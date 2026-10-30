# -*- coding: utf-8 -*-
"""M5 集成：导出 ONNX 推理模型（含输入归一化 + domain one-hot + 输出反归一化）。

用法：
    conda activate sofc
    python export_onnx.py --config config_m4b.yaml --ckpt runs/stage3_pde_m4b/best.pt

输入（推理侧直接给物理量）：
    x_phys : (B, 7) float32  [x(m), y(m), t_s(s), V_applied(V), T_in_K(K), lambda_air, FU]
    dom_id : (B,)   int64    domain 索引：0 air_channel, 1 air_electrode,
                             2 electrolyte, 3 fuel_electrode, 4 fuel_channel
输出：(B, 12) float32 物理量 [T, u, v, p, w1, w2, w3, phis, phil, htQ, qav_an, qav_ca]
    （w4 = 1 - w1 - w2 - w3 由用户按需导出；NaN 掩码域由调用侧按 domain 处理）

验证：onnxruntime CPU 输出 vs PyTorch 输出（误差 < 1e-3），并测 CPU 单次推理延迟。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from data.dataset import SOFCDataModule, CONT_IDS, cont_ids_for_cfg
from data.split import DOMAIN_ORDER
from models.pinn import BaselineMLP


def build_net(cfg: dict, n_out: int, n_cont: int = len(CONT_IDS)) -> nn.Module:
    m = cfg["model"]
    return BaselineMLP(n_cont=n_cont, n_domain=5, n_out=n_out,
                       hidden=int(m["hidden"]), layers=int(m["layers"]),
                       dropout=0.0, fourier=bool(m["fourier"]), n_freq=int(m["n_freq"]),
                       f_min=m.get("f_min", 1.0), f_max=m.get("f_max", 512.0),
                       fourier_cols=m.get("fourier_cols", [0, 1, 2]))


class PinnInference(nn.Module):
    """PyTorch 推理 wrapper：物理量输入 -> 物理量输出（含归一化/反归一化）。"""

    def __init__(self, net: nn.Module, x_stats: dict, y_stats: dict,
                 target_vars: list, n_domain: int = 5,
                 cont: list = None):
        super().__init__()
        self.net = net
        self.target_vars = list(target_vars)
        cont = list(cont if cont is not None else CONT_IDS)
        xm = torch.tensor([x_stats[c]["mean"] for c in cont], dtype=torch.float32)
        xs = torch.tensor([x_stats[c]["std"] for c in cont], dtype=torch.float32)
        ym = torch.tensor([y_stats[v]["mean"] for v in target_vars], dtype=torch.float32)
        ys = torch.tensor([y_stats[v]["std"] for v in target_vars], dtype=torch.float32)
        self.register_buffer("xm", xm)
        self.register_buffer("xs", xs)
        self.register_buffer("ym", ym)
        self.register_buffer("ys", ys)
        # domain one-hot 用固定单位矩阵 embedding（ONNX 导出为 Gather，稳定）
        eye = torch.eye(n_domain, dtype=torch.float32)
        self.register_buffer("_eye", eye)

    def forward(self, x_phys: torch.Tensor, dom_id: torch.Tensor):
        # x_phys: (B, 7)；dom_id: (B,) int64
        Xn = (x_phys - self.xm) / self.xs
        oh = torch.index_select(self._eye, 0, dom_id.long())   # (B, 5) one-hot
        out_n = self.net(torch.cat([Xn, oh], dim=-1))
        return out_n * self.ys + self.ym                       # 物理量


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_m4b.yaml")
    ap.add_argument("--ckpt", default="runs/stage3_pde_m4b/best.pt")
    ap.add_argument("--out", default=None, help="输出 .onnx 路径")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--bench", action="store_true", default=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 归一化统计（直接读 norm_stats.json，避免重载数据）
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    with open(os.path.join(run_dir, "norm_stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)
    x_stats, y_stats = stats["x"], stats["y"]
    target_vars = stats["target_vars"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cont = cont_ids_for_cfg(cfg)
    net = build_net(cfg, len(target_vars), n_cont=len(cont))
    net.load_state_dict(ck["model"])
    net.eval()
    wrapper = PinnInference(net, x_stats, y_stats, target_vars, cont=cont).eval()
    print(f"[model] loaded {args.ckpt} (epoch={ck.get('epoch')}, best_val={ck.get('best_val'):.4f})")

    # ---- 导出 ----
    out = args.out or os.path.join(run_dir, "pinn_sofc_m4b.onnx")
    B = 4
    x_dummy = torch.zeros(B, len(cont), dtype=torch.float32)
    d_dummy = torch.zeros(B, dtype=torch.int64)
    torch.onnx.export(
        wrapper, (x_dummy, d_dummy), out,
        input_names=["x_phys", "dom_id"], output_names=["y_phys"],
        dynamic_axes={"x_phys": {0: "batch"}, "dom_id": {0: "batch"},
                      "y_phys": {0: "batch"}},
        opset_version=args.opset,
    )
    import onnx
    model = onnx.load(out)
    onnx.checker.check_model(model)
    print(f"[onnx] exported -> {out}  (opset {args.opset}, inputs x_phys(B,{len(cont)}) float32, "
          f"dom_id(B,) int64, outputs y_phys(B,12))")

    # ---- onnxruntime 一致性 ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (256, len(cont))).astype(np.float32)
    # 物理范围随机点
    X[:, 0] = rng.uniform(0.0, 0.1, 256)          # x
    X[:, 1] = rng.uniform(0.0, 2.62e-3, 256)      # y
    X[:, 2] = rng.uniform(0.0, 1200.0, 256)       # t
    X[:, 3] = rng.uniform(0.75, 1.0, 256)         # V_cell
    if len(cont) > 7:                             # v19 waveform params
        X[:, 4] = rng.uniform(0.0, 0.25, 256)     # amp
        X[:, 5] = rng.uniform(0.0, 200.0, 256)    # stepT
        X[:, 6] = rng.uniform(0.0, 4.0, 256)      # wave_type
        X[:, 7] = rng.uniform(1073.0, 1123.0, 256)  # T_in
        X[:, 8] = rng.uniform(4.0, 8.0, 256)      # lambda_air
        X[:, 9] = rng.uniform(0.6, 0.8, 256)      # FU
    else:
        X[:, 4] = rng.uniform(1073.0, 1123.0, 256)  # T_in
        X[:, 5] = rng.uniform(4.0, 8.0, 256)        # lambda_air
        X[:, 6] = rng.uniform(0.6, 0.8, 256)        # FU
    D = rng.integers(0, 5, 256).astype(np.int64)
    with torch.no_grad():
        y_torch = wrapper(torch.from_numpy(X), torch.from_numpy(D)).numpy()
    y_ort = sess.run(None, {"x_phys": X, "dom_id": D})[0]
    err = np.abs(y_torch - y_ort).max()
    denom = np.sqrt((y_torch ** 2).sum()) + 1e-12
    rel = err / denom
    print(f"[onnx] max|torch-ort|={err:.3e}  rel_l2={rel:.3e}  (pass: {rel < 1e-3})")
    assert rel < 1e-3, "ONNX 输出与 PyTorch 不一致"

    # ---- CPU 延迟基准 ----
    if args.bench:
        xb = X[:1].copy(); db = D[:1].copy()
        # warmup
        for _ in range(20):
            sess.run(None, {"x_phys": xb, "dom_id": db})
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            sess.run(None, {"x_phys": xb, "dom_id": db})
        t_single = (time.perf_counter() - t0) / N * 1e3
        # batch=64
        x64 = X[:64].copy(); d64 = D[:64].copy()
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run(None, {"x_phys": x64, "dom_id": d64})
        t_batch = (time.perf_counter() - t0) / 20 * 1e3
        print(f"[bench] CPU single(batch=1) avg={t_single:.3f} ms/次; "
              f"batch=64 avg={t_batch:.3f} ms/次 ({64/t_batch:.0f} 点/ms)")
        ok = "PASS" if t_single <= 10.0 else "FAIL"
        print(f"[bench] 验收（单次 ≤10 ms）: {ok}")
    print("[onnx] done")


if __name__ == "__main__":
    main()
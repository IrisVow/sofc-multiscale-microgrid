# -*- coding: utf-8 -*-
"""D1 管线自检（v18 7 维 / v19 10 维 ONNX）：
ORT 会话加载/形状、确定性、domain_guard 联动、ONNX vs PyTorch 一致性（<1e-3）、
CPU 单次延迟（<=10 ms）、缓存命中。
运行：python scripts/check_surrogate.py [--onnx PATH] [--cont-version 7|10]
      （默认读 configs/coupling.yaml；输出 results/coupling/surrogate_check_v{ver}.json）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[1]

CHECKS = []


def report(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def make_ops(n_cont, v, t_in, lam, fu, t=0.0, x=0.05, y=1.31e-3,
             amp=0.0, stepT=0.0, wave_type=0.0):
    """按 cont_version 构造探针行：v19 列序 [x,y,t,V,amp,stepT,wave_type,T_in,lambda_air,FU]。"""
    import numpy as np
    row = [x, y, t, v]
    if n_cont >= 10:
        row += [amp, stepT, wave_type, t_in, lam, fu]
    else:
        row += [t_in, lam, fu]
    return np.array([row], dtype=np.float32)


def main():
    import numpy as np
    from src.coupling import PinnSurrogate
    from src.coupling.domain_guard import check_dict
    from src.coupling.pinn_coupling import load_coupling_params

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--cont-version", type=int, default=None)
    ap.add_argument("--ckpt", default="final.pt",
                    help="onnx 同目录中的 PyTorch checkpoint 文件名（v20部署模型对应 final.pt）")
    args = ap.parse_args()

    p = load_coupling_params()
    cont_version = args.cont_version or int(p.get("cont_version", 7))
    onnx = Path(args.onnx) if args.onnx else ROOT / p["onnx_path"]
    print(f"== D1 管线自检（cont_version={cont_version}，ONNX: {onnx.name}）==")

    ok_load = onnx.exists()
    report("ONNX 文件存在", ok_load, str(onnx))
    sur = PinnSurrogate(onnx)
    report("输入特征维 = cont_version", sur.N_CONT == cont_version, f"N_CONT={sur.N_CONT}")

    ins = {i.name: tuple(i.shape) for i in sur.sess.get_inputs()}
    outs = [o.name for o in sur.sess.get_outputs()]
    report("输入名 x_phys/dom_id", set(ins) == {"x_phys", "dom_id"}, str(ins))
    report("输出名 y_phys", "y_phys" in outs, str(outs))

    # ---- 确定性 / 形状 / 缓存 ----
    ops = make_ops(cont_version, 0.85, 1123.0, 6.0, 0.7)
    y1 = sur.infer(ops, np.array([2], dtype=np.int64))
    y2 = sur.infer(ops, np.array([2], dtype=np.int64))
    report("输出形状 (1,12)", y1.shape == (1, 12), str(y1.shape))
    report("输出有限", bool(np.isfinite(y1).all()))
    report("确定性（重复调用一致）", float(np.abs(y1 - y2).max()) == 0.0)
    report("缓存命中", sur.stats["cache_hits"] >= 1, f"hits={sur.stats['cache_hits']}")

    # ---- domain_guard 联动 ----
    ok_in, _ = check_dict({"x": 0.05, "y": 1.31e-3, "t": 0.0, "V_applied": 0.9,
                           "T_in": 1073.0, "lambda_air": 6.0, "FU": 0.7})
    report("域内点通过", ok_in)
    out_g, in_dom_g, info_g = sur.infer_guarded(make_ops(cont_version, 0.60, 1123.0, 6.0, 0.7))
    viol_vars = [d["var"] for d in info_g.get("violation_detail", [])]
    report("域外点拒绝推理（NaN）", (not in_dom_g[0]) and np.isnan(out_g).all(), f"viol={viol_vars}")
    report("域外计数", sur.stats["n_out_of_domain"] == 1, str(sur.stats["n_out_of_domain"]))

    # ---- 延迟基准 ----
    bench = sur.benchmark(n=100)
    ok_lat = bench["single_ms"] <= 10.0
    report("CPU 单次延迟 <=10 ms", ok_lat, f"{bench['single_ms']:.3f} ms")
    report("batch=64 延迟合理", bench["batch64_ms"] <= 10.0 * 4, f"{bench['batch64_ms']:.3f} ms")

    # ---- ONNX vs PyTorch（从 ONNX 所在 run 目录加载）----
    torch_ok, detail = False, ""
    try:
        import torch
        import yaml
        run_dir = onnx.resolve().parent
        torch_ok_path = run_dir / args.ckpt
        if not torch_ok_path.exists():
            detail = f"ckpt not found: {torch_ok_path}"
            raise FileNotFoundError(detail)
        sys.path.insert(0, str(ROOT.parent / "pinn_sofc"))
        from export_onnx import PinnInference, build_net
        from data.dataset import cont_ids_for_cfg

        with io.open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with io.open(run_dir / "norm_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
        cont = cont_ids_for_cfg(cfg)
        net = build_net(cfg, len(stats["target_vars"]), n_cont=len(cont))
        ck = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        net.eval()
        wrap = PinnInference(net, stats["x"], stats["y"], stats["target_vars"], cont=cont).eval()

        rng = np.random.default_rng(0)
        n_cont = len(cont)
        X = rng.uniform(0.0, 1.0, (256, n_cont)).astype(np.float32)
        X[:, 0] = rng.uniform(0.0, 0.1, 256)            # x
        X[:, 1] = rng.uniform(0.0, 2.62e-3, 256)        # y
        X[:, 2] = rng.uniform(0.0, 1200.0, 256)         # t
        X[:, 3] = rng.uniform(0.75, 1.0, 256)           # V_applied
        if n_cont > 7:                                  # v19 波形参数
            X[:, 4] = rng.uniform(0.0, 0.25, 256)       # amp
            X[:, 5] = rng.uniform(0.0, 200.0, 256)      # stepT
            X[:, 6] = rng.uniform(0.0, 4.0, 256)        # wave_type
            X[:, 7] = rng.uniform(1073.0, 1123.0, 256)  # T_in
            X[:, 8] = rng.uniform(4.0, 8.0, 256)        # lambda_air
            X[:, 9] = rng.uniform(0.6, 0.8, 256)        # FU
        else:
            X[:, 4] = rng.uniform(1073.0, 1123.0, 256)  # T_in
            X[:, 5] = rng.uniform(4.0, 8.0, 256)        # lambda_air
            X[:, 6] = rng.uniform(0.6, 0.8, 256)        # FU
        D = rng.integers(0, 5, 256).astype(np.int64)
        with torch.no_grad():
            y_torch = wrap(torch.from_numpy(X), torch.from_numpy(D)).numpy()
        y_ort = sur.infer(X, D)
        err = float(np.abs(y_torch - y_ort).max())
        rel = err / (float(np.sqrt((y_torch ** 2).sum())) + 1e-12)
        torch_ok = bool(rel < 1e-3)
        detail = f"max|d|={err:.3e} rel_l2={rel:.3e}"
    except Exception as e:
        detail = f"SKIP: {type(e).__name__}: {e}"
    report("ONNX vs PyTorch rel_l2 < 1e-3", torch_ok, detail)

    n_ok = sum(1 for c in CHECKS if c["ok"])
    total = len(CHECKS)
    ok_all = n_ok == total
    print(f"\n== {'全部通过' if ok_all else '存在失败'}：{n_ok}/{total} 项检查 ==")
    out_dir = ROOT / "results" / "coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    with io.open(out_dir / f"surrogate_check_v{cont_version}.json", "w", encoding="utf-8") as f:
        json.dump({"checks": CHECKS, "all_ok": ok_all, "bench": bench,
                   "onnx": str(onnx), "cont_version": cont_version},
                  f, ensure_ascii=False, indent=2)
    print(f"-> {out_dir / ('surrogate_check_v' + str(cont_version) + '.json')}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

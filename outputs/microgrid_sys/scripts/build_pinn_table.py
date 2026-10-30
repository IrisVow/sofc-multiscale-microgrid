# -*- coding: utf-8 -*-
"""生成 PINN 查找表（阶段 D1 骨架）：枚举 V × T_in × λ_air × FU 工况网格，
对每个工况点用 ONNX 推理探针场并汇总 Tmax / qav_an / qav_ca / Δφ。

用法：
    python scripts/build_pinn_table.py                              # 用 configs/coupling.yaml 的 onnx/lut 路径
    python scripts/build_pinn_table.py --onnx <v19.onnx> --out data/pinn_lut_v19.npz --cont-version 10
输出：data/pinn_lut_<tag>.npz（PinnLUT 格式）+ results/coupling/pinn_lut_<tag>.csv + summary.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_grid(spec: str):
    nv, nt, nl, nf = [int(x) for x in spec.lower().split("x")]
    return nv, nt, nl, nf


def make_axes(nv, nt, nl, nf):
    v = np.linspace(0.75, 1.0, nv)
    t_in = np.linspace(1073.0, 1123.0, nt)
    lam = np.linspace(4.0, 8.0, nl)
    fu = np.linspace(0.6, 0.8, nf)
    return v, t_in, lam, fu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cont-version", type=int, default=None)
    ap.add_argument("--grid", default="11x3x3x3")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from src.coupling import PinnSurrogate
    from src.coupling.domain_guard import DOMAIN_ORDER, check as guard_check
    from src.coupling.pinn_coupling import _DOMAIN_MID_Y, load_coupling_params

    p = load_coupling_params()
    onnx = args.onnx or p.get("onnx_path")
    cont_version = args.cont_version or int(p.get("cont_version", 7))
    out = args.out or p.get("lut_path")
    out_path = Path(out) if os.path.isabs(out) else ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tag = out_path.stem.replace("pinn_lut_", "")

    sur = PinnSurrogate(onnx if os.path.isabs(onnx) else ROOT / onnx)

    probes = p.get("probes")
    if not probes:
        probes = [{"domain": d, "y_m": y} for d, y in _DOMAIN_MID_Y.items()]
    dom_id = np.array([DOMAIN_ORDER.index(pr["domain"]) for pr in probes], dtype=np.int64)
    ys = np.array([float(pr["y_m"]) for pr in probes], dtype=float)
    x0 = float(p.get("probe_x_m", 0.05))
    t0 = float(p.get("t_s", 0.0))

    v, t_in, lam, fu = make_axes(*parse_grid(args.grid))
    n_grid = v.size * t_in.size * lam.size * fu.size
    n_p = len(ys)
    print(f"== 生成 PINN 查找表：grid={args.grid}（{n_grid} 工况 × {n_p} 探针）=")
    print(f"   onnx={onnx}（cont_version={cont_version}）-> {out_path}")

    # ---- 构建全部探针输入 ----
    rows = []
    gidx = []
    for V in v:
        for T in t_in:
            for L in lam:
                for F in fu:
                    base = np.array([[x0, y, t0, V, T, L, F] for y in ys], dtype=float)
                    if cont_version == 10:
                        # v19 列序：x,y,t,V,amp,stepT,wave_type,T_in,lambda_air,FU
                        base = np.concatenate([base[:, :4], np.zeros((n_p, 3)), base[:, 4:]], axis=1)
                    rows.append(base)
    ops = np.concatenate(rows, axis=0)
    dom_full = np.tile(dom_id, n_grid)
    guard_cols = ops[:, :7] if cont_version == 7 else ops[:, [0, 1, 2, 3, 7, 8, 9]]

    # ---- 域检查 ----
    in_domain, ginfo = guard_check(guard_cols)
    n_oov = int((~in_domain).sum())
    print(f"   域外探针点: {n_oov}/{ops.shape[0]}")

    # ---- 批量推理（域内点）----
    out = np.full((ops.shape[0], sur.N_OUT), np.nan, dtype=np.float32)
    ok_idx = np.where(in_domain)[0]
    t_start = time.perf_counter()
    for s in range(0, len(ok_idx), args.batch):
        idx = ok_idx[s:s + args.batch]
        out[idx] = sur.infer(ops[idx], dom_full[idx])
    dt_s = time.perf_counter() - t_start
    print(f"   推理 {len(ok_idx)} 点 -> {dt_s:.2f}s（batch={args.batch}）")

    # ---- 按工况点聚合 ----
    TMAX = np.full(n_grid, np.nan); QAN = np.full(n_grid, np.nan)
    QCA = np.full(n_grid, np.nan); PHI = np.full(n_grid, np.nan)
    el_mask = dom_id == 2
    for g in range(n_grid):
        sl = slice(g * n_p, (g + 1) * n_p)
        yg = out[sl]
        TMAX[g] = np.nanmax(yg[:, 0])
        QAN[g] = np.nanmax(yg[:, 10])
        QCA[g] = np.nanmax(yg[:, 11])
        el = yg[el_mask]
        PHI[g] = np.nanmax(np.abs(el[:, 7] - el[:, 8])) if len(el) else np.nan
    shape = (v.size, t_in.size, lam.size, fu.size)
    meta = {"onnx": str(onnx), "cont_version": cont_version, "probes": probes,
            "grid": args.grid, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_out_of_domain_probe_points": n_oov,
            "tmax_limit_k": float(p.get("tmax_limit_k", 1123.0))}
    np.savez(out_path,
             v_axis=v, t_in_axis=t_in, lam_axis=lam, fu_axis=fu,
             tmax_k=TMAX.reshape(shape), qav_an_max=QAN.reshape(shape),
             qav_ca_max=QCA.reshape(shape), phi_drop_v=PHI.reshape(shape),
             meta=np.array(meta))

    # ---- CSV（人类可读，工况点级）----
    csv_rows = []
    k = 0
    for V in v:
        for T in t_in:
            for L in lam:
                for F in fu:
                    csv_rows.append(f"{V:.4f},{T:.1f},{L:.2f},{F:.3f},"
                                    f"{TMAX[k]:.2f},{QAN[k]:.4f},{QCA[k]:.4f},{PHI[k]:.6f}")
                    k += 1
    res_dir = ROOT / "results" / "coupling"
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"pinn_lut_{tag}.csv"
    with io.open(csv_path, "w", encoding="utf-8") as f:
        f.write("V,T_in,lambda_air,FU,tmax_k,qav_an_max,qav_ca_max,phi_drop_v\n")
        f.write("\n".join(csv_rows) + "\n")

    # ---- 标定摘要 ----
    nm = np.argmin(np.abs(t_in - 1123.0)); lm = np.argmin(np.abs(lam - 6.0))
    fm = np.argmin(np.abs(fu - 0.7))
    tmax_nom = TMAX.reshape(shape)[:, nm, lm, fm]
    phi_nom = PHI.reshape(shape)[:, nm, lm, fm]
    summary = {"tag": tag, "grid": args.grid, "n_grid": n_grid, "n_probes": n_p,
               "n_out_of_domain_probe_points": n_oov, "onnx": str(onnx),
               "tmax_range_k": [float(np.nanmin(TMAX)), float(np.nanmax(TMAX))],
               "tmax_v_nominal": [float(x) for x in tmax_nom],
               "phi_drop_v_nominal": [float(x) for x in phi_nom],
               "lut_path": str(out_path.relative_to(ROOT)),
               "csv_path": str(csv_path.relative_to(ROOT))}
    with io.open(res_dir / f"pinn_lut_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"   Tmax 全网格范围: [{summary['tmax_range_k'][0]:.1f}, {summary['tmax_range_k'][1]:.1f}] K")
    print(f"   Tmax(V) @ T_in=1123,λ=6,FU=0.7: " +
          ", ".join(f"{x:.1f}" for x in tmax_nom))
    print(f"   Δφ(V) @ 名义: " + ", ".join(f"{x:.5f}" for x in phi_nom))
    print(f"   -> {out_path.relative_to(ROOT)} / {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
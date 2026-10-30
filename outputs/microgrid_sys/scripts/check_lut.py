# -*- coding: utf-8 -*-
"""D2 自检：PinnLUT 4D 线性插值（默认）与最近邻对照。

验证：
1. 网格点精确还原（线性在格点上 == 表值）
2. 中点线性性质（单轴相邻两点中点 == 均值）
3. Tmax 连续性：沿 V 稠密采样，线性最大相邻跳变远小于最近邻
4. 域外 clamp：V>上界 返回边界值（不外推）
5. 最近邻兼容：method="nearest" 仍等于 argmin 取点
6. 数值合理：域内随机点插值有限且在表值范围内
7. 性能：单次查询 < 0.1 ms（10 万次基准）
运行：python scripts/check_lut.py
输出：results/coupling/lut_check_<lut 版本>.json（lut_path 取自 configs/coupling.yaml）
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[1]

CHECKS = []


def report(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def main():
    import numpy as np
    from src.coupling import PinnLUT

    import yaml
    with io.open(ROOT / "configs" / "coupling.yaml", encoding="utf-8") as f:
        _lut_rel = yaml.safe_load(f)["pinn"].get("lut_path", "data/pinn_lut_v19.npz")
    npz = ROOT / str(_lut_rel)
    lut = PinnLUT(npz)
    print(f"== D2 PinnLUT 插值自检（{npz.name}，grid {lut.shape}）==")

    v, t_in, lam, fu = (lut.axes[k] for k in lut.AXES)

    # 1. 网格点精确还原
    errs = []
    for i in range(v.size):
        for j in range(t_in.size):
            for k in range(lam.size):
                for m in range(fu.size):
                    r = lut.lookup(v[i], t_in[j], lam[k], fu[m])
                    for f in lut.FIELDS:
                        errs.append(abs(r[f] - float(lut.data[f][(i, j, k, m)])))
    report("网格点线性还原（max|err| < 1e-6）", max(errs) < 1e-6, f"max={max(errs):.2e}")

    # 2. 中点线性性质（沿 V 轴，其余取网格点）
    vi = 5
    v_mid = 0.5 * (v[vi] + v[vi + 1])
    r = lut.lookup(v_mid, t_in[1], lam[1], fu[1])
    exp = 0.5 * (float(lut.data["tmax_k"][vi, 1, 1, 1]) + float(lut.data["tmax_k"][vi + 1, 1, 1, 1]))
    report("中点线性性质（Tmax=两点均值）", abs(r["tmax_k"] - exp) < 1e-6,
           f"mid={r['tmax_k']:.4f} exp={exp:.4f}")

    # 3. Tmax 连续性：沿 V 稠密采样（T_in=1100, λ=6, FU=0.7）
    t_fix, l_fix, f_fix = 1100.0, 6.0, 0.7
    xs = np.linspace(float(v[0]), float(v[-1]), 2001)
    lin = np.array([lut.lookup(x, t_fix, l_fix, f_fix)["tmax_k"] for x in xs])
    nn = np.array([lut.lookup(x, t_fix, l_fix, f_fix, method="nearest")["tmax_k"] for x in xs])
    jump_lin = float(np.max(np.abs(np.diff(lin))))
    jump_nn = float(np.max(np.abs(np.diff(nn))))
    ok_cont = jump_lin < 0.5 * jump_nn + 1e-9 and jump_lin < 2.0
    report("Tmax 连续性（线性最大相邻跳变 << 最近邻）", ok_cont,
           f"linear={jump_lin:.4f} K vs nearest={jump_nn:.4f} K")

    # 4. 域外 clamp
    r_hi = lut.lookup(v[-1] + 0.05, t_fix, l_fix, f_fix)
    r_edge = lut.lookup(v[-1], t_fix, l_fix, f_fix)
    report("域外 clamp（V>上界 == 边界值，不外推）", abs(r_hi["tmax_k"] - r_edge["tmax_k"]) < 1e-9,
           f"{r_hi['tmax_k']:.4f} == {r_edge['tmax_k']:.4f}")

    # 5. 最近邻兼容
    r_nn = lut.lookup(v[3], t_in[0], lam[2], fu[1], method="nearest")
    r_arg = float(lut.data["tmax_k"][3, 0, 2, 1])
    report("nearest 兼容（== 原 argmin 取点）", abs(r_nn["tmax_k"] - r_arg) < 1e-12,
           f"{r_nn['tmax_k']:.4f}")

    # 6. 随机域内点数值合理
    rng = np.random.default_rng(42)
    N = 500
    ok_range = True
    for _ in range(N):
        xv = float(rng.uniform(v[0], v[-1]))
        xt = float(rng.uniform(t_in[0], t_in[-1]))
        xl = float(rng.uniform(lam[0], lam[-1]))
        xf = float(rng.uniform(fu[0], fu[-1]))
        r = lut.lookup(xv, xt, xl, xf)
        if not (np.isfinite(r["tmax_k"])
                and float(lut.data["tmax_k"].min()) <= r["tmax_k"] <= float(lut.data["tmax_k"].max())):
            ok_range = False
            break
    report("随机域内插值有限且在表值范围内", ok_range, f"N={N}")

    # 7. 性能（10 万次）
    t0 = time.perf_counter()
    for _ in range(100_000):
        lut.lookup(0.83, 1100.0, 6.0, 0.71)
    dt_ms = (time.perf_counter() - t0) / 100_000 * 1e3
    report("单次查询 < 0.1 ms（10 万次基准）", dt_ms < 0.1, f"{dt_ms:.5f} ms/次")

    n_ok = sum(1 for c in CHECKS if c["ok"])
    ok_all = n_ok == len(CHECKS)
    print(f"\n== {'全部通过' if ok_all else '存在失败'}：{n_ok}/{len(CHECKS)} 项检查 ==")
    out_dir = ROOT / "results" / "coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"lut_check_{npz.stem.replace('pinn_lut_', '')}.json"
    with io.open(out_dir / out_name, "w", encoding="utf-8") as f:
        json.dump({"checks": CHECKS, "all_ok": ok_all, "npz": str(npz.relative_to(ROOT)),
                   "continuity": {"linear_max_jump_k": jump_lin, "nearest_max_jump_k": jump_nn}},
                  f, ensure_ascii=False, indent=2)
    print(f"-> {out_dir / out_name}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
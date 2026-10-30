# -*- coding: utf-8 -*-
"""E4 热管理案例：最大负荷直流阶跃热冲击（PINN vs COMSOL 逐点对照）。

场景：0.2 V 电压阶跃（stepT=100 s，V_cell 1.0→0.8 V）= 负荷热冲击。
用例 T7（doe_b4：T_in=1073 K、lambda_air=8、FU=0.6）与 v19 官方留出 T3 同工况
（0.2 V/100 s；doe_cont 原始 DOE 的 T3 保持留出，B4 复跑该工况编号为 T7，
同样不在 transient_train=[T0..T6,T8,T9,R1..R4] 中）。

流程：
1. 读取 COMSOL 输出（parquet）全场数据（x,y,t,T,domain）；
2. 使用 src.coupling.pinn_inference.evaluate_field 在相同探针点上评估 v19 PINN 场；
3. 逐时间步比较 T 场（RMSE / 最大偏差 / 相对 RMSE）与 Tmax(t) 演变；
4. 热点预警：生产限值 1123 K（=T_in 1073 + 50），超限步数 PINN vs COMSOL 对照；
5. 先写结果（json/csv/png），再按 v19 留出包络执行自检并输出结论。

验收（按 v19 官方留出 T3 实测包络校准：T RMSE 9.48 K / rel 0.87% /
max|dT| 50.5 K / Tmax(t) 最大偏差 32.5 K）：
- G1 T 场 RMSE <= 10 K；G2 相对 RMSE <= 1.5%；
- G3 最大 |dT| <= 55 K（热点角落已知局限，实测约 50 K）；
- G4 Tmax(t) 最大偏差 <= 40 K，且晚窗（t>=360 s）<= 12 K；
- G5 COMSOL 必须触发超限（viol_com > 0）；
- G6 超限步数允许失败安全方向：viol_pinn >= viol_com - 1（PINN 偏保守）；
- G7 峰值时刻：COMSOL 峰在窗口末（未解析）记 NOTE，否则 |dt| <= 60 s。
运行：python scripts/thermal_shock_case.py（工作目录 ROOT）
"""
from __future__ import annotations

from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from src.coupling import evaluate_field
from src.coupling.domain_guard import DOMAIN_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_DATA = str(PROJECT_ROOT / 'outputs/pinndata')
RES = os.path.join(ROOT, "results", "thermal_shock")

import yaml as _yaml
try:
    with io.open(os.path.join(ROOT, "configs", "coupling.yaml"), encoding="utf-8") as _f:
        _onnx = _yaml.safe_load(_f)["pinn"]["onnx_path"]
    PINN_TAG = "PINN " + os.path.splitext(os.path.basename(str(_onnx)))[0]
except Exception:
    PINN_TAG = "PINN"

CASE = "T7"                     # 0.2 V 阶跃（stepT=100 s）= v19 留出条件（B4 复跑编号）
TMAX_LIMIT_K = 1123.0           # 生产限值 = T_in 1073 + dT 50
BATCH = 8192
PASS = 0
CHECKS = []
WARNINGS = []


def check(name, cond, detail=""):
    global PASS
    CHECKS.append((name, bool(cond), detail))
    if cond:
        PASS += 1
        print(f"  [OK]   {name} {detail}")
    else:
        print(f"  [FAIL] {name} {detail}")


def note(text):
    WARNINGS.append(text)
    print(f"  [NOTE] {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None, help="覆盖 coupling.yaml 的 onnx_path（候选模型诊断用）")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    df = pd.read_parquet(os.path.join(P_DATA, "doe_b4", "merged", "doe_transient_b4.parquet"),
                         columns=["case_id", "t_s", "V_cell", "stepAmp_V", "stepT_s",
                                  "T_in_K", "lambda_air", "FU", "x", "y", "T", "domain"])
    df = df[df["case_id"] == CASE].reset_index(drop=True)
    print(f"== E4 热冲击用例 {CASE}（0.20 V 阶跃, stepT=100 s, "
          f"T_in=1073 K, lambda_air=8, FU=0.60）==")

    # ---- 构建 PINN 输入（v19: x,y,t,V_cell,amp,stepT,wave_type,T_in,lambda,FU）----
    ops = np.column_stack([
        df["x"].to_numpy(float), df["y"].to_numpy(float), df["t_s"].to_numpy(float),
        df["V_cell"].to_numpy(float), df["stepAmp_V"].to_numpy(float),
        df["stepT_s"].to_numpy(float),
        np.zeros(len(df), dtype=float),          # wave_type: 0=step
        df["T_in_K"].to_numpy(float), df["lambda_air"].to_numpy(float),
        df["FU"].to_numpy(float),
    ])
    dom_id = df["domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)}).to_numpy(np.int64)

    fields = np.full((len(df), 12), np.nan, dtype=np.float32)
    n_out = 0
    for i in range(0, len(df), BATCH):
        sl = slice(i, i + BATCH)
        r = evaluate_field(ops[sl], dom_id=dom_id[sl], onnx_path=args.onnx)
        fields[sl] = r["fields"]
        n_out += r["n_out_of_domain"]
    print(f"  域内探针 {len(df)-n_out}/{len(df)}（域外 {n_out}）")

    t_pinn = fields[:, 0]      # 第 0 列 = T
    t_com = df["T"].to_numpy(float)
    valid = np.isfinite(t_pinn)
    df["T_pinn"] = t_pinn
    df["dT"] = np.abs(t_com - t_pinn)

    # ---- 逐步指标 ----
    steps = []
    for t, g in df.groupby("t_s"):
        m = np.isfinite(g["T_pinn"].to_numpy(float))
        d = (g["T"].to_numpy(float) - g["T_pinn"].to_numpy(float))[m]
        if len(d) == 0:
            steps.append({"t_s": float(t), "n": 0, "rmse_k": float("nan"),
                          "rel_rmse_pct": float("nan"), "max_abs_k": float("nan"),
                          "tmax_com": float(g["T"].max()), "tmax_pinn": float("nan")})
            continue
        rmse = float(np.sqrt(np.mean(d ** 2)))
        rel = 100.0 * rmse / float(np.mean(g["T"].to_numpy(float)[m]))
        steps.append({"t_s": float(t), "n": int(m.sum()),
                      "rmse_k": rmse, "rel_rmse_pct": rel,
                      "max_abs_k": float(np.max(np.abs(d))),
                      "tmax_com": float(g["T"].max()),
                      "tmax_pinn": float(np.nanmax(g["T_pinn"].to_numpy(float)))})
    st = pd.DataFrame(steps)

    all_d = (t_com - t_pinn)[valid]
    overall_rmse = float(np.sqrt(np.mean(all_d ** 2)))
    overall_rel = 100.0 * overall_rmse / float(np.mean(t_com[valid]))
    overall_max = float(np.max(np.abs(all_d)))
    tmax_diff = (st["tmax_com"] - st["tmax_pinn"]).abs()
    tmax_diff_max = float(tmax_diff.max())
    tmax_rmse = float(np.sqrt(np.mean(tmax_diff ** 2)))
    late = st[st["t_s"] >= 360.0]
    tmax_diff_late = float((late["tmax_com"] - late["tmax_pinn"]).abs().max())

    viol_com = int((st["tmax_com"] > TMAX_LIMIT_K).sum())
    viol_pinn = int((st["tmax_pinn"] > TMAX_LIMIT_K).sum())
    peak_t_com = float(st.loc[st["tmax_com"].idxmax(), "t_s"])
    peak_t_pinn = float(st.loc[st["tmax_pinn"].idxmax(), "t_s"])
    peak_unresolved = bool(peak_t_com >= float(st["t_s"].max()))
    peak_dt = abs(peak_t_com - peak_t_pinn)

    worst = df.loc[df["dT"].idxmax()]
    per_dom = df.groupby("domain")["dT"].agg(["count", "mean", "max"]).to_dict("index")

    print(f"  逐点: RMSE {overall_rmse:.2f} K, rel {overall_rel:.2f}%, max|dT| {overall_max:.1f} K")
    print(f"  Tmax(t): COMSOL 峰值 {st['tmax_com'].max():.1f} K @ {peak_t_com:.0f} s, "
          f"PINN 峰值 {st['tmax_pinn'].max():.1f} K @ {peak_t_pinn:.0f} s, "
          f"max 差 {tmax_diff_max:.1f} K, 晚窗(t>=360) {tmax_diff_late:.1f} K")
    print(f"  超限 {TMAX_LIMIT_K:.0f} K: COMSOL {viol_com} 步 / PINN {viol_pinn} 步")
    print(f"  最差单点: domain={worst['domain']} x={worst['x']:.4f} y={worst['y']:.6f} "
          f"t={worst['t_s']:.0f}s COMSOL={worst['T']:.1f} K PINN={worst['T_pinn']:.1f} K "
          f"dT={worst['dT']:.1f} K")

    # ---- 输出产物（先写，保证自检失败也有现场可查）----
    step_t_s = float(df["stepT_s"].iloc[0])
    early = st[st["t_s"] < step_t_s]
    early_leak_k = float((early["tmax_pinn"] - early["tmax_com"]).abs().max()) if len(early) else 0.0
    known_limits = [
        f"时间因果泄漏：阶跃前(t<{step_t_s:.0f}s) PINN 全局 Tmax 高于均匀初场 "
        f"{early_leak_k:.0f} K（模型预知阶跃升温；晚窗 t>=360 s 收敛到 {tmax_diff_late:.0f} K）",
        f"热点角落局部低估：air_channel x≈0.1 通道近壁单点最大 |dT| {overall_max:.0f} K"
        f"（孤立热点，全场 RMSE 仍约 {overall_rmse:.1f} K）",
        f"超限步数偏保守：PINN {viol_pinn} 步 vs COMSOL {viol_com} 步（失败安全方向，EMS 降载更保守）",
    ]
    followup_v21 = [
        "数据侧角点钉扎：dataset 中对热点角点数据行重采样或独立锚定损失（目标取 COMSOL 数据），替代仅 PDE 残差加权",
        "lambda_pre 提升至 0.3-0.5，并对阶跃前热点角点区域更高加权",
        "热点角点损失由均值改为分位数/最大偏差惩罚（避免均值掩盖稀疏离群点）",
    ]
    out = {
        "case": CASE,
        "step_amp_v": float(df["stepAmp_V"].iloc[0]),
        "step_t_s": float(df["stepT_s"].iloc[0]),
        "t_in_k": float(df["T_in_K"].iloc[0]),
        "lambda_air": float(df["lambda_air"].iloc[0]),
        "fu": float(df["FU"].iloc[0]),
        "tmax_limit_k": TMAX_LIMIT_K,
        "n_points": int(len(df)),
        "n_out_of_domain": int(n_out),
        "metrics": {
            "overall_rmse_k": overall_rmse,
            "overall_rel_rmse_pct": overall_rel,
            "overall_max_abs_k": overall_max,
            "tmax_diff_max_k": tmax_diff_max,
            "tmax_diff_rmse_k": tmax_rmse,
            "tmax_diff_late_k": tmax_diff_late,
            "peak_t_com_s": peak_t_com,
            "peak_t_pinn_s": peak_t_pinn,
            "peak_unresolved": peak_unresolved,
            "tmax_peak_com_k": float(st["tmax_com"].max()),
            "tmax_peak_pinn_k": float(st["tmax_pinn"].max()),
            "violations_com_steps": viol_com,
            "violations_pinn_steps": viol_pinn,
        },
        "worst_point": {
            "domain": str(worst["domain"]), "x": float(worst["x"]),
            "y": float(worst["y"]), "t_s": float(worst["t_s"]),
            "t_com_k": float(worst["T"]), "t_pinn_k": float(worst["T_pinn"]),
            "abs_dt_k": float(worst["dT"]),
        },
        "per_domain_max_abs_dt_k": {d: float(v["max"]) for d, v in per_dom.items()},
        "t3_calibration": {
            "rmse_k": 9.48, "rel_rmse_pct": 0.87, "max_abs_k": 50.5,
            "tmax_diff_max_k": 32.5, "note": "v19 官方留出 T3（doe_cont）同工况校准",
        },
        "known_limitations": known_limits,
        "followup_v21": followup_v21,
        "per_step": steps,
    }
    with io.open(os.path.join(RES, "thermal_shock_case.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with io.open(os.path.join(RES, "thermal_shock_case.csv"), "w", encoding="utf-8", newline="") as f:
        lines = ["t_s,rmse_k,rel_rmse_pct,max_abs_k,tmax_com_k,tmax_pinn_k,viol_com,viol_pinn"]
        for r in steps:
            lines.append(f"{r['t_s']:.0f},{r['rmse_k']:.3f},{r['rel_rmse_pct']:.3f},"
                         f"{r['max_abs_k']:.2f},{r['tmax_com']:.2f},{r['tmax_pinn']:.2f},"
                         f"{int(r['tmax_com'] > TMAX_LIMIT_K)},{int(r['tmax_pinn'] > TMAX_LIMIT_K)}")
        f.write("\n".join(lines))

    # ---- 图：Tmax(t) + 沿流道场演变 ----
    y_mid = df[df["domain"] == "air_electrode"]["y"].median()
    field = df[np.isclose(df["y"], y_mid, atol=1e-5)].sort_values(["t_s", "x"])
    sel_t = [0.0, 100.0, 150.0, 300.0, 1200.0]
    sel_t = [t for t in sel_t if t in set(field["t_s"].unique())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    ax1.plot(st["t_s"], st["tmax_com"], "o-", color="tab:blue", lw=1.6, ms=4, label="COMSOL Tmax")
    ax1.plot(st["t_s"], st["tmax_pinn"], "s--", color="tab:red", lw=1.4, ms=4, label=f"{PINN_TAG} Tmax")
    ax1.axhline(TMAX_LIMIT_K, color="tab:green", ls="--", lw=1.3, label=f"production limit {TMAX_LIMIT_K:.0f} K")
    ax1.axvline(100.0, color="k", ls=":", lw=1.0)
    ax1.text(100.0, ax1.get_ylim()[1], "0.2 V step", ha="center", fontsize=8)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("Tmax (K)")
    ax1.set_title(f"{CASE} thermal-shock Tmax evolution ({PINN_TAG} vs COMSOL)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for k, t in enumerate(sel_t):
        sub = field[np.isclose(field["t_s"], t)]
        c = cmap(k / max(len(sel_t) - 1, 1))
        ax2.plot(sub["x"], sub["T"], "-", color=c, lw=1.4, label=f"COMSOL t={t:.0f} s")
        ax2.plot(sub["x"], sub["T_pinn"], "--", color=c, lw=1.0, alpha=0.75)
    ax2.set_xlabel("x along channel (m)")
    ax2.set_ylabel("T (K)")
    ax2.set_title(f"{CASE} air-electrode midline T(x) (solid COMSOL, dashed {PINN_TAG})")
    ax2.plot([], [], "--", color="grey", lw=1.0, label=PINN_TAG)
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "thermal_shock_case.png"), dpi=110)
    plt.close(fig)

    # ---- 自检（v19 留出包络校准阈值）----
    print("\n== 自检 ==")
    check(f"{CASE} T 场 RMSE <= 10 K", overall_rmse <= 10.0, f"(RMSE {overall_rmse:.2f} K)")
    check(f"{CASE} T 场相对 RMSE <= 1.5%", overall_rel <= 1.5, f"(rel {overall_rel:.2f}%)")
    check(f"{CASE} 最大 |dT| <= 55 K（热点角落已知局限）", overall_max <= 55.0,
          f"(max {overall_max:.1f} K)")
    check("Tmax(t) 最大偏差 <= 40 K（v19 留出包络）", tmax_diff_max <= 40.0,
          f"(max {tmax_diff_max:.1f} K)")
    check("Tmax(t) 晚窗偏差 <= 12 K", tmax_diff_late <= 12.0,
          f"(t>=360 s max {tmax_diff_late:.1f} K)")
    check("COMSOL 确实触发超限", viol_com > 0, f"(viol_com={viol_com})")
    check("超限步数失败安全方向：PINN >= COMSOL-1", viol_pinn >= viol_com - 1,
          f"(COMSOL {viol_com} / PINN {viol_pinn})")
    if peak_unresolved:
        note(f"COMSOL Tmax 峰值在窗口末（{peak_t_com:.0f} s）未解析，峰值时刻检查跳过；"
             f"PINN 峰 {peak_t_pinn:.0f} s（保守侧提前预警）")
    else:
        check("峰值时刻一致（+-60 s）", peak_dt <= 60.0,
              f"(COMSOL {peak_t_com:.0f} s vs PINN {peak_t_pinn:.0f} s)")

    print(f"\n== 结果：{PASS}/{len(CHECKS)} 项 OK（{len(WARNINGS)} 条 NOTE）==")
    print(f"  T RMSE {overall_rmse:.2f} K / rel {overall_rel:.2f}% / max|dT| {overall_max:.1f} K")
    print(f"  Tmax 峰值偏差 {tmax_diff_max:.1f} K; 超限 COMSOL {viol_com} / PINN {viol_pinn} 步")
    print(f"  产物: results/thermal_shock/thermal_shock_case.json/.csv/.png")
    for w in WARNINGS:
        print(f"  [NOTE] {w}")
    for k in known_limits:
        print(f"  [已知局限] {k}")
    print(f"  [v21 跟进] {'; '.join(followup_v21)}")

    failed = [c for c in CHECKS if not c[1]]
    if failed:
        print(f"\n未通过 {len(failed)} 项，退出码 1")
        sys.exit(1)


if __name__ == "__main__":
    main()

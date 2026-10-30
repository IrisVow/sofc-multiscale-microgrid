# -*- coding: utf-8 -*-
"""E3 敏感性分析：推荐方案（SOFC 150 kW + 电池 100 kWh + 柴油 100 kW）
OFAT tornado。因子（PLANNING.md 阶段 E3）：

经济类（重算 economics，无需重仿）：
- LNG 价格 ±25%、SOFC CAPEX ±30%、电池 CAPEX ±40%、折现率 6%/10%、年电量 ±10%
物理类（带 PINN 联动重仿 7 天）：
- 降解速率 deg_scale ×0.5/×1.5（新增旋钮）
- T_in 1123→1098 K（基线为域上界）、λ_air 6±1、FU 0.70±0.05
指标：LCOE/GHG 变动 %；热管理因子额外报 Tmax 裕度与降载能量
（生产限值 1173 K = T_in + 50 K）。输出 results/capacity/e3_sensitivity.json/.csv/.png。
运行：python scripts/sensitivity_tornado.py（工作目录 ROOT）
"""
from __future__ import annotations

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
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

from src.analysis.capacity import CapacityOptimizer
from src.coupling import PinnCoupling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "capacity")

def load_best():
    """从 capacity_opt.json 读取推荐方案（SOFC/电池/柴油 三元组）。"""
    with io.open(os.path.join(RES, "capacity_opt.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    b = data["selection"]["best"]
    return float(b["sofc_kw"]), float(b["battery_kwh"]), float(b["genset_kw"])

SOFC_KW, BAT_KWH, GENSET_KW = load_best()
PASS = 0

def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")

def load_base_row():
    with io.open(os.path.join(RES, "capacity_opt.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data["rows"]:
        if (float(r["sofc_kw"]) == SOFC_KW and float(r["battery_kwh"]) == BAT_KWH
                and float(r["genset_kw"]) == GENSET_KW):
            return r
    raise RuntimeError("base row not found")

def econ_lcoe(base_row, overrides):
    """用覆盖参数重算年化经济性（无重仿）。"""
    opt = CapacityOptimizer(overrides=overrides)
    row = dict(base_row)
    row.update(opt.economics(row))
    return row

def sim_lcoe(overrides=None, coupling=None, sofc_extra=None):
    """重仿 7 天带联动/降解旋铮的推荐方案。"""
    opt = CapacityOptimizer()
    co = {}
    if sofc_extra:
        co["sofc"] = dict(sofc_extra)
    return opt.run_point(SOFC_KW, BAT_KWH, genset_kw=GENSET_KW, coupling=coupling,
                         component_overrides=co or None)

def pct(a, b):
    return (a / b - 1.0) * 100.0

def main():
    base = load_base_row()
    base_lcoe = base["lcoe_usd_mwh"]
    base_ghg = base["ghg_kg_co2eq"]
    print("== E3 敏感性 tornado（基线 LCOE "
          f"{base_lcoe:.1f} $/MWh, GHG {base_ghg:.0f} kg/7d）==")

    econ_factors = [
        ("fuel_lng_usd_kwh", "LNG 价格", 0.032, 0.054),
        ("capex_sofc_usd_kw", "SOFC CAPEX", 1050.0, 1950.0),
        ("capex_battery_usd_kwh", "电池 CAPEX", 210.0, 490.0),
        ("discount_rate", "折现率", 0.06, 0.10),
        ("annual_energy_mwh", "年电量（负荷）", 761.7, 930.9),
    ]
    econ_results = []
    for key, label, lo, hi in econ_factors:
        r_lo = econ_lcoe(base, {key: lo})
        r_hi = econ_lcoe(base, {key: hi})
        econ_results.append({
            "factor": label, "key": key, "lo_value": lo, "hi_value": hi,
            "lcoe_lo_usd_mwh": r_lo["lcoe_usd_mwh"],
            "lcoe_hi_usd_mwh": r_hi["lcoe_usd_mwh"],
            "dlcoe_lo_pct": pct(r_lo["lcoe_usd_mwh"], base_lcoe),
            "dlcoe_hi_pct": pct(r_hi["lcoe_usd_mwh"], base_lcoe),
        })
        print(f"  {label:>14}: ↓{lo} -> {econ_results[-1]['dlcoe_lo_pct']:+.1f}%  "
              f"↑{hi} -> {econ_results[-1]['dlcoe_hi_pct']:+.1f}%")

    # ---- 物理因子（重仿 + PINN 联动）----
    coup_base = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate"})
    rb = sim_lcoe(coupling=coup_base)
    csb = coup_base.summary()
    base_max_t = csb["pinn_max_tmax_est_k"]
    base_derate = csb["pinn_derate_steps"]
    print(f"  基线联动: Tmax max {base_max_t:.1f} K, "
          f"降载 {base_derate} 步 (limit 1173 K)")

    phys = []
    # 降解速率
    r_d = sim_lcoe(sofc_extra={"deg_scale": 1.5})
    r_d2 = sim_lcoe(sofc_extra={"deg_scale": 0.5})
    phys.append({"factor": "降解速率", "kind": "soh",
                 "lcoe_lo_usd_mwh": r_d2["lcoe_usd_mwh"],
                 "lcoe_hi_usd_mwh": r_d["lcoe_usd_mwh"],
                 "dlcoe_lo_pct": pct(r_d2["lcoe_usd_mwh"], base_lcoe),
                 "dlcoe_hi_pct": pct(r_d["lcoe_usd_mwh"], base_lcoe),
                 "soh_end_lo": r_d2["sofc_soh_end"], "soh_end_hi": r_d["sofc_soh_end"]})
    print(f"  降解速率: ×0.5 -> {phys[-1]['dlcoe_lo_pct']:+.3f}% "
          f"(SOH {r_d2['sofc_soh_end']:.5f})  ×1.5 -> {phys[-1]['dlcoe_hi_pct']:+.3f}% "
          f"(SOH {r_d['sofc_soh_end']:.5f})")
    # T_in / lambda_air / FU（联动重仿）
    for key, label, lo, hi, base_val in [
        ("t_in_nominal_k", "T_in", 1098.0, 1123.0, 1123.0),
        ("lambda_air_nominal", "lambda_air", 5.0, 7.0, 6.0),
        ("fu_nominal", "FU", 0.65, 0.75, 0.70),
    ]:
        entries = []
        for tag, val in [("lo", lo), ("hi", hi), ("base", base_val)]:
            if abs(val - base_val) < 1e-9:
                entries.append((tag, rb, coup_base, csb))
            else:
                coup = PinnCoupling(params={"use_pinn_table": True,
                                            "constraint_mode": "derate", key: val})
                r = sim_lcoe(coupling=coup)
                cs = coup.summary()
                entries.append((tag, r, coup, cs))
        t_lo = next(e[3]["pinn_max_tmax_est_k"] for e in entries if e[0] == "lo")
        t_hi = next(e[3]["pinn_max_tmax_est_k"] for e in entries if e[0] == "hi")
        r_lo = next(e[1] for e in entries if e[0] == "lo")
        r_hi = next(e[1] for e in entries if e[0] == "hi")
        d_lo = next(e[3]["pinn_derate_steps"] for e in entries if e[0] == "lo")
        d_hi = next(e[3]["pinn_derate_steps"] for e in entries if e[0] == "hi")
        phys.append({"factor": label, "kind": "thermal", "key": key,
                     "tmax_lo_k": t_lo, "tmax_hi_k": t_hi,
                     "derate_lo": d_lo, "derate_hi": d_hi,
                     "lcoe_lo_usd_mwh": r_lo["lcoe_usd_mwh"],
                     "lcoe_hi_usd_mwh": r_hi["lcoe_usd_mwh"],
                     "dlcoe_lo_pct": pct(r_lo["lcoe_usd_mwh"], base_lcoe),
                     "dlcoe_hi_pct": pct(r_hi["lcoe_usd_mwh"], base_lcoe)})
        print(f"  {label:>10}: Tmax ↓{lo} -> {t_lo:.1f} K, ↑{hi} -> {t_hi:.1f} K "
              f"(降载 {d_lo}/{d_hi} 步)")

    # ---- 自检 ----
    for e in econ_results:
        check(f"{e['factor']} 方向合理", e["dlcoe_lo_pct"] <= 0.0 <= e["dlcoe_hi_pct"]
              or e["dlcoe_hi_pct"] <= 0.0 <= e["dlcoe_lo_pct"],
              f"(lo {e['dlcoe_lo_pct']:+.1f}%, hi {e['dlcoe_hi_pct']:+.1f}%)")
    check("年电量 ±10% → LCOE 反向 1-6%（CAPEX 年金固定稀释）",
          econ_results[-1]["dlcoe_hi_pct"] < 0.0 and econ_results[-1]["dlcoe_lo_pct"] > 0.0
          and abs(econ_results[-1]["dlcoe_hi_pct"]) < 6.0,
          f"(hi {econ_results[-1]['dlcoe_hi_pct']:+.1f}%)")
    check("电池 CAPEX ±40% 影响有限（< 8%）",
          abs(econ_results[2]["dlcoe_hi_pct"]) < 8.0 and abs(econ_results[2]["dlcoe_lo_pct"]) < 8.0,
          f"(lo {econ_results[2]['dlcoe_lo_pct']:+.1f}%, hi {econ_results[2]['dlcoe_hi_pct']:+.1f}%)")
    check("7 天窗口降解影响 < 0.1% LCOE",
          abs(phys[0]["dlcoe_hi_pct"]) < 0.1 and abs(phys[0]["dlcoe_lo_pct"]) < 0.1,
          f"(×1.5 {phys[0]['dlcoe_hi_pct']:+.3f}%)")
    check("域内 T_in/λ_air/FU 变动在生产限值 1173 K 下降载 0 步",
          all(p["derate_lo"] == 0 and p["derate_hi"] == 0 for p in phys if p["kind"] == "thermal"),
          "(production limit)")
    check("T_in 降低→ Tmax 降低（裕度增加）",
          phys[1]["tmax_lo_k"] < phys[1]["tmax_hi_k"],
          f"({phys[1]['tmax_lo_k']:.1f} vs {phys[1]['tmax_hi_k']:.1f} K)")

    out = {"base_lcoe_usd_mwh": base_lcoe, "base_ghg_kg_7d": base_ghg,
           "tmax_limit_k": 1173.0, "base_tmax_max_k": base_max_t,
           "base_derate_steps": base_derate,
           "economic_factors": econ_results, "physical_factors": phys}
    with io.open(os.path.join(RES, "e3_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with io.open(os.path.join(RES, "e3_sensitivity.csv"), "w", encoding="utf-8", newline="") as f:
        lines = ["factor,dlcoe_lo_pct,dlcoe_hi_pct,note"]
        for e in econ_results:
            lines.append(f"{e['factor']},{e['dlcoe_lo_pct']:.2f},{e['dlcoe_hi_pct']:.2f},economic")
        for p in phys:
            lines.append(f"{p['factor']},{p['dlcoe_lo_pct']:.3f},{p['dlcoe_hi_pct']:.3f},physical")
        f.write("\n".join(lines))

    # ---- tornado 图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    items = econ_results + [p for p in phys if p["kind"] == "soh"]
    items = sorted(items, key=lambda e: max(abs(e["dlcoe_lo_pct"]), abs(e["dlcoe_hi_pct"])))
    ypos = np.arange(len(items))
    for i, e in enumerate(items):
        lo, hi = e["dlcoe_lo_pct"], e["dlcoe_hi_pct"]
        ax1.barh(i, max(lo, hi) - min(lo, hi), left=min(lo, hi), height=0.55,
                 color="tab:blue" if e["dlcoe_hi_pct"] >= 0 else "tab:red", alpha=0.8)
        ax1.text(lo, i, f"{lo:+.1f}", va="center", ha="right", fontsize=7.5)
        ax1.text(hi, i, f"{hi:+.1f}", va="center", ha="left", fontsize=7.5)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([e["factor"] for e in items], fontsize=9)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("LCOE 变动 (%)")
    ax1.set_title("E3 — 经济性/降解 tornado（7d）")
    ax1.grid(alpha=0.3, axis="x")

    th = [p for p in phys if p["kind"] == "thermal"]
    names = [p["factor"] for p in th]
    xs = np.arange(len(th))
    for i, p in enumerate(th):
        lo_v, hi_v = p["tmax_lo_k"], p["tmax_hi_k"]
        ax2.barh(i, hi_v - lo_v, left=lo_v, height=0.5, color="tab:orange", alpha=0.85)
        ax2.text(lo_v, i, f"{lo_v:.0f}", va="center", ha="right", fontsize=7.5)
        ax2.text(hi_v, i, f"{hi_v:.0f}", va="center", ha="left", fontsize=7.5)
    ax2.set_yticks(xs)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.axvline(1173.0, color="tab:red", ls="--", lw=1.3, label="生产限值 1173 K")
    ax2.axvline(base_max_t, color="tab:blue", ls=":", lw=1.2, label=f"基线 Tmax {base_max_t:.0f} K")
    ax2.set_xlabel("Tmax max (K)")
    ax2.set_title("E3 — 热管理因子对 Tmax 裕度")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "e3_sensitivity.png"), dpi=110)
    plt.close(fig)

    print(f"\n== 全部通过：{PASS} 项检查 OK ==\n")
    print(f"  Tmax max: base {base_max_t:.1f} K; 域内因子变动 "
          f"{min(p['tmax_lo_k'] for p in th):.1f}~{max(p['tmax_hi_k'] for p in th):.1f} K; "
          "生产限值下 0 降载")

if __name__ == "__main__":
    main()

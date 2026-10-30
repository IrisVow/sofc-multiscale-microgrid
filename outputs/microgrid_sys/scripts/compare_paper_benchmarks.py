# -*- coding: utf-8 -*-
"""E2 结果指标：与论文（Energy Conversion & Management 326 (2025) 119477）三档设计场景对照。

论文基准（游轮 5 年仿真，MGO+LNG 双燃料）：
- SOFC AUX   17.5% 装机占比 -> GHG -21%、污染物 -38~46%
- SOFC MAN   31%   -> GHG -33%、污染物 -60~70%
- SOFC CRUISE 51%  -> GHG -49%、污染物 -94~96%
本工程（海岛孤网 200 kW，7 天夏季最坏工况窗口）：
- 推荐 SOFC 150 kW + 电池 100 kWh + 柴油 100 kW（LNG）-> 42.9% 占比 -> GHG -48.7%

指标：GHG/NOx/SOx/PM 减排、燃料（LNG/MGO）、体积/重量（C5 sizing）、
LCOE/NPV/回收期。输出 results/capacity/e2_paper_benchmarks.json/.csv/.png。
运行：python scripts/compare_paper_benchmarks.py（工作目录 ROOT）
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

from src.components.sizing import SizingCalculator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "capacity")

# 论文三档基准（装机占比 %, GHG 减排 %, 污染物减排区间 %）
PAPER = [
    {"scenario": "SOFC AUX",    "share_pct": 17.5, "ghg_reduction_pct": 21, "pollutant_range": (38, 46)},
    {"scenario": "SOFC MAN",    "share_pct": 31.0, "ghg_reduction_pct": 33, "pollutant_range": (60, 70)},
    {"scenario": "SOFC CRUISE", "share_pct": 51.0, "ghg_reduction_pct": 49, "pollutant_range": (94, 96)},
]

LHV = {"LNG": 50000.0 / 3600.0, "MGO": 42800.0 / 3600.0}   # kWh/kg

PASS = 0

def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


def fuel_kg(kwh: float, fuel: str) -> float:
    return kwh / LHV[fuel]


def annualize(row: dict, key: str) -> float:
    return float(row[key]) * float(row.get("annual_scale", 1.0))


def main():
    with io.open(os.path.join(RES, "capacity_opt.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = {(float(r["sofc_kw"]), float(r["battery_kwh"]), float(r["genset_kw"])): r
            for r in data["rows"]}
    sel = data["selection"]
    base_bat = float(sel["baseline"]["battery_kwh"]) if sel.get("baseline") else 100.0
    base = rows[(0.0, base_bat, 250.0)]
    best = sel["best"]
    best_row = rows[(float(best["sofc_kw"]), float(best["battery_kwh"]),
                     float(best["genset_kw"]))]

    print("== E2 结果指标与论文基准对照 ==")

    # ---- 减排指标 ----
    ghg_red = 1.0 - best_row["ghg_kg_co2eq"] / base["ghg_kg_co2eq"]
    def pct_red(k):
        return 1.0 - best_row[k] / base[k]
    nox_red, sox_red, pm_red = pct_red("nox_kg"), pct_red("sox_kg"), pct_red("pm_kg")

    # ---- 燃料（7 天窗口）----
    fuel_lng_7d = best_row["fuel_sofc_kwh"] + best_row["fuel_genset_kwh"] + best_row["fuel_boiler_kwh"]
    fuel_mgo_7d = base["fuel_sofc_kwh"] + base["fuel_genset_kwh"] + base["fuel_boiler_kwh"]
    load_7d = best_row["energy_load_kwh"]
    spec_base = fuel_mgo_7d / load_7d            # kWh燃料/kWh电
    spec_best = fuel_lng_7d / load_7d
    fuel_eff_gain = spec_base / spec_best - 1.0  # 相对基线燃料效率提升

    # ---- 体积/重量（C5 sizing，7 天自持）----
    sz_best = SizingCalculator(overrides={"sofc_kw": float(best["sofc_kw"]),
                                          "genset_kw": float(best["genset_kw"]),
                                          "battery_kwh": float(best["battery_kwh"])}).compute()
    sz_base = SizingCalculator(overrides={"sofc_kw": 1e-3, "genset_kw": 250.0,
                                          "battery_kwh": base_bat}).compute()

    # ---- 经济性 ----
    # NPV 口径（2026-08-18 修订）：折现率取 economics.yaml 的 discount_rate（8%），
    # 项目期 n=20；此前误用 CRF（0.10185）作折现率导致 NPV20 系统性低估（903 -> ~1080 kUSD）。
    vs = best["vs_baseline"]
    r = float(best_row.get("discount_rate") or 0.08)
    n = int(best_row.get("project_years") or 20)
    annuity = (1.0 - (1.0 + r) ** -n) / r
    op_save = vs["annual_op_base_usd"] - vs["annual_op_usd"]
    npv_20y = -vs["incremental_capex_usd"] + op_save * annuity

    out = {
        "baseline": {
            "genset_kw": base["genset_kw"], "fuel": base["genset_fuel"],
            "ghg_kg_7d": base["ghg_kg_co2eq"], "nox_kg_7d": base["nox_kg"],
            "sox_kg_7d": base["sox_kg"], "pm_kg_7d": base["pm_kg"],
            "fuel_energy_kwh_7d": fuel_mgo_7d, "fuel_kg_7d": fuel_kg(fuel_mgo_7d, "MGO"),
            "specific_fuel_kwh_per_kwh": spec_base,
            "lcoe_usd_mwh": base["lcoe_usd_mwh"], "capex_usd": base["capex_usd"],
            "sizing": {"equipment_volume_m3": sz_base["equipment_volume_m3"],
                       "total_volume_m3": sz_base["total_volume_m3"],
                       "equipment_weight_ton": sz_base["equipment_weight_ton"],
                       "total_weight_ton": sz_base["total_weight_ton"]},
        },
        "recommended": {
            "sofc_kw": best_row["sofc_kw"], "battery_kwh": best_row["battery_kwh"],
            "genset_kw": best_row["genset_kw"], "fuel": best_row["genset_fuel"],
            "sofc_share_pct": 100.0 * best_row["sofc_kw"]
                             / (best_row["sofc_kw"] + best_row["genset_kw"]),
            "ghg_kg_7d": best_row["ghg_kg_co2eq"], "nox_kg_7d": best_row["nox_kg"],
            "sox_kg_7d": best_row["sox_kg"], "pm_kg_7d": best_row["pm_kg"],
            "fuel_energy_kwh_7d": fuel_lng_7d, "fuel_kg_7d": fuel_kg(fuel_lng_7d, "LNG"),
            "specific_fuel_kwh_per_kwh": spec_best,
            "lcoe_usd_mwh": best_row["lcoe_usd_mwh"], "capex_usd": best_row["capex_usd"],
            "annual_cost_usd": best_row["annual_cost_usd"],
            "sizing": {"equipment_volume_m3": sz_best["equipment_volume_m3"],
                       "total_volume_m3": sz_best["total_volume_m3"],
                       "equipment_weight_ton": sz_best["equipment_weight_ton"],
                       "total_weight_ton": sz_best["total_weight_ton"]},
        },
        "reductions": {
            "ghg_pct": ghg_red * 100.0, "nox_pct": nox_red * 100.0,
            "sox_pct": sox_red * 100.0, "pm_pct": pm_red * 100.0,
            "fuel_efficiency_gain_pct": fuel_eff_gain * 100.0,
        },
        "economics": {
            "lcoe_delta_pct": (best_row["lcoe_usd_mwh"] / base["lcoe_usd_mwh"] - 1.0) * 100.0,
            "incremental_capex_usd": vs["incremental_capex_usd"],
            "annual_op_saving_usd": op_save,
            "simple_payback_years": vs["simple_payback_years"],
            "npv_20y_usd": npv_20y,
        },
        "paper_benchmarks": [
            {"scenario": p["scenario"], "sofc_share_pct": p["share_pct"],
             "ghg_reduction_pct": p["ghg_reduction_pct"],
             "pollutant_reduction_range": list(p["pollutant_range"])}
            for p in PAPER
        ],
    }

    # ---- 自检 ----
    print(f"  推荐 SOFC {best_row['sofc_kw']:.0f} kW 占比 {out['recommended']['sofc_share_pct']:.1f}%")
    check("污染物减排：NOx/SOx/PM 均 > 90%",
          nox_red > 0.9 and sox_red > 0.9 and pm_red > 0.9,
          f"(NOx {nox_red*100:.1f}% / SOx {sox_red*100:.1f}% / PM {pm_red*100:.1f}%)")
    check("GHG 减排超过 IMO 2030 目标 30%", ghg_red > 0.30, f"(GHG {ghg_red*100:.1f}%)")
    check("与论文 51% 装机档（GHG -49%）同量级（40-60%）",
          0.40 <= ghg_red <= 0.60, f"(GHG {ghg_red*100:.1f}%)")
    check("燃料效率提升在论文 19-59% 区间", 0.19 <= fuel_eff_gain <= 0.59,
          f"(+{fuel_eff_gain*100:.1f}%)")
    check("LCOE 降低且 NPV20 > 0", best_row["lcoe_usd_mwh"] < base["lcoe_usd_mwh"]
          and npv_20y > 0, f"(LCOE {best_row['lcoe_usd_mwh']:.1f} vs {base['lcoe_usd_mwh']:.1f}, NPV {npv_20y/1000:.0f} kUSD)")
    check("体积重量可行（推荐方案设备体积 < 20 m3）",
          sz_best["equipment_volume_m3"] < 20.0,
          f"(eq_vol {sz_best['equipment_volume_m3']:.1f} m3, total {sz_best['total_volume_m3']:.1f} m3)")

    # ---- 输出 ----
    with io.open(os.path.join(RES, "e2_paper_benchmarks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with io.open(os.path.join(RES, "e2_paper_benchmarks.csv"), "w", encoding="utf-8", newline="") as f:
        lines = ["metric,value,unit"]
        for k, v in sorted(out["reductions"].items()):
            lines.append(f"reductions.{k},{v:.2f},%")
        for k, v in sorted(out["economics"].items()):
            lines.append(f"economics.{k},{v:.4f},{'usd' if 'usd' in k else ''}")
        lines.append(f"recommended.sofc_share_pct,{out['recommended']['sofc_share_pct']:.2f},%")
        lines.append(f"recommended.total_volume_m3,{sz_best['total_volume_m3']:.2f},m3")
        lines.append(f"recommended.total_weight_ton,{sz_best['total_weight_ton']:.2f},ton")
        lines.append(f"recommended.lcoe_usd_mwh,{best_row['lcoe_usd_mwh']:.2f},usd/mwh")
        lines.append(f"recommended.ghg_kg_7d,{best_row['ghg_kg_co2eq']:.1f},kg")
        f.write("\n".join(lines))

    # ---- 对比图 ----
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    xs = [p["share_pct"] for p in PAPER] + [out["recommended"]["sofc_share_pct"]]
    ys = [p["ghg_reduction_pct"] for p in PAPER] + [out["reductions"]["ghg_pct"]]
    ax[0].plot([p["share_pct"] for p in PAPER], [p["ghg_reduction_pct"] for p in PAPER],
               "o-", color="tab:blue", label="论文设计场景")
    for p in PAPER:
        ax[0].annotate(p["scenario"].replace("SOFC ", ""), (p["share_pct"], p["ghg_reduction_pct"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[0].plot(out["recommended"]["sofc_share_pct"], out["reductions"]["ghg_pct"], "s",
               color="tab:red", ms=9, label="本工程推荐")
    ax[0].axhline(30, color="tab:green", ls="--", lw=1.2, label="IMO 2030 目标 30%")
    ax[0].set_xlabel("SOFC installed power share (%)")
    ax[0].set_ylabel("GHG reduction vs MGO gensets (%)")
    ax[0].set_title("E2 — GHG 减排 vs 论文基准")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    names = ["GHG", "NOx", "SOx", "PM"]
    vals = [ghg_red * 100, nox_red * 100, sox_red * 100, pm_red * 100]
    bars = ax[1].bar(names, vals, color=["tab:red", "tab:orange", "tab:purple", "tab:brown"])
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax[1].set_ylim(0, 105)
    ax[1].set_ylabel("减排比例 (%)")
    ax[1].set_title("E2 — 推荐方案排放减排（7d）")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "e2_paper_benchmarks.png"), dpi=110)
    plt.close(fig)

    print(f"\n== 全部通过：{PASS} 项检查 OK ==\n")
    print(f"  GHG {ghg_red*100:.1f}% / NOx {nox_red*100:.1f}% / SOx {sox_red*100:.1f}% / PM {pm_red*100:.1f}%")
    print(f"  燃料效率 +{fuel_eff_gain*100:.1f}%  年燃料 LNG {annualize(best_row,'fuel_sofc_kwh')+annualize(best_row,'fuel_genset_kwh'):,.0f} kWh")
    print(f"  体积 {sz_best['equipment_volume_m3']:.1f} m3（总 {sz_best['total_volume_m3']:.1f} m3）  重量 {sz_best['equipment_weight_ton']:.1f} t（总 {sz_best['total_weight_ton']:.1f} t）")
    print(f"  LCOE {best_row['lcoe_usd_mwh']:.1f} vs {base['lcoe_usd_mwh']:.1f} $/MWh  NPV20 {npv_20y/1000:,.0f} kUSD  回收期 {vs['simple_payback_years']:.2f} y")


if __name__ == "__main__":
    main()

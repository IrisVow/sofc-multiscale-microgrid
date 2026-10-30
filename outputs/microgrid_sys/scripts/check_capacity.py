# -*- coding: utf-8 -*-
"""E1 容量配置优化自检（小网格 + 手算经济性 + PINN 复核推荐方案）。
运行：python scripts/check_capacity.py（工作目录 ROOT）
通过时逐项打印 OK；任一断言失败退出码非 0。

校验项（小网格 sofc∈{0,100,150} kW × bat∈{100,500} kWh）：
- 纯柴油基线（250 kW MGO 机组）可靠（LOLP=0）且 GHG 高于所有 SOFC 方案；
- LOLP/EENS 与 df 手算一致；SOC 在 [0.2,0.9]；
- 经济性：CRF 手算（8%/20y）、CAPEX 手算、LCOE = 年成本/846.3 MWh；
- 确定性（同点重跑一致）；select 取可行最小年成本；
- PINN 复核推荐方案：Tmax 生产限值 1173 K 下违反 0 步、降载 0 步。
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from src.analysis.capacity import CapacityOptimizer
from src.analysis.load_profile import LoadProfile
from src.ems import Microgrid

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6

opt = CapacityOptimizer()
print("== E1 容量优化（小网格：sofc {0,100,150} kW x bat {100,500} kWh）==")
rows = [opt.run_point(s, b) for s in (0.0, 100.0, 150.0) for b in (100.0, 500.0)]

# ---- 可靠性指标手算对照 ----
for r in rows:
    prof = LoadProfile(os.path.join(ROOT, "configs", "scenario.yaml"))
    p_load, q_load = prof.generate_series(days=7.0, dt_s=60.0, season="summer",
                                          weekend=False, time_index=False)
    ov = {"sofc": {"p_nom_kw": r["sofc_kw"] if r["sofc_kw"] > 0 else 1e-3},
          "battery": {"e_nom_kwh": r["battery_kwh"]}}
    if r["sofc_kw"] <= 0:
        ov["genset"] = {"p_nom_kw": r["genset_kw"]}
    mg = Microgrid(component_overrides=ov, genset_fuel=r["genset_fuel"])
    mg.run_days(7.0, np.asarray(p_load, dtype=float), np.asarray(q_load, dtype=float), dt_s=60.0)
    df = mg.to_dataframe()
    deficit = np.maximum(0.0, df["p_load_kw"].values - df["net_kw"].values)
    dt_h = 60.0 / 3600.0
    check(f"LOLP/EENS 手算一致（SOFC {r['sofc_kw']:.0f} / bat {r['battery_kwh']:.0f}）",
          int((deficit > 1.0).sum()) == r["lolp_steps"]
          and abs(float(deficit.sum()) * dt_h - r["eens_kwh"]) < TOL,
          f"(lolp={r['lolp_steps']}, eens={r['eens_kwh']:.3f})")

base = next(r for r in rows if r["sofc_kw"] == 0.0)
check("纯柴油基线可靠（250 kW MGO 机组，LOLP=0）", base["lolp_frac"] == 0.0
      and base["genset_kw"] == 250.0 and base["genset_fuel"] == "MGO",
      f"(genset={base['genset_kw']:.0f} kW, lolp={base['lolp_frac']:.4f})")
check("基线 GHG 高于所有 SOFC 方案", all(base["ghg_kg_co2eq"] > r["ghg_kg_co2eq"]
      for r in rows if r["sofc_kw"] > 0),
      f"(base={base['ghg_kg_co2eq']:.0f}, min={min(r['ghg_kg_co2eq'] for r in rows if r['sofc_kw'] > 0):.0f})")
check("SOC 均在 [0.2, 0.9]", all(0.2 - 1e-6 <= r["soc_min"] and r["soc_max"] <= 0.9 + 1e-6
      for r in rows))

# ---- 经济性手算 ----
crf_exp = 0.08 * 1.08 ** 20 / (1.08 ** 20 - 1.0)
check("CRF 手算（8% / 20y）", abs(rows[0]["crf"] - crf_exp) < TOL, f"(crf={rows[0]['crf']:.5f})")
r100 = next(r for r in rows if r["sofc_kw"] == 100.0 and r["battery_kwh"] == 100.0)
capex_exp = 100 * 1500 + 100 * 500 + 100 * 350
check("CAPEX 手算（SOFC 100 + 柴油 100 + 电池 100 kWh）= 235 kUSD",
      abs(r100["capex_usd"] - capex_exp) < TOL, f"(={r100['capex_usd']:.0f})")
check("LCOE = 年总成本 / 846.3 MWh", abs(r100["lcoe_usd_mwh"]
      - r100["annual_cost_usd"] / 846.3) < 1e-6, f"(={r100['lcoe_usd_mwh']:.1f})")

# ---- 确定性 ----
r_a = opt.run_point(100.0, 100.0)
r_b = opt.run_point(100.0, 100.0)
check("确定性（同点重跑一致）", abs(r_a["energy_sofc_kwh"] - r_b["energy_sofc_kwh"]) < 1e-6)

# ---- 选择 ----
sel = opt.select(rows)
check("select：全部可行 且 best = 最小年成本", sel["n_feasible"] == len(rows)
      and sel["best"]["annual_cost_usd"] == min(r["annual_cost_usd"] for r in rows),
      f"(best SOFC {sel['best']['sofc_kw']:.0f} / bat {sel['best']['battery_kwh']:.0f})")
check("best vs 基线：GHG 减排 > 0 且回收期为正", sel["best"]["vs_baseline"]["ghg_reduction_frac"] > 0
      and sel["best"]["vs_baseline"]["simple_payback_years"] > 0,
      f"(ΔGHG={sel['best']['vs_baseline']['ghg_reduction_frac'] * 100:.1f}%, "
      f"payback={sel['best']['vs_baseline']['simple_payback_years']:.2f} y)")

# ---- 导出 ----
fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="check_capacity_")
os.close(fd)
os.remove(tmp)
try:
    csv_p = opt.to_csv(rows, tmp)
    with io.open(tmp, "r", encoding="utf-8") as f:
        content = f.read()
    check("CSV 导出可运行（表头 + 6 行）", os.path.exists(csv_p) and content.startswith("sofc_kw")
          and len(content.splitlines()) == 7, f"(lines={len(content.splitlines())})")
finally:
    if os.path.exists(tmp):
        os.remove(tmp)

# ---- PINN 复核推荐方案 ----
from src.coupling import PinnCoupling
coup = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate"})
r_opt = opt.run_point(sel["best"]["sofc_kw"], sel["best"]["battery_kwh"], coupling=coup)
cs = coup.summary()
check("PINN 复核推荐方案：Tmax 生产限值 1173 K 违反 0 步、降载 0 步",
      cs["pinn_tmax_violations"] == 0 and cs["pinn_derate_steps"] == 0,
      f"(viol={cs['pinn_tmax_violations']}, max={cs['pinn_max_tmax_est_k']:.1f} K, "
      f"derate={cs['pinn_derate_steps']} steps)")

print(f"\n== 全部通过：{PASS} 项检查 OK ==")
# -*- coding: utf-8 -*-
"""D4 电堆/系统放大律自检（单电池 → 电堆 → 系统 → 装机/几何/热 + 敏感性）。
运行：python scripts/check_scaling.py（工作目录 ROOT）
通过时逐项打印 OK；任一断言失败退出码非 0。

校验项（默认配置，手算对照，容差 1e-6）：
- 电池：节距 2.5 mm（W_ch 1.5 + W_rib 1.0）、40 通道、A_cell=100 cm²、
  I_nom=18 A、P_cell=15.3 W DC。
- 电堆：V_target=400 V → N_series=471、V_stack=400.35 V、P_stack≈7.21 kW DC。
- 系统：P_dc_req=150/(0.96·0.98·0.95)≈167.83 kW；N_cells_total=10970；
  N_parallel=24、N_cells_installed=11304；P_net_ac≥150 kW、利用率≈0.970≤1。
- 几何：堆高 471×3 mm=1.413 m；堆芯体积 24 堆≈0.441 m³；系统包络 10 m³
  （Table 2），堆芯占比 3–8%。
- 热：燃料预算 vs 0D 名义点≈1.074（=1/(1−bop)+取整）；后燃热=P_fuel×(1−U_f)、
  回收热=P_fuel×η_heat(1.0) 量级自洽。
- 敏感性：P_nom×1.5 → 电池数 ≈1.46×（取整）；i_nom ±20% → 电池数 ∓20%
  （线性）；V_stack 200/600 V 改变串并联配比；bop 3–8% 改变装机。
"""
from __future__ import annotations

import io
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.components import StackScaling

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


TOL = 1e-6

calc = StackScaling()
res = calc.compute()
cell, stk, sys_, th = res["cell"], res["stack"], res["system"], res["thermal"]

print("== D4 电池级（PINN/COMSOL DOE 几何 → 电化学放大）==")
check("节距 = W_ch + W_rib = 2.5 mm", abs(cell["pitch_m"] - 2.5e-3) < TOL,
      f"(={cell['pitch_m'] * 1e3:.2f} mm)")
check("全电池 40 通道周期（10 cm / 2.5 mm）", abs(cell["n_channels"] - 40.0) < TOL,
      f"(={cell['n_channels']:.0f})")
check("A_cell = L×W = 100 cm²（0.1 m × 0.1 m）", abs(cell["a_cell_cm2"] - 100.0) < TOL,
      f"(={cell['a_cell_cm2']:.1f} cm²)")
check("I_nom = 0.18 A/cm² × 100 cm² = 18 A", abs(cell["i_cell_A"] - 18.0) < TOL,
      f"(={cell['i_cell_A']:.3f} A)")
check("P_cell = 0.85 V × 18 A = 15.3 W DC", abs(cell["p_cell_dc_w"] - 15.3) < TOL,
      f"(={cell['p_cell_dc_w']:.3f} W)")

print("== D4 电堆级（V_target → 串联节数）==")
check("N_series = round(400/0.85) = 471", stk["n_series"] == 471, f"(={stk['n_series']})")
check("V_stack = 471 × 0.85 = 400.35 V", abs(stk["v_stack_dc"] - 400.35) < TOL,
      f"(={stk['v_stack_dc']:.3f} V)")
check("P_stack = 471 × 15.3 W = 7.2063 kW DC", abs(stk["p_stack_dc_kW"] - 471 * 0.0153) < TOL,
      f"(={stk['p_stack_dc_kW']:.4f} kW)")

print("== D4 系统级（P_nom → 并联堆数/装机/净功率）==")
p_dc_req_exp = 150.0 / 0.96 / 0.98 / 0.95
check("P_dc_req = 150/(η_inv·η_dcdc·(1−bop)) ≈ 167.83 kW", abs(sys_["p_dc_req_kW"] - p_dc_req_exp) < TOL,
      f"(={sys_['p_dc_req_kW']:.3f} kW)")
n_total_exp = math.ceil(p_dc_req_exp / 0.0153)
check("N_cells_total = ceil(P_dc_req/P_cell) = 10970", sys_["n_cells_total"] == n_total_exp,
      f"(={sys_['n_cells_total']})")
check("N_parallel = ceil(10970/471) = 24", sys_["n_parallel"] == 24, f"(={sys_['n_parallel']})")
check("N_cells_installed = 24 × 471 = 11304", sys_["n_cells_installed"] == 11304,
      f"(={sys_['n_cells_installed']})")
check("P_net_ac ≥ 150 kW 且利用率 ≤ 1", sys_["p_net_ac_kW"] >= 150.0 and sys_["utilization"] <= 1.0,
      f"(P_net_ac={sys_['p_net_ac_kW']:.2f} kW, util={sys_['utilization']:.4f})")
check("净 AC 精确 = P_dc×(1−bop)×η_dcdc×η_inv", abs(sys_["p_net_ac_kW"]
      - sys_["p_installed_dc_kW"] * (1.0 - sys_["bop_frac"]) * sys_["eta_dcdc"] * sys_["eta_inv"]) < TOL)
check("cells/kW = 11304/150 = 75.36", abs(sys_["cells_per_kw"] - 11304 / 150.0) < TOL,
      f"(={sys_['cells_per_kw']:.2f})")

print("== D4 几何 / Table 2 交叉核对 ==")
check("堆高 = 471 × 3 mm = 1.413 m", abs(stk["height_m"] - 1.413) < TOL,
      f"(={stk['height_m']:.3f} m)")
vol_exp = 24 * (1.413 * (0.01 * 1.3))
check("堆芯体积（24 堆 × 1.413×0.013 m³）≈ 0.441 m³", abs(sys_["plant_stack_volume_m3"] - vol_exp) < TOL,
      f"(={sys_['plant_stack_volume_m3']:.4f} m³)")
check("系统包络 150 kW → 10 m³ / 6 ton（Table 2）",
      abs(sys_["envelope_volume_m3"] - 10.0) < TOL and abs(sys_["envelope_weight_ton"] - 6.0) < TOL,
      f"(V={sys_['envelope_volume_m3']:.2f}, W={sys_['envelope_weight_ton']:.2f})")
check("堆芯占系统包络 3–8%（工程经验 5–10% 下界）", 0.03 <= sys_["stack_core_share"] <= 0.08,
      f"(={sys_['stack_core_share']:.4f})")

print("== D4 热/燃料（0D 交叉核对）==")
check("燃料预算 vs 0D 名义点 ≈ 1.074（=1/(1−bop)+取整裕度）",
      1.05 <= th["fuel_budget_ratio_vs_0d"] <= 1.10, f"(={th['fuel_budget_ratio_vs_0d']:.4f})")
check("后燃热 = P_fuel×(1−U_f)（U_f=0.72）", abs(th["q_afterburner_kW"]
      - th["p_fuel_req_kW"] * (1.0 - th["u_f_nominal"])) < TOL, f"(={th['q_afterburner_kW']:.1f} kW)")
check("回收热 = P_fuel×η_heat(1.0)=0.25", abs(th["q_recover_kW"]
      - th["p_fuel_req_kW"] * 0.25) < TOL, f"(={th['q_recover_kW']:.1f} kW)")
check("回收热与 C4 热负荷量级匹配（60–80 kW 热峰附近）", 60.0 <= th["q_recover_kW"] <= 100.0,
      f"(={th['q_recover_kW']:.1f} kW)")

print("== D4 敏感性 ==")
sens = res["sensitivity"]
check("P_nom×1.5 → 电池数 ≈1.46×（取整影响）", 1.44 <= sens["p_nom_1p5x"]["ratio_n_cells"] <= 1.48,
      f"(={sens['p_nom_1p5x']['ratio_n_cells']:.4f})")
check("i_nom −20% → 电池数 ×1.25（线性）", 1.24 <= sens["i_nom_minus20pct"]["ratio_n_cells"] <= 1.26,
      f"(={sens['i_nom_minus20pct']['ratio_n_cells']:.4f})")
check("i_nom +20% → 电池数 ×0.833（线性）", 0.82 <= sens["i_nom_plus20pct"]["ratio_n_cells"] <= 0.84,
      f"(={sens['i_nom_plus20pct']['ratio_n_cells']:.4f})")
check("V_stack 200 V → N_series=235、并联 47", sens["v_stack_200v"]["n_series"] == 235
      and sens["v_stack_200v"]["n_parallel"] == 47,
      f"(series={sens['v_stack_200v']['n_series']}, par={sens['v_stack_200v']['n_parallel']})")
check("V_stack 600 V → N_series=706、并联 16", sens["v_stack_600v"]["n_series"] == 706
      and sens["v_stack_600v"]["n_parallel"] == 16,
      f"(series={sens['v_stack_600v']['n_series']}, par={sens['v_stack_600v']['n_parallel']})")
check("bop 3% 减装、8% 增装（单调）", sens["bop_3pct"]["n_cells"] < sys_["n_cells_installed"]
      < sens["bop_8pct"]["n_cells"],
      f"(3%={sens['bop_3pct']['n_cells']}, base={sys_['n_cells_installed']}, 8%={sens['bop_8pct']['n_cells']})")

print("== CSV 导出 ==")
fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="check_scaling_")
os.close(fd)
try:
    out = calc.to_csv(tmp)
    with io.open(tmp, "r", encoding="utf-8") as f:
        content = f.read()
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
check("CSV 导出可运行（写读回：表头 + 23 行）", out == tmp and content.startswith("key,value,unit")
      and len(content.splitlines()) == 24, f"(lines={len(content.splitlines())})")

# ---------------------------------------------------------------------------
print("\n== D4 标定汇总 ==")
print(f"  单电池            : {cell['a_cell_cm2']:.0f} cm²，{cell['i_cell_A']:.1f} A @ {cell['i_nom_A_cm2']:.2f} A/cm²，"
      f"{cell['p_cell_dc_w']:.1f} W")
print(f"  电堆（{stk['n_series']} 节串联）: {stk['v_stack_dc']:.0f} V DC，{stk['p_stack_dc_kW']:.1f} kW，"
      f"高 {stk['height_m']:.2f} m")
print(f"  系统（{sys_['n_parallel']} 堆并联）: {sys_['n_cells_installed']} 节 / {sys_['p_installed_dc_kW']:.1f} kW DC "
      f"-> {sys_['p_net_ac_kW']:.1f} kW AC（util={sys_['utilization']:.3f}，{sys_['cells_per_kw']:.1f} cells/kW）")
print(f"  几何              : 堆芯 {sys_['plant_stack_volume_m3']:.2f} m³ + BOP "
      f"-> {sys_['plant_volume_m3']:.2f} m³；Table 2 包络 {sys_['envelope_volume_m3']:.1f} m³"
      f"（堆芯占 {sys_['stack_core_share'] * 100:.1f}%）")
print(f"  燃料/热（名义点）  : P_fuel {th['p_fuel_req_kW']:.0f} kW（vs 0D {th['p_fuel_0d_kW']:.0f} kW，"
      f"×{th['fuel_budget_ratio_vs_0d']:.3f}）；后燃 {th['q_afterburner_kW']:.0f} kW / 回收 {th['q_recover_kW']:.0f} kW")
print(f"\n== 全部通过：{PASS} 项检查 OK ==")
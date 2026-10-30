# -*- coding: utf-8 -*-
"""C2 组件模型自检（论文 Eq.8/9/10/11/12 + Table 2/3 参数）。
运行：python scripts/check_components.py
通过时逐项打印 OK；任一断言失败退出码非 0。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.components import (SOFCSystem, Reformer, Genset, Battery, Boiler,
                            EmissionsBook, CHEM, PARAMS)

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


# ---------------------------------------------------------------------------
print("== SOFC: Eq.9/10 V-I / 名义电压 ==")
sofc = SOFCSystem()
p_nom = sofc.p_nom
i_nom = sofc.current_for_power(1.0)
v_nom = sofc.v_cell(i_nom)
check("V_cell(P_nom, BOL) = 0.85 V", abs(v_nom - 0.85) < 1e-6, f"(got {v_nom:.4f})")
check("i_pu(P_nom) = 1.0（恒功率名义点）", abs(i_nom - 1.0) < 1e-9, f"(got {i_nom:.4f})")
check("P = V*I 回代（Eq.10 自洽）",
      abs(sofc.power_for_current(i_nom) - 1.0) < 1e-9,
      f"(P_pu={sofc.power_for_current(i_nom):.4f})")
check("SOH(BOL) = 1.0", abs(sofc.soh() - 1.0) < 1e-9, f"(got {sofc.soh():.4f})")

print("== SOFC: EOL 电压 / 效率缩放（Eq.12，恒功率定义）==")
sofc2 = SOFCSystem()
sofc2.r_deg = sofc2.r_deg_eol                 # 推到 EOL（V/pu）
i_eol = sofc2.current_for_power(1.0)
v_eol = sofc2.v_cell(i_eol)
check("V_cell(P_nom, EOL) = 0.71 V", abs(v_eol - 0.71) < 1e-6, f"(got {v_eol:.4f})")
check("EOL 电流上升（恒功率维持）", i_eol > 1.0, f"(I_pu={i_eol:.4f})")
check("SOH(EOL) = 0", abs(sofc2.soh()) < 1e-9, f"(got {sofc2.soh():.4f})")
eta_ratio = sofc2.eta_net(1.0) / sofc.eta_net(1.0)
check("EOL/BOL 效率 = V_EOL/V_BOL", abs(eta_ratio - 0.71 / 0.85) < 1e-6,
      f"(ratio={eta_ratio:.4f})")

print("== SOFC: 部分负荷效率峰值（Fig.3 结构）==")
e10, e50, e100 = sofc.eta_net(0.10), sofc.eta_net(0.50), sofc.eta_net(1.00)
check("eta(0.5) > eta(0.1)", e50 > e10, f"({e50:.3f} vs {e10:.3f})")
check("eta(0.5) > eta(1.0)", e50 > e100, f"({e50:.3f} vs {e100:.3f})")
check("eta 处于 40-62% 区间", 0.40 <= e50 <= 0.62, f"(eta50={e50:.3f})")
check("CHP: eta_el+eta_heat(0.5) <= 0.92",
      sofc.eta_net(0.5) + sofc.eta_heat(0.5) <= 0.92,
      f"({sofc.eta_net(0.5)+sofc.eta_heat(0.5):.3f})")

print("== SOFC: 速率限制（Eq.8，R=2%/min，F=8%/min，作用在电流）==")
sofc3 = SOFCSystem()
r1 = sofc3.step(60.0, p_nom)
check("60s 升载 i_pu <= 2%", r1["i_pu"] <= 0.02 + 1e-9, f"(i_pu={r1['i_pu']:.4f})")
r2 = sofc3.step(60.0, p_nom)
check("120s 升载 i_pu <= 4%", r2["i_pu"] <= 0.04 + 1e-9, f"(i_pu={r2['i_pu']:.4f})")
sofc4 = SOFCSystem()
sofc4.i_pu = sofc4.current_for_power(1.0)
r3 = sofc4.step(60.0, 0.0)
check("60s 降载 i_pu >= 1-8%", r3["i_pu"] >= 1.0 - 0.08 - 1e-9, f"(i_pu={r3['i_pu']:.4f})")

print("== SOFC: 降解积分（Eq.11 -> R_deg -> SOH）==")
sofc5 = SOFCSystem()
sofc5.step(60.0, p_nom)
soh0 = sofc5.soh()
for _ in range(10):
    sofc5.step(3600.0, p_nom)                 # 10 h 名义运行
check("SOH 单调下降", sofc5.soh() < soh0, f"({soh0:.5f} -> {sofc5.soh():.5f})")
# 30k h 名义运行（粗步长 100h）：dSOH 由 Eq.11 常数（0.0573, 1.25）标定。
# i_nom=0.18 A/cm2 -> r_rate=0.0718 %/1000h -> dSOH≈0.18；
# 对应 40-50k h 更换时 SOH≈0.75-0.80，与论文 Table 7 一致（5 年 0.08-0.47 为高动态工况）。
sofc6 = SOFCSystem()
sofc6.step(60.0, p_nom)
for _ in range(300):
    sofc6.step(100.0 * 3600.0, p_nom)
soh_30k = sofc6.soh()
check("30k h 名义运行 SOH 降幅 12-25%", 0.12 <= (1.0 - soh_30k) <= 0.25,
      f"(SOH={soh_30k:.4f}, dSOH={1-soh_30k:.4f})")
check("运行小时累计正确", abs(sofc6.t_oper_h - 30000.0) < 100.0,
      f"(t_oper={sofc6.t_oper_h:.0f} h)")

print("== SOFC: 瞬态降解因子 a_trans = 1.5 ==")
r = sofc5.deg_rate(0.18, 0.0)
rt = sofc5.deg_rate(0.18, 0.10)
check("a_trans 施加", abs(rt / r - 1.5) < 1e-9, f"({r:.4f} -> {rt:.4f} %/1000h)")

print("== Reformer: 物料/能量守恒 ==")
ref = Reformer()
out = ref.run(100.0)                          # 100 kW CH4
n_ch4 = 100.0 * 1000.0 / (CHEM["LHV_CH4_kJ_kg"] * CHEM["M_CH4_kg_kmol"])
check("n_h2 = 4 * n_ch4 * eta_reform",
      abs(out["n_h2_kmol_s"] - 4.0 * n_ch4 * ref.eta_reform) < 1e-12)
check("H2 功率 <= CH4 功率*1.21", out["p_h2_kw"] <= 100.0 * 1.21,
      f"(p_h2={out['p_h2_kw']:.1f} kW, 理论上限~121 kW)")
check("蒸汽需求 S/C=2", abs(out["n_h2o_kmol_s"] / n_ch4 - ref.sc) < 1e-12)
back = ref.h2_flow_for_power(out["p_h2_kw"])
check("h2_flow 逆变换自洽", abs(back - 100.0) < 1e-6, f"(back={back:.2f} kW)")

print("== Genset: 最低负荷 / 速率 / 效率 ==")
g = Genset(p_nom_kw=100.0, fuel="MGO")
r = g.step(60.0, 5.0)                         # 5% < LL -> 停机
check("低于 10% 最低负荷归零", r["p_kw"] == 0.0 and r["curtail_kw"] == 5.0,
      f"(p={r['p_kw']}, curt={r['curtail_kw']})")
g2 = Genset(p_nom_kw=100.0, fuel="MGO")
r2 = g2.step(60.0, 100.0)
check("MGO 升载 <= 2.7%P/s x 60s（饱和在 P_nom）", r2["p_kw"] == 100.0,
      f"(p={r2['p_kw']:.1f})")
check("效率在 20-45% 区间", 0.20 <= r2["eta"] <= 0.45, f"(eta={r2['eta']:.3f})")
g3 = Genset(p_nom_kw=100.0, fuel="LNG")
r3 = g3.step(1.0, 100.0)
check("LNG 升载 <= 1.4%P/s", r3["p_kw"] <= 1.4 + 1e-6, f"(p={r3['p_kw']:.2f})")

print("== Battery: 效率 / C-rate / SOC 边界 ==")
bat = Battery({"e_nom_kwh": 100.0})
lim = bat.limits(360.0)                       # 6 min 步长下 C-rate 主导
check("C-rate 限 300 kW（100 kWh x 3）", abs(lim["dis_max"] - 300.0) < 1e-6,
      f"(dis_max={lim['dis_max']:.1f})")
r = bat.step(3600.0, 10.0)                    # 放 10 kW x 1h
check("放电后 SOC 下降", r["soc"] < 0.55, f"(soc={r['soc']:.4f})")
r2 = bat.step(3600.0, -10.0)                  # 充回 10 kW x 1h
check("充放往返存在损耗", r2["soc"] < 0.55, f"(soc={r2['soc']:.4f})")
bat2 = Battery({"e_nom_kwh": 100.0, "soc_init": 0.95, "soc_ub": 0.90})
r3 = bat2.step(3600.0, -50.0)               # 充电指令；已在 UB 之上 -> 拒绝充电并钳位到 UB
check("SOC 上界钳位（0.95 超界被夹到 0.90）", abs(r3["soc"] - 0.90) < 1e-9,
      f"(soc={r3['soc']:.4f})")
bat3 = Battery({"e_nom_kwh": 100.0, "soc_init": 0.10, "soc_lb": 0.20})
r4 = bat3.step(3600.0, 200.0)                   # 放电指令；已在 LB 之下 -> 拒绝放电并钳位到 LB
check("SOC 下界钳位（0.10 超界被夹到 0.20）", abs(r4["soc"] - 0.20) < 1e-9,
      f"(soc={r4['soc']:.4f})")

print("== Boiler: 热平衡 ==")
b = Boiler()
r = b.step(q_demand_kw=100.0, q_recovered_kw=60.0, fuel="LNG")
check("回收热先满足（经换热器）", abs(r["q_supplied_kw"] - 60.0 * 0.85) < 1e-9,
      f"(supplied={r['q_supplied_kw']:.2f})")
check("缺口由锅炉补足（eta=0.90）",
      abs(r["q_boiler_kw"] - (100.0 - 60.0 * 0.85) / 0.90) < 1e-9,
      f"(boiler={r['q_boiler_kw']:.2f})")

print("== Emissions: GHG / 因子 ==")
book = EmissionsBook()
row = book.add_fuel("test_mgo", "MGO", 1.0)   # 1 kWh MGO 燃料
check("MGO CO2 ~ 269.6 g/kWh 燃料",
      abs(row["co2_g"] - 3.206 / 42.8e3 * 3.6e6) < 1.0, f"(co2={row['co2_g']:.1f} g)")
row2 = book.add_fuel("test_lng", "LNG", 1.0)
check("LNG CO2 ~ 198 g/kWh 燃料",
      abs(row2["co2_g"] - 2.75 / 50.0e3 * 3.6e6) < 1.0, f"(co2={row2['co2_g']:.1f} g)")
row3 = book.add_fuel("test_lng_slip", "LNG", 1.0, ch4_slip=0.035)
ghg_add = row3["ch4_g"] * book.gwp_ch4
check("CH4 滑移 3.5% -> GHG 增量 = 25 x 1.4 g",
      abs(ghg_add - 25.0 * 0.035 * (1.0 / 50.0e3 * 3.6e6)) < 1.0,
      f"(ghg_add={ghg_add:.1f} g)")

# 燃料强度对照：MGO vs LNG（LNG 应更低）
book2 = EmissionsBook()
book2.add_fuel("a", "MGO", 1000.0, ch4_slip=0.0)
book2.add_fuel("b", "LNG", 1000.0, ch4_slip=0.0)
s = book2.summary()
check("LNG CO2 强度 < MGO（~30% 降幅）",
      book2.sources["b"]["co2_g"] < book2.sources["a"]["co2_g"],
      f"(LNG {book2.sources['b']['co2_g']/1000:.1f} kg vs MGO {book2.sources['a']['co2_g']/1000:.1f} kg)")

print(f"\n== 全部通过：{PASS} 项检查 OK ==")
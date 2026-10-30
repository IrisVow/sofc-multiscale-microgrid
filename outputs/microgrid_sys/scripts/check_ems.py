# -*- coding: utf-8 -*-
"""C3 EMS 与整定自检（论文 Eq.17-20 + 层级调度 + 热平衡 + 7 天整定仿真）。
运行：python scripts/check_ems.py
通过时逐项打印 OK；任一断言失败退出码非 0。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.ems import EMS, Microgrid

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


P_SOFC = 150.0
P_GENSET = 100.0

print("== EMS: SOC 控制（Eq.20 + 双滞环）==")
ems = EMS()
a_low = ems.p_add(0.30)                       # SOC 低 -> 充电
check("低 SOC 充电（P_add>0）", a_low > 0, f"(p_add={a_low:.1f})")
check("P_add 限幅", abs(a_low) <= ems.p_add_max_kw + 1e-9, f"(p_add={a_low:.1f})")
a_high = ems.p_add(0.80)                      # SOC 高 -> 放电
check("高 SOC 放电（P_add<0）", a_high < 0, f"(p_add={a_high:.1f})")
ems2 = EMS()
a_mid = ems2.p_add(0.55)
check("目标死区 P_add=0", abs(a_mid) < 1e-9, f"(p_add={a_mid:.2f})")
ems3 = EMS()
a1 = ems3.p_add(0.35)                         # < h1 -> 强制充电 ON
check("SOC<h1 强制充电", ems3.charge_mode and a1 >= ems3.p_add_charge_min_kw - 1e-9)
a2 = ems3.p_add(0.50)                         # 仍在 (h1,h2) 保持 ON
check("滞环保持充电（h1-h2 内不切换）",
      ems3.charge_mode and a2 >= ems3.p_add_charge_min_kw - 1e-9)
a3 = ems3.p_add(0.62)                         # > h2 -> OFF
check("SOC>h2 停止强制充电", (not ems3.charge_mode) and a3 < 0, f"(a3={a3:.1f})")

print("== EMS: SOFC 低通滤波与启停滞环（Eq.17）==")
ems4 = EMS()
sp = None
for _ in range(300):                          # 5 h 稳态负荷 180 kW
    sp = ems4.compute_setpoints(60.0, 180.0, 0.55, P_SOFC, P_GENSET)
check("稳态 SOFC 设定 <= 装机", sp["sofc_set_kw"] <= P_SOFC + 1e-9,
      f"(={sp['sofc_set_kw']:.1f})")
check("稳态 SOFC 已启动", sp["sofc_on"], f"(lpf={sp['lpf_kw']:.1f})")
ems5 = EMS()
v0 = ems5.compute_setpoints(60.0, 180.0, 0.55, P_SOFC, P_GENSET)["lpf_kw"]
v1 = ems5.compute_setpoints(60.0, 180.0, 0.55, P_SOFC, P_GENSET)["lpf_kw"]
check("LPF 单步增量温和（<=10% 装机）", v1 - v0 <= 0.10 * P_SOFC + 1e-9,
      f"(dv={v1-v0:.1f})")
ems6 = EMS()
for _ in range(200):
    ems6.compute_setpoints(60.0, 100.0, 0.55, P_SOFC, P_GENSET)   # 建立 on
check("预热后 SOFC 已启动", ems6.sofc_on)
toggles = 0
prev = ems6.sofc_on
for k in range(200):                          # 5/15 kW 交替，滤波后 ~10
    spk = ems6.compute_setpoints(60.0, 5.0 if k % 2 == 0 else 15.0, 0.55, P_SOFC, P_GENSET)
    if spk["sofc_on"] != prev:
        toggles += 1
        prev = spk["sofc_on"]
check("负荷波动不触发频繁启停（切换<=2）", toggles <= 2, f"(toggles={toggles})")

print("== EMS: 层级分配（Eq.18/19）==")
ems7 = EMS()
for _ in range(600):                          # 10 h 稳态 200 kW
    sp7 = ems7.compute_setpoints(60.0, 200.0, 0.55, P_SOFC, P_GENSET)
check("SOFC 饱和在装机 150", abs(sp7["sofc_set_kw"] - P_SOFC) < 1e-6,
      f"(={sp7['sofc_set_kw']:.1f})")
check("机组补足剩余 ~50", abs(sp7["genset_set_kw"] - 50.0) < 1.0,
      f"(={sp7['genset_set_kw']:.1f})")
check("电池兜底 = 真实负荷 - 实际出力",
      abs(EMS.battery_setpoint(200.0, 150.0, 50.0)) < 1e-9)
check("电池兜底 = P_load - gen（SOC 修正进入电池）",
      abs(EMS.battery_setpoint(200.0, 160.0, 30.0) - 10.0) < 1e-9)

print("== Microgrid: 7 天整定仿真 ==")
mg = Microgrid()
rng = np.random.default_rng(7)


def load_fn(t_h):
    base = 85.0 + 65.0 * np.sin(2 * np.pi * (t_h - 9.0) / 24.0)
    peak = 45.0 if 11.0 <= (t_h % 24.0) < 13.0 else 0.0
    return float(max(0.0, base + peak + 8.0 * rng.normal()))


def heat_fn(t_h):
    h = 35.0 + 20.0 * np.sin(2 * np.pi * (t_h - 7.0) / 24.0) + 10.0 * np.sin(2 * np.pi * (t_h - 19.0) / 24.0)
    return float(max(5.0, h + 3.0 * rng.normal()))


mg.run_days(7.0, load_fn, heat_fn, dt_s=60.0)
df = mg.to_dataframe()
s = mg.summary()
dt_h = float(df["dt_s"].iloc[0]) / 3600.0
check("功率平衡：累计不平衡 < 1 kWh", abs(s["imbalance_kwh"]) < 1.0,
      f"(={s['imbalance_kwh']:.3f} kWh)")
check("功率平衡：最大瞬时不平衡 < 5 kW", s["max_imbalance_kw"] < 5.0,
      f"(={s['max_imbalance_kw']:.2f} kW)")
check("SOC 在 [0.2, 0.9]", 0.20 <= s["soc_min"] and s["soc_max"] <= 0.90,
      f"([{s['soc_min']:.2f}, {s['soc_max']:.2f}])")
ramp_i = float(df["sofc_i_pu"].diff().abs().max())
check("SOFC 电流爬坡 <= 2%/min（论文 Table 3）",
      ramp_i <= 0.02 * (dt_h * 60.0) + 1e-9, f"(dI_pu={ramp_i:.4f}/min)")
check("机组参与发电（负荷超过 SOFC 装机时段）", s["energy_genset_kwh"] > 0,
      f"(={s['energy_genset_kwh']:.0f} kWh)")
check("SOFC 为主供电源", s["energy_sofc_kwh"] > s["energy_genset_kwh"],
      f"(sofc={s['energy_sofc_kwh']:.0f} vs gen={s['energy_genset_kwh']:.0f} kWh)")
check("锅炉参与补热", bool((df["q_boiler_kw"] > 0).any()))
check("余热回收非负", bool((df["q_rec_kw"] >= -1e-9).all()))
check("排放记账 > 0", s["ghg_kg_co2eq"] > 0, f"(ghg={s['ghg_kg_co2eq']:.1f} kg)")
check("SOFC 降解单调（SOH 非增）", bool(df["sofc_soh"].is_monotonic_decreasing))
check("机组余热回收 > 0（运行时）", bool((df.loc[df["genset_kw"] > 1.0, "q_rec_kw"] > 0).all()))

print("== D3: PINN Tmax 约束降载（软惩罚进 EMS，紧限值 1135 K 演示）==")
from src.coupling import PinnCoupling
rng3 = np.random.default_rng(7)          # 独立种子：基线/降载用同一负荷序列


def load_fn3(t_h):
    base = 85.0 + 65.0 * np.sin(2 * np.pi * (t_h - 9.0) / 24.0)
    peak = 45.0 if 11.0 <= (t_h % 24.0) < 13.0 else 0.0
    return float(max(0.0, base + peak + 8.0 * rng3.normal()))


def heat_fn3(t_h):
    h = 35.0 + 20.0 * np.sin(2 * np.pi * (t_h - 7.0) / 24.0) + 10.0 * np.sin(2 * np.pi * (t_h - 19.0) / 24.0)
    return float(max(5.0, h + 3.0 * rng3.normal()))


mg_b = Microgrid()
mg_b.run_days(7.0, load_fn3, heat_fn3, dt_s=60.0)
s_b = mg_b.summary()
coup3 = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate",
                             "tmax_limit_k": 1135.0, "derate_deadband_k": 0.0,
                             "derate_floor_pu": 0.5})
mg_d = Microgrid(coupling=coup3)
mg_d.run_days(7.0, load_fn3, heat_fn3, dt_s=60.0)
df_d = mg_d.to_dataframe()
s_d = mg_d.summary()
cs3 = coup3.summary()
check("D3 触发降载（紧限值下存在降载事件）", cs3["pinn_derate_steps"] > 0,
      f"(steps={cs3['pinn_derate_steps']}, kwh={cs3['pinn_derate_kwh']:.0f})")
check("D3 降载后 Tmax 违反 0 步", cs3["pinn_tmax_violations"] == 0,
      f"(viol={cs3['pinn_tmax_violations']}, max={cs3['pinn_max_tmax_est_k']:.1f} K)")
check("D3 功率平衡保持（累计 <1 kWh）", abs(s_d["imbalance_kwh"]) < 1.0,
      f"(={s_d['imbalance_kwh']:.3f} kWh)")
check("D3 SOFC 出力低于基线（降载转移）", s_d["energy_sofc_kwh"] < s_b["energy_sofc_kwh"],
      f"(base={s_b['energy_sofc_kwh']:.0f} vs derate={s_d['energy_sofc_kwh']:.0f} kWh)")
check("D3 机组/电池补足缺口", (s_d["energy_genset_kwh"] + s_d["energy_bat_kwh"]) >
      (s_b["energy_genset_kwh"] + s_b["energy_bat_kwh"]),
      f"(base={s_b['energy_genset_kwh'] + s_b['energy_bat_kwh']:.0f} vs "
      f"derate={s_d['energy_genset_kwh'] + s_d['energy_bat_kwh']:.0f} kWh)")
check("D3 降载幅度有界（<= 装机 150 kW）", float(df_d["pinn_derate_kw"].max()) <= 150.0 + 1e-9,
      f"(max={float(df_d['pinn_derate_kw'].max()):.1f} kW)")
check("D3 未触发全降载（floor 未击穿）", cs3["pinn_derate_full_steps"] == 0,
      f"(full={cs3['pinn_derate_full_steps']})")
print(f"\n== 全部通过：{PASS} 项检查 OK ==")
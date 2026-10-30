# -*- coding: utf-8 -*-
"""C5 尺寸/重量模型自检（论文 Table 2 密度 + S_fuel 燃料舱 + C4 电量量级）。
运行：python scripts/check_sizing.py（工作目录 ROOT）
通过时逐项打印 OK 并输出标定汇总；任一断言失败退出码非 0。

校验项：SOFC 150 kW -> 10.0 m3 / 6.0 ton；电池 300 kWh（components.yaml）-> 3.333 m3
/ 3.75 ton；柴油机/锅炉按 dg/boiler_lng 密度；燃料舱 = 日燃料能量 × 天数 × (1+S_fuel)
/ 能量密度；S_fuel 裕度与 autonomy 线性；与 C4 年电量 846.3 MWh 的日燃料能量量级自洽；
总体积/总重量为正且 SOFC 占比合理；CSV 导出可运行。"""
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

from src.components.params import DENSITIES
from src.components.sizing import SizingCalculator

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


TOL = 1e-6

# ---------------------------------------------------------------------------
calc = SizingCalculator()
res = calc.compute()

print("== C5 组件体积/重量（Table 2 密度，手算对照，容差 1e-6）==")
sofc = res["components"]["sofc"]
check("SOFC 150 kW -> 10.0 m3 / 6.0 ton",
      abs(sofc["volume_m3"] - 10.0) < TOL and abs(sofc["weight_ton"] - 6.0) < TOL,
      f"(V={sofc['volume_m3']:.4f}, W={sofc['weight_ton']:.4f})")

bat = res["components"]["battery"]
check("电池 300 kWh（components.yaml 实际值）-> 3.3333 m3 / 3.75 ton",
      abs(bat["volume_m3"] - 300.0 / 90.0) < TOL and abs(bat["weight_ton"] - 3.75) < TOL,
      f"(V={bat['volume_m3']:.4f}, W={bat['weight_ton']:.4f}, E={bat['rating_kwh']:.0f} kWh)")

dg = res["components"]["genset"]
check("柴油机 100 kW（dg/MGO）-> 2.2222 m3 / 1.6667 ton",
      abs(dg["volume_m3"] - 100.0 / 45.0) < TOL and abs(dg["weight_ton"] - 100.0 / 60.0) < TOL,
      f"(V={dg['volume_m3']:.4f}, W={dg['weight_ton']:.4f})")

boiler = res["components"]["boiler"]
check("锅炉 120 kW（boiler_lng，双燃料取保守）-> 1.6 m3 / 0.3871 ton",
      abs(boiler["volume_m3"] - 120.0 / 75.0) < TOL
      and abs(boiler["weight_ton"] - 120.0 / 310.0) < TOL,
      f"(V={boiler['volume_m3']:.4f}, W={boiler['weight_ton']:.4f})")

KEY_DENSITY = {"sofc": "sofc", "genset": "dg", "gg": "gg",
               "boiler": "boiler_lng", "battery": "bat"}
ok_loop = True
for key, c in res["components"].items():
    d = DENSITIES[KEY_DENSITY[key]]
    if "rating_kw" in c:
        ok_loop = ok_loop and abs(c["volume_m3"] - c["rating_kw"] / d["p_vol_kW_m3"]) < TOL
        ok_loop = ok_loop and abs(c["weight_ton"] - c["rating_kw"] / d["p_grav_kW_ton"]) < TOL
    else:
        ok_loop = ok_loop and abs(c["volume_m3"] - c["rating_kwh"] / d["e_vol_kWh_m3"]) < TOL
        ok_loop = ok_loop and abs(c["weight_ton"] - c["rating_kwh"] / d["e_grav_kWh_ton"]) < TOL
check("每个组件体积/重量 = 额定功率(容量)/对应密度（循环 1e-6）", ok_loop)

calc_ov = SizingCalculator(overrides={"sofc_kw": 300.0})
res_ov = calc_ov.compute()
check("显式 dict 覆盖生效（sofc_kw=300 -> 20.0 m3）",
      abs(res_ov["components"]["sofc"]["volume_m3"] - 20.0) < TOL,
      f"(V={res_ov['components']['sofc']['volume_m3']:.3f})")

print("== 燃料舱：S_fuel 裕度 + autonomy 线性 + 能量密度自洽 ==")
lng = res["fuel_tanks"]["lng"]
mgo = res["fuel_tanks"]["mgo"]
d_lng = DENSITIES["fuel_lng"]
d_mgo = DENSITIES["fuel_mgo"]
vol_lng_exp = (lng["daily_fuel_energy_kwh"] * res["autonomy_days"]
               * (1.0 + res["safety_margin_fuel"]) / d_lng["e_vol_kWh_m3"])
check("LNG 舱容积 = 日燃料能量 × 7 × 1.1 / 3443（精确对照）",
      abs(lng["volume_m3"] - vol_lng_exp) < TOL, f"(V={lng['volume_m3']:.3f} m3)")
check("LNG 舱能量密度自洽（E/V ≈ 3443 kWh/m3）",
      abs(lng["tank_energy_kwh"] / lng["volume_m3"] - d_lng["e_vol_kWh_m3"]) < TOL,
      f"(={lng['tank_energy_kwh'] / lng['volume_m3']:.1f} kWh/m3)")
vol_mgo_exp = (mgo["daily_fuel_energy_kwh"] * res["autonomy_days"]
               * (1.0 + res["safety_margin_fuel"]) / d_mgo["e_vol_kWh_m3"])
check("MGO 舱容积 = 日燃料能量 × 7 × 1.1 / 8200（精确对照）",
      abs(mgo["volume_m3"] - vol_mgo_exp) < TOL, f"(V={mgo['volume_m3']:.3f} m3)")

calc0 = SizingCalculator(overrides={"safety_margin_fuel": 0.0})
res0 = calc0.compute()
ratio_margin = res["fuel_tanks"]["lng"]["volume_m3"] / res0["fuel_tanks"]["lng"]["volume_m3"]
check("S_fuel=0.10 -> 燃料舱容积 ×1.10（裕度正确放大）",
      abs(ratio_margin - 1.10) < 1e-9, f"(ratio={ratio_margin:.6f})")

calc14 = SizingCalculator(overrides={"autonomy_days": 14.0})
res14 = calc14.compute()
ratio_auto = res14["fuel_tanks"]["lng"]["volume_m3"] / res["fuel_tanks"]["lng"]["volume_m3"]
check("autonomy 线性（14 天 = 2 × 7 天容积）",
      abs(ratio_auto - 2.0) < 1e-9, f"(ratio={ratio_auto:.6f})")

print("== C4 年电量 846.3 MWh ↔ 日燃料能量量级自洽 ==")
daily_fuel_mwh = lng["daily_fuel_energy_kwh"] / 1000.0
check("日电量 = 846.3/365 精确 且 日燃料能量（eta=0.54）落在 3.5-6.5 MWh 量级",
      abs(res["daily_electricity_kwh"] - 846.3e3 / 365.0) < TOL
      and 3.5 <= daily_fuel_mwh <= 6.5,
      f"(日电 {res['daily_electricity_kwh'] / 1000:.4f} MWh，日燃料 {daily_fuel_mwh:.3f} MWh)")
check("回代：日燃料能量 × 365 × eta_sofc = 846.3 MWh",
      abs(lng["daily_fuel_energy_kwh"] * 365.0 * res["eta_sofc_nominal"] - 846.3e3) < 1e-3,
      f"(={lng['daily_fuel_energy_kwh'] * 365.0 * res['eta_sofc_nominal']:.1f} kWh)")

print("== 总量 / 甲板面积 / CSV 导出 ==")
tv, tw = res["total_volume_m3"], res["total_weight_ton"]
share = sofc["volume_m3"] / tv
check("总体积/总重量为正 且 SOFC 机组体积占比合理（20-60%）",
      tv > 0.0 and tw > 0.0 and 0.20 <= share <= 0.60,
      f"(V={tv:.3f} m3, W={tw:.3f} ton, share={share:.3f})")
check("甲板面积 = 总体积 / deck_height_m=3.0（精确对照）",
      abs(res["deck_area_m2"] - tv / 3.0) < TOL and res["deck_area_m2"] > 0.0,
      f"(A={res['deck_area_m2']:.2f} m2)")
fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="check_sizing_")
os.close(fd)
try:
    out = calc.to_csv(tmp)
    with io.open(tmp, "r", encoding="utf-8") as f:
        content = f.read()
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
n_rows = len(calc.to_table())
check("CSV 导出可运行（写读回：表头 + 每组件/燃料舱/合计一行）",
      out == tmp and content.startswith("component")
      and len(content.splitlines()) == n_rows + 1,
      f"(rows={n_rows} + header)")

# ---------------------------------------------------------------------------
print("\n== 标定汇总 ==")
c = res["components"]
print(f"  SOFC 150 kW          : {c['sofc']['volume_m3']:.3f} m3 / {c['sofc']['weight_ton']:.3f} ton")
print(f"  柴油机 100 kW        : {c['genset']['volume_m3']:.3f} m3 / {c['genset']['weight_ton']:.3f} ton")
print(f"  锅炉 120 kW          : {c['boiler']['volume_m3']:.3f} m3 / {c['boiler']['weight_ton']:.3f} ton")
print(f"  电池 {bat['rating_kwh']:.0f} kWh         : {bat['volume_m3']:.3f} m3 / {bat['weight_ton']:.3f} ton")
print(f"  LNG 燃料舱 ({res['autonomy_days']:.0f} d)   : {lng['volume_m3']:.3f} m3 / {lng['weight_ton']:.3f} ton")
print(f"  MGO 燃料舱 ({res['autonomy_days']:.0f} d)   : {mgo['volume_m3']:.3f} m3 / {mgo['weight_ton']:.3f} ton")
print(f"  总排布体积          : {res['total_volume_m3']:.2f} m3（设备 {res['equipment_volume_m3']:.2f} + 燃料 {res['fuel_volume_m3']:.2f}）")
print(f"  总重量              : {res['total_weight_ton']:.2f} ton")
print(f"  甲板面积 (H={res['deck_height_m']:.1f} m) : {res['deck_area_m2']:.2f} m2")
print(f"  日燃料能量          : LNG {lng['daily_fuel_energy_kwh'] / 1000:.3f} MWh/d，MGO {mgo['daily_fuel_energy_kwh'] / 1000:.3f} MWh/d")
print(f"\n== 全部通过：{PASS} 项检查 OK ==")
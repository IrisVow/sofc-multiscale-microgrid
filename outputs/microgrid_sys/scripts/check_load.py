# -*- coding: utf-8 -*-
"""C4 负荷/可再生模块自检（阶段 C4）。
运行：python scripts/check_load.py（工作目录 ROOT）
通过时逐项打印 OK 并输出标定汇总；任一断言失败抛出 AssertionError（退出码非 0）。

校验项：年峰值 200 kW（±0.5%）、年电量 720-880 MWh（365 d @ 60 s）、
淡化恒 40 kW、曲线非负、热负荷峰值/均值区间、RE 关闭全零、60 s 相邻变化有界。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.analysis.load_profile import LoadProfile, SEASONS
from src.analysis.renewables import PVProfile, WindProfile

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


# ---------------------------------------------------------------------------
print("== C4 负荷剖面标定（365 d @ 60 s 全年合成序列）==")
lp = LoadProfile()
df = lp.generate_series(365, dt_s=60.0, start="2026-01-01")
p_el = df["p_el_kw"].to_numpy()
q = df["q_load_kw"].to_numpy()
res = df["p_res_kw"].to_numpy()
tou = df["p_tou_kw"].to_numpy()
desal = df["p_desal_kw"].to_numpy()

peak = float(p_el.max())
i_peak = int(p_el.argmax())
t_peak = df.index[i_peak]
check("年峰值负荷 = 200 kW（±0.5%）", abs(peak - 200.0) <= 1.0,
      f"(peak={peak:.3f} kW)")
check("峰值时刻 = 夏季工作日 19:00", t_peak.month in (6, 7, 8)
      and t_peak.hour == 19 and t_peak.weekday() < 5, f"(t={t_peak})")

e_mwh = float(p_el.sum() * 60.0 / 3600.0 / 1000.0)
cal = lp.calibrate()
check("年电量 720-880 MWh（365 d @ 60 s）", 720.0 <= e_mwh <= 880.0,
      f"(E={e_mwh:.1f} MWh, scale_factor={cal['scale_factor']:.4f})")

check("淡化负荷恒 40 kW", float(np.max(np.abs(desal - 40.0))) < 1e-9,
      f"(max|Δ|={np.max(np.abs(desal - 40.0)):.2e})")
check("电/热/分项曲线非负",
      min(p_el.min(), q.min(), res.min(), tou.min(), desal.min()) >= 0.0,
      f"(min={min(p_el.min(), q.min()):.3f} kW)")
check("热负荷峰值 60-80 kW", 60.0 <= q.max() <= 80.0, f"(max={q.max():.2f} kW)")
check("热负荷均值 35-50 kW", 35.0 <= q.mean() <= 50.0, f"(mean={q.mean():.2f} kW)")

dmax = float(np.max(np.abs(np.diff(p_el))))
check("60 s 相邻步变化有界（<=5 kW，无尖刺）", dmax <= 5.0,
      f"(max|ΔP|={dmax:.3f} kW/步)")

print("== 季节 x 工作日/周末 24 h 基准曲线 ==")
for s in SEASONS:
    for w in (False, True):
        d24 = lp.profile_24h(s, weekend=w, dt_s=3600.0)
        p24 = float(d24["p_el_kw"].max())
        q24 = float(d24["q_load_kw"].max())
        check(f"24h 基准曲线 {s}/{'周末' if w else '工作日'} 峰值<=200 且热峰<=80",
              p24 <= 200.0 + 1e-9 and q24 <= 80.0 + 1e-9,
              f"(P={p24:.2f}, Q={q24:.2f})")
check("夏季工作日 24h 曲线峰值 = 200 kW",
      abs(float(lp.profile_24h("summer", weekend=False)["p_el_kw"].max()) - 200.0) <= 1.0)

print("== 可再生能源（C1 关闭 -> 全零占位）==")
pv = PVProfile()
wd = WindProfile()
pvs = pv.generate_series(7, dt_s=60.0, time_index=False)
wds = wd.generate_series(7, dt_s=60.0, time_index=False)
check("include_pv=false（scenario 读取）", pv.include is False)
check("include_wind=false（scenario 读取）", wd.include is False)
check("PV 关闭 -> 全零序列", bool(np.all(pvs == 0.0)) and pvs.shape[0] == 7 * 1440,
      f"(n={pvs.shape[0]})")
check("Wind 关闭 -> 全零序列", bool(np.all(wds == 0.0)) and wds.shape[0] == 7 * 1440,
      f"(n={wds.shape[0]})")

print("== 淡化可平移接口（EMS C3/D 期）==")
meta = lp.metadata()
check("desalination_schedulable=true（元数据）", meta["desalination_schedulable"] is True)
check("可平移元数据含 kw/horizon 字段",
      meta["desalination_shift"]["kw"] == 40.0
      and meta["desalination_shift"]["shift_horizon_h"] == 24.0)
df7 = lp.generate_series(7, dt_s=60.0, start="2026-06-01")
sched = np.zeros(len(df7), dtype=float)          # 示意：淡化全部移走（峰时削峰）
df7b = lp.apply_desalination_schedule(df7, sched)
check("平移接口：替换淡化后 p_el 重算一致",
      bool(np.allclose(df7b["p_el_kw"],
                       df7b["p_res_kw"] + df7b["p_tou_kw"] + sched)))
check("平移接口：削峰效果（峰时 p_el 下降 40 kW）",
      float(df7["p_el_kw"].max()) - float(df7b["p_el_kw"].max()) == 40.0,
      f"({df7['p_el_kw'].max():.1f} -> {df7b['p_el_kw'].max():.1f} kW)")

print("== time_index=False 返回 (p_el, q_load) 数组 ==")
p_a, q_a = lp.generate_series(3, dt_s=60.0, time_index=False)
check("数组模式返回 (p_el_kw, q_load_kw)", p_a.shape == q_a.shape == (3 * 1440,))

# ---------------------------------------------------------------------------
print(f"\n== 标定汇总 ==")
print(f"  年峰值电负荷 : {peak:.2f} kW（@{t_peak}）")
print(f"  年电量       : {e_mwh:.1f} MWh（目标区间 720-880；负荷系数 {e_mwh/8760/0.2:.3f}）")
print(f"  淡化年电量   : {40.0*8760/1000:.1f} MWh（恒 40 kW）")
print(f"  热负荷       : 峰值 {q.max():.1f} kW / 均值 {q.mean():.1f} kW")
print(f"  60 s 相邻最大变化 : {dmax:.3f} kW/步")
print(f"\n== 全部通过：{PASS} 项检查 OK ==")
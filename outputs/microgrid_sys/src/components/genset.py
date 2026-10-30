# -*- coding: utf-8 -*-
"""柴油/燃气发电机组 0D 模型（阶段 C2，论文 2.3.2）。

- 负荷-电效率查找表（含 η_gen=0.95，Table 3）
- 排放查找表（NOx/SOx/PM/CH4 滑移，随负荷；MGO 与 LNG 两套）
- 速率限制（Eq.8 结构：MGO 2.7%P/s、LNG 1.4%P/s，Table 3）
- 最低负荷 10%（Table 3）；LNG 双燃料机组 3% 引燃油（Table 3）

调用：rep = genset.step(dt_s, p_set_kw, fuel='LNG')
rep: {p_kw, p_set_kw, eta, fuel_kwh, pilot_kwh, ch4_g, nox_g, sox_g, pm_g, curtail_kw}
"""
from __future__ import annotations

import numpy as np

from .params import PARAMS


def _interp(tbl, x):
    xs = np.array([r[0] for r in tbl], dtype=float)
    ys = np.array([r[1] for r in tbl], dtype=float)
    return float(np.interp(x, xs, ys))


class Genset:
    def __init__(self, params: dict = None, p_nom_kw: float = 100.0, fuel: str = "LNG"):
        p = dict(PARAMS["genset"])
        if params:
            p.update(params)
        self.p_nom = float(p_nom_kw)
        self.min_load = float(p["min_load"])
        self.pilot_frac = float(p["pilot_fuel_frac"])
        self.ch4_slip = float(p["ch4_slip"])
        self.ramp = {"MGO": float(p["ramp_mgo_pct_per_s"]) / 100.0,   # pu/s
                     "LNG": float(p["ramp_lng_pct_per_s"]) / 100.0}
        self.fuel = fuel
        self.p_prev = 0.0

    def set_fuel(self, fuel: str):
        self.fuel = fuel

    def reset(self):
        self.p_prev = 0.0
        return self

    def _table(self, key: str) -> list:
        return list(PARAMS["genset"].get(f"{key}_{self.fuel.lower()}", []))

    def eta(self, load_pu: float) -> float:
        return _interp(self._table("eta_eff"), float(load_pu))

    def step(self, dt_s: float, p_set_kw: float) -> dict:
        """推进 dt 秒；返回本步燃料/排放/实际功率。"""
        p_req = max(0.0, float(p_set_kw))
        p_set = p_req
        if 0.0 < p_req / self.p_nom < self.min_load:
            # 最低负荷 10%（Table 3）：低于 LL 的指令归零（由 EMS 决策抬载或停机）
            p_set = 0.0
        curt = p_req - p_set
        # 速率限制（pu/s）
        ramp = self.ramp[self.fuel]
        p_new_pu = float(np.clip(p_set / self.p_nom,
                                 self.p_prev - ramp * dt_s,
                                 self.p_prev + ramp * dt_s))
        self.p_prev = p_new_pu
        load_pu = p_new_pu
        p = p_new_pu * self.p_nom
        eta = self.eta(load_pu)
        fuel_kwh = p / max(eta, 1e-6)                     # 燃料 LHV 能量（kWh）
        pilot_kwh = fuel_kwh * self.pilot_frac if self.fuel == "LNG" else 0.0
        # 排放（g）：排放因子随负荷
        rep = {
            "p_kw": p, "p_set_kw": p_set, "load_pu": load_pu,
            "eta": eta, "fuel_kwh": fuel_kwh, "pilot_kwh": pilot_kwh,
            "ch4_slip_frac": self.ch4_slip,               # CH4 质量滑移份额（emissions 按燃料质量折算）
            "nox_g": p * _interp(self._table("nox_g_kwh"), load_pu),
            "sox_g": p * _interp(self._table("sox_g_kwh"), load_pu),
            "pm_g": p * _interp(self._table("pm_g_kwh"), load_pu),
            "q_heat_kw": fuel_kwh * _interp(self._table("eta_heat"), load_pu),
            "curtail_kw": p_req - p,
            "fuel": self.fuel,
            "dt_s": dt_s,
        }
        return rep
# -*- coding: utf-8 -*-
"""锅炉与热平衡 0D 模型（阶段 C2）。

热链：SOFC/发电机组余热回收 -> 锅炉补足热需求（淡化/生活热水）-> 余热排放。
锅炉效率：MGO 85% / LNG 90%（Table 3）；换热器效率 85%（项目文档 Eq.5）。

调用：rep = boiler.step(q_demand_kw, q_recovered_kw, fuel='LNG')
rep: {q_deficit_kw, q_boiler_kw, fuel_kwh, q_dumped_kw, q_supplied_kw}
"""
from __future__ import annotations

from .params import PARAMS


class Boiler:
    def __init__(self, params: dict = None):
        p = dict(PARAMS["boiler"])
        if params:
            p.update(params)
        self.eta = {"MGO": float(p["eta_mgo"]), "LNG": float(p["eta_lng"])}
        self.hex_eta = float(p["hex_eta"])

    def step(self, q_demand_kw: float, q_recovered_kw: float, fuel: str = "LNG",
              dt_s: float = 0.0) -> dict:
        q_dem = max(0.0, float(q_demand_kw))
        q_rec = max(0.0, float(q_recovered_kw))
        q_supplied = min(q_dem, q_rec * self.hex_eta)     # 回收热先满足（经换热器）
        q_deficit = max(0.0, q_dem - q_supplied)
        q_boiler = q_deficit / self.eta[fuel]
        q_dumped = max(0.0, q_rec * self.hex_eta - q_supplied)
        return {
            "q_demand_kw": q_dem, "q_recovered_kw": q_rec,
            "q_supplied_kw": q_supplied, "q_deficit_kw": q_deficit,
            "q_boiler_kw": q_boiler, "fuel_kwh": q_boiler,   # 锅炉燃料能量（LHV）
            "q_dumped_kw": q_dumped, "fuel": fuel, "dt_s": float(dt_s),
        }
# -*- coding: utf-8 -*-
"""电池 0D 模型（阶段 C2，论文 Table 3 参数）。

- SOC 状态：初值 0.55，边界 [0.2, 0.9]（含 10% 安全裕度）
- 充放效率 90%/95%；最大 C-rate 3
- 功率限幅：C-rate 限 + SOC 边界限（充不满/放不尽的余量 -> curtail）

调用：rep = bat.step(dt_s, p_req_kw)   # p_req>0 放电（供网），<0 充电（吸收）
rep: {p_kw, soc, e_kwh, curt_kw, eta_eff}
"""
from __future__ import annotations

from .params import PARAMS


class Battery:
    def __init__(self, params: dict = None):
        p = dict(PARAMS["battery"])
        if params:
            p.update(params)
        self.e_nom = float(p["e_nom_kwh"])
        self.eta_ch = float(p["eta_ch"])
        self.eta_dis = float(p["eta_dis"])
        self.c_rate = float(p["c_rate_max"])
        self.soc_ub = float(p["soc_ub"])
        self.soc_lb = float(p["soc_lb"])
        self.soc_init = float(p["soc_init"])
        self.reset()

    def reset(self):
        self.soc = self.soc_init
        self.e_kwh = self.soc * self.e_nom
        return self

    def limits(self, dt_s: float) -> dict:
        """当前可放/可充功率（kW）。放电为正。"""
        dt_h = max(dt_s, 1e-9) / 3600.0
        p_c_rate = self.e_nom * self.c_rate
        # 放电上限：SOC 下界限制（可放出 e=(soc-lb)*E*eta_dis 的能量）
        p_soc_dis = (self.soc - self.soc_lb) * self.e_nom * self.eta_dis / dt_h
        # 充电上限：SOC 上界限制（可存入 e=(ub-soc)*E/eta_ch 的能量）
        p_soc_ch = (self.soc_ub - self.soc) * self.e_nom / (self.eta_ch * dt_h)
        return {"dis_max": min(p_c_rate, max(0.0, p_soc_dis)),
                "ch_max": min(p_c_rate, max(0.0, p_soc_ch))}

    def step(self, dt_s: float, p_req_kw: float) -> dict:
        """推进 dt 秒。p_req>0 放电、<0 充电；超限部分记 curtail。"""
        dt_h = max(dt_s, 1e-9) / 3600.0
        lim = self.limits(dt_s)
        p_req = float(p_req_kw)
        p_clip = float(min(max(p_req, -lim["ch_max"]), lim["dis_max"]))
        curt = p_req - p_clip
        if p_clip >= 0:                                   # 放电
            self.e_kwh -= p_clip * dt_h / self.eta_dis
        else:                                             # 充电
            self.e_kwh += (-p_clip) * dt_h * self.eta_ch
        self.e_kwh = float(max(self.soc_lb * self.e_nom,
                               min(self.e_kwh, self.soc_ub * self.e_nom)))
        self.soc = self.e_kwh / self.e_nom
        return {
            "p_kw": p_clip, "p_req_kw": p_req, "soc": self.soc,
            "e_kwh": self.e_kwh, "curt_kw": curt, "dt_s": dt_s,
            "lim": lim,
        }
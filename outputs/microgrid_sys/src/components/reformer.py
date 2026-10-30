# -*- coding: utf-8 -*-
"""LNG 重整器子模块（阶段 C2）。

将 LNG（视为 CH4）重整为富氢合成气供 SOFC 阳极：
    CH4 + 2 H2O -> CO2 + 4 H2   （蒸汽重整 + 完全水气变换，S/C>=2）
计入：重整效率（LHV 基准）、蒸汽需求（S/C）、CH4 滑移、重整热需求
（由 SOFC 余热/燃烧器承担，见 sofc_system.heat_balance）。

接口（供 EMS/耦合层）：
    reformer.run(p_fuel_ch4_kw) -> dict:
        n_h2_kmol_s, n_h2o_kmol_s, p_h2_kw, ch4_slip_kw, q_reform_kw, m_h2_kg_s
"""
from __future__ import annotations

from .params import CHEM, PARAMS


class Reformer:
    """蒸汽重整器 0D 模型（稳态代数；动态通过上游 SOFC 速率限制体现）。"""

    def __init__(self, params: dict = None):
        p = dict(PARAMS["sofc"])
        if params:
            p.update(params)
        self.eta_reform = float(p["eta_reform"])
        self.sc = float(p["sc_ratio"])
        self.ch4_slip = float(p["ch4_slip"])
        # 重整反应吸热 ~206 kJ/mol CH4（基准），加蒸汽汽化潜热计入 q_reform
        self.q_reform_kJ_per_kmol = 206.0 + 2.0 * 40.7  # ~287 kJ/mol（保守含蒸汽）

    def run(self, p_fuel_ch4_kw: float):
        """输入 CH4 燃料功率（LHV, kW）-> 合成气输出与热/水需求。"""
        p_ch4 = max(0.0, float(p_fuel_ch4_kw))
        lhv_ch4_kj_kmol = CHEM["LHV_CH4_kJ_kg"] * CHEM["M_CH4_kg_kmol"]   # kJ/kmol
        n_ch4 = p_ch4 * 1000.0 / lhv_ch4_kj_kmol                          # kmol/s
        n_h2 = 4.0 * n_ch4 * self.eta_reform                              # kmol/s（完全 WGS）
        n_h2o = self.sc * n_ch4                                           # kmol/s 蒸汽需求
        # 滑移：未重整 CH4 直接进燃烧器（热利用，不产 H2）
        slip = n_ch4 * self.ch4_slip
        p_h2_kw = n_h2 * CHEM["LHV_H2_kJ_kg"] * CHEM["M_H2_kg_kmol"] / 1000.0
        q_reform_kw = n_ch4 * (1.0 - self.ch4_slip) * self.q_reform_kJ_per_kmol / 1000.0
        return {
            "n_h2_kmol_s": n_h2,
            "n_h2o_kmol_s": n_h2o,
            "p_h2_kw": p_h2_kw,
            "p_ch4_kw": p_ch4,
            "slip_kw": slip * lhv_ch4_kj_kmol / 1000.0,
            "q_reform_kw": q_reform_kw,
            "m_h2_kg_s": n_h2 * CHEM["M_H2_kg_kmol"],
            "m_h2o_kg_s": n_h2o * CHEM["M_H2O_kg_kmol"],
            "eta_reform": self.eta_reform,
        }

    def h2_flow_for_power(self, p_h2_kw: float) -> float:
        """给定阳极 H2 功率需求（kW LHV），返回所需 CH4 燃料功率（kW）。"""
        lhv_h2_kmol = CHEM["LHV_H2_kJ_kg"] * CHEM["M_H2_kg_kmol"]
        lhv_ch4_kmol = CHEM["LHV_CH4_kJ_kg"] * CHEM["M_CH4_kg_kmol"]
        k = 4.0 * self.eta_reform * lhv_h2_kmol / lhv_ch4_kmol
        return p_h2_kw / k
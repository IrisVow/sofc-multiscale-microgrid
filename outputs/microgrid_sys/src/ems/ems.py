# -*- coding: utf-8 -*-
"""EMS 能量管理策略（阶段 C3；论文 Eq.17-20 + 双滞环）。

控制层级（论文 2.3.5，Fig.5a）：
1. SOFC 优先：对总需求（P_load + P_add）施加一阶低通滤波（Eq.17，时间常数
   由 SOFC/机组瞬态能力决定）作为 SOFC 设定值，限幅 [0, P_sofc_nom]，
   并带启停双滞环（防频繁启停）。
2. 发电机组次之（Eq.18）：P_genset,set = P_load + P_add - P_sofc,set，
   仅承担 SOFC 滤波后无法覆盖的部分；低于最低负荷则停机（电池兜底）。
3. 电池兜底（Eq.19）：P_bat,set = P_load - P_sofc,act - P_genset,act（真实负荷与实际出力之差）；
   P_add 只抬高发电设定（Eq.18），发电增量即电池充电量。
4. SOC 控制（Eq.20）：P_add = k_soc*(SOC_target - SOC)，比例增益 + 限幅 +
   死区；低 SOC 强制充电双滞环（SOC<h1 开始充、SOC>h2 停止）。

参数默认值见 DEFAULTS，可被 configs/ems.yaml 或构造 dict 覆盖。

接口（可插拔，阶段 D 可替换为 MPC/规则表）：
    ems = EMS(params)
    sp  = ems.compute_setpoints(dt_s, p_load_kw, soc, sofc_p_nom_kw, genset_p_nom_kw)
    bat = EMS.battery_setpoint(p_load_kw, p_sofc_act_kw, p_genset_act_kw)
"""
from __future__ import annotations

import io
import os

import numpy as np

DEFAULTS = {
    "soc_target": 0.55,          # 论文：SOC 目标/初值 0.55
    "k_soc_kw": 400.0,           # Eq.20 比例增益（kW / SOC 偏差）
    "p_add_max_kw": 100.0,       # P_add 限幅（峰值负荷 ~50%）
    "soc_deadband": 0.02,        # 目标死区（|dev|<db 时 P_add=0）
    "soc_h1": 0.40,              # 低 SOC 强制充电起点（双滞环下沿）
    "soc_h2": 0.60,              # 停止强制充电点（双滞环上沿）
    "p_add_charge_min_kw": 5.0,  # 强制充电期间 P_add 下限
    "lpf_tau_min": 20.0,         # SOFC 低通滤波时间常数（Eq.17）
    "sofc_on_thr_pu": 0.10,      # SOFC 启动阈值（相对 P_nom）
    "sofc_off_thr_pu": 0.05,     # SOFC 停机阈值
    "genset_min_load_pu": 0.10,  # 机组最低负荷（Table 3）
}


def load_ems_params(path: str = None) -> dict:
    """从 ems.yaml 读取覆盖项；path=None 仅返回 DEFAULTS。"""
    p = dict(DEFAULTS)
    if path:
        import yaml
        with io.open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ems = cfg.get("ems", cfg)
        for k in DEFAULTS:
            if k in ems:
                p[k] = ems[k]
    return p


class EMS:
    """SOC 滞环 EMS：SOFC -> 机组 -> 电池 层级 + SOC 比例修正。"""

    def __init__(self, params: dict = None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.soc_target = float(p["soc_target"])
        self.k_soc_kw = float(p["k_soc_kw"])
        self.p_add_max_kw = float(p["p_add_max_kw"])
        self.soc_deadband = float(p["soc_deadband"])
        self.soc_h1 = float(p["soc_h1"])
        self.soc_h2 = float(p["soc_h2"])
        self.p_add_charge_min_kw = float(p["p_add_charge_min_kw"])
        self.lpf_tau_min = float(p["lpf_tau_min"])
        self.sofc_on_thr_pu = float(p["sofc_on_thr_pu"])
        self.sofc_off_thr_pu = float(p["sofc_off_thr_pu"])
        self.genset_min_load_pu = float(p["genset_min_load_pu"])
        self.reset()

    def reset(self):
        self.lpf_state_kw = 0.0
        self.sofc_on = False
        self.charge_mode = False
        return self

    # ------------------------------------------------------------------
    # SOC 控制（Eq.20 + 双滞环）
    # ------------------------------------------------------------------
    def p_add(self, soc: float) -> float:
        """由 SOC 偏差计算人工负荷修正 P_add（kW）。"""
        soc = float(soc)
        if soc < self.soc_h1:
            self.charge_mode = True
        elif soc > self.soc_h2:
            self.charge_mode = False
        dev = self.soc_target - soc
        raw = float(np.clip(self.k_soc_kw * dev, -self.p_add_max_kw, self.p_add_max_kw))
        if abs(dev) < self.soc_deadband:
            raw = 0.0
        if self.charge_mode:
            raw = max(raw, self.p_add_charge_min_kw)
        return raw

    # ------------------------------------------------------------------
    # 低通滤波（Eq.17）
    # ------------------------------------------------------------------
    def lpf(self, x_kw: float, dt_s: float) -> float:
        """一阶低通：y += (x - y) * (1 - exp(-dt/tau))。"""
        tau_s = max(self.lpf_tau_min * 60.0, 1e-9)
        alpha = 1.0 - np.exp(-max(dt_s, 0.0) / tau_s)
        self.lpf_state_kw += alpha * (float(x_kw) - self.lpf_state_kw)
        return float(self.lpf_state_kw)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def compute_setpoints(self, dt_s: float, p_load_kw: float, soc: float,
                          sofc_p_nom_kw: float, genset_p_nom_kw: float) -> dict:
        """给定电网负荷与 SOC，返回 SOFC/机组设定值及中间量。"""
        p_add = self.p_add(soc)
        p_total = max(0.0, float(p_load_kw) + p_add)

        sofc_set_raw = float(np.clip(self.lpf(p_total, dt_s), 0.0, sofc_p_nom_kw))
        on_thr = self.sofc_on_thr_pu * sofc_p_nom_kw
        off_thr = self.sofc_off_thr_pu * sofc_p_nom_kw
        if not self.sofc_on and sofc_set_raw >= on_thr:
            self.sofc_on = True
        if self.sofc_on and sofc_set_raw < off_thr:
            self.sofc_on = False
        sofc_set = sofc_set_raw if self.sofc_on else 0.0

        # Eq.18：机组承担 SOFC 设定之外的高频/超装部分
        genset_set = max(0.0, p_total - sofc_set)
        genset_set = float(np.clip(genset_set, 0.0, genset_p_nom_kw))
        if genset_set < self.genset_min_load_pu * genset_p_nom_kw:
            genset_set = 0.0

        return {
            "sofc_set_kw": sofc_set,
            "sofc_operating": bool(self.sofc_on and sofc_set > 0.0),
            "genset_set_kw": genset_set,
            "p_add_kw": p_add,
            "p_total_kw": p_total,
            "lpf_kw": sofc_set_raw,
            "sofc_on": bool(self.sofc_on),
            "charge_mode": bool(self.charge_mode),
        }

    @staticmethod
    def battery_setpoint(p_load_kw: float,
                         p_sofc_act_kw: float, p_genset_act_kw: float) -> float:
        """Eq.19：电池设定 = 真实负荷 - 组件实际出力。
        P_add 只抬高 SOFC/机组发电设定（Eq.18），发电增量即电池充电量；
        电池吸收滤波/速率跟踪残差与 SOC 修正对应的充放电量。"""
        return float(p_load_kw) - float(p_sofc_act_kw) - float(p_genset_act_kw)
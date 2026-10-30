# -*- coding: utf-8 -*-
"""SOFC 系统 0D 模型（阶段 C2，论文 Eq.8/9/10/11/12 + Fig.3 效率结构）。

单位约定（恒功率名义点自洽）：
- 电流标幺 i_pu = I/I_nom，I_nom = 0.18 A/cm2（标定，论文范围 0.1-0.23）。
- 电阻以 V/pu 存储：R_bol = V0 - V_BOL = 1.05 - 0.85 = 0.20 V/pu；
  降解增量 R_deg 同单位（V/pu）。
- 电压为绝对电压（V）：V = V0 - i_pu * (R_bol + R_deg)。
- 功率标幺 p_pu = P/P_nom = i_pu * V / V_BOL；名义点：i_pu=1 -> V=0.85 V，p_pu=1。
  Eq.10 取小根（高电压支路）：
  i_pu = (V0 - sqrt(V0^2 - 4*(R_bol+R_deg)*V_BOL*p_pu)) / (2*(R_bol+R_deg))
- EOL 按论文恒功率定义：SOH 在名义功率点取值（V(P_nom)=V_EOL=0.71 V），
  R_deg_eol = (V0-V_EOL)*V_EOL/V_BOL - (V0-V_BOL) = 0.084 V/pu（电流随寿命上升）。

其余结构：
- 净电效率：BOL 查找表（Fig.3 结构，峰值在部分负荷）；EOL 曲线 = BOL * V/V_BOL
  （电压随降解下降 -> 效率同比例下降，与 Eq.12 SOH 定义一致）。
- 速率限制（Eq.8）：升载 R=2%/min、降载 F=8%/min（作用在电流标幺 i_pu 上）。
- 降解（Eq.11）：r_deg_rate = a_trans * 0.0573 * exp(1.25*i_A) [%eta_net/1000h]，
  瞬态（|di/dt| > 0.03%/min）时 a_trans=1.5；积分映射到 R_deg 增量
  （dR = (V/i_pu)*(r_rate/100)/1000*dt_h）。
- 热平衡：回收热 = P_fuel * eta_heat(P)（Fig.3，寿命内恒定）；重整热需求由
  reformer 报告，热链（燃烧器补热/锅炉）在 C3 耦合层实现。

调用约定：
    sofc = SOFCSystem(params)
    rep = sofc.step(dt_s, p_ac_set_kw, operating=True)
    rep: {p_ac, p_dc, p_fuel, p_h2_kw, q_heat_kw, q_reform_kw, v_cell, i_A_cm2,
          eta_net, soh, r_deg_pct_1000h, u_f}
"""
from __future__ import annotations

import numpy as np

from .params import PARAMS
from .reformer import Reformer


def _interp(table, x):
    """一维线性插值（table: [[x0,y0],...]；x 超界按端点值）。"""
    xs = np.array([r[0] for r in table], dtype=float)
    ys = np.array([r[1] for r in table], dtype=float)
    return float(np.interp(x, xs, ys))


class SOFCSystem:
    def __init__(self, params: dict = None):
        p = dict(PARAMS["sofc"])
        if params:
            p.update(params)
        self.p = p
        self.p_nom = float(p["p_nom_kw"])
        self.v0 = float(p["v0_ocv"])
        self.v_bol = float(p["v_cell_bol"])
        self.v_eol = float(p["v_cell_eol"])
        self.r_bol_pu = self.v0 - self.v_bol                 # BOL 电阻（V/pu）
        self.r_deg_eol = ((self.v0 - self.v_eol) * self.v_eol / self.v_bol
                          - (self.v0 - self.v_bol))          # 恒功率 EOL（V/pu）
        self.deg_scale = float(p.get("deg_scale", 1.0))      # E3 sensitivity: scale degradation rate
        self.ramp_rise = float(p["ramp_rise"])               # pu/min
        self.ramp_fall = float(p["ramp_fall"])               # pu/min
        self.t_transient = float(p["t_transient"])           # pu/min（0.03%）
        self.a_trans = float(p["a_trans"])
        self.i_nom = float(p["i_nom_A_cm2"])
        self.eta_inv = float(p["eta_inv"])
        self.eta_dcdc = float(p["eta_dcdc"])
        self.lifetime_h = float(p["lifetime_h"])
        self.eta_bol = list(p["eta_net_bol"])
        self.eta_heat_tbl = list(p["eta_heat"])
        self.u_f_tbl = list(p["u_f_load"])
        self.reformer = Reformer({"eta_reform": float(p["eta_reform"]),
                                  "sc_ratio": float(p["sc_ratio"]),
                                  "ch4_slip": float(p["ch4_slip"])})
        self.reset()

    def reset(self, soh: float = 1.0):
        self.i_pu = 0.0
        self.r_deg = (1.0 - soh) * self.r_deg_eol            # V/pu
        self.t_oper_h = 0.0
        self.p_ac = 0.0
        return self

    # ------------------------------------------------------------------
    # 代数关系（Eq.9/10）
    # ------------------------------------------------------------------
    def v_cell(self, i_pu: float, r_deg: float = None) -> float:
        """给定电流标幺（pu）求电池电压（V）。"""
        r = self.r_bol_pu + (self.r_deg if r_deg is None else r_deg)
        return self.v0 - i_pu * r

    def current_for_power(self, p_pu: float, r_deg: float = None) -> float:
        """Eq.10：给定功率（pu，DC）求电流标幺（pu）；取高电压支路小根。"""
        p_pu = max(0.0, float(p_pu))
        r = self.r_bol_pu + (self.r_deg if r_deg is None else r_deg)
        if r <= 0.0:
            return 0.0
        disc = self.v0 ** 2 - 4.0 * r * self.v_bol * p_pu
        if disc <= 0.0:
            # 超出当前退化状态下的最大可输出功率：回到最大功率点
            return self.v0 / (2.0 * r)
        return (self.v0 - np.sqrt(disc)) / (2.0 * r)

    def power_for_current(self, i_pu: float) -> float:
        """给定电流标幺求功率（pu，DC）：p_pu = i_pu * V / V_BOL。"""
        v = self.v_cell(i_pu)
        return max(0.0, i_pu * v / self.v_bol)

    def soh(self) -> float:
        """Eq.12：在名义功率点（恒功率）定义 SOH。
        随寿命电流上升以维持 P_nom，V(P_nom) 下降。"""
        i_nom = self.current_for_power(1.0)
        v_nom = self.v_cell(i_nom)
        return float(np.clip((v_nom - self.v_eol) / (self.v_bol - self.v_eol), 0.0, 1.0))

    def eta_net(self, p_pu: float, soh: float = None) -> float:
        """净电效率（DC）：BOL 曲线 * SOH 电压比。"""
        eta0 = _interp(self.eta_bol, float(p_pu))
        if soh is None:
            soh = self.soh()
        v_ratio = (self.v_eol + soh * (self.v_bol - self.v_eol)) / self.v_bol
        return eta0 * v_ratio

    def eta_heat(self, p_pu: float) -> float:
        return _interp(self.eta_heat_tbl, float(p_pu))

    def u_f(self, p_pu: float) -> float:
        return _interp(self.u_f_tbl, float(p_pu))

    def deg_rate(self, i_A_cm2: float, di_pu_dt: float = 0.0) -> float:
        """Eq.11：降解速率 [%eta_net/1000h]。"""
        a = self.a_trans if abs(di_pu_dt) > self.t_transient else 1.0
        return a * 0.0573 * np.exp(1.25 * i_A_cm2)

    # ------------------------------------------------------------------
    # 动态步进
    # ------------------------------------------------------------------
    def step(self, dt_s: float, p_ac_set_kw: float, operating: bool = True) -> dict:
        """推进 dt 秒。
        输入 EMS 的交流功率设定值（电网侧）；内部换算 DC 需求 -> 电流目标
        -> Eq.8 速率限制 -> Eq.10 电流/电压 -> 燃料/热/降解。"""
        dt_h = max(0.0, dt_s) / 3600.0
        if dt_s <= 0:
            return self.report()

        if not operating:
            target_p_dc = 0.0                                # 停机（受降载速率限制）
        else:
            p_ac = max(0.0, float(p_ac_set_kw))
            p_dc_set = p_ac / self.eta_inv                   # 交流 -> DC 净输出
            target_p_dc = min(p_dc_set, self.p_nom)          # 上限 P_nom
        target_i = self.current_for_power(target_p_dc / self.p_nom)

        # Eq.8 速率限制（pu/min）
        dt_min = dt_s / 60.0
        i_prev = self.i_pu
        di_max = self.ramp_rise * dt_min
        di_min = -self.ramp_fall * dt_min
        i_new = float(np.clip(target_i, i_prev + di_min, i_prev + di_max))
        di_pu_dt = (i_new - i_prev) / max(dt_min, 1e-9)      # pu/min

        p_dc_pu = self.power_for_current(i_new)
        p_dc = p_dc_pu * self.p_nom
        v_cell = self.v_cell(i_new)
        i_A = i_new * self.i_nom
        p_pu_eff = float(np.clip(p_dc_pu, 1e-3, 1.0))
        eta_net = self.eta_net(p_pu_eff)
        p_fuel = p_dc / max(eta_net, 1e-6)                   # LHV 燃料功率（CH4）
        q_heat = p_fuel * self.eta_heat(p_pu_eff)            # 回收热（燃烧器/重整余热）
        uf = self.u_f(p_pu_eff)
        ref = self.reformer.run(p_fuel)                      # 重整 H2 输出/热需求

        # 降解积分（Eq.11 -> dEta/Eta = -r_deg% per 1000h）；
        # dV/V = -r/100 per 1000h，dV = -i_pu*dR_deg -> dR_deg = (V/i_pu)*(r/100)/1000
        r_deg_rate = self.deg_rate(i_A, di_pu_dt) * self.deg_scale            # %/1000h
        dR = 0.0
        if i_new > 1e-3:
            dR = (v_cell / i_new) * (r_deg_rate / 100.0) / 1000.0 * dt_h
        self.r_deg = max(0.0, self.r_deg + dR)

        self.i_pu = i_new
        self.p_ac = p_dc * self.eta_inv
        if operating and i_new > 1e-3:
            self.t_oper_h += dt_h

        rep = self.report()
        rep.update({
            "p_fuel": p_fuel, "p_h2_kw": ref["p_h2_kw"],
            "q_heat": q_heat, "q_reform_kw": ref["q_reform_kw"],
            "u_f": uf, "i_A_cm2": i_A, "v_cell": v_cell,
            "eta_net": eta_net, "r_deg_pct_1000h": r_deg_rate,
            "dt_s": dt_s,
        })
        return rep

    def report(self) -> dict:
        soh = self.soh()
        return {
            "p_ac": self.p_ac,
            "p_dc": self.p_ac / self.eta_inv,
            "p_fuel": 0.0, "p_h2_kw": 0.0,
            "q_heat": 0.0, "q_reform_kw": 0.0, "u_f": 0.0,
            "i_pu": self.i_pu, "i_A_cm2": 0.0, "v_cell": self.v_cell(self.i_pu),
            "eta_net": 0.0, "soh": soh, "r_deg_pct_1000h": 0.0,
            "t_oper_h": self.t_oper_h, "dt_s": 0.0,
        }

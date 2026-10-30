"""Corrected differential equation core derived from preserved v20 implementation.
Public loss/sampling/boundary API is physics.pde.PDEResiduals.
Only inherited algebraic derivative/equation helpers are used; legacy training helpers below are overridden.
"""
from __future__ import annotations

import numpy as np
import torch

from . import params as P
from .electrochem import ionic_conductivity, reaction_current
from .domains import DOMAIN_ORDER, DOMAIN_ID, sample_collocation, sample_interfaces

# 输出变量列索引（与 config.yaml target_vars 顺序一致）
I = {"T": 0, "u": 1, "v": 2, "p": 3, "w1": 4, "w2": 5, "w3": 6,
     "phis": 7, "phil": 8, "htQ": 9, "qav_an": 10, "qav_ca": 11}

CONT_IDS = ["x", "y", "t_s", "V_applied", "T_in_K", "lambda_air", "FU"]
CHANNELS = {"air_channel", "fuel_channel"}
ELECTRODES = {"air_electrode", "fuel_electrode"}
ANODE = "fuel_electrode"
CATHODE = "air_electrode"
T_REF_BV = 1073.0          # BV 平衡电位参考温度 K

# ---------------- 参考尺度（把各方程残差缩放到 O(1)） ----------------
L_REF = 0.1                # m 流道长度
U_REF = 1.0                # m/s
RHO_REF = 0.165            # kg/m^3
DT_REF = 50.0              # K 温升参考
W_REF = 1.0
PHIS_REF = 1.0             # V
PHIL_REF = 0.2             # V
K_REF = 0.35               # W/(m*K)
SIGS_REF = 1.0e4           # S/m
SIGL_REF = 3.34e4          # S/m
J0_AN_VOL = P.AV_AN * P.I0_AN    # 5e7 A/m^3
J0_CA_VOL = P.AV_CA * P.I0_CA    # 5e6 A/m^3

REF = {
    "mass": RHO_REF * U_REF / L_REF,
    "mom": 2000.0,                                   # ~ dp 尺度 (200 Pa / 0.1 m)
    "w": RHO_REF * U_REF * W_REF / L_REF,
    "T": RHO_REF * P.CP_G * U_REF * DT_REF / L_REF,  # 对流主导 ~1.2e5
    "phis": SIGS_REF * PHIS_REF / L_REF ** 2,
    "phil": SIGL_REF * PHIL_REF / L_REF ** 2,
    "bv_an": J0_AN_VOL,
    "bv_ca": J0_CA_VOL,
    "flux_T": P.K_S * DT_REF / 2.62e-3,
    "flux_phil": SIGL_REF * PHIL_REF / L_REF,
    "flux_phis": SIGS_REF * PHIS_REF / L_REF,
    "mass_flux": RHO_REF * U_REF,
}


def k_eff(dom: str) -> float:
    """有效导热系数 W/(m*K)。"""
    if dom == "electrolyte":
        return P.K_S
    if dom in ELECTRODES:
        return P.K_S
    return P.K_G


def sigma_s_eff(dom: str) -> float:
    if dom == ANODE:
        return P.SIGMA_S_AN
    if dom == CATHODE:
        return P.SIGMA_S_CA
    return 0.0


def sigma_l_eff(dom: str, T=None):
    """The saved source uses NoCorr on both ionic and electronic conduction."""
    if dom == "electrolyte" or dom in ELECTRODES:
        if T is None:
            raise ValueError("Ionic conductivity requires the local temperature")
        return ionic_conductivity(T)
    return 0.0 if T is None else torch.zeros_like(T)


def rho_cp_eff(dom: str, rho_gas: torch.Tensor) -> torch.Tensor:
    """有效体积热容 (rho*cp)_eff。"""
    if dom in ELECTRODES or dom == "electrolyte":
        return torch.ones_like(rho_gas) * P.RHO_S * P.CP_S
    return rho_gas * P.CP_G



class PDEResiduals:
    """Differential operators; all policies supplied by the corrected public subclass."""

    def __init__(self, net, dm, device=None, balance=True, eq_weights=None,
                 hotspot_weights=None):
        self.net = net
        self.dm = dm
        self.device = device if device is not None else torch.device("cpu")
        self.balance = bool(balance)
        self.waveform_input = bool(getattr(self.dm, "waveform_input", False))
        self._ema = {}
        self._eq_weights = dict(eq_weights or {})   # extra multipliers on balanced terms
        # v20: hotspot corner oversampling/weighting (x~outlet, channel|electrode wall)
        # spec per domain: [x0, x1, y0, y1, weight]
        self._hotspot = {}
        for _dom, _spec in (hotspot_weights or {}).items():
            self._hotspot[str(_dom)] = (float(_spec[0]), float(_spec[1]),
                                        float(_spec[2]), float(_spec[3]), float(_spec[4]))
        self._stats()
        self._case_params()

    def _stats(self):
        xs, ys = self.dm.x_stats, self.dm.y_stats
        self.cont_ids = list(getattr(self.dm, "cont_ids", CONT_IDS))
        self.xm = torch.tensor([xs[c]["mean"] for c in self.cont_ids], dtype=torch.float32)
        self.xs = torch.tensor([xs[c]["std"] for c in self.cont_ids], dtype=torch.float32)
        self.ym = torch.tensor([ys[v]["mean"] for v in self.dm.target_vars], dtype=torch.float32)
        self.ys = torch.tensor([ys[v]["std"] for v in self.dm.target_vars], dtype=torch.float32)

    def _build_input(self, dom_id, x_p, y_p, t_p, case):
        """物理坐标/工况 -> 归一化网络输入 (B, n_cont+n_domain)。

        case: dict —— v19 波形参数 (Vc, amp, stepT, wtype, Tin, la, fu)；
              或 v18 瞬时电压 (V, Tin, la, fu)。
        """
        dev = x_p.device

        def _t(v):
            if isinstance(v, torch.Tensor):
                return v.to(dev)
            return torch.full((x_p.shape[0],), float(v), dtype=torch.float32, device=dev)

        if self.waveform_input:
            phys = torch.stack([x_p, y_p, t_p,
                                _t(case["Vc"]), _t(case["amp"]), _t(case["stepT"]),
                                _t(case["wtype"]),
                                _t(case["Tin"]), _t(case["la"]), _t(case["fu"])], dim=-1)
        else:
            phys = torch.stack([x_p, y_p, t_p, _t(case["V"]),
                                _t(case["Tin"]), _t(case["la"]), _t(case["fu"])], dim=-1)
        Xn = (phys - self.xm.to(dev)) / self.xs.to(dev)
        oh = torch.zeros(x_p.shape[0], len(DOMAIN_ORDER), device=dev)
        oh[:, dom_id] = 1.0
        return torch.cat([Xn, oh], dim=-1)

    def _forward(self, dom_id, x_p, y_p, t_p, case):
        """网络前向 + 反归一化 -> 物理量 (B, 12)。"""
        X = self._build_input(dom_id, x_p, y_p, t_p, case)
        out_n = self.net(X)
        dev = out_n.device
        return out_n * self.ys.to(dev) + self.ym.to(dev)

    def _d1(self, out, x_p, y_p, cols):
        """指定列对 (x, y) 的一阶导：返回 (dict col->gx, dict col->gy)。"""
        gx = {j: self._grad(out[:, j], x_p) for j in cols}
        gy = {j: self._grad(out[:, j], y_p) for j in cols}
        return gx, gy

    def _d2(self, gx, gy, x_p, y_p, cols):
        """指定列的二阶导（基于一阶导 dict）：gxx=∂gx/∂x, gyy=∂gy/∂y。"""
        gxx = {j: self._grad(gx[j], x_p) for j in cols}
        gyy = {j: self._grad(gy[j], y_p) for j in cols}
        return gxx, gyy

    def residuals(self, dom, x_p, y_p, t_p, case, transient=True):
        """计算一个域的全部残差（已按参考尺度缩放），返回 dict[str, Tensor]。

        稳态调用显式 transient=False（关闭时间导数）；瞬态调用传物理时间 t，
        保留 ∂/∂t 项（连续性/动量/组分/能量），电荷与 BV 按准稳态。
        case: v19 波形参数 dict 或 v18 瞬时电压 dict（见 _build_input）。
        """
        x_p = x_p.to(self.device)
        y_p = y_p.to(self.device)
        t_p = t_p.to(self.device)

        dom_id = DOMAIN_ID[dom]
        out = self._forward(dom_id, x_p, y_p, t_p, case)

        u = out[:, I["u"]];      v = out[:, I["v"]];      p = out[:, I["p"]]
        T = out[:, I["T"]];      w1 = out[:, I["w1"]];    w2 = out[:, I["w2"]]
        w3 = out[:, I["w3"]];    phis = out[:, I["phis"]]; phil = out[:, I["phil"]]
        htQ = out[:, I["htQ"]];  qav_an = out[:, I["qav_an"]]; qav_ca = out[:, I["qav_ca"]]

        # 一阶导（按域裁剪列）
        d1_cols = ["u", "v", "p", "T", "w1", "w2", "w3"]
        if dom in ELECTRODES:
            d1_cols += ["phis", "phil"]
        elif dom == "electrolyte":
            d1_cols += ["phil"]
        gx, gy = self._d1(out, x_p, y_p, [I[c] for c in d1_cols])
        ux = gx[I["u"]];   uy = gy[I["u"]]
        vx = gx[I["v"]];   vy = gy[I["v"]]
        px = gx[I["p"]];   py = gy[I["p"]]
        Tx = gx[I["T"]];   Ty = gy[I["T"]]
        w1x = gx[I["w1"]]; w1y = gy[I["w1"]]
        w2x = gx[I["w2"]]; w2y = gy[I["w2"]]
        w3x = gx[I["w3"]]; w3y = gy[I["w3"]]

        rho, (dMp, dMT, dMw1, dMw2, dMw3) = self._rho(out)
        rhox = dMp * px + dMT * Tx + dMw1 * w1x + dMw2 * w2x + dMw3 * w3x
        rhoy = dMp * py + dMT * Ty + dMw1 * w1y + dMw2 * w2y + dMw3 * w3y

        # 时间一阶导（瞬态项，M4）；t 为物理时间秒
        gt = {j: (self._grad(out[:, j], t_p) if transient else out[:, j] * 0.) for j in
              [I["u"], I["v"], I["p"], I["T"], I["w1"], I["w2"], I["w3"]]}
        ut = gt[I["u"]];   vt = gt[I["v"]];   pt = gt[I["p"]];   Tt = gt[I["T"]]
        w1t = gt[I["w1"]]; w2t = gt[I["w2"]]; w3t = gt[I["w3"]]
        rho_t = dMp * pt + dMT * Tt + dMw1 * w1t + dMw2 * w2t + dMw3 * w3t

        res = {}
        if dom != "electrolyte":
            # 二阶导（通道/电极）
            d2_cols = ["u", "v", "T", "w1", "w2", "w3"]
            if dom in ELECTRODES:
                d2_cols += ["phis", "phil"]
            gxx, gyy = self._d2(gx, gy, x_p, y_p, [I[c] for c in d2_cols])
            uxx = gxx[I["u"]]; uyy = gyy[I["u"]]
            vxx = gxx[I["v"]]; vyy = gyy[I["v"]]
            Txx = gxx[I["T"]]; Tyy = gyy[I["T"]]
            w1xx = gxx[I["w1"]]; w1yy = gyy[I["w1"]]
            w2xx = gxx[I["w2"]]; w2yy = gyy[I["w2"]]
            w3xx = gxx[I["w3"]]; w3yy = gyy[I["w3"]]

            # ---------------- 不可压缩连续性 div(u)=0 ----------------
            Rm = ux + vy
            res["mass"] = Rm / (U_REF / L_REF)

            # ---------------- 动量（Brinkman：电极含 Darcy 项） ----------------
            vx_y = self._grad(vx, y_p)          # d2v/dxdy
            uy_x = self._grad(uy, x_p)          # d2u/dydx
            divx = uxx + vx_y
            divy = vyy + uy_x
            Rmx = rho * (ut + u * ux + v * uy) + px - P.MU_G * (uxx + uyy) - P.MU_G * divx
            Rmy = rho * (vt + u * vx + v * vy) + py - P.MU_G * (vxx + vyy) - P.MU_G * divy
            if dom in ELECTRODES:
                Rmx = Rmx + (P.MU_G / P.KAPPA_P) * u
                Rmy = Rmy + (P.MU_G / P.KAPPA_P) * v
            res["momx"] = Rmx / REF["mom"]
            res["momy"] = Rmy / REF["mom"]

            # ---------------- 组分输运（源项按电化学计量自洽） ----------------
            div_ru = u * rhox + rho * ux + v * rhoy + rho * vy    # div(rho*u)
            R1 = w1 * rho_t + rho * w1t + w1 * div_ru + rho * (u * w1x + v * w1y) \
                 - P.D_EFF_GAS * (rhox * w1x + rho * w1xx + rhoy * w1y + rho * w1yy)
            R2 = w2 * rho_t + rho * w2t + w2 * div_ru + rho * (u * w2x + v * w2y) \
                 - P.D_EFF_GAS * (rhox * w2x + rho * w2xx + rhoy * w2y + rho * w2yy)
            R3 = w3 * rho_t + rho * w3t + w3 * div_ru + rho * (u * w3x + v * w3y) \
                 - P.D_EFF_GAS * (rhox * w3x + rho * w3xx + rhoy * w3y + rho * w3yy)
            if dom == ANODE:
                R1 = R1 + qav_an * P.M_H2 / (2.0 * P.F_CONST)     # H2 消耗
                R2 = R2 - qav_an * P.M_H2O / (2.0 * P.F_CONST)    # H2O 生成
            elif dom == CATHODE:
                R3 = R3 + qav_ca * P.M_O2 / (4.0 * P.F_CONST)     # O2 消耗
            res["w1"] = R1 / REF["w"]
            res["w2"] = R2 / REF["w"]
            res["w3"] = R3 / REF["w"]

            # ---------------- 能量（瞬态含 (ρc_p)_eff ∂T/∂t） ----------------
            rcp = rho_cp_eff(dom, rho)
            advection = rho * P.CP_G * (u * Tx + v * Ty) if dom in CHANNELS else T * 0.
            RT = rcp * Tt + advection - k_eff(dom) * (Txx + Tyy) - htQ
            res["T"] = RT / REF["T"]

            # ---------------- 电荷守恒 + BV 闭合（电极） ----------------
            if dom in ELECTRODES:
                sxx = gxx[I["phis"]]; syy = gyy[I["phis"]]
                lxx = gxx[I["phil"]]; lyy = gyy[I["phil"]]
                qav = qav_an if dom == ANODE else qav_ca
                # COMSOL: div(i_s)=-Av*i_loc; div(i_l)=+Av*i_loc.
                Rphis = -sigma_s_eff(dom) * (sxx + syy) + qav
                sigma_l = sigma_l_eff(dom, T)
                ionic_divergence = -self._grad(sigma_l * gx[I["phil"]], x_p) - self._grad(sigma_l * gy[I["phil"]], y_p)
                Rphil = ionic_divergence - qav
                res["phis"] = Rphis / REF["phis"]
                res["phil"] = Rphil / REF["phil"]

                local_current = reaction_current(dom, phis, phil, T, rho, w1, w3)
                area = P.AV_AN if dom == ANODE else P.AV_CA
                scale = REF["bv_an"] if dom == ANODE else REF["bv_ca"]
                qav_bv = area * local_current
                res["bv"] = (qav - qav_bv) / scale
        else:
            # ---------------- 电解质：纯导热（含 ρ_s c_p,s ∂T/∂t）+ 离子电荷 ----------------
            Tt_ely = self._grad(out[:, I["T"]], t_p) if transient else out[:, I["T"]] * 0.
            gxx, gyy = self._d2(gx, gy, x_p, y_p, [I["T"], I["phil"]])
            Txx = gxx[I["T"]]; Tyy = gyy[I["T"]]
            lxx = gxx[I["phil"]]; lyy = gyy[I["phil"]]
            RT = P.RHO_S * P.CP_S * Tt_ely - P.K_S * (Txx + Tyy) - htQ
            res["T"] = RT / REF["T"]
            sigma_l = sigma_l_eff(dom, T)
            Rphil = -self._grad(sigma_l * gx[I["phil"]], x_p) - self._grad(sigma_l * gy[I["phil"]], y_p)
            res["phil"] = Rphil / REF["phil"]
        return res

    def _hotspot_mask(self, dom, pts):
        """v20: per-point weight for hotspot corner regions (1.0 outside)."""
        w = np.ones(pts.shape[0], dtype=np.float64)
        hs = self._hotspot.get(dom)
        if hs is not None:
            x0, x1, y0, y1, ww = hs
            m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            w[m] = ww
        return w

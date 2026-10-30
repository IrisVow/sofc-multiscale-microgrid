# -*- coding: utf-8 -*-
"""M3/M4 PDE 残差：5 域守恒方程（稳态 + 瞬态时间导数）+ 界面连续性 + BV 闭合。

方程与符号见 outputs/pinndata/PINN构建方案.md §4（与 sofc1_FINAL2_solved.mph 一致）。
- 组分：w1=H2, w2=H2O, w3=O2, w4=N2（质量分数，w4=1-w1-w2-w3）
- 源项一致性（数据已自洽）：rH2=-qav_an*M_H2/(2F), rH2O=+qav_an*M_H2O/(2F),
  rO2=-qav_ca*M_O2/(4F)（单位 kg/(m^3*s)）
- 瞬态项（M4）：连续性 ρ∂ρ/∂t、动量 ρ·∂u/∂t、组分 ∂(ρw_i)/∂t、
-   能量 (ρc_p)_eff·∂T/∂t（电解质 ρ_s·c_p,s·∂T/∂t）；电荷守恒按电化学
-   准稳态假设保留（电荷弛豫 ~µs 远快于热/流动 ~100s）。
- 残差在物理量空间计算（网络输出反归一化），配点为物理坐标；
  所有 autograd 导数保留计算图，使残差可回传到网络参数。
"""
from __future__ import annotations

import numpy as np
import torch

from . import params as P
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
    "flux_T": K_REF * DT_REF / L_REF,
    "flux_phil": SIGL_REF * PHIL_REF / L_REF,
    "flux_phis": SIGS_REF * PHIS_REF / L_REF,
    "mass_flux": RHO_REF * U_REF,
}


def k_eff(dom: str) -> float:
    """有效导热系数 W/(m*K)。"""
    if dom == "electrolyte":
        return P.K_S
    if dom in ELECTRODES:
        return P.EPS_P * P.K_G + (1.0 - P.EPS_P) * P.K_S
    return P.K_G


def sigma_s_eff(dom: str) -> float:
    if dom == ANODE:
        return P.SIGMA_S_AN * (1.0 - P.EPS_P) / P.TAU_P
    if dom == CATHODE:
        return P.SIGMA_S_CA * (1.0 - P.EPS_P) / P.TAU_P
    return 0.0


def sigma_l_eff(dom: str) -> float:
    if dom == "electrolyte":
        return P.SIGMA_L_YSZ
    if dom in ELECTRODES:
        return P.SIGMA_L_YSZ * (1.0 - P.EPS_P) / P.TAU_P
    return 0.0


def rho_cp_eff(dom: str, rho_gas: torch.Tensor) -> torch.Tensor:
    """有效体积热容 (rho*cp)_eff。"""
    if dom in ELECTRODES:
        return P.EPS_P * rho_gas * P.CP_G + (1.0 - P.EPS_P) * P.RHO_S * P.CP_S
    return rho_gas * P.CP_G


class PDEResiduals:
    """对给定网络计算 5 域稳态 PDE 残差与界面连续性损失。"""

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

    # ------------------------------------------------------------------
    def _stats(self):
        xs, ys = self.dm.x_stats, self.dm.y_stats
        self.cont_ids = list(getattr(self.dm, "cont_ids", CONT_IDS))
        self.xm = torch.tensor([xs[c]["mean"] for c in self.cont_ids], dtype=torch.float32)
        self.xs = torch.tensor([xs[c]["std"] for c in self.cont_ids], dtype=torch.float32)
        self.ym = torch.tensor([ys[v]["mean"] for v in self.dm.target_vars], dtype=torch.float32)
        self.ys = torch.tensor([ys[v]["std"] for v in self.dm.target_vars], dtype=torch.float32)

    def _case_params(self):
        keys = ["V_applied", "T_in_K", "lambda_air", "FU"]
        self._cases = {}
        df = self.dm.df_steady
        # all steady cases (train+val+test): PDE/BC also constrain the S9
        # temperature-extrapolation operating point
        for cid in df["case_id"].unique():
            sub = df[df["case_id"] == cid]
            if len(sub):
                self._cases[str(cid)] = [float(sub[k].iloc[0]) for k in keys]
        if not self._cases:
            self._cases = {"S1": [0.9, 1073.0, 7.2, 0.64]}
        self._case_ids = list(self._cases.keys())
        self._case_list = list(self._cases.values())
        # inlet mean velocity u_in (data mean at x<1e-3, per case x channel)
        self._u_in = {}
        for cid, sub in df.groupby("case_id"):
            for dom in CHANNELS:
                subd = sub[(sub["domain"] == dom) & (sub["x"] < 1e-3)]
                if len(subd):
                    self._u_in[(str(cid), dom)] = float(subd["u"].mean())
        # transient case table: (V_cell, T_in, lambda, FU, amp, stepT, wave_type, t_max)
        self._tcases = {}
        if self.dm.df_transient is not None:
            tdf = self.dm.df_transient
            for cid, sub in tdf.groupby("case_id"):
                r0 = sub.iloc[0]
                Vc = float(r0["V_cell"]) if "V_cell" in sub.columns else float(r0["V_applied"])
                Tin = float(r0["T_in_K"]); la = float(r0["lambda_air"]); fu = float(r0["FU"])
                if self.waveform_input:
                    amp = float(r0["amp"]); stepT = float(r0["stepT"]); wtype = int(r0["wave_type"])
                else:
                    stepT = float(sub["stepT_s"].iloc[0]) if "stepT_s" in sub.columns else 0.0
                    amp = float(sub["stepAmp_V"].iloc[0]) if "stepAmp_V" in sub.columns else 0.0
                    wtype = 0
                self._tcases[str(cid)] = (Vc, Tin, la, fu, amp, stepT, wtype,
                                          float(sub["t_s"].max()))
        self._tcase_ids = list(self._tcases.keys())
        self._tcase_list = list(self._tcases.values())

    def _rand_case(self, rng: np.random.Generator):
        i = int(rng.integers(len(self._case_list)))
        cid = self._case_ids[i]
        c = self._case_list[i]
        return str(cid), float(c[0]), float(c[1]), float(c[2]), float(c[3])

    def _rand_t(self, rng: np.random.Generator, t_max: float, stepT: float = None):
        """时间采样（阶跃窗加权）：
        - 20% 落在阶跃窗 [stepT-20, stepT+60]（响应最剧烈段）；
        - 40% 落在动态段 [0, min(400,t_max)]；
        - 40% 均匀全区间 [0, t_max]。"""
        if stepT is not None and rng.random() < 0.2:
            lo, hi = max(0.0, stepT - 20.0), min(t_max, stepT + 60.0)
            if hi > lo:
                return float(rng.uniform(lo, hi))
        if rng.random() < 0.5:
            return float(rng.uniform(0.0, min(400.0, t_max)))
        return float(rng.uniform(0.0, t_max))

    def _rand_transient_case(self, rng: np.random.Generator):
        """随机瞬态工况 + 随机时间 t（波形参数用于生成 V_applied 与网络输入）。"""
        i = int(rng.integers(len(self._tcase_list)))
        cid = self._tcase_ids[i]
        Vc, Tin, la, fu, amp, stepT, wtype, t_max = self._tcase_list[i]
        st_sam = stepT if wtype == 0 else None    # 阶跃窗仅对 step 波形
        t = self._rand_t(rng, t_max, stepT=st_sam)
        return str(cid), t, Vc, Tin, la, fu, amp, stepT, wtype

    # ------------------------------------------------------------------
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

    def _grad(self, out, x):
        """d(out)/dx，out 与 x 同形 (B,)，保留计算图（供二阶与回传）。"""
        return torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out),
                                   create_graph=True, retain_graph=True)[0]

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

    # ------------------------------------------------------------------
    def _rho(self, out):
        """理想气体密度 rho=p*M_mix/(R*T) 及其对 (p,T,w1,w2,w3) 的解析偏导。"""
        p = out[:, I["p"]]
        T = out[:, I["T"]]
        w1 = out[:, I["w1"]]
        w2 = out[:, I["w2"]]
        w3 = out[:, I["w3"]]
        w4 = 1.0 - w1 - w2 - w3
        invM = w1 / P.M_H2 + w2 / P.M_H2O + w3 / P.M_O2 + w4 / P.M_N2
        Mmix = 1.0 / invM.clamp(min=1e-9)
        Mmix = Mmix.clamp(0.002, 0.03)            # ???? [H2, N2]
        rho = (p * Mmix / (P.R_CONST * T)).clamp(min=1e-6, max=5.0)
        dM1 = -Mmix ** 2 * (1.0 / P.M_H2 - 1.0 / P.M_N2)
        dM2 = -Mmix ** 2 * (1.0 / P.M_H2O - 1.0 / P.M_N2)
        dM3 = -Mmix ** 2 * (1.0 / P.M_O2 - 1.0 / P.M_N2)
        c = p / (P.R_CONST * T)
        dMp = Mmix / (P.R_CONST * T)
        dMT = -p * Mmix / (P.R_CONST * T ** 2)
        return rho, (dMp, dMT, c * dM1, c * dM2, c * dM3)

    # ------------------------------------------------------------------
    def residuals(self, dom, x_p, y_p, t_p, case):
        """计算一个域的全部残差（已按参考尺度缩放），返回 dict[str, Tensor]。

        稳态调用传 t_p=0（时间导数项自动消失）；瞬态调用传物理时间 t，
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
        gt = {j: self._grad(out[:, j], t_p) for j in
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

            # ---------------- 连续性（瞬态含 ∂ρ/∂t） ----------------
            Rm = rho_t + u * rhox + rho * ux + v * rhoy + rho * vy
            res["mass"] = Rm / REF["mass"]

            # ---------------- 动量（Brinkman：电极含 Darcy 项） ----------------
            vx_y = self._grad(vx, y_p)          # d2v/dxdy
            uy_x = self._grad(uy, x_p)          # d2u/dydx
            divx = uxx + vx_y
            divy = vyy + uy_x
            Rmx = rho * (ut + u * ux + v * uy) + px - P.MU_G * (uxx + uyy) - (P.MU_G / 3.0) * divx
            Rmy = rho * (vt + u * vx + v * vy) + py - P.MU_G * (vxx + vyy) - (P.MU_G / 3.0) * divy
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
            RT = rcp * (Tt + u * Tx + v * Ty) - k_eff(dom) * (Txx + Tyy) - htQ
            res["T"] = RT / REF["T"]

            # ---------------- 电荷守恒 + BV 闭合（电极） ----------------
            if dom in ELECTRODES:
                sxx = gxx[I["phis"]]; syy = gyy[I["phis"]]
                lxx = gxx[I["phil"]]; lyy = gyy[I["phil"]]
                qav = qav_an if dom == ANODE else qav_ca
                Rphis = -sigma_s_eff(dom) * (sxx + syy) - qav
                Rphil = -sigma_l_eff(dom) * (lxx + lyy) + qav
                res["phis"] = Rphis / REF["phis"]
                res["phil"] = Rphil / REF["phil"]

                if dom == ANODE:
                    eta = phis - phil - P.EEQ_AN
                    jv, scale = J0_AN_VOL, REF["bv_an"]
                else:
                    eta = phis - phil - (P.EEQ_CA + P.DEEQDT_CA * (T - T_REF_BV))
                    jv, scale = J0_CA_VOL, REF["bv_ca"]
                z = (P.F_CONST * eta / (P.R_CONST * T)).clamp(-20.0, 20.0)
                qav_bv = jv * (torch.exp(P.ALPHA_A * z) - torch.exp(-P.ALPHA_C * z))
                res["bv"] = (qav - qav_bv) / scale
        else:
            # ---------------- 电解质：纯导热（含 ρ_s c_p,s ∂T/∂t）+ 离子电荷 ----------------
            Tt_ely = self._grad(out[:, I["T"]], t_p)
            gxx, gyy = self._d2(gx, gy, x_p, y_p, [I["T"], I["phil"]])
            Txx = gxx[I["T"]]; Tyy = gyy[I["T"]]
            lxx = gxx[I["phil"]]; lyy = gyy[I["phil"]]
            RT = P.RHO_S * P.CP_S * Tt_ely - P.K_S * (Txx + Tyy) - htQ
            res["T"] = RT / REF["T"]
            Rphil = -P.SIGMA_L_YSZ * (lxx + lyy)
            res["phil"] = Rphil / REF["phil"]
        return res

    # ------------------------------------------------------------------
    def _hotspot_mask(self, dom, pts):
        """v20: per-point weight for hotspot corner regions (1.0 outside)."""
        w = np.ones(pts.shape[0], dtype=np.float64)
        hs = self._hotspot.get(dom)
        if hs is not None:
            x0, x1, y0, y1, ww = hs
            m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            w[m] = ww
        return w

    def domain_loss(self, dom, rng, n_per_domain, use_bv=True,
                    transient=False, t_fixed=None, pts_override=None):
        """域 PDE 残差损失。

        transient=False：稳态配点（t=0，V 取稳态工况）。
        transient=True：瞬态配点 (x,y,t)，工况参数与 V(t) 由阶跃波形生成；
            t_fixed 给定时固定时刻（如 t=0 用于初始条件 L_ic）。
        pts_override：外部给定的配点数组 (N,2)，跳过 sample_collocation（IC 角点加密用）。
        """
        if pts_override is not None:
            pts = pts_override[dom] if isinstance(pts_override, dict) else pts_override
        else:
            pts = sample_collocation(n_per_domain, rng)[dom]
        w_pt = self._hotspot_mask(dom, pts)
        wt = torch.tensor(w_pt, dtype=torch.float32, device=self.device)
        wsum = float(wt.sum().clamp(min=1.0))
        x = torch.tensor(pts[:, 0], dtype=torch.float32, requires_grad=True, device=self.device)
        y = torch.tensor(pts[:, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        if transient:
            _cid, t, Vc, Tin, la, fu, amp, stepT, wtype = self._rand_transient_case(rng)
            if t_fixed is not None:
                t = float(t_fixed)
            t_p = torch.full_like(x, t, requires_grad=True)
            if self.waveform_input:
                case = dict(Vc=Vc, amp=amp, stepT=stepT, wtype=wtype,
                            Tin=Tin, la=la, fu=fu)
            else:
                # V 由阶跃波形给出：heaviside 不可导，用 t_p.detach() 使 V
                # 成为常量输入（V 对 t 的导数本应为 0，不参与网络回传）
                V = Vc - amp * torch.heaviside(t_p.detach() - stepT, torch.zeros_like(t_p))
                case = dict(V=V, Tin=Tin, la=la, fu=fu)
        else:
            _cid, V, Tin, la, fu = self._rand_case(rng)
            t_p = torch.zeros_like(x, requires_grad=True)
            if self.waveform_input:
                case = dict(Vc=V, amp=0.0, stepT=0.0, wtype=0, Tin=Tin, la=la, fu=fu)
            else:
                case = dict(V=V, Tin=Tin, la=la, fu=fu)
        res = self.residuals(dom, x, y, t_p, case)
        means = {}
        for name, r in res.items():
            if r is None or (name == "bv" and not use_bv):
                continue
            # v20: hotspot-weighted L1-like mean (w=1 outside hotspot -> plain mean)
            means[name] = torch.sum(torch.sqrt(r * r + 1e-8) * wt) / wsum
        loss = torch.zeros((), device=x.device)
        if self.balance and means:
            # per-(domain,equation) EMA inverse scaling: each term contributes
            # ~m/EMA ~ O(1), so huge 2nd-derivative residuals cannot dominate
            # the gradient; weights track the running residual magnitude
            for name, m in means.items():
                key = f"{dom}|{name}"
                ema = self._ema.get(key)
                if ema is None:
                    ema = float(m.detach())
                ema = 0.9 * ema + 0.1 * float(m.detach())
                self._ema[key] = ema
                w = min(max(1.0 / max(ema, 1e-6), 1e-6), 1e6)
                w = w * float(self._eq_weights.get(name, 1.0))
                loss = loss + w * m
        else:
            for m in means.values():
                loss = loss + m
        return loss

    def interface_loss(self, rng, n_per_iface, scale=1.0):
        """???????? backward?????????????"""
        total = 0.0
        cnt = 0
        for a, b, pts in sample_interfaces(n_per_iface, rng):
            _cid, V, Tin, la, fu = self._rand_case(rng)
            x = torch.tensor(pts[:, 0], dtype=torch.float32, requires_grad=True, device=self.device)
            y = torch.tensor(pts[:, 1], dtype=torch.float32, requires_grad=True, device=self.device)
            tz = torch.zeros_like(x)
            if self.waveform_input:
                case = dict(Vc=V, amp=0.0, stepT=0.0, wtype=0, Tin=Tin, la=la, fu=fu)
            else:
                case = dict(V=V, Tin=Tin, la=la, fu=fu)
            oa = self._forward(DOMAIN_ID[a], x, y, tz, case)
            ob = self._forward(DOMAIN_ID[b], x, y, tz, case)
            _, ga = self._d1(oa, x, y, [I["T"], I["phil"], I["phis"]])
            _, gb = self._d1(ob, x, y, [I["T"], I["phil"], I["phis"]])
            Ty_a = ga[I["T"]]; Ty_b = gb[I["T"]]
            items = [(oa[:, I["T"]] - ob[:, I["T"]]) / DT_REF]
            items.append((k_eff(a) * Ty_a - k_eff(b) * Ty_b) / REF["flux_T"])
            if (a in ELECTRODES and b == "electrolyte"):
                ly_a = ga[I["phil"]]; ly_b = gb[I["phil"]]
                items.append((oa[:, I["phil"]] - ob[:, I["phil"]]) / PHIL_REF)
                items.append((sigma_l_eff(a) * ly_a - sigma_l_eff(b) * ly_b) / REF["flux_phil"])
                items.append(sigma_s_eff(a) * ga[I["phis"]] / REF["flux_phis"])
            elif (a == "electrolyte" and b in ELECTRODES):
                ly_a = ga[I["phil"]]; ly_b = gb[I["phil"]]
                items.append((oa[:, I["phil"]] - ob[:, I["phil"]]) / PHIL_REF)
                items.append((sigma_l_eff(a) * ly_a - sigma_l_eff(b) * ly_b) / REF["flux_phil"])
                items.append(sigma_s_eff(b) * gb[I["phis"]] / REF["flux_phis"])
            elif (a in CHANNELS and b in ELECTRODES) or (b in CHANNELS and a in ELECTRODES):
                rho_a = self._rho(oa)[0]
                rho_b = self._rho(ob)[0]
                items.append((rho_a * oa[:, I["v"]] - rho_b * ob[:, I["v"]]) / REF["mass_flux"])
            loss_iface = sum(torch.mean(torch.sqrt(it * it + 1e-8)) for it in items)
            total = total + float(loss_iface.detach())
            (scale * loss_iface).backward()
            cnt += 1
        return total / max(1, cnt)

    W_IN = {"fuel_channel": (0.95, 0.05, 0.0),
            "air_channel": (0.0, 0.0, 0.21)}

    def _edge_forward(self, dom, x, y, case, t=0.0):
        n = len(x)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return self._forward(DOMAIN_ID[dom], x_t, y_t,
                             torch.full((n,), t, dtype=torch.float32, device=self.device),
                             case)

    def boundary_loss(self, rng, n_per_edge, scale=1.0, do_backward=True,
                      include_transient=True):
        """Inlet/outlet/collector soft BCs (plan sec 4.4). Backward per edge.

        v19: 额外对瞬态工况施加 collector BC phis = v_wave(t; 波形参数) 与
        inlet BC（T_in/w/v），使网络从波形参数端到端复现外加电压轨迹。
        """
        total = 0.0
        cnt = 0
        for cid, c in zip(self._case_ids, self._case_list):
            V, Tin, la, fu = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            if self.waveform_input:
                case = dict(Vc=V, amp=0.0, stepT=0.0, wtype=0, Tin=Tin, la=la, fu=fu)
            else:
                case = dict(V=V, Tin=Tin, la=la, fu=fu)
            for dom in ("fuel_channel", "air_channel"):
                y0, y1 = P.Y_LAYERS[dom]
                # inlet x=0: T=T_in, v=0, w fixed, u=u_in(data mean)
                y = rng.uniform(y0, y1, n_per_edge)
                x = np.zeros_like(y)
                o = self._edge_forward(dom, x, y, case)
                w_in = self.W_IN[dom]
                items = [(o[:, I["T"]] - Tin) / DT_REF,
                         o[:, I["v"]] / U_REF,
                         (o[:, I["w1"]] - w_in[0]) / W_REF,
                         (o[:, I["w2"]] - w_in[1]) / W_REF,
                         (o[:, I["w3"]] - w_in[2]) / W_REF]
                u_in = self._u_in.get((cid, dom))
                if u_in is not None:
                    items.append((o[:, I["u"]] - u_in) / U_REF)
                loss_e = sum(torch.mean(torch.sqrt(it * it + 1e-8)) for it in items)
                total += float(loss_e.detach()); cnt += 1
                if do_backward:
                    (scale * loss_e).backward()
                # outlet x=0.1: p = P_REF
                x = np.full(n_per_edge, 0.1)
                y = rng.uniform(y0, y1, n_per_edge)
                o = self._edge_forward(dom, x, y, case)
                loss_o = torch.mean(torch.sqrt((o[:, I["p"]] - P.P_REF) ** 2 + 1e-8)) / 1.0e5
                total += float(loss_o.detach()); cnt += 1
                if do_backward:
                    (scale * loss_o).backward()
            # collector surfaces (electrode | channel interface y):
            #   air electrode bottom phis=V_applied, fuel electrode top phis=0
            for dom, yc, vapp in (("air_electrode", 0.8e-3, True),
                                  ("fuel_electrode", 1.82e-3, False)):
                x = rng.uniform(0.0, 0.1, n_per_edge)
                y = np.full_like(x, yc)
                o = self._edge_forward(dom, x, y, case)
                target = V if vapp else 0.0
                loss_c = torch.mean(torch.sqrt((o[:, I["phis"]] - target) ** 2 + 1e-8)) / PHIS_REF
                total += float(loss_c.detach()); cnt += 1
                if do_backward:
                    (scale * loss_c).backward()
        # v19: transient collector/inlet BC —— phis = v_wave(t; waveform params)
        if include_transient and self.waveform_input and self._tcase_list:
            from physics.waveforms import v_wave
            for cid, c in zip(self._tcase_ids, self._tcase_list):
                Vc, Tin, la, fu = float(c[0]), float(c[1]), float(c[2]), float(c[3])
                amp, stepT, wtype, t_max = float(c[4]), float(c[5]), int(c[6]), float(c[7])
                st_sam = stepT if wtype == 0 else None
                case = dict(Vc=Vc, amp=amp, stepT=stepT, wtype=wtype,
                            Tin=Tin, la=la, fu=fu)
                for dom, yc, vapp in (("air_electrode", 0.8e-3, True),
                                      ("fuel_electrode", 1.82e-3, False)):
                    x = rng.uniform(0.0, 0.1, n_per_edge)
                    y = np.full_like(x, yc)
                    t = self._rand_t(rng, t_max, stepT=st_sam)
                    o = self._edge_forward(dom, x, y, case, t=t)
                    target = v_wave(t, Vc, amp, wtype, stepT).to(self.device)
                    loss_c = torch.mean(torch.sqrt((o[:, I["phis"]] - target) ** 2 + 1e-8)) / PHIS_REF
                    total += float(loss_c.detach()); cnt += 1
                    if do_backward:
                        (scale * loss_c).backward()
                for dom in ("fuel_channel", "air_channel"):
                    y0, y1 = P.Y_LAYERS[dom]
                    y = rng.uniform(y0, y1, n_per_edge)
                    x = np.zeros_like(y)
                    t = self._rand_t(rng, t_max, stepT=st_sam)
                    o = self._edge_forward(dom, x, y, case, t=t)
                    w_in = self.W_IN[dom]
                    items = [(o[:, I["T"]] - Tin) / DT_REF,
                             o[:, I["v"]] / U_REF,
                             (o[:, I["w1"]] - w_in[0]) / W_REF,
                             (o[:, I["w2"]] - w_in[1]) / W_REF,
                             (o[:, I["w3"]] - w_in[2]) / W_REF]
                    loss_e = sum(torch.mean(torch.sqrt(it * it + 1e-8)) for it in items)
                    total += float(loss_e.detach()); cnt += 1
                    if do_backward:
                        (scale * loss_e).backward()
        return total / max(1, cnt)

    def ic_loss(self, rng, n_per_domain, use_bv=True, scale=1.0, t_fixed=0.0,
               n_hotspot=0):
        """时间端点条件损失：瞬态工况在固定时刻 t_fixed 满足其（准）稳态 PDE。

        - t_fixed=0（L_ic，M4）：瞬态工况 t=0 处 NN(x,y,0,·)=初始稳态解
          （数据快照 t=0 为均匀初始场，V=V_cell）；
        - t_fixed=t_max（L_final，M4b）：瞬态末态 NN(x,y,t_max,·) 应逼近
          对应电压的稳态解（如 T3 末态 V=0.8 -> S5），缓解末态系统偏差。
        - n_hotspot>0（v20）：在热点角点区域（x~0.1 出口近电极壁）追加配点，
          加强 t=0 初值约束（E4 局限修复项 1）。
        逐域 backward 以省显存。
        """
        total = 0.0
        cnt = 0
        for dom in DOMAIN_ORDER:
            base = sample_collocation(n_per_domain, rng)[dom]
            pts = base
            if n_hotspot:
                hs = self._hotspot.get(dom)
                if hs is not None:
                    x0, x1, y0, y1, _w = hs
                    n_add = int(n_hotspot[dom]) if isinstance(n_hotspot, dict) else int(n_hotspot)
                    hx = rng.uniform(x0, x1, n_add)
                    hy = rng.uniform(y0, y1, n_add)
                    pts = np.concatenate([base, np.stack([hx, hy], axis=1)], axis=0)
            loss = self.domain_loss(dom, rng, n_per_domain, use_bv=use_bv,
                                    transient=True, t_fixed=float(t_fixed),
                                    pts_override=pts)
            total += float(loss.detach())
            (scale * loss).backward()
            cnt += 1
        return total / max(1, cnt)

    def pre_step_loss(self, rng, n_per_domain, use_bv=True, scale=1.0):
        """阶跃前时间因果损失（v20，E4 局限修复项 2）。

        对 step 型瞬态工况强制 t<0.9*stepT 时全场 T≈T_in（数据/COMSOL 均验证
        阶跃前为均匀初场），消除模型“预知”阶跃导致的提前升温（t=0..stepT
        PINN Tmax 曾偏高 27-38 K）。仅 wave_type==0（阶跃）工况，热点区域加权。
        """
        total = 0.0
        cnt = 0
        step_cases = [(cid, c) for cid, c in zip(self._tcase_ids, self._tcase_list)
                      if int(c[6]) == 0 and float(c[5]) > 1.0]
        if not step_cases or not self.waveform_input:
            return 0.0
        for cid, c in step_cases:
            Vc, Tin, la, fu, amp, stepT, wtype, t_max = c
            case = dict(Vc=Vc, amp=amp, stepT=stepT, wtype=wtype, Tin=Tin, la=la, fu=fu)
            for dom in DOMAIN_ORDER:
                n = int(n_per_domain[dom]) if isinstance(n_per_domain, dict) else int(n_per_domain)
                pts = sample_collocation(n, rng)[dom]
                t = rng.uniform(0.0, 0.9 * stepT, n)
                x = torch.tensor(pts[:, 0], dtype=torch.float32, device=self.device)
                y = torch.tensor(pts[:, 1], dtype=torch.float32, device=self.device)
                t_p = torch.tensor(t, dtype=torch.float32, device=self.device)
                out = self._forward(DOMAIN_ID[dom], x, y, t_p, case)
                r = torch.sqrt((out[:, I["T"]] - Tin) ** 2 + 1e-8) / DT_REF
                wt = torch.tensor(self._hotspot_mask(dom, pts), dtype=torch.float32,
                                  device=self.device)
                loss = torch.sum(r * wt) / float(wt.sum().clamp(min=1.0))
                total += float(loss.detach())
                (scale * loss).backward()
                cnt += 1
        return total / max(1, cnt)

    def epoch_pde_loss(self, rng, n_per_domain, n_iface, n_edge=256,
                       use_bv=True, lambda_int=1.0, lambda_bc=1.0):
        total = torch.zeros((), device=self.device)
        for dom in DOMAIN_ORDER:
            total = total + self.domain_loss(dom, rng, n_per_domain, use_bv)
        lpde = total / len(DOMAIN_ORDER)
        lint = self.interface_loss(rng, n_iface) * lambda_int
        lbc = self.boundary_loss(rng, n_edge, do_backward=False) * lambda_bc
        return lpde, lint, lbc
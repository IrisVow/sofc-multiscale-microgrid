# -*- coding: utf-8 -*-
"""D4 电堆/系统放大律：单电池（PINN 单位通道）→ 全电池 → 电堆（串并联）→
系统（BOP 功耗、后燃、热回收）→ 装机功率与几何。

自下而上（本模块）：
    单电池几何（COMSOL DOE：L=0.1 m 流道长度，W_ch=1.5 mm，W_rib=1.0 mm，
    节距 2.5 mm；2D 模型 = 单通道重复单元）→ 全电池（cell_width_m 沿节距
    重复 n_channel 次，默认 10 cm → A_cell = 100 cm²）→ 名义电流
    I = i_nom × A_cell（0.18 A/cm² × 100 = 18 A）→ 单电池功率 V_BOL × I
    （0.85 V × 18 A = 15.3 W）→ 电堆 N_series（V_stack_target/V_BOL，
    默认 400 V → 471 节）→ 系统 N_parallel = ceil(P_dc_req/P_stack_dc)，
    P_dc_req = P_nom/(η_inv·η_dcdc·(1−bop_frac)) → 净 AC 功率（含 BOP 扣除）。

自上而下交叉核对（与 C5/Table 2）：
    系统包络体积 = P_nom/15 kW/m³（150 kW → 10 m³）；电堆核心体积只占
    ~4–5%（其余为 BOP/换热/保温/封装）——与 SOFC 工程经验（堆芯占系统
    体积 5–10%）下界一致。

热/燃料交叉核对（与 0D SOFCSystem）：
    0D 名义点 P_fuel = P_dc/η_net(1.0)；本模块按 P_dc_req（含 BOP + 取整）
    给出更保守的燃料预算（~7% 差异 ≈ 1/(1−bop) − 1 + 取整裕度）；
    后燃热 = P_fuel×(1−U_f)、回收热 = P_fuel×η_heat(1.0) 与 0D 量级一致。

不确定性（详见 docs/model_assumptions.md §D4）：
    1) 电流密度 i_nom（0.18 A/cm² 为标定值，论文范围 0.1–0.23）：±20% →
       电池数 ∓20%（线性）。
    2) 2D 单位通道 → 3D 全电池外推：假设通道均匀、忽略端部/歧管损失
       （~5–10% 面积损失）。
    3) V_stack_target 影响串并联配比与取整裕度（利用率 ≤ 1）。
    4) BOP 功耗默认 5%（工程范围 3–8%）：每 +1% 需 +~2% 堆直流功率。
    5) 堆几何为下限估计（不含歧管/端板/保温），系统级排布以 Table 2 为准。

调用：
    calc = StackScaling()                 # 读 configs/components.yaml + scaling.yaml
    res  = calc.compute()                 # 汇总 dict（电池/电堆/系统/几何/热/敏感性）
    calc.to_csv("results/scaling_summary.csv")   # 扁平汇总 CSV（UTF-8）
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import yaml

from .params import DENSITIES, load_params
from .sofc_system import SOFCSystem


def _read_yaml(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class StackScaling:
    """自下而上电堆/系统放大计算器（阶段 D4）。"""

    DEFAULTS = {
        # 单电池几何（PINN/COMSOL DOE）
        "cell_length_m": 0.10,        # 流道方向长度（PINN L=100 mm）
        "channel_width_m": 1.5e-3,    # 通道宽 W_ch
        "rib_width_m": 1.0e-3,        # 肋宽 W_rib（节距 = W_ch + W_rib = 2.5 mm）
        "cell_width_m": 0.10,         # 全电池宽度（沿节距方向，10 cm → 40 通道周期）
        # 电堆电气设计
        "v_stack_target_v": 400.0,    # 单堆直流母线目标电压（N_series = V_target/V_bol）
        # 系统
        "bop_frac": 0.05,             # BOP 功耗占电堆直流功率（鼓风机/泵/控制/热管理）
        "cell_pitch_stack_m": 3.0e-3, # 堆内重复单元厚度（电池+互联+框架，~3 mm）
        "frame_frac": 1.30,           # 电池面积 → 堆足迹放大（边框/歧管）
        "bop_volume_frac": 0.50,      # 堆外 BOP/辅机体积占电堆核心体积比例
    }

    def __init__(self, config_dir: str = None, overrides: dict = None,
                 with_sensitivity: bool = True):
        cfg_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
        self.config_dir = cfg_dir
        self.params = load_params(str(cfg_dir / "components.yaml"))
        scaling_yaml = _read_yaml(cfg_dir / "scaling.yaml")
        self.cfg = dict(self.DEFAULTS)
        self.cfg.update(scaling_yaml or {})
        if overrides:
            self.cfg.update(overrides)
        self.p_sofc = self.params["sofc"]
        self.p_nom = float(self.cfg.get("p_nom_kw") or self.p_sofc.get("p_nom_kw", 150.0))
        self.sofc = SOFCSystem(self.p_sofc)   # 0D 交叉核对用
        self.with_sensitivity = bool(with_sensitivity)
        self.result = None

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        c = self.cfg
        p = self.p_sofc
        # ---- 电池级 ----
        pitch_m = c["channel_width_m"] + c["rib_width_m"]
        n_channels = c["cell_width_m"] / pitch_m
        a_cell_m2 = c["cell_length_m"] * c["cell_width_m"]
        a_cell_cm2 = a_cell_m2 * 1e4
        i_nom = float(c.get("i_nom_A_cm2") or p["i_nom_A_cm2"])
        v_bol = float(p["v_cell_bol"])
        i_cell_A = i_nom * a_cell_cm2
        p_cell_dc_kW = v_bol * i_cell_A / 1000.0

        # ---- 电堆级 ----
        n_series = int(round(c["v_stack_target_v"] / v_bol))
        v_stack_dc = n_series * v_bol
        p_stack_dc_kW = n_series * p_cell_dc_kW

        # ---- 系统级 ----
        eta_inv = float(p["eta_inv"])
        eta_dcdc = float(p["eta_dcdc"])
        bop = float(c["bop_frac"])
        p_dc_req_kW = self.p_nom / eta_inv / eta_dcdc / (1.0 - bop)
        n_cells_total = int(math.ceil(p_dc_req_kW / p_cell_dc_kW))
        n_parallel = int(math.ceil(n_cells_total / n_series))
        n_cells_installed = n_parallel * n_series
        p_installed_dc_kW = n_cells_installed * p_cell_dc_kW
        p_bop_kW = bop * p_installed_dc_kW
        p_net_ac_kW = p_installed_dc_kW * (1.0 - bop) * eta_dcdc * eta_inv  # BOP 为直流侧寄生，先扣再变换
        utilization = self.p_nom / p_net_ac_kW

        # ---- 几何（堆芯为下限估计）----
        stack_height_m = n_series * c["cell_pitch_stack_m"]
        stack_footprint_m2 = a_cell_m2 * c["frame_frac"]
        stack_volume_m3 = stack_height_m * stack_footprint_m2
        plant_stack_vol_m3 = n_parallel * stack_volume_m3
        plant_vol_m3 = plant_stack_vol_m3 * (1.0 + c["bop_volume_frac"])

        # ---- 系统包络（Table 2 密度，自上而下交叉核对）----
        d = DENSITIES["sofc"]
        envelope_vol_m3 = self.p_nom / d["p_vol_kW_m3"]
        envelope_weight_ton = self.p_nom / d["p_grav_kW_ton"]
        stack_share = plant_stack_vol_m3 / envelope_vol_m3

        # ---- 热/燃料（0D 交叉核对）----
        eta_net_nom = self.sofc.eta_net(1.0)
        uf_nom = self.sofc.u_f(1.0)
        eta_heat_nom = self.sofc.eta_heat(1.0)
        p_fuel_req_kW = p_dc_req_kW / eta_net_nom
        q_afterburner_kW = p_fuel_req_kW * (1.0 - uf_nom)
        q_recover_kW = p_fuel_req_kW * eta_heat_nom
        p_dc_0d_kW = self.p_nom / eta_inv
        p_fuel_0d_kW = p_dc_0d_kW / eta_net_nom
        fuel_ratio = p_fuel_req_kW / p_fuel_0d_kW

        res = {
            "p_nom_kw": self.p_nom,
            "cell": {
                "pitch_m": pitch_m, "n_channels": n_channels,
                "a_cell_m2": a_cell_m2, "a_cell_cm2": a_cell_cm2,
                "i_nom_A_cm2": i_nom, "i_cell_A": i_cell_A,
                "v_cell_nom_v": v_bol, "p_cell_dc_w": p_cell_dc_kW * 1000.0,
            },
            "stack": {
                "v_target_v": c["v_stack_target_v"], "n_series": n_series,
                "v_stack_dc": v_stack_dc, "p_stack_dc_kW": p_stack_dc_kW,
                "height_m": stack_height_m, "footprint_m2": stack_footprint_m2,
                "volume_m3": stack_volume_m3,
            },
            "system": {
                "eta_inv": eta_inv, "eta_dcdc": eta_dcdc, "bop_frac": bop,
                "p_dc_req_kW": p_dc_req_kW, "n_cells_total": n_cells_total,
                "n_parallel": n_parallel, "n_cells_installed": n_cells_installed,
                "p_installed_dc_kW": p_installed_dc_kW, "p_bop_kW": p_bop_kW,
                "p_net_ac_kW": p_net_ac_kW, "utilization": utilization,
                "cells_per_kw": n_cells_installed / self.p_nom,
                "stacks": n_parallel,
                "plant_stack_volume_m3": plant_stack_vol_m3,
                "plant_volume_m3": plant_vol_m3,
                "envelope_volume_m3": envelope_vol_m3,
                "envelope_weight_ton": envelope_weight_ton,
                "stack_core_share": stack_share,
            },
            "thermal": {
                "eta_net_nominal": eta_net_nom, "u_f_nominal": uf_nom,
                "p_fuel_req_kW": p_fuel_req_kW, "q_afterburner_kW": q_afterburner_kW,
                "q_recover_kW": q_recover_kW, "p_fuel_0d_kW": p_fuel_0d_kW,
                "fuel_budget_ratio_vs_0d": fuel_ratio,
            },
        }
        self.result = res
        if self.with_sensitivity:
            res["sensitivity"] = self._sensitivity()
        return self.result

    # ------------------------------------------------------------------
    def _sensitivity(self) -> dict:
        """关键参数变动的电池数/配比/利用率（相对基准）。"""
        base = self.result
        n0 = base["system"]["n_cells_installed"]
        cases = {
            "p_nom_1p5x": {"p_nom_kw": self.p_nom * 1.5},
            "i_nom_minus20pct": {"i_nom_A_cm2": self.p_sofc["i_nom_A_cm2"] * 0.8},
            "i_nom_plus20pct": {"i_nom_A_cm2": self.p_sofc["i_nom_A_cm2"] * 1.2},
            "v_stack_200v": {"v_stack_target_v": 200.0},
            "v_stack_600v": {"v_stack_target_v": 600.0},
            "bop_3pct": {"bop_frac": 0.03},
            "bop_8pct": {"bop_frac": 0.08},
        }
        out = {}
        for name, ov in cases.items():
            r = StackScaling(config_dir=str(self.config_dir), overrides=ov,
                              with_sensitivity=False).compute()
            out[name] = {"n_cells": r["system"]["n_cells_installed"],
                         "n_parallel": r["system"]["n_parallel"],
                         "n_series": r["stack"]["n_series"],
                         "utilization": r["system"]["utilization"],
                         "ratio_n_cells": r["system"]["n_cells_installed"] / n0}
        return out

    # ------------------------------------------------------------------
    def to_csv(self, path: str = None):
        """导出扁平汇总 CSV（UTF-8）。path 给出时写文件并返回 path；否则返回文本。"""
        if self.result is None:
            self.compute()
        res = self.result
        rows = [
            ("p_nom_kw", f"{res['p_nom_kw']:.1f}", "kW"),
            ("cell.pitch_m", f"{res['cell']['pitch_m'] * 1e3:.2f}", "mm"),
            ("cell.n_channels", f"{res['cell']['n_channels']:.0f}", "-"),
            ("cell.a_cell_cm2", f"{res['cell']['a_cell_cm2']:.1f}", "cm2"),
            ("cell.i_cell_A", f"{res['cell']['i_cell_A']:.2f}", "A"),
            ("cell.p_cell_dc_w", f"{res['cell']['p_cell_dc_w']:.2f}", "W"),
            ("stack.n_series", f"{res['stack']['n_series']}", "cells"),
            ("stack.v_stack_dc", f"{res['stack']['v_stack_dc']:.2f}", "V"),
            ("stack.p_stack_dc_kW", f"{res['stack']['p_stack_dc_kW']:.2f}", "kW"),
            ("stack.height_m", f"{res['stack']['height_m']:.3f}", "m"),
            ("system.n_parallel", f"{res['system']['n_parallel']}", "stacks"),
            ("system.n_cells_installed", f"{res['system']['n_cells_installed']}", "cells"),
            ("system.cells_per_kw", f"{res['system']['cells_per_kw']:.1f}", "cells/kW"),
            ("system.p_installed_dc_kW", f"{res['system']['p_installed_dc_kW']:.2f}", "kW"),
            ("system.p_bop_kW", f"{res['system']['p_bop_kW']:.2f}", "kW"),
            ("system.p_net_ac_kW", f"{res['system']['p_net_ac_kW']:.2f}", "kW"),
            ("system.utilization", f"{res['system']['utilization']:.4f}", "-"),
            ("system.envelope_volume_m3", f"{res['system']['envelope_volume_m3']:.2f}", "m3"),
            ("system.stack_core_share", f"{res['system']['stack_core_share']:.4f}", "-"),
            ("thermal.p_fuel_req_kW", f"{res['thermal']['p_fuel_req_kW']:.1f}", "kW"),
            ("thermal.q_afterburner_kW", f"{res['thermal']['q_afterburner_kW']:.1f}", "kW"),
            ("thermal.q_recover_kW", f"{res['thermal']['q_recover_kW']:.1f}", "kW"),
            ("thermal.fuel_budget_ratio_vs_0d", f"{res['thermal']['fuel_budget_ratio_vs_0d']:.4f}", "-"),
        ]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["key", "value", "unit"])
        writer.writerows(rows)
        if path:
            with io.open(path, "w", encoding="utf-8", newline="") as f:
                f.write(buf.getvalue())
            return path
        return buf.getvalue()
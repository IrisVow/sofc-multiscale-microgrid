# -*- coding: utf-8 -*-
"""微电网调度与整定（阶段 C3）：按论文层级（SOFC -> 机组 -> 电池）协调组件。

每步流程：
    EMS 设定值 -> SOFC.step -> Genset.step -> 电池按实际出力兜底（Eq.19）
    -> Boiler 热平衡（回收热优先，缺口锅炉补足，多余排弃）
    -> EmissionsBook 记账 -> 功率平衡检查 -> 记录历史。

调用：
    mg = Microgrid()                       # 默认读 configs/components.yaml + ems.yaml
    rep = mg.step(60.0, p_load_kw, q_load_kw)
    df  = mg.to_dataframe()
    mg.run_days(7, load_fn, heat_fn, dt_s=60)   # load_fn(hour)->(p_kw, q_kw) 或序列
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np

from ..components import (Battery, Boiler, EmissionsBook, Genset, SOFCSystem,
                          load_params as load_component_params, merge_params)
from .ems import EMS, load_ems_params


class Microgrid:
    """组件 + EMS + 排放记账的整定容器。"""

    def __init__(self, config_dir: str = None, ems_params: dict = None,
                 genset_fuel: str = None, boiler_fuel: str = None, coupling=None,
                 component_overrides: dict = None):
        cfg_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
        comp = load_component_params(str(cfg_dir / "components.yaml"))
        if component_overrides:                    # E1 容量扫描：覆盖装机（sofc.p_nom_kw / battery.e_nom_kwh 等）
            comp = merge_params(comp, component_overrides)
        ems_p = load_ems_params(str(cfg_dir / "ems.yaml"))
        if ems_params:
            ems_p.update(ems_params)

        self.sofc = SOFCSystem(comp["sofc"])
        g = comp["genset"]
        self.genset = Genset(params=g, p_nom_kw=float(g.get("p_nom_kw", 100.0)),
                             fuel=genset_fuel or str(g.get("fuel", "LNG")))
        self.battery = Battery(comp["battery"])
        self.boiler = Boiler(comp["boiler"])
        self.ems = EMS(ems_p)
        self.emissions = EmissionsBook()
        self.boiler_fuel = boiler_fuel or "LNG"
        self.history = []
        self.coupling = coupling      # PinnCoupling | None（阶段 D1：PINN 监视）
        self.reset()

    def reset(self):
        self.sofc.reset()
        self.genset.reset()
        self.battery.reset()
        self.ems.reset()
        self.emissions.reset()
        self.history = []
        return self

    # ------------------------------------------------------------------
    def step(self, dt_s: float, p_load_kw: float, q_load_kw: float) -> dict:
        """推进一个时间步，返回该步明细。"""
        dt_s = float(dt_s)
        sp = self.ems.compute_setpoints(dt_s, p_load_kw, self.battery.soc,
                                        self.sofc.p_nom, self.genset.p_nom)
        req_kw = float(sp["sofc_set_kw"])
        cinfo = {"pinn_constraint": "off", "pinn_tmax_req_k": float("nan"),
                 "pinn_derate_kw": 0.0, "pinn_derate_full": False}
        if (self.coupling is not None and self.coupling.constraint_mode == "derate"
                and sp["sofc_operating"] and req_kw > 0.0):
            p_der, cinfo = self._tmax_derate(req_kw)
            self.coupling.note_derate(cinfo.get("pinn_derate_kw", 0.0), dt_s,
                                      full=cinfo.get("pinn_derate_full", False))
            sp = dict(sp)
            sp["sofc_set_kw"] = p_der
        r_sofc = self.sofc.step(dt_s, sp["sofc_set_kw"], operating=sp["sofc_operating"])
        r_gen = self.genset.step(dt_s, sp["genset_set_kw"])
        pin = self.coupling.monitor_step(r_sofc) if self.coupling is not None else {}
        bat_set = EMS.battery_setpoint(p_load_kw, r_sofc["p_ac"], r_gen["p_kw"])
        r_bat = self.battery.step(dt_s, bat_set)

        # 热平衡：回收热（SOFC + 机组）优先，锅炉补足
        q_rec = r_sofc["q_heat"] + r_gen["q_heat_kw"]
        r_boil = self.boiler.step(q_load_kw, q_rec, fuel=self.boiler_fuel, dt_s=dt_s)

        # 排放记账
        self.emissions.add_sofc("sofc", r_sofc)
        self.emissions.add_genset("genset", r_gen)
        self.emissions.add_boiler("boiler", r_boil, fuel=self.boiler_fuel)

        # 平衡与记录
        net_kw = r_sofc["p_ac"] + r_gen["p_kw"] + r_bat["p_kw"]
        imbalance_kw = float(p_load_kw) - net_kw      # >0 欠供；<0 富余
        row = {
            "t_s": (len(self.history) * dt_s),
            "p_load_kw": float(p_load_kw), "q_load_kw": float(q_load_kw),
            "p_add_kw": sp["p_add_kw"], "soc": r_bat["soc"],
            "sofc_kw": r_sofc["p_ac"], "genset_kw": r_gen["p_kw"],
            "bat_kw": r_bat["p_kw"], "net_kw": net_kw,
            "imbalance_kw": imbalance_kw,
            "sofc_eta": r_sofc["eta_net"], "sofc_i_pu": r_sofc["i_pu"],
            "sofc_soh": r_sofc["soh"],
            "sofc_r_deg": r_sofc["r_deg_pct_1000h"],
            "sofc_v_cell": r_sofc["v_cell"], "sofc_u_f": r_sofc["u_f"],
            "q_rec_kw": q_rec, "q_boiler_kw": r_boil["q_boiler_kw"],
            "q_dumped_kw": r_boil["q_dumped_kw"],
            "fuel_sofc_kw": r_sofc["p_fuel"], "fuel_genset_kw": r_gen["fuel_kwh"],
            "fuel_boiler_kw": r_boil["fuel_kwh"],
            "sofc_on": sp["sofc_on"], "charge_mode": sp["charge_mode"],
            "pinn_req_kw": req_kw, "pinn_derate_kw": cinfo["pinn_derate_kw"],
            "pinn_tmax_req_k": cinfo["pinn_tmax_req_k"],
            "pinn_constraint": cinfo["pinn_constraint"],
            "dt_s": dt_s,
        }
        row.update(pin)      # 阶段 D1：PINN 监视列（pinn_*）
        self.history.append(row)
        return row

    # ------------------------------------------------------------------
    def _tmax_derate(self, p_set_kw: float) -> tuple:
        """D3：PINN Tmax 约束降载（软惩罚）。

        在 [derate_floor_pu*P_nom, p_set] 内二分找最大满足
        tmax <= tmax_limit + deadband 的功率；返回 (derated_p, info)。
        Tmax 随功率单调增（电流↑→产热↑），二分有效。
        """
        c = self.coupling
        limit = c.tmax_limit + c.derate_deadband_k - c.derate_margin_k  # D2: 目标=限值-裕度
        floor = c.derate_floor_pu * self.sofc.p_nom
        info = {"pinn_constraint": "ok", "pinn_tmax_req_k": float("nan"),
                "pinn_derate_kw": 0.0, "pinn_derate_full": False}

        def query(p_kw):
            p_pu = min(float(p_kw) / self.sofc.p_nom, 1.0)
            p_dc = max(0.0, float(p_kw)) / self.sofc.eta_inv
            i_pu = self.sofc.current_for_power(p_dc / self.sofc.p_nom)
            v = self.sofc.v_cell(i_pu)
            fu = self.sofc.u_f(p_pu)
            return c.evaluate(v, fu)

        q0 = query(p_set_kw)
        info["pinn_tmax_req_k"] = float(q0.get("tmax", float("nan")))
        if not q0.get("ok") or q0["tmax"] <= limit:
            return p_set_kw, info                      # 未超限：不降载
        lo, hi = max(floor, 0.0), p_set_kw
        q_lo = query(lo)
        if not q_lo.get("ok") or q_lo["tmax"] > limit:
            best, full = lo, True                      # 下限仍超限：降到底并标记
        else:
            full = False
            for _ in range(int(c.derate_bisect_iters)):
                mid = 0.5 * (lo + hi)
                if query(mid)["tmax"] <= limit:
                    lo = mid
                else:
                    hi = mid
            best = lo
        info["pinn_derate_kw"] = max(0.0, p_set_kw - best)
        info["pinn_derate_full"] = full
        info["pinn_constraint"] = "full" if full else "derate"
        info["pinn_tmax_derated_k"] = float(query(best).get("tmax", float("nan")))
        return best, info
    def run_days(self, days: float, load_fn, heat_fn=None, dt_s: float = 60.0,
                 seed: int = 0) -> list:
        """按天运行：load_fn(hour_of_day)->p_kw、heat_fn(hour_of_day)->q_kw。
        支持返回元组 (p, q)。若传入数组（长度 = 天*24h/dt 步），按索引取。"""
        dt_s = float(dt_s)
        n_steps = int(round(days * 86400.0 / dt_s))
        for k in range(n_steps):
            t_day_h = (k * dt_s) / 3600.0
            if callable(load_fn):
                v = load_fn(t_day_h)
                if isinstance(v, (tuple, list)):
                    p_l, q_l = v[0], v[1]
                else:
                    p_l, q_l = v, (heat_fn(t_day_h) if heat_fn else 0.0)
            else:
                p_l = float(load_fn[int(k) % len(load_fn)])
                q_l = float(heat_fn[int(k) % len(heat_fn)]) if heat_fn is not None else 0.0
            self.step(dt_s, p_l, q_l)
        return self.history

    # ------------------------------------------------------------------
    def to_dataframe(self):
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return None

    def summary(self) -> dict:
        """整段仿真的汇总指标。"""
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return {}
        s = self.emissions.summary()
        dt_h = float(df["dt_s"].iloc[0]) / 3600.0
        out = {
            "hours": float(len(df)) * dt_h,
            "energy_load_kwh": float(df["p_load_kw"].sum() * dt_h),
            "energy_sofc_kwh": float(df["sofc_kw"].sum() * dt_h),
            "energy_genset_kwh": float(df["genset_kw"].sum() * dt_h),
            "energy_bat_kwh": float(df["bat_kw"].sum() * dt_h),
            "imbalance_kwh": float(df["imbalance_kw"].sum() * dt_h),
            "max_imbalance_kw": float(df["imbalance_kw"].abs().max()),
            "soc_min": float(df["soc"].min()), "soc_max": float(df["soc"].max()),
            "sofc_soh_end": float(df["sofc_soh"].iloc[-1]),
            "ghg_kg_co2eq": float(s["ghg_kg_co2eq"]),
            "nox_kg": float(s["nox_kg"]),
            "fuel_sofc_kwh": float(df["fuel_sofc_kw"].sum() * dt_h),
            "fuel_genset_kwh": float(df["fuel_genset_kw"].sum() * dt_h),
            "fuel_boiler_kwh": float(df["fuel_boiler_kw"].sum() * dt_h),
        }
        if self.coupling is not None:
            out.update(self.coupling.summary())     # 阶段 D1：PINN 监视统计
        return out
# -*- coding: utf-8 -*-
"""排放核算（阶段 C2）：GHG（CO2 + CH4 x GWP25）+ NOx/SOx/PM。

口径（TTW，tank-to-wake）：
- CO2：由燃料含碳量按 LHV 化学计量（MGO 3.206 kgCO2/kg；CH4/LNG 2.750 kgCO2/kg）
- CH4 滑移：SOFC 重整 0.5%、LNG 贫燃机 3.5%（Comer 政策默认 6% 可调）、锅炉按 0.1%
- NOx/SOx/PM：发电机组按负荷查找表（g/kWh）；SOFC 与锅炉近似为 0（电化学/低排放燃烧）

调用：
    book = EmissionsBook()
    book.add_source("sofc", p_fuel_kwh=..., fuel="LNG", ch4_slip=0.005)
    book.add_source("genset", p_kwh=..., fuel="LNG", load_pu=0.6)
    summary = book.summary()
"""
from __future__ import annotations

import numpy as np

from .params import CHEM, PARAMS


def _interp(tbl, x):
    xs = np.array([r[0] for r in tbl], dtype=float)
    ys = np.array([r[1] for r in tbl], dtype=float)
    return float(np.interp(x, xs, ys))


class EmissionsBook:
    def __init__(self, gwp_ch4: float = None):
        self.gwp_ch4 = float(gwp_ch4 if gwp_ch4 is not None else CHEM["GWP_CH4"])
        self.sources = {}
        self._reset_totals()

    def _reset_totals(self):
        self.tot = {"co2_g": 0.0, "ch4_g": 0.0, "nox_g": 0.0,
                    "sox_g": 0.0, "pm_g": 0.0, "fuel_kwh": 0.0}

    def reset(self):
        self.sources = {}
        self._reset_totals()
        return self

    def _fuel_kg_per_kwh(self, fuel: str) -> float:
        lhv = {"MGO": CHEM["LHV_MGO_kJ_kg"], "LNG": CHEM["LHV_LNG_kJ_kg"]}[fuel]
        return 3.6e3 / lhv                                   # kg per kWh(LHV)

    def _co2_kg_per_kg(self, fuel: str) -> float:
        return {"MGO": CHEM["MGO_CO2_kg_per_kg"],
                "LNG": CHEM["CH4_CO2_kg_per_kg"]}[fuel]

    def add_fuel(self, source: str, fuel: str, p_fuel_kwh: float,
                 ch4_slip: float = 0.0, eta: float = 1.0,
                 p_el_kwh: float = None, load_pu: float = 1.0) -> dict:
        """按燃料能量（kWh LHV）核算 CO2/CH4；NOx/SOx/PM 仅对发电机组（按发电量）。"""
        e = max(0.0, float(p_fuel_kwh))
        kg_fuel = e * self._fuel_kg_per_kwh(fuel)
        co2_g = kg_fuel * self._co2_kg_per_kg(fuel) * 1000.0
        ch4_g = kg_fuel * ch4_slip * 1000.0
        nox = sox = pm = 0.0
        if p_el_kwh:
            g = PARAMS["genset"]
            nox = p_el_kwh * _interp(g[f"nox_g_kwh_{fuel.lower()}"], load_pu)
            sox = p_el_kwh * _interp(g[f"sox_g_kwh_{fuel.lower()}"], load_pu)
            pm = p_el_kwh * _interp(g[f"pm_g_kwh_{fuel.lower()}"], load_pu)
        row = {"co2_g": co2_g, "ch4_g": ch4_g, "nox_g": nox,
               "sox_g": sox, "pm_g": pm, "fuel_kwh": e}
        self.sources[source] = row
        for k in row:
            self.tot[k] += row[k]
        return row

    def add_genset(self, source: str, rep: dict) -> dict:
        """从 Genset.step() 报告直接入账（含引燃油 3% MGO）。"""
        fuel = rep.get("fuel", "LNG")
        p_el = rep["p_kw"] * rep["dt_s"] / 3600.0            # kWh
        p_fuel = rep["fuel_kwh"] * rep["dt_s"] / 3600.0      # kWh
        slip = PARAMS["genset"]["ch4_slip"] if fuel == "LNG" else 0.0
        row = self.add_fuel(source, fuel, p_fuel, ch4_slip=slip,
                            p_el_kwh=p_el, load_pu=rep["load_pu"])
        # 双燃料机组引燃油（LNG + 3% MGO 引燃油）
        pilot = rep.get("pilot_kwh", 0.0) * rep["dt_s"] / 3600.0
        if pilot > 0:
            pr = self.add_fuel(f"{source}_pilot", "MGO", pilot, ch4_slip=0.0)
            row["pilot_co2_g"] = pr["co2_g"]
        return row

    def add_sofc(self, source: str, rep: dict, ch4_slip: float = None) -> dict:
        """从 SOFCSystem.step() 报告入账（LNG 重整，CH4 滑移默认 0.5%）。"""
        p_fuel = rep["p_fuel"] * rep["dt_s"] / 3600.0        # kWh
        if ch4_slip is None:
            ch4_slip = PARAMS["sofc"]["ch4_slip"]
        return self.add_fuel(source, "LNG", p_fuel, ch4_slip=ch4_slip)

    def add_boiler(self, source: str, rep: dict, fuel: str = "LNG",
                   ch4_slip: float = 0.001) -> dict:
        p_fuel = rep["fuel_kwh"] * rep.get("dt_s", 0.0) / 3600.0
        return self.add_fuel(source, fuel, p_fuel, ch4_slip=ch4_slip)

    def summary(self) -> dict:
        t = self.tot
        ghg = t["co2_g"] + self.gwp_ch4 * t["ch4_g"]
        return {
            "co2_kg": t["co2_g"] / 1000.0,
            "ch4_kg": t["ch4_g"] / 1000.0,
            "ghg_kg_co2eq": ghg / 1000.0,
            "nox_kg": t["nox_g"] / 1000.0,
            "sox_kg": t["sox_g"] / 1000.0,
            "pm_kg": t["pm_g"] / 1000.0,
            "fuel_kwh": t["fuel_kwh"],
        }
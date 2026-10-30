# -*- coding: utf-8 -*-
"""C5 尺寸/重量模型（论文 Table 2 功率/能量密度 + S_fuel 燃料舱）。

依据 configs/components.yaml（C2 装机）与 configs/scenario.yaml（C1 定稿）计算
各组件排布体积（m3）与重量（ton），并按自持天数 + 燃料裕度 S_fuel 估算
LNG（主燃料）/MGO（备用）燃料舱容积；计算参数可被 configs/sizing.yaml 覆盖
（autonomy_days、deck_height_m、燃料效率假设等）。

单位与公式约定：
- 功率型组件（sofc/dg/gg/boiler）：体积 = P / p_vol，重量 = P / p_grav；
- 能量型组件（电池/燃料舱）：体积 = E / e_vol，重量 = E / e_grav；
- 日燃料能量 = 日电量 / 名义效率（日电量 = 年电量(846.3 MWh, C4 标定)/365；
  主燃料 LNG 用 SOFC 名义效率 eta_net(1.0)≈0.54，备用 MGO 用柴油机满负荷
  效率 ≈0.42；可用参数 fuel_energy_per_day_kwh 直接覆盖日燃料能量）。
- 燃料舱容积 = 日燃料能量 × autonomy_days × (1 + S_fuel) / e_vol。

调用：
    calc = SizingCalculator()                 # 读 configs/ 下 components/scenario/sizing yaml
    res  = calc.compute()                     # 汇总 dict（组件 + 燃料舱 + 合计）
    rows = calc.to_table()                    # 逐组件行（含合计），便于打印
    calc.to_csv("results/sizing_summary.csv") # 导出 CSV（UTF-8）
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import yaml

from .params import DENSITIES, load_params
from .sofc_system import SOFCSystem


def _interp(table, x):
    """一维线性插值（table: [[x0,y0],...]；x 超界按端点值）。"""
    xs = np.array([r[0] for r in table], dtype=float)
    ys = np.array([r[1] for r in table], dtype=float)
    return float(np.interp(x, xs, ys))


def _read_yaml(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _power_size(key, rating_kw):
    """功率型组件：体积/重量 = 额定功率 / 功率密度（kW/m3、kW/ton）。"""
    d = DENSITIES[key]
    return {"volume_m3": rating_kw / d["p_vol_kW_m3"],
            "weight_ton": rating_kw / d["p_grav_kW_ton"]}


def _energy_size(key, rating_kwh):
    """能量型组件（电池/燃料舱）：体积/重量 = 能量 / 能量密度（kWh/m3、kWh/ton）。"""
    d = DENSITIES[key]
    return {"volume_m3": rating_kwh / d["e_vol_kWh_m3"],
            "weight_ton": rating_kwh / d["e_grav_kWh_ton"]}


class SizingCalculator:
    """尺寸/重量计算器（阶段 C5）。"""

    DEFAULTS = {
        "annual_energy_mwh": 846.3,   # C4 标定年电量（check_load.py）
        "autonomy_days": 7.0,         # 燃料自持天数
        "deck_height_m": 3.0,         # 甲板高度假设（甲板面积 = 体积/高度）
        "boiler_kw": 120.0,           # 锅炉装机（params.py 无默认；C4 热峰 60-80 kW 上浮 50%）
        "gg_kw": 0.0,                 # 燃气机（可选，0 = 不装）
    }

    def __init__(self, config_dir: str = None, overrides: dict = None):
        """读 configs/components.yaml + scenario.yaml + sizing.yaml；overrides 最高优先。"""
        cfg_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
        self.config_dir = cfg_dir
        self.params = load_params(str(cfg_dir / "components.yaml"))
        self.scenario = _read_yaml(cfg_dir / "scenario.yaml")
        sizing_yaml = _read_yaml(cfg_dir / "sizing.yaml")
        self.cfg = dict(self.DEFAULTS)
        self.cfg.update(sizing_yaml or {})
        if overrides:
            self.cfg.update(overrides)
        self.result = None

    # ------------------------------------------------------------------
    def _get(self, *keys, default=None):
        """按路径取 scenario.yaml 中的值。"""
        node = self.scenario
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ------------------------------------------------------------------
    # 装机与假设解析（sizing 配置 > yaml 装机 > 内置默认）
    # ------------------------------------------------------------------
    def _sofc_kw(self):
        return float(self.cfg.get("sofc_kw")
                     or self._get("components", "sofc_capacity_kw")
                     or self.params["sofc"].get("p_nom_kw") or 150.0)

    def _genset_kw(self):
        return float(self.cfg.get("genset_kw")
                     or self._get("components", "genset_capacity_kw")
                     or self.params["genset"].get("p_nom_kw") or 100.0)

    def _battery_kwh(self):
        return float(self.cfg.get("battery_kwh")
                     or self.params["battery"].get("e_nom_kwh") or 100.0)

    def _annual_energy_mwh(self):
        v = self.cfg.get("annual_energy_mwh")
        if v is None:
            v = self._get("load", "annual_energy_mwh")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(self.DEFAULTS["annual_energy_mwh"])

    def _eta_sofc(self):
        """主燃料名义净效率：默认 SOFC eta_net(1.0)（BOL 名义点 0.54）。"""
        v = self.cfg.get("eta_sofc_nominal")
        if v is None:
            v = SOFCSystem(self.params["sofc"]).eta_net(1.0)
        return float(v)

    def _eta_genset(self):
        """备用机组名义效率：默认柴油机 MGO 满负荷点（0.42）。"""
        v = self.cfg.get("eta_genset_nominal")
        if v is None:
            v = _interp(self.params["genset"]["eta_eff_mgo"], 1.0)
        return float(v)

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        """计算组件体积/重量、燃料舱容积与合计；返回汇总 dict。"""
        sofc_kw = self._sofc_kw()
        genset_kw = self._genset_kw()
        gg_kw = float(self.cfg.get("gg_kw") or 0.0)
        boiler_kw = float(self.cfg.get("boiler_kw") or self.DEFAULTS["boiler_kw"])
        battery_kwh = self._battery_kwh()
        annual_mwh = self._annual_energy_mwh()
        autonomy = float(self.cfg.get("autonomy_days", 7.0))
        s_margin = float(self.cfg.get("safety_margin_fuel",
                                      self.params["general"].get("safety_margin_fuel", 0.10)))
        deck_h = float(self.cfg.get("deck_height_m", 3.0))
        eta_sofc = self._eta_sofc()
        eta_gen = self._eta_genset()

        # 日燃料能量 = 日电量 / 名义效率（可用 fuel_energy_per_day_kwh 直接覆盖）
        daily_el = annual_mwh * 1000.0 / 365.0
        fep = self.cfg.get("fuel_energy_per_day_kwh")
        daily_lng = float(fep) if fep is not None else daily_el / eta_sofc
        daily_mgo = float(fep) if fep is not None else daily_el / eta_gen

        # 组件：功率型（功率密度） + 能量型（能量密度）
        components = {
            "sofc": {"name": "SOFC 系统 (LNG)", "rating_kw": sofc_kw,
                     **_power_size("sofc", sofc_kw)},
            "genset": {"name": "柴油机组 (MGO 备用)", "rating_kw": genset_kw,
                       **_power_size("dg", genset_kw)},
            "boiler": {"name": "锅炉 (LNG/MGO 双燃料)", "rating_kw": boiler_kw,
                       **_power_size("boiler_lng", boiler_kw)},
            "battery": {"name": "电池 (Li-ion)", "rating_kwh": battery_kwh,
                        **_energy_size("bat", battery_kwh)},
        }
        if gg_kw > 0:
            components["gg"] = {"name": "燃气机 (LNG)", "rating_kw": gg_kw,
                                **_power_size("gg", gg_kw)}

        # 燃料舱：舱能量 = 日燃料能量 × 自持天数 × (1 + S_fuel)
        tank_e_lng = daily_lng * autonomy * (1.0 + s_margin)
        tank_e_mgo = daily_mgo * autonomy * (1.0 + s_margin)
        fuel_tanks = {
            "lng": {"fuel": "LNG", "daily_fuel_energy_kwh": daily_lng,
                    "tank_energy_kwh": tank_e_lng, **_energy_size("fuel_lng", tank_e_lng)},
            "mgo": {"fuel": "MGO", "daily_fuel_energy_kwh": daily_mgo,
                    "tank_energy_kwh": tank_e_mgo, **_energy_size("fuel_mgo", tank_e_mgo)},
        }

        eq_vol = sum(c["volume_m3"] for c in components.values())
        eq_wt = sum(c["weight_ton"] for c in components.values())
        fuel_vol = sum(t["volume_m3"] for t in fuel_tanks.values())
        fuel_wt = sum(t["weight_ton"] for t in fuel_tanks.values())
        total_vol = eq_vol + fuel_vol
        total_wt = eq_wt + fuel_wt

        self.result = {
            "annual_energy_mwh": annual_mwh,
            "daily_electricity_kwh": daily_el,
            "autonomy_days": autonomy,
            "safety_margin_fuel": s_margin,
            "deck_height_m": deck_h,
            "eta_sofc_nominal": eta_sofc,
            "eta_genset_nominal": eta_gen,
            "components": components,
            "fuel_tanks": fuel_tanks,
            "equipment_volume_m3": eq_vol,
            "equipment_weight_ton": eq_wt,
            "fuel_volume_m3": fuel_vol,
            "fuel_weight_ton": fuel_wt,
            "total_volume_m3": total_vol,
            "total_weight_ton": total_wt,
            "deck_area_m2": total_vol / deck_h if deck_h > 0 else None,
        }
        return self.result

    # ------------------------------------------------------------------
    def to_table(self) -> list:
        """逐组件/燃料舱/合计行（component, rating, volume_m3, weight_ton）。"""
        if self.result is None:
            self.compute()
        res = self.result
        rows = []
        for c in res["components"].values():
            rating = (f'{c["rating_kw"]:.1f} kW' if "rating_kw" in c
                      else f'{c["rating_kwh"]:.1f} kWh')
            rows.append({"component": c["name"], "rating": rating,
                         "volume_m3": round(c["volume_m3"], 4),
                         "weight_ton": round(c["weight_ton"], 4)})
        for t in res["fuel_tanks"].values():
            rows.append({"component": f"燃料舱 {t['fuel']}",
                         "rating": (f'{t["daily_fuel_energy_kwh"]:.1f} kWh/d × '
                                    f'{res["autonomy_days"]:.0f} d'),
                         "volume_m3": round(t["volume_m3"], 4),
                         "weight_ton": round(t["weight_ton"], 4)})
        rows.append({"component": "合计", "rating": "含燃料舱",
                     "volume_m3": round(res["total_volume_m3"], 4),
                     "weight_ton": round(res["total_weight_ton"], 4)})
        return rows

    def to_csv(self, path: str = None):
        """导出 CSV（UTF-8）。path 给出时写文件并返回 path；否则返回 CSV 文本。"""
        fieldnames = ["component", "rating", "volume_m3", "weight_ton"]
        if path:
            with io.open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.to_table())
            return path
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(self.to_table())
        return buf.getvalue()
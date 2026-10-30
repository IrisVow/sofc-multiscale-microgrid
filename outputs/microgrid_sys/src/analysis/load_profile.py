# -*- coding: utf-8 -*-
"""负荷剖面合成模块（阶段 C4）。

依据 configs/scenario.yaml（C1 定稿，UTF-8）合成海岛微电网电/热负荷曲线：
- 居民负荷：早峰 07-09、晚峰 18-22、夜间低谷、午间回落；峰值 90 kW。
- 旅游/商业负荷：日间主导 08-18、傍晚延续；夏季峰值 x1.6、冬季低谷 x0.6
  （C1 范围：夏季 x1.5-2.0、冬季 x0.5-0.7）；肩季参考峰值 43.75 kW，
  夏季峰值 70 kW，与居民晚峰同时 -> 年峰值 200 kW。
- 海水淡化：恒定 40 kW；desalination_schedulable=true 时可平移
  （EMS 削峰填谷决策空间，C3/D 期使用）。
- 热负荷：生活热水/采暖/淡化预热，早晚双峰，峰值约 75 kW、均值约 49 kW
  （工程假设；后续可用论文游轮热负荷数据标定）。

季节维度：4 季（spring/summer/autumn/winter）x 工作日/周末 x 24 h 基准曲线。
generate_series(days, dt_s, season=None, weekend=None, ...) 合成任意长度序列，
time_index=True 返回带时间索引的 DataFrame，False 返回 (p_el_kw, q_load_kw)。

标定（scripts/check_load.py 校验）：峰值季节工作日最大电负荷 = 200 kW（±0.5%）；
全年电量（365 d @ 60 s）落在 720-880 MWh；淡化恒 40 kW；曲线非负；
60 s 步长相邻变化有界（无尖刺）。
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import yaml

_DEFAULT_SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "scenario.yaml")

SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_MONTHS = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}

# 24 h 基准曲线（标幺，峰值 1.0 固定在设计峰值时刻）
RES_WEEKDAY = np.array([
    0.08, 0.06, 0.06, 0.06, 0.06, 0.08, 0.22, 0.50, 0.60, 0.35, 0.30, 0.28,
    0.25, 0.24, 0.28, 0.32, 0.38, 0.55, 0.80, 1.00, 0.90, 0.72, 0.40, 0.20,
])
RES_WEEKEND = np.array([
    0.10, 0.08, 0.08, 0.08, 0.08, 0.10, 0.30, 0.70, 0.85, 0.55, 0.45, 0.40,
    0.38, 0.35, 0.40, 0.45, 0.55, 0.70, 0.90, 1.00, 0.95, 0.85, 0.55, 0.30,
])
TOU_WEEKDAY = np.array([
    0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.12, 0.30, 0.50, 0.62, 0.68,
    0.72, 0.92, 1.00, 0.94, 0.72, 0.55, 0.75, 1.00, 0.88, 0.55, 0.20, 0.06,
])
TOU_WEEKEND = np.array([
    0.04, 0.03, 0.03, 0.03, 0.03, 0.04, 0.07, 0.18, 0.45, 0.65, 0.78, 0.85,
    0.88, 0.95, 1.00, 0.96, 0.85, 0.70, 0.85, 1.00, 0.92, 0.65, 0.28, 0.08,
])
HEAT_ANCHORS = np.array([
    0.45, 0.40, 0.38, 0.36, 0.35, 0.40, 0.55, 0.85, 1.00, 0.90, 0.75, 0.65,
    0.60, 0.58, 0.60, 0.65, 0.70, 0.78, 0.88, 0.95, 0.90, 0.80, 0.65, 0.52,
])

# 季节系数（tourism 满足 C1：夏 x1.6、冬 x0.6；residential 全年恒 1.0；
# heat 冬季略高、夏季略低，峰值仍在 60-80 kW 区间）
RES_SEASONAL = {"spring": 1.0, "summer": 1.0, "autumn": 1.0, "winter": 1.0}
TOU_SEASONAL = {"spring": 1.0, "summer": 1.6, "autumn": 1.0, "winter": 0.6}
HEAT_SEASONAL = {"spring": 1.0, "summer": 0.95, "autumn": 1.0, "winter": 1.05}


def _season_of_month(month: int) -> str:
    """月份 -> 季节（冬季含 12/1/2）。"""
    for s, ms in SEASON_MONTHS.items():
        if month in ms:
            return s
    return "winter"


class LoadProfile:
    """海岛微电网电/热负荷剖面合成器（阶段 C4）。"""

    def __init__(self, scenario_path: str = None, params: dict = None):
        """读取 configs/scenario.yaml（UTF-8）；params 可覆盖标定/季节系数。"""
        path = scenario_path or _DEFAULT_SCENARIO
        with io.open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        scen = cfg.get("scenario", {})
        load = cfg.get("load", {})
        seg = load.get("segments", {})
        self.scenario_path = os.path.abspath(path)
        self.dt_s_default = float(scen.get("time_step_s", 60.0))
        self.peak_load_kw = float(load.get("peak_load_kw", 200.0))
        self.res_ratio = float(seg.get("residential_ratio", 0.45))
        self.tou_ratio = float(seg.get("tourism_ratio", 0.35))
        self.desal_kw = float(seg.get("desalination_kw", 40.0))
        self.desalination_schedulable = bool(seg.get("desalination_schedulable", False))
        # 峰值构成：居民 90 + 旅游 70（夏季）+ 淡化 40 = 200 kW（同时）
        self.res_peak_kw = self.peak_load_kw * self.res_ratio
        self.tou_peak_kw = self.peak_load_kw * self.tou_ratio

        p = dict(params or {})
        self.seasonal = {
            "residential": dict(p.get("residential_seasonal", RES_SEASONAL)),
            "tourism": dict(p.get("tourism_seasonal", TOU_SEASONAL)),
            "heat": dict(p.get("heat_seasonal", HEAT_SEASONAL)),
        }
        # 旅游肩季参考峰值：夏季系数归一，使夏季峰值恰为 70 kW
        self.tou_ref_peak_kw = self.tou_peak_kw / max(self.seasonal["tourism"].values())
        self.heat_peak_kw = float(p.get("heat_peak_kw", 75.0))
        self.heat_anchors = np.asarray(p.get("heat_anchors", HEAT_ANCHORS), dtype=float)
        self.anchor_tables = {
            "res_weekday": np.asarray(p.get("res_weekday", RES_WEEKDAY), dtype=float),
            "res_weekend": np.asarray(p.get("res_weekend", RES_WEEKEND), dtype=float),
            "tou_weekday": np.asarray(p.get("tou_weekday", TOU_WEEKDAY), dtype=float),
            "tou_weekend": np.asarray(p.get("tou_weekend", TOU_WEEKEND), dtype=float),
        }
        for key, tbl in self.anchor_tables.items():
            if tbl.shape != (24,):
                raise ValueError(f"{key} 锚点需为 24 小时 1 维数组")
        self._xph = np.arange(25, dtype=float)   # 周期延拓锚点（小时）

    # ------------------------------------------------------------------
    def _pu_24h(self, anchors: np.ndarray, n_steps: int) -> np.ndarray:
        """小时锚点线性插值 -> n_steps 点 24 h 标幺曲线（周期首尾相连）。"""
        fp = np.concatenate([anchors, anchors[:1]])
        t_h = np.arange(n_steps) * (24.0 / n_steps)
        return np.interp(t_h, self._xph, fp)

    def _day_arrays(self, season: str, weekend: bool, n_steps: int):
        """单日 60 s 级数组：电/热/分项（kW）。"""
        if season not in SEASONS:
            raise ValueError(f"season 需为 {SEASONS} 之一，got {season!r}")
        res_key = "res_weekend" if weekend else "res_weekday"
        tou_key = "tou_weekend" if weekend else "tou_weekday"
        res_pu = self._pu_24h(self.anchor_tables[res_key], n_steps)
        tou_pu = self._pu_24h(self.anchor_tables[tou_key], n_steps)
        heat_pu = self._pu_24h(self.heat_anchors, n_steps)
        p_res = res_pu * self.res_peak_kw * self.seasonal["residential"][season]
        p_tou = tou_pu * self.tou_ref_peak_kw * self.seasonal["tourism"][season]
        p_desal = np.full(n_steps, self.desal_kw, dtype=float)
        p_el = p_res + p_tou + p_desal
        q_load = heat_pu * self.heat_peak_kw * self.seasonal["heat"][season]
        return p_el, q_load, p_res, p_tou, p_desal

    # ------------------------------------------------------------------
    def profile_24h(self, season: str = "summer", weekend: bool = False,
                    dt_s: float = 3600.0) -> pd.DataFrame:
        """单个 24 h 基准曲线（默认夏季工作日，1 h 步长）。"""
        n = int(round(86400.0 / dt_s))
        if n * dt_s != 86400.0:
            raise ValueError(f"dt_s={dt_s} 不能整除 86400 s")
        p_el, q, res, tou, desal = self._day_arrays(season, bool(weekend), n)
        idx = pd.date_range("2000-01-01", periods=n, freq=f"{int(dt_s)}s")
        return pd.DataFrame({"p_el_kw": p_el, "q_load_kw": q, "p_res_kw": res,
                             "p_tou_kw": tou, "p_desal_kw": desal}, index=idx)

    def generate_series(self, days: int = 365, dt_s: float = None,
                        season: str = None, weekend: bool = None,
                        start: str = "2026-01-01", time_index: bool = True):
        """合成 days 天序列。

        season=None 时按日历自动分配季节（start 起）；weekend=None 时按日期
        自动区分工作日/周末；time_index=True 返回带 60 s 时间索引的 DataFrame，
        False 返回 (p_el_kw 数组, q_load_kw 数组)。
        """
        dt_s = float(dt_s if dt_s is not None else self.dt_s_default)
        n_steps = int(round(86400.0 / dt_s))
        if n_steps * dt_s != 86400.0:
            raise ValueError(f"dt_s={dt_s} 不能整除 86400 s")
        dates = pd.date_range(start=start, periods=int(days), freq="D")
        el, q, res, tou, desal = [], [], [], [], []
        for d in dates:
            s = season if season is not None else _season_of_month(d.month)
            wk = bool(weekend) if weekend is not None else (d.weekday() >= 5)
            e, q_, r, t, dd = self._day_arrays(s, wk, n_steps)
            el.append(e)
            q.append(q_)
            res.append(r)
            tou.append(t)
            desal.append(dd)
        out = {
            "p_el_kw": np.concatenate(el),
            "q_load_kw": np.concatenate(q),
            "p_res_kw": np.concatenate(res),
            "p_tou_kw": np.concatenate(tou),
            "p_desal_kw": np.concatenate(desal),
        }
        if not time_index:
            return out["p_el_kw"], out["q_load_kw"]
        idx = pd.date_range(start=start, periods=len(out["p_el_kw"]),
                            freq=f"{int(dt_s)}s")
        return pd.DataFrame(out, index=idx)

    # ------------------------------------------------------------------
    def annual_energy_mwh(self, dt_s: float = None, days: int = 365) -> float:
        """全年电量（MWh）：默认 365 d @ 60 s 合成序列积分。"""
        dt_s = float(dt_s if dt_s is not None else self.dt_s_default)
        p_el = self.generate_series(days, dt_s=dt_s, time_index=False)[0]
        return float(np.sum(p_el) * dt_s / 3600.0 / 1000.0)

    def calibrate(self, dt_s: float = None, lo_mwh: float = 720.0,
                  hi_mwh: float = 880.0) -> dict:
        """年电量校核：超限时给出非淡化部分建议缩放因子并报告实际值。"""
        dt_s = float(dt_s if dt_s is not None else self.dt_s_default)
        e = self.annual_energy_mwh(dt_s)
        in_range = lo_mwh <= e <= hi_mwh
        desal_annual = self.desal_kw * 8760.0 / 1000.0
        var_annual = e - desal_annual
        if in_range:
            factor = 1.0
        else:
            target = float(np.clip(e, lo_mwh, hi_mwh))
            factor = (target - desal_annual) / var_annual if var_annual > 0 else 1.0
        return {
            "annual_energy_mwh": e,
            "desal_annual_mwh": desal_annual,
            "variable_annual_mwh": var_annual,
            "scale_factor": factor,
            "in_range": bool(in_range),
            "target_range_mwh": [lo_mwh, hi_mwh],
        }

    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        """负荷构成/淡化可平移接口元数据（供 EMS C3/D 期使用）。"""
        cal = self.calibrate()
        return {
            "source": "configs/scenario.yaml（C1 定稿）",
            "peak_load_kw": self.peak_load_kw,
            "segments_kw": {
                "residential": self.res_peak_kw,
                "tourism": self.tou_peak_kw,
                "desalination": self.desal_kw,
            },
            "ratios": {"residential": self.res_ratio, "tourism": self.tou_ratio},
            "desalination_schedulable": self.desalination_schedulable,
            "desalination_shift": {
                "kw": self.desal_kw,
                "shift_horizon_h": 24.0,
                "max_shift_kw": self.desal_kw,
                "note": "淡化负荷恒 40 kW，可平移调度（EMS 削峰填谷，C3/D 期）",
            },
            "seasons": list(self.seasonal["tourism"].keys()),
            "tourism_seasonal": dict(self.seasonal["tourism"]),
            "heat": {
                "peak_kw": self.heat_peak_kw * max(self.seasonal["heat"].values()),
                "mean_kw": float(np.mean(self.heat_anchors) * self.heat_peak_kw),
                "assumption": "工程假设（早晚双峰），后续可用论文游轮热负荷数据标定",
            },
            "calibration": cal,
        }

    # ------------------------------------------------------------------
    def desalination_series(self, days: int = 365, dt_s: float = None) -> np.ndarray:
        """恒定淡化序列（可平移基准）；实际调度由 EMS 用 apply_desalination_schedule 重塑。"""
        dt_s = float(dt_s if dt_s is not None else self.dt_s_default)
        n = int(days) * int(round(86400.0 / dt_s))
        return np.full(n, self.desal_kw, dtype=float)

    def apply_desalination_schedule(self, df: pd.DataFrame,
                                    desal_kw: np.ndarray) -> pd.DataFrame:
        """可平移接口：将淡化负荷替换为 EMS 给定序列，重算净电负荷。"""
        if not self.desalination_schedulable:
            raise ValueError("scenario 中 desalination_schedulable=false，淡化不可平移")
        d = np.asarray(desal_kw, dtype=float)
        if len(d) != len(df):
            raise ValueError("desal 序列长度与 df 不一致")
        out = df.copy()
        out["p_desal_kw"] = d
        out["p_el_kw"] = out["p_res_kw"] + out["p_tou_kw"] + d
        return out
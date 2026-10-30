# -*- coding: utf-8 -*-
"""可再生能源出力模块（阶段 C4 占位，E 期启用）。

C1 定稿：configs/scenario.yaml 中 renewables.include_pv=false、
include_wind=false，本模块当前一律返回全零序列（E 期启用占位）。
E 期启用后（将 include_pv/include_wind 置 true 并给出装机容量）：
- PVProfile：晴空余弦曲线 P = cap x av x max(0, cos(pi*(t-12)/12))，
  06:00-18:00 有出力、正午最大；av 为可用系数（云量/温度/灰尘损失）。
- WindProfile：Weibull 日曲线（k=2、c=7 m/s 默认，Rayleigh 型），
  以日均风速构造平滑日形状（午后峰、夜间低），功率按 v^3 截断在额定风速；
  E 期可替换为实测风速时序 + Weibull 随机采样。

两类的 generate_series 接口与 src/analysis/load_profile.py 一致：
time_index=True 返回带时间索引的 DataFrame，False 返回 numpy 数组。
"""
from __future__ import annotations

import io
import math
import os

import numpy as np
import pandas as pd
import yaml

_DEFAULT_SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "scenario.yaml")


def _load_scenario(scenario_path: str) -> dict:
    path = scenario_path or _DEFAULT_SCENARIO
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _freq(dt_s: float) -> str:
    return f"{int(dt_s)}s"


class PVProfile:
    """晴空余弦光伏出力（E 期启用占位；include_pv=false 时全零）。"""

    def __init__(self, capacity_kw: float = None, availability: float = 0.9,
                 scenario_path: str = None, include: bool = None):
        cfg = _load_scenario(scenario_path)
        ren = cfg.get("renewables", {})
        self.include = bool(include) if include is not None else bool(ren.get("include_pv", False))
        cap = capacity_kw if capacity_kw is not None else ren.get("pv_capacity_kw")
        self.capacity_kw = float(cap) if cap else 0.0
        self.availability = float(availability)

    def power(self, t_h_local: float) -> np.ndarray:
        """t_h_local：当地时间（0-24 h，可向量化）。include=false 时返回全零。"""
        t = np.asarray(t_h_local, dtype=float)
        if not self.include:
            return np.zeros_like(t)
        # 晴空余弦：6-18 时出力，正午 12 时最大；其余时段为零
        return self.capacity_kw * self.availability * np.maximum(
            0.0, np.cos(np.pi * (t - 12.0) / 12.0))

    def generate_series(self, days: int = 365, dt_s: float = 60.0,
                        start: str = "2026-01-01", time_index: bool = True):
        n = int(days) * int(round(86400.0 / dt_s))
        if self.include:
            t = (np.arange(n) * dt_s / 3600.0) % 24.0
            out = self.power(t)
        else:
            out = np.zeros(n, dtype=float)
        if not time_index:
            return out
        idx = pd.date_range(start=start, periods=n, freq=_freq(dt_s))
        return pd.DataFrame({"p_pv_kw": out}, index=idx)


class WindProfile:
    """Weibull 日曲线风电出力（E 期启用占位；include_wind=false 时全零）。"""

    def __init__(self, capacity_kw: float = None, weibull_k: float = 2.0,
                 weibull_c: float = 7.0, rated_speed: float = 12.0,
                 scenario_path: str = None, include: bool = None):
        cfg = _load_scenario(scenario_path)
        ren = cfg.get("renewables", {})
        self.include = bool(include) if include is not None else bool(ren.get("include_wind", False))
        cap = capacity_kw if capacity_kw is not None else ren.get("wind_capacity_kw")
        self.capacity_kw = float(cap) if cap else 0.0
        self.weibull_k = float(weibull_k)
        self.weibull_c = float(weibull_c)
        self.rated_speed = float(rated_speed)
        # 日均风速 = c * Gamma(1 + 1/k)（Weibull 均值）
        self.v_mean = self.weibull_c * math.gamma(1.0 + 1.0 / self.weibull_k)

    def power(self, t_h_local: float) -> np.ndarray:
        """t_h_local：当地时间（0-24 h，可向量化）。include=false 时返回全零。"""
        t = np.asarray(t_h_local, dtype=float)
        if not self.include:
            return np.zeros_like(t)
        # 平滑日形状：午后峰（~14 h）、夜间低；日均风速保持 Weibull 均值
        v = self.v_mean * (1.0 + 0.55 * np.sin(np.pi * (t - 5.0) / 14.0))
        v = np.maximum(v, 0.0)
        pu = np.clip((v / self.rated_speed) ** 3, 0.0, 1.0)
        return self.capacity_kw * pu

    def generate_series(self, days: int = 365, dt_s: float = 60.0,
                        start: str = "2026-01-01", time_index: bool = True):
        n = int(days) * int(round(86400.0 / dt_s))
        if self.include:
            t = (np.arange(n) * dt_s / 3600.0) % 24.0
            out = self.power(t)
        else:
            out = np.zeros(n, dtype=float)
        if not time_index:
            return out
        idx = pd.date_range(start=start, periods=n, freq=_freq(dt_s))
        return pd.DataFrame({"p_wind_kw": out}, index=idx)
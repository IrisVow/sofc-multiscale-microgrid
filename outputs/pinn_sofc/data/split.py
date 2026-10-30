# -*- coding: utf-8 -*-
"""工况切分与域定义（M1 数据管线）"""
from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# 域定义（与 DOE 数据一致，见 pinndata/doe/README.md）
# ---------------------------------------------------------------------------
DOMAIN_ORDER = [
    "air_channel",      # y 0.00-0.80 mm，空气流道
    "air_electrode",    # y 0.80-1.30 mm，空气侧电极（阴极, cd.iloc_pcar1）
    "electrolyte",      # y 1.30-1.32 mm，YSZ 致密电解质
    "fuel_electrode",   # y 1.32-1.82 mm，燃料侧电极（阳极, cd.iloc_panr1）
    "fuel_channel",     # y 1.82-2.62 mm，燃料流道
]
DOMAIN_ID = {d: i for i, d in enumerate(DOMAIN_ORDER)}

# 几何边界（米）：用于域判定与后续 PDE 界面
GEOM = {
    "x_min": 0.0, "x_max": 0.1,
    "y_air_ch": (0.0, 0.8e-3),
    "y_air_el": (0.8e-3, 1.30e-3),
    "y_ely": (1.30e-3, 1.32e-3),
    "y_fuel_el": (1.32e-3, 1.82e-3),
    "y_fuel_ch": (1.82e-3, 2.62e-3),
}

# 输出变量
PRIMARY_VARS = ["T", "u", "v", "p", "w1", "w2", "w3", "phis", "phil"]
AUX_VARS = ["htQ", "qav_an", "qav_ca"]
RATE_VARS = ["rH2", "rH2O", "rO2"]          # 阶段2+ 可选
CONT_VARS = ["x", "y", "t_s", "V_applied", "T_in_K", "lambda_air", "FU"]

# 工况分类
def classify_case(case_id: str) -> str:
    """steady / transient / unknown"""
    if case_id.startswith("S"):
        return "steady"
    if case_id.startswith("T"):
        return "transient"
    return "unknown"


def make_splits(cfg: dict) -> dict:
    """从 config 生成切分字典。"""
    d = cfg["data"]
    return {
        "steady_train": list(d["steady_train"]),
        "steady_val": list(d["steady_val"]),
        "steady_test": list(d["steady_test"]),
        "transient_train": list(d["transient_train"]),
        "transient_test": list(d["transient_test"]),
    }
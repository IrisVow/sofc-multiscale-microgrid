# -*- coding: utf-8 -*-
"""C2 组件参数库：论文（Energy Conversion & Management 326 (2025) 119477）
Table 2/3 参数 + 化学常量 + 效率/排放查找表参考点。

单位约定：功率 kW、能量 kWh、质量 kg、热值按 LHV。
所有默认值可被 configs/components.yaml 或构造时 dict 覆盖。
"""
from __future__ import annotations

import copy

# ---------------------------------------------------------------------------
# 化学与燃料常量（LHV 基准）
# ---------------------------------------------------------------------------
CHEM = {
    "LHV_MGO_kJ_kg": 42800.0,          # Table 3
    "LHV_LNG_kJ_kg": 50000.0,          # Table 3（取 CH4 LHV 近似）
    "LHV_H2_kJ_kg": 119960.0,          # H2 LHV
    "LHV_CH4_kJ_kg": 50000.0,          # CH4 LHV（与 LNG 一致）
    "M_CH4_kg_kmol": 16.043,
    "M_H2_kg_kmol": 2.016,
    "M_H2O_kg_kmol": 18.015,
    "M_CO2_kg_kmol": 44.010,
    "MGO_CO2_kg_per_kg": 3.206,        # 燃料完全燃烧 CO2（IPCC 口径）
    "CH4_CO2_kg_per_kg": 2.750,        # CH4 -> CO2 化学计量
    "GWP_CH4": 25.0,                   # Table 3: CO2-eq CH4 = 25
}

# ---------------------------------------------------------------------------
# 论文 Table 3：仿真模型参数
# ---------------------------------------------------------------------------
PARAMS = {
    "general": {
        "safety_margin_fuel": 0.10,    # S_fuel（燃料容量裕度，C5 用）
    },
    "sofc": {
        "p_nom_kw": 150.0,             # 名义电功率（海岛场景定稿，scenario.yaml）
        "v0_ocv": 1.05,                # 开路电压（Eq.10 V0；校准参数）
        "v_cell_bol": 0.85,            # Table 3：BOL 名义电压
        "v_cell_eol": 0.71,            # Table 3：EOL 名义电压
        "ramp_rise": 0.02,             # Table 3：R = 2% P/min（升载）
        "ramp_fall": 0.08,             # Table 3：F = 8% P/min（降载）
        "t_transient": 0.0003,         # Table 3：0.03%/min 瞬态阈值
        "a_trans": 1.5,                # Table 3：瞬态降解因子
        "lifetime_h": 30000.0,         # Table 3：名义堆寿命
        "i_nom_A_cm2": 0.18,           # 名义电流密度（Eq.11 用；标定 0.18 A/cm2，论文范围 0.1-0.23）
        "eta_reform": 0.80,            # LNG 重整器 LHV 效率（C2 默认，待标定）
        "sc_ratio": 2.0,               # 重整水碳比 S/C
        "ch4_slip": 0.005,             # SOFC 重整 CH4 滑移（默认 0.5%）
        # 净电效率 BOL（DC，Fig.3 结构；峰值在部分负荷，50-60% 区间）
        "eta_net_bol": [[0.10, 0.42], [0.25, 0.52], [0.40, 0.56],
                        [0.60, 0.57], [0.80, 0.56], [1.00, 0.54]],
        # 热效率（Fig.3：寿命内恒定；部分负荷 U_f 降低 -> 燃烧器补热 -> 热效率上升）
        "eta_heat": [[0.10, 0.45], [0.25, 0.40], [0.50, 0.33],
                     [0.75, 0.28], [1.00, 0.25]],
        # 燃料利用率 U_f（部分负荷降低，为燃烧器供热）
        "u_f_load": [[0.10, 0.55], [0.25, 0.60], [0.50, 0.65],
                     [0.75, 0.70], [1.00, 0.72]],
        "eta_inv": 0.96,               # Table 3：逆变器 AC|DC
        "eta_dcdc": 0.98,              # Table 3：DC|DC
    },
    "genset": {
        "min_load": 0.10,              # Table 3：最低负荷 10%
        "eta_gen": 0.95,               # Table 3：发电机效率（查找表已含）
        "ramp_mgo_pct_per_s": 2.7,     # Table 3：MGO 瞬态能力 %P/s
        "ramp_lng_pct_per_s": 1.4,     # Table 3：LNG 瞬态能力 %P/s
        "pilot_fuel_frac": 0.03,       # Table 3：DFG 引燃油 3%
        # 负荷-电效率查找表（已含 η_gen=0.95；典型中速机/贫燃燃气机）
        "eta_eff_mgo": [[0.10, 0.22], [0.25, 0.28], [0.50, 0.34],
                        [0.75, 0.38], [1.00, 0.42]],
        "eta_eff_lng": [[0.10, 0.20], [0.25, 0.28], [0.50, 0.35],
                        [0.75, 0.40], [1.00, 0.44]],
        # 排放因子（g/kWh，随负荷，含 η_gen；SOx 按 0.5%S MGO）
        "nox_g_kwh_mgo": [[0.10, 13.0], [0.50, 11.5], [1.00, 10.0]],
        "nox_g_kwh_lng": [[0.10, 2.2], [0.50, 2.0], [1.00, 1.8]],
        "sox_g_kwh_mgo": [[0.10, 0.9], [0.50, 0.85], [1.00, 0.8]],
        "sox_g_kwh_lng": [[0.10, 0.0], [0.50, 0.0], [1.00, 0.0]],
        "pm_g_kwh_mgo": [[0.10, 0.6], [0.50, 0.5], [1.00, 0.45]],
        "pm_g_kwh_lng": [[0.10, 0.03], [0.50, 0.02], [1.00, 0.02]],
        "ch4_slip": 0.035,             # LNG 贫燃机 CH4 滑移（Comer 建议 6% 作政策默认，3.5% 代表性）
        # 余热回收效率（回收热/燃料 LHV，随负载；用于 C3 热平衡）
        "eta_heat_mgo": [[0.10, 0.28], [0.50, 0.34], [1.00, 0.36]],
        "eta_heat_lng": [[0.10, 0.28], [0.50, 0.35], [1.00, 0.38]],
    },
    "battery": {
        "e_nom_kwh": 300.0,            # 名义容量（scenario.yaml 扫描范围中点 0.3 MWh）
        "eta_ch": 0.90,                # Table 3：充电效率
        "eta_dis": 0.95,               # Table 3：放电效率
        "c_rate_max": 3.0,             # Table 3：最大 C-rate
        "soc_init": 0.55,              # Table 3：初始 SOC
        "soc_ub": 0.90,                # Table 3：SOC 上界
        "soc_lb": 0.20,                # Table 3：SOC 下界
        "safety_margin": 0.10,         # Table 3：电池安全裕度
    },
    "boiler": {
        "eta_mgo": 0.85,               # Table 3：MGO 锅炉效率
        "eta_lng": 0.90,               # Table 3：LNG 锅炉效率
        "hex_eta": 0.85,               # 换热器效率（中文项目文档 Eq.5：Q_desal <= eta_hex * Q_exhaust）
    },
}

# ---------------------------------------------------------------------------
# 论文 Table 2：尺寸/能量密度（C5 尺寸模型用；此处先行登记）
# ---------------------------------------------------------------------------
DENSITIES = {
    "sofc": {"p_vol_kW_m3": 15.0, "p_grav_kW_ton": 25.0},          # LNG
    "dg": {"p_vol_kW_m3": 45.0, "p_grav_kW_ton": 60.0},            # MGO
    "dfg": {"p_vol_kW_m3": 40.0, "p_grav_kW_ton": 50.0},           # both
    "gg": {"p_vol_kW_m3": 40.0, "p_grav_kW_ton": 50.0},            # LNG
    "boiler_mgo": {"p_vol_kW_m3": 104.0, "p_grav_kW_ton": 360.0},
    "boiler_lng": {"p_vol_kW_m3": 75.0, "p_grav_kW_ton": 310.0},
    "bat": {"e_vol_kWh_m3": 90.0, "e_grav_kWh_ton": 80.0},         # Li-ion
    "fuel_mgo": {"e_vol_kWh_m3": 8200.0, "e_grav_kWh_ton": 8300.0},
    "fuel_lng": {"e_vol_kWh_m3": 3443.0, "e_grav_kWh_ton": 8050.0},
}


def merge_params(base: dict, override: dict) -> dict:
    """深度合并默认参数（override 可来自 YAML/构造 dict）。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_params(out[k], v)
        else:
            out[k] = v
    return out


def load_params(path: str = None) -> dict:
    """从 components.yaml 覆盖默认参数；path=None 时仅返回内置默认。"""
    params = copy.deepcopy(PARAMS)
    if path:
        import io
        import yaml
        with io.open(path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        params = merge_params(params, override)
    return params
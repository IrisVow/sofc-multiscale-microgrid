# -*- coding: utf-8 -*-
"""模型参数（取自 sofc1_FINAL2_solved.mph，见 pinndata/doe/README.md §3.3）"""
from __future__ import annotations

# 物理常数
F_CONST = 96485.33212       # C/mol
R_CONST = 8.314462618       # J/(mol·K)
P_REF = 101325.0            # Pa (1 atm)

# 摩尔质量 kg/mol
M_H2 = 0.002016
M_H2O = 0.018015
M_O2 = 0.031998
M_N2 = 0.028013

# 几何
L = 0.1                     # m 流道长度
Y_LAYERS = {                # m
    "air_channel": (0.0, 0.8e-3),
    "air_electrode": (0.8e-3, 1.30e-3),
    "electrolyte": (1.30e-3, 1.32e-3),
    "fuel_electrode": (1.32e-3, 1.82e-3),
    "fuel_channel": (1.82e-3, 2.62e-3),
}

# 多孔电极
EPS_P = 0.35                # 孔隙率
TAU_P = 3.0                 # 迂曲度
KAPPA_P = 1e-12             # m^2 渗透率
AV_AN = 1e5                 # 1/m 阳极比表面积
AV_CA = 1e5                 # 1/m 阴极比表面积

# 电化学
I0_AN = 500.0               # A/m^2 阳极交换电流密度 (i0_an=5e3*k_i0, k_i0=0.1)
I0_CA = 50.0                # A/m^2 阴极交换电流密度
ALPHA_A = 0.5               # 阳极传递系数（待标定）
ALPHA_C = 0.5               # 阴极传递系数（待标定）
EEQ_AN = 0.0                # V 阳极平衡电位（数据）
EEQ_CA = 1.0                # V 阴极平衡电位（数据）
DEEQDT_CA = -2.3e-4         # V/K

# 电导率 S/m
SIGMA_S_AN = 3.5e4
SIGMA_S_CA = 1e4
SIGMA_L_YSZ = 3.34e4

# 气体/固体物性
MU_G = 3e-5                 # Pa·s
K_G = 0.1                   # W/(m·K)
CP_G = 1500.0               # J/(kg·K)
K_S = 2.0                   # W/(m·K)
CP_S = 500.0                # J/(kg·K)
RHO_S = 5000.0              # kg/m^3
D_EFF_GAS = 1e-4            # m^2/s 有效扩散系数

# 二元扩散系数 m^2/s (Maxwell-Stefan, 阶段3 可选)
D_BINARY = {"D12": 9e-4, "D13": 8e-4, "D14": 8e-4,
            "D23": 3e-4, "D24": 3e-4, "D34": 2e-4}

# 散热
H_CONV = 5.0                # W/(m^2·K)
T_AMB = 293.15              # K
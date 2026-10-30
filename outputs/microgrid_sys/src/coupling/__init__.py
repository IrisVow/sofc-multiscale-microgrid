# -*- coding: utf-8 -*-
"""双层耦合包（阶段 D1–D5）：domain_guard（B5）+ surrogate（ORT）+ LUT + PinnCoupling 监视器 + PinnInference 公共 API。"""
from .domain_guard import DOMAIN_ORDER, OPS_BOUNDS, COL_NAMES, check, check_dict
from .lut import PinnLUT
from .pinn_coupling import PinnCoupling, load_coupling_params
from .pinn_inference import PinnInference, evaluate_field, evaluate_table
from .surrogate import PinnSurrogate

__all__ = ["DOMAIN_ORDER", "OPS_BOUNDS", "COL_NAMES", "check", "check_dict",
           "PinnSurrogate", "PinnLUT", "PinnCoupling", "load_coupling_params",
    "PinnInference", "evaluate_field", "evaluate_table"]
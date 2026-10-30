# -*- coding: utf-8 -*-
"""域/界面几何工具（阶段3 PDE 配点与界面配对使用）"""
from __future__ import annotations

import numpy as np

from .params import Y_LAYERS

DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]
DOMAIN_ID = {d: i for i, d in enumerate(DOMAIN_ORDER)}

# 相邻域界面（y 方向层间界面，x 方向全覆盖）
INTERFACES = [
    ("air_channel", "air_electrode", 0.8e-3),
    ("air_electrode", "electrolyte", 1.30e-3),
    ("electrolyte", "fuel_electrode", 1.32e-3),
    ("fuel_electrode", "fuel_channel", 1.82e-3),
]


def sample_collocation(n_per_domain, rng: np.random.Generator, x_range=(0.0, 0.1)):
    """Sample random collocation points per domain.

    n_per_domain may be an int (same count everywhere) or a dict
    {domain: count} to over-sample thin/source-active layers
    (electrolyte, electrodes for charge conservation).
    Returns dict[domain, (N,2) x,y].
    """
    out = {}
    for d in DOMAIN_ORDER:
        y0, y1 = Y_LAYERS[d]
        n = n_per_domain[d] if isinstance(n_per_domain, dict) else n_per_domain
        x = rng.uniform(x_range[0], x_range[1], n)
        y = rng.uniform(y0, y1, n)
        out[d] = np.stack([x, y], axis=1)
    return out


def sample_interfaces(n_per_iface: int, rng: np.random.Generator):
    """界面配点：返回 list of (dom_a, dom_b, (N,2) x,y)。"""
    out = []
    for a, b, yc in INTERFACES:
        x = rng.uniform(0.0, 0.1, n_per_iface)
        pts = np.stack([x, np.full(n_per_iface, yc)], axis=1)
        out.append((a, b, pts))
    return out
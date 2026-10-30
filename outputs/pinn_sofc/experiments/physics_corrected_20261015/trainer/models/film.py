# -*- coding: utf-8 -*-
"""FiLM 条件块：domain 条件注入共享主干（阶段2+ 使用；M1 基线用残差块）"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """h = act(γ_dom ⊙ (W h) + β_dom)；γ/β 由 domain 特征经小 MLP 生成。"""

    def __init__(self, d_in: int, d_cond: int, act=nn.SiLU()):
        super().__init__()
        self.fc = nn.Linear(d_in, d_in)
        self.cond = nn.Sequential(nn.Linear(d_cond, d_in), nn.SiLU(), nn.Linear(d_in, 2 * d_in))
        self.act = act

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = self.fc(h)
        gamma, beta = self.cond(c).chunk(2, dim=-1)
        return self.act(gamma * z + beta)
# -*- coding: utf-8 -*-
"""基线 MLP（M1）：输入 = 连续条件(7) + Fourier(x,y,t) + domain one-hot(5) -> 输出全部变量"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fourier import FourierFeatures


class ResidualBlock(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(x))))


class BaselineMLP(nn.Module):
    """纯数据基线：残差 MLP + 可选 Fourier 特征 + domain 拼接。

    输入张量 X：(B, n_cont + n_domain)，其中前 n_spatial(=3) 列为归一化 (x,y,t)，
    随后为其余连续条件，最后 n_domain 列为域 one-hot。
    """

    def __init__(self, n_cont: int = 7, n_domain: int = 5, n_out: int = 12,
                 hidden: int = 256, layers: int = 6, dropout: float = 0.05,
                 fourier: bool = True, n_freq: int = 10,
                 f_min: float = 1.0, f_max: float = 512.0,
                 fourier_cols: list = None):
        super().__init__()
        self.n_cont = n_cont
        self.n_domain = n_domain
        self.fourier = fourier
        self.n_spatial = 3  # x, y, t
        if fourier_cols is None:
            fourier_cols = [0, 1, 2]
        self.fourier_cols = list(fourier_cols)
        n_fourier = len(self.fourier_cols)
        if fourier:
            self.ff = FourierFeatures(n_spatial=n_fourier, n_freq=n_freq,
                                      f_min=f_min, f_max=f_max)
            in_dim = 2 * n_fourier * n_freq + n_cont + n_domain
        else:
            in_dim = n_cont + n_domain
        self.in_layer = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(layers)])
        self.head = nn.Linear(hidden, n_out)
        self.act = nn.SiLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        cont = X[:, :self.n_cont]
        dom = X[:, self.n_cont:]
        feats = []
        if self.fourier:
            feats.append(self.ff(cont[:, self.fourier_cols]))
        feats.append(cont)
        feats.append(dom)
        z = torch.cat(feats, dim=-1)
        h = self.act(self.in_layer(z))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)
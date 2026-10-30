# -*- coding: utf-8 -*-
"""Fourier 特征层：对归一化后的 (x, y, t) 做对数等距频率编码"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """gamma(s) = [sin(2*pi*f_k*s), cos(2*pi*f_k*s)] with log-spaced f_k.

    Supports per-coordinate frequency ranges: f_min/f_max may be floats
    (shared across coordinates) or sequences of length n_spatial.
    Input s is the normalized (x, y, t) coordinates (B, n_spatial).
    Output dim = n_spatial * 2 * n_freq.
    """

    def __init__(self, n_spatial: int = 3, n_freq: int = 10,
                 f_min: float = 1.0, f_max: float = 512.0):
        super().__init__()
        if isinstance(f_min, (list, tuple)):
            assert len(f_min) == n_spatial
            fmin = torch.tensor([float(v) for v in f_min], dtype=torch.float32)
        else:
            fmin = torch.full((n_spatial,), float(f_min), dtype=torch.float32)
        if isinstance(f_max, (list, tuple)):
            assert len(f_max) == n_spatial
            fmax = torch.tensor([float(v) for v in f_max], dtype=torch.float32)
        else:
            fmax = torch.full((n_spatial,), float(f_max), dtype=torch.float32)
        # (n_freq, n_spatial): per-coordinate log-spaced frequencies
        freqs = torch.stack([
            torch.logspace(math.log10(fmin[i].item()), math.log10(fmax[i].item()), n_freq)
            for i in range(n_spatial)
        ], dim=1)
        self.register_buffer("freqs", freqs.contiguous())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, n_spatial) -> (B, n_spatial * 2 * n_freq)
        ph = 2 * math.pi * self.freqs[None] * s[:, None, :]      # (B, n_freq, n_spatial)
        out = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)  # (B, n_freq, 2*n_spatial)
        return out.reshape(s.shape[0], -1)


    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, n_spatial) -> (B, n_spatial * 2 * n_freq)
        ph = 2 * math.pi * self.freqs[None] * s[:, None, :]      # (B, n_freq, n_spatial)
        out = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)  # (B, n_freq, 2*n_spatial)
        return out.reshape(s.shape[0], -1)
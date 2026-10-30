# -*- coding: utf-8 -*-
"""v19 waveform parameterization: analytic applied-voltage waveforms.

The PINN input switches from the instantaneous `V_applied(t)` (v18) to
end-to-end waveform parameters (V_cell, amp, stepT, wave_type). This module
is the single source of truth for the analytic waveforms that drive the
COMSOL DOE transient cases, used by:
  - data/dataset.py   : fill per-case (amp, stepT, wave_type) columns
  - physics/pde.py    : transient collector BC phis = v_wave(t; params)

Waveforms (must match scripts/doe_cont_clean.py CONT_META + v_applied):
  step      Vc - amp * H(t - stepT)                     (T0-T9)
  ramp      Vc - amp * min(t/600, 1)                    (R1)
  multistep Vc - amp * w(t), breaks=[200,400,600,800,1000],
             levels=[0,.2,.6,1,.4,.8]                  (R2)
  sine      Vc - amp * (0.5 + 0.5*sin(2*pi*t/600))      (R3)
  random    Vc - amp * w(t), breaks=[150,300,500,700,900,1100],
             levels=[.6,.2,1,.4,.8,.6]                 (R4)
"""
from __future__ import annotations

import torch

WAVE_TYPES = ["step", "ramp", "multistep", "sine", "random"]
WAVE_CODE = {w: i for i, w in enumerate(WAVE_TYPES)}

# continuous-wave case metadata (amp/type; T-cases carry stepT_s/stepAmp_V in data)
R_CASE_META = {
    "R1": {"amp": 0.20, "wave_type": "ramp",      "char_t": 600.0},
    "R2": {"amp": 0.25, "wave_type": "multistep", "char_t": 200.0},
    "R3": {"amp": 0.20, "wave_type": "sine",      "char_t": 600.0},
    "R4": {"amp": 0.25, "wave_type": "random",    "char_t": 150.0},
}

_MULTI_BREAKS = [200.0, 400.0, 600.0, 800.0, 1000.0]
_MULTI_LEVELS = [0.0, 0.2, 0.6, 1.0, 0.4, 0.8]
_RAND_BREAKS = [150.0, 300.0, 500.0, 700.0, 900.0, 1100.0]
_RAND_LEVELS = [0.6, 0.2, 1.0, 0.4, 0.8, 0.6]


def _piecewise(t: torch.Tensor, breaks, levels, base: float = 0.0) -> torch.Tensor:
    w = torch.full_like(t, float(base))
    for b, lv in zip(breaks, levels):
        w = torch.where(t >= b, torch.full_like(t, float(lv)), w)
    return w


def v_wave(t, V_cell: float, amp: float, wave_type, stepT: float = 0.0):
    """Applied voltage at time t for a waveform parameterization.

    t: float or tensor; returns float32 tensor with the same shape.
    wave_type: string name or int code (WAVE_CODE).
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    if isinstance(wave_type, str):
        wave_type = WAVE_CODE[wave_type]
    wave_type = int(wave_type)
    Vc = float(V_cell)
    a = float(amp)
    if wave_type == 0:                       # step
        return Vc - a * (t >= float(stepT)).to(t.dtype)
    if wave_type == 1:                       # ramp
        return Vc - a * torch.clamp(t / 600.0, 0.0, 1.0)
    if wave_type == 2:                       # multistep
        return Vc - a * _piecewise(t, _MULTI_BREAKS, _MULTI_LEVELS[1:], base=_MULTI_LEVELS[0])
    if wave_type == 3:                       # sine
        return Vc - a * (0.5 + 0.5 * torch.sin(2.0 * 3.141592653589793 * t / 600.0))
    if wave_type == 4:                       # random
        return Vc - a * _piecewise(t, _RAND_BREAKS, _RAND_LEVELS)
    raise ValueError(f"unknown wave_type: {wave_type}")

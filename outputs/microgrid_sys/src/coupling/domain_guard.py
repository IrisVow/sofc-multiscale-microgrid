# -*- coding: utf-8 -*-
"""B5 域外检测器：PINN 输入域检查（阶段 B，2026-08-13）。

PINN（m4b）的 DOE 覆盖域（与 pinndata/README.md 一致）：
    x(m)      [0, 0.1]
    y(m)      [0, 2.62e-3]
    t(s)      [0, 1200]（稳态 t=0）
    V_applied [0.75, 1.0] V
    T_in      [1073, 1123] K
    lambda_air [4, 8]
    FU        [0.6, 0.8]

用途：系统级仿真在调用 PINN 前先做域检查，超界输入必须告警，
避免在 PINN 盲区静默推理（PLANNING.md B5 / 风险表）。
"""
from __future__ import annotations

import numpy as np

# 列序与 export_onnx.py 的 x_phys 一致：(B,7) = x, y, t, V, T_in, lambda_air, FU
COL_NAMES = ["x", "y", "t", "V_applied", "T_in", "lambda_air", "FU"]
OPS_BOUNDS = np.array([
    [0.0, 0.1],        # x
    [0.0, 2.62e-3],    # y
    [0.0, 1200.0],     # t
    [0.75, 1.0],       # V_applied
    [1073.0, 1123.0],  # T_in
    [4.0, 8.0],        # lambda_air
    [0.6, 0.8],        # FU
])

# v19（waveform 10 维，export_onnx.py 列序）：x,y,t,V,amp,stepT,wave_type,T_in,lambda_air,FU
COL_NAMES_V19 = ["x", "y", "t", "V_applied", "amp", "stepT", "wave_type",
                 "T_in", "lambda_air", "FU"]
OPS_BOUNDS_V19 = np.array([
    [0.0, 0.1],        # x
    [0.0, 2.62e-3],    # y
    [0.0, 1200.0],     # t
    [0.75, 1.0],       # V_applied
    [0.0, 0.25],       # amp（波形幅值，v19）
    [0.0, 200.0],      # stepT（阶跃时长，v19）
    [0.0, 4.0],        # wave_type（0=稳态，v19）
    [1073.0, 1123.0],  # T_in
    [4.0, 8.0],        # lambda_air
    [0.6, 0.8],        # FU
])

# 与 export_onnx.py 相同的域索引：0 air_channel, 1 air_electrode, 2 electrolyte,
# 3 fuel_electrode, 4 fuel_channel
DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]


def check(ops: np.ndarray, margin_frac: float = 0.0) -> tuple[np.ndarray, dict]:
    """检查操作点是否在 PINN 输入域内。

    参数
    ----
    ops : (N, 7) float 数组，列序见 COL_NAMES
    margin_frac : 边界缓冲比例（0.01 = 距边界 1% 内视为"临界"，计入 warning）

    返回
    ----
    (in_domain, info)
      in_domain : (N,) bool，True = 严格在域内（含 margin 之外）
      info : dict，含 'warnings'（临界点索引）、'violations'（越界点索引与明细）
    """
    ops = np.asarray(ops, dtype=np.float64)
    if ops.ndim == 1:
        ops = ops[None, :]
    n_cols = ops.shape[1]
    if n_cols == len(COL_NAMES):
        names, bounds = COL_NAMES, OPS_BOUNDS
    elif n_cols == len(COL_NAMES_V19):
        names, bounds = COL_NAMES_V19, OPS_BOUNDS_V19
    else:
        raise ValueError(f"ops 须为 (N,{len(COL_NAMES)}) 或 (N,{len(COL_NAMES_V19)})，实际 {ops.shape}")

    lo = bounds[:, 0]
    hi = bounds[:, 1]
    margin = margin_frac * (hi - lo)

    below = ops < lo[None, :]
    above = ops > hi[None, :]
    viol_mask = below | above
    in_domain = ~viol_mask.any(axis=1)

    near_lo = (~below) & (ops < (lo + margin)[None, :])
    near_hi = (~above) & (ops > (hi - margin)[None, :])
    warn_mask = (near_lo | near_hi).any(axis=1) & in_domain

    info = {"warnings": np.where(warn_mask)[0], "violations": np.where(~in_domain)[0]}
    if viol_mask.any():
        detail = []
        idx = np.where(viol_mask.any(axis=1))[0]
        for i in idx:
            cols = np.where(viol_mask[i])[0]
            for c in cols:
                detail.append({
                    "row": int(i),
                    "var": names[c],
                    "value": float(ops[i, c]),
                    "range": [float(lo[c]), float(hi[c])],
                })
        info["violation_detail"] = detail
    return in_domain, info


def check_dict(op: dict, margin_frac: float = 0.0) -> tuple[bool, dict]:
    """单点字典形式检查（便于阅读）：op = {"V_applied": 0.9, "T_in": 1073, ...}。"""
    row = [op.get(k, 0.0) for k in COL_NAMES]
    mask, info = check(np.array([row]), margin_frac=margin_frac)
    return bool(mask[0]), info


if __name__ == "__main__":
    # 自检
    ok, inf = check(np.array([[0.05, 1.5e-3, 0.0, 0.9, 1073.0, 6.0, 0.7]]))
    assert ok and len(inf["violations"]) == 0, "域内点应通过"
    bad, inf2 = check(np.array([[0.05, 1.5e-3, 0.0, 0.6, 1073.0, 6.0, 0.7]]))
    assert not bad and inf2["violation_detail"][0]["var"] == "V_applied", "越界应告警"
    near, inf3 = check(np.array([[0.05, 1.5e-3, 0.0, 0.755, 1073.0, 6.0, 0.7]]), margin_frac=0.01)
    assert near and len(inf3["warnings"]) == 1, "临界应告警"
    print("[domain_guard] self-check PASS")
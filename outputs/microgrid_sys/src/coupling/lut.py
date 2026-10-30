# -*- coding: utf-8 -*-
"""PINN 查找表（阶段 D2）：离线枚举 V×T_in×λ_air×FU 的场汇总 + 4D 线性插值查询。

生成：scripts/build_pinn_table.py（v19 ONNX，探针场汇总）。
表项（每个工况点）：
    tmax_k      温度代理量（K），具体取样口径由 meta.temperature_metric 标识。
                原 v18/v19/v20 表为 x=50 mm 的五探针最大值，不能称为全场热点。
                新候选表区分沿流道探针最大值和训练网格点最大值；均非连续场极值保证。
    qav_an_max  阳极体积热源最大值（诊断量，与局部电流密度相关）
    qav_ca_max  阴极体积热源最大值
    phi_drop_v  电解质中面 phis-phil 差（局部电势降代理）
查询：lookup(v, t_in, lam, fu) 默认 4D 多线性插值（clamp 到网格边界、不外推），
      域外点等价于边界插值，比最近邻平滑；method="nearest" 保留原最近邻（对照/回退）。
实现：向量化——16 角点掩码 + 四场堆叠后一次张量点积（单次 ~20 µs）。
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np


class PinnLUT:
    """npz 查找表：axes{v_axis,t_in_axis,lam_axis,fu_axis} + data{...} + meta。"""

    AXES = ("v_axis", "t_in_axis", "lam_axis", "fu_axis")
    FIELDS = ("tmax_k", "qav_an_max", "qav_ca_max", "phi_drop_v")

    def __init__(self, npz_path):
        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=True) as z:
            self.axes = {k: np.asarray(z[k], dtype=float) for k in self.AXES}
            self.data = {k: np.asarray(z[k], dtype=float) for k in self.FIELDS}
            self.meta = dict(z["meta"].item()) if "meta" in z else {}
        self.shape = tuple(len(v) for v in self.axes.values())
        # Explicit semantics, including historical tables without this metadata.
        # These attributes do not change interpolation or the legacy lookup return schema.
        self.temperature_metric = str(self.meta.get("temperature_metric", "legacy_five_probe_max"))
        self.temperature_sampling = str(self.meta.get("temperature_sampling", "legacy_5_probes"))
        self.quasisteady_only = bool(self.meta.get("quasisteady_only", True))
        # D2：16 角点 0/1 掩码 + 四场堆叠（向量化多线性插值）
        self._corners = np.array(list(product((0, 1), repeat=4)), dtype=np.int64)
        self._stack = np.stack([self.data[f] for f in self.FIELDS]).reshape(len(self.FIELDS), -1)

    def _idx(self, name: str, x: float) -> int:
        return int(np.argmin(np.abs(self.axes[name] - float(x))))

    def _bracket(self, name: str, x: float) -> tuple:
        """返回 (lo, hi, w)：x 在 axis 内的插值包围格点与权重（clamp，不外推）。

        x <= axis[0] -> (0, 0, 0.0)；x >= axis[-1] -> (n-1, n-1, 0.0)；
        内部 -> (k, k+1, w)，w = (x-axis[k]) / (axis[k+1]-axis[k])。
        """
        axis = self.axes[name]
        n = len(axis)
        x = float(x)
        if x <= axis[0]:
            return 0, 0, 0.0
        if x >= axis[-1]:
            return n - 1, n - 1, 0.0
        hi = int(np.searchsorted(axis, x, side="right"))
        hi = min(max(hi, 1), n - 1)
        lo = hi - 1
        w = (x - axis[lo]) / (axis[hi] - axis[lo])
        return lo, hi, w

    def lookup(self, v, t_in, lam, fu, method: str = "linear") -> dict:
        """4D 网格查询，返回 {field: float} + idx + method。

        method="linear"（默认）：4D 多线性插值，clamp 到网格边界（不外推）；
        method="nearest"：原最近邻（对照/回退）。
        idx：linear 时为插值下角格点；nearest 时为最近格点。
        """
        if method == "nearest":
            idx = tuple(self._idx(n, x) for n, x in zip(self.AXES, (v, t_in, lam, fu)))
            out = {f: float(self.data[f][idx]) for f in self.FIELDS}
            out["idx"] = idx
            out["method"] = "nearest"
            return out

        br = [self._bracket(n, x) for n, x in zip(self.AXES, (v, t_in, lam, fu))]
        lo = np.array([b[0] for b in br], dtype=np.int64)
        hi = np.array([b[1] for b in br], dtype=np.int64)
        w = np.array([b[2] for b in br], dtype=float)
        corner = lo + self._corners * (hi - lo)             # (16, 4)
        flat = np.ravel_multi_index(corner.T, self.shape)   # (16,)
        vals = self._stack[:, flat]                         # (4, 16)
        wgt = np.prod(np.where(self._corners == 1, w, 1.0 - w), axis=1)  # (16,)
        if np.isfinite(vals).all():
            res = vals @ wgt                                # (4,)
        else:
            ok = np.isfinite(vals)
            w2 = np.where(ok, wgt, 0.0)
            den = w2.sum(axis=1)
            res = np.divide((vals * w2).sum(axis=1), den,
                            out=np.zeros(len(self.FIELDS)), where=den > 0)
        out = {f: float(res[i]) for i, f in enumerate(self.FIELDS)}
        out["idx"] = tuple(int(x) for x in lo)
        out["method"] = "linear"
        return out

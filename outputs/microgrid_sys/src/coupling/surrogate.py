# -*- coding: utf-8 -*-
"""PINN ONNX 推理封装（阶段 D1 骨架）：onnxruntime + domain_guard + 缓存 + 延迟统计。

用法：
    sur = PinnSurrogate(onnx_path)
    out = sur.infer(ops, dom_id)        # ops (N,7) 物理量；返回 (N,12) float32
    out, in_domain, info = sur.infer_guarded(ops)
    bench = sur.benchmark()             # CPU 延迟基准

说明：ONNX 图内已含输入归一化 + domain one-hot + 输出反归一化（见 pinn_sofc/export_onnx.py），
调用侧直接给物理量。v18 列序 [x,y,t,V_applied,T_in,lambda_air,FU]（与 domain_guard.COL_NAMES 一致）。
输出 12 列 [T,u,v,p,w1,w2,w3,phis,phil,htQ,qav_an,qav_ca]；w4=1-w1-w2-w3；NaN 掩码域由调用侧处理。
"""
from __future__ import annotations

import time

import numpy as np

from .domain_guard import check as guard_check


class PinnSurrogate:
    """单 ONNX 会话封装：批量推理、结果缓存、CPU 延迟基准。"""

    N_CONT = 7          # v18 默认；v19=10（初始化时按 ONNX 输入维覆盖）
    N_OUT = 12          # T,u,v,p,w1,w2,w3,phis,phil,htQ,qav_an,qav_ca

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        self.onnx_path = str(onnx_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(self.onnx_path, so,
                                         providers=providers or ["CPUExecutionProvider"])
        for i in self.sess.get_inputs():
            if i.name == "x_phys":
                n_in = int(i.shape[1])
                if n_in in (7, 10):
                    self.N_CONT = n_in
                break
        self._cache = {}
        self.stats = {"n_calls": 0, "cache_hits": 0, "n_out_of_domain": 0,
                      "single_ms": []}

    # ------------------------------------------------------------------
    def infer(self, ops, dom_id=None) -> np.ndarray:
        """ops (N,7) float；dom_id (N,) int64（默认 electrolyte=2）。返回 (N,12) float32。"""
        ops = np.asarray(ops, dtype=np.float32)
        if ops.ndim == 1:
            ops = ops[None, :]
        n = int(ops.shape[0])
        if ops.shape[1] != self.N_CONT:
            raise ValueError(f"ops 须为 (N,{self.N_CONT})，实际 {ops.shape}")
        if dom_id is None:
            dom_id = np.zeros(n, dtype=np.int64) + 2
        else:
            dom_id = np.asarray(dom_id, dtype=np.int64).reshape(-1)
            if dom_id.size == 1 and n > 1:
                dom_id = np.full(n, int(dom_id[0]), dtype=np.int64)
        self.stats["n_calls"] += 1
        key = (ops.tobytes(), dom_id.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        t0 = time.perf_counter()
        y = self.sess.run(None, {"x_phys": ops, "dom_id": dom_id})[0]
        self.stats["single_ms"].append((time.perf_counter() - t0) * 1e3)
        self._cache[key] = y
        return y

    def infer_guarded(self, ops, dom_id=None, margin_frac: float = 0.0):
        """先 domain_guard 检查；越界行输出 NaN 且不推理。返回 (out, in_domain, info)。"""
        ops64 = np.asarray(ops, dtype=np.float64)
        in_domain, info = guard_check(ops64, margin_frac=margin_frac)
        n_viol = int((~in_domain).sum())
        self.stats["n_out_of_domain"] += n_viol
        if n_viol:
            out = np.full((int(ops64.shape[0]), self.N_OUT), np.nan, dtype=np.float32)
        else:
            out = self.infer(ops64, dom_id=dom_id)
        return out, in_domain, info

    # ------------------------------------------------------------------
    def benchmark(self, n: int = 200, batch: int = 64, warmup: int = 20) -> dict:
        """CPU 延迟基准（单点 + batch=64）。返回 ms 值。"""
        ops = np.zeros((batch, self.N_CONT), dtype=np.float32)
        ops[:, 0] = 0.05; ops[:, 1] = 1.31e-3; ops[:, 2] = 0.0; ops[:, 3] = 0.85
        if self.N_CONT >= 10:      # v19：零波形参数 + T_in,lambda_air,FU 在 7-9 列
            ops[:, 7] = 1123.0; ops[:, 8] = 6.0; ops[:, 9] = 0.7
        else:                      # v18：T_in,lambda_air,FU 在 4-6 列
            ops[:, 4] = 1123.0; ops[:, 5] = 6.0; ops[:, 6] = 0.7
        dom = np.zeros(batch, dtype=np.int64) + 2
        for _ in range(warmup):
            self.sess.run(None, {"x_phys": ops[:1], "dom_id": dom[:1]})
        t0 = time.perf_counter()
        for _ in range(n):
            self.sess.run(None, {"x_phys": ops[:1], "dom_id": dom[:1]})
        t_single = (time.perf_counter() - t0) / n * 1e3
        n_b = max(5, warmup // 4)
        t0 = time.perf_counter()
        for _ in range(n_b):
            self.sess.run(None, {"x_phys": ops, "dom_id": dom})
        t_batch = (time.perf_counter() - t0) / n_b * 1e3
        return {"single_ms": t_single, "batch64_ms": t_batch}

    def close(self):
        try:
            del self.sess
        except Exception:
            pass
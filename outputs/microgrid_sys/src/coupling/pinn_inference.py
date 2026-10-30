# -*- coding: utf-8 -*-
"""D5 公共 PINN 推理 API：evaluate_field / evaluate_table + 域外告警（ORT）。

封装 PinnSurrogate + domain_guard 为两个高层接口：
    evaluate_field(ops)      -> 全场物理量（任意探针点批量；域外行 NaN + 逐变量告警）
    evaluate_table(ops_grid) -> 工况网格批量推理 -> 场汇总查找表（Tmax / qav_an / qav_ca / Δφ）
二者均：
    - 自动读取 configs/coupling.yaml 的 onnx_path（可显式覆盖）；
    - 自动识别 v18（7 维）/ v19（10 维）列序（x,y,t,V,amp,stepT,wave_type,T_in,λ,FU）；
    - 域外行不推理（NaN 掩码）并给出逐变量告警明细；
    - 同一 onnx 路径的 API 实例缓存复用（模块级单例）。

用法：
    from src.coupling.pinn_inference import evaluate_field, evaluate_table
    r = evaluate_field(np.array([[0.05, 1.31e-3, 0.0, 0.85, 1123.0, 6.0, 0.7]]))
    r["fields"]      # (N,12) float32：T,u,v,p,w1..w3,phis,phil,htQ,qav_an,qav_ca
    r["in_domain"]   # (N,) bool；r["violations"] -> [{'row','var','value','range'}, ...]
    t = evaluate_table(grid_ops, out_npz="results/table_api.npz")
    t["tmax_k"]      # (M,) 每工况全场最高温度估计（域外行 NaN）
验证：python scripts/check_inference.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .domain_guard import DOMAIN_ORDER, check as guard_check
from .surrogate import PinnSurrogate
from .pinn_coupling import _DOMAIN_MID_Y, load_coupling_params

FIELD_NAMES = ["T", "u", "v", "p", "w1", "w2", "w3", "phis", "phil", "htQ",
               "qav_an", "qav_ca"]


def _nanmax(x) -> float:
    """全 NaN 安全的最大值（返回 NaN 而不告警）。"""
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


class PinnInference:
    """高层公共推理 API（阶段 D5）。"""

    def __init__(self, onnx_path=None, config_dir=None):
        p = load_coupling_params(str(Path(config_dir) / "coupling.yaml") if config_dir else None)
        root = Path(__file__).resolve().parents[2]
        self.onnx_path = str(Path(onnx_path) if onnx_path else root / p["onnx_path"])
        self.sur = PinnSurrogate(self.onnx_path)
        self.cont_version = self.sur.N_CONT
        self.probes = [(d, float(y)) for d, y in _DOMAIN_MID_Y.items()]
        self.dom_id = np.array([DOMAIN_ORDER.index(d) for d, _ in self.probes], dtype=np.int64)
        self.probe_x_m = float(p.get("probe_x_m", 0.05))

    # ------------------------------------------------------------------
    def evaluate_field(self, ops, dom_id=None, margin_frac: float = 0.0) -> dict:
        """任意探针点批量推理，返回 12 列物理量 + 域掩码 + 告警。

        ops : (N, 7|10) float（每行含 x,y,t 与工况列）；dom_id 缺省全为电解质。
        """
        ops = np.asarray(ops, dtype=np.float64)
        if ops.ndim == 1:
            ops = ops[None, :]
        n, nc = ops.shape
        if nc != self.cont_version:
            raise ValueError(f"ops 须为 (N,{self.cont_version})（cont_version={self.cont_version}），"
                             f"实际 {ops.shape}")
        out, in_domain, info = self.sur.infer_guarded(ops, dom_id=dom_id,
                                                      margin_frac=margin_frac)
        return {
            "cont_version": self.cont_version,
            "fields": out,                       # (N,12) float32；域外行 NaN
            "columns": list(FIELD_NAMES),
            "in_domain": in_domain,              # (N,) bool
            "n_valid": int(in_domain.sum()),
            "n_out_of_domain": int((~in_domain).sum()),
            "violations": list(info.get("violation_detail", [])),
            "warnings": [int(i) for i in info.get("warnings", [])],
            "stats": {"n_calls": self.sur.stats["n_calls"],
                      "cache_hits": self.sur.stats["cache_hits"],
                      "n_out_of_domain_total": self.sur.stats["n_out_of_domain"]},
        }

    # ------------------------------------------------------------------
    def evaluate_table(self, ops_grid, probes=None, batch: int = 512,
                       out_npz: str = None, margin_frac: float = 0.0) -> dict:
        """工况网格批量推理 -> 场汇总查找表（Tmax / qav_an / qav_ca / Δφ）。

        ops_grid : (M, 7|10) 每行一个工况点（x/y 列被探针坐标覆盖，t 保留）；
        probes   : 探针 [(domain, y_m), ...]，缺省为 5 域中线（与 build_pinn_table 一致）。
        out_npz  : 给定路径时导出 npz（ops_grid + 各场汇总 + meta）。
        返回 dict：tmax_k/qav_an_max/qav_ca_max/phi_drop_v (M,)，in_domain (M,)，
        violations（含 op/probe 索引），n_out_of_domain_probe_points。
        """
        ops_grid = np.asarray(ops_grid, dtype=np.float64)
        if ops_grid.ndim == 1:
            ops_grid = ops_grid[None, :]
        m, nc = ops_grid.shape
        if nc != self.cont_version:
            raise ValueError(f"ops_grid 须为 (M,{self.cont_version})，实际 {ops_grid.shape}")
        if probes is None:
            probes = self.probes
        n_p = len(probes)
        ys = np.array([y for _, y in probes], dtype=float)
        dom_full = np.tile(self.dom_id, m)

        # 展开：每工况复制 n_p 行，x/y 覆盖为探针坐标
        rows = np.repeat(ops_grid, n_p, axis=0)
        rows[:, 0] = self.probe_x_m
        rows[:, 1] = np.tile(ys, m)

        # 探针点级域检查 + 推理
        in_dom_all, info = guard_check(rows, margin_frac=margin_frac)
        n_oov = int((~in_dom_all).sum())
        fields = np.full((m * n_p, len(FIELD_NAMES)), np.nan, dtype=np.float32)
        ok_idx = np.where(in_dom_all)[0]
        for s in range(0, len(ok_idx), max(batch, 1)):
            idx = ok_idx[s:s + batch]
            fields[idx] = self.sur.infer(rows[idx], dom_full[idx])

        # 每工况聚合（与 build_pinn_table 相同口径）
        el_mask = self.dom_id == 2
        tmax_k = np.full(m, np.nan)
        qan = np.full(m, np.nan)
        qca = np.full(m, np.nan)
        phi = np.full(m, np.nan)
        for g in range(m):
            yg = fields[g * n_p:(g + 1) * n_p]
            tmax_k[g] = _nanmax(yg[:, 0])
            qan[g] = _nanmax(yg[:, 10])
            qca[g] = _nanmax(yg[:, 11])
            el = yg[el_mask]
            if len(el):
                phi[g] = _nanmax(np.abs(el[:, 7] - el[:, 8]))

        op_in = in_dom_all.reshape(m, n_p).all(axis=1)
        viols = []
        for v in info.get("violation_detail", []):
            row_i = int(v["row"])
            viols.append({"op": row_i // n_p, "probe": row_i % n_p,
                          "var": v["var"], "value": v["value"], "range": v["range"]})

        result = {
            "cont_version": self.cont_version,
            "n_ops": m, "n_probes": n_p,
            "tmax_k": tmax_k, "qav_an_max": qan, "qav_ca_max": qca,
            "phi_drop_v": phi,
            "in_domain": op_in,
            "n_valid": int(op_in.sum()),
            "n_out_of_domain_probe_points": n_oov,
            "violations": viols,
            "stats": {"n_calls": self.sur.stats["n_calls"],
                      "cache_hits": self.sur.stats["cache_hits"],
                      "n_out_of_domain_total": self.sur.stats["n_out_of_domain"]},
        }
        if out_npz:
            out_path = Path(out_npz)
            if not out_path.is_absolute():
                out_path = Path(__file__).resolve().parents[2] / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import time as _time
            meta = {"onnx": self.onnx_path, "cont_version": self.cont_version,
                    "probes": [{"domain": d, "y_m": y} for d, y in probes],
                    "created": _time.strftime("%Y-%m-%d %H:%M:%S")}
            np.savez(out_path, ops_grid=ops_grid, tmax_k=tmax_k,
                     qav_an_max=qan, qav_ca_max=qca, phi_drop_v=phi,
                     in_domain=op_in, meta=np.array(meta))
            result["npz_path"] = str(out_path)
        return result


# ---------------------------------------------------------------------------
# 模块级接口（单例缓存）
# ---------------------------------------------------------------------------
_API_CACHE = {}


def _get_api(onnx_path=None, config_dir=None) -> PinnInference:
    key = str(onnx_path) if onnx_path else "default"
    if key not in _API_CACHE:
        _API_CACHE[key] = PinnInference(onnx_path=onnx_path, config_dir=config_dir)
    return _API_CACHE[key]


def evaluate_field(ops, dom_id=None, onnx_path=None, config_dir=None,
                   margin_frac: float = 0.0) -> dict:
    """模块级 evaluate_field（等价 PinnInference().evaluate_field）。"""
    return _get_api(onnx_path, config_dir).evaluate_field(ops, dom_id=dom_id,
                                                          margin_frac=margin_frac)


def evaluate_table(ops_grid, probes=None, onnx_path=None, config_dir=None,
                   batch: int = 512, out_npz: str = None,
                   margin_frac: float = 0.0) -> dict:
    """模块级 evaluate_table（等价 PinnInference().evaluate_table）。"""
    return _get_api(onnx_path, config_dir).evaluate_table(ops_grid, probes=probes,
                                                          batch=batch, out_npz=out_npz,
                                                          margin_frac=margin_frac)
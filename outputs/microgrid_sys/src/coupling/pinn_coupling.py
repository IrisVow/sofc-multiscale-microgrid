# -*- coding: utf-8 -*-
"""EMS/PINN 双层耦合监视器（阶段 D1 骨架）。

职责：
1. 工况映射：SOFC 步进后的实际工作点（v_cell, u_f）-> PINN 探针输入
   ops (N_probes,7) = [x, y_probe, t=0, V_cell, T_in_nominal, lambda_air_nominal, FU]。
2. 域检查：domain_guard 判定是否在 PINN 训练域（V∈[0.75,1.0]、T_in∈[1073,1123]、
   λ∈[4,8]、FU∈[0.6,0.8]）；越界步骤不推理并计数。
3. 查询：lut 模式（默认，快）返回温度代理量/qav/Δφ；live 模式逐点推理（慢，仅供验证）。
   原生产 LUT 的 tmax 是 x=50 mm 五探针温度最大值，并非全场热点。
   新表的温度取样口径见 PinnLUT.temperature_metric；准稳态表均省略热历史。
4. 在线约束（监视）：Tmax 估计 > tmax_limit 记为违反；D2 起转 EMS 软惩罚。

模式：off / lut / live；由 configs/coupling.yaml 的 use_pinn_table + live_infer 决定。
接入：Microgrid(coupling=PinnCoupling(...))；历史行带 pinn_* 列。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .domain_guard import check as guard_check
from .lut import PinnLUT
from .surrogate import PinnSurrogate

_DOMAIN_MID_Y = {
    "air_channel": 0.40e-3,
    "air_electrode": 1.05e-3,
    "electrolyte": 1.31e-3,
    "fuel_electrode": 1.57e-3,
    "fuel_channel": 2.22e-3,
}


def load_coupling_params(path=None) -> dict:
    """读 configs/coupling.yaml（或显式路径），返回 pinn 段。"""
    import yaml
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "coupling.yaml"
    with io.open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("pinn", cfg)


class PinnCoupling:
    """微电网侧 PINN 耦合监视器。"""

    def __init__(self, params: dict = None, config_dir: str = None):
        p = load_coupling_params(str(Path(config_dir) / "coupling.yaml") if config_dir else None)
        if params:
            p.update(params)
        self.p = p
        self.cont_version = int(p.get("cont_version", 7))
        self.t_s = float(p.get("t_s", 0.0))
        self.x = float(p.get("probe_x_m", 0.05))
        self.t_in = float(p.get("t_in_nominal_k", 1123.0))
        self.lam = float(p.get("lambda_air_nominal", 6.0))
        self.fu_nom = float(p.get('fu_nominal', 0.7))
        self.lut_method = str(p.get('lut_method', 'linear')).lower()  # D2: lookup method (linear/nearest)
        # tmax_limit：热点温度上界 = t_in_nominal + max_dt_k（2026-08-15 重标定语义）；
        # 显式 tmax_limit_k 优先（coupling.yaml 已写 1173.0 = 1123 + 50）。
        self.tmax_limit = float(p.get("tmax_limit_k") if p.get("tmax_limit_k") is not None
                                else self.t_in + float(p.get("max_dt_k", 50.0)))
        # D3 在线约束（软惩罚进 EMS）
        self.constraint_mode = str(p.get("constraint_mode", "monitor")).lower()
        self.derate_floor_pu = float(p.get("derate_floor_pu", 0.50))
        self.derate_bisect_iters = int(p.get("derate_bisect_iters", 10))
        self.derate_deadband_k = float(p.get("derate_deadband_k", 3.0))
        self.derate_margin_k = float(p.get("derate_margin_k", 2.0))  # D2: 降载目标安全裕度（覆盖插值/模型不确定度）

        probes = p.get("probes")
        if not probes:
            probes = [{"domain": d, "y_m": y} for d, y in _DOMAIN_MID_Y.items()]
        self.probes = [(pr["domain"], float(pr["y_m"])) for pr in probes]
        from .domain_guard import DOMAIN_ORDER
        self.dom_id = np.array([DOMAIN_ORDER.index(d) for d, _ in self.probes], dtype=np.int64)
        self.ys = np.array([y for _, y in self.probes], dtype=float)

        self.mode = "off"
        self._surrogate = None
        self._lut = None
        self.reset()
        self._init_backend()

    # ------------------------------------------------------------------
    def _init_backend(self):
        root = Path(__file__).resolve().parents[2]
        if bool(self.p.get("use_pinn_table", False)) and self.p.get("lut_path"):
            self._lut = PinnLUT(root / self.p["lut_path"])
            self.mode = "lut"
        elif bool(self.p.get("live_infer", False)) and self.p.get("onnx_path"):
            self._surrogate = PinnSurrogate(root / self.p["onnx_path"])
            self.mode = "live"

    def enabled(self) -> bool:
        return self.mode != "off"

    def reset(self):
        self.n_steps = 0          # monitor_step 总调用
        self.n_queried = 0        # 发电步（p_ac>0）数
        self.n_in_domain = 0
        self.n_out_of_domain = 0
        self.n_tmax_viol = 0
        self.max_tmax_est = 0.0
        self.derate_stats = {"steps": 0, "full_derate_steps": 0,
                             "derate_kwh": 0.0, "max_derate_kw": 0.0}
        self._last = {}

    # ------------------------------------------------------------------
    def _ops_rows(self, v_cell: float, fu: float) -> np.ndarray:
        """(N_probes, 7) 物理量；v19（10 维）时在 4-6 列插入零波形参数。"""
        rows = np.array([[self.x, y, self.t_s, v_cell, self.t_in, self.lam, fu]
                         for y in self.ys], dtype=float)
        if self.cont_version == 10:
            # v19 列序：x,y,t,V,amp,stepT,wave_type,T_in,lambda_air,FU（export_onnx.py）
            rows = np.concatenate([rows[:, :4], np.zeros((len(self.ys), 3)), rows[:, 4:]], axis=1)
        return rows

    def _query_lut(self, v_cell: float, fu: float) -> dict:
        r = self._lut.lookup(v_cell, self.t_in, self.lam, fu, method=self.lut_method)
        return {"tmax": r["tmax_k"], "qav_an": r["qav_an_max"],
                "qav_ca": r["qav_ca_max"], "phi_drop": r["phi_drop_v"]}

    def _query_live(self, ops: np.ndarray) -> dict:
        y = self._surrogate.infer(ops, self.dom_id)
        tmax = float(np.nanmax(y[:, 0]))
        qav_an = float(np.nanmax(y[:, 10]))
        qav_ca = float(np.nanmax(y[:, 11]))
        el = np.where(self.dom_id == 2)[0]
        phi_drop = float(np.nanmax(np.abs(y[el, 7] - y[el, 8]))) if len(el) else np.nan
        return {"tmax": tmax, "qav_an": qav_an, "qav_ca": qav_ca, "phi_drop": phi_drop}

    def evaluate(self, v_cell: float, fu: float) -> dict:
        """D3：按工作点 (v_cell, fu) 评估 PINN（lut/live 模式），供约束降载用。

        返回 {ok, in_domain, tmax, qav_an, qav_ca, phi_drop}；mode=off 时 ok=False。
        """
        if self.mode == "off":
            return {"ok": False, "in_domain": False, "tmax": float("nan"),
                    "qav_an": float("nan"), "qav_ca": float("nan"),
                    "phi_drop": float("nan")}
        ops = self._ops_rows(v_cell, fu)
        in_domain, info = guard_check(ops)
        if not bool(in_domain.all()):
            self.n_out_of_domain += 1
            return {"ok": False, "in_domain": False, "tmax": float("nan"),
                    "qav_an": float("nan"), "qav_ca": float("nan"),
                    "phi_drop": float("nan"),
                    "violations": info.get("violation_detail", [])}
        self.n_in_domain += 1
        q = self._query_lut(v_cell, fu) if self.mode == "lut" else self._query_live(ops)
        return {"ok": True, "in_domain": True, "tmax": float(q["tmax"]),
                "qav_an": float(q["qav_an"]), "qav_ca": float(q["qav_ca"]),
                "phi_drop": float(q["phi_drop"])}

    def note_derate(self, derate_kw: float, dt_s: float, full: bool = False):
        """D3：记录一次降载事件（软惩罚量：功率×时长）。"""
        if derate_kw > 0.0:
            self.derate_stats["steps"] += 1
            self.derate_stats["derate_kwh"] += float(derate_kw) * float(dt_s) / 3600.0
            self.derate_stats["max_derate_kw"] = max(self.derate_stats["max_derate_kw"],
                                                     float(derate_kw))
        if full:
            self.derate_stats["full_derate_steps"] += 1
    def monitor_step(self, r_sofc: dict) -> dict:
        """SOFC 步进结果 -> PINN 监视记录（并入 Microgrid 历史行）。"""
        rec = {"pinn_mode": self.mode, "pinn_in_domain": False,
               "pinn_tmax_est_k": np.nan, "pinn_qav_an": np.nan, "pinn_qav_ca": np.nan,
               "pinn_phi_drop_v": np.nan, "pinn_tmax_viol": False}
        self.n_steps += 1
        if self.mode == "off":
            rec["pinn_in_domain"] = True
            return rec
        p_ac = float(r_sofc.get("p_ac", 0.0))
        if p_ac <= 0.0:                       # 停机/待机步：无需查询
            return rec
        self.n_queried += 1
        v_cell = float(r_sofc.get("v_cell", 0.85))
        fu = float(r_sofc.get("u_f", self.fu_nom)) or self.fu_nom
        ops = self._ops_rows(v_cell, fu)
        in_domain, info = guard_check(ops)
        if not bool(in_domain.all()):
            self.n_out_of_domain += 1
            self._last = {"ok": False, "violations": info.get("violation_detail", [])}
            return rec
        self.n_in_domain += 1
        q = self._query_lut(v_cell, fu) if self.mode == "lut" else self._query_live(ops)
        viol = bool(q["tmax"] > self.tmax_limit)
        self.n_tmax_viol += int(viol)
        self.max_tmax_est = max(self.max_tmax_est, q["tmax"])
        rec.update({"pinn_in_domain": True, "pinn_tmax_est_k": q["tmax"],
                    "pinn_qav_an": q["qav_an"], "pinn_qav_ca": q["qav_ca"],
                    "pinn_phi_drop_v": q["phi_drop"], "pinn_tmax_viol": viol})
        self._last = rec
        return rec

    def summary(self) -> dict:
        oov_share = self.n_out_of_domain / max(self.n_queried, 1)
        out = {
            "pinn_mode": self.mode,
            "pinn_steps": self.n_steps,
            "pinn_queried": self.n_queried,
            "pinn_in_domain": self.n_in_domain,
            "pinn_out_of_domain": self.n_out_of_domain,
            "pinn_oov_share": oov_share,
            "pinn_tmax_violations": self.n_tmax_viol,
            "pinn_max_tmax_est_k": self.max_tmax_est,
            "pinn_constraint_mode": self.constraint_mode,
            "pinn_derate_steps": self.derate_stats["steps"],
            "pinn_derate_full_steps": self.derate_stats["full_derate_steps"],
            "pinn_derate_kwh": self.derate_stats["derate_kwh"],
            "pinn_derate_max_kw": self.derate_stats["max_derate_kw"],
        }
        if self._surrogate is not None:
            out["surrogate_cache_hits"] = self._surrogate.stats["cache_hits"]
        return out

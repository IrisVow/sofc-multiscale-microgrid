# -*- coding: utf-8 -*-
"""M1 数据管线：parquet 加载、归一化、域掩码、平衡采样、工况切分"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .split import DOMAIN_ORDER, DOMAIN_ID, PRIMARY_VARS, AUX_VARS, CONT_VARS, make_splits

META_COLS = ["case_id", "t_s", "V_applied", "T_in_K", "lambda_air", "FU"]
CONT_IDS = ["x", "y", "t_s", "V_applied", "T_in_K", "lambda_air", "FU"]
# v19: waveform-parameterized input (drop instantaneous V_applied)
CONT_IDS_V19 = ["x", "y", "t_s", "V_cell", "amp", "stepT", "wave_type",
                "T_in_K", "lambda_air", "FU"]


def cont_ids_for_cfg(cfg: dict):
    """Choose input columns: v19 waveform params when waveform_input=true."""
    return CONT_IDS_V19 if bool(cfg.get("data", {}).get("waveform_input", False)) else CONT_IDS


class PointPool:
    """一个切分（train/val/test）的数值池：输入/输出/域标签。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, dom: np.ndarray,
                 case_id: np.ndarray, t: np.ndarray, n_out: int):
        self.x = x.astype(np.float32)          # (N, n_in)
        self.y = y.astype(np.float32)          # (N, n_out) 未归一化
        self.dom = dom.astype(np.int64)        # (N,)
        self.case_id = case_id                 # (N,) object
        self.t = t.astype(np.float32)          # (N,)
        self.n_out = n_out
        self.n = len(x)
        # (case_id@t) 桶标签：用于瞬态按时间步分层采样
        self.case_t = np.char.add(np.asarray(case_id, dtype=str).astype("U32") + "@",
                                  t.astype(str).astype("U32"))
        # 可选瞬态快照权重（t=0/t_max/阶跃窗加权），None 表示均匀
        self.sample_weight = None
        # Pool membership is immutable after construction; cache indices, not weights.
        self._domain_indices = {}
        self._strat_indices = {}

    def _candidates(self, domain):
        if domain not in self._domain_indices:
            self._domain_indices[domain] = np.where(self.dom == domain)[0]
        return self._domain_indices[domain]

    def _buckets(self, domain):
        if domain not in self._strat_indices:
            cand = self._candidates(domain)
            labels, inv = np.unique(self.case_t[cand], return_inverse=True)
            # Stable grouping preserves the original ascending indices in every bucket.
            order = np.argsort(inv, kind='stable')
            groups = np.split(cand[order], np.cumsum(np.bincount(inv))[:-1]) if len(labels) else []
            self._strat_indices[domain] = groups
        return self._strat_indices[domain]

    def __len__(self):
        return self.n

    def sample_indices(self, n_per_domain: Dict[int, int], rng: np.random.Generator) -> np.ndarray:
        """按域配额随机采样（可重复）。"""
        idx = []
        for d, quota in n_per_domain.items():
            cand = self._candidates(d)
            if len(cand) == 0:
                continue
            k = min(quota, len(cand))
            idx.append(rng.choice(cand, size=k, replace=False))
        if not idx:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(idx)

    def sample_indices_strat(self, n_per_domain: Dict[int, int],
                             rng: np.random.Generator,
                             sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """按域配额 + 域内 (case_id,t) 分层采样：保证每个时间步快照都被覆盖。

        瞬态数据 ~100 万点分布在多个 (工况, 时间步) 快照上；若纯随机，
        单 epoch 可能漏掉 t=0 初始态或阶跃附近快照。这里对每个快照桶
        抽 ceil(quota/n_buckets) 个点（快照内再随机），再拼接；
        sample_weight 给定时，配额按桶均权重的比例分配（初始态/末态/
        阶跃窗可过采样）。
        """
        idx = []
        for d, quota in n_per_domain.items():
            cand = self._candidates(d)
            if len(cand) == 0:
                continue
            groups = self._buckets(d)
            if sample_weight is not None:
                wb = np.array([float(sample_weight[sub].mean()) for sub in groups])
                wb = wb / wb.sum()
                per_bucket = np.maximum(1, np.round(quota * wb).astype(int))
            else:
                per_bucket = np.full(len(groups), max(1, int(np.ceil(quota / len(groups)))))
            picked = []
            for b, sub in enumerate(groups):
                k = min(int(per_bucket[b]), len(sub))
                picked.append(rng.choice(sub, size=k, replace=False))
            idx.append(np.concatenate(picked))
        if not idx:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(idx)


class SOFCDataModule:
    """构建训练/验证/测试池 + 归一化统计 + 域掩码。"""

    def __init__(self, cfg: dict, device: torch.device):
        cfg = resolve_config_paths(cfg)
        self.cfg = cfg
        self.device = device
        d = cfg["data"]
        self.target_vars: List[str] = list(d["target_vars"])
        self.n_out = len(self.target_vars)
        self.waveform_input = bool(d.get("waveform_input", False))
        self.cont_ids = cont_ids_for_cfg(cfg)
        self.n_cont = len(self.cont_ids)
        self.t_scale = float(d.get("t_scale_s", 1200.0))
        self.splits = make_splits(cfg)
        self._tensor_cache: Dict[str, tuple] = {}

        self._load_dataframes()
        self._build_masks()
        self._fit_normalization()
        self._build_pools()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_dataframes(self):
        root = self.cfg["paths"]["data_root"]
        cols_keep = META_COLS + ["x", "y", "domain"] + self.target_vars
        self.df_steady = pd.read_parquet(
            os.path.join(root, self.cfg["paths"]["merged_steady"]), columns=cols_keep)
        self.df_transient = None
        if self.cfg["data"]["use_transient"]:
            # 瞬态额外读 stepT_s（阶跃时刻，用于快照加权/波形）；
            # v19 再读 V_cell/stepAmp_V/wave_type 以构建波形参数输入
            extra = ["stepT_s"] + (["V_cell", "stepAmp_V", "wave_type"] if self.waveform_input else [])
            self.df_transient = pd.read_parquet(
                os.path.join(root, self.cfg["paths"]["merged_transient"]),
                columns=cols_keep + extra)
        if self.waveform_input:
            self._add_waveform_cols()
        self.df_all = pd.concat([self.df_steady]
                                + ([self.df_transient] if self.df_transient is not None else []),
                                ignore_index=True)

    # ------------------------------------------------------------------
    # v19: 波形参数列（V_cell/amp/stepT/wave_type）
    def _add_waveform_cols(self):
        """稳态无动态 -> 默认参数（V_cell=V_applied, amp=0, stepT=0, wave_type=step）。

        瞬态：T 用例用数据列 stepT_s/stepAmp_V；R 连续用例用 R_CASE_META
        （amp/wave_type/char_t），wave_type 缺失（NaN）按 'step'。
        """
        from physics.waveforms import R_CASE_META, WAVE_CODE
        self.df_steady["V_cell"] = self.df_steady["V_applied"]
        self.df_steady["amp"] = 0.0
        self.df_steady["stepT"] = 0.0
        self.df_steady["wave_type"] = 0.0
        if self.df_transient is None:
            return
        tdf = self.df_transient
        if "V_cell" not in tdf.columns:
            tdf["V_cell"] = tdf["V_applied"]
        if "stepAmp_V" not in tdf.columns:
            tdf["stepAmp_V"] = np.nan
        if "wave_type" not in tdf.columns:
            tdf["wave_type"] = np.nan
        cid = tdf["case_id"].astype(str).to_numpy()
        amp = tdf["stepAmp_V"].to_numpy(dtype=np.float64).copy()
        stepT = tdf["stepT_s"].to_numpy(dtype=np.float64).copy()
        wtype = tdf["wave_type"].astype(str).to_numpy().copy()
        for rc, meta in R_CASE_META.items():
            m = cid == rc
            amp[m] = meta["amp"]
            stepT[m] = meta["char_t"]
            wtype[m] = meta["wave_type"]
        # T 用例 wave_type 缺失 -> step
        wtype = np.where((wtype == "nan") | (wtype == "None") | (wtype == ""), "step", wtype)
        tdf["amp"] = np.where(np.isfinite(amp), amp, 0.0)
        tdf["stepT"] = np.where(np.isfinite(stepT), stepT, 0.0)
        tdf["wave_type"] = [WAVE_CODE.get(w, 0) for w in wtype]

    # ------------------------------------------------------------------
    # 域掩码：按各域变量覆盖情况构建（NaN 缺口自动置 0）
    def _build_masks(self):
        masks = {}
        for dom in DOMAIN_ORDER:
            g = self.df_steady[self.df_steady["domain"] == dom]
            row = {}
            for v in self.target_vars:
                row[v] = 1 if g[v].isna().mean() == 0.0 else 0
            masks[dom] = row
        self.masks = masks                      # {dom: {var: 0/1}}
        # (n_domain, n_out) 掩码矩阵
        M = np.zeros((len(DOMAIN_ORDER), self.n_out), dtype=np.float32)
        for i, dom in enumerate(DOMAIN_ORDER):
            for j, v in enumerate(self.target_vars):
                M[i, j] = masks[dom][v]
        self.M = torch.from_numpy(M).to(self.device)
        print("[data] 域掩码（行=域, 列=变量 %s）:" % self.target_vars)
        print("       ", np.where(M.T > 0, "1", ".").T)

    # ------------------------------------------------------------------
    # 归一化统计（仅用 train 工况拟合；瞬态 t 用固定时间尺度）
    # ------------------------------------------------------------------
    def _fit_normalization(self):
        train_cases = set(self.splits["steady_train"])
        df = self.df_steady[self.df_steady["case_id"].isin(train_cases)]
        stats_df = df
        if self.waveform_input and self.df_transient is not None:
            # v19: 波形参数统计并入瞬态训练工况（amp/stepT/wave_type 在稳态为 0）
            ttrain = set(self.splits["transient_train"])
            tdf = self.df_transient[self.df_transient["case_id"].isin(ttrain)]
            stats_df = pd.concat([df[self.cont_ids], tdf[self.cont_ids]], ignore_index=True)

        x_stats, y_stats = {}, {}
        for c in self.cont_ids:
            s = float(stats_df[c].std())
            x_stats[c] = {"mean": float(stats_df[c].mean()), "std": s if s > 1e-12 else 1.0}
        # t_s 特殊处理：以 t=0 为参考，用固定时间尺度归一化
        x_stats["t_s"] = {"mean": 0.0, "std": self.t_scale}
        for v in self.target_vars:
            vals = df[v].to_numpy(dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            y_std = float(np.nanstd(vals)) if len(vals) else 1.0
            y_stats[v] = {"mean": float(np.nanmean(vals)), "std": y_std if y_std > 1e-12 else 1.0}
        self.x_stats, self.y_stats = x_stats, y_stats
        print("[data] 输入统计:", {c: (round(s["mean"], 4), round(s["std"], 4))
                                   for c, s in x_stats.items()})
        print("[data] 输出 σ:", {v: round(s["std"], 4) for v, s in y_stats.items()})

    # ------------------------------------------------------------------
    # 从 DataFrame 构建数值池
    # ------------------------------------------------------------------
    def _pool_from_df(self, df: pd.DataFrame) -> PointPool:
        x = df[self.cont_ids].to_numpy(np.float32)
        y = np.full((len(df), self.n_out), np.nan, dtype=np.float32)
        for j, v in enumerate(self.target_vars):
            y[:, j] = df[v].to_numpy(np.float32)
        dom = df["domain"].map(DOMAIN_ID).to_numpy(np.int64)
        case_id = df["case_id"].to_numpy()
        t = df["t_s"].to_numpy(np.float32)
        return PointPool(x, y, dom, case_id, t, self.n_out)

    def _build_pools(self):
        self.pools: Dict[str, PointPool] = {}
        for name, cases in self.splits.items():
            if not cases:
                continue
            cases_set = set(cases)
            parts = []
            for df in ([self.df_steady] + ([self.df_transient] if self.df_transient is not None else [])):
                sub = df[df["case_id"].isin(cases_set)]
                if len(sub):
                    parts.append(sub)
            if parts:
                dfp = pd.concat(parts, ignore_index=True)
                pool = self._pool_from_df(dfp)
                if name.startswith("transient"):
                    # 快照权重：t=0 初始态、t=t_max 末态 ×3，阶跃窗 ×2.5
                    t = dfp["t_s"].to_numpy()
                    stepT = dfp["stepT_s"].to_numpy()
                    t_max = float(dfp["t_s"].max())
                    w = np.ones(len(dfp), dtype=np.float32)
                    w[t == 0] *= 3.0
                    w[t >= t_max - 1e-6] *= 3.0
                    w[np.abs(t - stepT) <= 60.0] *= float(self.cfg["data"].get("step_window_weight", 2.5))
                    pool.sample_weight = w
                elif name == "steady_train":
                    cw = self.cfg["data"].get("steady_case_weights", {})
                    if cw:
                        cid = np.asarray(dfp["case_id"], dtype=str)
                        w = np.ones(len(dfp), dtype=np.float32)
                        for c, wt in cw.items():
                            w[cid == str(c)] *= float(wt)
                        pool.sample_weight = w
                self.pools[name] = pool
                print(f"[data] pool {name}: cases={cases} n={len(self.pools[name])}")

    # ------------------------------------------------------------------
    # 输入/输出归一化与掩码张量
    # ------------------------------------------------------------------
    def _norm_x(self, pool: PointPool) -> torch.Tensor:
        nc = len(self.cont_ids)
        X = np.empty((pool.n, nc + len(DOMAIN_ORDER)), dtype=np.float32)
        for j, c in enumerate(self.cont_ids):
            X[:, j] = (pool.x[:, j] - self.x_stats[c]["mean"]) / self.x_stats[c]["std"]
        X[:, nc:] = (pool.dom[:, None] == np.arange(len(DOMAIN_ORDER))[None, :])
        return torch.from_numpy(X).to(self.device)

    def _norm_y(self, pool: PointPool) -> torch.Tensor:
        Y = np.empty_like(pool.y)
        for j, v in enumerate(self.target_vars):
            Y[:, j] = (pool.y[:, j] - self.y_stats[v]["mean"]) / self.y_stats[v]["std"]
        return torch.from_numpy(Y).to(self.device)

    def _mask_for_pool(self, pool: PointPool) -> torch.Tensor:
        m = self.M[pool.dom].cpu().numpy()       # (N, n_out) float32
        # NaN 位置强制置掩码 0（保险）
        m = m * (~np.isnan(pool.y)).astype(np.float32)
        return torch.from_numpy(m).to(self.device)

    def pool_tensors(self, name: str):
        pool = self.pools[name]
        return self._norm_x(pool), self._norm_y(pool), self._mask_for_pool(pool), pool

    def _tensors(self, name: str):
        """归一化张量（缓存，避免每 epoch 重算大瞬态池）。"""
        if name not in self._tensor_cache:
            pool = self.pools[name]
            self._tensor_cache[name] = (self._norm_x(pool), self._norm_y(pool),
                                        self._mask_for_pool(pool))
        return self._tensor_cache[name]

    # ------------------------------------------------------------------
    # 训练采样：按域配额生成一个 epoch 的批次
    # ------------------------------------------------------------------
    def domain_quota(self, batch_size: int) -> Dict[int, int]:
        w = self.cfg["data"]["domain_weights"]
        tot = sum(w.values())
        return {DOMAIN_ID[d]: max(1, int(batch_size * w[d] / tot)) for d in w}

    def iter_epoch(self, batch_size: int, rng: np.random.Generator):
        """产出一个 epoch 的 (X, Y, M) 批次张量（稳态 + 瞬态联合采样）。

        瞬态比例由 data.transient_frac 控制；瞬态池按 (case_id, t) 分层，
        保证 t=0 初始态与阶跃附近快照每 epoch 都被覆盖。
        """
        d = self.cfg["data"]
        n_target = int(d["samples_per_epoch"])
        tf = float(d.get("transient_frac", 0.5))
        quota = self.domain_quota(batch_size)
        jobs = []
        if tf < 1.0 and "steady_train" in self.pools:
            jobs.append(("steady_train", int(n_target * (1.0 - tf))))
        if tf > 0.0 and "transient_train" in self.pools:
            jobs.append(("transient_train", n_target - sum(j[1] for j in jobs)))
        for name, n_tot in jobs:
            pool = self.pools[name]
            X, Y, M = self._tensors(name)
            n_seen = 0
            while n_seen < n_tot:
                if pool.case_t is not None and (name.startswith("transient")
                                                   or pool.sample_weight is not None):
                    idx = pool.sample_indices_strat(quota, rng,
                                                    sample_weight=pool.sample_weight)
                else:
                    idx = pool.sample_indices(quota, rng)
                if len(idx) == 0:
                    break
                yield X[idx], Y[idx], M[idx]
                n_seen += len(idx)

    # ------------------------------------------------------------------
    # 持久化统计与掩码
    # ------------------------------------------------------------------
    def save_stats(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"x": self.x_stats, "y": self.y_stats, "target_vars": self.target_vars,
                       "domain_order": DOMAIN_ORDER}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "masks.json"), "w", encoding="utf-8") as f:
            json.dump(self.masks, f, ensure_ascii=False, indent=2)

    def denorm_y(self, Y: torch.Tensor) -> torch.Tensor:
        """归一化输出 -> 物理单位。"""
        Y = Y.detach().cpu().numpy()
        out = np.empty_like(Y)
        for j, v in enumerate(self.target_vars):
            out[:, j] = Y[:, j] * self.y_stats[v]["std"] + self.y_stats[v]["mean"]
        return out


def resolve_config_paths(cfg):
    """Resolve released config paths from the repository or trainer directory."""
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p / '.sofc-project').is_file())
    trainer = Path(__file__).resolve().parents[1]
    for key in ('data_root', 'run_dir'):
        if key in cfg['paths']:
            path = Path(cfg['paths'][key])
            if not path.is_absolute():
                anchor = root if path.parts[0] in ('outputs', 'work') else trainer
                cfg['paths'][key] = str((anchor / path).resolve())
    for section in ('data', 'pde'):
        for key in ('initial_temperature_file',):
            if key in cfg.get(section, {}):
                path = Path(cfg[section][key])
                if not path.is_absolute():
                    anchor = root if path.parts[0] in ('outputs', 'work') else trainer
                    cfg[section][key] = str((anchor / path).resolve())
    return cfg

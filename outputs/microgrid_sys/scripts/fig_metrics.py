# -*- coding: utf-8 -*-
"""F2 fig_metrics: cell-level validation metrics (steady_val / steady_test S9 / transient_test T3).

Reads eval_metrics_final.csv (version's run dir; falls back to best) and plots
(a) per-variable RMSE by pool and (b) T RMSE trend across v18/v19/v20.
Exports data first.

Usage (cwd = microgrid_sys):
    python scripts/fig_metrics.py --version v20
Outputs (results/figures/<version>/):
    metrics_summary_<version>.csv
    fig_metrics_<version>.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from scripts.fig_common import RUN_DIRS, style, outdir, resolve_run_dir, write_csv, savefig

POOL_LABEL = {"steady_val": "steady_val", "steady_test": "S9 (steady_test)",
              "transient_test": "T3 (transient_test)"}
VARS = ["T", "u", "v", "p", "w1", "w2", "w3", "phis", "phil", "htQ", "qav_an", "qav_ca", "Tmax(K)"]


def load_metrics(version):
    d = resolve_run_dir(version)
    for fn in ["eval_metrics_final.csv", "eval_metrics_best.csv"]:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["version"] = version
            df["ckpt"] = fn.replace("eval_metrics_", "").replace(".csv", "")
            return df
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v20")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    out = args.out or outdir(ver)
    style()

    df = load_metrics(ver)
    if df is None:
        raise SystemExit(f"[fig_metrics] no eval metrics for {ver} in {resolve_run_dir(ver)}")

    # pivot: pool x var -> rmse
    piv = df.pivot_table(index="pool", columns="var", values="rmse")
    summ = piv[VARS].reset_index()
    summ.insert(0, "version", ver)
    csv_path = os.path.join(out, f"metrics_summary_{ver}.csv")
    write_csv(summ, csv_path)
    print(f"[fig_metrics] exported {csv_path}")

    # version trend for T RMSE
    trend = []
    for v in ["v18", "v19", ver]:
        d = load_metrics(v)
        if d is None:
            continue
        t = d[d["var"] == "T"].copy()
        for _, r in t.iterrows():
            trend.append({"version": v, "pool": r["pool"],
                          "t_rmse_k": float(r["rmse"]),
                          "ckpt": r["ckpt"]})
    tr = pd.DataFrame(trend)

    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # (a) per-variable RMSE by pool (log scale for wide ranges)
    x = np.arange(len(VARS))
    width = 0.28
    colors = {"steady_val": "tab:blue", "steady_test": "tab:orange",
              "transient_test": "tab:green"}
    for k, pool in enumerate(["steady_val", "steady_test", "transient_test"]):
        if pool not in piv.index:
            continue
        vals = [piv.loc[pool, v] if v in piv.columns else np.nan for v in VARS]
        axes[0].bar(x + (k - 1) * width, np.abs(vals), width, label=POOL_LABEL[pool],
                    color=colors[pool], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(VARS, rotation=45, ha="right", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE (log)")
    axes[0].set_title(f"Cell-level validation RMSE by pool - PINN {ver} final")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # (b) T RMSE trend
    for pool in ["steady_val", "steady_test", "transient_test"]:
        sub = tr[tr["pool"] == pool].sort_values("version")
        if len(sub):
            axes[1].plot(sub["version"], sub["t_rmse_k"], "-o", label=POOL_LABEL[pool], lw=1.5)
    axes[1].set_xlabel("Model version")
    axes[1].set_ylabel("T RMSE [K]")
    axes[1].set_title("T RMSE trend across versions (v18/v19 best, v20 final)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    if "transient_test" in set(tr["pool"]):
        axes[1].axhline(10.0, color="k", ls="--", lw=1.0)
        axes[1].annotate("T3 gate 10 K", xy=(0.02, 10.4), fontsize=8)

    fig.suptitle(f"PINN validation metrics (version {ver})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, os.path.join(out, f"fig_metrics_{ver}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

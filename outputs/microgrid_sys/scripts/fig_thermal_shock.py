# -*- coding: utf-8 -*-
"""F2 fig_thermal_shock: thermal shock case time series (E4 / T7 step).

Reads results/thermal_shock/thermal_shock_case.csv + .json (produced by
scripts/thermal_shock_case.py) and plots Tmax(t), RMSE(t), max|dT|(t) vs COMSOL.
Exports a tidy csv first.

Usage (cwd = microgrid_sys):
    python scripts/fig_thermal_shock.py --version v20
Outputs (results/figures/<version>/):
    fig_thermal_shock_<version>.csv
    fig_thermal_shock_<version>.png
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from scripts.fig_common import style, outdir, write_csv, savefig

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "thermal_shock")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=os.path.join(RES, "thermal_shock_case.csv"))
    ap.add_argument("--json", default=os.path.join(RES, "thermal_shock_case.json"))
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    out = args.out or outdir(ver)
    style()

    if not os.path.exists(args.csv):
        raise SystemExit(f"[fig_thermal_shock] {args.csv} not found - run thermal_shock_case.py first")
    df = pd.read_csv(args.csv)
    meta = {}
    if os.path.exists(args.json):
        with io.open(args.json, encoding="utf-8") as f:
            meta = json.load(f)

    tidy = df.rename(columns={
        "t_s": "t_s", "rmse_k": "rmse_k", "rel_rmse_pct": "rel_rmse_pct",
        "max_abs_k": "max_abs_k", "tmax_com_k": "tmax_com_k",
        "tmax_pinn_k": "tmax_pinn_k", "viol_com": "viol_com", "viol_pinn": "viol_pinn"})
    csv_path = os.path.join(out, f"fig_thermal_shock_{ver}.csv")
    write_csv(tidy, csv_path)
    print(f"[fig_thermal_shock] exported {csv_path} ({len(df)} steps)")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    t = df["t_s"].to_numpy(float)

    ax = axes[0]
    ax.plot(t, df["tmax_com_k"], "-o", ms=3, lw=1.4, color="tab:blue", label="COMSOL Tmax")
    ax.plot(t, df["tmax_pinn_k"], "--s", ms=3, lw=1.4, color="tab:red", label="PINN Tmax")
    limit = float(meta.get("tmax_limit_k", 1123.0))
    ax.axhline(limit, color="k", ls=":", lw=1.2)
    ax.annotate(f"limit {limit:.0f} K", xy=(t[1], limit + 1.2), fontsize=8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("Tmax [K]")
    ax.set_title(f"Thermal shock Tmax(t) - {meta.get('case', 'T7')} "
                 f"({meta.get('step_amp_v', 0.2):.2f} V / {meta.get('step_t_s', 100):.0f} s step)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(t, df["rmse_k"], width=8.0, color="tab:blue", alpha=0.75, label="T RMSE")
    ax.set_xlabel("t [s]"); ax.set_ylabel("T RMSE [K]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(t, df["max_abs_k"], "-o", ms=3, lw=1.2, color="tab:red", label="max|dT|")
    ax2.set_ylabel("max|dT| [K]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.legend([ax.containers[0], ax2.lines[0]], ["T RMSE", "max|dT|"], fontsize=8, loc="upper left")
    ax.set_title(f"Thermal shock error(t) - overall RMSE "
                 f"{meta.get('metrics', {}).get('overall_rmse_k', float('nan')):.2f} K")
    ax.grid(alpha=0.3)

    m = meta.get("metrics", {})
    fig.suptitle(
        f"Thermal shock case {meta.get('case', 'T7')} - PINN {ver} vs COMSOL "
        f"(violations COMSOL {m.get('violations_com_steps', '?')} / PINN "
        f"{m.get('violations_pinn_steps', '?')})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, os.path.join(out, f"fig_thermal_shock_{ver}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

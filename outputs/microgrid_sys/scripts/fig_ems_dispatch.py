# -*- coding: utf-8 -*-
"""F2 fig_ems_dispatch: 7-day EMS dispatch + PINN Tmax constraint.

Reads results/ems_day_sim_pinn/ems_day_sim.csv and results/coupling/compare_pinn_ems.json
and plots (a) power dispatch, (b) SOC & SOFC efficiency, (c) PINN Tmax estimate with
production/derate limit references and constraint summary from the comparison run.
Exports a downsampled csv first.

Usage (cwd = microgrid_sys):
    python scripts/fig_ems_dispatch.py --version v20
Outputs (results/figures/<version>/):
    ems_dispatch_<version>.csv
    fig_ems_dispatch_<version>.png
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMS_CSV = os.path.join(ROOT, "results", "ems_day_sim_pinn", "ems_day_sim.csv")
EMS_JSON = os.path.join(ROOT, "results", "coupling", "compare_pinn_ems.json")
EMS_M2_CSV = os.path.join(ROOT, "results", "coupling", "compare_pinn_ems_derate_m2.csv")
EMS_M15_CSV = os.path.join(ROOT, "results", "coupling", "compare_pinn_ems_derate_m15.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=EMS_CSV)
    ap.add_argument("--json", default=EMS_JSON)
    ap.add_argument("--csv-m2", default=EMS_M2_CSV)
    ap.add_argument("--csv-m15", default=EMS_M15_CSV)
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    out = args.out or outdir(ver)
    style()

    if not os.path.exists(args.csv):
        raise SystemExit(f"[fig_ems_dispatch] {args.csv} not found - run compare_pinn_ems.py / run_day_sim.py first")
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    print(f"[fig_ems_dispatch] loaded {len(df)} steps")

    json_sum = {}
    if os.path.exists(args.json):
        with io.open(args.json, encoding="utf-8") as f:
            json_sum = json.load(f)

    def _load_derate_csv(path):
        if not os.path.exists(path):
            print(f"[fig_ems_dispatch] {path} not found - skipping derate trace")
            return None
        d = pd.read_csv(path, encoding="utf-8-sig")
        print(f"[fig_ems_dispatch] loaded {len(d)} steps from {os.path.basename(path)}")
        return d

    df_m2 = _load_derate_csv(args.csv_m2)
    df_m15 = _load_derate_csv(args.csv_m15)

    keep = ["t_s", "p_load_kw", "q_load_kw", "soc", "sofc_kw", "genset_kw", "bat_kw",
            "sofc_eta", "q_rec_kw", "q_boiler_kw", "pinn_tmax_est_k", "pinn_derate_kw",
            "pinn_tmax_viol"]
    cols = [c for c in keep if c in df.columns]
    sub = df[cols].iloc[::15].reset_index(drop=True)   # 15-min downsample for export/plot
    sub_m2 = df_m2[[c for c in cols if c in df_m2.columns]].iloc[::15].reset_index(drop=True) \
        if df_m2 is not None else None
    sub_m15 = df_m15[[c for c in cols if c in df_m15.columns]].iloc[::15].reset_index(drop=True) \
        if df_m15 is not None else None
    t_h_m2 = sub_m2["t_s"].to_numpy() / 3600.0 if sub_m2 is not None else None
    t_h_m15 = sub_m15["t_s"].to_numpy() / 3600.0 if sub_m15 is not None else None
    csv_path = os.path.join(out, f"ems_dispatch_{ver}.csv")
    write_csv(sub, csv_path)
    print(f"[fig_ems_dispatch] exported {csv_path} ({len(sub)} rows, 15-min)")

    import matplotlib.pyplot as plt
    t_h = sub["t_s"].to_numpy() / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(t_h, sub["p_load_kw"], lw=1.0, color="k", label="Load")
    ax.fill_between(t_h, 0, sub["sofc_kw"], step="mid", color="tab:red", alpha=0.6, label="SOFC")
    ax.fill_between(t_h, 0, sub["genset_kw"], step="mid", color="tab:orange", alpha=0.7, label="Genset")
    ax.fill_between(t_h, 0, sub["bat_kw"].clip(lower=0), step="mid", color="tab:green", alpha=0.5, label="Battery discharge")
    ax.fill_between(t_h, 0, sub["bat_kw"].clip(upper=0), step="mid", color="tab:blue", alpha=0.5, label="Battery charge")
    if sub_m15 is not None:
        ax.plot(t_h_m15, sub_m15["sofc_kw"], lw=1.1, ls="--", color="tab:red", alpha=0.85,
                label="SOFC derated (15 K margin)")
    ax.set_ylabel("Power [kW]")
    ax.set_title("7-day dispatch (EMS + PINN LUT; derate demo @1135 K)")
    ax.legend(fontsize=8, ncol=5, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_h, sub["soc"], lw=1.2, color="tab:blue", label="SOC")
    ax.set_ylabel("SOC [-]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.axhline(0.9, color="k", ls=":", lw=0.8); ax.axhline(0.2, color="k", ls=":", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(t_h, sub["sofc_eta"] * 100.0, lw=1.0, color="tab:green", label="SOFC eta")
    ax2.set_ylabel("SOFC efficiency [%]", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.legend([ax.lines[0], ax2.lines[0]], ["SOC", "SOFC eta"], fontsize=8, loc="lower left")
    ax.set_title("SOC & SOFC efficiency")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if "pinn_tmax_est_k" in sub.columns:
        ax.plot(t_h, sub["pinn_tmax_est_k"], lw=1.0, color="tab:blue",
                label="Tmax monitor (1173 K)")
    if sub_m2 is not None and "pinn_tmax_est_k" in sub_m2.columns:
        ax.plot(t_h_m2, sub_m2["pinn_tmax_est_k"], lw=1.0, color="tab:orange",
                label="Tmax derate @1135 K (2 K margin)")
    if sub_m15 is not None and "pinn_tmax_est_k" in sub_m15.columns:
        ax.plot(t_h_m15, sub_m15["pinn_tmax_est_k"], lw=1.0, ls="--", color="tab:purple",
                label="Tmax derate @1135 K (15 K margin)")
    ax.axhline(1173.0, color="k", ls="--", lw=1.2)
    ax.annotate("production limit 1173 K", xy=(0.02, 1174.5), fontsize=8)
    ax.axhline(1135.0, color="tab:orange", ls=":", lw=1.2)
    ax.annotate("derate demo limit 1135 K", xy=(0.02, 1136.5), fontsize=8)
    ax.axhline(1120.0, color="tab:purple", ls=":", lw=1.0)
    ax.annotate("m15 target 1120 K", xy=(0.02, 1121.0), fontsize=8, color="tab:purple")
    sum_m2 = json_sum.get("derate_hot_1135", {}).get("pinn", {})
    sum_m15 = json_sum.get("derate_hot_1135_m15", {}).get("pinn", {})
    stat_lines = []
    if sum_m2:
        stat_lines.append(
            f"2 K margin : {sum_m2.get('pinn_derate_steps', 0)} steps / "
            f"{sum_m2.get('pinn_derate_kwh', 0):.0f} kWh, cap "
            f"{sum_m2.get('pinn_max_tmax_est_k', 0):.1f} K, 0 viol")
    if sum_m15:
        stat_lines.append(
            f"15 K margin: {sum_m15.get('pinn_derate_steps', 0)} steps / "
            f"{sum_m15.get('pinn_derate_kwh', 0):.0f} kWh, cap "
            f"{sum_m15.get('pinn_max_tmax_est_k', 0):.1f} K, 0 viol")
    if stat_lines:
        ax.text(0.98, 0.06, "\n".join(stat_lines),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))
    ax.set_ylabel("Tmax [K]"); ax.set_xlabel("t [h]")
    ax.set_title("PINN Tmax constraint: monitor vs 2 K / 15 K margin derating (7-day)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

    fig.suptitle(f"EMS 7-day simulation with PINN coupling - v{ver}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, os.path.join(out, f"fig_ems_dispatch_{ver}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

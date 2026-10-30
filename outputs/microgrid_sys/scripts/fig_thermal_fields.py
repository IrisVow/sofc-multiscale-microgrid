# -*- coding: utf-8 -*-
"""F2 fig_thermal_fields: steady (S9) and transient (T3, t=1200 s) field slices.

Compares PINN (ONNX) against COMSOL point-by-point on the COMSOL mesh and plots
T / phis / htQ fields + absolute error (English labels). Exports data first.

Usage (cwd = microgrid_sys):
    python scripts/fig_thermal_fields.py --version v20 [--cases s9,t3]
Outputs (results/figures/<version>/):
    fields_s9_<version>.csv / fields_t3_<version>.csv
    fig_thermal_fields_s9_<version>.png / fig_thermal_fields_t3_<version>.png
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

import numpy as np
import pandas as pd

from scripts.fig_common import P_DATA, style, outdir, resolve_onnx, write_csv, savefig

DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]
VAR_COLS = {"T": (0, "T [K]"), "phis": (7, "phi_s [V]"), "htQ": (9, "htQ [W/m^3]")}
DOM_BOUNDS_MM = [0.0, 0.8, 1.30, 1.32, 1.82, 2.62]


def load_case(parquet, case_id, t_s=None):
    cols = ["case_id", "t_s", "V_cell", "stepAmp_V", "stepT_s", "T_in_K",
            "lambda_air", "FU", "x", "y", "domain", "T", "phis", "htQ"]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df["case_id"] == case_id].reset_index(drop=True)
    if t_s is not None:
        df = df[np.isclose(df["t_s"], t_s)].reset_index(drop=True)
    return df


def build_ops(df):
    """10-dim waveform input [x,y,t,V_cell,amp,stepT,wave_type,T_in,lam,FU]."""
    return np.column_stack([
        df["x"].to_numpy(float), df["y"].to_numpy(float), df["t_s"].to_numpy(float),
        df["V_cell"].to_numpy(float), df["stepAmp_V"].fillna(0.0).to_numpy(float),
        df["stepT_s"].fillna(0.0).to_numpy(float),
        np.zeros(len(df), dtype=float),
        df["T_in_K"].to_numpy(float), df["lambda_air"].to_numpy(float),
        df["FU"].to_numpy(float),
    ])


def run_case(onnx, parquet, case_id, t_s, ver, out, batch=16384):
    from src.coupling import evaluate_field
    df = load_case(parquet, case_id, t_s)
    ops = build_ops(df)
    dom_id = df["domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)}).to_numpy(np.int64)
    fields = np.full((len(df), 12), np.nan, dtype=np.float32)
    n_out = 0
    for i in range(0, len(df), batch):
        sl = slice(i, i + batch)
        r = evaluate_field(ops[sl], dom_id=dom_id[sl], onnx_path=onnx)
        fields[sl] = r["fields"]
        n_out += r["n_out_of_domain"]
    tag = f"{case_id.lower()}_" + (f"t{int(t_s)}" if t_s is not None else "steady")
    out_df = pd.DataFrame({
        "x_mm": df["x"].to_numpy() * 1e3,
        "y_mm": df["y"].to_numpy() * 1e3,
        "domain": df["domain"].to_numpy(),
        "t_s": df["t_s"].to_numpy(),
        "T_com": df["T"].to_numpy(),
        "T_pinn": fields[:, 0],
        "phis_com": df["phis"].to_numpy(),
        "phis_pinn": fields[:, 7],
        "htQ_com": df["htQ"].to_numpy(),
        "htQ_pinn": fields[:, 9],
    })
    csv_path = os.path.join(out, f"fields_{tag}_{ver}.csv")
    write_csv(out_df, csv_path)
    print(f"[fig_thermal_fields] exported {csv_path} "
          f"({len(out_df)} pts, out-of-domain {n_out})")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for col, (var, (j, label)) in enumerate(VAR_COLS.items()):
        com = out_df[f"{var}_com"].to_numpy(float)
        pinn = out_df[f"{var}_pinn"].to_numpy(float)
        m = np.isfinite(com) & np.isfinite(pinn)
        err = np.abs(pinn - com)
        rmse = float(np.sqrt(np.mean(err[m] ** 2))) if m.sum() else float("nan")
        mae = float(np.mean(err[m])) if m.sum() else float("nan")
        for row, (vals, cmap, vmin, vmax, title) in enumerate([
            (com, "turbo", np.nanpercentile(com, 0.5), np.nanpercentile(com, 99.5),
             f"COMSOL {label}"),
            (err, "RdBu_r", None, None,
             f"PINN |err| (RMSE={rmse:.3g}, MAE={mae:.3g})"),
        ]):
            ax = axes[row, col]
            mm = np.isfinite(vals)
            sc = ax.scatter(out_df.loc[mm, "x_mm"], out_df.loc[mm, "y_mm"],
                            c=vals[mm], s=1.2, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            for b in DOM_BOUNDS_MM:
                ax.axhline(b, color="k", lw=0.4, alpha=0.5)
            ax.set_xlim(0, 100); ax.set_ylim(0, 2.62)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            fig.colorbar(sc, ax=ax, fraction=0.046)
    case_title = (f"{case_id} steady (T_in={df['T_in_K'].iloc[0]:.0f} K, "
                  f"V={df['V_cell'].iloc[0]:.2f} V)")
    if t_s is not None:
        case_title = (f"{case_id} transient t={int(t_s)} s "
                      f"(step {df['stepAmp_V'].iloc[0]:.2f} V / {df['stepT_s'].iloc[0]:.0f} s, "
                      f"T_in={df['T_in_K'].iloc[0]:.0f} K)")
    fig.suptitle(f"Thermal fields - PINN {ver} vs COMSOL ({case_title})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(out, f"fig_thermal_fields_{tag}_{ver}.png")
    savefig(fig, png)
    plt.close(fig)
    return {"case": case_id, "t_s": t_s, "n_pts": len(df), "out_of_domain": n_out,
            "png": os.path.basename(png), "csv": os.path.basename(csv_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default="s9,t3")
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    onnx = args.onnx or resolve_onnx(ver)
    out = args.out or outdir(ver)
    style()
    print(f"[fig_thermal_fields] version={ver} onnx={onnx}")

    manifest = []
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c == "s9":
            manifest.append(run_case(onnx, os.path.join(P_DATA, "doe_ext", "merged",
                                                        "doe_steady_combined.parquet"),
                                     "S9", None, ver, out))
        elif c == "t3":
            manifest.append(run_case(onnx, os.path.join(P_DATA, "doe_cont", "merged",
                                                        "doe_transient_combined2.parquet"),
                                     "T3", 1200.0, ver, out))
        else:
            print(f"[fig_thermal_fields] skip unknown case {c}")
    save_json(manifest, os.path.join(out, f"fields_manifest_{ver}.json"))


if __name__ == "__main__":
    from scripts.fig_common import save_json
    main()

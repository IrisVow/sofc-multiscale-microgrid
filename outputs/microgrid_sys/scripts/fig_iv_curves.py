# -*- coding: utf-8 -*-
"""F2 fig_iv_curves: I-V / P-V / efficiency / Tmax curves from PINN ONNX (version-parameterized).

For a fixed operating point (T_in, lambda_air, FU), scans V_cell on the COMSOL
electrode mesh points, integrates qav_an/qav_ca -> i_avg (A/m^2), and overlays the
COMSOL DOE reference I-V points. Tmax / phi_drop come from the coupling LUT csv.

Usage (cwd = microgrid_sys):
    python scripts/fig_iv_curves.py --version v20
Outputs (results/figures/<version>/):
    iv_curves_<version>.csv   exported data (before plotting)
    fig_iv_curves_<version>.png
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

from scripts.fig_common import (P_DATA, style, outdir, resolve_onnx, write_csv, savefig)

F = 96485.33
LHV_H2_MOLAR = 241.8e3
Y_THICK_EL = 5.0e-4
DOM_FUEL_EL, DOM_AIR_EL = 3, 1

DOE_STEADY = os.path.join(P_DATA, "doe_ext", "merged", "doe_steady_combined.parquet")


def _predict_avg(onnx, df_pts, dom_id, V, T_in, lam, fu, batch=8192):
    """Return qav mean on given domain points (A/m^3)."""
    from src.coupling import evaluate_field
    x = df_pts["x"].to_numpy(float)
    y = df_pts["y"].to_numpy(float)
    ops = np.column_stack([x, y, np.zeros(len(x), dtype=float),
                           np.full(len(x), V, dtype=float),
                           np.zeros(len(x), dtype=float),
                           np.zeros(len(x), dtype=float),
                           np.zeros(len(x), dtype=float),
                           np.full(len(x), T_in, dtype=float),
                           np.full(len(x), lam, dtype=float),
                           np.full(len(x), fu, dtype=float)])
    did = np.full(len(x), dom_id, dtype=np.int64)
    qav = np.full(len(x), np.nan, dtype=np.float32)
    for i in range(0, len(x), batch):
        sl = slice(i, i + batch)
        r = evaluate_field(ops[sl], dom_id=did[sl], onnx_path=onnx)
        qav[sl] = r["fields"][:, 10] if dom_id == DOM_FUEL_EL else r["fields"][:, 11]
    return qav


def load_electrode_mesh(doe_path, T_in, lam, fu):
    cols = ["case_id", "type", "T_in_K", "lambda_air", "FU", "V_applied",
            "x", "y", "domain", "qav_an", "qav_ca"]
    df = pd.read_parquet(doe_path, columns=cols)
    df = df[(df["type"] == "steady") & (df["T_in_K"] == T_in) &
            (df["lambda_air"] == lam) & (df["FU"] == fu)].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"[iv] no COMSOL cases at T_in={T_in}, lam={lam}, FU={fu}")
    pts_fuel = df[df["domain"] == "fuel_electrode"][["x", "y"]].drop_duplicates()
    pts_air = df[df["domain"] == "air_electrode"][["x", "y"]].drop_duplicates()
    ref = []
    for case, g in df.groupby("case_id"):
        g_fuel = g[g["domain"] == "fuel_electrode"]
        ref.append({"case": case, "V": float(g["V_applied"].iloc[0]),
                    "i_avg_ref": float(g_fuel["qav_an"].mean() * Y_THICK_EL)})
    return pts_fuel, pts_air, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="model tag, e.g. v20")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--T_in", type=float, default=1073.0)
    ap.add_argument("--lambda_air", type=float, default=8.0)
    ap.add_argument("--FU", type=float, default=0.6)
    ap.add_argument("--doe", default=DOE_STEADY)
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    onnx = args.onnx or resolve_onnx(ver)
    out = args.out or outdir(ver)
    style()
    print(f"[fig_iv_curves] version={ver} onnx={onnx}")

    pts_fuel, pts_air, ref = load_electrode_mesh(args.doe, args.T_in, args.lambda_air, args.FU)
    print(f"[fig_iv_curves] electrode mesh: fuel={len(pts_fuel)} air={len(pts_air)} "
          f"COMSOL ref cases={[r['case'] for r in ref]}")

    Vs = np.round(np.linspace(0.75, 1.0, 26), 4)
    rows = []
    for V in Vs:
        q_an = _predict_avg(onnx, pts_fuel, DOM_FUEL_EL, V, args.T_in, args.lambda_air, args.FU)
        q_ca = _predict_avg(onnx, pts_air, DOM_AIR_EL, V, args.T_in, args.lambda_air, args.FU)
        i_an = float(np.nanmean(q_an) * Y_THICK_EL)
        i_ca = float(np.nanmean(-q_ca) * Y_THICK_EL)
        i_avg = 0.5 * (i_an + i_ca)
        rows.append({
            "V_applied": V, "T_in": args.T_in, "lambda_air": args.lambda_air, "FU": args.FU,
            "i_avg_an": i_an, "i_avg_ca": i_ca, "i_avg": i_avg,
            "P_density_W_m2": V * i_avg,
            "eta_el": V * 2.0 * F * args.FU / LHV_H2_MOLAR,
        })
    df = pd.DataFrame(rows)

    # merge Tmax / phi_drop from the coupling LUT (nominal rows)
    lut_csv = os.path.join(out, "..", "..", "coupling", f"pinn_lut_{ver}.csv")
    if os.path.exists(lut_csv):
        lut = pd.read_csv(lut_csv)
        lut = lut[(lut["T_in"] == args.T_in) & (lut["lambda_air"] == args.lambda_air) &
                  (lut["FU"] == args.FU)].sort_values("V")
        if len(lut):
            df["Tmax_K"] = np.interp(df["V_applied"], lut["V"].to_numpy(), lut["tmax_k"].to_numpy())
            df["phi_drop_v"] = np.interp(df["V_applied"], lut["V"].to_numpy(),
                                         lut["phi_drop_v"].to_numpy())
        else:
            df["Tmax_K"] = np.nan
            df["phi_drop_v"] = np.nan
    else:
        df["Tmax_K"] = np.nan
        df["phi_drop_v"] = np.nan

    csv_path = os.path.join(out, f"iv_curves_{ver}.csv")
    write_csv(df, csv_path)
    print(f"[fig_iv_curves] exported {csv_path} ({len(df)} rows)")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    v = df["V_applied"].to_numpy()
    axes[0, 0].plot(v, df["i_avg"] / 1e4, "-o", ms=3, lw=1.4,
                    label=f"PINN lookup (T_in={args.T_in:.0f} K, lam={args.lambda_air:.0f}, FU={args.FU:.1f})")
    axes[0, 0].scatter([r["V"] for r in ref], [r["i_avg_ref"] / 1e4 for r in ref],
                       marker="s", s=48, zorder=3, facecolor="none", edgecolor="tab:red",
                       label="COMSOL DOE (i_avg)")
    axes[0, 0].set_xlabel("V_cell [V]"); axes[0, 0].set_ylabel("i_avg [10^4 A/m^2]")
    axes[0, 0].set_title("(a) I-V curve: PINN vs COMSOL")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(v, df["P_density_W_m2"] / 1e3, "-o", ms=3, lw=1.4, color="tab:green")
    axes[0, 1].set_xlabel("V_cell [V]"); axes[0, 1].set_ylabel("P_density [kW/m^2]")
    axes[0, 1].set_title("(b) P-V curve")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(v, df["eta_el"] * 100.0, "-o", ms=3, lw=1.4, color="tab:purple")
    axes[1, 0].set_xlabel("V_cell [V]"); axes[1, 0].set_ylabel("Electrical efficiency [%]")
    axes[1, 0].set_title("(c) Efficiency vs V_cell")
    axes[1, 0].grid(alpha=0.3)

    ax = axes[1, 1]
    if df["Tmax_K"].notna().any():
        ax.plot(v, df["Tmax_K"], "-o", ms=3, lw=1.4, color="tab:red", label="Tmax (PINN LUT)")
        ax.set_ylabel("Tmax [K]", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax.twinx()
        ax2.plot(v, df["phi_drop_v"], "--s", ms=3, lw=1.0, color="tab:blue", label="phi_drop")
        ax2.set_ylabel("phi_drop [V]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax.axhline(1123.0, color="k", ls=":", lw=1.0)
        ax.annotate("1123 K", xy=(0.76, 1124), fontsize=8, color="k")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Tmax: LUT csv not found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("V_cell [V]")
    ax.set_title("(d) Tmax & phi_drop vs V_cell")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Cell polarization curves - PINN {ver} (qav inversion)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, os.path.join(out, f"fig_iv_curves_{ver}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

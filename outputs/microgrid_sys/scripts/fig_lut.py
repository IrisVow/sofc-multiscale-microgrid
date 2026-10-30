# -*- coding: utf-8 -*-
"""F2 fig_lut: coupling lookup table diagnostics (continuity, latency, Tmax vs V).

Reads results/coupling/lut_check_{v19,v20}.json, pinn_lut_{ver}.csv and
pinn_lut_{ver}_summary.json; exports a comparison csv and plots
(a) Tmax & phi_drop vs V, (b) continuity linear vs nearest (v19/v20),
(c) LUT latency.

Usage (cwd = microgrid_sys):
    python scripts/fig_lut.py --version v20
Outputs (results/figures/<version>/):
    lut_compare_<version>.csv
    fig_lut_<version>.png
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
RES_COUPLING = os.path.join(ROOT, "results", "coupling")


def _load_json(p):
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scripts.fig_common import current_version
    ver = args.version or current_version()
    out = args.out or outdir(ver)
    style()

    l20 = _load_json(os.path.join(RES_COUPLING, f"lut_check_{ver}.json"))
    l19 = _load_json(os.path.join(RES_COUPLING, "lut_check_v19.json"))
    if l20 is None:
        raise SystemExit(f"[fig_lut] lut_check_{ver}.json not found - run check_lut.py first")
    summ = _load_json(os.path.join(RES_COUPLING, f"pinn_lut_{ver}_summary.json")) or {}

    rows = []
    for v, d in [("v19", l19), (ver, l20)]:
        if not d:
            continue
        cont = d.get("continuity", {})
        rows.append({
            "version": v, "all_ok": d.get("all_ok"),
            "linear_max_jump_k": cont.get("linear_max_jump_k"),
            "nearest_max_jump_k": cont.get("nearest_max_jump_k"),
            "latency_ms": next((c.get("detail") for c in d.get("checks", [])
                                if "0.1 ms" in c.get("name", "")), None),
        })
    cmp = pd.DataFrame(rows)
    csv_path = os.path.join(out, f"lut_compare_{ver}.csv")
    write_csv(cmp, csv_path)
    print(f"[fig_lut] exported {csv_path}")

    lut_csv = os.path.join(RES_COUPLING, f"pinn_lut_{ver}.csv")
    lut = pd.read_csv(lut_csv) if os.path.exists(lut_csv) else None

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))

    ax = axes[0, 0]
    if lut is not None:
        nom = lut[(lut["lambda_air"] == 8.0) & (lut["FU"] == 0.6)].sort_values("V")
        ax.plot(nom["V"], nom["tmax_k"], "-o", ms=3, lw=1.4, color="tab:red", label="Tmax (lam=8, FU=0.6)")
        ax.axhline(1123.0, color="k", ls=":", lw=1.0)
        ax.set_ylabel("Tmax [K]", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax.twinx()
        ax2.plot(nom["V"], nom["phi_drop_v"], "--s", ms=3, lw=1.0, color="tab:blue", label="phi_drop")
        ax2.set_ylabel("phi_drop [V]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="lower left")
    ax.set_xlabel("V_cell [V]")
    ax.set_title(f"(a) LUT Tmax & phi_drop vs V (nominal, {ver})")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if len(cmp):
        x = np.arange(len(cmp))
        w = 0.36
        ax.bar(x - w / 2, cmp["linear_max_jump_k"], w, label="linear", color="tab:green")
        ax.bar(x + w / 2, cmp["nearest_max_jump_k"], w, label="nearest", color="tab:gray")
        ax.set_xticks(x); ax.set_xticklabels(cmp["version"])
        ax.set_yscale("log")
        ax.set_ylabel("max adjacent jump [K] (log)")
        ax.set_title("(b) Tmax continuity: linear vs nearest")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    lat = None
    for d in [l20]:
        if d:
            for c in d.get("checks", []):
                if "0.1 ms" in c.get("name", ""):
                    m = c.get("detail", "")
                    try:
                        lat = float(m.split("=")[-1].replace(" ms/次", "").replace(" ms", ""))
                    except Exception:
                        lat = None
    if lat is not None:
        ax.bar(["LUT query"], [lat], color="tab:purple", width=0.45)
        ax.axhline(0.1, color="k", ls="--", lw=1.0)
        ax.annotate("gate 0.1 ms", xy=(0, 0.105), fontsize=8)
        ax.set_ylabel("ms/query")
        ax.set_title(f"(c) LUT latency: {lat:.4f} ms (<< 10 ms EMS step)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    tr = summ.get("tmax_range_k", [])
    g = summ.get("grid", "?")
    txt = f"LUT grid: {g}\n"
    txt += f"cases: {summ.get('n_grid', '?')}\n"
    txt += f"probes/domain: {summ.get('n_probes', '?')}\n"
    if len(tr) == 2:
        txt += f"Tmax range: {tr[0]:.1f} - {tr[1]:.1f} K\n"
    txt += f"out-of-domain probe pts: {summ.get('n_out_of_domain_probe_points', '?')}\n"
    tv = summ.get("tmax_v_nominal", [])
    if tv:
        txt += f"Tmax @V 0.75/1.0 (nominal): {tv[0]:.1f} / {tv[-1]:.1f} K"
    ax.text(0.06, 0.94, txt, transform=ax.transAxes, va="top", fontsize=10,
            family="monospace", bbox=dict(boxstyle="round", fc="lightyellow"))
    ax.axis("off")
    ax.set_title("(d) LUT coverage summary")

    fig.suptitle(f"Coupling lookup table diagnostics - {ver}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, os.path.join(out, f"fig_lut_{ver}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

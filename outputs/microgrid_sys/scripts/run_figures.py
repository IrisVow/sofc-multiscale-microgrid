# -*- coding: utf-8 -*-
"""F2 figure pipeline orchestrator (version-parameterized).

Runs all fig_*.py scripts for a model version, verifies outputs and writes a
manifest json into results/figures/<version>/.

Usage (cwd = microgrid_sys):
    python scripts/run_figures.py --version v20
    python scripts/run_figures.py --version v20 --skip metrics
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURES = {
    "iv": "fig_iv_curves.py",
    "fields": "fig_thermal_fields.py",
    "metrics": "fig_metrics.py",
    "shock": "fig_thermal_shock.py",
    "ems": "fig_ems_dispatch.py",
    "lut": "fig_lut.py",
}
PNG_PREFIX = {
    "iv": "fig_iv_curves",
    "fields": "fig_thermal_fields",
    "metrics": "fig_metrics",
    "shock": "fig_thermal_shock",
    "ems": "fig_ems_dispatch",
    "lut": "fig_lut",
}

# Static figures: editable draw.io source is authoritative; a preview PNG/SVG is
# rendered by scripts/fig1_render.py when missing (matplotlib, no draw.io CLI needed).
STATIC_FIGURES = {
    "fig1_architecture": {
        "script": "docs/figures/fig1_architecture.drawio",
        "png": ["fig1_architecture.png"],
        "renderer": "scripts/fig1_render.py",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--skip", default="", help="comma-separated figure keys to skip")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from scripts.fig_common import current_version, outdir
    ver = args.version or current_version()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    out = outdir(ver)
    os.makedirs(out, exist_ok=True)

    manifest = {"version": ver, "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "figures": {}}
    for key, script in FIGURES.items():
        if key in skip:
            print(f"[run_figures] skip {key}")
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "scripts", script),
                            "--version", ver], cwd=ROOT)
        dt = time.time() - t0
        ok = p.returncode == 0
        manifest["figures"][key] = {"script": script, "ok": ok, "seconds": round(dt, 1)}
        print(f"[run_figures] {key}: {'OK' if ok else 'FAIL'} ({dt:.1f} s)")
        if not ok:
            sys.exit(f"[run_figures] {key} failed")

    # verify outputs
    for key, f in FIGURES.items():
        if key in skip:
            continue
        base = PNG_PREFIX[key]
        found = sorted(fn for fn in os.listdir(out) if fn.startswith(base) and fn.endswith(".png"))
        manifest["figures"][key]["png"] = found

    # static figures: render a preview if missing, then register
    for key, info in STATIC_FIGURES.items():
        pngs = [fn for fn in info["png"] if os.path.exists(os.path.join(out, fn))]
        renderer = info.get("renderer")
        if not pngs and renderer:
            rp = os.path.join(ROOT, renderer)
            if os.path.exists(rp):
                r = subprocess.run(
                    [sys.executable, "-u", rp, "--drawio",
                     os.path.join(ROOT, info["script"]), "--out", out],
                    cwd=ROOT)
                if r.returncode == 0:
                    pngs = [fn for fn in info["png"] if os.path.exists(os.path.join(out, fn))]
        manifest["figures"][key] = {
            "script": info["script"],
            "ok": bool(pngs),
            "seconds": 0.0,
            "png": pngs,
        }

    mp = os.path.join(out, "figures_manifest.json")
    with io.open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[run_figures] manifest: {mp}")
    print(f"[run_figures] all figures done for {ver} -> {out}")


if __name__ == "__main__":
    main()

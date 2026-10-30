# -*- coding: utf-8 -*-
"""F2 figure pipeline: shared helpers (version resolution, ONNX paths, output dirs, style).

English-only plot labels; data is always exported (csv/json) before plotting.
"""
from __future__ import annotations

import io
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # microgrid_sys
PINN_ROOT = os.path.normpath(os.path.join(ROOT, "..", "pinn_sofc"))
# pinndata/ lives at the repository root (sibling of microgrid_sys/ and pinn_sofc/)
P_DATA = os.path.normpath(os.path.join(ROOT, "..", "pinndata"))

ONNX_CANDIDATES = {
    "v18": os.path.join(PINN_ROOT, "runs", "stage3_pde_v18", "pinn_sofc_v18.onnx"),
    "v19": os.path.join(PINN_ROOT, "runs", "stage3_pde_v19", "pinn_sofc_v19.onnx"),
    "v20": os.path.join(PINN_ROOT, "runs", "stage3_pde_v20", "pinn_sofc_v20.onnx"),
}

RUN_DIRS = {v: os.path.join(PINN_ROOT, "runs", "stage3_pde_" + v) for v in ONNX_CANDIDATES}

# 5-domain order (matches domain_guard.DOMAIN_ORDER and export_onnx)
DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]


def load_coupling():
    import yaml
    with io.open(os.path.join(ROOT, "configs", "coupling.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def current_version() -> str:
    cfg = load_coupling()
    p = str(cfg["pinn"]["onnx_path"])
    m = re.search(r"stage3_pde_?(v\d+)", p)
    return m.group(1) if m else "v20"


def resolve_onnx(version: str | None = None) -> str:
    version = version or current_version()
    p = ONNX_CANDIDATES.get(version)
    if p and os.path.exists(p):
        return p
    d = RUN_DIRS.get(version)
    if d and os.path.isdir(d):
        cand = sorted(f for f in os.listdir(d)
                      if f.endswith(".onnx") and "best" not in f and "final" not in f)
        if cand:
            return os.path.join(d, cand[0])
    return p


def resolve_run_dir(version: str | None = None) -> str:
    version = version or current_version()
    return RUN_DIRS.get(version, os.path.join(PINN_ROOT, "runs", "stage3_pde_" + version))


def outdir(version: str | None = None) -> str:
    d = os.path.join(ROOT, "results", "figures", version or current_version())
    os.makedirs(d, exist_ok=True)
    return d


def style():
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save_json(obj, path):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8")


def savefig(fig, path):
    fig.savefig(path)
    print("saved:", path)

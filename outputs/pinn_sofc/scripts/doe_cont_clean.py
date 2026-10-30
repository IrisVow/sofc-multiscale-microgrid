# -*- coding: utf-8 -*-
"""B4-cont transient DOE cleaning: continuous load trajectories (R1-R4).
Same rules as doe_b4_clean (35-col COMSOL export, dedup, rO2 sign, domain by coverage).
meta: type=transient, stepT_s=NaN (no single step), V_applied from analytic wave V(t).

Usage:
  python scripts/doe_cont_clean.py --validate
  python scripts/doe_cont_clean.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pinn_sofc
OUTPUTS = os.path.dirname(BASE)                                       # outputs
WORK = os.path.join(os.path.dirname(OUTPUTS), "work")                 # .../work
DOE = os.path.join(OUTPUTS, "pinndata", "doe")
DOE_B4 = os.path.join(OUTPUTS, "pinndata", "doe_b4")
DOE_CONT = os.path.join(OUTPUTS, "pinndata", "doe_cont")

COLS35 = ["x", "y", "T", "u", "v", "p", "w1", "w2", "w3", "w4", "phis", "phil",
          "rho", "CH2", "CH2O", "CO2", "CN2", "htQ", "isx", "isy", "ilx", "ily",
          "iloc_pcar1", "iloc_panr1", "eta_pcar1", "eta_panr1", "Eeq_pcar1",
          "Eeq_panr1", "ilim_pcar1", "ilim_panr1", "qav_an", "qav_ca",
          "rH2", "rH2O", "rO2"]

DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]

# tag -> (V_cell, T_in_K, lambda_air, FU, amp_V, wave_type)
CONT_META = {
    "R1": (1.0, 1073.0, 8.0, 0.6, 0.20, "ramp"),
    "R2": (1.0, 1073.0, 8.0, 0.6, 0.25, "multistep"),
    "R3": (1.0, 1073.0, 8.0, 0.6, 0.20, "sine"),
    "R4": (1.0, 1073.0, 8.0, 0.6, 0.25, "random"),
}


def v_applied(t: float, amp: float, wtype: str) -> float:
    if wtype == "ramp":
        return 1.0 - amp * min(t / 600.0, 1.0)
    if wtype == "multistep":
        w = 0.0
        if t >= 200.0: w = 0.2
        if t >= 400.0: w = 0.6
        if t >= 600.0: w = 1.0
        if t >= 800.0: w = 0.4
        if t >= 1000.0: w = 0.8
        return 1.0 - amp * w
    if wtype == "sine":
        return 1.0 - amp * (0.5 + 0.5 * math.sin(2 * math.pi * t / 600.0))
    if wtype == "random":
        w = 0.0
        if t >= 150.0: w = 0.6
        if t >= 300.0: w = 0.2
        if t >= 500.0: w = 1.0
        if t >= 700.0: w = 0.4
        if t >= 900.0: w = 0.8
        if t >= 1100.0: w = 0.6
        return 1.0 - amp * w
    raise ValueError(wtype)


def parse_raw(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    n_comment = 0
    for l in lines:
        if l.startswith("%"):
            n_comment += 1
        else:
            break
    df = pd.read_csv(path, comment="%", encoding="utf-8", header=None,
                     skiprows=n_comment, names=COLS35, skipinitialspace=True)
    return df


def assign_domain(row: pd.Series) -> str:
    if pd.notna(row["iloc_pcar1"]):
        return "air_electrode"
    if pd.notna(row["iloc_panr1"]):
        return "fuel_electrode"
    if pd.notna(row["phil"]):
        return "electrolyte"
    if pd.notna(row["u"]):
        return "air_channel" if row["y"] < 1.3e-3 else "fuel_channel"
    return "unknown"


def clean_snapshot(path: str, case_id: str, t_s: float) -> pd.DataFrame:
    df = parse_raw(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["rO2"] = -df["rO2"]
    df["domain"] = df.apply(assign_domain, axis=1)
    assert (df["domain"] != "unknown").all(), f"{path}: unknown domain"
    v, tin, lam, fu, amp, wtype = CONT_META[case_id]
    meta = {"case_id": case_id, "type": "transient", "t_s": float(t_s), "V_cell": v,
            "T_in_K": tin, "lambda_air": lam, "FU": fu, "stepT_s": np.nan,
            "wave_type": wtype, "V_applied": float(v_applied(t_s, amp, wtype))}
    for k in reversed(list(meta)):
        df.insert(0, k, meta[k])
    return df


def t_from_name(fname: str) -> float:
    base = os.path.basename(fname)
    m = re.search(r"_t(\d+)s\.csv$", base)
    if m:
        return float(int(m.group(1)))
    if base.endswith("_steady.csv"):
        return 0.0
    raise ValueError("cannot parse t from " + base)


def clean_case(raw_dir: str, case_id: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"sofc1_doe_{case_id}_*.csv")))
    files = [f for f in files if not os.path.basename(f).endswith("_steady.csv")]
    assert files, f"no time-step raws for {case_id} in {raw_dir}"
    frames = []
    for f in files:
        frames.append(clean_snapshot(f, case_id, t_from_name(f)))
    df = pd.concat(frames, ignore_index=True)
    return df


def validate() -> None:
    raw_root = os.path.join(WORK, "doe_cont", "transient")
    for cid in CONT_META:
        df = clean_case(os.path.join(raw_root, cid), cid)
        per_snap = df.groupby("t_s").size()
        assert (per_snap == 15567).all(), f"{cid}: snapshot rows != 15567: {dict(per_snap)}"
        assert (df["domain"] != "unknown").all(), f"{cid}: unknown domain"
        v = df["V_applied"].to_numpy()
        assert np.isfinite(v).all() and (v >= 0.7).all() and (v <= 1.01).all(), f"{cid}: V_applied range"
        print(f"[validate] {cid}: {len(df):,} rows, {len(per_snap)} snapshots x {int(per_snap.iloc[0])} rows, "
              f"V_applied {v.min():.3f}-{v.max():.3f} OK")


def build() -> None:
    out_clean = os.path.join(DOE_CONT, "cleaned", "transient")
    out_merge = os.path.join(DOE_CONT, "merged")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_merge, exist_ok=True)
    raw_root = os.path.join(WORK, "doe_cont", "transient")
    frames = []
    for cid in CONT_META:
        df = clean_case(os.path.join(raw_root, cid), cid)
        frames.append(df)
        df.to_parquet(os.path.join(out_clean, f"{cid}.parquet"), index=False)
        print(f"[build] {cid}: {len(df):,} rows -> cleaned/transient/{cid}.parquet")
    cont = pd.concat(frames, ignore_index=True)
    cont.to_parquet(os.path.join(out_merge, "doe_transient_cont.parquet"), index=False)
    b4 = pd.read_parquet(os.path.join(DOE_B4, "merged", "doe_transient_b4.parquet"))
    orig = pd.read_parquet(os.path.join(DOE, "merged", "doe_transient.parquet"))
    comb = pd.concat([orig, b4, cont], ignore_index=True)
    comb.to_parquet(os.path.join(out_merge, "doe_transient_combined2.parquet"), index=False)
    print(f"[build] cont={len(cont):,} rows; combined2={len(comb):,} rows (14 cases: T0-T9 + R1-R4)")
    cond = {"model_source": "sofc1_FINAL2_solved.mph (COMSOL 6.3)",
            "note": "B4-cont continuous load trajectories: ramp(R1) / multistep(R2) / sine(R3) / random(R4), "
                    "V 0.70-1.0 V window 1200 s; stepT_s=NaN; cleaning identical to transient DOE",
            "cases": {cid: {"V_cell": c[0], "T_in_K": c[1], "lambda_air": c[2], "FU": c[3],
                            "amp_V": c[4], "wave_type": c[5]} for cid, c in CONT_META.items()}}
    with open(os.path.join(DOE_CONT, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(cond, f, ensure_ascii=False, indent=2)
    print("[build] conditions.json written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        build()


if __name__ == "__main__":
    main()
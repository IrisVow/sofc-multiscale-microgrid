# -*- coding: utf-8 -*-
"""B4 transient DOE cleaning: larger V-step trajectories (T4-T9).
Rules identical to original DOE transient cleaning (validated by reproducing T0):
1. skip '%' comment lines; 35-column header by position;
2. drop fully-duplicate rows -> 15,567 rows/snapshot;
3. rO2 = -rO2; domain by variable coverage;
4. meta: type=transient, t_s from filename, V_applied = V_cell - stepAmp if t_s >= stepT else V_cell.

Usage:
  python scripts/doe_b4_clean.py --validate   # reproduce T0 and compare to existing cleaned T0.parquet
  python scripts/doe_b4_clean.py              # clean T4-T9 -> pinndata/doe_b4/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pinn_sofc
OUTPUTS = os.path.dirname(BASE)                                       # outputs
WORK = os.path.join(os.path.dirname(OUTPUTS), "work")                 # .../work
DOE = os.path.join(OUTPUTS, "pinndata", "doe")
DOE_B4 = os.path.join(OUTPUTS, "pinndata", "doe_b4")

COLS35 = ["x", "y", "T", "u", "v", "p", "w1", "w2", "w3", "w4", "phis", "phil",
          "rho", "CH2", "CH2O", "CO2", "CN2", "htQ", "isx", "isy", "ilx", "ily",
          "iloc_pcar1", "iloc_panr1", "eta_pcar1", "eta_panr1", "Eeq_pcar1",
          "Eeq_panr1", "ilim_pcar1", "ilim_panr1", "qav_an", "qav_ca",
          "rH2", "rH2O", "rO2"]

META_BASE = ["case_id", "type", "t_s", "V_cell", "T_in_K", "lambda_air", "FU",
             "stepT_s", "stepAmp_V", "V_applied"]
DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]

# tag -> (V_cell, T_in_K, lambda_air, FU, stepT_s, stepAmp_V)
VALIDATE_META = {
    "T0": (1.0, 1073.0, 8.0, 0.6, 100.0, 0.10),
    "T1": (1.0, 1073.0, 8.0, 0.6, 50.0, 0.10),
    "T2": (1.0, 1073.0, 8.0, 0.6, 200.0, 0.10),
    "T3": (1.0, 1073.0, 8.0, 0.6, 100.0, 0.20),
}

CASE_META = {
    "T4": (1.0, 1073.0, 8.0, 0.6, 100.0, 0.15),
    "T5": (1.0, 1073.0, 8.0, 0.6, 200.0, 0.15),
    "T6": (1.0, 1073.0, 8.0, 0.6, 50.0, 0.15),
    "T7": (1.0, 1073.0, 8.0, 0.6, 100.0, 0.20),
    "T8": (1.0, 1073.0, 8.0, 0.6, 200.0, 0.20),
    "T9": (1.0, 1073.0, 8.0, 0.6, 50.0, 0.20),
}


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
    meta_src = CASE_META if case_id in CASE_META else VALIDATE_META
    v, t, lam, fu, stepT, amp = meta_src[case_id]
    v_app = v - amp if t_s >= stepT else v
    meta = {"case_id": case_id, "type": "transient", "t_s": float(t_s), "V_cell": v,
            "T_in_K": t, "lambda_air": lam, "FU": fu, "stepT_s": float(stepT),
            "stepAmp_V": float(amp), "V_applied": float(v_app)}
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
    """Reproduce T0 from original raw and compare with existing cleaned T0.parquet."""
    raw_dir = os.path.join(DOE, "raw", "transient", "T0")
    df = clean_case(raw_dir, "T0")
    ref = pd.read_parquet(os.path.join(DOE, "cleaned", "transient", "T0.parquet"))
    df = df.sort_values(["domain", "t_s", "x", "y"]).reset_index(drop=True)
    ref = ref.sort_values(["domain", "t_s", "x", "y"]).reset_index(drop=True)
    assert df.shape == ref.shape, f"shape {df.shape} vs {ref.shape}"
    assert list(df.columns) == list(ref.columns), "cols differ"
    bad = []
    for c in ref.columns:
        if not pd.api.types.is_numeric_dtype(ref[c].dtype):
            if not df[c].equals(ref[c]):
                bad.append(c)
            continue
        a = df[c].to_numpy(dtype=np.float64)
        b = ref[c].to_numpy(dtype=np.float64)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            bad.append(c)
            continue
        both = np.isfinite(a) & np.isfinite(b)
        if both.any() and not np.allclose(a[both], b[both], rtol=1e-10, atol=1e-12):
            bad.append(c)
    if bad:
        raise SystemExit(f"[validate] T0 MISMATCH cols {bad}")
    print(f"[validate] T0 OK ({len(df):,} rows, {len(df.columns)} cols, per-point identical)")


def build() -> None:
    out_clean = os.path.join(DOE_B4, "cleaned", "transient")
    out_merge = os.path.join(DOE_B4, "merged")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_merge, exist_ok=True)
    raw_root = os.path.join(WORK, "doe_b4", "transient")
    frames = []
    for cid in CASE_META:
        df = clean_case(os.path.join(raw_root, cid), cid)
        frames.append(df)
        df.to_parquet(os.path.join(out_clean, f"{cid}.parquet"), index=False)
        print(f"[build] {cid}: {len(df):,} rows -> cleaned/transient/{cid}.parquet")
    b4 = pd.concat(frames, ignore_index=True)
    b4.to_parquet(os.path.join(out_merge, "doe_transient_b4.parquet"), index=False)
    # combined: original T0-T3 + B4 T4-T9
    orig = pd.read_parquet(os.path.join(DOE, "merged", "doe_transient.parquet"))
    comb = pd.concat([orig, b4], ignore_index=True)
    comb.to_parquet(os.path.join(out_merge, "doe_transient_combined.parquet"), index=False)
    print(f"[build] b4={len(b4):,} rows; combined={len(comb):,} rows ({len(CASE_META)+4} cases)")
    cond = {"model_source": "sofc1_FINAL2_solved.mph (COMSOL 6.3)",
            "note": "B4 transient DOE: larger V-step amplitudes 0.15/0.20 V x stepT {50,100,200}; cleaning identical to original transient DOE",
            "cases": {cid: {"V_cell": c[0], "T_in_K": c[1], "lambda_air": c[2], "FU": c[3],
                            "stepT_s": c[4], "stepAmp_V": c[5]} for cid, c in CASE_META.items()}}
    with open(os.path.join(DOE_B4, "conditions.json"), "w", encoding="utf-8") as f:
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

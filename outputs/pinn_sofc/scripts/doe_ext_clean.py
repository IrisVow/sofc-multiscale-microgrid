# -*- coding: utf-8 -*-
"""B1 DOE 扩展数据清洗管线（与原始 DOE 管线一致，经验证可逐点复现 S1-S9）。

规则（已在 raw/steady/sofc1_doe_S1_*.csv 上验证，cleaned 计数与数值完全一致）：
1. 跳过 '%' 注释行；表头为 'x,y,<33 表达式>'；
2. 删除完全重复行（35 列全同）-> 每工况 15,567 行；
3. 按变量覆盖定域（与 split.py DOMAIN_ORDER 一致）；
4. 追加元数据列并合并。

用法：
  python scripts/doe_ext_clean.py --validate   # 复现 S1-S9 并与既有 parquet 对比
  python scripts/doe_ext_clean.py              # 处理 E1-E3 -> doe_ext/cleaned + merged
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pinn_sofc
OUTPUTS = os.path.dirname(BASE)                                     # outputs
DOE = os.path.join(OUTPUTS, "pinndata", "doe")
DOE_EXT = os.path.join(OUTPUTS, "pinndata", "doe_ext")

# 35 列表头（x,y + 33 表达式），与 COMSOL Data 导出顺序一致
COLS35 = ["x", "y", "T", "u", "v", "p", "w1", "w2", "w3", "w4", "phis", "phil",
          "rho", "CH2", "CH2O", "CO2", "CN2", "htQ", "isx", "isy", "ilx", "ily",
          "iloc_pcar1", "iloc_panr1", "eta_pcar1", "eta_panr1", "Eeq_pcar1",
          "Eeq_panr1", "ilim_pcar1", "ilim_panr1", "qav_an", "qav_ca",
          "rH2", "rH2O", "rO2"]

META_COLS = ["case_id", "type", "t_s", "V_cell", "T_in_K", "lambda_air", "FU",
             "stepT_s", "stepAmp_V", "V_applied"]

DOMAIN_ORDER = ["air_channel", "air_electrode", "electrolyte", "fuel_electrode", "fuel_channel"]

# case_id -> (V_cell, T_in_K, lambda_air, FU)
CASE_META = {
    "S1": (1.00, 1073.0, 8.0, 0.6), "S2": (0.95, 1073.0, 8.0, 0.6),
    "S3": (0.90, 1073.0, 8.0, 0.6), "S4": (0.85, 1073.0, 8.0, 0.6),
    "S5": (0.80, 1073.0, 8.0, 0.6), "S6": (0.75, 1073.0, 8.0, 0.6),
    "S7": (0.90, 1073.0, 4.0, 0.6), "S8": (0.90, 1073.0, 8.0, 0.8),
    "S9": (0.90, 1123.0, 8.0, 0.6),
    "E1": (0.75, 1123.0, 8.0, 0.6), "E2": (0.80, 1123.0, 8.0, 0.6),
    "E3": (0.85, 1123.0, 8.0, 0.6),
    "F1": (0.90, 1123.0, 4.0, 0.6), "F2": (0.90, 1123.0, 6.0, 0.6),
    "F3": (0.85, 1123.0, 4.0, 0.6), "F4": (0.75, 1123.0, 4.0, 0.6),
    "F5": (0.85, 1123.0, 6.0, 0.6), "F6": (0.75, 1123.0, 6.0, 0.6),
    "F7": (0.90, 1123.0, 8.0, 0.7), "F8": (0.90, 1123.0, 8.0, 0.8),
    "F9": (0.85, 1123.0, 8.0, 0.8), "F10": (0.75, 1123.0, 8.0, 0.8),
    "F11": (0.90, 1073.0, 6.0, 0.6), "F12": (0.90, 1073.0, 8.0, 0.7),
}


def parse_raw(path: str) -> pd.DataFrame:
    """解析 COMSOL Data 导出 CSV -> 35 列 DataFrame（含重复行）。"""
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
    """按变量覆盖定域：iloc_pcar1->air_el, iloc_panr1->fuel_el, phil->ely, u->channel。"""
    if pd.notna(row["iloc_pcar1"]):
        return "air_electrode"
    if pd.notna(row["iloc_panr1"]):
        return "fuel_electrode"
    if pd.notna(row["phil"]):
        return "electrolyte"
    if pd.notna(row["u"]):
        return "air_channel" if row["y"] < 1.3e-3 else "fuel_channel"
    return "unknown"


def clean_case(path: str, case_id: str) -> pd.DataFrame:
    df = parse_raw(path)
    df = df.drop_duplicates().reset_index(drop=True)
    # 原始 DOE 清洗对 rO2 做了物理符号修正（O2 消耗为负）；rH2/rH2O 符号已正确
    df["rO2"] = -df["rO2"]
    df["domain"] = df.apply(assign_domain, axis=1)
    assert (df["domain"] != "unknown").all(), f"{case_id}: unknown domain rows"
    v, t, lam, fu = CASE_META[case_id]
    meta = {"case_id": case_id, "type": "steady", "t_s": 0.0, "V_cell": v,
            "T_in_K": t, "lambda_air": lam, "FU": fu, "stepT_s": float("nan"),
            "stepAmp_V": float("nan"), "V_applied": v}
    for k in reversed(list(meta)):
        df.insert(0, k, meta[k])
    return df


def validate_s_cases() -> None:
    """用 S1-S9 原始 CSV 复现清洗结果并与既有 parquet 逐列对比。"""
    n_bad = 0
    for cid in ["S%d" % i for i in range(1, 10)]:
        raw = os.path.join(DOE, "raw", "steady", f"sofc1_doe_{cid}_*.csv")
        import glob
        matches = glob.glob(raw)
        assert matches, f"no raw for {cid}"
        df = clean_case(matches[0], cid).sort_values(["domain", "x", "y"]).reset_index(drop=True)
        ref = pd.read_parquet(os.path.join(DOE, "cleaned", "steady", f"{cid}.parquet"))
        ref = ref.sort_values(["domain", "x", "y"]).reset_index(drop=True)
        assert df.shape == ref.shape, f"{cid}: shape {df.shape} vs {ref.shape}"
        assert list(df.columns) == list(ref.columns), f"{cid}: cols differ"
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
            n_bad += 1
            print(f"[validate] {cid}: MISMATCH cols {bad}")
        else:
            print(f"[validate] {cid}: OK (15,567 行，46 列，逐点一致)")
    if n_bad:
        sys.exit(f"[validate] {n_bad} 个工况不一致")
    print("[validate] S1-S9 全部复现成功")


def build_ext() -> None:
    """清洗 E1-E3 -> doe_ext/cleaned/steady + merged/doe_steady_ext.parquet。"""
    out_clean = os.path.join(DOE_EXT, "cleaned", "steady")
    out_merge = os.path.join(DOE_EXT, "merged")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_merge, exist_ok=True)
    ext_ids = [cid for cid in CASE_META if not cid.startswith("S")]
    frames = []
    for cid in ext_ids:
        import glob
        raw = glob.glob(os.path.join(DOE_EXT, "raw", "steady", f"sofc1_doe_{cid}_*.csv"))
        assert raw, f"no raw for {cid}"
        df = clean_case(raw[0], cid)
        frames.append(df)
        df.to_parquet(os.path.join(out_clean, f"{cid}.parquet"), index=False)
        print(f"[build] {cid}: {len(df)} 行 -> cleaned/steady/{cid}.parquet")
    # 合并：仅 E 工况（训练集新增部分）
    ext = pd.concat(frames, ignore_index=True)
    ext.to_parquet(os.path.join(out_merge, "doe_steady_ext.parquet"), index=False)
    # 合并：全稳态（S1-S9 + E1-E3），供 v11 训练/验证/测试切分
    all_frames = []
    for cid in list(CASE_META):
        import glob
        if cid.startswith("S"):
            raw = glob.glob(os.path.join(DOE, "raw", "steady", f"sofc1_doe_{cid}_*.csv"))[0]
        else:
            raw = glob.glob(os.path.join(DOE_EXT, "raw", "steady", f"sofc1_doe_{cid}_*.csv"))[0]
        all_frames.append(clean_case(raw, cid))
    merged = pd.concat(all_frames, ignore_index=True)
    merged.to_parquet(os.path.join(out_merge, "doe_steady_combined.parquet"), index=False)
    print(f"[build] merged: ext={len(ext)} 行, combined={len(merged)} 行 ({len(CASE_META)} 工况)")
    # conditions.json
    import json
    cond = {"model_source": "sofc1_fixed.mph (COMSOL 6.3)",
            "notes": "E1-E3 (T_in=1123K x V 0.75/0.8/0.85) + F1-F12 (lambda_air/FU boundary pts) as B1 additions; cleaning identical to original DOE",
            "cases": {cid: {"V_cell": c[0], "T_in_K": c[1], "lambda_air": c[2], "FU": c[3]}
                      for cid, c in CASE_META.items()}}
    with open(os.path.join(DOE_EXT, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(cond, f, ensure_ascii=False, indent=2)
    print("[build] conditions.json 已写出")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="复现并校验 S1-S9")
    args = ap.parse_args()
    if args.validate:
        validate_s_cases()
    else:
        build_ext()


if __name__ == "__main__":
    main()
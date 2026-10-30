# -*- coding: utf-8 -*-
"""M5 回归门槛检查（阶段 A4）：对 M4b 基线关键指标做阈值回归。

用法：
    conda activate sofc
    python run_regression.py --metrics runs/stage3_pde_m4b/eval_metrics_best.csv

退出码：0 = 全部 PASS；1 = 存在 FAIL。
阈值对应 PLANNING.md 阶段 A4：steady_val T rel<=1.5%、S9 T rel<=5%、T3 T RMSE<=9 K。
"""
from __future__ import annotations

import argparse
import csv
import sys

# pool, var -> (指标, 阈值, 说明)
THRESHOLDS = [
    ("steady_val", "T", "rel_l2", 0.015, "稳态验证 T rel<=1.5%"),
    ("steady_test", "T", "rel_l2", 0.050, "S9 温度外推 T rel<=5%"),
    ("transient_test", "T", "rmse", 9.0, "T3 瞬态外推 T RMSE<=9 K"),
    ("transient_test", "T", "rel_l2", 0.010, "T3 瞬态外推 T rel<=1%"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/stage3_pde_m4b/eval_metrics_best.csv")
    ap.add_argument("--json-out", default=None, help="可选：结果写为 JSON")
    args = ap.parse_args()

    rows = {}
    with open(args.metrics, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[(r["pool"], r["var"])] = r

    fails = 0
    print(f"{'pool':<16}{'var':<6}{'metric':<8}{'value':>12}{'limit':>10}  verdict")
    for pool, var, metric, limit, desc in THRESHOLDS:
        key = (pool, var)
        if key not in rows:
            print(f"[WARN] 缺少 {pool}/{var} 行，跳过 {desc}")
            continue
        value = float(rows[key][metric])
        ok = value <= limit
        fails += 0 if ok else 1
        print(f"{pool:<16}{var:<6}{metric:<8}{value:>12.4g}{limit:>10.4g}  {'PASS' if ok else 'FAIL'}  ({desc})")

    if args.json_out:
        import json
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"pass": fails == 0, "fails": fails}, f, ensure_ascii=False, indent=2)

    print(f"\n结果: {'全部 PASS' if fails == 0 else str(fails) + ' 项 FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
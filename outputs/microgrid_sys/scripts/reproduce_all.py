# -*- coding: utf-8 -*-
"""F3 一键复现脚本（v20 定稿基线）。

串行执行阶段 F 复现链路，每步独立 subprocess（失败即停，结果摘要写
results/reproduce_<version>_summary.json）：

  1. PINN v20 评估 + A4 回归（pinn_sofc/，需 torch 环境；--retrain 时先重训）
  2. 耦合 LUT 4D 查找表（build_pinn_table.py -> data/pinn_lut_v20.npz）
  3. F2 图集 v20（run_figures.py -> results/figures/v20/ + figures_manifest.json）
  4. EMS 7 天调度 + Tmax 降载（compare_pinn_ems.py）
  5. 热冲击案例（thermal_shock_case.py -> results/thermal_shock/）
  6. 可选：E1 容量优化 + E2 论文对照 + E3 敏感性（--with-e，耗时长）

用法（cwd = microgrid_sys）：
    python scripts/reproduce_all.py
    python scripts/reproduce_all.py --with-e
    python scripts/reproduce_all.py --force        # 强制重跑已存在的评估/LUT
    python scripts/reproduce_all.py --retrain      # 先重训 v20 PINN（约 30 min+，热启动 v19 best.pt）
    python scripts/reproduce_all.py --no-shock     # 按需跳过步骤
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINN_ROOT = ROOT.parent / "pinn_sofc"


def run(name: str, cmd: list[str], cwd: Path) -> dict:
    print(f"\n[reproduce] ====== {name} ======")
    print(f"[reproduce] cwd: {cwd}\n[reproduce] $ {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd))
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[reproduce] {name}: {'OK' if ok else 'FAIL'} ({dt:.1f} s)")
    if not ok:
        raise SystemExit(f"[reproduce] 步骤失败：{name}")
    return {"ok": ok, "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v20", help="模型版本（默认 v20）")
    ap.add_argument("--with-e", action="store_true", help="包含 E1 容量优化 + E2 论文对照 + E3 敏感性")
    ap.add_argument("--retrain", action="store_true", help="先重训 PINN（热启动前一版本 best.pt）")
    ap.add_argument("--force", action="store_true", help="强制重跑已存在的评估/LUT 产物")
    for step in ("pinn", "lut", "figures", "ems", "shock", "e"):
        ap.add_argument(f"--no-{step}", action="store_true", help=f"跳过 {step}")
    args = ap.parse_args()
    ver = args.version
    py = sys.executable
    summary = {"version": ver, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    if not args.no_pinn:
        run_dir = PINN_ROOT / "runs" / f"stage3_pde_{ver}"
        if args.retrain:
            prev = f"stage3_pde_v{int(ver[1:]) - 1}"
            summary["steps"]["train"] = run(
                f"PINN 重训 {ver}",
                [py, "train.py", "--config", f"config_{ver}.yaml",
                 "--init", f"runs/{prev}/best.pt"],
                PINN_ROOT,
            )
        eval_csv = run_dir / "eval_metrics_final.csv"
        if args.force or not eval_csv.exists():
            summary["steps"]["evaluate"] = run(
                f"PINN 评估 {ver}",
                [py, "-u", "evaluate.py", "--run_dir", f"runs/stage3_pde_{ver}",
                 "--config", f"config_{ver}.yaml", "--ckpt", "final.pt",
                 "--pools", "steady_val", "steady_test", "transient_test"],
                PINN_ROOT,
            )
        else:
            print(f"[reproduce] 已有 {eval_csv}，跳过评估（--force 重跑）")
            summary["steps"]["evaluate"] = {"ok": True, "skipped": True}
        if eval_csv.exists():
            summary["steps"]["regression"] = run(
                "A4 回归门槛",
                [py, "-u", "run_regression.py", "--metrics", str(eval_csv)],
                PINN_ROOT,
            )

    if not args.no_lut:
        lut = ROOT / "data" / f"pinn_lut_{ver}.npz"
        if args.force or not lut.exists():
            summary["steps"]["lut"] = run(
                f"LUT 构建 {ver}",
                [py, "-u", "scripts/build_pinn_table.py", "--out", f"data/pinn_lut_{ver}.npz"],
                ROOT,
            )
        else:
            print(f"[reproduce] 已有 {lut}，跳过 LUT（--force 重跑）")
            summary["steps"]["lut"] = {"ok": True, "skipped": True}

    if not args.no_figures:
        summary["steps"]["figures"] = run(
            f"F2 图集 {ver}",
            [py, "-u", "scripts/run_figures.py", "--version", ver],
            ROOT,
        )

    if not args.no_ems:
        summary["steps"]["ems"] = run(
            "EMS 7 天调度 + Tmax 降载",
            [py, "-u", "scripts/compare_pinn_ems.py", "7"],
            ROOT,
        )

    if not args.no_shock:
        summary["steps"]["shock"] = run(
            "热冲击案例",
            [py, "-u", "scripts/thermal_shock_case.py"],
            ROOT,
        )

    if args.with_e and not args.no_e:
        summary["steps"]["e1_capacity"] = run(
            "E1 容量优化", [py, "-u", "scripts/run_capacity_opt.py", "--days", "7"], ROOT)
        summary["steps"]["e2_benchmark"] = run(
            "E2 论文对照", [py, "-u", "scripts/compare_paper_benchmarks.py"], ROOT)
        summary["steps"]["e3_sensitivity"] = run(
            "E3 敏感性", [py, "-u", "scripts/sensitivity_tornado.py"], ROOT)

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "results" / f"reproduce_{ver}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[reproduce] 完成：{out}")
    for k, v in summary["steps"].items():
        print(f"[reproduce]   {k:<16} {'OK' if v.get('ok') else 'SKIP'}")


if __name__ == "__main__":
    main()

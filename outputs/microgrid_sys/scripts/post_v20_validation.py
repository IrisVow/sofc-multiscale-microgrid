# -*- coding: utf-8 -*-
"""v20 验证管线（训练+eval 完成后一键执行，数据导出 + 英文图注脚本已就绪）。

流程：
  1. 等待 pinn_sofc/runs/stage3_pde_v20/pinn_sofc_v20.onnx 出现（训练+eval 完成）；
  2. build_pinn_table 生成 data/pinn_lut_v20.npz（grid 11x3x3x3，与 v19 同规格）；
  3. 备份 coupling.yaml -> coupling.v19_backup.yaml，切换 onnx/lut 到 v20；
  4. 运行 check_surrogate / check_lut / compare_pinn_ems / thermal_shock_case；
  5. 任一 FAIL -> 回滚 coupling.yaml（v19），写 FAIL marker；全过 -> 写 DONE marker + 摘要 json。
用法：python scripts/post_v20_validation.py [--wait-hours 10]
"""
from __future__ import annotations

from pathlib import Path as _ProjectPath
PROJECT_ROOT = next(p for p in _ProjectPath(__file__).resolve().parents if (p / ".sofc-project").is_file())

import argparse
import io
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = __import__('sys').executable
ONNX = (str(PROJECT_ROOT / 'outputs/pinn_sofc/runs/stage3_pde_v20/pinn_sofc_v20.onnx'))
CFG = os.path.join(ROOT, "configs", "coupling.yaml")
CFG_BACKUP = os.path.join(ROOT, "configs", "coupling.v19_backup.yaml")
OUT = os.path.join(ROOT, "results", "coupling")
MARKER = os.path.join(OUT, "v20_validation.marker")
SUMMARY = os.path.join(OUT, "v20_validation_summary.json")


def run(cmd, log, tag, tail=6):
    print(f"[{tag}] run: {cmd}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True)
    out_txt = (r.stdout or b"").decode("utf-8", errors="replace")
    err_txt = (r.stderr or b"").decode("utf-8", errors="replace")
    lines = out_txt.strip().splitlines()[-tail:]
    for ln in lines:
        print(f"    {ln}", flush=True)
    if r.returncode != 0:
        print(f"[{tag}] FAILED rc={r.returncode}", flush=True)
        for ln in err_txt.strip().splitlines()[-tail:]:
            print(f"    ERR {ln}", flush=True)
    return r.returncode


V19_ONNX = "../pinn_sofc/runs/stage3_pde_v19/pinn_sofc_v19.onnx"
V20_ONNX = "../pinn_sofc/runs/stage3_pde_v20/pinn_sofc_v20.onnx"


def switch_coupling(to_v20: bool):
    with io.open(CFG, "r", encoding="utf-8") as f:
        s = f.read()
    if to_v20:
        s = s.replace(V19_ONNX, V20_ONNX).replace("pinn_lut_v19.npz", "pinn_lut_v20.npz")
    else:
        s = s.replace(V20_ONNX, V19_ONNX).replace("pinn_lut_v20.npz", "pinn_lut_v19.npz")
    with io.open(CFG, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-hours", type=float, default=10.0)
    args = ap.parse_args()

    deadline = time.time() + args.wait_hours * 3600
    while not os.path.exists(ONNX) and time.time() < deadline:
        print(f"waiting for {ONNX} ...", flush=True)
        time.sleep(300)
    if not os.path.exists(ONNX):
        with io.open(MARKER, "w", encoding="utf-8") as f:
            f.write("fail_missing_onnx")
        print("FAIL: onnx not found within wait window")
        sys.exit(1)
    print("onnx found:", ONNX, flush=True)

    # 1) LUT
    rc = run(["scripts/build_pinn_table.py", "--onnx", ONNX,
              "--out", os.path.join("data", "pinn_lut_v20.npz"),
              "--cont-version", "10", "--grid", "11x3x3x3"], None, "lut")
    if rc != 0:
        with io.open(MARKER, "w", encoding="utf-8") as f:
            f.write("fail_lut")
        print("FAIL: LUT build")
        sys.exit(1)

    # 2) switch coupling.yaml (backup v19 first)
    shutil.copy2(CFG, CFG_BACKUP)
    switch_coupling(True)
    print("coupling.yaml -> v20 (backup: coupling.v19_backup.yaml)", flush=True)

    # 3) checks
    checks = {
        "check_surrogate": ["scripts/check_surrogate.py", "--onnx", ONNX, "--cont-version", "10",
                          "--ckpt", "final.pt"],
        "check_lut": ["scripts/check_lut.py"],
        "compare_pinn_ems": ["scripts/compare_pinn_ems.py", "7"],
        "thermal_shock_case": ["scripts/thermal_shock_case.py"],
    }
    results = {}
    all_ok = True
    for name, cmd in checks.items():
        rc = run(cmd, None, name)
        results[name] = "PASS" if rc == 0 else "FAIL"
        all_ok = all_ok and rc == 0

    summary = {
        "version": "v20",
        "onnx": ONNX,
        "coupling_switched": True,
        "checks": results,
        "all_pass": all_ok,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with io.open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not all_ok:
        switch_coupling(False)
        print("some checks FAILED -> coupling.yaml reverted to v19", flush=True)
        with io.open(MARKER, "w", encoding="utf-8") as f:
            f.write("fail_checks")
        print("FAIL marker written")
        sys.exit(1)
    with io.open(MARKER, "w", encoding="utf-8") as f:
        f.write("done " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("ALL v20 CHECKS PASS -> coupling.yaml stays v20; marker written")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""E1 容量配置优化运行器：SOFC kW × 电池 kWh 网格扫描 + 约束 + 经济性。
运行：python scripts/run_capacity_opt.py [--days 7] [--pinn]
输出：results/capacity/capacity_scan.csv + capacity_opt.json + capacity_scan.png
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]


def main(days: float = 7.0, with_pinn: bool = False):
    from src.analysis.capacity import CapacityOptimizer

    opt = CapacityOptimizer()
    coupling = None
    if with_pinn:
        from src.coupling import PinnCoupling
        coupling = PinnCoupling(params={"use_pinn_table": True,
                                        "constraint_mode": "derate"})
        print("== E1 容量扫描（含 PINN Tmax 约束 EMS）==")
    else:
        print("== E1 容量扫描（无 PINN；推荐方案单独复核）==")

    grid = opt.scan_grid()
    print(f"  网格 {len(grid)} 点: SOFC {sorted({g[0] for g in grid})} kW x "
          f"电池 {sorted({g[1] for g in grid})} kWh x "
          f"柴油 {sorted({g[2] for g in grid})} kW（夏季工作日代表周 7 d @ 60 s；"
          f"年电量 {opt.annual_energy_kwh / 1000:.1f} MWh = 365 天四季合成）")
    t0 = time.perf_counter()
    rows = opt.scan(days=days, coupling=coupling)
    dt = time.perf_counter() - t0
    print(f"  扫描完成 {dt:.1f}s（{dt / len(rows):.1f} s/点）")

    sel = opt.select(rows)
    csv_p = opt.to_csv(rows)
    json_p = opt.to_json(rows, sel)
    png_p = opt.plot(rows, sel.get("best"))

    print("\n== E1 结果 ==")
    base = sel.get("baseline") or {}
    best = sel.get("best")
    print(f"  可行点 {sel['n_feasible']}/{sel['n_points']}")
    if base:
        print(f"  纯柴油基线（MGO）: LCOE {base['lcoe_usd_mwh']:.1f} USD/MWh, "
              f"年成本 {base['annual_cost_usd'] / 1000:.1f} kUSD, "
              f"GHG {base['ghg_kg_co2eq']:.0f} kg/7d")
    if best:
        vb = best["vs_baseline"]
        print(f"  推荐: SOFC {best['sofc_kw']:.0f} kW + 电池 {best['battery_kwh']:.0f} kWh "
              f"+ 柴油 {best['genset_kw']:.0f} kW")
        print(f"    年总成本 {best['annual_cost_usd'] / 1000:.1f} kUSD, "
              f"LCOE {best['lcoe_usd_mwh']:.1f} USD/MWh, "
              f"CAPEX {best['capex_usd'] / 1000:.1f} kUSD")
        print(f"    vs 基线: GHG 减排 {vb['ghg_reduction_frac'] * 100:.1f}%, "
              f"LCOE Δ {vb['lcoe_delta_usd_mwh']:+.1f} USD/MWh, "
              f"增量投资回收期 {vb['simple_payback_years'] if vb['simple_payback_years'] else 'N/A'} 年")
        print(f"    LOLP {best['lolp_frac']:.5f}, EENS {best['eens_kwh']:.1f} kWh")
    else:
        print("  无可行点！")
    print(f"\n  -> {csv_p}")
    print(f"  -> {json_p}")
    print(f"  -> {png_p}")
    return 0


if __name__ == "__main__":
    days = 7.0
    with_pinn = "--pinn" in sys.argv
    for a in sys.argv[1:]:
        if a.replace(".", "", 1).isdigit():
            days = float(a)
    sys.exit(main(days=days, with_pinn=with_pinn))
# -*- coding: utf-8 -*-
"""C3+C4 集成演示：海岛微电网 7 天调度仿真（LoadProfile + Microgrid + 绘图）。
运行：python scripts/run_day_sim.py [days] [--pinn] [--start YYYY-MM-DD] [--margin-k K] [--limit-k K]
默认窗口：2026-07-06 起 7 天（夏季最恶劣窗口：5 工作日 + 2 周末）。
输出：results/ems_day_sim/ems_day_sim.csv + ems_day_sim.png + summary.json
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(days: int = 7, with_pinn: bool = False, start: str = "2026-07-06",
         margin_k: float = None, limit_k: float = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis import LoadProfile
    from src.ems import Microgrid

    prof = LoadProfile()
    p_load, q_load = prof.generate_series(days=days, dt_s=60.0, start=start, time_index=False)
    p_load = np.asarray(p_load, dtype=float)
    q_load = np.asarray(q_load, dtype=float)

    out_tag = ""
    if with_pinn:
        from src.coupling import PinnCoupling
        pinn_params = {"use_pinn_table": True}
        if margin_k is not None:
            pinn_params["derate_margin_k"] = float(margin_k)
        if limit_k is not None:
            pinn_params["tmax_limit_k"] = float(limit_k)
        if margin_k is not None or limit_k is not None:
            out_tag = f"_m{margin_k or 2:.0f}_l{limit_k or 1173:.0f}"
        mg = Microgrid(coupling=PinnCoupling(params=pinn_params))
    else:
        mg = Microgrid()
    OUT = ROOT / "results" / (("ems_day_sim_pinn" + out_tag) if with_pinn else "ems_day_sim")
    mg.run_days(days, p_load, q_load, dt_s=60.0)
    df = mg.to_dataframe()
    s = mg.summary()

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ems_day_sim.csv", index=False, encoding="utf-8-sig")
    with io.open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

    # ---- 绘图 ----
    t_h = df["t_s"].values / 3600.0
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    ax[0].plot(t_h, df["p_load_kw"], color="k", lw=1.0, label="load")
    ax[0].plot(t_h, df["sofc_kw"], color="tab:green", lw=1.2, label="SOFC")
    ax[0].plot(t_h, df["genset_kw"], color="tab:orange", lw=1.2, label="genset")
    ax[0].plot(t_h, df["bat_kw"], color="tab:blue", lw=0.8, label="battery(+dis)")
    ax[0].plot(t_h, df["net_kw"], color="tab:red", lw=0.8, ls="--", label="net")
    ax[0].set_ylabel("electrical (kW)")
    ax[0].legend(ncol=5, fontsize=8)
    ax[0].set_title(f"Island microgrid {days:.0f}-day dispatch ({start} +{days:.0f}d, SOFC {mg.sofc.p_nom:.0f} kW / genset {mg.genset.p_nom:.0f} kW / battery {mg.battery.e_nom:.0f} kWh)")

    ax[1].plot(t_h, df["soc"], color="tab:purple", lw=1.0, label="SOC")
    ax[1].axhline(0.55, color="gray", ls=":", lw=0.8)
    ax[1].set_ylabel("SOC")
    ax[1].legend(fontsize=8)
    ax2 = ax[1].twinx()
    ax2.plot(t_h, df["p_add_kw"], color="tab:cyan", lw=0.8, alpha=0.7, label="P_add")
    ax2.set_ylabel("P_add (kW)")
    ax2.legend(fontsize=8, loc="lower right")

    ax[2].plot(t_h, df["q_load_kw"], color="k", lw=1.0, label="heat load")
    ax[2].plot(t_h, df["q_rec_kw"], color="tab:green", lw=0.9, label="recovered")
    ax[2].plot(t_h, df["q_boiler_kw"], color="tab:red", lw=0.9, label="boiler fuel")
    ax[2].set_ylabel("heat (kW)")
    ax[2].legend(fontsize=8)

    ax[3].plot(t_h, df["fuel_sofc_kw"], color="tab:green", lw=1.0, label="SOFC fuel")
    ax[3].plot(t_h, df["fuel_genset_kw"], color="tab:orange", lw=1.0, label="genset fuel")
    ax[3].plot(t_h, df["fuel_boiler_kw"], color="tab:red", lw=1.0, label="boiler fuel")
    ax[3].set_ylabel("fuel (kW LHV)")
    ax[3].set_xlabel("time (h)")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "ems_day_sim.png", dpi=110)
    plt.close(fig)

    print(f"== {days:.0f}-day island dispatch (start {start}) ==")
    print(f"  energy: load {s['energy_load_kwh']:.0f} | SOFC {s['energy_sofc_kwh']:.0f} | "
          f"genset {s['energy_genset_kwh']:.0f} | battery {s['energy_bat_kwh']:.0f} kWh")
    print(f"  imbalance: {s['imbalance_kwh']:.3f} kWh (max {s['max_imbalance_kw']:.2f} kW)")
    print(f"  SOC range [{s['soc_min']:.2f}, {s['soc_max']:.2f}]  SOFC SOH end {s['sofc_soh_end']:.4f}")
    print(f"  GHG {s['ghg_kg_co2eq']:.1f} kgCO2eq | NOx {s['nox_kg']:.1f} kg")
    print(f"  fuel: SOFC {s['fuel_sofc_kwh']:.0f} | genset {s['fuel_genset_kwh']:.0f} | boiler {s['fuel_boiler_kwh']:.0f} kWh")
    print(f"  -> {OUT / 'ems_day_sim.csv'} / ems_day_sim.png / summary.json")


if __name__ == "__main__":
    days = 7
    with_pinn = "--pinn" in sys.argv
    start = "2026-07-06"
    margin_k = None
    limit_k = None
    argv = list(sys.argv[1:])
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--start" and i + 1 < len(argv):
            start = argv[i + 1]; i += 2; continue
        if a == "--margin-k" and i + 1 < len(argv):
            margin_k = float(argv[i + 1]); i += 2; continue
        if a == "--limit-k" and i + 1 < len(argv):
            limit_k = float(argv[i + 1]); i += 2; continue
        if a.isdigit():
            days = int(a)
        i += 1
    main(days=days, with_pinn=with_pinn, start=start,
         margin_k=margin_k, limit_k=limit_k)
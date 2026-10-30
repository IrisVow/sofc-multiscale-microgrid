# -*- coding: utf-8 -*-
"""D3：7 天重仿四路对比（基线 / D1 监视 / D3 约束降载 / 紧限值降载演示）。
运行：python scripts/compare_pinn_ems.py [days]
输出：results/coupling/compare_pinn_ems.json + compare_pinn_ems.png
说明：
- 监视模式不改调度（应逐点一致，ΔE=0）。
- 降载模式（生产 1173 K）在超限时削减 SOFC 设定值，缺口由机组/电池自然补足。
- 紧限值 1135 K 仅用于演示 D3 机制确实动作（触发降载、违反清零）。
- 2026-08-18（审稿修订）：统一为夏季最恶劣 7 天窗口（2026-07-06 起，5 工作日 + 2 周末）；
  紧限值降载分别以 2 K 与 15 K 安全裕度演示，量化裕度对降载量与 Tmax 上限的影响。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="?", type=float, default=7.0)
    ap.add_argument("--start", default="2026-07-06")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.analysis import LoadProfile
    from src.coupling import PinnCoupling
    from src.ems import Microgrid

    days = args.days
    start = getattr(args, "start", "2026-07-06")
    prof = LoadProfile()
    p_load, q_load = prof.generate_series(days=days, dt_s=60.0, start=start,
                                          time_index=False)
    p_load = np.asarray(p_load, dtype=float)
    q_load = np.asarray(q_load, dtype=float)

    def run(coupling=None):
        mg = Microgrid(coupling=coupling)
        t0 = time.perf_counter()
        mg.run_days(days, p_load, q_load, dt_s=60.0)
        dt = time.perf_counter() - t0
        return mg.to_dataframe(), mg.summary(), dt, mg

    df0, s0, dt_base, _ = run(None)                                   # 基线
    coup_m = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "monitor"})
    df1, s1, dt_mon, _ = run(coup_m)                                  # D1 监视
    coup_d = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate"})
    df2, s2, dt_der, _ = run(coup_d)                                  # D3 降载（1173 K）
    coup_h = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate",
                                  "tmax_limit_k": 1135.0, "derate_deadband_k": 0.0,
                                  "derate_floor_pu": 0.5, "derate_margin_k": 2.0})
    df3, s3, dt_hot, _ = run(coup_h)                                  # D3 紧限值演示（1135 K, margin 2 K）
    coup_m15 = PinnCoupling(params={"use_pinn_table": True, "constraint_mode": "derate",
                                    "tmax_limit_k": 1135.0, "derate_deadband_k": 0.0,
                                    "derate_floor_pu": 0.5, "derate_margin_k": 15.0})
    df4, s4, dt_m15, _ = run(coup_m15)                                # D3 紧限值演示（1135 K, margin 15 K）

    n = len(df0)
    d_mon = {k: abs(s0[k] - s1[k]) for k in ("energy_sofc_kwh", "energy_genset_kwh", "energy_bat_kwh")}
    d_der = {k: s2[k] - s0[k] for k in ("energy_sofc_kwh", "energy_genset_kwh", "energy_bat_kwh")}
    d_hot = {k: s3[k] - s0[k] for k in ("energy_sofc_kwh", "energy_genset_kwh", "energy_bat_kwh")}
    d_m15 = {k: s4[k] - s0[k] for k in ("energy_sofc_kwh", "energy_genset_kwh", "energy_bat_kwh")}
    ok_mon = all(v < 0.5 for v in d_mon.values())
    cs_m, cs_d, cs_h = coup_m.summary(), coup_d.summary(), coup_h.summary()
    cs_m15 = coup_m15.summary()

    report = {
        "days": days, "n_steps": n,
        "monitor": {"dispatch_same": ok_mon, "dispatch_delta_kwh": d_mon, "pinn": cs_m},
        "derate": {"dispatch_delta_vs_base_kwh": d_der, "pinn": cs_d,
                   "tmax_est": {"max_k": cs_d["pinn_max_tmax_est_k"],
                                "limit_k": coup_d.tmax_limit,
                                "n_viol": cs_d["pinn_tmax_violations"]}},
        "derate_hot_1135": {"dispatch_delta_vs_base_kwh": d_hot, "pinn": cs_h,
                            "tmax_est": {"max_k": cs_h["pinn_max_tmax_est_k"],
                                         "limit_k": coup_h.tmax_limit,
                                         "margin_k": coup_h.derate_margin_k,
                                         "target_k": coup_h.tmax_limit - coup_h.derate_margin_k,
                                         "n_viol": cs_h["pinn_tmax_violations"]}},
        "derate_hot_1135_m15": {"dispatch_delta_vs_base_kwh": d_m15, "pinn": cs_m15,
                                "tmax_est": {"max_k": cs_m15["pinn_max_tmax_est_k"],
                                             "limit_k": coup_m15.tmax_limit,
                                             "margin_k": coup_m15.derate_margin_k,
                                             "target_k": coup_m15.tmax_limit - coup_m15.derate_margin_k,
                                             "n_viol": cs_m15["pinn_tmax_violations"]}},
        "baseline": {k: s0[k] for k in ("energy_load_kwh", "energy_sofc_kwh",
                                        "energy_genset_kwh", "energy_bat_kwh",
                                        "imbalance_kwh", "ghg_kg_co2eq")},
        "ghg_kg_co2eq": {"base": s0["ghg_kg_co2eq"], "monitor": s1["ghg_kg_co2eq"],
                         "derate": s2["ghg_kg_co2eq"], "derate_hot": s3["ghg_kg_co2eq"],
                         "derate_hot_m15": s4["ghg_kg_co2eq"]},
        "runtime_s": {"baseline": dt_base, "monitor": dt_mon, "derate": dt_der,
                      "derate_hot": dt_hot},
        "overhead_ms_step": {"monitor": (dt_mon - dt_base) / n * 1e3,
                             "derate": (dt_der - dt_base) / n * 1e3,
                             "derate_hot": (dt_hot - dt_base) / n * 1e3},
    }
    out_dir = ROOT / "results" / "coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    with io.open(out_dir / "compare_pinn_ems.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 导出逐时序列（F2 fig_ems_dispatch 复用：monitor / m2 / m15）----
    exp_cols = ["t_s", "p_load_kw", "soc", "sofc_kw", "genset_kw", "bat_kw",
                "pinn_req_kw", "pinn_derate_kw", "pinn_tmax_est_k", "pinn_tmax_viol"]
    for name, dfx in (("monitor", df1), ("derate_m2", df3), ("derate_m15", df4)):
        cols = [c for c in exp_cols if c in dfx.columns]
        dfx[cols].to_csv(out_dir / f"compare_pinn_ems_{name}.csv",
                         index=False, encoding="utf-8")
        print(f"[compare_pinn_ems] exported compare_pinn_ems_{name}.csv ({len(dfx)} steps)")

    # ---- 绘图（基线 + 紧限值降载）----
    t_h = df0["t_s"].values / 3600.0
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax[0].plot(t_h, df0["p_load_kw"], color="k", lw=1.0, label="load")
    ax[0].plot(t_h, df0["sofc_kw"], color="tab:green", lw=1.0, label="SOFC base")
    ax[0].plot(t_h, df3["sofc_kw"], color="tab:green", lw=0.8, ls="--", alpha=0.8,
               label="SOFC derate(hot)")
    ax[0].plot(t_h, df3["genset_kw"], color="tab:orange", lw=0.8, label="genset derate(hot)")
    ax[0].plot(t_h, df3["bat_kw"], color="tab:blue", lw=0.7, label="battery derate(hot)")
    ax[0].set_ylabel("electrical (kW)")
    ax[0].legend(ncol=5, fontsize=8)
    ax[0].set_title(f"D3 Tmax constraint EMS — {days:.0f} d island dispatch (v20 LUT, {start})")

    ax[1].plot(t_h, df3["pinn_req_kw"], color="tab:gray", lw=0.8, ls=":", label="SOFC req (pre-derate)")
    ax[1].plot(t_h, df3["sofc_kw"], color="tab:green", lw=1.0, label="SOFC actual")
    ax[1].fill_between(t_h, df3["sofc_kw"], df3["pinn_req_kw"],
                       where=df3["pinn_derate_kw"] > 0.1,
                       color="tab:red", alpha=0.35, label="derated energy")
    ax[1].set_ylabel("SOFC (kW)")
    ax[1].legend(fontsize=8)

    ax[2].plot(t_h, df1["pinn_tmax_est_k"], color="tab:blue", lw=0.9, label="Tmax monitor")
    ax[2].plot(t_h, df3["pinn_tmax_est_k"], color="tab:red", lw=1.0, label="Tmax derate(hot, m2)")
    ax[2].plot(t_h, df4["pinn_tmax_est_k"], color="tab:purple", lw=1.0, ls="--",
               label="Tmax derate(hot, m15)")
    ax[2].axhline(coup_h.tmax_limit, color="gray", ls=":", lw=1.0,
                  label=f"hot limit {coup_h.tmax_limit:.0f} K")
    ax[2].axhline(coup_h.tmax_limit - coup_m15.derate_margin_k, color="purple",
                  ls=":", lw=1.0, label=f"hot target (m15) {coup_h.tmax_limit - coup_m15.derate_margin_k:.0f} K")
    ax[2].axhline(coup_d.tmax_limit, color="black", ls=":", lw=1.0,
                  label=f"prod limit {coup_d.tmax_limit:.0f} K")
    ax[2].fill_between(t_h, 0, 1, where=df1["pinn_in_domain"].astype(bool),
                       color="tab:green", alpha=0.12, transform=ax[2].get_xaxis_transform(),
                       label="in-domain")
    ax[2].set_ylabel("Tmax est (K)")
    ax[2].set_xlabel("time (h)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_pinn_ems.png", dpi=110)
    plt.close(fig)

    print(f"== D3 7 天对比（{days:.0f} d, {n} 步）==")
    print(f"  监视: 调度一致 {'OK' if ok_mon else 'FAIL'} "
          f"(SOFC ΔE={d_mon['energy_sofc_kwh']:.4f} kWh)")
    print(f"  降载(1173K): SOFC {s2['energy_sofc_kwh']:.0f} vs 基线 {s0['energy_sofc_kwh']:.0f} kWh, "
          f"降载 {cs_d['pinn_derate_steps']} 步 / {cs_d['pinn_derate_kwh']:.0f} kWh, "
          f"违反 {cs_d['pinn_tmax_violations']} 步")
    print(f"  紧限值(1135K, m2): SOFC {s3['energy_sofc_kwh']:.0f} vs 基线 {s0['energy_sofc_kwh']:.0f} kWh, "
          f"降载 {cs_h['pinn_derate_steps']} 步 / {cs_h['pinn_derate_kwh']:.0f} kWh, "
          f"Tmax max {cs_h['pinn_max_tmax_est_k']:.1f} K, 违反 {cs_h['pinn_tmax_violations']} 步, "
          f"机组+电池补足 {d_hot['energy_genset_kwh'] + d_hot['energy_bat_kwh']:.0f} kWh")
    print(f"  紧限值(1135K, m15): SOFC {s4['energy_sofc_kwh']:.0f} vs 基线 {s0['energy_sofc_kwh']:.0f} kWh, "
          f"降载 {cs_m15['pinn_derate_steps']} 步 / {cs_m15['pinn_derate_kwh']:.0f} kWh, "
          f"Tmax max {cs_m15['pinn_max_tmax_est_k']:.1f} K, 违反 {cs_m15['pinn_tmax_violations']} 步")
    print(f"  GHG: 基线 {s0['ghg_kg_co2eq']:.0f} / 监视 {s1['ghg_kg_co2eq']:.0f} / "
          f"降载 {s2['ghg_kg_co2eq']:.0f} / 紧限值 {s3['ghg_kg_co2eq']:.0f} kgCO2eq")
    print(f"  开销: 监视 {(dt_mon-dt_base)/n*1e3:.3f} / 降载 {(dt_der-dt_base)/n*1e3:.3f} ms/步")
    print(f"  -> {out_dir / 'compare_pinn_ems.json'} / compare_pinn_ems.png")
    sys.exit(0 if ok_mon else 1)


if __name__ == "__main__":
    main()
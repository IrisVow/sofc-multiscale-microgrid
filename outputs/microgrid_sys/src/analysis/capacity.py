# -*- coding: utf-8 -*-
"""E1 容量配置优化：SOFC kW × 电池 kWh 网格扫描（+ 纯柴油基线）+ 可靠性约束 + 经济性。

方法（PLANNING.md 阶段 E）：
- 扫描空间：sofc_kw ∈ {0} ∪ sofc_capacity_scan（scenario.yaml，默认 [60,100,150]）
  × battery_kwh ∈ battery_mwh_range 三分点（默认 [100,300,500]）；柴油机组固定
  genset_capacity_kw（100 kW）。
- 每点跑 7 天最坏工况（夏季工作日，60 s 步长，LoadProfile 合成）；指标：能量分配、
  燃料、GHG/NOx/SOx/PM、LOLP/EENS、SOC 边界、经济性。
- 年化：以 C4 年电量（annual_energy_mwh=846.3）对 7 天窗口能量缩放燃料与 O&M；
  CAPEX 用 CRF 年金化（project_years / discount_rate）；LCOE = 年总成本/年电量；
  对比纯柴油基线（sofc=0，MGO）给 ΔGHG、ΔLCOE、增量投资回收期。
- 约束（configs/economics.yaml）：LOLP ≤ 0.1% 步数、EENS ≤ 0.1% 负荷能量、
  SOC ∈ [lb,ub]（模型已钳位，检查防漂移）。可行点取年总成本最小为推荐。
- 可选 coupling：传 PinnCoupling 时每点带 D3 Tmax 约束（生产 1173 K 零误触发，
  扫描结果与无耦合一致；推荐方案单独用 PINN 复核）。

不确定性（详见 docs/model_assumptions.md §E1）：
- 7 天夏季最坏工况 → 年化用能量缩放（忽略季节效率差异，±10% 量级）；
- 燃料价/CAPEX/O&M 为工程估计（E2 文献对照更新）；电池循环老化未建模；
- 纯柴油基线用 MGO（scenario backup_fuel），其余点机组按当前配置 LNG。

调用：
    opt = CapacityOptimizer()
    rows = opt.scan()                  # 全网格（12 点，约 20-40 s）
    best = opt.select(rows)["best"]    # 可行最优
    opt.to_csv(rows)                   # results/capacity/capacity_scan.csv
    opt.plot(rows, best)               # results/capacity/capacity_scan.png
    opt.to_json(rows, best)            # results/capacity/capacity_opt.json
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import yaml

from ..ems import Microgrid

ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class CapacityOptimizer:
    """E1 容量配置网格扫描 + 约束 + 经济性（阶段 E）。"""

    DEFAULTS = {
        "fuel_lng_usd_kwh": 0.043,
        "fuel_mgo_usd_kwh": 0.075,
        "om_sofc_usd_kwh": 0.020,
        "om_genset_usd_kwh": 0.025,
        "om_battery_usd_kwh": 0.010,
        "capex_sofc_usd_kw": 1500.0,
        "capex_genset_usd_kw": 500.0,
        "capex_battery_usd_kwh": 350.0,
        "project_years": 20,
        "discount_rate": 0.08,
        "annual_energy_mwh": 846.3,
        "lolp_max_frac": 0.001,
        "eens_max_frac": 0.001,
        "imbalance_tol_kw": 1.0,
        "baseline_genset_kw": 250.0,   # 纯柴油基线装机（峰值 200 × 1.25 裕度，保证可靠性）
    }

    def __init__(self, config_dir: str = None, overrides: dict = None):
        cfg_dir = Path(config_dir) if config_dir else ROOT / "configs"
        self.config_dir = cfg_dir
        self.scenario = _read_yaml(cfg_dir / "scenario.yaml")
        eco = _read_yaml(cfg_dir / "economics.yaml").get("economics", {})
        self.cfg = dict(self.DEFAULTS)
        self.cfg.update(eco)
        if overrides:
            self.cfg.update(overrides)
        comp = self.scenario.get("components", {})
        self.sofc_scan = [float(x) for x in comp.get("sofc_capacity_scan", [60, 100, 150])]
        self.bat_range = [float(x) for x in comp.get("battery_mwh_range", [0.1, 0.5])]
        self.bat_scan_list = [float(x) for x in comp.get("battery_capacity_scan", [])]
        self.genset_kw = float(comp.get("genset_capacity_kw", 100.0))
        self.genset_scan = [float(x) for x in comp.get("genset_capacity_scan", [self.genset_kw])]
        self.annual_energy_kwh = float(self.cfg["annual_energy_mwh"]) * 1000.0
        self.result_dir = ROOT / "results" / "capacity"

    # ------------------------------------------------------------------
    def bat_scan(self) -> list:
        """电池容量扫描点（kWh）：优先 scenario 显式列表，否则三分点。"""
        if self.bat_scan_list:
            return sorted({float(x) for x in self.bat_scan_list})
        lo, hi = self.bat_range
        return sorted({lo * 1000.0, 0.5 * (lo + hi) * 1000.0, hi * 1000.0})

    def scan_grid(self) -> list:
        """(SOFC kW, 电池 kWh, genset kW) 三元组网格；SOFC=0 为纯柴油基线行。"""
        return [(s, b, g) for s in ([0.0] + self.sofc_scan)
                for b in self.bat_scan() for g in self.genset_scan]

    # ------------------------------------------------------------------
    def run_point(self, sofc_kw: float, bat_kwh: float, genset_kw: float = None,
                  days: float = 7.0, coupling=None, genset_fuel: str = None,
                  component_overrides: dict = None) -> dict:
        """单点 7 天夏季代表窗口仿真（夏季工作日，最恶劣日型）+ 指标 + 经济性。

        component_overrides：额外组件参数覆盖（E3 敏感性用），
        如 {"sofc": {"deg_scale": 1.5}} 加速降解。"""
        from .load_profile import LoadProfile
        prof = LoadProfile(str(self.config_dir / "scenario.yaml"))
        p_load, q_load = prof.generate_series(days=days, dt_s=60.0,
                                              season="summer", weekend=False,
                                              time_index=False)
        p_load = np.asarray(p_load, dtype=float)
        q_load = np.asarray(q_load, dtype=float)
        # sofc=0（纯柴油基线）：机组配足峰值裕度（250 kW）保证可靠；SOFC 用 1e-3 kW
        # 占位避免 p_nom=0 除零（功率/燃料可忽略）
        ov = dict(component_overrides or {})
        ov["sofc"] = {"p_nom_kw": 1e-3 if sofc_kw <= 0.0 else float(sofc_kw),
                      **ov.get("sofc", {})}
        ov["battery"] = {"e_nom_kwh": float(bat_kwh), **ov.get("battery", {})}
        if sofc_kw <= 0.0:
            genset_kw_eff = float(self.cfg["baseline_genset_kw"])
            ov["genset"] = {"p_nom_kw": genset_kw_eff, **ov.get("genset", {})}
        else:
            genset_kw_eff = float(genset_kw) if genset_kw is not None else self.genset_kw
            ov["genset"] = {"p_nom_kw": genset_kw_eff, **ov.get("genset", {})}
        fuel = genset_fuel or ("MGO" if sofc_kw <= 0.0 else "LNG")
        mg = Microgrid(config_dir=str(self.config_dir), component_overrides=ov,
                       genset_fuel=fuel, coupling=coupling)
        mg.run_days(days, p_load, q_load, dt_s=60.0)
        df = mg.to_dataframe()
        s = mg.summary()
        es = mg.emissions.summary()
        dt_h = 60.0 / 3600.0
        deficit = np.maximum(0.0, df["p_load_kw"].values - df["net_kw"].values)
        n_steps = len(df)
        row = {
            "sofc_kw": sofc_kw, "battery_kwh": bat_kwh, "genset_kw": genset_kw_eff,
            "genset_fuel": fuel, "n_steps": n_steps,
            "energy_load_kwh": float(s["energy_load_kwh"]),
            "energy_sofc_kwh": float(s["energy_sofc_kwh"]),
            "energy_genset_kwh": float(s["energy_genset_kwh"]),
            "energy_bat_kwh": float(s["energy_bat_kwh"]),
            "energy_bat_cycled_kwh": float(df["bat_kw"].abs().sum()) * dt_h,
            "fuel_sofc_kwh": float(s["fuel_sofc_kwh"]),
            "fuel_genset_kwh": float(s["fuel_genset_kwh"]),
            "fuel_boiler_kwh": float(s["fuel_boiler_kwh"]),
            "ghg_kg_co2eq": float(s["ghg_kg_co2eq"]),
            "nox_kg": float(s["nox_kg"]),
            "sox_kg": float(es["sox_kg"]),
            "pm_kg": float(es["pm_kg"]),
            "lolp_steps": int((deficit > self.cfg["imbalance_tol_kw"]).sum()),
            "lolp_frac": float((deficit > self.cfg["imbalance_tol_kw"]).mean()),
            "eens_kwh": float(deficit.sum()) * dt_h,
            "soc_min": float(s["soc_min"]), "soc_max": float(s["soc_max"]),
            "sofc_soh_end": float(s["sofc_soh_end"]),
        }
        row.update(self.economics(row))
        return row

    # ------------------------------------------------------------------
    def economics(self, row: dict) -> dict:
        """年化经济性：CRF 年金化 CAPEX + 能量缩放燃料/O&M + LCOE。

        年电量口径：`annual_energy_kwh` 为 365 天四季合成年电量（LoadProfile 校验值
        846.3 MWh，见 economics.yaml）；年化系数 k = 年电量 / 7 天代表窗口负荷能量，
        即把代表窗口运行成本按年电量负荷加权放大（近似假设成本与负荷成正比）。"""
        c = self.cfg
        k = self.annual_energy_kwh / max(float(row["energy_load_kwh"]), 1e-9)
        cost_lng = ((float(row["fuel_sofc_kwh"]) + float(row["fuel_boiler_kwh"]))
                       * k * float(c["fuel_lng_usd_kwh"]))
        p_gen = (float(c["fuel_lng_usd_kwh"]) if row["genset_fuel"] == "LNG"
                 else float(c["fuel_mgo_usd_kwh"]))
        cost_gen = float(row["fuel_genset_kwh"]) * k * p_gen
        om = (float(row["energy_sofc_kwh"]) * float(c["om_sofc_usd_kwh"])
              + float(row["energy_genset_kwh"]) * float(c["om_genset_usd_kwh"])
              + float(row["energy_bat_cycled_kwh"]) * float(c["om_battery_usd_kwh"])) * k
        capex = (float(row["sofc_kw"]) * float(c["capex_sofc_usd_kw"])
                 + float(row["genset_kw"]) * float(c["capex_genset_usd_kw"])
                 + float(row["battery_kwh"]) * float(c["capex_battery_usd_kwh"]))
        crf = self._crf()
        annual_capex = capex * crf
        annual_cost = cost_lng + cost_gen + om + annual_capex
        return {
            "annual_scale": k,
            "capex_usd": capex,
            "crf": crf,
            "annual_capex_usd": annual_capex,
            "annual_fuel_usd": cost_lng + cost_gen,
            "annual_om_usd": om,
            "annual_cost_usd": annual_cost,
            "lcoe_usd_mwh": annual_cost / (self.annual_energy_kwh / 1000.0),
        }

    def _crf(self) -> float:
        r = float(self.cfg["discount_rate"])
        n = float(self.cfg["project_years"])
        if r <= 0.0:
            return 1.0 / n
        return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)

    # ------------------------------------------------------------------
    def feasible(self, row: dict) -> bool:
        c = self.cfg
        ok_lolp = float(row["lolp_frac"]) <= float(c["lolp_max_frac"])
        ok_eens = float(row["eens_kwh"]) <= (float(c["eens_max_frac"])
                              * float(row["energy_load_kwh"]))
        ok_soc = float(row["soc_min"]) >= 0.2 - 1e-6 and float(row["soc_max"]) <= 0.9 + 1e-6
        return bool(ok_lolp and ok_eens and ok_soc)

    def select(self, rows: list) -> dict:
        """可行点中取年总成本最小者；并计算相对纯柴油基线的减排/回收期。"""
        feas = [r for r in rows if self.feasible(r)]
        best = min(feas, key=lambda r: r["annual_cost_usd"]) if feas else None
        base = None
        for r in rows:
            if abs(r["sofc_kw"]) < 1e-9:
                base = r
                break
        out = {"n_points": len(rows), "n_feasible": len(feas),
               "feasible": [[r["sofc_kw"], r["battery_kwh"], r["genset_kw"]] for r in feas]}
        if best is None:
            out["best"] = None
            return out
        b = {k: best[k] for k in ("sofc_kw", "battery_kwh", "genset_kw",
                                  "annual_cost_usd", "lcoe_usd_mwh", "capex_usd",
                                  "annual_fuel_usd", "annual_om_usd", "ghg_kg_co2eq",
                                  "lolp_frac", "eens_kwh")}
        if base is not None:
            b["vs_baseline"] = {
                "ghg_reduction_frac": 1.0 - best["ghg_kg_co2eq"] / base["ghg_kg_co2eq"],
                "lcoe_usd_mwh_base": base["lcoe_usd_mwh"],
                "lcoe_delta_usd_mwh": best["lcoe_usd_mwh"] - base["lcoe_usd_mwh"],
                "annual_op_base_usd": base["annual_fuel_usd"] + base["annual_om_usd"],
                "annual_op_usd": best["annual_fuel_usd"] + best["annual_om_usd"],
            }
            op_save = b["vs_baseline"]["annual_op_base_usd"] - b["vs_baseline"]["annual_op_usd"]
            inc_capex = best["capex_usd"] - base["capex_usd"]
            b["vs_baseline"]["incremental_capex_usd"] = inc_capex
            b["vs_baseline"]["simple_payback_years"] = (inc_capex / op_save) if op_save > 0 else None

        out["best"] = b
        out["baseline"] = {"sofc_kw": 0.0, "battery_kwh": base["battery_kwh"] if base else None,
                           "annual_cost_usd": base["annual_cost_usd"] if base else None,
                           "lcoe_usd_mwh": base["lcoe_usd_mwh"] if base else None,
                           "ghg_kg_co2eq": base["ghg_kg_co2eq"] if base else None}
        return out

    # ------------------------------------------------------------------
    def scan(self, days: float = 7.0, coupling=None) -> list:
        """全网格扫描（sofc=0 用 MGO 基线，其余 LNG）。"""
        rows = []
        for sofc_kw, bat_kwh, genset_kw in self.scan_grid():
            rows.append(self.run_point(sofc_kw, bat_kwh, genset_kw=genset_kw,
                                       days=days, coupling=coupling))
        return rows

    # ------------------------------------------------------------------
    def to_csv(self, rows: list, path: str = None) -> str:
        keys = ["sofc_kw", "battery_kwh", "genset_kw", "genset_fuel",
                "energy_load_kwh", "energy_sofc_kwh", "energy_genset_kwh",
                "energy_bat_kwh", "fuel_sofc_kwh", "fuel_genset_kwh",
                "fuel_boiler_kwh", "ghg_kg_co2eq", "nox_kg", "sox_kg", "pm_kg",
                "lolp_steps", "lolp_frac", "eens_kwh", "soc_min", "soc_max",
                "capex_usd", "annual_capex_usd", "annual_fuel_usd", "annual_om_usd",
                "annual_cost_usd", "lcoe_usd_mwh"]
        path = Path(path) if path else self.result_dir / "capacity_scan.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            w = None
            for r in rows:
                row = {k: r.get(k) for k in keys}
                if w is None:
                    w = __import__("csv").DictWriter(f, fieldnames=keys)
                    w.writeheader()
                w.writerow(row)
        return str(path)

    def to_json(self, rows: list, selection: dict, path: str = None) -> str:
        path = Path(path) if path else self.result_dir / "capacity_opt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump({"selection": selection, "rows": rows}, f,
                      ensure_ascii=False, indent=2)
        return str(path)

    def plot(self, rows: list, best: dict = None, path: str = None):
        """双面板：年总成本 / GHG vs SOFC 装机（按电池容量分线）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path = Path(path) if path else self.result_dir / "capacity_scan.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        # 多 genset 维度下，每个 (sofc,bat) 只画该组合最优 genset 的点（年总成本最小）
        best_by_sb = {}
        for r in rows:
            key = (r["sofc_kw"], r["battery_kwh"])
            if key not in best_by_sb or r["annual_cost_usd"] < best_by_sb[key]["annual_cost_usd"]:
                best_by_sb[key] = r
        rows = list(best_by_sb.values())
        bats = sorted({r["battery_kwh"] for r in rows})
        for b in bats:
            sub = [r for r in rows if r["battery_kwh"] == b]
            sub = sorted(sub, key=lambda r: r["sofc_kw"])
            xs = [r["sofc_kw"] for r in sub]
            ax[0].plot(xs, [r["annual_cost_usd"] / 1000.0 for r in sub],
                       marker="o", label=f"bat {b:.0f} kWh")
            ax[1].plot(xs, [r["ghg_kg_co2eq"] for r in sub],
                       marker="o", label=f"bat {b:.0f} kWh")
        if best is not None:
            ax[0].axvline(best["sofc_kw"], color="tab:red", ls=":", lw=1.2,
                          label=f"opt SOFC {best['sofc_kw']:.0f} kW")
        ax[0].set_xlabel("SOFC capacity (kW)")
        ax[0].set_ylabel("annual total cost (kUSD)")
        ax[0].set_title("E1 capacity scan — annual cost")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].set_xlabel("SOFC capacity (kW)")
        ax[1].set_ylabel("GHG (kgCO2eq / 7d)")
        ax[1].set_title("E1 capacity scan — GHG (7-day summer rep. week)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)

# -*- coding: utf-8 -*-
"""D5 公共 PINN 推理 API 自检（evaluate_field / evaluate_table + 域外告警）。
运行：python scripts/check_inference.py（工作目录 ROOT）
通过时逐项打印 OK；任一断言失败退出码非 0。

校验项：
- evaluate_field：单点域内（12 列、T 有限）、5 探针批量、域外行 NaN + V_applied 告警、
  确定性（重复调用一致）、v19（10 维）与 v18（7 维）列序自动识别。
- evaluate_table：2×2 工况网格汇总 shape/有限、Tmax 随 V 单调（低 V 高电流 → 高温）、
  与 PinnLUT 在网格节点交叉核对（<=0.1 K）、域外行 NaN + 告警、npz 导出可加载。
- 性能：evaluate_field 单点（不同工况，绕过缓存）< 5 ms/次。
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from src.coupling import PinnLUT
from src.coupling.pinn_inference import PinnInference, evaluate_field, evaluate_table

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        raise AssertionError(f"FAIL: {name} {detail}")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# v19 行（10 维）：x,y,t,V,amp,stepT,wave_type,T_in,lambda_air,FU
def row_v19(v, t_in, lam=6.0, fu=0.7, x=0.05, y=1.31e-3):
    return [x, y, 0.0, v, 0.0, 0.0, 0.0, t_in, lam, fu]


# v18 行（7 维）：x,y,t,V,T_in,lambda_air,FU
def row_v18(v, t_in, lam=6.0, fu=0.7, x=0.05, y=1.31e-3):
    return [x, y, 0.0, v, t_in, lam, fu]


api = PinnInference()
print(f"== D5 公共推理 API（cont_version={api.cont_version}，{api.onnx_path.split('/')[-1]}）==")

print("-- evaluate_field --")
r = api.evaluate_field(np.array([row_v19(0.85, 1123.0)]))
check("单点域内：12 列、T 有限", r["fields"].shape == (1, 12) and np.isfinite(r["fields"][0, 0])
      and bool(r["in_domain"][0]) and r["n_valid"] == 1,
      f"(T={float(r['fields'][0, 0]):.2f} K)")
check("列名 = 12 物理量", r["columns"] == ["T", "u", "v", "p", "w1", "w2", "w3",
                                            "phis", "phil", "htQ", "qav_an", "qav_ca"])

probes = [0.00040, 0.00105, 0.00131, 0.00157, 0.00222]
r5 = api.evaluate_field(np.array([row_v19(0.9, 1100.0, y=y) for y in probes]))
check("5 探针批量：shape (5,12)、全域内", r5["fields"].shape == (5, 12)
      and bool(r5["in_domain"].all()) and r5["n_valid"] == 5)

r_ood = api.evaluate_field(np.array([row_v19(0.60, 1123.0)]))
viol_vars = [v["var"] for v in r_ood["violations"]]
check("域外行（V=0.6）：全 NaN + V_applied 告警", np.isnan(r_ood["fields"]).all()
      and not bool(r_ood["in_domain"][0]) and "V_applied" in viol_vars,
      f"(viol={viol_vars})")

r_a = api.evaluate_field(np.array([row_v19(0.87, 1090.0)]))
r_b = api.evaluate_field(np.array([row_v19(0.87, 1090.0)]))
check("确定性（重复调用一致）", float(np.abs(r_a["fields"] - r_b["fields"]).max()) == 0.0)

# v18 兼容（7 维）
v18_path = os.path.join(ROOT, "..", "pinn_sofc", "runs", "stage3_pde_v18", "pinn_sofc_v18.onnx")
if os.path.exists(v18_path):
    r18 = evaluate_field(np.array([row_v18(0.85, 1123.0)]), onnx_path=v18_path)
    check("v18（7 维）自动识别：T 有限、域内", r18["cont_version"] == 7
          and np.isfinite(r18["fields"][0, 0]) and bool(r18["in_domain"][0]),
          f"(N_CONT=7, T={float(r18['fields'][0, 0]):.2f} K)")
else:
    print("  [SKIP] v18历史模型兼容性：本发布包仅提供v20部署模型")

print("-- evaluate_table --")
grid = np.array([row_v19(v, t) for v in (0.85, 0.95) for t in (1073.0, 1123.0)])
t = evaluate_table(grid)
check("2×2 工况网格：tmax 等 4 场 shape (4,)、有限", t["tmax_k"].shape == (4,)
      and np.isfinite(t["tmax_k"]).all() and np.isfinite(t["qav_an_max"]).all()
      and np.isfinite(t["qav_ca_max"]).all() and np.isfinite(t["phi_drop_v"]).all(),
      f"(tmax={np.round(t['tmax_k'], 1).tolist()})")
check("Tmax 随 V 单调（低 V → 高电流 → 高温）",
      t["tmax_k"][0] > t["tmax_k"][2] and t["tmax_k"][1] > t["tmax_k"][3],
      f"(V0.85: {t['tmax_k'][0]:.1f}/{t['tmax_k'][1]:.1f} vs V0.95: {t['tmax_k'][2]:.1f}/{t['tmax_k'][3]:.1f} K)")

from src.coupling.pinn_coupling import load_coupling_params
lut = PinnLUT(os.path.join(ROOT, load_coupling_params()["lut_path"]))
r_lut = lut.lookup(0.9, 1098.0, 6.0, 0.7)
t_node = evaluate_table(np.array([row_v19(0.9, 1098.0)]))
diff = abs(float(t_node["tmax_k"][0]) - r_lut["tmax_k"])
check("与 PinnLUT 网格节点交叉核对 <= 0.1 K", diff <= 0.1,
      f"(lut={r_lut['tmax_k']:.4f}, api={float(t_node['tmax_k'][0]):.4f}, d={diff:.4f})")

t_ood = evaluate_table(np.array([row_v19(0.85, 1123.0), row_v19(1.05, 1123.0)]))
check("表级域外行：该行 NaN + 告警含 op 索引", np.isnan(t_ood["tmax_k"][1])
      and not bool(t_ood["in_domain"][1]) and t_ood["violations"]
      and t_ood["violations"][0]["op"] == 1 and t_ood["violations"][0]["var"] == "V_applied",
      f"(viol={[(v['op'], v['var']) for v in t_ood['violations']]})")

fd, tmp = tempfile.mkstemp(suffix=".npz", prefix="check_inference_")
os.close(fd)
os.remove(tmp)
npz_ok = False
try:
    t_npz = evaluate_table(np.array([row_v19(0.85, 1100.0)]), out_npz=tmp)
    with np.load(tmp, allow_pickle=True) as z:
        npz_ok = ("tmax_k" in z and "ops_grid" in z and "meta" in z
                  and z["tmax_k"].shape == (1,))
    npz_ok = npz_ok and os.path.exists(t_npz["npz_path"])
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
check("npz 导出可加载（tmax_k/ops_grid/meta）", npz_ok,
      f"(path={t_npz['npz_path']})")

print("-- 性能（单点、不同工况绕过缓存）--")
t0 = time.perf_counter()
for i in range(100):
    api.evaluate_field(np.array([row_v19(0.80 + 0.001 * i, 1100.0)]))
avg_ms = (time.perf_counter() - t0) / 100.0 * 1e3
check("evaluate_field 单点 < 5 ms/次", avg_ms < 5.0, f"({avg_ms:.3f} ms/次)")

print(f"\n== 全部通过：{PASS} 项检查 OK ==")

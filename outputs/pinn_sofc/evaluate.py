# -*- coding: utf-8 -*-
"""M1 评估脚本：加载 checkpoint，计算按变量/按域指标（物理单位），输出 CSV + 对比图。

用法：
    python evaluate.py --run_dir runs/stage1_baseline --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from data.dataset import SOFCDataModule, CONT_IDS
from data.split import DOMAIN_ORDER, DOMAIN_ID
from models.pinn import BaselineMLP


def build_model_from_cfg(cfg: dict, n_out: int, n_cont: int = len(CONT_IDS)) -> torch.nn.Module:
    m = cfg["model"]
    from models.pinn import BaselineMLP
    return BaselineMLP(n_cont=n_cont, n_domain=5, n_out=n_out,
                       hidden=int(m["hidden"]), layers=int(m["layers"]),
                       dropout=0.0, fourier=bool(m["fourier"]), n_freq=int(m["n_freq"]),
                       f_min=m.get("f_min", 1.0), f_max=m.get("f_max", 512.0),
                       fourier_cols=m.get("fourier_cols", [0, 1, 2]))


def metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """按列计算 rmse/mae/rel_l2（仅有效位置）。"""
    out = {}
    for name, fn in [
        ("rmse", lambda e: np.sqrt(np.nanmean(e ** 2, axis=0))),
        ("mae", lambda e: np.nanmean(np.abs(e), axis=0)),
        ("rel_l2", lambda e: np.sqrt(np.nansum(e ** 2, axis=0)) / np.sqrt(np.nansum(y_true ** 2, axis=0))),
    ]:
        e = np.where(mask > 0, y_pred - y_true, np.nan)
        out[name] = fn(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint filename in run_dir")
    ap.add_argument("--pools", nargs="*", default=["steady_val", "steady_test"])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join(args.run_dir, args.ckpt), map_location="cpu", weights_only=False)
    release_paths = cfg["paths"].copy()
    cfg = ck.get("config", cfg).copy()
    cfg["paths"] = release_paths

    dm = SOFCDataModule(cfg, torch.device("cpu"))
    model = build_model_from_cfg(cfg, dm.n_out, n_cont=dm.n_cont).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = []
    for pool_name in args.pools:
        X, Y, M, pool = dm.pool_tensors(pool_name)
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 32768):
                preds.append(model(X[i:i + 32768].to(device)).cpu())
        P = torch.cat(preds, 0)
        y_true = Y.numpy()
        y_pred = dm.denorm_y(P)
        y_true_raw = pool.y
        mask = M.numpy()
        met = metrics(y_true_raw, y_pred, mask)

        for j, v in enumerate(dm.target_vars):
            rows.append({
                "pool": pool_name, "var": v,
                "rmse": met["rmse"][j], "mae": met["mae"][j], "rel_l2": met["rel_l2"][j],
                "n_valid": int(mask[:, j].sum()),
            })

        # 按域分解
        for d, di in enumerate(DOMAIN_ORDER):
            sel = pool.dom == di
            if not sel.any():
                continue
            sub_mask = mask[sel]
            sub_true = y_true_raw[sel]
            sub_pred = y_pred[sel]
            for j, v in enumerate(dm.target_vars):
                e = np.where(sub_mask[:, j] > 0, sub_pred[:, j] - sub_true[:, j], np.nan)
                if np.isnan(e).all():
                    continue
                rows.append({
                    "pool": pool_name, "var": f"{v}@{d}",
                    "rmse": np.sqrt(np.nanmean(e ** 2)), "mae": np.nanmean(np.abs(e)),
                    "rel_l2": np.sqrt(np.nansum(e ** 2)) / np.sqrt(np.nansum(sub_true[:, j] ** 2)),
                    "n_valid": int((sub_mask[:, j] > 0).sum()),
                })

        # Tmax 误差（工程指标）
        if "T" in dm.target_vars:
            jT = dm.target_vars.index("T")
            sel = mask[:, jT] > 0
            tmax_true = y_true_raw[sel, jT].max()
            tmax_pred = y_pred[sel, jT].max()
            rows.append({"pool": pool_name, "var": "Tmax(K)", "rmse": tmax_pred - tmax_true,
                         "mae": abs(tmax_pred - tmax_true), "rel_l2": (tmax_pred - tmax_true) / tmax_true,
                         "n_valid": int(sel.sum())})

    df = pd.DataFrame(rows)
    stem = os.path.splitext(args.ckpt)[0]
    out_csv = os.path.join(args.run_dir, f"eval_metrics_{stem}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"[eval] saved -> {out_csv}")

    # ---- M4: transient per-(case,t) metrics + T(t) probe ----
    if "transient_test" in args.pools:
        Xt, Yt, Mt, pt = dm.pool_tensors("transient_test")
        y_true_t = pt.y
        with torch.no_grad():
            Pt = torch.cat([model(Xt[i:i + 32768].to(device)).cpu()
                            for i in range(0, Xt.shape[0], 32768)], 0)
        y_pred_t = dm.denorm_y(Pt)
        mask_t = Mt.numpy()
        trows = []
        for cid in np.unique(pt.case_id):
            for t in np.unique(pt.t[pt.case_id == cid]):
                sel = (pt.case_id == cid) & (np.isclose(pt.t, t))
                sm = mask_t[sel]; st = y_true_t[sel]; sp = y_pred_t[sel]
                for j, v in enumerate(dm.target_vars):
                    e = np.where(sm[:, j] > 0, sp[:, j] - st[:, j], np.nan)
                    if np.isnan(e).all():
                        continue
                    trows.append({"case": cid, "t_s": float(t), "var": v,
                                  "rmse": float(np.sqrt(np.nanmean(e ** 2))),
                                  "mae": float(np.nanmean(np.abs(e))),
                                  "rel_l2": float(np.sqrt(np.nansum(e ** 2)) / np.sqrt(np.nansum(st[:, j] ** 2))),
                                  "n_valid": int((sm[:, j] > 0).sum())})
        df_t = pd.DataFrame(trows)
        df_t.to_csv(os.path.join(args.run_dir, f"eval_transient_by_t_{stem}.csv"),
                    index=False, encoding="utf-8-sig")
        # T(t) probe: y=electrolyte center (1.31e-3 m), x~0.05 m
        jT = dm.target_vars.index("T")
        # PointPool.x 是 (N,7) CONT_IDS 矩阵：x=col0, y=col1, t=col2
        probe_sel = ((pt.dom == DOMAIN_ID["electrolyte"]) &
                     (np.abs(pt.x[:, 0] - 0.05) < 2e-3) &
                     (np.abs(pt.x[:, 1] - 1.31e-3) < 2e-5))
        if probe_sel.any():
            prows = []
            for cid in np.unique(pt.case_id[probe_sel]):
                for t in np.unique(pt.t[probe_sel & (pt.case_id == cid)]):
                    sel = probe_sel & (pt.case_id == cid) & (np.isclose(pt.t, t))
                    prows.append({"case": cid, "t_s": float(t),
                                  "T_true": float(np.nanmean(y_true_t[sel, jT])),
                                  "T_pred": float(np.nanmean(y_pred_t[sel, jT])),
                                  "T_err": float(np.nanmean(y_pred_t[sel, jT] - y_true_t[sel, jT]))})
            pd.DataFrame(prows).to_csv(
                os.path.join(args.run_dir, f"probe_T_t_{stem}.csv"),
                index=False, encoding="utf-8-sig")
            print(f"[eval] saved transient by-t -> eval_transient_by_t_{stem}.csv, probe_T_t_{stem}.csv")

    # 汇总打印
    print("\n==== 按变量汇总（rmse/mae/rel_l2）====")
    for pool_name in args.pools:
        sub = df[(df.pool == pool_name) & (~df["var"].str.contains("@"))]
        print(f"\n-- {pool_name} --")
        for _, r in sub.iterrows():
            print(f"  {r['var']:>9s}: rmse={r['rmse']:.5g} mae={r['mae']:.5g} rel_l2={r['rel_l2']:.5f}")


if __name__ == "__main__":
    main()
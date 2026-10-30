# -*- coding: utf-8 -*-
"""M5 可视化：与 COMSOL 全场逐点对比图 + T3 温度云图动画（GIF）。

用法：
    conda activate sofc
    python plot_fields.py --config config_m4b.yaml --ckpt runs/stage3_pde_m4b/best.pt --cases all
    python plot_fields.py --config config_m4b.yaml --ckpt runs/stage3_pde_m4b/best.pt --animate

输出（runs/<run_dir>/plots/）：
    s9_fields_T_phis_htQ.png   S9 稳态 T/phis/htQ 预测 vs COMSOL vs 误差
    t3_t1200_fields.png        T3 末态（t=1200 s）同上
    t3_T_snapshots.png         T3 关键时刻（100/360/1200 s）温度场对比
    t3_T_animation.gif         T3 全 22 时间步温度场动画
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import SOFCDataModule, CONT_IDS
from data.split import DOMAIN_ORDER, DOMAIN_ID
from models.pinn import BaselineMLP

# 域边界 y (mm)（用于图上画分隔线）
DOM_BOUNDS_MM = [0.0, 0.8, 1.30, 1.32, 1.82, 2.62]
VAR_LABEL = {
    "T": "T [K]", "phis": "phis [V]", "phil": "phil [V]",
    "htQ": "htQ [W/m^3]", "qav_an": "qav_an [A/m^3]", "qav_ca": "qav_ca [A/m^3]",
    "u": "u [m/s]", "v": "v [m/s]", "p": "p [Pa]", "w1": "w1 [-]",
}


def build_net(cfg, n_out, n_cont=len(CONT_IDS)):
    m = cfg["model"]
    return BaselineMLP(n_cont=n_cont, n_domain=5, n_out=n_out,
                       hidden=int(m["hidden"]), layers=int(m["layers"]),
                       dropout=0.0, fourier=bool(m["fourier"]), n_freq=int(m["n_freq"]),
                       f_min=m.get("f_min", 1.0), f_max=m.get("f_max", 512.0),
                       fourier_cols=m.get("fourier_cols", [0, 1, 2]))


def predict(dm, model, pool, device):
    X, _, _, p = dm.pool_tensors(pool)
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 32768):
            preds.append(model(X[i:i + 32768].to(device)).cpu())
    P = torch.cat(preds, 0)
    return p, dm.denorm_y(P)


def field_axes(ax, p, y_pred, y_true, var, j, vmin=None, vmax=None, title=None, cmap="turbo"):
    """单轴 scatter 场图：着色=误差或值，返回 im 用于 colorbar。"""
    if y_true is None:
        m = ~np.isnan(y_pred[:, j])
        val_src = y_pred[m, j]
    else:
        m = ~np.isnan(y_true[:, j])
        val_src = y_pred[m, j] - y_true[m, j]
    if m.sum() == 0:
        ax.set_visible(False)
        return None
    x = p.x[m, 0] * 1e3          # mm
    y = p.x[m, 1] * 1e3          # mm
    val = val_src
    if vmin is None:
        vmin = float(np.nanpercentile(val, 0.5))
    if vmax is None:
        vmax = float(np.nanpercentile(val, 99.5))
    sc = ax.scatter(x, y, c=val, s=1.2, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    for b in DOM_BOUNDS_MM:
        ax.axhline(b, color="k", lw=0.4, alpha=0.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 2.62)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return sc


def err_stats(y_pred, y_true, j):
    m = ~np.isnan(y_true[:, j])
    e = y_pred[m, j] - y_true[m, j]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def plot_s9(dm, model, device, outdir):
    p, yp = predict(dm, model, "steady_test", device)
    yt = p.y
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, var in enumerate(["T", "phis", "htQ"]):
        j = dm.target_vars.index(var)
        rmse, mae = err_stats(yp, yt, j)
        sc = field_axes(axes[0, col], p, yp, yt, var, j, title=f"COMSOL {VAR_LABEL[var]}")
        if sc:
            fig.colorbar(sc, ax=axes[0, col], fraction=0.046)
        sc = field_axes(axes[1, col], p, yp, yt, var, j,
                        title=f"PINN 误差 (RMSE={rmse:.3g}, MAE={mae:.3g})", cmap="RdBu_r")
        if sc:
            fig.colorbar(sc, ax=axes[1, col], fraction=0.046)
    fig.suptitle("S9 稳态外推（T_in=1123 K）：COMSOL vs PINN 误差场", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "s9_fields_T_phis_htQ.png"), dpi=130)
    plt.close(fig)
    print("[plot] saved s9_fields_T_phis_htQ.png")


def plot_t3_t1200(dm, model, device, outdir):
    p, yp = predict(dm, model, "transient_test", device)
    yt = p.y
    sel = np.isclose(p.t, 1200.0)
    ps = _subset(p, sel); yps = yp[sel]; yts = yt[sel]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, var in enumerate(["T", "phis", "htQ"]):
        j = dm.target_vars.index(var)
        rmse, mae = err_stats(yps, yts, j)
        sc = field_axes(axes[0, col], ps, yps, yts, var, j, title=f"COMSOL {VAR_LABEL[var]} (t=1200 s)")
        if sc:
            fig.colorbar(sc, ax=axes[0, col], fraction=0.046)
        sc = field_axes(axes[1, col], ps, yps, yts, var, j,
                        title=f"PINN 误差 (RMSE={rmse:.3g}, MAE={mae:.3g})", cmap="RdBu_r")
        if sc:
            fig.colorbar(sc, ax=axes[1, col], fraction=0.046)
    fig.suptitle("T3 瞬态外推末态（0.2 V 阶跃, t=1200 s）：COMSOL vs PINN 误差场", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "t3_t1200_fields.png"), dpi=130)
    plt.close(fig)
    print("[plot] saved t3_t1200_fields.png")


def _subset(p, sel):
    from data.dataset import PointPool
    return PointPool(p.x[sel], p.y[sel], p.dom[sel], p.case_id[sel], p.t[sel], p.n_out)


def plot_t3_snapshots(dm, model, device, outdir, ts=(100.0, 360.0, 1200.0)):
    p, yp = predict(dm, model, "transient_test", device)
    yt = p.y
    j = dm.target_vars.index("T")
    fig, axes = plt.subplots(len(ts), 3, figsize=(15, 3.4 * len(ts)))
    for r, t in enumerate(ts):
        sel = np.isclose(p.t, t)
        ps = _subset(p, sel); yps = yp[sel]; yts = yt[sel]
        rmse, mae = err_stats(yps, yts, j)
        sc = field_axes(axes[r, 0], ps, yps, None, "T", j, title=f"PINN T (t={t:.0f} s)")
        if sc:
            fig.colorbar(sc, ax=axes[r, 0], fraction=0.046)
        sc = field_axes(axes[r, 1], ps, yps, yts, "T", j, title=f"COMSOL T (t={t:.0f} s)")
        if sc:
            fig.colorbar(sc, ax=axes[r, 1], fraction=0.046)
        sc = field_axes(axes[r, 2], ps, yps, yts, "T", j,
                        title=f"误差 (RMSE={rmse:.2f} K, MAE={mae:.2f} K)", cmap="RdBu_r")
        if sc:
            fig.colorbar(sc, ax=axes[r, 2], fraction=0.046)
    fig.suptitle("T3 瞬态外推：温度场时刻对比（PINN | COMSOL | 误差）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "t3_T_snapshots.png"), dpi=130)
    plt.close(fig)
    print("[plot] saved t3_T_snapshots.png")


def animate_t3_T(dm, model, device, outdir):
    """T3 全时间步温度场 GIF。"""
    from PIL import Image
    p, yp = predict(dm, model, "transient_test", device)
    yt = p.y
    j = dm.target_vars.index("T")
    ts = sorted(np.unique(p.t))
    vmin = float(np.nanmin(yt[:, j])); vmax = float(np.nanmax(yt[:, j]))
    frames = []
    for t in ts:
        sel = np.isclose(p.t, t)
        ps = _subset(p, sel)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        sc = field_axes(ax, ps, yp[sel], yt[sel], "T", j, vmin=vmin, vmax=vmax,
                        title=f"T3 T [K]  t={t:.0f} s")
        fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout()
        tmp = os.path.join(outdir, "_frame.png")
        fig.savefig(tmp, dpi=90)
        plt.close(fig)
        frames.append(Image.open(tmp).convert("RGB"))
        os.remove(tmp)
    out = os.path.join(outdir, "t3_T_animation.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=350, loop=0)
    print(f"[plot] saved {out} ({len(ts)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_m4b.yaml")
    ap.add_argument("--ckpt", default="runs/stage3_pde_m4b/best.pt")
    ap.add_argument("--cases", default="all", help="all / s9 / t3_t1200 / t3_snaps")
    ap.add_argument("--animate", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = SOFCDataModule(cfg, device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_net(cfg, dm.n_out, n_cont=dm.n_cont).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    outdir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "plots")
    os.makedirs(outdir, exist_ok=True)
    print(f"[plot] ckpt={args.ckpt} out={outdir}")

    cases = {"s9", "t3_t1200", "t3_snaps"} if args.cases == "all" else {args.cases}
    if "s9" in cases:
        plot_s9(dm, model, device, outdir)
    if "t3_t1200" in cases:
        plot_t3_t1200(dm, model, device, outdir)
    if "t3_snaps" in cases:
        plot_t3_snapshots(dm, model, device, outdir)
    if args.animate:
        animate_t3_T(dm, model, device, outdir)
    print("[plot] done")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""M1-M4 训练脚本：稳态/瞬态数据 + PDE 物理约束（能量/动量/组分/电荷守恒）。

用法：
    conda activate sofc
    python train.py --config config.yaml            # 稳态
    python train.py --config config_m4.yaml         # 瞬态扩展（use_transient: true）
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import random
import hashlib
from pathlib import Path

from checkpointing import (atomic_save, data_identity, code_identity, snapshot, restore, PauseRequest, training_config)

import numpy as np
import torch
import torch.nn as nn
import yaml

from data.dataset import SOFCDataModule, CONT_IDS
from data.split import DOMAIN_ORDER
from models.pinn import BaselineMLP
from losses.losses import MaskedDataLoss


def setup_device(cfg: dict) -> torch.device:
    dev = str(cfg["train"]["device"])
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ramp_weight(ep1: int, start: int, end: int, target: float) -> float:
    """课程权重：start 前为 0，end 后为目标值，中间线性爬升。"""
    if ep1 < start:
        return 0.0
    if ep1 >= end:
        return float(target)
    return float(target) * (ep1 - start) / max(1, end - start)


def marked_batches(iterator):
    """One-batch lookahead, preserving the original ordered sample sequence."""
    iterator = iter(iterator)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for following in iterator:
        yield (*current, False)
        current = following
    yield (*current, True)


def model_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def build_model(cfg: dict, n_out: int, n_cont: int = len(CONT_IDS)) -> nn.Module:
    m = cfg["model"]
    return BaselineMLP(
        n_cont=n_cont,
        n_domain=5,
        n_out=n_out,
        hidden=int(m["hidden"]),
        layers=int(m["layers"]),
        dropout=float(m["dropout"]),
        fourier=bool(m["fourier"]),
        n_freq=int(m["n_freq"]),
        f_min=m.get("f_min", 1.0),
        f_max=m.get("f_max", 512.0),
        fourier_cols=m.get("fourier_cols", [0, 1, 2]),
    )


def evaluate_pool(model, X, Y, M, device, chunk=32768):
    """全池评估：返回加权损失与各变量掩码 RMSE（归一化空间）。"""
    model.eval()
    n = X.shape[0]
    n_out = Y.shape[1]
    sq_err = torch.zeros(n_out, device=device)
    wsum = torch.zeros(n_out, device=device)
    with torch.no_grad():
        for i in range(0, n, chunk):
            xb, yb, mb = X[i:i + chunk], Y[i:i + chunk], M[i:i + chunk]
            pb = model(xb)
            d2 = (pb - torch.nan_to_num(yb, nan=0.0)) ** 2 * mb
            sq_err += d2.sum(0)
            wsum += mb.sum(0)
    rmse = torch.sqrt(sq_err / wsum.clamp(min=1.0))
    return rmse


def validation_temperature_hotspot_score(model, X, pool, dm, device, chunk=32768):
    """Equal-case (T RMSE + absolute full-field Tmax bias) / 10 K, validation only."""
    model.eval()
    j = dm.target_vars.index("T")
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            predictions.append(model(X[i:i + chunk].to(device))[:, j].detach().cpu().numpy())
    predicted = np.concatenate(predictions) * dm.y_stats["T"]["std"] + dm.y_stats["T"]["mean"]
    true = pool.y[:, j]
    values = []
    for cid in np.unique(pool.case_id):
        use = pool.case_id == cid
        error = predicted[use] - true[use]
        values.append((np.sqrt(np.mean(error ** 2)) + abs(predicted[use].max() - true[use].max())) / 10.0)
    return float(np.mean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config 中的 epochs")
    ap.add_argument("--init", default=None, help="pretrained checkpoint for warm start")
    ap.add_argument("--paired-init", help="Shared random initial weights for a locked paired experiment")
    ap.add_argument("--seed", type=int, default=None, help="override seed in config")
    ap.add_argument("--run-dir", default=None, help="override run_dir in config")
    ap.add_argument("--resume", help="Full state checkpoint, distinct from --init")
    ap.add_argument("--pause-file", help="Request graceful pause by creating this file")
    ap.add_argument("--stop-after-epoch", type=int, help="Pause after this completed epoch; keep total epoch budget")
    args = ap.parse_args()
    if args.resume and args.init:
        ap.error("--resume and --init are mutually exclusive")
    if args.paired_init and (args.resume or args.init):
        ap.error("--paired-init is distinct from warm start and resume")
    if args.resume and (args.seed is not None or args.epochs is not None):
        ap.error("Resume keeps the original seed and total epoch budget")
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_ck.get("resume_schema") != 1:
            ap.error("This is a weight-only checkpoint; use --init in a new run directory")


    if resume_ck is not None:
        cfg = copy.deepcopy(resume_ck["config"])
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                supplied = yaml.safe_load(f)
            if training_config(supplied) != training_config(cfg):
                ap.error("Supplied config differs from the checkpoint")
    else:
        with open(args.config or "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.run_dir:
        cfg["paths"]["run_dir"] = args.run_dir

    from data.dataset import resolve_config_paths
    cfg = resolve_config_paths(cfg)
    device = setup_device(cfg)
    print(f"[env] device={device}  torch={torch.__version__}")
    set_seed(int(cfg["train"]["seed"]))

    run_dir = cfg["paths"]["run_dir"]
    folder = Path(run_dir)
    if folder.exists():
        allowed = {"train_stdout.log", "train_stderr.log", "train.pid", "PAUSE.request"}
        existing = {p.name for p in folder.iterdir()}
        if resume_ck is None and existing - allowed:
            raise FileExistsError("Run directory already contains results; choose --run-dir or --resume")
        if resume_ck is not None and folder.resolve() != Path(args.resume).resolve().parent:
            if existing - allowed:
                raise FileExistsError("A resumed fork requires a new empty --run-dir")
        elif resume_ck is not None and (folder / "latest.pt").exists():
            latest_epoch = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False).get("epoch", -1)
            if latest_epoch > resume_ck["epoch"]:
                raise ValueError("Checkpoint is older than this run; resume into a new --run-dir")
    os.makedirs(run_dir, exist_ok=True)

    # ---------------- 数据 ----------------
    t0 = time.time()
    dm = SOFCDataModule(cfg, device)
    print(f"[data] 构建完成, {time.time()-t0:.1f}s")
    Xv, Yv, Mv, validation_pool = dm.pool_tensors("steady_val")
    # Test feedback is disabled by default; model selection uses validation only.
    inspect_tests = bool(cfg["train"].get("evaluate_tests_during_training", False))
    Xt = Yt = Mt = Xtt = Ytt = Mtt = None
    if inspect_tests:
        Xt, Yt, Mt, _ = dm.pool_tensors("steady_test")
        if "transient_test" in dm.pools:
            Xtt, Ytt, Mtt, _ = dm.pool_tensors("transient_test")

    # ---------------- 模型/损失/优化 ----------------
    model = build_model(cfg, dm.n_out, n_cont=dm.n_cont).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__} 参数={n_params/1e6:.2f}M")

    # 数据损失权重 1/σ²
    var_w = torch.tensor([1.0 / dm.y_stats[v]["std"] ** 2 for v in dm.target_vars],
                         dtype=torch.float32, device=device)
    boost = cfg["data"].get("loss_var_weights", {})
    if boost:
        for j, v in enumerate(dm.target_vars):
            if v in boost:
                var_w[j] = var_w[j] * float(boost[v])
        print("[loss] var weight boost:", {v: float(boost[v]) for v in boost})
    # B2: ??????qav/htQ ?????????????/??????
    rel_cfg = cfg["data"].get("loss_rel", {})
    rel_idx = {}
    for v, lam in rel_cfg.items():
        if v in dm.target_vars:
            rel_idx[dm.target_vars.index(v)] = float(lam)
        else:
            print(f"[loss] WARN: loss_rel ?? {v} ?? target_vars")
    loss_fn = MaskedDataLoss(
        var_w,
        learnable=(cfg["train"]["loss_weight_mode"] == "learned"),
        relative_vars=rel_idx,
        lambda_rel=float(cfg["data"].get("lambda_rel", 1.0)),
        eps=float(cfg["data"].get("loss_rel_eps", 0.1)),
    )
    if rel_idx:
        print(f"[loss] relative vars (idx->lambda): {rel_idx}  lambda_rel={cfg['data'].get('lambda_rel', 1.0)} eps={cfg['data'].get('loss_rel_eps', 0.1)}")

    # warm start from pretrained weights (curriculum: data first, physics later)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        # warm start ???????? Fourier ???f_min/f_max ????????
        # ? buffer ?????????????????
        sd = {k: v for k, v in ck["model"].items() if not k.startswith("ff.freqs")}
        model.load_state_dict(sd, strict=False)
        print(f"[model] init from {args.init} (best_val={ck.get('best_val', float('nan')):.4f})"
              f"  freqs kept from config f_max={cfg['model'].get('f_max', 512.0)}")
    if args.paired_init:
        shared = torch.load(args.paired_init, map_location=device, weights_only=False)
        if shared.get('kind') != 'shared_random_initialization' or shared['seed'] != int(cfg['train']['seed']):
            raise ValueError('Paired initialization seed or kind mismatch')
        model.load_state_dict(shared['model'], strict=True)
        if model_state_hash(model.state_dict()) != shared['model_sha256']:
            raise ValueError('Paired initialization weights mismatch')
        print(f"[paired-init] seed={shared['seed']} model_sha256={shared['model_sha256']}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"]["warmup_epochs"])
    if cfg["train"].get("lr_schedule", "warmup") == "warmup_cosine":
        import math as _math
        def _lr_lambda(e: int) -> float:
            if e < warmup:
                return (e + 1) / max(1, warmup)
            frac = (e - warmup) / max(1, epochs - warmup)
            return 0.05 + 0.95 * 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, frac)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda e: min(1.0, (e + 1) / max(1, warmup)))

    # ---------------- 训练 ----------------
    # ---------------- PDE（M3 物理约束） ----------------
    pde = None
    pde_cfg = cfg.get("pde", {})
    if pde_cfg.get("enabled", False):
        from physics.pde import PDEResiduals
        pde = PDEResiduals(model, dm, device,
                              balance=bool(pde_cfg.get("balance", True)),
                              eq_weights=pde_cfg.get("eq_weights", {}),
                              hotspot_weights=pde_cfg.get("hotspot_weights", {}))
        print(f"[pde] enabled: colloc/dom={pde_cfg.get('collocation_per_domain')} "
              f"iface={pde_cfg.get('interface_per_iface')} "
              f"lambda_pde={pde_cfg.get('lambda_pde')} lambda_int={pde_cfg.get('lambda_int')} "
              f"ramp={pde_cfg.get('ramp_start_epoch')}->{pde_cfg.get('ramp_end_epoch')}")
    rng_pde = np.random.default_rng(int(cfg["train"]["seed"]) + 1)

    # ---------------- 训练 ----------------
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    batch_size = int(cfg["data"]["batch_size"])
    best_val = float("inf")
    log = []
    print(f"[train] epochs={epochs} batch={batch_size} lr={cfg['train']['lr']}")

    identity, code = data_identity(cfg, dm), code_identity()
    start_epoch = 0
    if resume_ck is not None:
        restore(resume_ck, model, loss_fn, opt, sched, pde, rng, rng_pde, cfg, identity, code, device)
        start_epoch = int(resume_ck["epoch"])
        best_val, log = resume_ck["best_val"], copy.deepcopy(resume_ck["history"])
        print(f"[resume] epoch {start_epoch}, lr={opt.param_groups[0]['lr']:.8g}")
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    dm.save_stats(run_dir)
    atomic_save(log, Path(run_dir) / "train_log.json", as_json=True)
    pause = PauseRequest(args.pause_file or str(Path(run_dir) / "PAUSE.request"))
    # Initial state also allows a pause before the first epoch.
    initial = snapshot(model, loss_fn, opt, sched, pde, rng, rng_pde, cfg,
                       start_epoch, best_val, log, identity, code, device)
    atomic_save(initial, Path(run_dir) / "latest.pt")
    if resume_ck is None:
        atomic_save(initial, Path(run_dir) / 'initial.pt')
        atomic_save({'model_sha256':model_state_hash(model.state_dict()),'seed':int(cfg['train']['seed'])},Path(run_dir)/'initial_identity.json',as_json=True)
    if pause.pending():
        pause.finish(run_dir, start_epoch)
        pause.close()
        return
    for ep in range(start_epoch, epochs):
        model.train()
        ep_loss, n_b = 0.0, 0
        tb = time.time()
        joint = cfg['train'].get('physics_update_mode', 'separate') == 'joint_last_batch'
        joint_pending = False
        updates = batches = 0
        iterator = dm.iter_epoch(batch_size, rng)
        iterator = marked_batches(iterator) if joint else ((*batch, False) for batch in iterator)
        for Xb, Yb, Mb, last in iterator:
            batches += 1
            pred = model(Xb)
            loss = loss_fn(pred, Yb, Mb)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite data loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if joint and pde is not None and last:
                joint_pending = True
            else:
                if cfg["train"]["grad_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]), error_if_nonfinite=True)
                opt.step();updates += 1
            ep_loss += float(loss) * len(Xb)
            n_b += len(Xb)
        ep_pde, ep_int, ep_bc = None, None, None
        ep_pde_t, ep_ic, ep_final, lpre = None, None, None, None
        if pde is not None:
            lam = ramp_weight(ep + 1, int(pde_cfg["ramp_start_epoch"]),
                              int(pde_cfg["ramp_end_epoch"]), float(pde_cfg["lambda_pde"]))
            lam_int = float(pde_cfg.get("lambda_int", 0.05))
            lam_bc = ramp_weight(ep + 1, int(pde_cfg["ramp_start_epoch"]),
                                 int(pde_cfg["ramp_end_epoch"]),
                                 float(pde_cfg.get("lambda_bc", 0.05)))
            if lam > 0.0 or lam_int > 0.0 or lam_bc > 0.0:
                # 逐域/逐界面 backward：二阶导图大，立即释放以省显存
                if not joint_pending:
                    opt.zero_grad(set_to_none=True)
                n_coll = pde_cfg.get("collocation_per_domain", 512)
                if isinstance(n_coll, dict):
                    n_coll = {d: int(v) for d, v in n_coll.items()}
                n_coll_t = pde_cfg.get("transient_collocation_per_domain", None)
                if isinstance(n_coll_t, dict):
                    n_coll_t = {d: int(v) for d, v in n_coll_t.items()}
                n_iface = int(pde_cfg.get("interface_per_iface", 256))
                lam_t = ramp_weight(ep + 1, int(pde_cfg["ramp_start_epoch"]),
                                    int(pde_cfg["ramp_end_epoch"]),
                                    float(pde_cfg.get("lambda_pde_t", 0.2)))
                lam_ic = float(pde_cfg.get("lambda_ic", 0.05))
                lam_final = float(pde_cfg.get("lambda_final", 0.0))
                lam_pre = ramp_weight(ep + 1, int(pde_cfg["ramp_start_epoch"]),
                                      int(pde_cfg["ramp_end_epoch"]),
                                      float(pde_cfg.get("lambda_pre", 0.0)))
                t_end = float(cfg["data"].get("t_scale_s", 1200.0))
                lpde_sum, lpde_t_sum, lic, lfinal = 0.0, 0.0, 0.0, 0.0
                lpre = None
                for _dom in DOMAIN_ORDER:
                    _l = pde.domain_loss(_dom, rng_pde, n_coll,
                                         use_bv=bool(pde_cfg.get("use_bv", True)))
                    (lam * _l).backward()
                    lpde_sum += float(_l.detach())
                    if n_coll_t:
                        _lt = pde.domain_loss(_dom, rng_pde, n_coll_t,
                                              use_bv=bool(pde_cfg.get("use_bv", True)),
                                              transient=True)
                        (lam_t * _lt).backward()
                        lpde_t_sum += float(_lt.detach())
                lint = pde.interface_loss(rng_pde, n_iface, scale=lam_int)
                if lam_ic > 0.0 and n_coll_t:
                    lic = pde.ic_loss(rng_pde, n_coll_t,
                                      use_bv=bool(pde_cfg.get("use_bv", True)),
                                      scale=lam_ic,
                                      n_hotspot=pde_cfg.get("ic_hotspot_points", 0))
                if lam_pre > 0.0 and n_coll_t:
                    lpre = pde.pre_step_loss(rng_pde, n_coll_t,
                                             use_bv=bool(pde_cfg.get("use_bv", True)),
                                             scale=lam_pre)
                if lam_final > 0.0 and n_coll_t:
                    lfinal = pde.ic_loss(rng_pde, n_coll_t,
                                         use_bv=bool(pde_cfg.get("use_bv", True)),
                                         scale=lam_final, t_fixed=t_end)
                lbc = 0.0
                if lam_bc > 0.0:
                    n_edge = int(pde_cfg.get("boundary_per_edge", 256))
                    lbc = pde.boundary_loss(rng_pde, n_edge, scale=lam_bc)
                if not joint:
                    if cfg["train"]["grad_clip"]:
                        nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]), error_if_nonfinite=True)
                    opt.step();updates += 1
                ep_pde, ep_int, ep_bc = lpde_sum / len(DOMAIN_ORDER), lint, lbc
                ep_pde_t = lpde_t_sum / len(DOMAIN_ORDER) if n_coll_t else None
                ep_ic = lic if n_coll_t else None
                ep_final = lfinal if n_coll_t else None
        if joint_pending:
            if cfg['train']['grad_clip']:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg['train']['grad_clip']),error_if_nonfinite=True)
            opt.step();updates += 1
        if joint and updates != batches:
            raise AssertionError('Joint mode requires exactly one optimizer update per supervised batch')
        sched.step()

        msg = f"[train] ep={ep+1:3d}/{epochs} loss={ep_loss/max(1,n_b):.5f} lr={opt.param_groups[0]['lr']:.2e} {time.time()-tb:.1f}s"
        if ep_pde is not None:
            msg += f" pde={ep_pde:.4f}/{ep_int:.4f}/{ep_bc:.4f}"
            if ep_pde_t is not None:
                msg += f" pde_t={ep_pde_t:.4f}/ic={ep_ic:.4f}/final={ep_final:.4f}"
                if lpre is not None:
                    msg += f"/pre={lpre:.4f}"
        improved = False
        selected_score = aggregate_validation = None
        if (ep + 1) % int(cfg["train"]["eval_every"]) == 0 or ep == epochs - 1:
            vrmse = evaluate_pool(model, Xv, Yv, Mv, device)
            trmse = evaluate_pool(model, Xt, Yt, Mt, device) if inspect_tests else None
            # best-ckpt selection: T-weighted (gates are T-based; multi-var mean dilutes T)
            bw = float(cfg["train"].get("best_t_weight", 1.0))
            wsel = torch.ones_like(vrmse)
            wsel[dm.target_vars.index("T")] = bw
            aggregate_validation = float((vrmse.pow(2) * wsel).mean()) ** 0.5
            metric = cfg["train"].get("selection_metric", "normalized_t_weighted")
            if metric == "temperature_hotspot":
                selected_score = validation_temperature_hotspot_score(model, Xv, validation_pool, dm, device)
            elif metric == "normalized_t_weighted":
                selected_score = aggregate_validation
            else:
                raise ValueError(f"Unknown selection_metric: {metric}")
            vloss = selected_score ** 2
            msg += f" | val_rmse_mean={aggregate_validation:.5f} selection_score={selected_score:.5f}"
            if Xtt is not None:
                ttrmse = evaluate_pool(model, Xtt, Ytt, Mtt, device)
                msg += f" ttest_rmse={float(ttrmse.pow(2).mean())**0.5:.5f}"
            for j, v in enumerate(dm.target_vars):
                msg += f"\n    {v:>8s}: val_rmse={vrmse[j].item():.4f}"
                if trmse is not None:
                    msg += f" test_rmse={trmse[j].item():.4f}"
            if selected_score < best_val ** 0.5 - 1e-12:
                best_val = vloss
                improved = True
                msg += " | SAVED(best)"
        print(msg)

        log.append({"epoch": ep + 1, "loss": ep_loss / max(1, n_b),
                     'supervised_points':n_b,'supervised_batches':batches,'optimizer_updates':updates,
                     'sampling_sha256':getattr(dm,'epoch_sampling_sha256',None),'epoch_wall_seconds':time.time()-tb,
                     "val_rmse_mean": aggregate_validation, "selection_score": selected_score,
                     "pde_loss": ep_pde, "pde_t_loss": ep_pde_t, "ic_loss": ep_ic,
                     "pre_loss": lpre, "final_loss": ep_final, "int_loss": ep_int,
                     "bc_loss": ep_bc})
        state = snapshot(model, loss_fn, opt, sched, pde, rng, rng_pde, cfg,
                         ep + 1, best_val, log, identity, code, device)
        atomic_save(state, Path(run_dir) / "latest.pt")
        atomic_save(log, Path(run_dir) / "train_log.json", as_json=True)
        if improved:
            atomic_save(state, Path(run_dir) / "best.pt")
        if pause.pending() or (args.stop_after_epoch is not None and ep + 1 >= args.stop_after_epoch):
            pause.finish(run_dir, ep + 1)
            pause.close()
            return

    # final.pt is also a resumable full-state checkpoint, and retains legacy model/epoch keys.
    state = snapshot(model, loss_fn, opt, sched, pde, rng, rng_pde, cfg,
                     epochs, best_val, log, identity, code, device)
    atomic_save(state, Path(run_dir) / "final.pt")
    pause.close()
    atomic_save({"status": "complete", "completed_epoch": epochs}, Path(run_dir) / "pause_status.json", as_json=True)
    print(f"[done] best_val_rmse={best_val**0.5:.5f} -> {run_dir}")


if __name__ == "__main__":
    main()

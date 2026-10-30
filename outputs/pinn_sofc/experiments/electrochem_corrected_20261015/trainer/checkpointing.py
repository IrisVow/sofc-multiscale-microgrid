"""Atomic epoch-boundary checkpoints. Resume is distinct from weight-only init."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile

import numpy as np
import torch

SCHEMA = 3


def atomic_save(value, filename, *, as_json=False):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        mode = "w" if as_json else "wb"
        with os.fdopen(fd, mode, **({"encoding": "utf-8"} if as_json else {})) as stream:
            if as_json:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=True)
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_config(cfg):
    from data.dataset import resolve_config_paths
    value = resolve_config_paths(copy.deepcopy(cfg))
    value["paths"].pop("run_dir", None)
    return value


def data_identity(cfg, dm):
    paths = cfg["paths"]
    files = {}
    for key in ("merged_steady", "merged_transient"):
        if key == "merged_transient" and not cfg["data"].get("use_transient"):
            continue
        if key in paths:
            path = Path(paths["data_root"]) / paths[key]
            files[key] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    if cfg["data"].get("initial_temperature_file"):
        path=Path(cfg["data"]["initial_temperature_file"]);files["initial_temperature_file"]={"bytes":path.stat().st_size,"sha256":file_hash(path)}
    return {"files": files, "x_stats": dm.x_stats, "y_stats": dm.y_stats,
            "target_vars": dm.target_vars, "masks": dm.masks}


def code_identity():
    root = Path(__file__).resolve().parent
    names = ["train.py", "checkpointing.py", "data/dataset.py", "data/split.py",
             "models/pinn.py", "models/fourier.py", "losses/losses.py", "physics/pde.py",
             "physics/params.py", "physics/domains.py", "physics/waveforms.py", "physics/equations_core.py", "physics/gradient_budget.py", "physics/thermal_operators.py", "physics/electrochem.py"]
    return {name: file_hash(root / name) for name in names}


def capture_rng(rng, rng_pde):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampling": copy.deepcopy(rng.bit_generator.state),
            "collocation": copy.deepcopy(rng_pde.bit_generator.state)}


def restore_rng(state, rng, rng_pde):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact resume requires the original runtime")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])
    rng.bit_generator.state = state["sampling"]
    rng_pde.bit_generator.state = state["collocation"]


def runtime_identity(device):
    return {"torch": str(torch.__version__), "device_type": device.type,
            "cuda": torch.version.cuda,
            "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def snapshot(model, loss_fn, opt, sched, pde, rng, rng_pde, cfg, epoch, best_val,
             history, identity, code, device):
    return {"resume_schema": SCHEMA, "model": model.state_dict(),
            "loss_state": loss_fn.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch, "best_val": best_val,
            "config": copy.deepcopy(cfg), "history": copy.deepcopy(history),
            "pde_ema": copy.deepcopy(pde._ema) if pde is not None else None,
            "rng": capture_rng(rng, rng_pde), "data_identity": identity,
            "code_identity": code, "runtime": runtime_identity(device)}


def restore(ck, model, loss_fn, opt, sched, pde, rng, rng_pde, cfg, identity, code, device):
    if ck.get("resume_schema") != SCHEMA:
        raise ValueError("Weight-only best.pt/final.pt cannot fully resume; use --init for a new run")
    if training_config(ck["config"]) != training_config(cfg):
        raise ValueError("Training configuration changed; resume preserves the original epoch budget and schedule")
    if ck["data_identity"] != identity:
        raise ValueError("Data files, normalization or masks changed; refusing inconsistent resume")
    if ck["code_identity"] != code:
        raise ValueError("Training implementation changed; refusing inconsistent resume")
    if ck["runtime"] != runtime_identity(device):
        raise ValueError("Runtime or deterministic computation settings changed")
    if len(ck["history"]) != ck["epoch"]:
        raise ValueError("Checkpoint history does not match its completed epoch")
    model.load_state_dict(ck["model"], strict=True)
    loss_fn.load_state_dict(ck["loss_state"], strict=True)
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    if pde is not None:
        pde._ema = copy.deepcopy(ck["pde_ema"])
    restore_rng(ck["rng"], rng, rng_pde)


class PauseRequest:
    def __init__(self, filename):
        self.path = Path(filename)
        self.requested = False
        self.previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, signum, frame):
        self.requested = True
        print("[pause] requested; finishing this epoch before saving latest.pt", flush=True)

    def pending(self):
        return self.requested or self.path.exists()

    def finish(self, run_dir, epoch):
        atomic_save({"status": "paused", "completed_epoch": epoch,
                     "resume_checkpoint": str(Path(run_dir) / "latest.pt")},
                    Path(run_dir) / "pause_status.json", as_json=True)
        if self.path.exists():
            os.replace(self.path, self.path.with_name(self.path.name + f".handled-{epoch}"))
        print(f"[paused] completed epoch {epoch}; resume with --resume {Path(run_dir) / 'latest.pt'}", flush=True)

    def close(self):
        signal.signal(signal.SIGINT, self.previous)

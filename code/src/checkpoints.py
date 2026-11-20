"""Trusted-local PyTorch recovery snapshots with atomic pointer updates.

Caller must checkpoint only after a complete optimizer step, and provide sampler
order/cursor. Exact mid-epoch replay requires num_workers=0 and no prefetched
augmentation. External backups remain the caller's responsibility.
"""
import hashlib
import random
from pathlib import Path
import uuid

from src.atomic import atomic_write, write_json


def save_checkpoint(run_dir, *, model, optimizer, scheduler, scaler,
                    epoch, global_step, sampler_state, config, manifest_hash,
                    metrics, best_metric, is_best=False):
    import numpy as np
    import torch
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "global_step": global_step,
        "sampler_state": sampler_state, "config": config,
        "manifest_hash": manifest_hash, "metrics": metrics,
        "best_metric": best_metric,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    # Unique immutable files preserve older results even if pointer updates fail.
    name = f"step-{global_step:09d}-{uuid.uuid4().hex[:8]}.pt"
    target = root / name
    atomic_write(target, lambda stream: torch.save(state, stream))
    pointer = {"file": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "epoch": epoch, "global_step": global_step}
    write_json(root / "latest.json", pointer)
    if is_best:
        write_json(root / "best.json", pointer)
    return target


def load_checkpoint(run_dir, *, model, optimizer, scheduler, scaler,
                    expected_manifest_hash, expected_config, which="latest"):
    import json
    import numpy as np
    import torch
    if which not in {"latest", "best"}:
        raise ValueError("which must be latest or best")
    root = Path(run_dir).resolve()
    pointer = json.loads((root / f"{which}.json").read_text())
    target = (root / pointer["file"]).resolve()
    if target.parent != root:
        raise ValueError("Checkpoint path escapes run directory")
    if hashlib.sha256(target.read_bytes()).hexdigest() != pointer["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    # Full optimizer/RNG recovery requires pickle: load only your trusted snapshots.
    state = torch.load(target, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or state["manifest_hash"] != expected_manifest_hash:
        raise ValueError("Checkpoint schema or dataset manifest mismatch")
    if state["config"] != expected_config:
        raise ValueError("Configuration changed; create a new run instead of resuming")
    for key, component in [("scheduler", scheduler), ("scaler", scaler)]:
        if (state[key] is None) != (component is None):
            raise ValueError(f"Recovery component mismatch: {key}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Exact CUDA recovery requires a compatible CUDA environment")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state

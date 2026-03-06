"""
train/utils.py — Training utility functions.

Provides:
    get_cosine_schedule_with_warmup  : LambdaLR scheduler with linear warmup + cosine decay
    save_checkpoint                  : Persist model/optimizer/scheduler state to disk
    load_checkpoint                  : Restore state from a saved checkpoint directory
    get_grad_norm                    : Compute total L2 gradient norm across all parameters
    setup_ddp                        : Initialise NCCL distributed process group
    cleanup_ddp                      : Tear down distributed process group
    is_main_process                  : True when this process is rank 0 (or non-distributed)
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a LambdaLR scheduler with:
      - Linear warmup: lr scales from 0 → 1 over [0, warmup_steps)
      - Cosine decay:  lr scales from 1 → min_lr_ratio over [warmup_steps, total_steps]

    Args:
        optimizer:     The wrapped optimizer.
        warmup_steps:  Number of linear-warmup steps.
        total_steps:   Total number of training steps.
        min_lr_ratio:  Minimum lr as a fraction of the peak lr (default 0.1).

    Returns:
        A LambdaLR scheduler instance.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase.
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # After total_steps, hold at min_lr_ratio.
        if current_step >= total_steps:
            return min_lr_ratio

        # Cosine decay phase.
        decay_steps = total_steps - warmup_steps
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale cosine output from [0, 1] into [min_lr_ratio, 1].
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    step: int,
    loss: float,
    path: str | Path,
    suffix: str | None = None,
) -> Path:
    """
    Save a training checkpoint to ``path/checkpoint-{step:07d}/``.

    Saves:
        - model.pt        : model state_dict
        - optimizer.pt    : optimizer state_dict
        - scheduler.pt    : scheduler state_dict
        - train_state.pt  : step and loss scalars
        - config.yaml     : model LMConfig (if the model exposes a ``.config`` attribute)

    Handles both plain ``nn.Module`` and DDP-wrapped models by unwrapping
    via ``.module`` when present.

    Args:
        model:     The model (plain or DDP-wrapped).
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        step:      Current training step (used in directory name).
        loss:      Current loss value (stored for reference).
        path:      Root checkpoint directory.

    Returns:
        Path to the created checkpoint sub-directory.
    """
    dir_name = f"checkpoint-{suffix}" if suffix else f"checkpoint-{step:07d}"
    ckpt_dir = Path(path) / dir_name
    tmp_dir = Path(path) / f".tmp_{dir_name}"

    # Write to temp directory first for crash safety
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model: torch.nn.Module = getattr(model, "module", model)

    torch.save(raw_model.state_dict(), tmp_dir / "model.pt")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")

    import random as _random
    train_state = {
        "step": step,
        "loss": loss,
        "rng_state": {
            "python": _random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }
    torch.save(train_state, tmp_dir / "train_state.pt")

    # Persist the model config when available.
    if hasattr(raw_model, "config"):
        cfg = raw_model.config
        if hasattr(cfg, "to_dict"):
            config_dict = cfg.to_dict()
        else:
            # Fallback: try __dict__ for plain dataclasses.
            config_dict = {
                k: v for k, v in vars(cfg).items() if not k.startswith("_")
            }
        with open(tmp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Atomic swap: rename old → trash, tmp → final, delete trash
    trash_dir = Path(path) / f".trash_{dir_name}"
    if trash_dir.exists():
        shutil.rmtree(trash_dir)
    if ckpt_dir.exists():
        ckpt_dir.rename(trash_dir)
    tmp_dir.rename(ckpt_dir)
    if trash_dir.exists():
        shutil.rmtree(trash_dir)

    # Clean up old checkpoints (keep recent N + best)
    cleanup_old_checkpoints(Path(path))

    return ckpt_dir


def cleanup_old_checkpoints(path: Path, keep: int = 5) -> None:
    """Remove old checkpoints, keeping the most recent `keep` plus checkpoint-best."""
    ckpts = sorted(
        [d for d in path.glob("checkpoint-[0-9]*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    for old in ckpts[:-keep]:
        shutil.rmtree(old)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
) -> Tuple[int, float]:
    """
    Load a checkpoint from a directory created by :func:`save_checkpoint`.

    The model weights are always restored.  Optimizer and scheduler states are
    only restored when the corresponding objects are provided.

    Args:
        path:      Path to the checkpoint directory (e.g. ``checkpoints/checkpoint-0001000``).
        model:     Model to load weights into (plain or DDP-wrapped).
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional LR scheduler to restore state into.

    Returns:
        ``(step, loss)`` — the training step and loss recorded at save time.
    """
    ckpt_dir = Path(path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Unwrap DDP model if necessary.
    raw_model: torch.nn.Module = getattr(model, "module", model)

    # Determine the device the model lives on.
    try:
        device = next(raw_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Validate architecture match between checkpoint and current model
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists() and hasattr(raw_model, "config"):
        import yaml as _yaml
        with open(config_path, "r") as f:
            ckpt_cfg = _yaml.safe_load(f)
        if "model" in ckpt_cfg and isinstance(ckpt_cfg["model"], dict):
            ckpt_cfg = ckpt_cfg["model"]
        cur_cfg = raw_model.config.to_dict()
        critical = [
            "d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
            "use_hybrid", "hybrid_pattern", "mamba_d_state", "mamba_expand",
            "mamba_head_dim", "mamba_n_groups", "mamba_d_ffn",
        ]
        mismatches = [
            f"  {k}: ckpt={ckpt_cfg.get(k)} vs cur={cur_cfg.get(k)}"
            for k in critical
            if ckpt_cfg.get(k) is not None and cur_cfg.get(k) is not None
            and ckpt_cfg.get(k) != cur_cfg.get(k)
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint architecture mismatch!\n" + "\n".join(mismatches)
                + "\nUse matching config or start fresh."
            )

    raw_model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=True)
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler.pt", map_location=device, weights_only=True)
        )

    train_state = torch.load(
        ckpt_dir / "train_state.pt", map_location="cpu", weights_only=True
    )
    step: int = int(train_state["step"])
    loss: float = float(train_state["loss"])

    # Restore RNG states if available (for exact resume reproducibility)
    rng_state = train_state.get("rng_state")
    if rng_state is not None:
        import random as _random
        try:
            _random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        except Exception as e:
            print(f"[WARN] RNG state restore failed (non-fatal): {e}")

    return step, loss


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def get_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 norm of all parameter gradients.

    Uses a single GPU kernel + one GPU-CPU sync instead of one sync per
    parameter (the naive loop approach).  Only parameters with non-None
    ``.grad`` attribute contribute.

    Args:
        model: The model (plain or DDP-wrapped).

    Returns:
        Scalar float — the global gradient L2 norm.
    """
    raw_model: torch.nn.Module = getattr(model, "module", model)
    grads = [p.grad.detach().float() for p in raw_model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    # Stack individual norms and compute the L2 norm of norms — single sync.
    return torch.stack([g.norm(2) for g in grads]).norm(2).item()


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Initialise the NCCL distributed process group for DDP training.

    Reads ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` from the environment
    (set automatically by ``torchrun``).

    Returns:
        ``(rank, local_rank, world_size, device)``
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Limit CPU thread count per process to avoid contention across 8 ranks.
    # 72 cores / 8 ranks = 9; use 4 to leave headroom for DataLoader workers.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    import datetime as _dt
    dist.init_process_group(
        backend="nccl",
        timeout=_dt.timedelta(seconds=7200),  # 2h for large checkpoint loads
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    """Tear down the distributed process group (call at end of training)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Return ``True`` when this process is rank 0 or when running without DDP.

    Reads the ``RANK`` environment variable; if it is absent the process is
    assumed to be the sole process (rank 0).
    """
    return int(os.environ.get("RANK", "0")) == 0

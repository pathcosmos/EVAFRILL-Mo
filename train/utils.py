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

    state_dict = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    # Auto-convert TransformerEngine (FP8) state dict when loading into non-TE model
    has_te_keys = any("_extra_state" in k or ".fc1_weight" in k for k in state_dict)
    has_std_keys = any(".gate_proj.weight" in k for k in state_dict)
    if has_te_keys and not has_std_keys:
        from model.transformer import LLM
        state_dict = LLM._convert_te_state_dict(state_dict)
    raw_model.load_state_dict(state_dict)

    optimizer_restored = False
    if optimizer is not None:
        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            try:
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True)
                )
                optimizer_restored = True
            except (RuntimeError, ValueError) as e:
                print(f"[WARN] Could not restore optimizer state (TE→non-TE migration?): {e}")
                # Attempt partial momentum restoration: match parameters by shape.
                try:
                    saved_opt = torch.load(opt_path, map_location=device, weights_only=True)
                    saved_states = list(saved_opt.get("state", {}).values())
                    cur_params = [p for group in optimizer.param_groups for p in group["params"]]
                    matched = 0
                    si = 0  # index into saved_states
                    for pi, p in enumerate(cur_params):
                        if si >= len(saved_states):
                            break
                        ss = saved_states[si]
                        if (
                            "exp_avg" in ss
                            and ss["exp_avg"].shape == p.shape
                            and "exp_avg_sq" in ss
                            and ss["exp_avg_sq"].shape == p.shape
                        ):
                            optimizer.state[p]["step"] = ss.get("step", torch.tensor(0))
                            optimizer.state[p]["exp_avg"] = ss["exp_avg"].to(device=device, dtype=p.dtype)
                            optimizer.state[p]["exp_avg_sq"] = ss["exp_avg_sq"].to(device=device, dtype=p.dtype)
                            matched += 1
                            si += 1
                        else:
                            # Shape mismatch — log and skip this saved state slot.
                            print(
                                f"[WARN]   param[{pi}] shape={tuple(p.shape)} vs "
                                f"saved state[{si}] exp_avg shape="
                                f"{tuple(ss['exp_avg'].shape) if 'exp_avg' in ss else 'N/A'} — skipping"
                            )
                            si += 1  # advance saved pointer regardless
                    print(
                        f"[WARN] Partial optimizer restore: {matched}/{len(cur_params)} params "
                        f"matched by shape. Unmatched params start with zero momentum."
                    )
                    optimizer_restored = True  # partial restore is better than nothing
                except Exception as e2:
                    print(f"[WARN] Partial optimizer restore also failed: {e2}")
                    print("[WARN] Optimizer starts fully fresh.")

    if scheduler is not None:
        sched_restored = False
        sched_path = ckpt_dir / "scheduler.pt"
        if sched_path.exists():
            try:
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device, weights_only=True)
                )
                sched_restored = True
            except (RuntimeError, ValueError) as e:
                print(f"[WARN] Could not restore scheduler state: {e}")
        if not sched_restored:
            # Fast-forward the scheduler to the correct step so that LR
            # reflects the resume position rather than replaying warmup from 0.
            # train_state.pt has not been read yet at this point, so we derive
            # the step from the checkpoint directory name as a fallback.
            resume_step: Optional[int] = None
            try:
                # Directory names like "checkpoint-0010000"
                dir_stem = ckpt_dir.name  # e.g. "checkpoint-0010000"
                numeric = dir_stem.split("-")[-1]
                if numeric.isdigit():
                    resume_step = int(numeric)
            except Exception:
                pass
            if resume_step is not None and resume_step > 0:
                print(
                    f"[WARN] Fast-forwarding scheduler to step {resume_step} "
                    f"(skipping {resume_step} no-op scheduler.step() calls)."
                )
                # LambdaLR tracks last_epoch internally; advance it without
                # touching the optimizer (use_count trick: call step() in a loop
                # but only update last_epoch, not optimizer state).
                scheduler.last_epoch = resume_step - 1
                scheduler.step()  # advances to resume_step and recomputes _last_lr
                print(
                    f"[WARN] Scheduler fast-forwarded: last_epoch={scheduler.last_epoch}, "
                    f"lr={scheduler.get_last_lr()}"
                )

    # weights_only=False: train_state contains numpy RNG arrays for exact resume
    train_state = torch.load(
        ckpt_dir / "train_state.pt", map_location="cpu", weights_only=False
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

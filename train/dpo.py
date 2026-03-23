"""
train/dpo.py — Direct Preference Optimization (DPO) training.

Native DPO implementation (no TRL dependency) for EVAFRILL-Mo hybrid models.
Supports LoRA adapters for memory-efficient training on single GPU.

Launch:
    python train/dpo.py \
        --sft_checkpoint checkpoints/3b_sft_v2/checkpoint-best \
        --dpo_data data/preference/combined_preference.jsonl \
        --config configs/h100_mig/dpo_3b_1gpu.yaml \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import LLM
from model.lora import apply_lora, get_lora_params, merge_lora, save_lora
from data.dpo_dataset import DPODataset, dpo_collate_fn
from train.utils import (
    get_cosine_schedule_with_warmup,
    is_main_process,
    save_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO Training for EVAFRILL-Mo")

    # Paths
    parser.add_argument("--sft_checkpoint", type=Path, required=True,
                        help="Path to SFT checkpoint directory")
    parser.add_argument("--dpo_data", type=Path, required=True,
                        help="Path to preference JSONL data")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/3b_dpo"),
                        help="Output checkpoint directory")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)

    # Infra
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    args, _ = parser.parse_known_args()

    # Load YAML config
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("train", {})
        yaml_map = {
            "max_steps": "max_steps", "batch_size": "batch_size",
            "grad_accum_steps": "grad_accum", "lr": "lr",
            "weight_decay": "weight_decay", "warmup_steps": "warmup_steps",
            "beta": "beta", "max_length": "max_length",
            "save_interval": "save_interval", "log_interval": "log_interval",
            "use_lora": "use_lora", "lora_rank": "lora_rank", "lora_alpha": "lora_alpha",
        }
        defaults = {}
        for yk, ak in yaml_map.items():
            if yk in train_cfg:
                defaults[ak] = train_cfg[yk]
        if defaults:
            parser.set_defaults(**defaults)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log probabilities over non-masked tokens.

    Args:
        model: The LLM model
        input_ids: (B, T) token ids
        labels: (B, T) target ids, -1 for masked positions

    Returns:
        (B,) sum of log probs per sample
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)  # (B, T, V)

    # Shift: predict next token
    # logits[:, :-1] predicts labels[:, 1:]
    # But our labels already have the shifted targets (same as SFT convention)
    # labels[i] = token_id means input_ids[i] should predict labels[i]
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    # Gather log probs for target tokens
    # For each position, get log_prob of the label token
    mask = labels != -1  # (B, T)
    # Clamp labels for gather (replace -1 with 0, will be masked out)
    safe_labels = labels.clamp(min=0)  # (B, T)
    per_token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    per_token_logps = per_token_logps * mask.float()  # zero out masked positions

    return per_token_logps.sum(dim=-1)  # (B,)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards  # (B,)
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_rewards.detach().mean(), rejected_rewards.detach().mean()


def _resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer is not None:
        return Path(args.tokenizer)
    ckpt_tok = args.sft_checkpoint / "tokenizer.json"
    if ckpt_tok.exists():
        return ckpt_tok
    default_tok = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    if default_tok.exists():
        return default_tok
    raise FileNotFoundError("Cannot find tokenizer.json")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Validate checkpoint
    if not args.sft_checkpoint.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {args.sft_checkpoint}")

    # Load SFT model as policy
    print(f"Loading SFT model from {args.sft_checkpoint}...")
    model = LLM.from_pretrained(args.sft_checkpoint)
    model.config.use_fp8 = False  # H100 MIG: BF16 only
    model = model.to(device=device, dtype=torch.bfloat16)

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled")

    # Compute reference log probs BEFORE applying LoRA
    # (reference model = SFT model without LoRA)
    # We'll compute ref logps on-the-fly with LoRA disabled via a context manager
    # Actually for simplicity: precompute nothing, just use model without LoRA adapters
    # For LoRA DPO: ref_model is the base (original weights), policy is base + LoRA
    # Since LoRA is initialized to zero, at start policy = ref

    # Apply LoRA
    if args.use_lora:
        n_lora_params = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        lora_params = get_lora_params(model)
        print(f"[INFO] LoRA: {n_lora_params:,} trainable params")
    else:
        # Full fine-tuning (risky for VRAM)
        lora_params = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Tokenizer
    tokenizer_path = _resolve_tokenizer_path(args)
    print(f"Loading tokenizer from {tokenizer_path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Dataset
    train_dataset = DPODataset(
        data_path=args.dpo_data,
        tokenizer=tokenizer,
        max_seq_len=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dpo_collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Optimizer — only LoRA params if using LoRA
    if lora_params is not None:
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=torch.cuda.is_available(),
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=torch.cuda.is_available(),
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
    )

    # Resume
    start_step = 0
    if args.resume is not None:
        start_step, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from step {start_step}")

    # Checkpoint dir
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer
    dest_tok = args.checkpoint_dir / "tokenizer.json"
    if not dest_tok.exists():
        shutil.copy2(str(tokenizer_path), str(dest_tok))

    # Log file
    log_fh = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "a", encoding="utf-8", buffering=1)

    def log(msg: str, level: str = "INFO"):
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")

    # Banner
    eff_batch = args.batch_size * args.grad_accum
    log(f"{'='*60}")
    log(f"DPO Training — EVAFRILL-Mo 3B")
    log(f"  SFT ckpt: {args.sft_checkpoint}")
    log(f"  DPO data: {args.dpo_data} ({len(train_dataset):,} samples)")
    log(f"  LoRA: rank={args.lora_rank} alpha={args.lora_alpha}")
    log(f"  beta={args.beta}, lr={args.lr:.2e}, eff_batch={eff_batch}")
    log(f"  max_steps={args.max_steps}, max_length={args.max_length}")
    log(f"  device={device}")
    log(f"{'='*60}")

    # Training loop
    import time
    model.train()
    loader_iter = iter(train_loader)
    epoch = 0

    def next_batch():
        nonlocal loader_iter, epoch
        try:
            return next(loader_iter)
        except StopIteration:
            epoch += 1
            loader_iter = iter(train_loader)
            return next(loader_iter)

    shutdown_requested = False
    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log(f"Shutdown signal received ({signum})", "WARN")

    signal.signal(signal.SIGHUP, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    t0 = time.perf_counter()
    running_loss = 0.0
    running_chosen_reward = 0.0
    running_rejected_reward = 0.0
    log_step_count = 0

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(args.grad_accum):
            batch = next_batch()
            chosen_ids = batch[0].to(device, dtype=torch.long, non_blocking=True)
            chosen_labels = batch[1].to(device, dtype=torch.long, non_blocking=True)
            rejected_ids = batch[2].to(device, dtype=torch.long, non_blocking=True)
            rejected_labels = batch[3].to(device, dtype=torch.long, non_blocking=True)

            # Policy log probs (with LoRA active)
            policy_chosen_logps = compute_log_probs(model, chosen_ids, chosen_labels)
            policy_rejected_logps = compute_log_probs(model, rejected_ids, rejected_labels)

            # Reference log probs (LoRA disabled)
            # For LoRA: temporarily set lora scaling to 0
            with torch.no_grad():
                # Save and zero LoRA params
                if args.use_lora:
                    saved_B = []
                    for m in model.modules():
                        from model.lora import LoRALinear
                        if isinstance(m, LoRALinear):
                            saved_B.append(m.lora_B.data.clone())
                            m.lora_B.data.zero_()

                ref_chosen_logps = compute_log_probs(model, chosen_ids, chosen_labels)
                ref_rejected_logps = compute_log_probs(model, rejected_ids, rejected_labels)

                # Restore LoRA params
                if args.use_lora:
                    idx = 0
                    for m in model.modules():
                        from model.lora import LoRALinear
                        if isinstance(m, LoRALinear):
                            m.lora_B.data.copy_(saved_B[idx])
                            idx += 1

            # DPO loss
            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta,
            )

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        ).item()

        optimizer.step()
        scheduler.step()

        avg_loss = accum_loss / args.grad_accum
        running_loss += avg_loss
        running_chosen_reward += chosen_reward.item()
        running_rejected_reward += rejected_reward.item()
        log_step_count += 1

        # Shutdown check
        if shutdown_requested:
            log(f"Graceful shutdown at step {step + 1}", "WARN")
            save_checkpoint(model, optimizer, scheduler, step + 1, avg_loss, str(args.checkpoint_dir))
            if args.use_lora:
                save_lora(model, args.checkpoint_dir / f"lora-{step+1:07d}")
            break

        # Logging
        if (step + 1) % args.log_interval == 0:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            avg_l = running_loss / log_step_count
            avg_cr = running_chosen_reward / log_step_count
            avg_rr = running_rejected_reward / log_step_count
            margin = avg_cr - avg_rr
            lr = scheduler.get_last_lr()[0]
            mem_gb = torch.cuda.memory_allocated() / 1e9

            log(f"step {step+1:>6d} | loss {avg_l:.4f} | "
                f"margin {margin:.4f} (c={avg_cr:.3f} r={avg_rr:.3f}) | "
                f"lr {lr:.2e} | gnorm {grad_norm:.3f} | mem {mem_gb:.1f}GB")

            running_loss = 0.0
            running_chosen_reward = 0.0
            running_rejected_reward = 0.0
            log_step_count = 0
            t0 = t1

        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, step + 1, avg_loss, str(args.checkpoint_dir)
            )
            if args.use_lora:
                save_lora(model, args.checkpoint_dir / f"lora-{step+1:07d}")
            log(f"Checkpoint saved -> {ckpt_path}")

    # Final save
    final_path = save_checkpoint(
        model, optimizer, scheduler, args.max_steps, avg_loss, str(args.checkpoint_dir)
    )
    if args.use_lora:
        save_lora(model, args.checkpoint_dir / "lora-final")
        # Also merge and save merged model
        log("Merging LoRA weights into base model...")
        merge_lora(model)
        model.save_pretrained(args.checkpoint_dir / "checkpoint-merged")
        log(f"Merged model saved -> {args.checkpoint_dir / 'checkpoint-merged'}")

    log(f"DPO training complete. Final checkpoint -> {final_path}")

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()

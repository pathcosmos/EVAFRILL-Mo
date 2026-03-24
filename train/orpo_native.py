"""
train/orpo_native.py — ORPO (Odds Ratio Preference Optimization) training.

Native ORPO implementation (no TRL, no HuggingFace Trainer) for EVAFRILL-Mo
hybrid Mamba-2+Transformer models. Unlike DPO, ORPO requires NO reference model
and performs SFT + alignment in a single training stage, making it ideal for
starting from a raw pretrained checkpoint.

Reference: Hong et al., "ORPO: Monolithic Preference Optimization without
Reference Model" (2024), https://arxiv.org/abs/2403.07691

Loss:
    L_ORPO = L_SFT + λ * L_OR
    L_SFT  = CrossEntropy(chosen_logits, chosen_labels)
    L_OR   = -E[log σ(log(odds_chosen / odds_rejected))]
    odds(x) = P(x) / (1 - P(x)),  P(x) = exp(avg_log_prob(x))

Launch:
    python train/orpo_native.py \
        --pretrained_checkpoint checkpoints/3b_final/checkpoint-0319772 \
        --preference_data data/preference/combined_preference.jsonl \
        --config configs/h100_mig/dpo_3b_1gpu.yaml \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import datetime
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORPO Training for EVAFRILL-Mo")

    # Paths
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True,
                        help="Path to pretrained model checkpoint directory "
                             "(e.g. checkpoints/3b_final/checkpoint-0319772)")
    parser.add_argument("--preference_data", type=Path, required=True,
                        help="Path to preference JSONL data (prompt/chosen/rejected)")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/3b_orpo"),
                        help="Output checkpoint directory (default: checkpoints/3b_orpo)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from an existing ORPO checkpoint directory")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to tokenizer.json (auto-detected if omitted)")
    parser.add_argument("--log_file", type=Path, default=None,
                        help="Append logs to this file in addition to stdout")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config to load defaults from (train: section)")

    # ORPO hyperparameters
    parser.add_argument("--lambda_or", type=float, default=1.0,
                        help="ORPO odds-ratio loss weight λ (default: 1.0)")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Total optimisation steps (default: 3000)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-step micro-batch size (default: 1)")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Peak learning rate (default: 5e-6; higher than DPO because "
                             "ORPO starts from pretrained, not SFT)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA adapters for memory-efficient training (default: on)")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)

    # Infrastructure
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. cuda:0 (auto-detected if omitted)")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    args, _ = parser.parse_known_args()

    # Load YAML config and apply as defaults (CLI flags override YAML)
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("train", {})
        yaml_map = {
            "max_steps":       "max_steps",
            "batch_size":      "batch_size",
            "grad_accum_steps": "grad_accum",
            "lr":              "lr",
            "weight_decay":    "weight_decay",
            "warmup_steps":    "warmup_steps",
            "lambda_or":       "lambda_or",
            "max_length":      "max_length",
            "save_interval":   "save_interval",
            "log_interval":    "log_interval",
            "use_lora":        "use_lora",
            "lora_rank":       "lora_rank",
            "lora_alpha":      "lora_alpha",
        }
        defaults: dict = {}
        for yk, ak in yaml_map.items():
            if yk in train_cfg:
                defaults[ak] = train_cfg[yk]
        if defaults:
            parser.set_defaults(**defaults)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    """Find tokenizer.json: explicit flag > checkpoint dir > project default."""
    if args.tokenizer is not None:
        return Path(args.tokenizer)
    ckpt_tok = args.pretrained_checkpoint / "tokenizer.json"
    if ckpt_tok.exists():
        return ckpt_tok
    default_tok = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    if default_tok.exists():
        return default_tok
    raise FileNotFoundError(
        "Cannot find tokenizer.json. Provide --tokenizer or place it in the checkpoint dir."
    )


# ---------------------------------------------------------------------------
# ORPO loss
# ---------------------------------------------------------------------------

def get_avg_log_prob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute average log probability over non-masked (response) tokens.

    Args:
        logits: (B, T, V) raw model logits — already in float32.
        labels: (B, T) token ids; -1 marks prompt/padding positions to ignore.

    Returns:
        (B,) mean log probability over response tokens per sample.
        Returns 0 for samples where no response token is present (shouldn't
        happen with well-formed data, but guarded for safety).
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    mask = labels != -1                                # (B, T)  True = response token
    safe_labels = labels.clamp(min=0)                  # replace -1 with 0 for gather
    per_token_logps = log_probs.gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)                                      # (B, T)

    # Zero out masked positions
    per_token_logps = per_token_logps * mask.float()   # (B, T)

    # Average over response tokens; clamp denominator to avoid div-by-zero
    n_tokens = mask.float().sum(dim=-1).clamp(min=1.0)  # (B,)
    return per_token_logps.sum(dim=-1) / n_tokens        # (B,)


def compute_orpo_loss(
    model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    lambda_or: float = 1.0,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Compute ORPO loss = SFT loss + λ * OR loss.

    No reference model is needed.  The SFT loss trains the model to generate
    chosen responses; the OR loss simultaneously teaches the model to prefer
    chosen over rejected by maximising the log odds ratio.

    Args:
        model:            The policy model (frozen base + trainable LoRA).
        chosen_ids:       (B, T) token ids for chosen sequences.
        chosen_labels:    (B, T) labels for chosen; -1 on prompt tokens.
        rejected_ids:     (B, T) token ids for rejected sequences.
        rejected_labels:  (B, T) labels for rejected; -1 on prompt tokens.
        lambda_or:        Weight of the OR loss term (paper default = 1.0).
        vocab_size:       Vocabulary size for reshape; inferred from logits if None.

    Returns:
        (total_loss, sft_loss_scalar, or_loss_scalar)
    """
    # -----------------------------------------------------------------------
    # 1. Forward pass — chosen
    # -----------------------------------------------------------------------
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chosen_logits, _ = model(chosen_ids)   # (B, T, V)

    # Infer vocab size from logits if not given
    V = chosen_logits.size(-1) if vocab_size is None else vocab_size

    # SFT loss: next-token prediction on response positions only.
    # logits[:, :-1] predicts labels[:, 1:] (standard causal shift).
    sft_logits = chosen_logits[:, :-1].contiguous().reshape(-1, V).float()
    sft_targets = chosen_labels[:, 1:].contiguous().reshape(-1)

    # F.cross_entropy ignores index -1 via ignore_index; -1 covers prompt tokens
    # AND the last padding position shifted out of the window.
    sft_loss: torch.Tensor = F.cross_entropy(sft_logits, sft_targets, ignore_index=-1)

    # Average log-prob over response tokens (used for OR computation)
    # Labels are NOT shifted here — get_avg_log_prob handles the alignment
    # by using labels directly as targets at each position.
    chosen_avg_logp: torch.Tensor = get_avg_log_prob(chosen_logits.float(), chosen_labels)

    # -----------------------------------------------------------------------
    # 2. Forward pass — rejected
    # -----------------------------------------------------------------------
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        rejected_logits, _ = model(rejected_ids)  # (B, T, V)

    rejected_avg_logp: torch.Tensor = get_avg_log_prob(rejected_logits.float(), rejected_labels)

    # -----------------------------------------------------------------------
    # 3. Odds ratio loss
    #
    # odds(x)  = P(x) / (1 - P(x))
    # log odds = log P(x) - log(1 - P(x)) = log P(x) - log1p(-exp(log P(x)))
    #
    # We use log1p(-exp(·)) with clamping to keep values numerically stable:
    #   - avg_log_prob is always ≤ 0
    #   - exp(avg_log_prob) ∈ (0, 1] → 1 - exp ∈ [0, 1)
    #   - clamp to avoid log(0) when avg_log_prob ≈ 0 (very high confidence)
    # -----------------------------------------------------------------------
    # Clamp to (-33, -1e-6): upper bound avoids 1-exp≈0 → log(0); lower keeps
    # values finite (exp(-33) ≈ 5e-15, no underflow in float32).
    eps_low, eps_high = -33.0, -1e-6

    chosen_avg_logp_clamped   = chosen_avg_logp.clamp(eps_low, eps_high)
    rejected_avg_logp_clamped = rejected_avg_logp.clamp(eps_low, eps_high)

    log_odds_chosen   = chosen_avg_logp_clamped   - torch.log1p(-chosen_avg_logp_clamped.exp())
    log_odds_rejected = rejected_avg_logp_clamped - torch.log1p(-rejected_avg_logp_clamped.exp())

    log_odds_ratio = log_odds_chosen - log_odds_rejected  # (B,)
    or_loss: torch.Tensor = -F.logsigmoid(log_odds_ratio).mean()

    # -----------------------------------------------------------------------
    # 4. Combined loss
    # -----------------------------------------------------------------------
    total_loss = sft_loss + lambda_or * or_loss

    return total_loss, sft_loss.item(), or_loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------------
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained_checkpoint}"
        )

    print(f"Loading pretrained model from {args.pretrained_checkpoint} ...")
    model: nn.Module = LLM.from_pretrained(args.pretrained_checkpoint)
    model.config.use_fp8 = False   # H100 MIG: BF16 only; B200 may set fp8 via config
    model = model.to(device=device, dtype=torch.bfloat16)

    # Gradient checkpointing — reduces VRAM at cost of ~20% speed
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    if args.use_lora:
        n_lora = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        lora_params = get_lora_params(model)
        print(f"[INFO] LoRA: {n_lora:,} trainable params "
              f"(rank={args.lora_rank}, alpha={args.lora_alpha})")
    else:
        lora_params = None
        print("[INFO] Full fine-tuning (all parameters trainable)")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer_path = _resolve_tokenizer_path(args)
    print(f"Loading tokenizer from {tokenizer_path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    train_dataset = DPODataset(
        data_path=args.preference_data,
        tokenizer=tokenizer,
        max_seq_len=args.max_length,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"Preference dataset is empty: {args.preference_data}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dpo_collate_fn,
        prefetch_factor=2,
        persistent_workers=(args.num_workers > 0),
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if lora_params is not None:
        opt_params = lora_params
    else:
        opt_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        opt_params,
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

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_step, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Output directory & tokenizer copy
    # ------------------------------------------------------------------
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest_tok = args.checkpoint_dir / "tokenizer.json"
    if not dest_tok.exists():
        shutil.copy2(str(tokenizer_path), str(dest_tok))

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    log_fh = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "a", encoding="utf-8", buffering=1)

    def log(msg: str, level: str = "INFO") -> None:
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")

    # ------------------------------------------------------------------
    # Training banner
    # ------------------------------------------------------------------
    eff_batch = args.batch_size * args.grad_accum
    log("=" * 65)
    log("ORPO Training — EVAFRILL-Mo")
    log(f"  Pretrained ckpt : {args.pretrained_checkpoint}")
    log(f"  Preference data : {args.preference_data} ({len(train_dataset):,} samples)")
    log(f"  LoRA            : rank={args.lora_rank} alpha={args.lora_alpha} "
        f"enabled={args.use_lora}")
    log(f"  lambda_or={args.lambda_or}, lr={args.lr:.2e}, eff_batch={eff_batch}")
    log(f"  max_steps={args.max_steps}, warmup={args.warmup_steps}, "
        f"max_len={args.max_length}")
    log(f"  device={device}")
    log("=" * 65)

    # ------------------------------------------------------------------
    # Graceful shutdown handler
    # ------------------------------------------------------------------
    shutdown_requested = False

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log(f"Shutdown signal received (sig={signum}). Saving checkpoint ...", "WARN")

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT,  shutdown_handler)
    try:
        signal.signal(signal.SIGHUP, shutdown_handler)
    except AttributeError:
        pass  # Windows does not have SIGHUP

    # ------------------------------------------------------------------
    # Data iterator (infinite, cycling through epochs)
    # ------------------------------------------------------------------
    import time

    epoch = 0
    loader_iter = iter(train_loader)

    def next_batch() -> tuple[torch.Tensor, ...]:
        nonlocal loader_iter, epoch
        try:
            return next(loader_iter)
        except StopIteration:
            epoch += 1
            log(f"--- Epoch {epoch} begin ---")
            loader_iter = iter(train_loader)
            return next(loader_iter)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()

    # Running statistics (reset every log_interval steps)
    running_total_loss = 0.0
    running_sft_loss   = 0.0
    running_or_loss    = 0.0
    log_step_count     = 0
    t0 = time.perf_counter()

    # Keep track of the last loss value for the final checkpoint call
    avg_loss = float("nan")

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)

        accum_total = 0.0
        accum_sft   = 0.0
        accum_or    = 0.0

        # ---- Gradient accumulation ----------------------------------------
        for _micro in range(args.grad_accum):
            batch = next_batch()
            chosen_ids      = batch[0].to(device, dtype=torch.long, non_blocking=True)
            chosen_labels   = batch[1].to(device, dtype=torch.long, non_blocking=True)
            rejected_ids    = batch[2].to(device, dtype=torch.long, non_blocking=True)
            rejected_labels = batch[3].to(device, dtype=torch.long, non_blocking=True)

            loss, sft_l, or_l = compute_orpo_loss(
                model,
                chosen_ids, chosen_labels,
                rejected_ids, rejected_labels,
                lambda_or=args.lambda_or,
            )

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            accum_total += loss.item()
            accum_sft   += sft_l
            accum_or    += or_l

        # ---- Gradient clipping --------------------------------------------
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        ).item()

        optimizer.step()
        scheduler.step()

        # ---- Accumulate stats ---------------------------------------------
        avg_total = accum_total / args.grad_accum
        avg_sft   = accum_sft   / args.grad_accum
        avg_or    = accum_or    / args.grad_accum

        running_total_loss += avg_total
        running_sft_loss   += avg_sft
        running_or_loss    += avg_or
        log_step_count     += 1
        avg_loss            = avg_total   # for use in checkpoint call

        # ---- Graceful shutdown check --------------------------------------
        if shutdown_requested:
            log(f"Graceful shutdown at step {step + 1}", "WARN")
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler,
                step + 1, avg_loss, str(args.checkpoint_dir)
            )
            if args.use_lora:
                save_lora(model, args.checkpoint_dir / f"lora-{step+1:07d}")
            log(f"Checkpoint saved -> {ckpt_path}")
            break

        # ---- Logging ------------------------------------------------------
        if (step + 1) % args.log_interval == 0:
            t1      = time.perf_counter()
            elapsed = t1 - t0

            mean_total = running_total_loss / log_step_count
            mean_sft   = running_sft_loss   / log_step_count
            mean_or    = running_or_loss    / log_step_count
            lr_now     = scheduler.get_last_lr()[0]
            mem_gb     = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            sps        = log_step_count / max(elapsed, 1e-6)  # steps per second

            log(
                f"step {step+1:>6d}/{args.max_steps} | "
                f"loss {mean_total:.4f} "
                f"(sft {mean_sft:.4f} or {mean_or:.4f}) | "
                f"lr {lr_now:.2e} | "
                f"gnorm {grad_norm:.3f} | "
                f"mem {mem_gb:.1f}GB | "
                f"{sps:.2f}step/s"
            )

            running_total_loss = 0.0
            running_sft_loss   = 0.0
            running_or_loss    = 0.0
            log_step_count     = 0
            t0 = t1

        # ---- Periodic checkpoint ------------------------------------------
        if (step + 1) % args.save_interval == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler,
                step + 1, avg_loss, str(args.checkpoint_dir)
            )
            if args.use_lora:
                save_lora(model, args.checkpoint_dir / f"lora-{step+1:07d}")
            log(f"Checkpoint saved -> {ckpt_path}")

    # -----------------------------------------------------------------------
    # Final checkpoint
    # -----------------------------------------------------------------------
    if not shutdown_requested:
        final_path = save_checkpoint(
            model, optimizer, scheduler,
            args.max_steps, avg_loss, str(args.checkpoint_dir)
        )
        if args.use_lora:
            save_lora(model, args.checkpoint_dir / "lora-final")
        log(f"Final checkpoint -> {final_path}")

    # -----------------------------------------------------------------------
    # LoRA merge + save merged model
    # -----------------------------------------------------------------------
    if args.use_lora:
        log("Merging LoRA weights into base model ...")
        merge_lora(model)
        merged_dir = args.checkpoint_dir / "checkpoint-merged"
        model.save_pretrained(merged_dir)
        # Also copy tokenizer into merged dir for easy inference
        shutil.copy2(str(dest_tok), str(merged_dir / "tokenizer.json"))
        log(f"Merged model saved -> {merged_dir}")

    log("ORPO training complete.")

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()

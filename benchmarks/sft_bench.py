"""
sft_bench.py — SFT training throughput benchmark for EVAFRILL-Mo 2.94B hybrid model.

Finds the stable upper limit for SFT training on H100 MIG 3g.40gb (40 GB VRAM, ~42 SMs).

Usage:
    python benchmarks/sft_bench.py
    python benchmarks/sft_bench.py --checkpoint checkpoints/3b_final/checkpoint-0319772
    python benchmarks/sft_bench.py --steps 10 --warmup 5
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root → sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# TF32 flags (recommended for Ampere / Hopper)
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model import LLM  # noqa: E402  (after sys.path patch)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    batch_size: int
    seq_len: int
    tokens_per_step: int
    tokens_per_sec: float
    ms_per_step: float
    peak_vram_mb: float
    vram_util_pct: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


def get_total_vram_mb() -> float:
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 2)


def get_peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated(0) / (1024 ** 2)


def reset_peak_vram() -> None:
    torch.cuda.reset_peak_memory_stats(0)


def cleanup() -> None:
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_bench(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    total_vram_mb: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> Optional[BenchResult]:
    """Run forward+backward+optimizer.step() and measure throughput.

    Returns None on OOM or any other failure.
    """
    # Build optimizer fresh each run to avoid state accumulation across configs.
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,  # fused AdamW kernel (bf16-compatible on CUDA)
        )
    except Exception:
        # Fused AdamW may not be available; fall back to standard
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    reset_peak_vram()

    def step() -> None:
        input_ids = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len),
            device=device, dtype=torch.long,
        )
        targets = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len),
            device=device, dtype=torch.long,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _logits, loss = model(input_ids, targets=targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Warmup (not timed)
    try:
        for _ in range(n_warmup):
            step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM] bs={batch_size} seq={seq_len} — OOM during warmup")
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR] bs={batch_size} seq={seq_len} — {exc}")
        cleanup()
        return None

    # Timed steps
    reset_peak_vram()
    try:
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM] bs={batch_size} seq={seq_len} — OOM during timed steps")
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR] bs={batch_size} seq={seq_len} — {exc}")
        cleanup()
        return None

    peak_vram_mb = get_peak_vram_mb()
    ms_per_step = elapsed / n_steps * 1000.0
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / (elapsed / n_steps)
    vram_util_pct = peak_vram_mb / total_vram_mb * 100.0

    # Free optimizer state before next run
    del optimizer
    cleanup()

    return BenchResult(
        batch_size=batch_size,
        seq_len=seq_len,
        tokens_per_step=tokens_per_step,
        tokens_per_sec=tokens_per_sec,
        ms_per_step=ms_per_step,
        peak_vram_mb=peak_vram_mb,
        vram_util_pct=vram_util_pct,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

HEADER = (
    f"{'bs':>4}  {'seq':>5}  {'tok/step':>9}  "
    f"{'tok/s':>10}  {'ms/step':>8}  {'VRAM MB':>8}  {'VRAM%':>6}"
)
SEP = "-" * len(HEADER)


def print_row(r: BenchResult) -> None:
    print(
        f"{r.batch_size:>4}  {r.seq_len:>5}  {r.tokens_per_step:>9,}  "
        f"{r.tokens_per_sec:>10,.0f}  {r.ms_per_step:>8.1f}  "
        f"{r.peak_vram_mb:>8.0f}  {r.vram_util_pct:>5.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT throughput benchmark for EVAFRILL-Mo 2.94B hybrid model"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/3b_final/checkpoint-0319772",
        help="Path to checkpoint directory (relative to project root or absolute)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of timed steps per configuration (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup steps before timing (default: 3)",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARNING] CUDA not available — results will not reflect GPU performance")

    total_vram_mb = get_total_vram_mb()
    print("=" * 70)
    print("  SFT Throughput Benchmark — EVAFRILL-Mo 2.94B Hybrid Mamba-2 + Transformer")
    print("=" * 70)
    print(f"  Device       : {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM   : {total_vram_mb:.0f} MB")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Warmup steps : {args.warmup}")
    print(f"  Timed steps  : {args.steps}")
    print()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading model from checkpoint ...")

    # Load config with FP8 disabled (no TransformerEngine on this machine)
    from model.config import LMConfig
    config = LMConfig.from_yaml(ckpt_path / "config.yaml")
    config.use_fp8 = False
    model = LLM(config)

    # Load state dict with TE→non-TE key conversion
    state_dict = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
    converted_sd = {}
    for k, v in state_dict.items():
        # Skip TransformerEngine _extra_state entries
        if "_extra_state" in k:
            continue
        # Convert TE LayerNormMLP keys to standard SwiGLU keys
        if ".ffn.layer_norm_weight" in k:
            new_k = k.replace(".ffn.layer_norm_weight", ".ffn_norm.weight")
            converted_sd[new_k] = v
        elif ".ffn.fc1_weight" in k:
            # TE fc1 = concat(gate_proj, up_proj) along dim 0
            d_ffn = v.shape[0] // 2
            converted_sd[k.replace(".ffn.fc1_weight", ".ffn.gate_proj.weight")] = v[:d_ffn]
            converted_sd[k.replace(".ffn.fc1_weight", ".ffn.up_proj.weight")] = v[d_ffn:]
        elif ".ffn.fc2_weight" in k:
            converted_sd[k.replace(".ffn.fc2_weight", ".ffn.down_proj.weight")] = v
        # Convert TE Linear keys for attention (strip _extra_state already done)
        elif ".attn.qkv_proj.weight" in k:
            converted_sd[k] = v
        elif ".attn.out_proj.weight" in k:
            converted_sd[k] = v
        else:
            converted_sd[k] = v

    model.load_state_dict(converted_sd)

    # Convert to bf16 and move to device
    model = model.to(dtype=torch.bfloat16, device=device)

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None

    # Train mode required for backward pass
    model.train()

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    cfg = model.config

    print()
    print("  Model parameter count:")
    print(f"    Total       : {fmt_params(total_params)}  ({total_params:,})")
    print(f"    Trainable   : {fmt_params(trainable_params)}  ({trainable_params:,})")
    print()
    print("  Model config summary:")
    print(f"    d_model     : {cfg.d_model}")
    print(f"    n_layers    : {cfg.n_layers}")
    print(f"    n_heads     : {cfg.n_heads}")
    print(f"    n_kv_heads  : {cfg.n_kv_heads}")
    print(f"    d_ffn       : {cfg.d_ffn}")
    print(f"    max_seq_len : {cfg.max_seq_len}")
    print(f"    vocab_size  : {cfg.vocab_size}")
    print(f"    use_hybrid  : {cfg.use_hybrid}")
    if cfg.use_hybrid:
        pattern_tokens = cfg.hybrid_pattern.strip().split()
        n_mamba = pattern_tokens.count("M")
        n_attn  = pattern_tokens.count("A")
        print(f"    hybrid_pat  : {n_mamba}× Mamba-2 + {n_attn}× Attention ({len(pattern_tokens)} layers total)")
        print(f"    mamba_d_state: {cfg.mamba_d_state}")
        print(f"    mamba_expand: {cfg.mamba_expand}")
        print(f"    chunk_size  : {cfg.mamba_chunk_size}")
    print(f"    use_fp8     : {cfg.use_fp8}  (disabled for this bench — no TransformerEngine)")
    print(f"    use_flash_attn: {cfg.use_flash_attn}")
    print()

    # -----------------------------------------------------------------------
    # Benchmark grid
    # -----------------------------------------------------------------------
    batch_sizes = [1, 2, 3, 4, 6, 8]
    seq_lens    = [512, 1024, 2048, 4096]

    total_configs = len(batch_sizes) * len(seq_lens)
    print(f"Running {total_configs} configurations ({len(batch_sizes)} batch sizes × {len(seq_lens)} seq lengths) ...")
    print()

    results: list[BenchResult] = []

    for seq_len in seq_lens:
        if seq_len > cfg.max_seq_len:
            print(f"  [SKIP] seq_len={seq_len} > model max_seq_len={cfg.max_seq_len}")
            continue
        for batch_size in batch_sizes:
            print(f"  Testing bs={batch_size}, seq={seq_len} ...", end=" ", flush=True)
            cleanup()
            result = run_bench(
                model=model,
                batch_size=batch_size,
                seq_len=seq_len,
                total_vram_mb=total_vram_mb,
                n_warmup=args.warmup,
                n_steps=args.steps,
                device=device,
            )
            if result is not None:
                results.append(result)
                print(
                    f"OK  {result.tokens_per_sec:,.0f} tok/s  "
                    f"{result.peak_vram_mb:.0f} MB  ({result.vram_util_pct:.1f}%)"
                )
            cleanup()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  RESULTS — sorted by tokens/sec (descending)")
    print("=" * 70)

    if not results:
        print("  No successful configurations found (all OOM).")
        print("  Try reducing batch_size or seq_len.")
        return

    results_sorted = sorted(results, key=lambda r: r.tokens_per_sec, reverse=True)

    print(HEADER)
    print(SEP)
    for r in results_sorted:
        print_row(r)
    print()

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    STABLE_THRESHOLD = 0.90  # < 90% VRAM = stable

    stable = [r for r in results_sorted if r.vram_util_pct < STABLE_THRESHOLD * 100]
    optimal: Optional[BenchResult] = stable[0] if stable else None
    max_possible: BenchResult = results_sorted[0]

    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    if optimal is not None:
        print(
            f"  Optimal (< {STABLE_THRESHOLD*100:.0f}% VRAM)  : "
            f"batch_size={optimal.batch_size}, seq_len={optimal.seq_len}  →  "
            f"{optimal.tokens_per_sec:,.0f} tok/s, "
            f"{optimal.peak_vram_mb:.0f} MB ({optimal.vram_util_pct:.1f}%)"
        )
    else:
        print(f"  Optimal (< {STABLE_THRESHOLD*100:.0f}% VRAM)  : none found — all configs exceed the stable margin")

    print(
        f"  Max possible (no margin): "
        f"batch_size={max_possible.batch_size}, seq_len={max_possible.seq_len}  →  "
        f"{max_possible.tokens_per_sec:,.0f} tok/s, "
        f"{max_possible.peak_vram_mb:.0f} MB ({max_possible.vram_util_pct:.1f}%)"
    )
    print()

    # -----------------------------------------------------------------------
    # Recommended SFT launch command
    # -----------------------------------------------------------------------
    rec = optimal if optimal is not None else max_possible
    print("=" * 70)
    print("  RECOMMENDED SFT LAUNCH COMMAND")
    print("=" * 70)
    print()
    print(f"  # Optimal batch_size={rec.batch_size}, seq_len={rec.seq_len}")
    print(f"  # Estimated throughput: {rec.tokens_per_sec:,.0f} tok/s")
    print(f"  # Peak VRAM: {rec.peak_vram_mb:.0f} MB / {total_vram_mb:.0f} MB ({rec.vram_util_pct:.1f}%)")
    print()
    print(
        f"  python train/sft.py \\\n"
        f"    --checkpoint {ckpt_path} \\\n"
        f"    --batch_size {rec.batch_size} \\\n"
        f"    --seq_len {rec.seq_len} \\\n"
        f"    --grad_accum 8 \\\n"
        f"    --lr 2e-5 \\\n"
        f"    --dtype bf16"
    )
    print()
    print(
        "  NOTE: grad_accum is not benchmarked here — it does NOT affect VRAM "
        "usage (only effective batch size = batch_size × grad_accum). "
        "Adjust --grad_accum to reach your desired effective batch size "
        f"(e.g., grad_accum=8 → effective batch={rec.batch_size * 8})."
    )
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

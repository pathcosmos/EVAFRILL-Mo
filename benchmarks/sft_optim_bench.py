"""
sft_optim_bench.py — Multi-phase SFT optimization benchmark for EVAFRILL-Mo 2.94B hybrid model.

Finds the optimal batch_size, seq_len, grad_accum, and torch.compile settings for SFT
training on H100 MIG 3g.40gb (40 GB VRAM, ~42 SMs).

Baseline: batch_size=1, grad_accum=28, eff_batch=28, ~24 GB VRAM, ~5,900 tok/s

Phases:
  Phase 1 — batch_size x seq_len grid (find VRAM limits and throughput)
  Phase 2 — grad_accum sweep for best Phase 1 config
  Phase 3 — torch.compile test on best Phase 2 config

Usage:
    python benchmarks/sft_optim_bench.py
    python benchmarks/sft_optim_bench.py --checkpoint checkpoints/3b_final/checkpoint-0319772
    python benchmarks/sft_optim_bench.py --skip-compile
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root -> sys.path
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
# Constants
# ---------------------------------------------------------------------------

# Current training progress
STEPS_DONE     = 410
TOTAL_STEPS    = 135000
STEPS_REMAINING = TOTAL_STEPS - STEPS_DONE

# Baseline: bs=1, ga=28, seq=1024, ~7.5 s/step
BASELINE_MS_PER_STEP = 7500.0

# VRAM safety threshold for "stable" recommendation
VRAM_STABLE_PCT = 0.90


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    """Phase 1: single forward+backward+optimizer.step() measurement."""
    batch_size: int
    seq_len: int
    tokens_per_step: int
    tokens_per_sec: float
    ms_per_step: float
    peak_vram_mb: float
    vram_util_pct: float


@dataclass
class GradAccumResult:
    """Phase 2: full optimizer step (all micro-steps of accumulation)."""
    batch_size: int
    seq_len: int
    grad_accum: int
    eff_batch: int
    tokens_per_opt_step: int
    eff_tokens_per_sec: float
    ms_per_opt_step: float
    peak_vram_mb: float
    vram_util_pct: float


@dataclass
class CompileResult:
    """Phase 3: torch.compile test on best config."""
    batch_size: int
    seq_len: int
    grad_accum: int
    eff_batch: int
    tokens_per_opt_step: int
    eff_tokens_per_sec: float
    ms_per_opt_step: float
    peak_vram_mb: float
    vram_util_pct: float
    compile_mode: str


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


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build a fresh AdamW optimizer, using the fused kernel when available."""
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except Exception:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )


def fmt_hours(ms: float) -> str:
    h = ms / 3_600_000.0
    if h >= 24:
        return f"{h:.1f} h  ({h/24:.1f} days)"
    return f"{h:.1f} h"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> LLM:
    """Load EVAFRILL-Mo from checkpoint with TE->non-TE state dict conversion."""
    from model.config import LMConfig

    config = LMConfig.from_yaml(ckpt_path / "config.yaml")
    config.use_fp8 = False
    model = LLM(config)

    state_dict = torch.load(
        ckpt_path / "model.pt", map_location="cpu", weights_only=True
    )

    converted_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        # Skip TransformerEngine _extra_state entries
        if "_extra_state" in k:
            continue
        # Convert TE LayerNormMLP keys to standard SwiGLU keys
        if ".ffn.layer_norm_weight" in k:
            converted_sd[k.replace(".ffn.layer_norm_weight", ".ffn_norm.weight")] = v
        elif ".ffn.fc1_weight" in k:
            # TE fc1 = concat(gate_proj, up_proj) along dim 0
            d_ffn = v.shape[0] // 2
            converted_sd[k.replace(".ffn.fc1_weight", ".ffn.gate_proj.weight")] = v[:d_ffn]
            converted_sd[k.replace(".ffn.fc1_weight", ".ffn.up_proj.weight")] = v[d_ffn:]
        elif ".ffn.fc2_weight" in k:
            converted_sd[k.replace(".ffn.fc2_weight", ".ffn.down_proj.weight")] = v
        else:
            converted_sd[k] = v

    model.load_state_dict(converted_sd)
    model = model.to(dtype=torch.bfloat16, device=device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()
    return model


# ---------------------------------------------------------------------------
# Phase 1: batch_size x seq_len grid
# ---------------------------------------------------------------------------

def run_grid_bench(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    total_vram_mb: float,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> Optional[GridResult]:
    """
    Run forward+backward+optimizer.step() and measure per-step throughput.
    Returns None on OOM or failure.
    """
    optimizer = build_optimizer(model)
    reset_peak_vram()

    vocab_size = model.config.vocab_size

    def single_step() -> None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets   = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _logits, loss = model(input_ids, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Warmup
    try:
        for _ in range(n_warmup):
            single_step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during warmup]")
        del optimizer
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during warmup: {exc}]")
        del optimizer
        cleanup()
        return None

    # Timed steps
    reset_peak_vram()
    try:
        t0 = time.perf_counter()
        for _ in range(n_steps):
            single_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during timed steps]")
        del optimizer
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during timed steps: {exc}]")
        del optimizer
        cleanup()
        return None

    peak_vram_mb   = get_peak_vram_mb()
    ms_per_step    = elapsed / n_steps * 1000.0
    tokens_per_step = batch_size * seq_len
    tokens_per_sec  = tokens_per_step / (elapsed / n_steps)
    vram_util_pct  = peak_vram_mb / total_vram_mb * 100.0

    del optimizer
    cleanup()

    return GridResult(
        batch_size=batch_size,
        seq_len=seq_len,
        tokens_per_step=tokens_per_step,
        tokens_per_sec=tokens_per_sec,
        ms_per_step=ms_per_step,
        peak_vram_mb=peak_vram_mb,
        vram_util_pct=vram_util_pct,
    )


def print_phase1_table(results: list[GridResult]) -> None:
    header = (
        f"{'bs':>4}  {'seq':>5}  {'tok/step':>9}  "
        f"{'tok/s':>10}  {'ms/step':>8}  {'VRAM MB':>8}  {'VRAM%':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.batch_size:>4}  {r.seq_len:>5}  {r.tokens_per_step:>9,}  "
            f"{r.tokens_per_sec:>10,.0f}  {r.ms_per_step:>8.1f}  "
            f"{r.peak_vram_mb:>8.0f}  {r.vram_util_pct:>5.1f}%"
        )


# ---------------------------------------------------------------------------
# Phase 2: grad_accum sweep
# ---------------------------------------------------------------------------

def run_grad_accum_bench(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    total_vram_mb: float,
    n_warmup_cycles: int,
    n_timed_cycles: int,
    device: torch.device,
) -> Optional[GradAccumResult]:
    """
    Measure a FULL optimizer step: all grad_accum micro-steps + optimizer.step().

    n_warmup_cycles  — number of full accumulation cycles (warmup, not timed)
    n_timed_cycles   — number of full accumulation cycles to time
    """
    optimizer = build_optimizer(model)
    reset_peak_vram()

    vocab_size = model.config.vocab_size

    def full_opt_step() -> None:
        """One full optimizer step = grad_accum micro-steps + optimizer.step()."""
        optimizer.zero_grad(set_to_none=True)
        for micro_idx in range(grad_accum):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets   = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _logits, loss = model(input_ids, targets=targets)
            # Divide loss by grad_accum before backward for correct gradient scaling
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Warmup cycles
    try:
        for _ in range(n_warmup_cycles):
            full_opt_step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during warmup]")
        del optimizer
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during warmup: {exc}]")
        del optimizer
        cleanup()
        return None

    # Timed cycles
    reset_peak_vram()
    try:
        t0 = time.perf_counter()
        for _ in range(n_timed_cycles):
            full_opt_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during timed steps]")
        del optimizer
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during timed steps: {exc}]")
        del optimizer
        cleanup()
        return None

    peak_vram_mb         = get_peak_vram_mb()
    ms_per_opt_step      = elapsed / n_timed_cycles * 1000.0
    tokens_per_opt_step  = batch_size * seq_len * grad_accum
    eff_tokens_per_sec   = tokens_per_opt_step / (elapsed / n_timed_cycles)
    vram_util_pct        = peak_vram_mb / total_vram_mb * 100.0

    del optimizer
    cleanup()

    return GradAccumResult(
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum=grad_accum,
        eff_batch=batch_size * grad_accum,
        tokens_per_opt_step=tokens_per_opt_step,
        eff_tokens_per_sec=eff_tokens_per_sec,
        ms_per_opt_step=ms_per_opt_step,
        peak_vram_mb=peak_vram_mb,
        vram_util_pct=vram_util_pct,
    )


def print_phase2_table(results: list[GradAccumResult]) -> None:
    header = (
        f"{'bs':>4}  {'seq':>5}  {'ga':>4}  {'eff_bs':>7}  "
        f"{'tok/opt_step':>13}  {'eff_tok/s':>10}  {'ms/opt_step':>12}  "
        f"{'VRAM MB':>8}  {'VRAM%':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.batch_size:>4}  {r.seq_len:>5}  {r.grad_accum:>4}  {r.eff_batch:>7}  "
            f"{r.tokens_per_opt_step:>13,}  {r.eff_tokens_per_sec:>10,.0f}  "
            f"{r.ms_per_opt_step:>12.1f}  "
            f"{r.peak_vram_mb:>8.0f}  {r.vram_util_pct:>5.1f}%"
        )


# ---------------------------------------------------------------------------
# Phase 3: torch.compile test
# ---------------------------------------------------------------------------

def run_compile_bench(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    total_vram_mb: float,
    n_warmup_cycles: int,
    n_timed_cycles: int,
    compile_mode: str,
    device: torch.device,
) -> Optional[CompileResult]:
    """
    Apply torch.compile to the model and measure full optimizer step throughput.
    Uses more warmup to allow the compiler to finish tracing.
    """
    print(f"    Compiling model with mode='{compile_mode}' (this may take ~1-2 min) ...", flush=True)
    try:
        compiled_model = torch.compile(model, mode=compile_mode)
    except Exception as exc:
        print(f"    [ERR] torch.compile failed: {exc}")
        return None

    optimizer = build_optimizer(compiled_model)
    reset_peak_vram()

    vocab_size = model.config.vocab_size

    def full_opt_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets   = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _logits, loss = compiled_model(input_ids, targets=targets)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), 1.0)
        optimizer.step()

    # Warmup (compile traces on first calls — needs more warmup)
    try:
        for _ in range(n_warmup_cycles):
            full_opt_step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during compile warmup]")
        del optimizer, compiled_model
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during compile warmup: {exc}]")
        del optimizer, compiled_model
        cleanup()
        return None

    # Timed cycles
    reset_peak_vram()
    try:
        t0 = time.perf_counter()
        for _ in range(n_timed_cycles):
            full_opt_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        print(f"    [OOM during compile timed steps]")
        del optimizer, compiled_model
        cleanup()
        return None
    except Exception as exc:
        print(f"    [ERR during compile timed steps: {exc}]")
        del optimizer, compiled_model
        cleanup()
        return None

    peak_vram_mb        = get_peak_vram_mb()
    ms_per_opt_step     = elapsed / n_timed_cycles * 1000.0
    tokens_per_opt_step = batch_size * seq_len * grad_accum
    eff_tokens_per_sec  = tokens_per_opt_step / (elapsed / n_timed_cycles)
    vram_util_pct       = peak_vram_mb / total_vram_mb * 100.0

    del optimizer, compiled_model
    cleanup()

    return CompileResult(
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum=grad_accum,
        eff_batch=batch_size * grad_accum,
        tokens_per_opt_step=tokens_per_opt_step,
        eff_tokens_per_sec=eff_tokens_per_sec,
        ms_per_opt_step=ms_per_opt_step,
        peak_vram_mb=peak_vram_mb,
        vram_util_pct=vram_util_pct,
        compile_mode=compile_mode,
    )


# ---------------------------------------------------------------------------
# Final recommendation helpers
# ---------------------------------------------------------------------------

def print_training_time_estimate(
    ms_per_opt_step: float,
    label: str,
    indent: str = "  ",
) -> None:
    est_ms    = STEPS_REMAINING * ms_per_opt_step
    baseline_ms = STEPS_REMAINING * BASELINE_MS_PER_STEP
    speedup   = baseline_ms / est_ms if est_ms > 0 else float("nan")
    print(f"{indent}{label}:")
    print(f"{indent}  ms/opt_step       : {ms_per_opt_step:,.1f} ms")
    print(f"{indent}  steps remaining   : {STEPS_REMAINING:,}  (= {TOTAL_STEPS} - {STEPS_DONE})")
    print(f"{indent}  estimated time    : {fmt_hours(est_ms)}")
    print(f"{indent}  baseline time     : {fmt_hours(baseline_ms)}  ({BASELINE_MS_PER_STEP:,.0f} ms/step)")
    print(f"{indent}  speedup vs baseline: {speedup:.2f}x")


def print_yaml_snippet(
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    lr: float = 7e-6,
) -> None:
    eff_batch = batch_size * grad_accum
    print()
    print("  Recommended YAML config snippet:")
    print()
    print("  train:")
    print(f"    batch_size: {batch_size}")
    print(f"    grad_accum_steps: {grad_accum}")
    print(f"    # eff_batch = {eff_batch}  (bs={batch_size} x ga={grad_accum})")
    print(f"    lr: {lr:.2e}")
    print(f"    max_seq_len: {seq_len}  # for data collation")
    print()
    print("  Recommended launch command:")
    print()
    print(f"  python train/sft.py \\")
    print(f"    --config configs/h100_mig/korean_3b_sft_1gpu.yaml \\")
    print(f"    --batch_size {batch_size} \\")
    print(f"    --seq_len {seq_len} \\")
    print(f"    --grad_accum {grad_accum} \\")
    print(f"    --lr {lr:.2e} \\")
    print(f"    --dtype bf16")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-phase SFT optimization benchmark for EVAFRILL-Mo 2.94B hybrid model"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/3b_final/checkpoint-0319772",
        help="Path to checkpoint directory (relative to project root or absolute)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip Phase 3 (torch.compile test)",
    )
    parser.add_argument(
        "--phase1-warmup",
        type=int,
        default=3,
        help="Warmup steps for Phase 1 grid (default: 3)",
    )
    parser.add_argument(
        "--phase1-steps",
        type=int,
        default=5,
        help="Timed steps for Phase 1 grid (default: 5)",
    )
    parser.add_argument(
        "--phase2-warmup-cycles",
        type=int,
        default=3,
        help="Warmup full-accumulation cycles for Phase 2 (default: 3)",
    )
    parser.add_argument(
        "--phase2-timed-cycles",
        type=int,
        default=3,
        help="Timed full-accumulation cycles for Phase 2 (default: 3)",
    )
    parser.add_argument(
        "--phase3-warmup-cycles",
        type=int,
        default=5,
        help="Warmup cycles for Phase 3 torch.compile (default: 5)",
    )
    parser.add_argument(
        "--phase3-timed-cycles",
        type=int,
        default=5,
        help="Timed cycles for Phase 3 torch.compile (default: 5)",
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

    print("=" * 78)
    print("  SFT Optimization Benchmark — EVAFRILL-Mo 2.94B Hybrid Mamba-2 + Transformer")
    print("=" * 78)
    print(f"  Device         : {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM     : {total_vram_mb:.0f} MB  ({total_vram_mb/1024:.2f} GB)")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Baseline       : bs=1, ga=28, eff_batch=28, ~{BASELINE_MS_PER_STEP:,.0f} ms/step")
    print(f"  Steps remaining: {STEPS_REMAINING:,}  ({STEPS_DONE} of {TOTAL_STEPS} done)")
    print()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading model from checkpoint ...")
    model = load_model(ckpt_path, device)

    total_params    = count_params(model)
    trainable_params = count_trainable_params(model)
    cfg             = model.config

    print()
    print("  Model parameters:")
    print(f"    Total       : {fmt_params(total_params)}  ({total_params:,})")
    print(f"    Trainable   : {fmt_params(trainable_params)}  ({trainable_params:,})")
    print()
    print("  Model config:")
    print(f"    d_model     : {cfg.d_model}")
    print(f"    n_layers    : {cfg.n_layers}")
    print(f"    n_heads     : {cfg.n_heads}  (kv_heads={cfg.n_kv_heads})")
    print(f"    d_ffn       : {cfg.d_ffn}")
    print(f"    max_seq_len : {cfg.max_seq_len}")
    print(f"    vocab_size  : {cfg.vocab_size}")
    if cfg.use_hybrid:
        pattern_tokens = cfg.hybrid_pattern.strip().split()
        n_mamba = pattern_tokens.count("M")
        n_attn  = pattern_tokens.count("A")
        print(
            f"    hybrid      : {n_mamba}x Mamba-2 + {n_attn}x Attention "
            f"({len(pattern_tokens)} layers)"
        )
    print(f"    use_fp8     : {cfg.use_fp8}  (disabled — no TE on this machine)")
    print(f"    flash_attn  : {cfg.use_flash_attn}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: batch_size x seq_len grid
    # -----------------------------------------------------------------------
    PHASE1_BATCH_SIZES = [1, 2, 3, 4]
    PHASE1_SEQ_LENS    = [512, 1024, 2048]

    total_p1 = len(PHASE1_BATCH_SIZES) * len(PHASE1_SEQ_LENS)

    print("=" * 78)
    print(f"  PHASE 1 — batch_size x seq_len grid  ({total_p1} configs)")
    print(f"           warmup={args.phase1_warmup}  timed={args.phase1_steps}")
    print(f"           Find VRAM limits; identify highest-throughput config under {VRAM_STABLE_PCT*100:.0f}% VRAM")
    print("=" * 78)
    print()

    phase1_results: list[GridResult] = []

    for seq_len in PHASE1_SEQ_LENS:
        if seq_len > cfg.max_seq_len:
            print(f"  [SKIP] seq_len={seq_len} exceeds model max_seq_len={cfg.max_seq_len}")
            continue
        for batch_size in PHASE1_BATCH_SIZES:
            print(
                f"  [P1] bs={batch_size}, seq={seq_len} ...",
                end="  ",
                flush=True,
            )
            cleanup()
            result = run_grid_bench(
                model=model,
                batch_size=batch_size,
                seq_len=seq_len,
                total_vram_mb=total_vram_mb,
                n_warmup=args.phase1_warmup,
                n_steps=args.phase1_steps,
                device=device,
            )
            if result is not None:
                phase1_results.append(result)
                stable_marker = "" if result.vram_util_pct < VRAM_STABLE_PCT * 100 else "  [>90% VRAM]"
                print(
                    f"OK  {result.tokens_per_sec:>10,.0f} tok/s  "
                    f"{result.peak_vram_mb:.0f} MB  ({result.vram_util_pct:.1f}%){stable_marker}"
                )
            cleanup()

    print()
    print("  Phase 1 Summary — sorted by tok/s descending:")
    print()

    if not phase1_results:
        print("  No successful configurations found in Phase 1 (all OOM).")
        print("  Cannot proceed to Phase 2.")
        return

    phase1_sorted = sorted(phase1_results, key=lambda r: r.tokens_per_sec, reverse=True)
    print_phase1_table(phase1_sorted)
    print()

    # Select best Phase 1 config under VRAM threshold
    stable_p1 = [r for r in phase1_sorted if r.vram_util_pct < VRAM_STABLE_PCT * 100]
    if stable_p1:
        best_p1 = stable_p1[0]
        print(
            f"  Best stable Phase 1 config (<{VRAM_STABLE_PCT*100:.0f}% VRAM): "
            f"bs={best_p1.batch_size}, seq={best_p1.seq_len}  "
            f"({best_p1.tokens_per_sec:,.0f} tok/s, {best_p1.peak_vram_mb:.0f} MB)"
        )
    else:
        best_p1 = phase1_sorted[0]
        print(
            f"  No stable config found (<{VRAM_STABLE_PCT*100:.0f}% VRAM). "
            f"Using best available: bs={best_p1.batch_size}, seq={best_p1.seq_len}"
        )
    print()

    # -----------------------------------------------------------------------
    # Phase 2: grad_accum sweep
    # -----------------------------------------------------------------------
    PHASE2_GRAD_ACCUMS = [7, 14, 28]

    print("=" * 78)
    print(f"  PHASE 2 — grad_accum sweep  ({len(PHASE2_GRAD_ACCUMS)} configs)")
    print(f"           base config: bs={best_p1.batch_size}, seq={best_p1.seq_len}")
    print(f"           grad_accum values: {PHASE2_GRAD_ACCUMS}")
    print(f"           warmup_cycles={args.phase2_warmup_cycles}  timed_cycles={args.phase2_timed_cycles}")
    print(f"           Measuring FULL optimizer step (all micro-steps + optimizer.step())")
    print("=" * 78)
    print()

    phase2_results: list[GradAccumResult] = []

    for ga in PHASE2_GRAD_ACCUMS:
        eff_batch = best_p1.batch_size * ga
        print(
            f"  [P2] bs={best_p1.batch_size}, seq={best_p1.seq_len}, "
            f"ga={ga}, eff_batch={eff_batch} ...",
            end="  ",
            flush=True,
        )
        cleanup()
        result = run_grad_accum_bench(
            model=model,
            batch_size=best_p1.batch_size,
            seq_len=best_p1.seq_len,
            grad_accum=ga,
            total_vram_mb=total_vram_mb,
            n_warmup_cycles=args.phase2_warmup_cycles,
            n_timed_cycles=args.phase2_timed_cycles,
            device=device,
        )
        if result is not None:
            phase2_results.append(result)
            print(
                f"OK  {result.eff_tokens_per_sec:>10,.0f} eff_tok/s  "
                f"{result.ms_per_opt_step:,.1f} ms/opt_step  "
                f"{result.peak_vram_mb:.0f} MB  ({result.vram_util_pct:.1f}%)"
            )
        cleanup()

    print()
    print("  Phase 2 Summary — sorted by eff_tok/s descending:")
    print()

    if not phase2_results:
        print("  No successful configurations found in Phase 2.")
        # Fall back to Phase 1 best for recommendation
        best_p2: Optional[GradAccumResult] = None
    else:
        phase2_sorted = sorted(phase2_results, key=lambda r: r.eff_tokens_per_sec, reverse=True)
        print_phase2_table(phase2_sorted)
        print()

        best_p2 = phase2_sorted[0]
        print(
            f"  Best Phase 2 config: "
            f"bs={best_p2.batch_size}, seq={best_p2.seq_len}, "
            f"ga={best_p2.grad_accum}, eff_batch={best_p2.eff_batch}  "
            f"({best_p2.eff_tokens_per_sec:,.0f} eff_tok/s, "
            f"{best_p2.ms_per_opt_step:,.1f} ms/opt_step)"
        )
    print()

    # -----------------------------------------------------------------------
    # Phase 3: torch.compile
    # -----------------------------------------------------------------------
    compile_result: Optional[CompileResult] = None

    if not args.skip_compile:
        # Use best Phase 2 config, or fall back to Phase 1
        if best_p2 is not None:
            p3_bs  = best_p2.batch_size
            p3_seq = best_p2.seq_len
            p3_ga  = best_p2.grad_accum
        else:
            p3_bs  = best_p1.batch_size
            p3_seq = best_p1.seq_len
            p3_ga  = 28  # default

        print("=" * 78)
        print(f"  PHASE 3 — torch.compile test")
        print(f"           config: bs={p3_bs}, seq={p3_seq}, ga={p3_ga}")
        print(f"           compile mode: reduce-overhead")
        print(f"           warmup_cycles={args.phase3_warmup_cycles}  (more warmup needed for tracing)")
        print(f"           timed_cycles={args.phase3_timed_cycles}")
        print("=" * 78)
        print()

        print(f"  [P3] bs={p3_bs}, seq={p3_seq}, ga={p3_ga}, compile=reduce-overhead ...", flush=True)
        cleanup()

        compile_result = run_compile_bench(
            model=model,
            batch_size=p3_bs,
            seq_len=p3_seq,
            grad_accum=p3_ga,
            total_vram_mb=total_vram_mb,
            n_warmup_cycles=args.phase3_warmup_cycles,
            n_timed_cycles=args.phase3_timed_cycles,
            compile_mode="reduce-overhead",
            device=device,
        )
        cleanup()

        if compile_result is not None:
            print(
                f"  OK  {compile_result.eff_tokens_per_sec:,.0f} eff_tok/s  "
                f"{compile_result.ms_per_opt_step:,.1f} ms/opt_step  "
                f"{compile_result.peak_vram_mb:.0f} MB  ({compile_result.vram_util_pct:.1f}%)"
            )

            if best_p2 is not None:
                speedup_vs_p2 = best_p2.ms_per_opt_step / compile_result.ms_per_opt_step
                print(
                    f"  Speedup vs Phase 2 best (no compile): {speedup_vs_p2:.2f}x  "
                    f"({best_p2.ms_per_opt_step:,.1f} -> {compile_result.ms_per_opt_step:,.1f} ms/opt_step)"
                )
        else:
            print("  torch.compile test failed or was skipped due to OOM/error.")
        print()
    else:
        print("  Phase 3 skipped (--skip-compile).")
        print()

    # -----------------------------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  FINAL RECOMMENDATION")
    print("=" * 78)
    print()

    # Determine the single best result
    # Priority: compile (if available and faster) > Phase 2 > Phase 1
    if compile_result is not None:
        best_ms = compile_result.ms_per_opt_step
        use_compile = True
        rec_bs  = compile_result.batch_size
        rec_seq = compile_result.seq_len
        rec_ga  = compile_result.grad_accum
        rec_label = f"Phase 3 (torch.compile, mode={compile_result.compile_mode})"
    elif best_p2 is not None:
        best_ms = best_p2.ms_per_opt_step
        use_compile = False
        rec_bs  = best_p2.batch_size
        rec_seq = best_p2.seq_len
        rec_ga  = best_p2.grad_accum
        rec_label = "Phase 2 (grad_accum sweep)"
    else:
        # Estimate ms/opt_step from Phase 1 using baseline ga=28
        # (Phase 2 failed; we cannot measure directly)
        best_ms = best_p1.ms_per_step * 28  # approximate
        use_compile = False
        rec_bs  = best_p1.batch_size
        rec_seq = best_p1.seq_len
        rec_ga  = 28
        rec_label = "Phase 1 (extrapolated, Phase 2 unavailable)"

    print(f"  Source   : {rec_label}")
    print(f"  Config   : batch_size={rec_bs}, seq_len={rec_seq}, grad_accum={rec_ga}")
    print(f"           : eff_batch = {rec_bs * rec_ga}  (bs={rec_bs} x ga={rec_ga})")
    if use_compile:
        print(f"           : torch.compile(mode='reduce-overhead') = ON")
    print()

    print_training_time_estimate(best_ms, rec_label)
    print()

    # Also print baseline estimate for comparison
    print("  Baseline (current): bs=1, seq=1024, ga=28, no compile")
    print(f"    ms/opt_step    : {BASELINE_MS_PER_STEP:,.1f} ms")
    print(f"    estimated time : {fmt_hours(STEPS_REMAINING * BASELINE_MS_PER_STEP)}")
    print()

    overall_speedup = BASELINE_MS_PER_STEP / best_ms if best_ms > 0 else float("nan")
    print(f"  Overall speedup vs baseline: {overall_speedup:.2f}x")
    print()

    # -----------------------------------------------------------------------
    # YAML snippet and launch command
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  RECOMMENDED CONFIG")
    print("=" * 78)
    print_yaml_snippet(
        batch_size=rec_bs,
        seq_len=rec_seq,
        grad_accum=rec_ga,
        lr=7e-6,
    )
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()

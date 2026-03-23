"""
compile_bench.py — torch.compile speedup test for EVAFRILL-Mo 2.94B.

Tests torch.compile with different modes on the best config from Phase 1/2:
  bs=4, seq=2048, ga=7 (eff_batch=28)

Also tests bs=1 ga=28 (current baseline) with/without compile for fair comparison.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model import LLM
from model.config import LMConfig


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def load_model(device):
    ckpt = PROJECT_ROOT / "checkpoints" / "3b_final" / "checkpoint-0319772"
    config = LMConfig.from_yaml(ckpt / "config.yaml")
    config.use_fp8 = False
    model = LLM(config)

    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    converted = {}
    for k, v in sd.items():
        if "_extra_state" in k:
            continue
        if ".ffn.layer_norm_weight" in k:
            converted[k.replace(".ffn.layer_norm_weight", ".ffn_norm.weight")] = v
        elif ".ffn.fc1_weight" in k:
            d = v.shape[0] // 2
            converted[k.replace(".ffn.fc1_weight", ".ffn.gate_proj.weight")] = v[:d]
            converted[k.replace(".ffn.fc1_weight", ".ffn.up_proj.weight")] = v[d:]
        elif ".ffn.fc2_weight" in k:
            converted[k.replace(".ffn.fc2_weight", ".ffn.down_proj.weight")] = v
        else:
            converted[k] = v

    model.load_state_dict(converted)
    model = model.to(dtype=torch.bfloat16, device=device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def bench_full_step(model, bs, seq, ga, n_warmup, n_timed, device):
    """Measure full optimizer step: ga micro-steps + optimizer.step()."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=True)
    vocab = model.config.vocab_size

    def full_step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ga):
            ids = torch.randint(0, vocab, (bs, seq), device=device)
            tgt = torch.randint(0, vocab, (bs, seq), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, targets=tgt)
            (loss / ga).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Warmup
    try:
        for _ in range(n_warmup):
            full_step()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"    [ERR warmup] {e}")
        del optimizer
        cleanup()
        return None

    # Timed
    torch.cuda.reset_peak_memory_stats(0)
    try:
        t0 = time.perf_counter()
        for _ in range(n_timed):
            full_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"    [ERR timed] {e}")
        del optimizer
        cleanup()
        return None

    peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    ms_per_step = elapsed / n_timed * 1000
    tok_per_step = bs * seq * ga
    tok_per_sec = tok_per_step / (elapsed / n_timed)

    del optimizer
    cleanup()

    return {
        "ms_per_step": ms_per_step,
        "tok_per_sec": tok_per_sec,
        "peak_mb": peak_mb,
        "vram_pct": peak_mb / total_mb * 100,
    }


def main():
    device = torch.device("cuda")
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    print("=" * 70)
    print("  torch.compile Benchmark — EVAFRILL-Mo 2.94B")
    print("=" * 70)
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {total_mb:.0f} MB")
    print()

    print("Loading model ...")
    model = load_model(device)
    print("Model loaded.")
    print()

    # Test configs
    configs = [
        # (label, bs, seq, ga, compile_mode)
        ("baseline (current)",   1, 2048, 28, None),
        ("bs=4 no-compile",      4, 2048,  7, None),
        ("bs=4 compile=default", 4, 2048,  7, "default"),
        ("bs=4 compile=reduce-overhead", 4, 2048, 7, "reduce-overhead"),
        ("bs=1 compile=default", 1, 2048, 28, "default"),
        ("bs=1 compile=reduce-overhead", 1, 2048, 28, "reduce-overhead"),
    ]

    results = []

    for label, bs, seq, ga, compile_mode in configs:
        eff = bs * ga
        print(f"  [{label}] bs={bs}, seq={seq}, ga={ga}, eff={eff}")

        if compile_mode is not None:
            print(f"    Compiling (mode={compile_mode}) ... ", end="", flush=True)
            try:
                test_model = torch.compile(model, mode=compile_mode)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                results.append((label, bs, seq, ga, eff, compile_mode, None))
                continue
            n_warmup = 5  # compile needs more warmup
        else:
            test_model = model
            n_warmup = 3

        r = bench_full_step(test_model, bs, seq, ga, n_warmup=n_warmup, n_timed=3, device=device)

        if r is not None:
            print(f"    => {r['tok_per_sec']:,.0f} tok/s | {r['ms_per_step']:,.1f} ms/step | "
                  f"{r['peak_mb']:.0f} MB ({r['vram_pct']:.1f}%)")
        else:
            print(f"    => FAILED")

        results.append((label, bs, seq, ga, eff, compile_mode, r))

        # Reset compiled model
        if compile_mode is not None:
            del test_model
        cleanup()
        print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Label':<35} {'tok/s':>8} {'ms/step':>10} {'VRAM%':>7} {'speedup':>8}")
    print("-" * 70)

    baseline_ms = None
    for label, bs, seq, ga, eff, cm, r in results:
        if r is None:
            print(f"{label:<35} {'FAILED':>8}")
            continue
        if baseline_ms is None:
            baseline_ms = r["ms_per_step"]
        speedup = baseline_ms / r["ms_per_step"]
        print(f"{label:<35} {r['tok_per_sec']:>8,.0f} {r['ms_per_step']:>10,.1f} "
              f"{r['vram_pct']:>6.1f}% {speedup:>7.2f}x")

    # Training time estimates
    print()
    steps_left = 135000 - 410
    print("  Training time estimates (134,590 steps remaining):")
    for label, bs, seq, ga, eff, cm, r in results:
        if r is None:
            continue
        hours = steps_left * r["ms_per_step"] / 3_600_000
        days = hours / 24
        # Adjust for different eff_batch sizes:
        # If eff_batch changes, total steps change proportionally
        current_eff = 28  # current eff_batch
        step_ratio = current_eff / eff  # if eff_batch doubles, steps halve
        adj_hours = hours * step_ratio
        adj_days = adj_hours / 24
        print(f"    {label:<35}: {adj_hours:>7.1f}h ({adj_days:.1f} days) "
              f"[eff_batch={eff}, adj_steps={int(steps_left * step_ratio):,}]")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

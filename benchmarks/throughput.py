"""
benchmarks/throughput.py — Measure model throughput (tokens/sec) and memory.

Usage:
    python benchmarks/throughput.py --config configs/hybrid_3b.yaml --batch_size 1 --seq_len 512 --steps 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import LLM, LMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Model throughput benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    config = LMConfig.from_yaml(args.config)
    model = LLM(config).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params/1e9:.2f}B)")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"use_hybrid={config.use_hybrid}")
    if config.use_hybrid:
        print(f"  mamba_d_ffn={config.mamba_d_ffn}, n_groups={config.mamba_n_groups}")

    tokens_per_step = args.batch_size * args.seq_len

    # Warmup
    print(f"\nWarmup: {args.warmup} steps...")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=x)
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Timed runs
    print(f"Benchmark: {args.steps} steps...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.steps):
        x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=x)
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    total_tokens = tokens_per_step * args.steps
    tokens_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / args.steps) * 1000

    print(f"\n{'='*50}")
    print(f"  Steps          : {args.steps}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Seq len        : {args.seq_len}")
    print(f"  Tokens/step    : {tokens_per_step:,}")
    print(f"  Total time     : {elapsed:.2f}s")
    print(f"  Time/step      : {ms_per_step:.1f}ms")
    print(f"  Throughput     : {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Peak GPU mem   : {peak_mem:,.0f} MB")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

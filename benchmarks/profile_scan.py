"""
benchmarks/profile_scan.py — Profile Mamba2Block selective_scan vs FFN vs conv1d.

Usage:
    python benchmarks/profile_scan.py --d_model 3072 --d_ffn 4608 --seq_len 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.mamba_block import Mamba2Block


def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba2Block profiler")
    parser.add_argument("--d_model", type=int, default=3072)
    parser.add_argument("--d_ffn", type=int, default=0)
    parser.add_argument("--n_groups", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace_dir", type=str, default="benchmarks/traces")
    args = parser.parse_args()

    block = Mamba2Block(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_groups=args.n_groups,
    ).cuda()

    param_count = sum(p.numel() for p in block.parameters())
    print(f"Mamba2Block: {param_count:,} params")
    print(f"  d_model={args.d_model}, d_ffn={args.d_ffn}, n_groups={args.n_groups}")

    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(args.warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = block(x)
        y.sum().backward()
        block.zero_grad(set_to_none=True)

    # Profile
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        for _ in range(2 + args.steps):  # wait + warmup + active
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = block(x)
            y.sum().backward()
            block.zero_grad(set_to_none=True)
            prof.step()

    # Print key averages
    print(f"\n{'='*70}")
    print("Key averages (CUDA time):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\nChrome trace saved to: {trace_dir}/")


if __name__ == "__main__":
    main()

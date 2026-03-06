"""
Fast PPL evaluation on B200 — bfloat16, proper CUDA device setup.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/fast_ppl.py \
        --checkpoint checkpoints/korean_3b_fp8_run1/checkpoint-0057000 \
        --data data/3b_val.bin \
        --max_tokens 10000000 \
        --batch_size 32 \
        --output eval/outputs/ppl_3b_val.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.transformer import LLM


class SlidingWindowDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = max(0, (len(tokens) - seq_len + stride - 1) // stride)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        actual_end = min(end, len(self.tokens))
        chunk_len = actual_end - start

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets = torch.full((self.seq_len,), -100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        if chunk_len > 1:
            toks = torch.from_numpy(self.tokens[start:actual_end].astype(np.int64))
            input_ids[:chunk_len] = toks
            targets[:chunk_len - 1] = toks[1:]
        new_start = 0 if idx == 0 else self.stride
        if chunk_len > 1:
            for pos in range(new_start, chunk_len - 1):
                loss_mask[pos] = True
        return input_ids, targets, loss_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Max tokens to evaluate (0=all)")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    device = "cuda:0"  # Use CUDA_VISIBLE_DEVICES to select GPU

    print(f"Loading model from {args.checkpoint}...")
    t0 = time.time()
    model = LLM.from_pretrained(args.checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params/1e6:.1f}M params, bfloat16, loaded in {time.time()-t0:.1f}s")

    tokens = np.fromfile(args.data, dtype=np.uint16)
    total_tokens = len(tokens)
    if args.max_tokens > 0 and total_tokens > args.max_tokens:
        tokens = tokens[:args.max_tokens]
        print(f"Using {len(tokens):,}/{total_tokens:,} tokens (sampled)")
    else:
        print(f"Using all {total_tokens:,} tokens")

    ds = SlidingWindowDataset(tokens, args.seq_len, args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    n_batches = len(dl)
    print(f"Windows: {len(ds):,}, Batches: {n_batches:,}, "
          f"seq_len={args.seq_len}, stride={args.stride}, bs={args.batch_size}")

    total_nll = 0.0
    total_count = 0
    t_start = time.time()

    with torch.inference_mode():
        for i, (inp, tgt, mask) in enumerate(dl):
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                reduction="none"
            ).view(mask.shape)

            nll = (ce * mask.float()).sum().item()
            cnt = mask.sum().item()
            total_nll += nll
            total_count += cnt

            if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                elapsed = time.time() - t_start
                running_ppl = math.exp(total_nll / total_count)
                speed = (i + 1) / elapsed
                eta = (n_batches - i - 1) / speed
                print(f"  [{i+1}/{n_batches}] PPL={running_ppl:.4f} "
                      f"({speed:.1f} batch/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t_start
    avg_nll = total_nll / total_count
    ppl = math.exp(avg_nll)
    bpt = avg_nll / math.log(2)

    data_name = Path(args.data).stem
    print(f"\n{'='*50}")
    print(f"  Dataset: {data_name}")
    print(f"  Tokens evaluated: {total_count:,}")
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Bits/token: {bpt:.4f}")
    print(f"  Avg NLL: {avg_nll:.6f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*50}")

    result = {
        "dataset": data_name,
        "data_file": args.data,
        "total_tokens": int(total_tokens),
        "eval_tokens": int(total_count),
        "max_tokens_used": args.max_tokens if args.max_tokens > 0 else int(total_tokens),
        "perplexity": round(ppl, 4),
        "bits_per_token": round(bpt, 4),
        "avg_nll": round(avg_nll, 6),
        "elapsed_sec": round(elapsed, 1),
        "config": {
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "dtype": "bfloat16",
        }
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")

    return result


if __name__ == "__main__":
    main()

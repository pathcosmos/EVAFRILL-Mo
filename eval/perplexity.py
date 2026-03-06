"""
Compute sliding-window perplexity of a trained LLM on a binary token dataset.

The sliding-window approach avoids the boundary effect of chunking: a window
of ``seq_len`` tokens is evaluated every ``stride`` tokens.  Positions in
the first (stride) tokens of each window are considered "fresh" context and
their NLL contributions are accumulated; positions in the overlap region are
not double-counted because only the *new* stride tokens are aggregated at
each step.

Reference: Press et al., 2022 "Train Short, Test Long" (sliding-window PPL).

Usage:
    python eval/perplexity.py \
        --checkpoint checkpoints/checkpoint-0100000 \
        --data data/val.bin \
        --seq_len 2048 \
        --batch_size 4 \
        --device cuda:0 \
        --stride 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model.transformer import LLM


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Yields (input_ids, targets, loss_mask) tuples for sliding-window PPL.

    ``loss_mask`` is 1 for positions that contribute to the perplexity
    estimate (i.e. the *new* stride tokens at the right end of the window)
    and 0 for the context-only positions.

    Args:
        tokens:   Flat 1-D numpy array of token IDs (uint16).
        seq_len:  Context window size.
        stride:   Step size between consecutive windows.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int) -> None:
        self.tokens  = tokens
        self.seq_len = seq_len
        self.stride  = stride
        # Number of windows that fit inside the token array.
        self.n_windows = max(0, (len(tokens) - seq_len + stride - 1) // stride)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end   = start + self.seq_len

        # Clamp end to array length; pad if needed.
        actual_end = min(end, len(self.tokens))
        chunk_len  = actual_end - start  # may be < seq_len for last window

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets   = torch.full((self.seq_len,), fill_value=-100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        if chunk_len > 1:
            toks = torch.from_numpy(
                self.tokens[start : actual_end].astype(np.int64)
            )
            input_ids[: chunk_len]     = toks
            targets  [: chunk_len - 1] = toks[1:]  # shifted labels

        # The "new" tokens start at stride positions from the beginning of the
        # window (they haven't been seen as targets in any previous window).
        # For the very first window (idx == 0) all positions are new.
        new_start_in_window = 0 if idx == 0 else self.stride
        # Loss mask covers [new_start_in_window, chunk_len - 1) because we
        # predict token[t+1] from token[t], so the last input position has no
        # target within this window.
        if chunk_len > 1:
            mask_end = chunk_len - 1  # positions 0 … chunk_len-2 have valid targets
            for pos in range(new_start_in_window, mask_end):
                loss_mask[pos] = True

        return input_ids, targets, loss_mask


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_perplexity(
    model: torch.nn.Module,
    data_path: str,
    seq_len: int,
    batch_size: int,
    device: str,
    stride: int,
) -> float:
    """
    Compute sliding-window perplexity on the token file at ``data_path``.

    Returns:
        Perplexity (float).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    tokens = np.memmap(path, dtype="uint16", mode="r")
    total_tokens = len(tokens)
    print(f"Loaded {total_tokens:,} tokens from {path}")

    dataset = SlidingWindowDataset(tokens, seq_len=seq_len, stride=stride)
    if len(dataset) == 0:
        raise ValueError(
            f"No windows fit in {total_tokens} tokens with seq_len={seq_len}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()

    # Accumulate log-probabilities (sum of NLL) and the count of evaluated tokens.
    total_nll   = 0.0
    total_count = 0

    for batch_input_ids, batch_targets, batch_loss_mask in tqdm(
        loader, desc="Evaluating perplexity", unit="batch"
    ):
        batch_input_ids = batch_input_ids.to(device)     # [B, seq_len]
        batch_targets   = batch_targets.to(device)       # [B, seq_len]
        batch_loss_mask = batch_loss_mask.to(device)     # [B, seq_len]

        logits, _ = model(batch_input_ids)              # [B, seq_len, vocab]

        # Cross-entropy loss per position (reduction='none').
        B, S, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * S, V),
            batch_targets.reshape(B * S),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, S)  # [B, seq_len]

        # Apply sliding-window loss mask.
        # Positions where targets == -100 are already zeroed by ignore_index;
        # we additionally zero positions outside the stride window.
        masked_ce = ce * batch_loss_mask.float()
        total_nll   += masked_ce.sum().item()
        total_count += batch_loss_mask.sum().item()

    if total_count == 0:
        raise RuntimeError("No valid token positions were evaluated.")

    avg_nll    = total_nll / total_count
    perplexity = math.exp(avg_nll)
    return perplexity


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sliding-window perplexity of a trained LLM."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the .bin token data file.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Context window length (default: 2048).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help=(
            "Stride for sliding window PPL; smaller = more accurate, "
            "slower (default: 512)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")
    model = LLM.from_pretrained(str(ckpt_path)).to(device=args.device, dtype=torch.float16)
    model.eval()
    print(f"Model parameters: {model.num_params / 1e6:.1f}M")

    print(
        f"\nPerplexity config: seq_len={args.seq_len}, "
        f"stride={args.stride}, batch_size={args.batch_size}"
    )

    ppl = compute_perplexity(
        model=model,
        data_path=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        stride=args.stride,
    )

    print("\n" + "=" * 50)
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Bits/token: {math.log2(math.e) * math.log(ppl):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

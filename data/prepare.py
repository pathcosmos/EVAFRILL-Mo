"""
Prepare raw text files (or a HuggingFace dataset) for LLM training.

Tokenizes all input text, concatenates all token IDs into a single flat
sequence, splits into train / validation sets, and saves each as a uint16
numpy binary file (.bin) ready for TextDataset / PackedDataset.

Usage — glob of local text files:
    python data/prepare.py \
        --input  "data/raw/*.txt" \
        --output  data/train.bin \
        --val_output data/val.bin \
        --tokenizer tokenizer/tokenizer.json \
        --val_split 0.005 \
        --seed 42

Usage — HuggingFace dataset (streaming):
    python data/prepare.py \
        --hf_dataset  allenai/c4 \
        --hf_subset   en \
        --hf_split    train \
        --hf_text_col text \
        --output      data/train.bin \
        --val_output  data/val.bin \
        --tokenizer   tokenizer/tokenizer.json \
        --val_split   0.005
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


def find_input_files(pattern: str) -> list[str]:
    """Resolve a glob pattern or a plain file path to a list of files."""
    if any(c in pattern for c in ("*", "?", "[")):
        files = sorted(glob.glob(pattern, recursive=True))
    else:
        files = [pattern] if Path(pattern).exists() else []
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")
    return files


def tokenize_file(path: str, tokenizer: Tokenizer) -> list[int]:
    """Read a single text file and return its token IDs."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return tokenizer.encode(text).ids


def derive_val_path(output_path: Path, val_output_arg: str | None) -> Path:
    """Return the val .bin path, either explicitly provided or auto-derived."""
    if val_output_arg:
        return Path(val_output_arg)
    # If the stem contains "train", swap it for "val".
    if "train" in output_path.name:
        candidate = output_path.parent / output_path.name.replace("train", "val")
        if candidate != output_path:
            return candidate
    # Generic fallback: append _val before the suffix.
    return output_path.with_name(output_path.stem + "_val" + output_path.suffix)


def save_bin(tokens: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array(tokens, dtype=np.uint16).tofile(str(path))


def _fmt_bytes(n_tokens: int) -> str:
    """Return a human-readable size string for a uint16 token array."""
    nbytes = n_tokens * 2  # uint16 = 2 bytes per token
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------

def iter_tokens_from_files(
    input_files: list[str],
    tokenizer: Tokenizer,
    seed: int,
) -> tuple[list[int], int]:
    """
    Tokenize every file, shuffle at file level, flatten, and return
    (all_tokens_shuffled, file_count).
    """
    per_file_tokens: list[list[int]] = []
    for fpath in tqdm(input_files, desc="Tokenizing", unit="file"):
        per_file_tokens.append(tokenize_file(fpath, tokenizer))

    rng = random.Random(seed)
    rng.shuffle(per_file_tokens)

    all_tokens: list[int] = []
    for toks in per_file_tokens:
        all_tokens.extend(toks)

    return all_tokens, len(input_files)


def iter_tokens_from_hf(
    hf_dataset: str,
    hf_subset: str | None,
    hf_split: str,
    hf_text_col: str,
    tokenizer: Tokenizer,
) -> tuple[list[int], int]:
    """
    Stream a HuggingFace dataset row-by-row, tokenize each row's text column,
    and return (all_tokens, row_count).

    Rows are appended in streaming order; no shuffle is performed here because
    the stream may be very large.  A seed-based split by position is used later.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for --hf_dataset. "
            "Install it with: pip install datasets"
        )

    print(f"Streaming HuggingFace dataset: {hf_dataset}"
          + (f" / {hf_subset}" if hf_subset else "")
          + f"  split={hf_split}")

    ds = load_dataset(
        hf_dataset,
        hf_subset,
        split=hf_split,
        streaming=True,
        trust_remote_code=True,
    )

    all_tokens: list[int] = []
    row_count = 0
    pbar = tqdm(desc="Tokenizing rows", unit="row")
    for row in ds:
        text = row.get(hf_text_col, "")
        if text:
            all_tokens.extend(tokenizer.encode(text).ids)
        row_count += 1
        pbar.update(1)
    pbar.close()

    return all_tokens, row_count


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize text sources and save as uint16 binary files for LLM training. "
            "Accepts either a glob of local text files (--input) or a HuggingFace "
            "dataset (--hf_dataset)."
        )
    )

    # --- Input source (mutually exclusive) ---
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        default=None,
        help='Glob pattern or path to a single text file, e.g. "data/raw/*.txt"',
    )
    source.add_argument(
        "--hf_dataset",
        default=None,
        metavar="DATASET",
        help="HuggingFace dataset name, e.g. allenai/c4 (alternative to --input)",
    )

    # --- HuggingFace-specific options ---
    parser.add_argument(
        "--hf_subset",
        default=None,
        metavar="SUBSET",
        help="Dataset subset / config name, e.g. 'en' for allenai/c4",
    )
    parser.add_argument(
        "--hf_split",
        default="train",
        metavar="SPLIT",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--hf_text_col",
        default="text",
        metavar="COLUMN",
        help="Name of the text column in the dataset (default: text)",
    )

    # --- Output paths ---
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the training binary, e.g. data/train.bin",
    )
    parser.add_argument(
        "--val_output",
        default=None,
        metavar="PATH",
        help=(
            "Explicit output path for the validation binary "
            "(default: auto-derived from --output, e.g. train.bin → val.bin)"
        ),
    )

    # --- Tokenizer ---
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/tokenizer.json",
        help="Path to a trained tokenizer JSON file (default: tokenizer/tokenizer.json)",
    )

    # --- Split / reproducibility ---
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.005,
        help="Fraction of tokens reserved for validation (default: 0.005)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split (default: 42)",
    )

    args = parser.parse_args()

    # Require at least one input source.
    if args.input is None and args.hf_dataset is None:
        parser.error("One of --input or --hf_dataset is required.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load tokenizer ----
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Warn early if IDs could overflow uint16.
    if vocab_size > 65535:
        print(
            "WARNING: vocab_size > 65535; token IDs above 65535 will be "
            "truncated when cast to uint16.",
            file=sys.stderr,
        )

    # ---- Collect tokens from the chosen source ----
    if args.hf_dataset:
        all_tokens, source_count = iter_tokens_from_hf(
            hf_dataset=args.hf_dataset,
            hf_subset=args.hf_subset,
            hf_split=args.hf_split,
            hf_text_col=args.hf_text_col,
            tokenizer=tokenizer,
        )
        source_label = f"{source_count:,} rows"
    else:
        input_files = find_input_files(args.input)
        print(f"Found {len(input_files)} input file(s).")
        all_tokens, source_count = iter_tokens_from_files(
            input_files=input_files,
            tokenizer=tokenizer,
            seed=args.seed,
        )
        source_label = f"{source_count:,} files"

    total_tokens = len(all_tokens)

    # ---- Split into train / val ----
    val_size   = max(1, int(total_tokens * args.val_split))
    train_size = total_tokens - val_size

    train_tokens = all_tokens[:train_size]
    val_tokens   = all_tokens[train_size:]

    # ---- Resolve output paths ----
    train_path = Path(args.output)
    val_path   = derive_val_path(train_path, args.val_output)

    # ---- Save ----
    print(f"\nSaving train data -> {train_path}")
    save_bin(train_tokens, train_path)

    print(f"Saving val data   -> {val_path}")
    save_bin(val_tokens, val_path)

    # ---- Final stats ----
    tokens_per_step = 8 * 2048 * 4 * 8  # bs=8, seq=2048, accum=4, 8 GPUs
    estimated_steps = train_size // tokens_per_step

    print()
    print(f"Tokenizer: {args.tokenizer} (vocab_size={vocab_size:,})")
    print(f"Total tokens: {total_tokens:,}")
    print(
        f"Train tokens: {train_size:,}"
        f" (stored in {train_path}, {_fmt_bytes(train_size)})"
    )
    print(
        f"Val tokens:   {val_size:,}"
        f"     (stored in {val_path}, {_fmt_bytes(val_size)})"
    )
    print(
        f"Estimated steps (bs=8, seq=2048, 8 GPUs, accum=4): {estimated_steps:,}"
    )


if __name__ == "__main__":
    main()

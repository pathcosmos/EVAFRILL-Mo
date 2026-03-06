"""
Train a Byte-Level BPE tokenizer on raw text files.

The tokenizer is saved in two formats:
  1. Native HuggingFace ``tokenizers`` format (vocab.json + merges.txt) inside
     the output directory — for fast loading with ByteLevelBPETokenizer.
  2. A ``tokenizer.json`` file (PreTrainedTokenizerFast) in the output directory
     — for easy loading with transformers.AutoTokenizer.

Usage:
    python tokenizer/train_tokenizer.py \
        --input  "data/raw/*.txt" \
        --output  tokenizer/ \
        --vocab_size 32000 \
        --min_frequency 2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: list[str] = ["<pad>", "<s>", "</s>", "<unk>"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_input_files(pattern: str) -> list[str]:
    """Resolve a glob pattern or a plain file path to a sorted list of paths."""
    if any(c in pattern for c in ("*", "?", "[")):
        files = sorted(glob.glob(pattern, recursive=True))
    else:
        files = [pattern] if Path(pattern).exists() else []
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Byte-Level BPE tokenizer and save to disk."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Glob pattern for training text files, e.g. "data/raw/*.txt"',
    )
    parser.add_argument(
        "--output",
        default="tokenizer/",
        help="Output directory for the trained tokenizer (default: tokenizer/)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Target vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a pair to be merged (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Discover input files ----
    input_files = find_input_files(args.input)
    print(f"Found {len(input_files)} training file(s).")

    # ---- Create output directory ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Initialise tokenizer ----
    tokenizer = ByteLevelBPETokenizer()

    # ---- Train ----
    print(
        f"\nTraining BPE tokenizer | vocab_size={args.vocab_size} "
        f"| min_frequency={args.min_frequency} ..."
    )
    tokenizer.train(
        files=input_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # ---- Add special tokens explicitly (ensures they have the right IDs) ----
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # ---- Save native format (vocab.json + merges.txt) ----
    tokenizer.save_model(str(output_dir))
    print(f"\nSaved vocab.json + merges.txt to: {output_dir}")

    # ---- Wrap in PreTrainedTokenizerFast and save tokenizer.json ----
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer_json_path = output_dir / "tokenizer.json"
    fast_tokenizer.save_pretrained(str(output_dir))
    print(f"Saved PreTrainedTokenizerFast to: {output_dir}")
    print(f"  -> tokenizer.json: {tokenizer_json_path}")

    # ---- Stats ----
    actual_vocab_size = tokenizer.get_vocab_size()
    print("\n" + "=" * 50)
    print("Tokenizer training statistics")
    print("=" * 50)
    print(f"  Training files  : {len(input_files):>10,}")
    print(f"  Target vocab    : {args.vocab_size:>10,}")
    print(f"  Actual vocab    : {actual_vocab_size:>10,}")
    print(f"  Min frequency   : {args.min_frequency:>10,}")
    print(f"  Special tokens  : {SPECIAL_TOKENS}")
    print(f"  Output dir      : {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()

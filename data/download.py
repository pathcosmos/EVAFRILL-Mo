"""
data/download.py — Download text corpora from HuggingFace datasets.

Default sources (no HF token required):
    1. wikimedia/wikipedia  20231101.ko  (Korean Wikipedia, ~600MB text)
    2. wikimedia/wikipedia  20231101.en  (English Wikipedia, streamed/sampled)

Usage:
    # Korean + English Wikipedia (default)
    python data/download.py

    # Korean only
    python data/download.py --langs ko

    # Custom sample sizes
    python data/download.py --langs ko en --ko_max 2000000 --en_max 500000

    # Custom dataset
    python data/download.py --dataset roneneldan/TinyStories --split train --text_col story
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Minimal text cleaning: strip whitespace, collapse excessive newlines."""
    text = text.strip()
    # Collapse 3+ consecutive newlines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Core download helpers
# ---------------------------------------------------------------------------

def _open_shard(output_dir: Path, prefix: str, shard_idx: int):
    """Return an open file handle for a new shard."""
    shard_path = output_dir / f"{prefix}_{shard_idx:04d}.txt"
    return open(shard_path, "w", encoding="utf-8")


def download_wikipedia(
    lang: str,
    output_dir: Path,
    max_articles: int,
    shard_size: int,
) -> dict:
    """
    Stream one Wikipedia language dump and write sharded plain-text files.

    Returns a stats dict with keys: articles, chars, tokens_est, files.
    """
    dataset_name = "wikimedia/wikipedia"
    config = f"20231101.{lang}"
    prefix = f"{lang}_wiki"

    print(f"\n[{lang}] Loading {dataset_name} / {config} …")

    try:
        ds = load_dataset(
            dataset_name,
            config,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"  WARNING: Failed to load {dataset_name}/{config}: {exc}", file=sys.stderr)
        return {"articles": 0, "chars": 0, "tokens_est": 0, "files": 0}

    count = 0
    total_chars = 0
    shard_idx = 0
    shard_count = 0  # articles written to the current shard

    shard_fh = _open_shard(output_dir, prefix, shard_idx)
    files = 1

    try:
        iterator = tqdm(ds, desc=f"  {lang}", unit="art", dynamic_ncols=True)
        for example in iterator:
            text = example.get("text", "")
            text = clean_text(text)

            if len(text) < 200:
                continue

            # Rotate shard if needed
            if shard_count > 0 and shard_count % shard_size == 0:
                shard_fh.close()
                shard_idx += 1
                shard_fh = _open_shard(output_dir, prefix, shard_idx)
                files += 1

            if shard_count == 0:
                shard_fh.write(text)
            else:
                shard_fh.write("\n\n" + text)

            shard_count += 1
            count += 1
            total_chars += len(text)

            # Progress print every 10,000 articles
            if count % 10_000 == 0:
                tqdm.write(f"  {lang}: {count:,} articles, {total_chars / 1e6:.1f}M chars")

            if max_articles and count >= max_articles:
                break

    except Exception as exc:
        print(f"\n  WARNING: Stream interrupted for {lang}: {exc}", file=sys.stderr)
    finally:
        shard_fh.close()

    tokens_est = total_chars // 4

    print(
        f"\n  [{lang}] Done — "
        f"{count:,} articles, "
        f"{total_chars / 1e6:.1f}M chars, "
        f"~{tokens_est / 1e6:.1f}M tokens (est.), "
        f"{files} shard file(s)"
    )

    return {"articles": count, "chars": total_chars, "tokens_est": tokens_est, "files": files}


def download_custom_dataset(
    dataset_name: str,
    output_dir: Path,
    subset: str | None,
    split: str,
    text_col: str,
    shard_size: int,
    max_rows: int = 0,
) -> dict:
    """
    Download an arbitrary HuggingFace dataset and write sharded plain-text files.

    Returns a stats dict with keys: articles, chars, tokens_est, files.
    """
    load_kwargs: dict = dict(split=split, streaming=True, trust_remote_code=True)
    if subset:
        load_kwargs["name"] = subset

    print(f"\n[custom] Loading {dataset_name}" + (f" / {subset}" if subset else "") + f" …")

    try:
        ds = load_dataset(dataset_name, **load_kwargs)
    except Exception as exc:
        print(f"  WARNING: Failed to load {dataset_name}: {exc}", file=sys.stderr)
        return {"articles": 0, "chars": 0, "tokens_est": 0, "files": 0}

    # Build a filesystem-safe prefix from the dataset name
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", dataset_name)
    prefix = f"{safe_name}_{split}"

    count = 0
    total_chars = 0
    shard_idx = 0
    shard_count = 0
    files = 1

    shard_fh = _open_shard(output_dir, prefix, shard_idx)

    try:
        iterator = tqdm(ds, desc="  custom", unit="row", dynamic_ncols=True)
        for example in iterator:
            text = example.get(text_col, "")
            if not isinstance(text, str):
                text = str(text)
            text = clean_text(text)

            if len(text) < 1:
                continue

            if shard_count > 0 and shard_count % shard_size == 0:
                shard_fh.close()
                shard_idx += 1
                shard_fh = _open_shard(output_dir, prefix, shard_idx)
                files += 1

            if shard_count == 0:
                shard_fh.write(text)
            else:
                shard_fh.write("\n\n" + text)

            shard_count += 1
            count += 1
            total_chars += len(text)

            if count % 10_000 == 0:
                tqdm.write(f"  custom: {count:,} rows, {total_chars / 1e6:.1f}M chars")

            if max_rows > 0 and count >= max_rows:
                break

    except Exception as exc:
        print(f"\n  WARNING: Stream interrupted: {exc}", file=sys.stderr)
    finally:
        shard_fh.close()

    tokens_est = total_chars // 4

    print(
        f"\n  [custom] Done — "
        f"{count:,} rows, "
        f"{total_chars / 1e6:.1f}M chars, "
        f"~{tokens_est / 1e6:.1f}M tokens (est.), "
        f"{files} shard file(s)"
    )

    return {"articles": count, "chars": total_chars, "tokens_est": tokens_est, "files": files}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download text corpora from HuggingFace datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where sharded .txt files are written.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["ko", "en"],
        metavar="LANG",
        help="Wikipedia language codes to download.",
    )
    parser.add_argument(
        "--ko_max",
        type=int,
        default=0,
        help="Max Korean Wikipedia articles (0 = all).",
    )
    parser.add_argument(
        "--en_max",
        type=int,
        default=300_000,
        help="Max English Wikipedia articles (0 = all).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000,
        help="Number of articles per shard file.",
    )
    # Custom dataset overrides
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override: HuggingFace dataset name (e.g. roneneldan/TinyStories).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset / config name (used with --dataset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (used with --dataset).",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Column name containing the text (used with --dataset).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Max rows to download from --dataset (0 = unlimited).",
    )
    return parser.parse_args()


def _lang_max(lang: str, args: argparse.Namespace) -> int:
    """Return the max-articles limit for a given Wikipedia language code."""
    mapping = {
        "ko": args.ko_max,
        "en": args.en_max,
    }
    return mapping.get(lang, 0)


def print_summary(all_stats: dict[str, dict]) -> None:
    """Print a final summary table for all downloaded sources."""
    print("\n" + "=" * 70)
    print(f"{'Source':<20} {'Articles':>12} {'Chars (M)':>12} {'Tokens est.(M)':>16} {'Files':>6}")
    print("-" * 70)
    totals: dict = {"articles": 0, "chars": 0, "tokens_est": 0, "files": 0}
    for name, stats in all_stats.items():
        print(
            f"{name:<20} "
            f"{stats['articles']:>12,} "
            f"{stats['chars'] / 1e6:>12.1f} "
            f"{stats['tokens_est'] / 1e6:>16.1f} "
            f"{stats['files']:>6}"
        )
        for key in totals:
            totals[key] += stats[key]
    print("-" * 70)
    print(
        f"{'TOTAL':<20} "
        f"{totals['articles']:>12,} "
        f"{totals['chars'] / 1e6:>12.1f} "
        f"{totals['tokens_est'] / 1e6:>16.1f} "
        f"{totals['files']:>6}"
    )
    print("=" * 70)


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    all_stats: dict[str, dict] = {}

    if args.dataset is not None:
        # Custom dataset mode — ignore --langs
        stats = download_custom_dataset(
            dataset_name=args.dataset,
            output_dir=output_dir,
            subset=args.subset,
            split=args.split,
            text_col=args.text_col,
            shard_size=args.shard_size,
            max_rows=args.max_rows,
        )
        all_stats[args.dataset] = stats
    else:
        # Wikipedia mode
        for lang in args.langs:
            max_articles = _lang_max(lang, args)
            try:
                stats = download_wikipedia(
                    lang=lang,
                    output_dir=output_dir,
                    max_articles=max_articles,
                    shard_size=args.shard_size,
                )
            except Exception as exc:
                print(
                    f"\n  WARNING: Unexpected error for lang={lang}: {exc}",
                    file=sys.stderr,
                )
                stats = {"articles": 0, "chars": 0, "tokens_est": 0, "files": 0}
            all_stats[f"{lang}_wiki"] = stats

    print_summary(all_stats)


if __name__ == "__main__":
    main()

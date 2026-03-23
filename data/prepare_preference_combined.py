#!/usr/bin/env python3
"""
data/prepare_preference_combined.py — Combine and normalize preference datasets.

Reads multiple JSONL preference files and normalizes them into a unified format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python data/prepare_preference_combined.py \
        --input_dir data/preference \
        --output data/preference/combined_preference.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalize_record(obj: dict) -> dict | None:
    """Normalize a single JSONL record to {prompt, chosen, rejected}."""
    # Extract prompt
    prompt = obj.get("prompt") or obj.get("question") or obj.get("instruction", "")

    # Handle instruction + input format
    inp = obj.get("input", "")
    if inp and inp.strip() and not obj.get("prompt"):
        prompt = f"{prompt}\n{inp.strip()}" if prompt else inp.strip()

    if not prompt or not prompt.strip():
        return None

    # Extract chosen/rejected
    chosen = obj.get("chosen", "")
    rejected = obj.get("rejected", "")

    # Some datasets have chosen/rejected as lists of messages
    if isinstance(chosen, list):
        # Extract assistant content from conversation format
        chosen = _extract_assistant_text(chosen)
    if isinstance(rejected, list):
        rejected = _extract_assistant_text(rejected)

    # Handle orig_response_A/B + orig_preference format (nayohan-style)
    if not chosen and not rejected:
        resp_a = obj.get("orig_response_A", "")
        resp_b = obj.get("orig_response_B", "")
        pref = obj.get("orig_preference", "")
        if resp_a and resp_b and pref:
            # Use orig_instruction as prompt if available
            if not prompt:
                prompt = obj.get("orig_instruction", "")
            if pref == "A":
                chosen, rejected = resp_a, resp_b
            elif pref == "B":
                chosen, rejected = resp_b, resp_a
            else:
                return None

    if not chosen or not rejected:
        return None

    chosen = chosen.strip()
    rejected = rejected.strip()
    prompt = prompt.strip()

    # Filter: chosen and rejected should be different
    if chosen == rejected:
        return None

    # Filter: minimum length
    if len(chosen) < 5 or len(rejected) < 5:
        return None

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _extract_assistant_text(messages: list) -> str:
    """Extract the last assistant message from a conversation list."""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "assistant" and content:
                return content
    # Fallback: join all content
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            parts.append(msg.get("content", ""))
        elif isinstance(msg, str):
            parts.append(msg)
    return "\n".join(parts)


def process_file(path: Path) -> tuple[list[dict], int, int]:
    """Process a single JSONL file, returning (records, total_lines, skipped)."""
    records = []
    total = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            normalized = normalize_record(obj)
            if normalized is None:
                skipped += 1
                continue

            records.append(normalized)

    return records, total, skipped


def main():
    parser = argparse.ArgumentParser(description="Combine preference datasets")
    parser.add_argument("--input_dir", type=Path, default=Path("data/preference"),
                        help="Directory containing preference JSONL files")
    parser.add_argument("--output", type=Path, default=Path("data/preference/combined_preference.jsonl"),
                        help="Output combined JSONL file")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    jsonl_files = sorted(args.input_dir.glob("*.jsonl"))
    # Exclude the output file itself
    jsonl_files = [f for f in jsonl_files if f.name != args.output.name]

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {args.input_dir}")

    print(f"Found {len(jsonl_files)} JSONL files in {args.input_dir}")
    print("=" * 70)

    all_records = []
    for fpath in jsonl_files:
        print(f"Processing: {fpath.name} ...", end=" ", flush=True)
        records, total, skipped = process_file(fpath)
        all_records.extend(records)
        print(f"  {len(records):,} records (from {total:,} lines, {skipped:,} skipped)")

    print("=" * 70)
    print(f"Total: {len(all_records):,} preference pairs")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Written to: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

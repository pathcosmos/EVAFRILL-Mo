#!/usr/bin/env python3
"""
data/merge_bins.py — 여러 uint16 .bin 파일을 하나로 병합.

Usage:
    python data/merge_bins.py input1.bin input2.bin ... output.bin

마지막 인수가 출력 경로, 나머지는 입력 파일.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def merge_bins(input_paths: list[Path], output_path: Path) -> None:
    arrays = [np.memmap(p, dtype="uint16", mode="r") for p in input_paths]
    total = sum(len(a) for a in arrays)
    print(f"Merging {len(arrays)} files → {total:,} tokens total")

    output = np.memmap(output_path, dtype="uint16", mode="w+", shape=(total,))
    offset = 0
    for p, arr in zip(input_paths, arrays):
        n = len(arr)
        output[offset : offset + n] = arr
        offset += n
        print(f"  {p.name}: {n:,} tokens")

    output.flush()
    print(f"\nSaved → {output_path}  ({total * 2 / 1e9:.2f} GB)")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python data/merge_bins.py input1.bin ... inputN.bin output.bin")
        sys.exit(1)

    *inputs, output = sys.argv[1:]
    input_paths = [Path(p) for p in inputs]
    output_path = Path(output)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        print(f"ERROR: Files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merge_bins(input_paths, output_path)


if __name__ == "__main__":
    main()

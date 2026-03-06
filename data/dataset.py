"""
Dataset classes for LLM training.

TextDataset:    Sliding window (stride 1) over a memory-mapped uint16 binary file.
PackedDataset: Non-overlapping windows (stride = seq_len) over the same file format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Sliding-window dataset over a memory-mapped numpy uint16 binary token file.

    Each sample is a (input_ids, targets) pair of length seq_len, where
    targets is input_ids shifted by one position.  Windows overlap by
    (seq_len - 1) tokens, i.e. stride = 1.

    Args:
        data_path: Path to the .bin file produced by data/prepare.py.
        seq_len:   Number of tokens per sample (context length).
    """

    def __init__(self, data_path: Union[str, Path], seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        # Memory-map for zero-copy random access.
        self.data: np.ndarray = np.memmap(path, dtype="uint16", mode="r")
        # Hint OS to preload entire file into page cache (2.2TB RAM available)
        import mmap as _mmap
        try:
            self.data._mmap.madvise(_mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass  # madvise not available on all platforms
        if len(self.data) < seq_len + 1:
            raise ValueError(
                f"Data file has only {len(self.data)} tokens, "
                f"need at least {seq_len + 1}."
            )

    def __len__(self) -> int:
        # Each window needs seq_len tokens plus one extra for the target shift.
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Slice from the memmap (returns a uint16 numpy view).
        chunk = self.data[idx : idx + self.seq_len + 1]
        # Cast to int32 (not int64) to halve CPU worker memory usage:
        # uint16 (2 B) → int32 (4 B) instead of uint16 → int64 (8 B, 4× bloat).
        # int32 is sufficient for vocab_size=64000 (max token id 65535 fits in int32).
        # The int32→int64 (long) promotion happens on GPU inside _step(), for free.
        chunk = torch.from_numpy(chunk.astype(np.int32))
        input_ids = chunk[:-1]   # [seq_len]
        targets   = chunk[1:]    # [seq_len]
        return input_ids, targets


class PackedDataset(Dataset):
    """
    Non-overlapping packed dataset over a memory-mapped uint16 binary token file.

    Intended for data that has already been packed (documents concatenated with
    EOS tokens).  Windows do not overlap; stride = seq_len.

    The target sequence is shifted by one token relative to input_ids.  Because
    the last token of a window shares its target with the *first* token of the
    next window, the final target position is filled with -1 (the standard
    ``ignore_index`` for ``nn.CrossEntropyLoss``).

    Args:
        data_path: Path to the .bin file produced by data/prepare.py.
        seq_len:   Number of tokens per sample (context length).
    """

    def __init__(self, data_path: Union[str, Path], seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        self.data: np.ndarray = np.memmap(path, dtype="uint16", mode="r")
        # Optimize mmap for shuffled random access pattern (DistributedSampler)
        import mmap as _mmap
        try:
            self.data._mmap.madvise(_mmap.MADV_RANDOM)    # disable kernel read-ahead (random access)
            self.data._mmap.madvise(_mmap.MADV_WILLNEED)  # async prefault into page cache
        except (AttributeError, OSError):
            pass
        if len(self.data) < seq_len:
            raise ValueError(
                f"Data file has only {len(self.data)} tokens, "
                f"need at least {seq_len}."
            )

    def __len__(self) -> int:
        return len(self.data) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end   = start + self.seq_len

        # Cast to int32 (not int64) to halve CPU worker memory usage.
        # int32 is sufficient for vocab_size=64000; int32→long promotion on GPU.
        input_ids = torch.from_numpy(
            self.data[start:end].astype(np.int32)
        )  # [seq_len]

        # Targets are shifted by one.  If end < len(data) we can read the
        # extra token normally; otherwise pad the last position with -1.
        if end < len(self.data):
            targets = torch.from_numpy(
                self.data[start + 1 : end + 1].astype(np.int32)
            )  # [seq_len]
        else:
            # Last window: all but the final position can be computed.
            # Use int32 for the filled portion; -1 fits in int32.
            targets = torch.full((self.seq_len,), fill_value=-1, dtype=torch.int32)
            if end - start - 1 > 0:
                targets[: self.seq_len - 1] = torch.from_numpy(
                    self.data[start + 1 : end].astype(np.int32)
                )

        return input_ids, targets

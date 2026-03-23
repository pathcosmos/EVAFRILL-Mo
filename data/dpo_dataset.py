"""
DPO (Direct Preference Optimization) dataset for the Korean LLM project.

Reads JSONL files containing preference pairs (prompt, chosen, rejected).
Applies the same chat template as SFT:

    <|user|>
    {prompt}
    <|assistant|>
    {response}</s>

Each sample produces two sequences:
  - chosen:   [prompt_tokens + chosen_response_tokens]
  - rejected: [prompt_tokens + rejected_response_tokens]

Labels use -1 for prompt tokens (ignore_index) so the DPO loss
only considers the response portion.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Chat template tags (same as SFT)
_USER_TAG = "<|user|>\n"
_ASSISTANT_TAG = "<|assistant|>\n"
_EOS_STRING = "</s>"


def _parse_preference_lines(args: tuple) -> list[tuple[str, str, str]]:
    """Parse a chunk of JSONL lines into (prompt, chosen, rejected) triples."""
    lines, path_str, start_lineno = args
    triples: list[tuple[str, str, str]] = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract prompt
        prompt = obj.get("prompt") or obj.get("question") or obj.get("instruction", "")
        if not prompt:
            continue

        # If instruction format has "input", append it
        inp = obj.get("input", "")
        if inp and inp.strip():
            prompt = f"{prompt}\n{inp.strip()}"

        chosen = obj.get("chosen", "")
        rejected = obj.get("rejected", "")

        if not chosen or not rejected:
            continue

        triples.append((prompt.strip(), chosen.strip(), rejected.strip()))

    return triples


class DPODataset(Dataset):
    """
    DPO preference dataset from JSONL files.

    Each sample returns a dict with:
        chosen_ids:    (seq_len,) — full token sequence for chosen response
        chosen_labels: (seq_len,) — -1 for prompt, token_ids for response
        rejected_ids:    (seq_len,) — full token sequence for rejected response
        rejected_labels: (seq_len,) — -1 for prompt, token_ids for response
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        eos_id = tokenizer.token_to_id(_EOS_STRING)
        if eos_id is None:
            raise ValueError(f"EOS token '{_EOS_STRING}' not found in tokenizer")
        self.eos_token_id = eos_id

        data_path = Path(data_path)

        # Try loading from cache
        cache_path = self._cache_path(data_path, max_seq_len)
        if cache_path is not None and cache_path.exists():
            import time as _time
            _t0 = _time.perf_counter()
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.samples = cached["samples"]
            _elapsed = _time.perf_counter() - _t0
            print(f"[DPODataset] Loaded {len(self.samples)} samples from cache in {_elapsed:.1f}s")
            return

        # Load raw JSONL
        raw_triples = self._load_jsonl(data_path)

        # Tokenize
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        skipped = 0
        truncated = 0

        for prompt_text, chosen_text, rejected_text in raw_triples:
            # Build chat-formatted strings
            prompt_str = f"{_USER_TAG}{prompt_text}\n{_ASSISTANT_TAG}"
            chosen_str = f"{chosen_text}{_EOS_STRING}"
            rejected_str = f"{rejected_text}{_EOS_STRING}"

            prompt_ids = tokenizer.encode(prompt_str).ids
            chosen_resp_ids = tokenizer.encode(chosen_str).ids
            rejected_resp_ids = tokenizer.encode(rejected_str).ids

            # Skip if prompt alone is too long
            if len(prompt_ids) >= max_seq_len - 10:
                skipped += 1
                continue

            # Build chosen sequence
            chosen_full = prompt_ids + chosen_resp_ids
            if len(chosen_full) > max_seq_len:
                truncated += 1
                allowed = max_seq_len - len(prompt_ids)
                if allowed <= 0:
                    skipped += 1
                    continue
                chosen_resp_ids = chosen_resp_ids[:allowed]
                if chosen_resp_ids[-1] != self.eos_token_id:
                    chosen_resp_ids[-1] = self.eos_token_id
                chosen_full = prompt_ids + chosen_resp_ids

            # Build rejected sequence
            rejected_full = prompt_ids + rejected_resp_ids
            if len(rejected_full) > max_seq_len:
                truncated += 1
                allowed = max_seq_len - len(prompt_ids)
                if allowed <= 0:
                    skipped += 1
                    continue
                rejected_resp_ids = rejected_resp_ids[:allowed]
                if rejected_resp_ids[-1] != self.eos_token_id:
                    rejected_resp_ids[-1] = self.eos_token_id
                rejected_full = prompt_ids + rejected_resp_ids

            # Build tensors
            chosen_input = torch.tensor(chosen_full, dtype=torch.long)
            chosen_labels = torch.full((len(chosen_full),), -1, dtype=torch.long)
            c_start = max(0, len(prompt_ids) - 1)
            c_end = c_start + len(chosen_resp_ids)
            chosen_labels[c_start:c_end] = torch.tensor(chosen_resp_ids, dtype=torch.long)

            rejected_input = torch.tensor(rejected_full, dtype=torch.long)
            rejected_labels = torch.full((len(rejected_full),), -1, dtype=torch.long)
            r_start = max(0, len(prompt_ids) - 1)
            r_end = r_start + len(rejected_resp_ids)
            rejected_labels[r_start:r_end] = torch.tensor(rejected_resp_ids, dtype=torch.long)

            self.samples.append((chosen_input, chosen_labels, rejected_input, rejected_labels))

        n = len(self.samples)
        print(f"[DPODataset] Loaded {n} samples (skipped={skipped}, truncated={truncated})")

        # Save cache
        if cache_path is not None and n > 0:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": self.samples}, cache_path)
                size_mb = cache_path.stat().st_size / 1e6
                print(f"[DPODataset] Cache saved: {cache_path} ({size_mb:.0f} MB)")
            except Exception as e:
                print(f"[DPODataset] WARNING: Failed to save cache: {e}")

    @staticmethod
    def _cache_path(data_path: Path, max_seq_len: int) -> Path | None:
        if not data_path.is_file():
            return None
        stat = data_path.stat()
        key = f"dpo|{data_path.resolve()}|{stat.st_size}|{stat.st_mtime}|{max_seq_len}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return data_path.parent / f".dpo_cache_{data_path.stem}_{h}.pt"

    def _load_jsonl(self, path: Path) -> list[tuple[str, str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")
        if path.is_dir():
            jsonl_files = sorted(path.glob("*.jsonl"))
            if not jsonl_files:
                raise ValueError(f"No .jsonl files found in: {path}")
        else:
            jsonl_files = [path]

        triples: list[tuple[str, str, str]] = []
        for f in jsonl_files:
            triples.extend(self._parse_jsonl_file(f))
        return triples

    def _parse_jsonl_file(self, path: Path) -> list[tuple[str, str, str]]:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        n_lines = len(lines)
        n_workers = min(os.cpu_count() or 1, 32)

        if n_lines > 100_000 and n_workers > 1:
            print(f"[DPODataset] Parsing {n_lines:,} lines with {n_workers} workers ...")
            chunk_size = (n_lines + n_workers - 1) // n_workers
            chunks = []
            for i in range(0, n_lines, chunk_size):
                chunks.append((lines[i:i+chunk_size], str(path), i + 1))

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_parse_preference_lines, chunks))

            triples: list[tuple[str, str, str]] = []
            for chunk_result in results:
                triples.extend(chunk_result)
            return triples
        else:
            return _parse_preference_lines((lines, str(path), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def dpo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate DPO batch with dynamic padding.
    Pads chosen and rejected sequences to the same max length within the batch.
    """
    import torch.nn.functional as F

    # Find max length across ALL sequences in the batch (chosen + rejected)
    all_lens = []
    for c_ids, c_lab, r_ids, r_lab in batch:
        all_lens.append(c_ids.size(0))
        all_lens.append(r_ids.size(0))

    raw_max = max(all_lens)
    max_len = max(512, ((raw_max + 63) // 64) * 64)

    chosen_ids_list, chosen_labels_list = [], []
    rejected_ids_list, rejected_labels_list = [], []

    for c_ids, c_lab, r_ids, r_lab in batch:
        c_pad = max_len - c_ids.size(0)
        r_pad = max_len - r_ids.size(0)

        chosen_ids_list.append(F.pad(c_ids, (0, c_pad), value=0))
        chosen_labels_list.append(F.pad(c_lab, (0, c_pad), value=-1))
        rejected_ids_list.append(F.pad(r_ids, (0, r_pad), value=0))
        rejected_labels_list.append(F.pad(r_lab, (0, r_pad), value=-1))

    return (
        torch.stack(chosen_ids_list),
        torch.stack(chosen_labels_list),
        torch.stack(rejected_ids_list),
        torch.stack(rejected_labels_list),
    )

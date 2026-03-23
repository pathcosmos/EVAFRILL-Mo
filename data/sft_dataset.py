"""
SFT (Supervised Fine-Tuning) dataset for the Korean LLM project.

Reads JSONL files in three supported formats:

  1. Alpaca format
     {"instruction": "...", "input": "...", "output": "..."}

  2. Alpaca format without optional input
     {"instruction": "...", "output": "..."}

  3. Conversation format
     {"conversations": [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]}

Chat template applied:

    <|user|>
    {instruction or user turn}
    <|assistant|>
    {output or assistant turn}</s>

Loss masking: ``labels`` is -1 for all prompt tokens so
``nn.CrossEntropyLoss`` (ignore_index=-1) only trains on
the assistant responses.
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
from tokenizers import Tokenizer  # HuggingFace tokenizers (fast, Rust-based)


# ---------------------------------------------------------------------------
# Role tags used in the chat template.
# ---------------------------------------------------------------------------
_USER_TAG      = "<|user|>\n"
_ASSISTANT_TAG = "<|assistant|>\n"
_EOS_STRING    = "</s>"


def _build_alpaca_turns(
    instruction: str,
    input_text: str,
    output: str,
) -> tuple[str, str]:
    """
    Convert an Alpaca-format sample into (prompt, response) strings.

    The *prompt* includes the user tag and instruction (+ optional input).
    The *response* includes the assistant tag and output, plus EOS.

    Args:
        instruction: The task instruction.
        input_text:  Optional additional input context. May be empty.
        output:      The expected assistant response.

    Returns:
        Tuple of (prompt_text, response_text).
    """
    user_body = instruction
    if input_text and input_text.strip():
        user_body = f"{instruction}\n{input_text.strip()}"

    prompt   = f"{_USER_TAG}{user_body}\n{_ASSISTANT_TAG}"
    response = f"{output}{_EOS_STRING}"
    return prompt, response


def _build_conversation_turns(
    conversations: list[dict],
) -> list[tuple[str, str]]:
    """
    Convert a conversation list into a sequence of (prompt, response) pairs.

    For a multi-turn conversation the prompt for turn *k* is the entire
    dialogue history up to (but not including) assistant turn *k*.

    Only user→assistant pairs contribute training samples.  Consecutive
    user messages are merged.  Conversations that start with an assistant
    turn, or that have no assistant turn, are skipped (return empty list).

    Args:
        conversations: List of dicts with ``role`` and ``content`` keys.
                       Roles are expected to be ``"user"`` or ``"assistant"``.

    Returns:
        List of (prompt_text, response_text) tuples, one per assistant turn.
    """
    pairs: list[tuple[str, str]] = []
    history = ""          # accumulated dialogue so far
    pending_user = ""     # user content not yet closed by an assistant turn

    for turn in conversations:
        role    = turn.get("role", "").lower()
        content = turn.get("content", "")

        if role == "user":
            if pending_user:
                # Two consecutive user turns — concatenate them.
                pending_user = f"{pending_user}\n{content}"
            else:
                pending_user = content

        elif role == "assistant":
            if not pending_user:
                # Assistant turn without a preceding user turn — skip.
                continue
            prompt   = f"{history}{_USER_TAG}{pending_user}\n{_ASSISTANT_TAG}"
            response = f"{content}{_EOS_STRING}"
            pairs.append((prompt, response))
            # Update history to include this full exchange (without the EOS
            # so the model does not treat it as a hard stop mid-context).
            history = f"{history}{_USER_TAG}{pending_user}\n{_ASSISTANT_TAG}{content}\n"
            pending_user = ""

    return pairs


def _parse_lines_chunk(args: tuple) -> list[tuple[str, str]]:
    """Parse a chunk of JSONL lines into (prompt, response) pairs.

    Top-level function (not a method) so it can be pickled for multiprocessing.
    """
    lines, path_str, start_lineno = args
    pairs: list[tuple[str, str]] = []

    for i, line in enumerate(lines):
        lineno = start_lineno + i
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        conv_list = obj.get("conversations") or obj.get("messages")
        if conv_list and isinstance(conv_list, list):
            turn_pairs = _build_conversation_turns(conv_list)
            pairs.extend(turn_pairs)
        elif "instruction" in obj and "output" in obj:
            prompt, response = _build_alpaca_turns(
                instruction=obj["instruction"],
                input_text=obj.get("input", ""),
                output=obj["output"],
            )
            pairs.append((prompt, response))

    return pairs


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset built from JSONL files.

    Each JSONL line must conform to one of three schemas described in the
    module docstring.  After tokenisation the sample is laid out as::

        [prompt tokens ...] [response tokens ...] [pad tokens ...]
        |<---- labels=-1 ---->| |<-- labels=token_id -->| |<- labels=-1 ->|

    The ``labels`` tensor uses -1 as the ignore value so that
    ``nn.CrossEntropyLoss(ignore_index=-1)`` only penalises the model on
    the assistant response tokens.

    Args:
        data_path:    Path to a single ``.jsonl`` file or a directory that
                      contains multiple ``.jsonl`` files (all are loaded).
        tokenizer:    A ``tokenizers.Tokenizer`` instance (HuggingFace fast
                      tokenizer loaded from ``tokenizer.json``).
        max_seq_len:  Maximum sequence length (tokens).  Samples exceeding
                      this are truncated from the *end of the response*.
                      Default: 4096.
        pad_token_id: Token id used for right-padding.  Default: 0.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_seq_len: int = 4096,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len
        self.pad_token_id = pad_token_id

        # Resolve EOS token id from the vocabulary.
        eos_id = tokenizer.token_to_id(_EOS_STRING)
        if eos_id is None:
            raise ValueError(
                f"EOS token '{_EOS_STRING}' not found in the tokenizer vocabulary. "
                "Check that the tokenizer was trained with this special token."
            )
        self.eos_token_id: int = eos_id

        data_path = Path(data_path)

        # ------------------------------------------------------------------
        # Try loading from token cache (fast path: ~seconds instead of minutes)
        # ------------------------------------------------------------------
        cache_path = self._cache_path(data_path, max_seq_len)
        if cache_path is not None and cache_path.exists():
            import time as _time
            _t0 = _time.perf_counter()
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.samples = cached["samples"]
            _elapsed = _time.perf_counter() - _t0
            print(
                f"[SFTDataset] Loaded {len(self.samples)} samples from cache "
                f"in {_elapsed:.1f}s ({cache_path.name})"
            )
            return

        # ------------------------------------------------------------------
        # Load raw JSONL samples (parallel JSON parsing).
        # ------------------------------------------------------------------
        raw_samples = self._load_jsonl(data_path)

        # ------------------------------------------------------------------
        # Tokenise and build (input_ids, labels) pairs.
        # ------------------------------------------------------------------
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        total_loaded      = 0
        total_tokens      = 0
        skipped_too_long  = 0
        truncated_count   = 0

        for prompt_text, response_text in raw_samples:
            total_loaded += 1

            prompt_ids   = tokenizer.encode(prompt_text).ids    # list[int]
            response_ids = tokenizer.encode(response_text).ids  # list[int]

            # Skip samples where the prompt alone leaves no room for output.
            if len(prompt_ids) >= max_seq_len - 10:
                skipped_too_long += 1
                continue

            full_ids = prompt_ids + response_ids

            # Truncate response if the combined sequence is too long.
            if len(full_ids) > max_seq_len:
                truncated_count += 1
                # Keep as many response tokens as possible after the prompt.
                allowed_response = max_seq_len - len(prompt_ids)
                if allowed_response <= 0:
                    # Prompt itself is too close to the limit (edge case).
                    skipped_too_long += 1
                    continue
                response_ids = response_ids[:allowed_response]
                # [BUG FIX] Force EOS at end after truncation so the model
                # always sees a proper sequence terminator.
                if response_ids[-1] != self.eos_token_id:
                    response_ids[-1] = self.eos_token_id
                full_ids = prompt_ids + response_ids

            seq_len = len(full_ids)
            total_tokens += seq_len

            # [BUG FIX] Do NOT pre-pad to max_seq_len here.
            # Store raw-length tensors so that dynamic_collate_fn can
            # batch-pad to the actual max length in each mini-batch,
            # saving significant compute on shorter sequences.
            input_ids = torch.tensor(full_ids, dtype=torch.long)

            # Build labels tensor (-1 everywhere, then fill response region).
            labels = torch.full((seq_len,), fill_value=-1, dtype=torch.long)
            resp_start = len(prompt_ids)
            resp_label_start = max(0, resp_start - 1)
            resp_label_end   = resp_label_start + len(response_ids)
            labels[resp_label_start:resp_label_end] = torch.tensor(
                response_ids, dtype=torch.long
            )

            self.samples.append((input_ids, labels))

        # ------------------------------------------------------------------
        # Print statistics.
        # ------------------------------------------------------------------
        n = len(self.samples)
        avg_len = (total_tokens / n) if n > 0 else 0.0

        print(
            f"[SFTDataset] Loaded {n} samples "
            f"(raw={total_loaded}, "
            f"skipped_too_long={skipped_too_long}, "
            f"truncated={truncated_count})"
        )
        print(
            f"[SFTDataset] avg_seq_len={avg_len:.1f}, "
            f"max_seq_len={max_seq_len}, "
            f"pad_token_id={pad_token_id}, "
            f"eos_token_id={self.eos_token_id}"
        )

        # ------------------------------------------------------------------
        # Save token cache for fast loading next time.
        # ------------------------------------------------------------------
        if cache_path is not None and n > 0:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": self.samples}, cache_path)
                size_mb = cache_path.stat().st_size / 1e6
                print(f"[SFTDataset] Token cache saved: {cache_path} ({size_mb:.0f} MB)")
            except Exception as e:
                print(f"[SFTDataset] WARNING: Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_path(data_path: Path, max_seq_len: int) -> Path | None:
        """Compute a deterministic cache path based on data file and params."""
        if not data_path.is_file():
            return None
        # Hash based on file path, size, mtime, and max_seq_len
        stat = data_path.stat()
        key = f"{data_path.resolve()}|{stat.st_size}|{stat.st_mtime}|{max_seq_len}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return data_path.parent / f".sft_cache_{data_path.stem}_{h}.pt"

    def _load_jsonl(self, path: Path) -> list[tuple[str, str]]:
        """
        Discover and parse JSONL files, returning (prompt, response) pairs.

        If ``path`` is a file, load that file only.  If it is a directory,
        load all ``*.jsonl`` files found directly inside it (non-recursive).

        Args:
            path: File or directory path.

        Returns:
            List of (prompt_text, response_text) tuples.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError:        If no ``.jsonl`` files are found under a
                               directory path.
        """
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        if path.is_dir():
            jsonl_files = sorted(path.glob("*.jsonl"))
            if not jsonl_files:
                raise ValueError(f"No .jsonl files found in directory: {path}")
        else:
            jsonl_files = [path]

        pairs: list[tuple[str, str]] = []
        for jsonl_file in jsonl_files:
            pairs.extend(self._parse_jsonl_file(jsonl_file))

        return pairs

    def _parse_jsonl_file(self, path: Path) -> list[tuple[str, str]]:
        """
        Parse a single JSONL file into (prompt, response) pairs.

        For large files (>100K lines), uses multiprocessing to parse in parallel.
        """
        # Read all lines first (I/O is sequential anyway)
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        n_lines = len(lines)
        n_workers = min(os.cpu_count() or 1, 32)

        if n_lines > 100_000 and n_workers > 1:
            # Parallel parsing for large files
            print(f"[SFTDataset] Parsing {n_lines:,} lines with {n_workers} workers ...")
            chunk_size = (n_lines + n_workers - 1) // n_workers
            chunks = []
            for i in range(0, n_lines, chunk_size):
                chunk_lines = lines[i : i + chunk_size]
                start_lineno = i + 1
                chunks.append((chunk_lines, str(path), start_lineno))

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_parse_lines_chunk, chunks))

            pairs: list[tuple[str, str]] = []
            for chunk_pairs in results:
                pairs.extend(chunk_pairs)
            return pairs
        else:
            # Small file: parse sequentially
            return _parse_lines_chunk((lines, str(path), 1))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(input_ids, labels)`` where both tensors have shape
            ``[seq_len]`` (variable per sample) and dtype ``torch.long``.
            Use a collate function to pad batches dynamically.

            - ``input_ids``: Full token sequence (prompt + response),
              NO padding (raw length).
            - ``labels``:    Response token ids at response positions,
              ``-1`` everywhere else (prompt tokens).
              Use ``ignore_index=-1`` in your loss function.
        """
        return self.samples[idx]

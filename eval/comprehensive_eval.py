"""
Comprehensive evaluation script for a trained 1B Korean language model.

Covers:
  1. Multi-source sliding-window perplexity (4 val sets)
  2. Token-level NLL distribution + top-50 highest/lowest-loss tokens
  3. Multi-prompt generation quality (10 diverse prompts)
  4. Repetition analysis (unigram..4-gram repetition ratio)
  5. Greedy vs. sampling comparison (3 prompts × 4 temperature settings)
  6. Calibration check (accuracy@1/5/10, mean prob, mean entropy)

Usage:
    python eval/comprehensive_eval.py \
        --checkpoint checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project root on sys.path (allow running from any cwd)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.transformer import LLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for a trained Korean LLM."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/korean_1b_fp8_run1/checkpoint-0034000",
        help="Path to the checkpoint directory (default: korean_1b_fp8_run1/checkpoint-0034000).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json. Defaults to <checkpoint>/tokenizer.json, "
             "then tokenizer/korean_sp/tokenizer.json.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing val .bin files. Defaults to <project>/data/.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sliding-window sequence length for PPL (default: 2048).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding-window PPL (default: 512).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for PPL evaluation (default: 4).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Max new tokens for generation (default: 200).",
    )
    parser.add_argument(
        "--calib_tokens",
        type=int,
        default=10000,
        help="Number of tokens used for calibration check (default: 10000).",
    )
    return parser.parse_args()


# ===========================================================================
# Model + tokenizer loading
# ===========================================================================

def load_model(checkpoint_dir: str, device: str) -> LLM:
    """Load LLM from checkpoint directory in BF16."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"  Loading model weights from: {ckpt_path}")
    model = LLM.from_pretrained(str(ckpt_path))
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params / 1e6:.1f}M  |  dtype: {next(model.parameters()).dtype}")
    return model


def load_tokenizer(checkpoint_dir: str, tokenizer_override: Optional[str]) -> Tokenizer:
    """Resolve and load tokenizer."""
    ckpt_path = Path(checkpoint_dir)
    candidates = []
    if tokenizer_override:
        candidates.append(Path(tokenizer_override))
    candidates += [
        ckpt_path / "tokenizer.json",
        _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Loading tokenizer from: {p}")
            return Tokenizer.from_file(str(p))
    raise FileNotFoundError(
        f"tokenizer.json not found. Tried: {[str(c) for c in candidates]}"
    )


# ===========================================================================
# Sliding-window Dataset (reused from perplexity.py logic)
# ===========================================================================

class SlidingWindowDataset(Dataset):
    """Sliding-window dataset yielding (input_ids, targets, loss_mask)."""

    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int) -> None:
        self.tokens  = tokens
        self.seq_len = seq_len
        self.stride  = stride
        self.n_windows = max(0, (len(tokens) - seq_len + stride - 1) // stride)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start      = idx * self.stride
        end        = start + self.seq_len
        actual_end = min(end, len(self.tokens))
        chunk_len  = actual_end - start

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets   = torch.full((self.seq_len,), fill_value=-100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        if chunk_len > 1:
            toks = torch.from_numpy(self.tokens[start:actual_end].astype(np.int64))
            input_ids[:chunk_len]     = toks
            targets[:chunk_len - 1]   = toks[1:]

        new_start = 0 if idx == 0 else self.stride
        if chunk_len > 1:
            for pos in range(new_start, chunk_len - 1):
                loss_mask[pos] = True

        return input_ids, targets, loss_mask


# ===========================================================================
# Sampling utilities (mirrors eval/generate.py)
# ===========================================================================

def top_p_filtering(
    logits: torch.Tensor,
    top_p: float = 0.9,
    top_k: int = 0,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_values = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_values, filter_value)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = (
            cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        )
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    if squeeze_output:
        logits = logits.squeeze(0)
    return logits


@torch.inference_mode()
def generate_text(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cuda:0",
) -> str:
    """Generate text and return the full string (prompt + generated)."""
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids], dtype=torch.long, device=device
    )
    eos_token_id: Optional[int] = tokenizer.token_to_id("</s>")
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        logits_all, _ = model(generated_ids)
        logits: torch.Tensor = logits_all[:, -1, :]  # [1, vocab]

        if temperature == 0.0:
            # Greedy decoding
            next_token_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs  = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    # Decode only the newly generated portion
    all_ids   = generated_ids[0].tolist()
    new_ids   = all_ids[len(tokenizer.encode(prompt).ids):]
    generated = tokenizer.decode(new_ids)
    return generated


# ===========================================================================
# Section 1 — Multi-source Perplexity
# ===========================================================================

@torch.inference_mode()
def eval_perplexity_on_file(
    model: LLM,
    data_path: Path,
    seq_len: int,
    stride: int,
    batch_size: int,
    device: str,
) -> Tuple[float, float, int]:
    """
    Sliding-window PPL on one .bin file.

    Returns:
        (perplexity, bits_per_token, n_tokens_evaluated)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    tokens = np.memmap(str(data_path), dtype="uint16", mode="r")
    n_total = len(tokens)
    # Cap at 2M tokens to keep eval time reasonable
    MAX_EVAL_TOKENS = 2_000_000
    if n_total > MAX_EVAL_TOKENS:
        tokens = tokens[:MAX_EVAL_TOKENS]
    print(f"    {data_path.name}: {n_total:,} tokens (using {len(tokens):,})")

    dataset = SlidingWindowDataset(tokens, seq_len=seq_len, stride=stride)
    if len(dataset) == 0:
        raise ValueError(f"No windows fit: {n_total} tokens, seq_len={seq_len}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    total_nll   = 0.0
    total_count = 0

    for batch_input_ids, batch_targets, batch_loss_mask in loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_targets   = batch_targets.to(device)
        batch_loss_mask = batch_loss_mask.to(device)

        logits, _ = model(batch_input_ids)  # [B, S, V]
        B, S, V = logits.shape

        ce = F.cross_entropy(
            logits.reshape(B * S, V),
            batch_targets.reshape(B * S),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, S)

        masked_ce    = ce * batch_loss_mask.float()
        total_nll   += masked_ce.sum().item()
        total_count += batch_loss_mask.sum().item()

    if total_count == 0:
        raise RuntimeError("No valid positions evaluated.")

    avg_nll    = total_nll / total_count
    ppl        = math.exp(avg_nll)
    bpt        = avg_nll / math.log(2)
    return ppl, bpt, total_count


def section_perplexity(
    model: LLM,
    data_dir: Path,
    seq_len: int,
    stride: int,
    batch_size: int,
    device: str,
) -> Dict[str, Tuple[float, float, int]]:
    """Run PPL on all 4 val sets. Returns {name: (ppl, bpt, n_tokens)}."""
    print_header("1. MULTI-SOURCE PERPLEXITY")
    val_files = [
        "3b_val.bin",
        "korean_wiki_val.bin",
        "korean_c4_val.bin",
        "korean_namuwiki_val.bin",
    ]
    results: Dict[str, Tuple[float, float, int]] = {}
    for fname in val_files:
        path = data_dir / fname
        name = fname.replace(".bin", "")
        print(f"  Evaluating {fname} ...")
        try:
            ppl, bpt, n_tok = eval_perplexity_on_file(
                model, path, seq_len, stride, batch_size, device
            )
            results[name] = (ppl, bpt, n_tok)
            print(f"    PPL = {ppl:.4f}  |  bits/token = {bpt:.4f}  |  tokens = {n_tok:,}")
        except Exception as exc:
            print(f"    [SKIPPED] {exc}")
            results[name] = (float("nan"), float("nan"), 0)

    print()
    print(f"  {'Dataset':<30} {'PPL':>10} {'bits/tok':>10} {'tokens':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")
    for name, (ppl, bpt, n_tok) in results.items():
        ppl_s = f"{ppl:.4f}" if math.isfinite(ppl) else "N/A"
        bpt_s = f"{bpt:.4f}" if math.isfinite(bpt) else "N/A"
        n_s   = f"{n_tok:,}" if n_tok else "N/A"
        print(f"  {name:<30} {ppl_s:>10} {bpt_s:>10} {n_s:>12}")
    return results


# ===========================================================================
# Section 2 — Token-level NLL Analysis
# ===========================================================================

@torch.inference_mode()
def section_token_analysis(
    model: LLM,
    tokenizer: Tokenizer,
    data_dir: Path,
    seq_len: int,
    batch_size: int,
    device: str,
    max_batches: int = 50,
) -> None:
    """Compute per-token NLL distribution and identify hardest/easiest tokens."""
    print_header("2. TOKEN-LEVEL NLL ANALYSIS")

    val_path = data_dir / "3b_val.bin"
    if not val_path.exists():
        print("  [SKIPPED] 3b_val.bin not found.")
        return

    tokens   = np.memmap(str(val_path), dtype="uint16", mode="r")
    dataset  = SlidingWindowDataset(tokens, seq_len=seq_len, stride=seq_len)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Accumulate per-token-id NLL sums and counts
    vocab_size = model.config.vocab_size
    token_nll_sum   = torch.zeros(vocab_size, dtype=torch.float64)
    token_nll_count = torch.zeros(vocab_size, dtype=torch.long)

    # Also store all NLL values for histogram
    all_nll_values: List[float] = []

    n_batches = 0
    for batch_input_ids, batch_targets, batch_loss_mask in loader:
        if n_batches >= max_batches:
            break

        batch_input_ids = batch_input_ids.to(device)
        batch_targets_dev = batch_targets.to(device)
        batch_loss_mask_dev = batch_loss_mask.to(device)

        logits, _ = model(batch_input_ids)  # [B, S, V]
        B, S, V = logits.shape

        # Per-position NLL (no reduction)
        nll = F.cross_entropy(
            logits.reshape(B * S, V),
            batch_targets_dev.reshape(B * S),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, S)  # [B, S]

        # Apply sliding-window mask (both tensors on GPU)
        mask = batch_loss_mask_dev & (batch_targets_dev != -100)
        valid_nll = nll[mask].float()
        valid_tok = batch_targets_dev[mask].long()  # use GPU targets for indexing

        # Histogram accumulation
        all_nll_values.extend(valid_nll.cpu().tolist())

        # Per-token accumulation (CPU scatter)
        for tok_id, nll_val in zip(valid_tok.tolist(), valid_nll.cpu().tolist()):
            if 0 <= tok_id < vocab_size:
                token_nll_sum[tok_id]   += nll_val
                token_nll_count[tok_id] += 1

        n_batches += 1

    if not all_nll_values:
        print("  [SKIPPED] No valid NLL values collected.")
        return

    all_nll = torch.tensor(all_nll_values, dtype=torch.float32)

    # --- NLL histogram ---
    bins   = [0, 1, 2, 3, 5, 10, float("inf")]
    labels = ["<1", "1-2", "2-3", "3-5", "5-10", ">10"]
    total  = len(all_nll)
    print(f"  Total token positions analysed: {total:,}")
    print()
    print(f"  {'NLL range':<10} {'count':>10} {'percentage':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*12}")
    for i, label in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1]
        if hi == float("inf"):
            cnt = int((all_nll >= lo).sum().item())
        else:
            cnt = int(((all_nll >= lo) & (all_nll < hi)).sum().item())
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  {label:<10} {cnt:>10,} {pct:>11.2f}%")

    print()
    print(f"  Mean NLL: {all_nll.mean().item():.4f}   Std: {all_nll.std().item():.4f}")
    print(f"  Median NLL: {all_nll.median().item():.4f}")

    # --- Top-50 highest-loss tokens ---
    has_data = token_nll_count > 0
    avg_nll_per_token = torch.where(
        has_data,
        token_nll_sum / token_nll_count.clamp(min=1).float(),
        torch.full_like(token_nll_sum, float("nan")),
    )

    # Mask NaN positions
    valid_mask = ~torch.isnan(avg_nll_per_token)
    valid_ids  = valid_mask.nonzero(as_tuple=True)[0]
    valid_avgs = avg_nll_per_token[valid_ids]

    if len(valid_ids) == 0:
        print("  [WARNING] No per-token averages computed.")
        return

    # Sort descending (highest NLL = hardest)
    sorted_idx   = valid_avgs.argsort(descending=True)
    top50_hard   = valid_ids[sorted_idx[:50]]
    top50_easy   = valid_ids[sorted_idx[-50:].flip(0)]

    def decode_token(tid: int) -> str:
        try:
            return repr(tokenizer.decode([tid]))
        except Exception:
            return f"<id={tid}>"

    print()
    print("  Top-50 HIGHEST-loss tokens (model struggles with):")
    print(f"  {'rank':<5} {'token_id':<10} {'avg_nll':>8} {'count':>8} {'decoded'}")
    print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*30}")
    for rank, tid in enumerate(top50_hard[:50].tolist(), start=1):
        avg  = avg_nll_per_token[tid].item()
        cnt  = token_nll_count[tid].item()
        text = decode_token(tid)
        print(f"  {rank:<5} {tid:<10} {avg:>8.3f} {cnt:>8,} {text}")

    print()
    print("  Top-50 LOWEST-loss tokens (model handles well):")
    print(f"  {'rank':<5} {'token_id':<10} {'avg_nll':>8} {'count':>8} {'decoded'}")
    print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*30}")
    for rank, tid in enumerate(top50_easy[:50].tolist(), start=1):
        avg  = avg_nll_per_token[tid].item()
        cnt  = token_nll_count[tid].item()
        text = decode_token(tid)
        print(f"  {rank:<5} {tid:<10} {avg:>8.3f} {cnt:>8,} {text}")


# ===========================================================================
# Section 3 — Multi-prompt Generation
# ===========================================================================

GENERATION_PROMPTS = [
    "한국의 수도는",
    "인공지능이란",
    "오늘 날씨가 좋아서",
    "대한민국의 역사에서 가장 중요한 사건은",
    "서울에서 부산까지 가는 방법은",
    "다음은 파이썬 코드입니다:\ndef hello():",
    "1 + 1 = 2이고, 2 + 2 =",
    "봄이 오면 꽃이 피고",
    "맛있는 김치찌개를 만들려면",
    "세종대왕은",
]


def compute_ngram_repetition(text: str, n: int) -> float:
    """Compute n-gram repetition ratio = 1 - unique_ngrams / total_ngrams.

    Returns a value in [0, 1] where 0 = no repetition, 1 = all repeated.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    total  = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


def section_generation(
    model: LLM,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    device: str,
) -> Dict[str, str]:
    """Generate text for each prompt and return {prompt: generated}."""
    print_header("3. MULTI-PROMPT GENERATION")
    generated: Dict[str, str] = {}

    for i, prompt in enumerate(GENERATION_PROMPTS, start=1):
        print(f"\n  [{i:02d}/{len(GENERATION_PROMPTS)}] Prompt: {prompt!r}")
        print("  " + "-" * 70)
        try:
            t0   = time.time()
            text = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                device=device,
            )
            elapsed = time.time() - t0
            generated[prompt] = text
            # Print generated text with wrapping at 80 chars
            full_output = prompt + text
            print(f"  {full_output}")
            print(f"\n  [generated {len(text.split()):,} words in {elapsed:.1f}s]")
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            generated[prompt] = ""

    return generated


# ===========================================================================
# Section 4 — Repetition Analysis
# ===========================================================================

REPETITION_THRESHOLD = 0.30  # 30% trigram repetition = degenerate


def section_repetition(generated: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Analyse n-gram repetition for each generated text."""
    print_header("4. REPETITION ANALYSIS")

    ns = [1, 2, 3, 4]
    header = f"  {'Prompt (truncated)':<35}"
    for n in ns:
        header += f" {'%rep-{n}gram':>12}"
    header += f"  {'FLAG':>6}"
    print(header)
    print("  " + "-" * (35 + 12 * len(ns) + 10))

    results: Dict[str, Dict[str, float]] = {}
    for prompt, text in generated.items():
        if not text.strip():
            continue
        row_results: Dict[str, float] = {}
        for n in ns:
            ratio = compute_ngram_repetition(text, n)
            row_results[f"{n}gram"] = ratio
        results[prompt] = row_results

        prompt_short = (prompt[:32] + "..") if len(prompt) > 34 else prompt
        row = f"  {prompt_short:<35}"
        for n in ns:
            pct = row_results[f"{n}gram"] * 100
            row += f" {pct:>11.1f}%"
        flag = "[DEGENERATE]" if row_results.get("3gram", 0.0) > REPETITION_THRESHOLD else ""
        row += f"  {flag}"
        print(row)

    # Summary
    degenerate = [
        p for p, r in results.items()
        if r.get("3gram", 0.0) > REPETITION_THRESHOLD
    ]
    print()
    if degenerate:
        print(f"  WARNING: {len(degenerate)} generation(s) exceed {REPETITION_THRESHOLD*100:.0f}% trigram repetition:")
        for p in degenerate:
            print(f"    - {p!r}")
    else:
        print(f"  All generations are below the {REPETITION_THRESHOLD*100:.0f}% trigram repetition threshold.")

    return results


# ===========================================================================
# Section 5 — Greedy vs. Sampling Comparison
# ===========================================================================

COMPARISON_PROMPTS = [
    "한국의 수도는",
    "인공지능이란",
    "봄이 오면 꽃이 피고",
]

TEMPERATURE_CONFIGS = [
    ("Greedy (T=0.0)", 0.0,  1,  0.0),
    ("Low    (T=0.3)", 0.3, 50,  0.9),
    ("Normal (T=0.8)", 0.8, 50,  0.9),
    ("High   (T=1.2)", 1.2, 50,  0.9),
]


def section_comparison(
    model: LLM,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    device: str,
) -> None:
    """Generate each comparison prompt at 4 temperature settings."""
    print_header("5. GREEDY vs. SAMPLING COMPARISON")

    for prompt in COMPARISON_PROMPTS:
        print(f"\n  Prompt: {prompt!r}")
        print("  " + "=" * 74)
        for label, temp, top_k, top_p in TEMPERATURE_CONFIGS:
            try:
                text = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=min(max_new_tokens, 100),
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    device=device,
                )
                print(f"\n  [{label}]")
                print(f"  {prompt + text}")
            except Exception as exc:
                print(f"\n  [{label}] FAILED: {exc}")
        print()


# ===========================================================================
# Section 6 — Calibration Check
# ===========================================================================

@torch.inference_mode()
def section_calibration(
    model: LLM,
    data_dir: Path,
    device: str,
    calib_tokens: int = 10000,
    seq_len: int = 512,
) -> Dict[str, float]:
    """
    Calibration check on first `calib_tokens` tokens of korean_val.bin.

    Computes:
      - mean predicted probability of correct token
      - mean entropy of predicted distributions
      - accuracy@1, @5, @10
    """
    print_header("6. CALIBRATION CHECK")

    val_path = data_dir / "3b_val.bin"
    if not val_path.exists():
        print("  [SKIPPED] 3b_val.bin not found.")
        return {}

    tokens_all = np.memmap(str(val_path), dtype="uint16", mode="r")
    n_use      = min(calib_tokens + seq_len, len(tokens_all))
    tokens     = tokens_all[:n_use]
    print(f"  Using first {n_use:,} tokens for calibration.")

    # Process in non-overlapping chunks of seq_len
    mean_correct_prob  = 0.0
    mean_entropy       = 0.0
    acc1 = acc5 = acc10 = 0
    n_positions        = 0

    n_chunks = (n_use - 1) // seq_len
    if n_chunks == 0:
        print("  [SKIPPED] Not enough tokens for calibration.")
        return {}

    for chunk_idx in range(n_chunks):
        start     = chunk_idx * seq_len
        end       = start + seq_len + 1
        if end > len(tokens):
            break

        chunk     = torch.from_numpy(tokens[start:end].astype(np.int64))
        input_ids = chunk[:-1].unsqueeze(0).to(device)   # [1, seq_len]
        target    = chunk[1:].to(device)                  # [seq_len]

        logits, _ = model(input_ids)                       # [1, seq_len, V]
        logits_2d = logits[0]                              # [seq_len, V]

        # Probabilities (fp32 for numerical stability)
        probs = F.softmax(logits_2d.float(), dim=-1)       # [seq_len, V]

        # Mean correct-token probability
        correct_probs = probs[torch.arange(seq_len, device=device), target]
        mean_correct_prob += correct_probs.sum().item()

        # Mean entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy   = -(probs * log_probs).sum(dim=-1)       # [seq_len]
        mean_entropy += entropy.sum().item()

        # Accuracy @k: check if correct token is in top-k
        top10 = logits_2d.topk(10, dim=-1).indices         # [seq_len, 10]
        target_col = target.unsqueeze(1)                    # [seq_len, 1]
        in_top10   = (top10 == target_col)                  # [seq_len, 10]
        acc1  += in_top10[:, :1].any(dim=1).sum().item()
        acc5  += in_top10[:, :5].any(dim=1).sum().item()
        acc10 += in_top10[:, :10].any(dim=1).sum().item()
        n_positions += seq_len

    if n_positions == 0:
        print("  [SKIPPED] No positions evaluated.")
        return {}

    metrics = {
        "mean_correct_prob": mean_correct_prob / n_positions,
        "mean_entropy_nats": mean_entropy / n_positions,
        "accuracy_at_1":     acc1  / n_positions,
        "accuracy_at_5":     acc5  / n_positions,
        "accuracy_at_10":    acc10 / n_positions,
    }

    print(f"  Positions evaluated:       {n_positions:,}")
    print(f"  Mean correct-token prob:   {metrics['mean_correct_prob']:.4f}")
    print(f"  Mean predicted entropy:    {metrics['mean_entropy_nats']:.4f} nats")
    print(f"  Accuracy @1:               {metrics['accuracy_at_1']*100:.2f}%")
    print(f"  Accuracy @5:               {metrics['accuracy_at_5']*100:.2f}%")
    print(f"  Accuracy @10:              {metrics['accuracy_at_10']*100:.2f}%")
    return metrics


# ===========================================================================
# Summary Table
# ===========================================================================

def print_summary(
    ppl_results: Dict[str, Tuple[float, float, int]],
    rep_results: Dict[str, Dict[str, float]],
    calib_results: Dict[str, float],
) -> None:
    print_header("SUMMARY TABLE")

    # Perplexity
    print("  [Perplexity]")
    print(f"  {'Dataset':<30} {'PPL':>10} {'bits/tok':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for name, (ppl, bpt, _) in ppl_results.items():
        ppl_s = f"{ppl:.4f}" if math.isfinite(ppl) else "N/A"
        bpt_s = f"{bpt:.4f}" if math.isfinite(bpt) else "N/A"
        print(f"  {name:<30} {ppl_s:>10} {bpt_s:>10}")

    # Repetition summary
    if rep_results:
        mean_tri = np.mean([r.get("3gram", 0.0) for r in rep_results.values()])
        degenerate_count = sum(
            1 for r in rep_results.values() if r.get("3gram", 0.0) > REPETITION_THRESHOLD
        )
        print()
        print("  [Repetition (avg over all prompts)]")
        for n in [1, 2, 3, 4]:
            vals = [r.get(f"{n}gram", 0.0) for r in rep_results.values()]
            if vals:
                print(f"  {n}-gram avg rep ratio:  {np.mean(vals)*100:.1f}%")
        print(f"  Degenerate outputs (>30% trigram): {degenerate_count}/{len(rep_results)}")

    # Calibration
    if calib_results:
        print()
        print("  [Calibration]")
        for key, val in calib_results.items():
            if "accuracy" in key:
                print(f"  {key:<30} {val*100:.2f}%")
            else:
                print(f"  {key:<30} {val:.4f}")

    print()
    print("  " + "=" * 60)
    print("  Evaluation complete.")
    print("  " + "=" * 60)


# ===========================================================================
# Formatting helpers
# ===========================================================================

def print_header(title: str) -> None:
    bar = "=" * 72
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root if not absolute
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path

    data_dir = Path(args.data_dir) if args.data_dir else _PROJECT_ROOT / "data"

    print_header("COMPREHENSIVE EVAL — Korean 1B LLM")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {args.device}")
    print(f"  Data dir   : {data_dir}")
    print(f"  seq_len    : {args.seq_len}  stride={args.stride}  batch={args.batch_size}")

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    print_header("LOADING MODEL & TOKENIZER")
    try:
        model = load_model(str(ckpt_path), args.device)
    except Exception as exc:
        print(f"  [FATAL] Could not load model: {exc}")
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(str(ckpt_path), args.tokenizer)
    except Exception as exc:
        print(f"  [FATAL] Could not load tokenizer: {exc}")
        sys.exit(1)

    # Collect results across sections for the summary table
    ppl_results:   Dict[str, Tuple[float, float, int]] = {}
    rep_results:   Dict[str, Dict[str, float]]         = {}
    calib_results: Dict[str, float]                    = {}

    # ------------------------------------------------------------------
    # Section 1 — Perplexity
    # ------------------------------------------------------------------
    try:
        ppl_results = section_perplexity(
            model, data_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print(f"  [SECTION 1 FAILED] {exc}")

    # ------------------------------------------------------------------
    # Section 2 — Token-level Analysis
    # ------------------------------------------------------------------
    try:
        section_token_analysis(
            model, tokenizer, data_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print(f"  [SECTION 2 FAILED] {exc}")

    # ------------------------------------------------------------------
    # Section 3 — Multi-prompt Generation
    # ------------------------------------------------------------------
    generated: Dict[str, str] = {}
    try:
        generated = section_generation(
            model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    except Exception as exc:
        print(f"  [SECTION 3 FAILED] {exc}")

    # ------------------------------------------------------------------
    # Section 4 — Repetition Analysis
    # ------------------------------------------------------------------
    if generated:
        try:
            rep_results = section_repetition(generated)
        except Exception as exc:
            print(f"  [SECTION 4 FAILED] {exc}")
    else:
        print_header("4. REPETITION ANALYSIS")
        print("  [SKIPPED] No generated texts available.")

    # ------------------------------------------------------------------
    # Section 5 — Greedy vs. Sampling Comparison
    # ------------------------------------------------------------------
    try:
        section_comparison(
            model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    except Exception as exc:
        print(f"  [SECTION 5 FAILED] {exc}")

    # ------------------------------------------------------------------
    # Section 6 — Calibration Check
    # ------------------------------------------------------------------
    try:
        calib_results = section_calibration(
            model, data_dir,
            device=args.device,
            calib_tokens=args.calib_tokens,
            seq_len=min(args.seq_len, 512),  # smaller chunks for calib
        )
    except Exception as exc:
        print(f"  [SECTION 6 FAILED] {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    try:
        print_summary(ppl_results, rep_results, calib_results)
    except Exception as exc:
        print(f"  [SUMMARY FAILED] {exc}")


if __name__ == "__main__":
    main()

"""
Prepare Korean instruction-following data for Supervised Fine-Tuning (SFT).

Downloads Korean SFT datasets from HuggingFace, normalises them to a common
JSONL format, applies quality filters, deduplicates, and splits into
train / validation sets.

Output format (one JSON object per line):
    {"instruction": "...", "input": "...", "output": "..."}

Usage:
    python data/prepare_sft_data.py
    python data/prepare_sft_data.py --output_dir data/sft/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Sample = Dict[str, str]  # {"instruction": str, "input": str, "output": str}


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _normalize_sample(
    instruction: str,
    input_text: str,
    output: str,
) -> Optional[Sample]:
    """
    Return a normalised sample dict, or None if any required field is missing.

    All fields are stripped of leading/trailing whitespace.  ``input`` is
    allowed to be empty (many alpaca-style datasets leave it blank).
    """
    instruction = (instruction or "").strip()
    input_text  = (input_text  or "").strip()
    output      = (output      or "").strip()

    if not instruction or not output:
        return None

    return {"instruction": instruction, "input": input_text, "output": output}


def load_kor_openorca_platypus(dataset_name: str) -> List[Sample]:
    """
    kyujinpy/KOR-OpenOrca-Platypus-v3

    Expected columns: instruction, input, output
    Falls back to system_prompt/question/response if needed.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    cols = set(ds.column_names)
    samples: List[Sample] = []

    for row in ds:
        # Primary column mapping
        if "instruction" in cols and "output" in cols:
            instruction = row.get("instruction", "") or ""
            input_text  = row.get("input", "") or ""
            output      = row.get("output", "") or ""
        # Fallback: question / response style
        elif "question" in cols and "response" in cols:
            instruction = row.get("question", "") or ""
            input_text  = ""
            output      = row.get("response", "") or ""
        # Fallback: conversations list
        elif "conversations" in cols:
            sample = _extract_from_conversations(row.get("conversations", []))
            if sample is None:
                continue
            instruction, input_text, output = sample
        else:
            # Last resort: dump all string fields and skip
            continue

        norm = _normalize_sample(instruction, input_text, output)
        if norm is not None:
            samples.append(norm)

    return samples


def load_kullm_v2(dataset_name: str) -> List[Sample]:
    """
    nlpai-lab/kullm-v2

    The KULLM-v2 dataset typically uses:
        - ``instruction``  (한국어 지시문)
        - ``input``        (추가 컨텍스트, optional)
        - ``output``       (응답)

    Some variants use ``context`` instead of ``input``, or nest content under
    ``text`` as a formatted prompt.  We inspect at runtime and adapt.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    cols = set(ds.column_names)
    samples: List[Sample] = []

    for row in ds:
        if "instruction" in cols and "output" in cols:
            instruction = row.get("instruction", "") or ""
            # Some KULLM records use "context" as the secondary input field.
            input_text  = (row.get("input", "") or row.get("context", "")) or ""
            output      = row.get("output", "") or ""

        elif "text" in cols:
            # Alpaca-formatted single-string: parse out the fields.
            parsed = _parse_alpaca_text(row.get("text", "") or "")
            if parsed is None:
                continue
            instruction, input_text, output = parsed

        elif "conversations" in cols:
            result = _extract_from_conversations(row.get("conversations", []))
            if result is None:
                continue
            instruction, input_text, output = result

        else:
            continue

        norm = _normalize_sample(instruction, input_text, output)
        if norm is not None:
            samples.append(norm)

    return samples


def load_ko_alpaca(dataset_name: str) -> List[Sample]:
    """
    junhochoi/ko-alpaca-12k

    Standard Alpaca format: instruction, input, output
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    cols = set(ds.column_names)
    samples: List[Sample] = []

    for row in ds:
        if "instruction" in cols and "output" in cols:
            instruction = row.get("instruction", "") or ""
            input_text  = row.get("input", "") or ""
            output      = row.get("output", "") or ""

        elif "conversations" in cols:
            result = _extract_from_conversations(row.get("conversations", []))
            if result is None:
                continue
            instruction, input_text, output = result

        else:
            continue

        norm = _normalize_sample(instruction, input_text, output)
        if norm is not None:
            samples.append(norm)

    return samples


def load_korean_safe_conversation(dataset_name: str) -> List[Sample]:
    """jojo0217/korean_safe_conversation — 안전 정렬 한국어 대화"""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", token=os.environ.get("HF_TOKEN"))
    samples: List[Sample] = []
    for item in ds:
        s = _normalize_sample(
            instruction=item.get("instruction", ""),
            input_text=item.get("input", ""),
            output=item.get("output", ""),
        )
        if s:
            samples.append(s)
    return samples


def load_evol_instruct_korean(dataset_name: str) -> List[Sample]:
    """FreedomIntelligence/Evol-Instruct-Korean — 복잡한 추론/코드"""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", token=os.environ.get("HF_TOKEN"))
    samples: List[Sample] = []
    for item in ds:
        conversations = item.get("conversations", [])
        if len(conversations) >= 2:
            instruction = conversations[0].get("value", "")
            output = conversations[1].get("value", "")
            s = _normalize_sample(instruction=instruction, input_text="", output=output)
            if s:
                samples.append(s)
    return samples


def load_kovast(dataset_name: str, max_samples: int = 50000) -> List[Sample]:
    """maywell/koVast — 멀티턴 대화 (첫 턴만 추출)"""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", token=os.environ.get("HF_TOKEN"))
    samples: List[Sample] = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        conversations = item.get("conversations", [])
        if len(conversations) >= 2:
            human_turn = next((c for c in conversations if c.get("from") == "human"), None)
            gpt_turn   = next((c for c in conversations if c.get("from") == "gpt"),   None)
            if human_turn and gpt_turn:
                s = _normalize_sample(
                    instruction=human_turn.get("value", ""),
                    input_text="",
                    output=gpt_turn.get("value", ""),
                )
                if s:
                    samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# Format-parsing helpers
# ---------------------------------------------------------------------------

def _extract_from_conversations(
    conversations: list,
) -> Optional[Tuple[str, str, str]]:
    """
    Extract (instruction, input, output) from a conversations list.

    Handles both dict-based conversation items (with "from"/"value" or
    "role"/"content" keys) and plain string lists.

    Returns None if the conversation does not contain at least one user turn
    followed by one assistant turn.
    """
    if not conversations:
        return None

    user_msg: Optional[str] = None
    assistant_msg: Optional[str] = None

    for item in conversations:
        if isinstance(item, dict):
            # OpenAI / ShareGPT style: {"role": "user", "content": "..."}
            role    = (item.get("role") or item.get("from") or "").lower()
            content = (item.get("content") or item.get("value") or "").strip()
        elif isinstance(item, str):
            # Occasionally items are raw strings; treat alternating as user/asst.
            content = item.strip()
            role    = "user" if user_msg is None else "assistant"
        else:
            continue

        if not content:
            continue

        if role in ("user", "human") and user_msg is None:
            user_msg = content
        elif role in ("assistant", "gpt", "bot") and user_msg is not None and assistant_msg is None:
            assistant_msg = content

        if user_msg is not None and assistant_msg is not None:
            break

    if user_msg is None or assistant_msg is None:
        return None

    return user_msg, "", assistant_msg


def _parse_alpaca_text(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an Alpaca-formatted text string of the form::

        Below is an instruction...

        ### Instruction:
        <instruction>

        ### Input:
        <input>

        ### Response:
        <response>

    Returns (instruction, input, response) or None on failure.
    """
    instruction = ""
    input_text  = ""
    output      = ""

    current_section: Optional[str] = None
    buffer: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        lower    = stripped.lower()

        if lower.startswith("### instruction"):
            if current_section == "input":
                input_text = "\n".join(buffer).strip()
            elif current_section == "response":
                output = "\n".join(buffer).strip()
            current_section = "instruction"
            buffer = []
        elif lower.startswith("### input"):
            if current_section == "instruction":
                instruction = "\n".join(buffer).strip()
            current_section = "input"
            buffer = []
        elif lower.startswith("### response") or lower.startswith("### output"):
            if current_section == "instruction":
                instruction = "\n".join(buffer).strip()
            elif current_section == "input":
                input_text = "\n".join(buffer).strip()
            current_section = "response"
            buffer = []
        else:
            if current_section is not None:
                buffer.append(line)

    # Flush final buffer
    if current_section == "instruction":
        instruction = "\n".join(buffer).strip()
    elif current_section == "input":
        input_text = "\n".join(buffer).strip()
    elif current_section == "response":
        output = "\n".join(buffer).strip()

    if not instruction or not output:
        return None

    return instruction, input_text, output


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

MIN_OUTPUT_LEN     = 10      # characters
MAX_OUTPUT_LEN     = 8_000   # characters


def _quality_filter(sample: Sample) -> bool:
    """품질 필터: 길이 + 반복 + 한국어 비율"""
    instruction = sample["instruction"]
    output      = sample["output"]

    # 길이 필터
    if len(instruction) < 10 or len(output) < 50:
        return False
    if len(output) > 3000:  # [수정] 4000→3000 긴 응답 제거
        return False

    # 한국어 비율 (최소 50% 이상 한글 문자) [수정] 30%→50%
    ko_chars = sum(1 for c in output if '가' <= c <= '힣')
    if len(output) > 0 and ko_chars / len(output) < 0.5:
        return False

    # 반복 퇴화 필터 (3-gram 반복 비율)
    words = output.split()
    if len(words) > 10:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if len(trigrams) > 0:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.5:  # 50% 이상 반복이면 제거
                return False

    return True


def _enhanced_quality_filter(sample: Sample) -> Optional[Sample]:
    """
    [추가] 데이터 품질 오염 필터:
      1. EOS 리터럴 텍스트 제거
      2. 질문:/답변: 패턴 오염 필터
      3. 50자 미만 output 필터
    """
    output = sample.get("output", "")
    # 1. EOS 리터럴 제거
    output = output.replace("</s>", "").replace("<|endoftext|>", "").strip()
    # 2. Q/A 패턴 오염 필터
    if re.search(r"(질문\s*:|답변\s*:|### Q|### A)", output):
        return None
    # 3. 너무 짧은 output 필터
    if len(output) < 50:
        return None
    sample["output"] = output
    return sample


def quality_filter(samples: List[Sample]) -> List[Sample]:
    """
    Remove samples that fail basic quality checks:
      - Empty instruction
      - Output shorter than MIN_OUTPUT_LEN characters
      - Output longer than MAX_OUTPUT_LEN characters
      - Korean character ratio below 30 %
      - 3-gram repetition ratio above 50 %
      - [추가] EOS 리터럴, Q/A 패턴 오염, 50자 미만
    """
    filtered: List[Sample] = []
    for s in samples:
        if not s["instruction"]:
            continue
        # [추가] Enhanced quality filter first (cleans output & rejects bad ones)
        s = _enhanced_quality_filter(s)
        if s is None:
            continue
        out_len = len(s["output"])
        if out_len < MIN_OUTPUT_LEN:
            continue
        if out_len > MAX_OUTPUT_LEN:
            continue
        if not _quality_filter(s):
            continue
        filtered.append(s)
    return filtered


def deduplicate(samples: List[Sample]) -> List[Sample]:
    """
    Remove duplicate samples based on instruction text (case-sensitive, exact).
    The first occurrence of each instruction is kept; subsequent ones are dropped.
    """
    seen: set[str] = set()
    unique: List[Sample] = []
    for s in samples:
        key = s["instruction"]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def apply_weighted_sampling(
    all_samples_with_source: Dict[str, List[Sample]],
    weights_dict: Dict[str, float],
) -> List[Sample]:
    """
    소스별 가중치에 따라 샘플을 업샘플링/다운샘플링.

    weights > 1.0: 업샘플링 (기본 + 추가 복제)
    weights < 1.0: 다운샘플링 (랜덤 제거, 최소 1개 유지)
    weights == 1.0: 변경 없음

    Args:
        all_samples_with_source: 소스명 → 샘플 리스트 매핑
        weights_dict: 소스명 → 가중치 매핑 (키 없으면 1.0 사용)

    Returns:
        가중치 적용 후 합쳐진 샘플 리스트
    """
    result: List[Sample] = []
    for source_name, samples in all_samples_with_source.items():
        if not samples:
            continue
        weight = weights_dict.get(source_name, 1.0)
        if weight >= 1.0:
            # 업샘플링: 원본 전체 포함 + 추가 복제
            result.extend(samples)
            extra = int(len(samples) * (weight - 1.0))
            if extra > 0:
                result.extend(random.choices(samples, k=extra))
        else:
            # 다운샘플링: weight 비율만큼만 유지 (최소 1개)
            keep = max(1, int(len(samples) * weight))
            result.extend(random.sample(samples, keep))
        target = int(len(samples) * weight)
        print(f"  {source_name}: {len(samples):,} → {target:,} (×{weight})")
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(samples: List[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def _avg_len(samples: List[Sample], field: str) -> float:
    if not samples:
        return 0.0
    return sum(len(s[field]) for s in samples) / len(samples)


# ---------------------------------------------------------------------------
# Dataset registry & sampling weights
# ---------------------------------------------------------------------------

# Weights control upsampling/downsampling relative to a baseline of 1.0.
# Values >1 cause the source to be overrepresented; values <1 underrepresent.
DATASET_WEIGHTS: Dict[str, float] = {
    # 키는 DATASET_REGISTRY 의 display_name 과 정확히 일치해야 합니다.
    "KOR-OpenOrca-Platypus-v3":   1.5,  # [수정] 2.0→1.5
    "kullm-v2":                   1.0,  # 기본값
    "ko-alpaca-12k":              2.0,  # 고품질 → 2배 샘플링
    "korean_safe_conversation":   1.5,
    "evol-instruct-korean":       2.0,  # [수정] 1.5→2.0
    "kovast":                     0.5,  # [수정] 0.8→0.5 다운샘플링 강화
}

# Each entry: (display_name, hf_repo_id, loader_function)
DATASET_REGISTRY = [
    (
        "KOR-OpenOrca-Platypus-v3",
        "kyujinpy/KOR-OpenOrca-Platypus-v3",
        load_kor_openorca_platypus,
    ),
    (
        "kullm-v2",
        "nlpai-lab/kullm-v2",
        load_kullm_v2,
    ),
    (
        "ko-alpaca-12k",
        "junhochoi/ko-alpaca-12k",
        load_ko_alpaca,
    ),
    (
        "korean_safe_conversation",
        "jojo0217/korean_safe_conversation",
        load_korean_safe_conversation,
    ),
    (
        "evol-instruct-korean",
        "FreedomIntelligence/Evol-Instruct-Korean",
        load_evol_instruct_korean,
    ),
    (
        "kovast",
        "maywell/koVast",
        load_kovast,
    ),
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare Korean SFT datasets from HuggingFace. "
            "Outputs train.jsonl and val.jsonl in the specified directory."
        )
    )
    parser.add_argument(
        "--output_dir",
        default="data/sft/",
        help="Directory where train.jsonl and val.jsonl will be written "
             "(default: data/sft/)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of samples reserved for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before the train/val split (default: 42)",
    )
    parser.add_argument(
        "--min_output_len",
        type=int,
        default=MIN_OUTPUT_LEN,
        help=f"Minimum output length in characters (default: {MIN_OUTPUT_LEN})",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=MAX_OUTPUT_LEN,
        help=f"Maximum output length in characters (default: {MAX_OUTPUT_LEN})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Allow overriding filter thresholds via CLI
    global MIN_OUTPUT_LEN, MAX_OUTPUT_LEN
    MIN_OUTPUT_LEN = args.min_output_len
    MAX_OUTPUT_LEN = args.max_output_len

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download and normalise each dataset --------------------------------

    samples_by_source: Dict[str, List[Sample]] = {}

    for display_name, repo_id, loader_fn in DATASET_REGISTRY:
        print(f"\nDownloading {display_name}...")
        try:
            raw = loader_fn(repo_id)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"  WARNING: Failed to load {display_name} ({repo_id}): {exc}",
                file=sys.stderr,
            )
            continue

        before = len(raw)
        filtered = quality_filter(raw)
        after = len(filtered)

        print(f"  Loaded {before:,} samples -> {after:,} after filtering")
        samples_by_source[display_name] = filtered

    # ---- Weighted sampling --------------------------------------------------

    print("\n[Weighted Sampling]")
    all_samples: List[Sample] = apply_weighted_sampling(samples_by_source, DATASET_WEIGHTS)

    if not all_samples:
        print(
            "\nERROR: No samples were collected. "
            "Check network connectivity and dataset availability.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Deduplication -------------------------------------------------------

    total_before_dedup = len(all_samples)
    all_samples = deduplicate(all_samples)
    total_after_dedup = len(all_samples)

    print(f"\nTotal: {total_before_dedup:,} samples")
    print(f"After deduplication: {total_after_dedup:,} samples")

    # ---- Shuffle and split ---------------------------------------------------

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    val_size   = max(1, int(len(all_samples) * args.val_split))
    train_size = len(all_samples) - val_size

    train_samples = all_samples[:train_size]
    val_samples   = all_samples[train_size:]

    print(f"Train: {len(train_samples):,} | Val: {len(val_samples):,}")

    # ---- Save ----------------------------------------------------------------

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples,   val_path)

    # ---- Statistics ----------------------------------------------------------

    avg_instr_train  = _avg_len(train_samples, "instruction")
    avg_output_train = _avg_len(train_samples, "output")
    avg_input_train  = _avg_len(train_samples, "input")

    print(f"\nSaved to:")
    print(f"  {train_path} ({len(train_samples):,} samples)")
    print(f"  {val_path}   ({len(val_samples):,} samples)")
    print()
    print("--- Statistics (train set) ---")
    print(f"  Avg instruction length : {avg_instr_train:.1f} chars")
    print(f"  Avg input length       : {avg_input_train:.1f} chars")
    print(f"  Avg output length      : {avg_output_train:.1f} chars")

    # Rough token estimate (Korean ~1.5 chars per token for BPE tokenizers)
    est_tokens = (avg_instr_train + avg_input_train + avg_output_train) * len(train_samples) / 1.5
    print(f"  Est. tokens (train)    : ~{est_tokens / 1e6:.1f}M  (rough, 1.5 chars/tok)")


if __name__ == "__main__":
    main()

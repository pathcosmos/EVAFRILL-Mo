#!/usr/bin/env python3
"""
data/generate_repetition_preference.py — Self-play preference data targeting repetition.

Generates (prompt, chosen, rejected) pairs by:
  - rejected: greedy decoding (temp=0, rep_penalty=1.0) → tends to repeat
  - chosen:   sampling with repetition penalty (temp=0.7, rep_penalty=1.2) → cleaner

Only keeps pairs where rejected has strictly higher 3-gram repetition rate than chosen.

Usage:
    python3 data/generate_repetition_preference.py \
        --checkpoint checkpoints/3b_dpo/checkpoint-slerp

    python3 data/generate_repetition_preference.py \
        --checkpoint checkpoints/3b_dpo/checkpoint-slerp \
        --output data/preference/repetition_preference.jsonl \
        --num_prompts 100 \
        --max_tokens 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import LLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Korean prompt bank — 100+ diverse prompts
# ---------------------------------------------------------------------------

# 15 existing eval prompts (completion style → wrapped in chat template)
_EVAL_PROMPTS = [
    "대한민국의 수도는 어디인지 설명해주세요.",
    "인공지능이란 무엇인지 자세히 설명해주세요.",
    "한국의 전통 음식 중에서 대표적인 것들을 소개해주세요.",
    "지구 온난화의 주요 원인은 무엇인가요?",
    "프로그래밍을 배우려면 어떻게 시작해야 하나요?",
    "조선시대에는 어떤 일들이 있었나요?",
    "물리학에서 에너지란 무엇인지 설명해주세요.",
    "한국어는 세계에서 어떤 특징을 가지고 있나요?",
    "경제 성장을 위해서는 무엇이 필요한가요?",
    "우주 탐사의 역사를 간단히 설명해주세요.",
    "머신러닝과 딥러닝의 차이는 무엇인가요?",
    "한국 문학의 대표적인 작품으로는 어떤 것들이 있나요?",
    "양자 컴퓨터란 무엇인지 설명해주세요.",
    "건강한 식습관을 위해서는 어떻게 해야 하나요?",
    "세계 2차 대전 이후 세계는 어떻게 변했나요?",
]

# Additional diverse prompts (~85 more)
_EXTRA_PROMPTS = [
    # 일상 대화
    "오늘 날씨가 좋은데 뭐 하면 좋을까요?",
    "주말에 뭐 하면 좋을지 추천해주세요.",
    "좋은 하루를 시작하는 방법을 알려주세요.",
    "집에서 할 수 있는 취미 활동을 추천해주세요.",
    "친구와 싸웠을 때 어떻게 화해하면 좋을까요?",
    "외로움을 느낄 때 어떻게 극복할 수 있나요?",
    "시간 관리를 잘 하는 방법을 알려주세요.",
    "아침 일찍 일어나는 습관을 만들려면 어떻게 해야 하나요?",
    "새로운 도시로 이사했을 때 적응하는 방법은?",
    "카페에서 혼자 시간 보내는 것의 장점은 무엇인가요?",

    # 지식 — 과학
    "DNA가 무엇인지 설명해주세요.",
    "블랙홀이란 무엇인가요?",
    "진화론이란 무엇인지 간단히 설명해주세요.",
    "기후 변화가 생태계에 미치는 영향은 무엇인가요?",
    "인체의 면역 시스템은 어떻게 작동하나요?",
    "빛의 속도는 왜 중요한가요?",
    "원자와 분자의 차이점은 무엇인가요?",
    "광합성이란 무엇인지 설명해주세요.",
    "중력파란 무엇인가요?",
    "줄기세포 치료란 무엇이며 어떻게 활용되나요?",

    # 지식 — 역사·사회
    "한국의 역사에서 가장 중요한 사건은 무엇인가요?",
    "민주주의란 무엇인지 설명해주세요.",
    "산업혁명이 세계에 미친 영향은 무엇인가요?",
    "냉전이란 무엇이었나요?",
    "한국 전쟁의 원인과 결과를 설명해주세요.",
    "세계화란 무엇이며 어떤 영향을 미치나요?",
    "인권이란 무엇이며 왜 중요한가요?",
    "실크로드가 역사적으로 중요한 이유는 무엇인가요?",
    "르네상스 시대는 어떤 시기였나요?",
    "한국의 독립운동에 대해 설명해주세요.",

    # 조언 — 직업·학습
    "취업 면접 잘 보는 방법은 무엇인가요?",
    "이력서를 잘 쓰는 방법을 알려주세요.",
    "대학 생활을 알차게 보내는 방법은?",
    "공부 집중력을 높이는 방법을 알려주세요.",
    "외국어를 빠르게 배우는 방법은 무엇인가요?",
    "직장에서 상사와 잘 지내는 방법은?",
    "프리랜서로 일하면 어떤 장단점이 있나요?",
    "자기소개서를 잘 쓰는 팁을 알려주세요.",
    "독서 습관을 기르는 방법은 무엇인가요?",
    "수학을 잘하기 위한 공부 방법은?",

    # 조언 — 건강·심리
    "스트레스 해소 방법을 알려주세요.",
    "우울감을 극복하는 방법은 무엇인가요?",
    "규칙적인 운동 습관을 만드는 방법은?",
    "수면의 질을 높이는 방법을 알려주세요.",
    "번아웃을 예방하는 방법은 무엇인가요?",
    "마음의 평화를 찾는 방법은?",
    "자존감을 높이는 방법을 알려주세요.",
    "명상을 시작하려면 어떻게 해야 하나요?",
    "건강한 체중을 유지하는 방법은?",
    "디지털 중독을 극복하는 방법을 알려주세요.",

    # 창작
    "짧은 동화를 하나 만들어주세요.",
    "봄에 대한 시를 써주세요.",
    "미래 도시를 배경으로 한 짧은 이야기를 써주세요.",
    "바다에 관한 짧은 수필을 써주세요.",
    "고양이를 주인공으로 한 짧은 이야기를 만들어주세요.",
    "가을 풍경을 묘사하는 글을 써주세요.",
    "우정에 관한 짧은 시를 써주세요.",
    "엄마에게 보내는 편지를 써주세요.",
    "미래의 나에게 쓰는 편지를 작성해주세요.",
    "어린 시절 추억에 관한 짧은 글을 써주세요.",

    # 기술·IT
    "클라우드 컴퓨팅이란 무엇인가요?",
    "블록체인이 무엇인지 설명해주세요.",
    "사이버 보안이 왜 중요한가요?",
    "빅데이터란 무엇이며 어떻게 활용되나요?",
    "5G 기술이 가져올 변화는 무엇인가요?",
    "인터넷 검색 엔진은 어떻게 작동하나요?",
    "스마트폰이 생활에 미친 영향은 무엇인가요?",
    "가상현실과 증강현실의 차이는 무엇인가요?",
    "자율주행 자동차 기술은 어디까지 왔나요?",
    "오픈소스 소프트웨어란 무엇인가요?",

    # 문화·예술
    "K-팝이 세계적으로 인기를 얻은 이유는 무엇인가요?",
    "한국 영화가 세계 시장에서 주목받는 이유는?",
    "전통 예술과 현대 예술의 차이는 무엇인가요?",
    "음악이 감정에 미치는 영향은 무엇인가요?",
    "독서가 삶에 미치는 긍정적인 영향은?",
    "미술 감상을 잘 하는 방법을 알려주세요.",
    "한국 전통 음악인 국악의 특징은 무엇인가요?",
    "영화 비평을 잘 쓰는 방법은?",
    "여행이 사람을 성장시키는 이유는 무엇인가요?",
    "사진 찍기를 잘 하는 팁을 알려주세요.",

    # 환경·사회
    "환경 보호를 위해 개인이 할 수 있는 일은?",
    "재활용의 중요성과 방법을 설명해주세요.",
    "채식주의의 장단점은 무엇인가요?",
    "동물 복지란 무엇이며 왜 중요한가요?",
    "지속 가능한 발전이란 무엇인가요?",
    "노령화 사회가 가져오는 문제점은 무엇인가요?",
    "교육 불평등을 해소하는 방법은?",
    "빈곤 문제를 해결하기 위한 방법은?",
    "다문화 사회에서 공존하는 방법은?",
    "봉사 활동이 사회에 미치는 영향은 무엇인가요?",
]

ALL_PROMPTS = _EVAL_PROMPTS + _EXTRA_PROMPTS  # 15 + 85 = 100

CHAT_TEMPLATE = "<|user|>\n{prompt}\n<|assistant|>\n"

EOS_TOKEN_ID = 2


# ---------------------------------------------------------------------------
# Repetition metric
# ---------------------------------------------------------------------------

def compute_ngram_repetition_rate(tokens: List[int], n: int = 3) -> float:
    """Fraction of n-gram positions that are repeats of an earlier occurrence."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    seen: set = set()
    repeated = 0
    for ng in ngrams:
        if ng in seen:
            repeated += 1
        seen.add(ng)
    return repeated / len(ngrams)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    eos_token_id: int,
) -> List[int]:
    """Auto-regressive generation with optional repetition penalty.

    Args:
        model: LLM instance already on device
        input_ids: (1, T) prompt token ids
        max_new_tokens: max tokens to generate
        temperature: sampling temperature (0 = greedy)
        repetition_penalty: penalty > 1 reduces prob of previously seen tokens
        eos_token_id: stop generation when this token is produced

    Returns:
        List of generated token ids (not including the prompt).
    """
    device = input_ids.device
    generated: List[int] = []
    current_ids = input_ids.clone()  # (1, T)

    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(current_ids)  # (1, T, V)

        next_logits = logits[0, -1, :].float()  # (V,)

        # Repetition penalty: discount logits for already-generated tokens
        if repetition_penalty != 1.0:
            all_seen_ids = current_ids[0].tolist() + generated
            for token_id in set(all_seen_ids):
                if token_id < next_logits.shape[0]:
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= repetition_penalty
                    else:
                        next_logits[token_id] /= repetition_penalty

        # Sample / greedy
        if temperature == 0.0:
            next_token = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_token)

        if next_token == eos_token_id:
            break

        # Append to context
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate self-play repetition preference data"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/3b_dpo/checkpoint-slerp"),
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/preference/repetition_preference.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="How many prompts to use (default: all ~100)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to tokenizer.json (default: auto-resolve)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min_rep_diff",
        type=float,
        default=0.0,
        help="Minimum difference (rejected_rep - chosen_rep) to keep a pair (default: >0)",
    )
    return parser.parse_args()


def _resolve_tokenizer(args: argparse.Namespace) -> Path:
    if args.tokenizer is not None:
        return Path(args.tokenizer)
    # Try checkpoint dir first
    ckpt_tok = args.checkpoint / "tokenizer.json"
    if ckpt_tok.exists():
        return ckpt_tok
    # Fall back to project default
    default_tok = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    if default_tok.exists():
        return default_tok
    raise FileNotFoundError(
        "Cannot find tokenizer.json — specify with --tokenizer"
    )


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prompts
    prompts = ALL_PROMPTS
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    print(f"[INFO] Using {len(prompts)} prompts")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Tokenizer
    tokenizer_path = _resolve_tokenizer(args)
    print(f"[INFO] Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Model
    checkpoint_path = _PROJECT_ROOT / args.checkpoint if not args.checkpoint.is_absolute() else args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading model from {checkpoint_path} ...")
    t0 = time.perf_counter()
    model = LLM.from_pretrained(checkpoint_path)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"[INFO] Model loaded in {time.perf_counter() - t0:.1f}s")

    # Output dir
    output_path = _PROJECT_ROOT / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stats
    valid_pairs = 0
    skipped = 0
    total_rejected_rep = 0.0
    total_chosen_rep = 0.0

    t_start = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, prompt_text in enumerate(prompts):
            prompt_str = CHAT_TEMPLATE.format(prompt=prompt_text)

            # Tokenize prompt
            encoding = tokenizer.encode(prompt_str)
            prompt_ids = encoding.ids
            if not prompt_ids:
                print(f"  [{idx+1}/{len(prompts)}] SKIP: empty tokenization for prompt")
                skipped += 1
                continue

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # --- Generate REJECTED: greedy, no rep penalty ---
            rej_tokens = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.0,
                repetition_penalty=1.0,
                eos_token_id=EOS_TOKEN_ID,
            )

            # --- Generate CHOSEN: sampling + rep penalty ---
            cho_tokens = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                repetition_penalty=1.2,
                eos_token_id=EOS_TOKEN_ID,
            )

            # Decode (strip EOS)
            rej_clean = [t for t in rej_tokens if t != EOS_TOKEN_ID]
            cho_clean = [t for t in cho_tokens if t != EOS_TOKEN_ID]

            rej_text = tokenizer.decode(rej_clean)
            cho_text = tokenizer.decode(cho_clean)

            # Compute 3-gram repetition rates on generated tokens
            rej_rep = compute_ngram_repetition_rate(rej_clean, n=3)
            cho_rep = compute_ngram_repetition_rate(cho_clean, n=3)

            # Filter: only keep if rejected is more repetitive than chosen
            diff = rej_rep - cho_rep
            if diff <= args.min_rep_diff:
                status = "SKIP"
                skipped += 1
            else:
                status = "KEEP"
                valid_pairs += 1
                total_rejected_rep += rej_rep
                total_chosen_rep += cho_rep
                record = {
                    "prompt": prompt_str,
                    "chosen": cho_text,
                    "rejected": rej_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            elapsed = time.perf_counter() - t_start
            print(
                f"  [{idx+1:3d}/{len(prompts)}] {status:4s} "
                f"rej_rep={rej_rep:.3f} cho_rep={cho_rep:.3f} diff={diff:+.3f} "
                f"| rej_len={len(rej_clean)} cho_len={len(cho_clean)} "
                f"| elapsed={elapsed:.1f}s"
            )

    # Summary
    elapsed_total = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"Generation complete in {elapsed_total:.1f}s")
    print(f"  Total prompts processed : {len(prompts)}")
    print(f"  Valid pairs kept        : {valid_pairs}")
    print(f"  Skipped (rep filter)    : {skipped}")
    if valid_pairs > 0:
        avg_rej = total_rejected_rep / valid_pairs
        avg_cho = total_chosen_rep / valid_pairs
        print(f"  Avg rejected 3-gram rep : {avg_rej:.4f}")
        print(f"  Avg chosen  3-gram rep  : {avg_cho:.4f}")
        print(f"  Avg improvement         : {avg_rej - avg_cho:+.4f}")
    print(f"  Output saved to         : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

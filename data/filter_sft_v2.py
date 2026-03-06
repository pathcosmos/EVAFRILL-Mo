#!/usr/bin/env python3
"""
filter_sft_v2.py — SFT 데이터 품질 필터 (JSONL messages 포맷)

필터 규칙:
  1. </s> 리터럴 제거 (assistant 메시지에서 </s> 태그 strip)
  2. Q:, A:, 질문:, 답변: 등 Q/A 마커 제거 (content 시작 부분)
  3. 50자 미만 극단 단문 제거 (assistant 응답 기준)
  4. 4-gram 반복률 >30% 제거 (assistant 응답 기준)

Usage:
  python data/filter_sft_v2.py \\
      --input  data/sft_combined/train.jsonl \\
      --output data/sft_combined/train_filtered.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# 필터 1: </s> 리터럴 제거
# ---------------------------------------------------------------------------
_EOS_PATTERN = re.compile(r"</s>", re.IGNORECASE)


def strip_eos_tag(text: str) -> str:
    """</s> 태그를 제거하고 앞뒤 공백을 정리한다."""
    return _EOS_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# 필터 2: Q/A 마커 제거
# ---------------------------------------------------------------------------
# content 시작 부분의 마커 패턴 (한국어·영어 모두 처리)
_QA_MARKER_PATTERN = re.compile(
    r"^\s*(?:"
    r"질문\s*[:：]\s*"
    r"|답변\s*[:：]\s*"
    r"|Q\s*[:：]\s*"
    r"|A\s*[:：]\s*"
    r"|Answer\s*[:：]\s*"
    r"|Question\s*[:：]\s*"
    r")+",
    re.IGNORECASE,
)


def strip_qa_markers(text: str) -> str:
    """content 시작 부분의 Q/A 마커를 제거한다."""
    return _QA_MARKER_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# 필터 3: 극단 단문 판단
# ---------------------------------------------------------------------------
MIN_ASSISTANT_LEN = 50  # 글자 수 기준


def is_too_short(text: str) -> bool:
    return len(text) < MIN_ASSISTANT_LEN


# ---------------------------------------------------------------------------
# 필터 4: 4-gram 반복률
# ---------------------------------------------------------------------------
NGRAM_SIZE = 4
MAX_REPEAT_RATIO = 0.30  # 30% 초과 시 제거


def _tokenize_ngrams(text: str, n: int):
    """공백 단위 토크나이즈 후 n-gram 리스트 반환. 한국어 fallback 포함."""
    tokens = text.split()
    # 한국어 fallback: 공백 토큰이 부족하면 문자 레벨 n-gram 사용
    if len(tokens) < n * 3:
        # 공백/구두점 제거 후 문자 단위
        chars = [c for c in text if not c.isspace()]
        if len(chars) < n:
            return []
        return [tuple(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_repeat_ratio(text: str, n: int = NGRAM_SIZE) -> float:
    """
    (중복 n-gram 수) / (전체 n-gram 수) 비율을 반환한다.
    전체 n-gram이 없으면 0.0 반환.
    """
    ngrams = _tokenize_ngrams(text, n)
    total = len(ngrams)
    if total == 0:
        return 0.0
    counts = Counter(ngrams)
    # 1회 초과 등장한 n-gram 개수(중복분)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / total


def is_repetitive(text: str) -> bool:
    return ngram_repeat_ratio(text) > MAX_REPEAT_RATIO


# ---------------------------------------------------------------------------
# 필터 5: 초장문 응답 필터
# ---------------------------------------------------------------------------
MAX_CHAR_LEN = 20000  # 20K 글자 초과 시 제거


def is_too_long(text: str) -> bool:
    return len(text) > MAX_CHAR_LEN


# ---------------------------------------------------------------------------
# 메시지 정제 / 샘플 수준 필터링
# ---------------------------------------------------------------------------

def clean_message_content(content: str, role: str) -> str:
    """단일 메시지의 content를 정제한다."""
    # 필터 1: </s> 태그 제거 (assistant 한정)
    if role == "assistant":
        content = strip_eos_tag(content)
    # 필터 2: Q/A 마커 제거 (모든 role)
    content = strip_qa_markers(content)
    return content


def filter_sample(sample: dict) -> tuple[dict | None, str]:
    """
    하나의 샘플을 검사·정제한다.
    반환: (정제된 샘플 또는 None, 제거 이유 또는 "")
    """
    messages = sample.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "no_messages"

    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = clean_message_content(content, role)
        cleaned_messages.append({**msg, "content": content})

    # assistant 응답 기준 필터 적용
    assistant_contents = [
        m["content"] for m in cleaned_messages if m.get("role") == "assistant"
    ]

    if not assistant_contents:
        return None, "no_assistant_turn"

    for ac in assistant_contents:
        # 필터 3: 극단 단문
        if is_too_short(ac):
            return None, "too_short"
        # 필터 5: 초장문
        if is_too_long(ac):
            return None, "too_long"
        # 필터 4: 4-gram 반복
        if is_repetitive(ac):
            return None, "repetitive"

    result = {**sample, "messages": cleaned_messages}
    return result, ""


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT 데이터 품질 필터 (JSONL messages 포맷)"
    )
    parser.add_argument("--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 파일 경로")
    return parser.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: 입력 파일을 찾을 수 없습니다: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 통계 카운터
    stats: dict[str, int] = {
        "total": 0,
        "no_messages": 0,
        "no_assistant_turn": 0,
        "too_short": 0,
        "too_long": 0,
        "repetitive": 0,
        "json_error": 0,
        "passed": 0,
    }

    with in_path.open("r", errors="replace") as fin, out_path.open("w") as fout:
        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            stats["total"] += 1

            try:
                sample = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[WARN] 라인 {lineno} JSON 파싱 실패: {e}", file=sys.stderr)
                stats["json_error"] += 1
                continue

            cleaned, reason = filter_sample(sample)
            if cleaned is None:
                stats[reason] = stats.get(reason, 0) + 1
            else:
                stats["passed"] += 1
                fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    # 통계 출력
    total = stats["total"]
    removed = total - stats["passed"]
    print("=" * 60)
    print(f"  입력 파일  : {in_path}")
    print(f"  출력 파일  : {out_path}")
    print("=" * 60)
    print(f"  총 입력    : {total:>10,}")
    print(f"  [제거] no_messages      : {stats['no_messages']:>10,}")
    print(f"  [제거] no_assistant_turn: {stats['no_assistant_turn']:>10,}")
    print(f"  [제거] too_short (<50자): {stats['too_short']:>10,}")
    print(f"  [제거] too_long (>{MAX_CHAR_LEN}자): {stats['too_long']:>10,}")
    print(f"  [제거] json_error          : {stats['json_error']:>10,}")
    print(f"  [제거] repetitive (4-gram >30%): {stats['repetitive']:>10,}")
    print(f"  총 제거    : {removed:>10,}  ({removed/total*100:.1f}%)")
    print(f"  최종 잔존  : {stats['passed']:>10,}  ({stats['passed']/total*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare_preference_combined.py — Preference 데이터 통합 + 포맷 정규화 스크립트
Phase 0F: ORPO 파이프라인 준비

입력 디렉토리: data/preference/
출력 파일:     data/preference/combined_preference.jsonl

지원 포맷:
  - {prompt, chosen, rejected}                        (표준 DPO/ORPO 포맷)
  - {question, chosen, rejected, [system]}            (heegyu, kuotient orca-math 계열)
  - {instruction, chosen, rejected}                   (instruction 키 변형)
  - {orig_instruction, orig_response_A/B, orig_preference}  (nayohan preference-collection)
  - {prompt, response_a, response_b, preferred}       (response_a/b + preferred 키)
  - {prompt, response_a, response_b, winner}          (winner 키 변형)
  - {instruction, preferred, dispreferred}            (preferred/dispreferred 키)
  - {prompt, winning_response, losing_response}       (Ultrafeedback 계열)
  - {conversations, chosen, rejected}                 (conversations 리스트 포맷)

품질 필터:
  - chosen, rejected 모두 비어있지 않을 것
  - chosen != rejected
  - 최소 20자 이상 (chosen 기준)

Usage:
    python data/prepare_preference_combined.py [--input_dir data/preference] [--output data/preference/combined_preference.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 필드명 자동 감지 로직
# ---------------------------------------------------------------------------

def _extract_text(val) -> str:
    """값이 str이면 그대로, list(conversations 포맷)이면 마지막 content 추출."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        # [{"role": ..., "content": ...}, ...] 형태
        parts = []
        for item in val:
            if isinstance(item, dict):
                content = item.get("content") or item.get("value") or item.get("text") or ""
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(val, dict):
        return (val.get("content") or val.get("value") or val.get("text") or "").strip()
    return str(val).strip()


def _build_prompt(record: dict) -> str:
    """레코드에서 prompt 문자열을 추출한다."""
    # 표준 prompt 키
    for key in ("prompt", "instruction", "question", "input", "user_prompt", "orig_instruction"):
        if key in record and record[key]:
            val = _extract_text(record[key])
            if val:
                # system 필드가 있으면 앞에 붙임
                system = record.get("system", "")
                if system:
                    return f"{system.strip()}\n{val}"
                return val

    # conversations 포맷: 첫 번째 human 턴
    if "conversations" in record:
        convs = record["conversations"]
        if isinstance(convs, list):
            for item in convs:
                role = (item.get("role") or item.get("from") or "").lower()
                if role in ("human", "user"):
                    return _extract_text(item.get("content") or item.get("value") or "")

    return ""


def normalize_record(record: dict, source_name: str) -> Optional[dict]:
    """
    단일 레코드를 {prompt, chosen, rejected} 로 정규화.
    변환 불가 시 None 반환.
    """
    chosen = ""
    rejected = ""

    # --- 패턴 1: 표준 {chosen, rejected} ---
    if "chosen" in record and "rejected" in record:
        chosen = _extract_text(record["chosen"])
        rejected = _extract_text(record["rejected"])

    # --- 패턴 2: nayohan preference-collection (orig_preference + orig_response_A/B) ---
    elif "orig_preference" in record:
        resp_a = _extract_text(record.get("orig_response_A", record.get("response_A", "")))
        resp_b = _extract_text(record.get("orig_response_B", record.get("response_B", "")))
        pref = str(record.get("orig_preference", "")).strip().upper()
        if pref == "B":
            chosen, rejected = resp_b, resp_a
        else:
            chosen, rejected = resp_a, resp_b

    # --- 패턴 3: preferred/dispreferred ---
    elif "preferred" in record and "dispreferred" in record:
        chosen = _extract_text(record["preferred"])
        rejected = _extract_text(record["dispreferred"])

    # --- 패턴 4: response_a/b + preferred or winner 키 ---
    elif "response_a" in record and "response_b" in record:
        resp_a = _extract_text(record["response_a"])
        resp_b = _extract_text(record["response_b"])
        winner_key = record.get("preferred") or record.get("winner") or ""
        winner = str(winner_key).strip().lower()
        if winner in ("b", "response_b", "model_b"):
            chosen, rejected = resp_b, resp_a
        else:
            # 기본: A가 chosen
            chosen, rejected = resp_a, resp_b

    # --- 패턴 5: winning_response / losing_response (Ultrafeedback 계열) ---
    elif "winning_response" in record and "losing_response" in record:
        chosen = _extract_text(record["winning_response"])
        rejected = _extract_text(record["losing_response"])

    # --- 패턴 6: completions 리스트 (일부 HH-RLHF 변형) ---
    elif "completions" in record:
        completions = record["completions"]
        if isinstance(completions, list) and len(completions) >= 2:
            # rating 있으면 내림차순 정렬
            def rating(c):
                return c.get("rating", c.get("score", 0)) if isinstance(c, dict) else 0
            sorted_c = sorted(completions, key=rating, reverse=True)
            chosen = _extract_text(sorted_c[0].get("text", sorted_c[0]) if isinstance(sorted_c[0], dict) else sorted_c[0])
            rejected = _extract_text(sorted_c[-1].get("text", sorted_c[-1]) if isinstance(sorted_c[-1], dict) else sorted_c[-1])

    else:
        return None  # 알 수 없는 포맷

    prompt = _build_prompt(record)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# ---------------------------------------------------------------------------
# 품질 필터
# ---------------------------------------------------------------------------

MIN_LEN = 20

def passes_quality_filter(record: dict) -> bool:
    """품질 필터: chosen/rejected 비어있지 않고, 다르고, 최소 길이 충족."""
    prompt = record.get("prompt", "")
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")

    if not chosen or not rejected:
        return False
    if chosen == rejected:
        return False
    if len(chosen) < MIN_LEN:
        return False
    if not prompt:
        # prompt 없으면 경고만 — 완전히 버리지는 않음 (ORPO는 prompt 필수이므로 실제로 제외)
        return False
    return True


# ---------------------------------------------------------------------------
# 파일별 로더
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    """JSONL 파일을 순차적으로 파싱하는 제너레이터."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  JSON 파싱 오류 {path.name}:{lineno} — {e}")


def process_file(src_path: Path, out_f, stats: dict) -> None:
    """단일 JSONL 파일을 읽어 정규화 후 out_f에 쓴다. stats 딕셔너리 갱신."""
    source_name = src_path.stem
    loaded = 0
    written = 0
    skipped_format = 0
    skipped_quality = 0

    log.info(f"  로딩: {src_path.name}")
    for record in load_jsonl(src_path):
        loaded += 1
        normalized = normalize_record(record, source_name)
        if normalized is None:
            skipped_format += 1
            continue
        if not passes_quality_filter(normalized):
            skipped_quality += 1
            continue
        out_f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        written += 1

    log.info(
        f"    {source_name}: 로딩 {loaded:,} → 포맷 스킵 {skipped_format:,} → 품질 스킵 {skipped_quality:,} → 출력 {written:,}"
    )
    stats[source_name] = {
        "loaded": loaded,
        "skipped_format": skipped_format,
        "skipped_quality": skipped_quality,
        "written": written,
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

# 처리할 파일 목록 (순서 고정 → 재현성)
TARGET_FILES = [
    "heegyu_orca-math-korean-preference-cleaned.jsonl",
    "kuotient_orca-math-korean-dpo-pairs.jsonl",
    "nayohan_preference-collection-ko-full.jsonl",
    "maywell_ko_Ultrafeedback_binarized.jsonl",
    "jojo0217_korean_rlhf_dataset.jsonl",
    "lemon-mint_korean-realqa-reasoning-v01-preference.jsonl",
    "tellang_yeji-preference-ko-v1.jsonl",
]


def main():
    parser = argparse.ArgumentParser(
        description="Preference 데이터 통합 + 포맷 정규화 (ORPO 호환)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/preference",
        help="입력 디렉토리 (기본: data/preference)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preference/combined_preference.jsonl",
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="TARGET_FILES 목록 외의 .jsonl 파일도 포함",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        log.error(f"입력 디렉토리 없음: {input_dir}")
        sys.exit(1)

    # 처리 파일 결정
    if args.include_all:
        src_files = sorted(input_dir.glob("*.jsonl"))
        # combined_preference.jsonl 자기 자신 제외
        src_files = [f for f in src_files if f.name != output_path.name]
    else:
        src_files = []
        for fname in TARGET_FILES:
            p = input_dir / fname
            if p.exists():
                src_files.append(p)
            else:
                log.warning(f"파일 없음 (스킵): {p}")

    if not src_files:
        log.error("처리할 JSONL 파일이 없습니다.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 0F: Preference 데이터 통합")
    log.info(f"  입력 파일 수  : {len(src_files)}")
    log.info(f"  출력 파일     : {output_path}")
    log.info(f"  최소 길이 기준: {MIN_LEN}자")
    log.info("=" * 60)

    stats: dict = {}
    total_written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for src_path in src_files:
            process_file(src_path, out_f, stats)
            total_written += stats.get(src_path.stem, {}).get("written", 0)

    # 최종 통계 요약
    log.info("")
    log.info("=" * 60)
    log.info("최종 통계 요약")
    log.info("=" * 60)
    log.info(f"{'데이터셋':<50} {'로딩':>8} {'포맷스킵':>8} {'품질스킵':>8} {'출력':>8}")
    log.info("-" * 86)
    grand_loaded = 0
    grand_fmt_skip = 0
    grand_qual_skip = 0
    for name, s in stats.items():
        log.info(
            f"{name:<50} {s['loaded']:>8,} {s['skipped_format']:>8,} {s['skipped_quality']:>8,} {s['written']:>8,}"
        )
        grand_loaded += s["loaded"]
        grand_fmt_skip += s["skipped_format"]
        grand_qual_skip += s["skipped_quality"]
    log.info("-" * 86)
    log.info(
        f"{'합계':<50} {grand_loaded:>8,} {grand_fmt_skip:>8,} {grand_qual_skip:>8,} {total_written:>8,}"
    )
    log.info("=" * 60)
    log.info(f"출력 완료: {output_path}  ({total_written:,}개 레코드)")


if __name__ == "__main__":
    main()

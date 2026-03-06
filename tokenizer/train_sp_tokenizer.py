#!/usr/bin/env python3
"""
tokenizer/train_sp_tokenizer.py — SentencePiece Unigram 한국어 토크나이저 학습.

한국어 1음절(UTF-8 3바이트) = 1토큰이 되도록 Unigram 모델을 사용.
character_coverage=0.9995로 한글 11,172 음절 전체 커버.

Usage:
    python tokenizer/train_sp_tokenizer.py \
        --input "data/raw/namuwiki_ko/*.txt,data/raw/ko_wiki_0000.txt" \
        --vocab_size 64000 \
        --output_dir tokenizer/korean_sp

Output:
    tokenizer/korean_sp/tokenizer.model   (SentencePiece 모델)
    tokenizer/korean_sp/tokenizer.vocab   (어휘 목록)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path


def expand_inputs(input_spec: str) -> list[str]:
    """콤마로 구분된 글로브 패턴들을 실제 파일 경로 목록으로 확장."""
    files: list[str] = []
    for pattern in input_spec.split(","):
        pattern = pattern.strip()
        if any(c in pattern for c in ("*", "?", "[")):
            matched = sorted(glob.glob(pattern, recursive=True))
            if not matched:
                print(f"WARNING: 패턴에 일치하는 파일 없음: {pattern!r}", file=sys.stderr)
            files.extend(matched)
        else:
            if Path(pattern).exists():
                files.append(pattern)
            else:
                print(f"WARNING: 파일 없음: {pattern!r}", file=sys.stderr)
    return files


def train(
    input_files: list[str],
    output_dir: Path,
    vocab_size: int,
    num_threads: int,
    input_sentence_size: int,
) -> None:
    try:
        import sentencepiece as spm
    except ImportError:
        print(
            "ERROR: sentencepiece가 설치되지 않음.\n"
            "  pip install --break-system-packages sentencepiece",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / "tokenizer")

    print(f"입력 파일 수: {len(input_files)}")
    for f in input_files[:5]:
        print(f"  {f}")
    if len(input_files) > 5:
        print(f"  ... 외 {len(input_files) - 5}개")
    print(f"어휘 크기: {vocab_size:,}")
    print(f"출력 경로: {model_prefix}.model / .vocab")
    print()

    # SentencePiece는 파일 목록을 콤마로 구분된 단일 문자열로 받는다
    input_str = ",".join(input_files)

    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",               # BPE보다 한국어에 자연스러움
        character_coverage=0.9995,           # 한글 11,172 음절 완전 커버
        normalization_rule_name="nfkc",      # Unicode NFKC 정규화 (한국어 호환문자 통일)
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        user_defined_symbols=[],
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        # 학습 안정성
        seed_sentencepiece_size=1_000_000,
        shrinking_factor=0.75,
        max_sentence_length=4096,
    )

    model_path = Path(f"{model_prefix}.model")
    vocab_path = Path(f"{model_prefix}.vocab")

    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print(f"학습 완료!")
        print(f"  모델: {model_path}  ({size_mb:.1f} MB)")
        print(f"  어휘: {vocab_path}")
        print()
        print("다음 단계:")
        print(f"  python tokenizer/convert_sp_to_hf.py \\")
        print(f"    --model {model_path} \\")
        print(f"    --output {output_dir}/tokenizer.json")
    else:
        print("ERROR: 학습 실패 — 출력 파일이 생성되지 않음", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentencePiece Unigram 한국어 토크나이저 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="콤마로 구분된 파일/글로브 패턴 (예: 'data/raw/ko/*.txt,data/raw/wiki.txt')",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=64000,
        help="어휘 크기",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tokenizer/korean_sp"),
        help="모델 저장 디렉토리",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=64,
        help="학습에 사용할 CPU 스레드 수",
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        default=10_000_000,
        help="학습에 사용할 최대 문장 수 (0 = 무제한)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = expand_inputs(args.input)
    if not input_files:
        print("ERROR: 입력 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)
    train(
        input_files=input_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentence_size,
    )


if __name__ == "__main__":
    main()

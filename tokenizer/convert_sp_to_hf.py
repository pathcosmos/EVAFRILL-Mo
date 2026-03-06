#!/usr/bin/env python3
"""
tokenizer/convert_sp_to_hf.py — SentencePiece 모델을 HuggingFace tokenizers.json으로 변환.

prepare.py의 load_tokenizer()는 Tokenizer.from_file()을 사용하므로
SentencePiece .model을 직접 읽지 못함 → HF tokenizers 포맷으로 변환 필요.

Usage:
    python tokenizer/convert_sp_to_hf.py \
        --model tokenizer/korean_sp/tokenizer.model \
        --output tokenizer/korean_sp/tokenizer.json

Requirements:
    pip install --break-system-packages sentencepiece tokenizers transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(model_path: Path, output_path: Path) -> None:
    """SentencePiece Unigram 모델을 HuggingFace tokenizers.json으로 변환."""

    # 방법 1: transformers의 XLNetTokenizer 계열 변환기 활용
    # (더 완전한 변환, special token 처리 포함)
    try:
        from transformers.convert_slow_tokenizer import SpmConverter
        from tokenizers import Tokenizer
        from tokenizers.models import Unigram

        print(f"변환 중: {model_path} → {output_path}")

        # SpmConverter는 tokenizers 라이브러리의 Unigram 모델로 변환
        # sentencepiece 모델 로드
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))

        vocab_size = sp.vocab_size()
        print(f"어휘 크기: {vocab_size:,}")

        # Unigram vocab 추출: (piece, score) 목록
        vocab: list[tuple[str, float]] = []
        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)
            vocab.append((piece, score))

        # HuggingFace Unigram 모델 생성
        # unk_id 확인
        unk_id = sp.unk_id()

        tokenizer = Tokenizer(Unigram(vocab, unk_id=unk_id))

        # Pre-tokenizer: Metaspace (SentencePiece 방식 — 공백을 ▁로 변환)
        # tokenizers >= 0.14: add_prefix_space → prepend_scheme='always'
        from tokenizers.pre_tokenizers import Metaspace
        tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")

        # Decoder: Metaspace (역변환)
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

        # Special token 설정 (SP 모델과 동일한 ID)
        from tokenizers import AddedToken
        pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
        bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
        eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2

        tokenizer.add_special_tokens([
            AddedToken("<pad>", special=True),
            AddedToken("<s>", special=True),
            AddedToken("</s>", special=True),
            AddedToken("<unk>", special=True),
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))

        # 저장 후 검증
        loaded = Tokenizer.from_file(str(output_path))
        test_text = "안녕하세요, 한국어 언어 모델입니다."
        encoded = loaded.encode(test_text)
        print(f"\n검증 통과:")
        print(f"  테스트 문자: {test_text!r}")
        print(f"  토큰 수: {len(encoded.ids)}")
        print(f"  토큰: {encoded.tokens[:15]}{'...' if len(encoded.tokens) > 15 else ''}")
        print(f"\n저장 완료: {output_path}")

    except ImportError as e:
        print(f"ERROR: 필요한 라이브러리 없음: {e}", file=sys.stderr)
        print("  pip install --break-system-packages sentencepiece tokenizers transformers", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 변환 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentencePiece 모델 → HuggingFace tokenizers.json 변환",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="SentencePiece .model 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="출력 tokenizers.json 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        print(f"ERROR: 모델 파일 없음: {args.model}", file=sys.stderr)
        sys.exit(1)
    convert(args.model, args.output)


if __name__ == "__main__":
    main()

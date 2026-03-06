"""
data/tokenize_extra.py — 대용량 korean_extra/ 데이터셋 병렬 토큰화

HuggingFace datasets disk 포맷(arrow), parquet, jsonl 등 세 가지 포맷을
자동 감지하여 SentencePiece 토크나이저로 토큰화하고, 결과를 uint16 memmap
(.bin) 파일로 저장한다.  881 GB 이상의 대용량 데이터도 스트리밍·청크 방식으로
처리한다.

출력 포맷은 data/dataset.py PackedDataset / TextDataset 과 완전히 호환되는
numpy uint16 플랫 배열이다.

사용 예시:
    # 단일 디렉토리
    python data/tokenize_extra.py \
        --input_dir data/korean_extra/fineweb2_edu_ko \
        --output    data/fineweb2_train.bin \
        --num_proc  8

    # korean_extra/ 전체 서브디렉토리 일괄 처리
    python data/tokenize_extra.py \
        --input_dir data/korean_extra \
        --auto_scan \
        --output_dir data \
        --num_proc 8

    # 공개 검증
    python -c "
    import numpy as np
    d = np.memmap('data/fineweb2_train.bin', dtype='uint16', mode='r')
    print(f'총 토큰: {len(d):,}')
    "
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path
from typing import Generator, Iterable, Iterator

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SentencePiece 임포트 (선택적 — 없으면 오류 메시지 출력 후 종료)
# ---------------------------------------------------------------------------
try:
    import sentencepiece as spm
except ImportError:
    print(
        "ERROR: sentencepiece 패키지가 설치되어 있지 않습니다.\n"
        "       pip install sentencepiece  로 설치 후 재실행하세요.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# datasets 임포트
# ---------------------------------------------------------------------------
try:
    import datasets as hf_datasets
except ImportError:
    print(
        "ERROR: datasets 패키지가 설치되어 있지 않습니다.\n"
        "       pip install datasets  로 설치 후 재실행하세요.",
        file=sys.stderr,
    )
    sys.exit(1)


# ===========================================================================
# 상수
# ===========================================================================

UINT16_MAX      = 65535           # uint16 오버플로 경계
MIN_TOKENS      = 100             # 최소 토큰 수 (미만이면 버림)
MAX_TOKENS      = 32_768          # 최대 토큰 수 (초과분은 버림)
HANGUL_RE_THRESHOLD = 0.10        # 한글 비율 최소 기준 (이 미만이고 한글 아닌 경우 버림)
CHUNK_TOKENS    = 500_000         # memmap 청크 단위 (tokens)
EOS_TOKEN_PLACEHOLDER = 1         # EOS id — SP 기본값, 실제 id는 모델에서 읽음

# ---------------------------------------------------------------------------
# 한글 비율 필터
# ---------------------------------------------------------------------------
# ord 범위: 가(AC00) ~ 힣(D7A3), ㄱ(3131) ~ ㅣ(3163)
_HANGUL_START = 0xAC00
_HANGUL_END   = 0xD7A3


def _has_enough_korean_or_english(text: str) -> bool:
    """
    한글 문자 비율이 HANGUL_RE_THRESHOLD 이상이거나,
    ASCII 알파벳 비율이 0.3 이상이면 True 반환.
    둘 다 아닌 경우 False (중국어, 일본어만 있는 등).
    """
    if not text:
        return False
    total = len(text)
    hangul_cnt = sum(1 for ch in text if _HANGUL_START <= ord(ch) <= _HANGUL_END)
    if hangul_cnt / total >= HANGUL_RE_THRESHOLD:
        return True
    ascii_alpha = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if ascii_alpha / total >= 0.30:
        return True
    return False


# ===========================================================================
# 토크나이저 래퍼 (프로세스 간 공유 불가 — 각 워커에서 reload)
# ===========================================================================

class SPTokenizer:
    """SentencePiece 모델을 wrapping한 간단한 토크나이저."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._sp: spm.SentencePieceProcessor | None = None

    # 프로세스 fork 후 _sp가 None인 경우 lazy load
    def _ensure_loaded(self) -> None:
        if self._sp is None:
            sp = spm.SentencePieceProcessor()
            sp.Load(self._model_path)
            self._sp = sp

    @property
    def eos_id(self) -> int:
        self._ensure_loaded()
        return self._sp.eos_id()

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return self._sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        return self._sp.EncodeAsIds(text)


# ===========================================================================
# 포맷 감지 & 이터레이터
# ===========================================================================

def _detect_format(input_dir: Path) -> str:
    """
    디렉토리 내용을 보고 포맷을 자동 감지한다.

    반환값:
        "hf_arrow"  — HuggingFace datasets disk 포맷 (dataset_info.json 존재)
        "parquet"   — .parquet 파일이 있음
        "jsonl"     — .jsonl 또는 .json 파일이 있음
        "unknown"   — 알 수 없음
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 경로가 디렉토리가 아닙니다: {input_dir}")

    # HF arrow 포맷 판별 — dataset_info.json 또는 state.json이 있으면 HF 포맷
    if (input_dir / "dataset_info.json").exists():
        return "hf_arrow"
    if (input_dir / "state.json").exists():
        return "hf_arrow"
    # 서브 디렉토리 안에 dataset_info.json이 있는 경우 (split 포함)
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "dataset_info.json").exists():
            return "hf_arrow"

    # parquet 파일 확인
    parquets = list(input_dir.rglob("*.parquet"))
    if parquets:
        return "parquet"

    # jsonl / json 파일 확인
    jsonls = list(input_dir.rglob("*.jsonl")) + list(input_dir.rglob("*.json"))
    if jsonls:
        return "jsonl"

    return "unknown"


def _iter_hf_arrow(
    input_dir: Path,
    text_col: str,
    num_proc: int,
) -> Iterator[str]:
    """HuggingFace datasets disk 포맷에서 텍스트를 스트리밍한다."""
    print(f"  [포맷] HuggingFace arrow (disk): {input_dir}")
    try:
        ds = hf_datasets.load_from_disk(str(input_dir))
    except Exception as exc:
        # DatasetDict일 수 있음 — 'train' split 시도
        try:
            ds_dict = hf_datasets.load_from_disk(str(input_dir))
            if isinstance(ds_dict, hf_datasets.DatasetDict):
                splits = list(ds_dict.keys())
                print(f"  DatasetDict 감지. splits={splits}, 'train' split 사용.")
                ds = ds_dict.get("train", ds_dict[splits[0]])
            else:
                raise exc
        except Exception:
            raise RuntimeError(
                f"HF arrow 포맷 로드 실패: {input_dir}\n원인: {exc}"
            ) from exc

    # 실제 텍스트 컬럼 이름 추정
    col = _resolve_text_col(list(ds.column_names), text_col)
    print(f"  텍스트 컬럼: '{col}', 총 행 수: {len(ds):,}")

    for row in ds:
        yield row[col]


def _iter_parquet(input_dir: Path, text_col: str) -> Iterator[str]:
    """parquet 파일에서 텍스트를 스트리밍한다."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        # datasets로 fallback
        print("  [경고] pyarrow 미설치, datasets로 parquet 로드 시도...")
        files = sorted(input_dir.rglob("*.parquet"))
        print(f"  [포맷] parquet ({len(files)} 파일): {input_dir}")
        ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"train": [str(f) for f in files]},
            split="train",
            streaming=True,
        )
        col = _resolve_text_col(list(ds.column_names), text_col)
        print(f"  텍스트 컬럼: '{col}'")
        for row in ds:
            yield row[col]
        return

    files = sorted(input_dir.rglob("*.parquet"))
    print(f"  [포맷] parquet ({len(files)} 파일): {input_dir}")
    for fpath in files:
        pf = pq.ParquetFile(str(fpath))
        cols = pf.schema_arrow.names
        col = _resolve_text_col(cols, text_col)
        for batch in pf.iter_batches(batch_size=1000, columns=[col]):
            for val in batch.column(col):
                yield val.as_py() or ""


def _iter_jsonl(input_dir: Path, text_col: str) -> Iterator[str]:
    """jsonl / json 파일에서 텍스트를 스트리밍한다."""
    files = sorted(input_dir.rglob("*.jsonl")) + sorted(input_dir.rglob("*.json"))
    # json 파일 중 jsonl이 아닌 것 제거 (파일 자체가 dict인 경우)
    print(f"  [포맷] jsonl ({len(files)} 파일): {input_dir}")
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, str):
                        yield obj
                    elif isinstance(obj, dict):
                        text = (
                            obj.get(text_col)
                            or obj.get("text")
                            or obj.get("content")
                            or obj.get("document")
                            or ""
                        )
                        yield str(text)
        except Exception as exc:
            print(f"  [경고] 파일 읽기 실패: {fpath} — {exc}", file=sys.stderr)


def _resolve_text_col(columns: list[str], preferred: str) -> str:
    """
    지정된 컬럼이 없을 경우, 일반적인 텍스트 컬럼 이름을 순서대로 탐색한다.
    """
    if preferred in columns:
        return preferred
    for candidate in ("text", "content", "document", "body", "passage"):
        if candidate in columns:
            print(
                f"  [INFO] 컬럼 '{preferred}' 미존재 → '{candidate}' 사용. "
                f"(전체 컬럼: {columns[:10]})"
            )
            return candidate
    # 마지막 수단: 첫 번째 문자열 컬럼
    print(
        f"  [경고] 텍스트 컬럼을 찾지 못함. 첫 번째 컬럼 '{columns[0]}' 사용.",
        file=sys.stderr,
    )
    return columns[0]


def get_text_iterator(
    input_dir: Path,
    text_col: str,
    num_proc: int,
) -> tuple[str, Iterator[str]]:
    """
    포맷을 자동 감지하고 알맞은 텍스트 이터레이터를 반환한다.

    Returns:
        (fmt, iterator)  fmt은 감지된 포맷 문자열
    """
    fmt = _detect_format(input_dir)
    if fmt == "hf_arrow":
        return fmt, _iter_hf_arrow(input_dir, text_col, num_proc)
    elif fmt == "parquet":
        return fmt, _iter_parquet(input_dir, text_col)
    elif fmt == "jsonl":
        return fmt, _iter_jsonl(input_dir, text_col)
    else:
        raise RuntimeError(
            f"지원하지 않는 포맷이거나 인식할 수 없습니다: {input_dir}\n"
            f"지원 포맷: HuggingFace arrow, parquet, jsonl"
        )


# ===========================================================================
# 단일 프로세스 토큰화 워커 (multiprocessing.Pool에서 호출)
# ===========================================================================

# 전역 토크나이저 — 각 워커 프로세스에서 한 번만 초기화
_g_sp: SPTokenizer | None = None
_g_model_path: str = ""


def _worker_init(model_path: str) -> None:
    """워커 초기화 함수: SentencePiece 모델 로드."""
    global _g_sp, _g_model_path
    _g_model_path = model_path
    _g_sp = SPTokenizer(model_path)
    _g_sp._ensure_loaded()


def _worker_tokenize_batch(texts: list[str]) -> list[list[int]]:
    """
    텍스트 배치를 토큰화하고 품질 필터를 적용한다.

    반환값: 유효한 토큰 리스트 목록 (필터 통과한 것만)
    """
    global _g_sp
    results: list[list[int]] = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        # 품질 필터: 언어
        if not _has_enough_korean_or_english(text):
            continue
        try:
            ids = _g_sp.encode(text)
        except Exception:
            continue
        # 길이 필터
        if len(ids) < MIN_TOKENS:
            continue
        if len(ids) > MAX_TOKENS:
            ids = ids[:MAX_TOKENS]
        results.append(ids)
    return results


# ===========================================================================
# memmap 청크 기반 기록기
# ===========================================================================

class MemmapWriter:
    """
    uint16 numpy memmap 파일에 토큰을 청크 단위로 기록하는 래퍼.

    초기에 작은 크기로 생성하고, 필요할 때 resize한다.
    최종적으로 실제 기록된 크기로 truncate하여 저장한다.
    """

    def __init__(self, path: Path, initial_size: int = CHUNK_TOKENS) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._alloc = max(initial_size, CHUNK_TOKENS)
        self._mm = np.memmap(
            str(path), dtype="uint16", mode="w+", shape=(self._alloc,)
        )
        self._pos = 0

    def write(self, tokens: Iterable[int]) -> int:
        """tokens를 기록하고 기록된 토큰 수를 반환한다."""
        arr = np.asarray(list(tokens), dtype=np.uint16)
        n = len(arr)
        if n == 0:
            return 0
        needed = self._pos + n
        if needed > self._alloc:
            # 두 배 또는 필요한 크기 중 큰 값으로 확장
            new_alloc = max(self._alloc * 2, needed + CHUNK_TOKENS)
            self._mm.flush()
            del self._mm
            self._alloc = new_alloc
            self._mm = np.memmap(
                str(self.path), dtype="uint16", mode="r+", shape=(self._alloc,)
            )
        self._mm[self._pos : self._pos + n] = arr
        self._pos += n
        return n

    def finalize(self) -> int:
        """기록된 실제 크기로 파일을 truncate하고 닫는다. 총 토큰 수를 반환한다."""
        self._mm.flush()
        del self._mm
        # 실제 기록된 크기로 truncate
        final_bytes = self._pos * 2  # uint16 = 2 bytes
        with open(str(self.path), "r+b") as fh:
            fh.truncate(final_bytes)
        return self._pos


# ===========================================================================
# 핵심 토큰화 파이프라인
# ===========================================================================

def tokenize_directory(
    input_dir: Path,
    output_path: Path,
    tokenizer_path: str,
    text_col: str = "text",
    num_proc: int = 8,
    batch_size: int = 512,
    eos_between_docs: bool = True,
    val_split: float = 0.002,
    seed: int = 42,
) -> dict:
    """
    단일 디렉토리를 토큰화하여 .bin 파일(들)로 저장한다.

    Args:
        input_dir:       입력 디렉토리 (포맷 자동 감지)
        output_path:     출력 .bin 파일 경로 (훈련 셋)
        tokenizer_path:  SentencePiece .model 파일 경로
        text_col:        텍스트 컬럼 이름 (arrow/parquet에서 사용)
        num_proc:        병렬 워커 수
        batch_size:      워커당 배치 크기
        eos_between_docs: 문서 사이에 EOS 토큰 삽입 여부
        val_split:       검증 분리 비율 (0 이면 val 파일 생성 안 함)
        seed:            재현성 시드

    Returns:
        통계 dict (total_tokens, train_tokens, val_tokens, skipped, elapsed_s)
    """
    t_start = time.time()

    # ─── 토크나이저 로드 (메인 프로세스: EOS id 확인) ─────────────────────
    sp_main = SPTokenizer(tokenizer_path)
    eos_id = sp_main.eos_id
    vocab_size = sp_main.vocab_size
    print(f"  토크나이저: {tokenizer_path}")
    print(f"  vocab_size={vocab_size:,}, eos_id={eos_id}")
    if vocab_size > UINT16_MAX:
        print(
            f"  [경고] vocab_size({vocab_size}) > {UINT16_MAX} "
            f"— uint16 오버플로 가능. 65535 이하 id만 안전.",
            file=sys.stderr,
        )

    # ─── 포맷 감지 & 이터레이터 생성 ─────────────────────────────────────
    fmt, text_iter = get_text_iterator(input_dir, text_col, num_proc)
    print(f"  포맷: {fmt}")

    # ─── 출력 경로 설정 ────────────────────────────────────────────────────
    train_path = output_path
    val_path: Path | None = None
    if val_split > 0:
        stem = output_path.stem
        if "train" in stem:
            val_path = output_path.parent / output_path.name.replace("train", "val")
        else:
            val_path = output_path.with_name(stem + "_val" + output_path.suffix)

    print(f"  출력(train): {train_path}")
    if val_path:
        print(f"  출력(val):   {val_path}")

    # ─── memmap 기록기 초기화 ─────────────────────────────────────────────
    writer = MemmapWriter(train_path)
    val_writer: MemmapWriter | None = MemmapWriter(val_path) if val_path else None

    # ─── multiprocessing Pool 생성 ────────────────────────────────────────
    pool = mp.Pool(
        processes=num_proc,
        initializer=_worker_init,
        initargs=(tokenizer_path,),
    )

    total_docs = 0
    skipped    = 0
    total_toks = 0

    # numpy rng for deterministic val split
    rng = np.random.default_rng(seed)

    def _submit_batch(batch_texts: list[str]) -> None:
        nonlocal total_docs, skipped, total_toks
        # 동기 map (배치 단위, 워커별 서브배치로 분할)
        sub_size = max(1, len(batch_texts) // num_proc)
        sub_batches = [
            batch_texts[i : i + sub_size]
            for i in range(0, len(batch_texts), sub_size)
        ]
        results_list = pool.map(_worker_tokenize_batch, sub_batches)

        for results in results_list:
            for ids in results:
                total_docs += 1
                n = len(ids)
                total_toks += n
                # EOS 토큰 삽입
                if eos_between_docs:
                    ids_out = ids + [eos_id]
                else:
                    ids_out = ids

                # val split: 무작위로 val_split 비율만큼 val 파일로
                if val_writer is not None and rng.random() < val_split:
                    val_writer.write(ids_out)
                else:
                    writer.write(ids_out)

            skipped_in_batch = sum(1 for _ in results) - len(results)

    # ─── 배치 수집 & tqdm 진행률 ─────────────────────────────────────────
    batch_buf: list[str] = []
    pbar = tqdm(desc=f"토큰화 [{input_dir.name}]", unit="doc", dynamic_ncols=True)

    for text in text_iter:
        batch_buf.append(text)
        if len(batch_buf) >= batch_size * num_proc:
            _submit_batch(batch_buf)
            pbar.update(len(batch_buf))
            pbar.set_postfix(
                tokens=f"{total_toks:,}",
                docs=f"{total_docs:,}",
                refresh=False,
            )
            batch_buf = []

    # 마지막 잔여 배치 처리
    if batch_buf:
        _submit_batch(batch_buf)
        pbar.update(len(batch_buf))

    pbar.close()
    pool.close()
    pool.join()

    # ─── 파일 마무리 ──────────────────────────────────────────────────────
    train_tokens = writer.finalize()
    val_tokens   = val_writer.finalize() if val_writer else 0

    elapsed = time.time() - t_start
    total_toks_with_eos = train_tokens + val_tokens

    print()
    print(f"  완료: {elapsed:.1f}초")
    print(f"  처리 문서: {total_docs:,}")
    print(f"  총 토큰(EOS 포함): {total_toks_with_eos:,}")
    print(f"    train: {train_tokens:,}  ({train_tokens*2/1e9:.2f} GB)")
    if val_tokens:
        print(f"    val:   {val_tokens:,}  ({val_tokens*2/1e9:.2f} GB)")
    throughput = total_toks_with_eos / elapsed if elapsed > 0 else 0
    print(f"  처리율: {throughput/1e6:.2f} M token/s")

    return {
        "total_docs"   : total_docs,
        "total_tokens" : total_toks_with_eos,
        "train_tokens" : train_tokens,
        "val_tokens"   : val_tokens,
        "elapsed_s"    : elapsed,
        "train_path"   : str(train_path),
        "val_path"     : str(val_path) if val_path else None,
    }


# ===========================================================================
# 서브디렉토리 자동 스캔 모드
# ===========================================================================

def auto_scan_and_tokenize(
    root_dir: Path,
    output_dir: Path,
    tokenizer_path: str,
    text_col: str,
    num_proc: int,
    batch_size: int,
    val_split: float,
    seed: int,
) -> list[dict]:
    """
    root_dir 의 직접 자식 디렉토리를 스캔하여 각각 토큰화한다.

    각 서브디렉토리에 대해:
        output_dir/korean_extra_{subdir_name}_train.bin 을 생성한다.
    """
    children = sorted(p for p in root_dir.iterdir() if p.is_dir())
    if not children:
        raise RuntimeError(f"서브디렉토리가 없습니다: {root_dir}")

    print(f"자동 스캔: {len(children)}개 서브디렉토리 발견")
    for ch in children:
        print(f"  - {ch.name}")
    print()

    all_stats = []
    for child in children:
        print("=" * 60)
        print(f"처리 중: {child}")
        print("=" * 60)
        safe_name = child.name.replace("/", "_").replace(" ", "_")
        out_name  = f"korean_extra_{safe_name}_train.bin"
        out_path  = output_dir / out_name
        try:
            stats = tokenize_directory(
                input_dir      = child,
                output_path    = out_path,
                tokenizer_path = tokenizer_path,
                text_col       = text_col,
                num_proc       = num_proc,
                batch_size     = batch_size,
                val_split      = val_split,
                seed           = seed,
            )
            stats["source"] = child.name
            all_stats.append(stats)
        except Exception as exc:
            print(f"  [오류] {child.name} 처리 실패: {exc}", file=sys.stderr)
            all_stats.append({"source": child.name, "error": str(exc)})
        print()

    return all_stats


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "korean_extra/ 대용량 데이터셋을 병렬 토큰화하여 uint16 memmap(.bin) 로 저장. "
            "HuggingFace arrow, parquet, jsonl 포맷 자동 감지."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 입력
    parser.add_argument(
        "--input_dir",
        required=True,
        help="토큰화할 디렉토리 경로. --auto_scan 시에는 루트 디렉토리.",
    )
    parser.add_argument(
        "--auto_scan",
        action="store_true",
        help=(
            "input_dir 의 직접 자식 디렉토리를 모두 순차 처리. "
            "이 경우 --output_dir 을 지정해야 함."
        ),
    )
    parser.add_argument(
        "--text_col",
        default="text",
        help="텍스트 컬럼 이름 (arrow/parquet/jsonl). 자동 추정 가능.",
    )

    # 출력
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--output",
        default=None,
        help="출력 .bin 파일 경로 (단일 디렉토리 처리 시 사용).",
    )
    out_group.add_argument(
        "--output_dir",
        default=None,
        help="출력 .bin 파일들을 저장할 디렉토리 (--auto_scan 시 사용).",
    )

    # 토크나이저
    parser.add_argument(
        "--tokenizer",
        default=(
            "/PROJECT/0325120031_A/ghong/taketimes/llm-bang"
            "/tokenizer/korean_64k.model"
        ),
        help="SentencePiece .model 파일 경로.",
    )

    # 처리 옵션
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="병렬 워커 수 (multiprocessing.Pool).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="워커당 배치 크기 (문서 수).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.002,
        help="검증 분리 비율 (0.0 이면 val 파일 미생성).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성 시드.",
    )
    parser.add_argument(
        "--no_eos",
        action="store_true",
        help="문서 사이에 EOS 토큰을 삽입하지 않는다.",
    )

    args = parser.parse_args()

    # 검증
    if not args.auto_scan and args.output is None:
        # 자동 출력 경로 생성
        input_name = Path(args.input_dir).name
        args.output = str(
            Path(args.input_dir).parent.parent
            / f"korean_extra_{input_name}_train.bin"
        )
        print(f"[INFO] --output 미지정 → 자동 설정: {args.output}")

    if args.auto_scan and args.output_dir is None:
        parser.error("--auto_scan 사용 시 --output_dir 을 지정해야 합니다.")

    return args


def main() -> None:
    args = parse_args()

    tokenizer_path = args.tokenizer
    if not Path(tokenizer_path).exists():
        # fallback: 상대경로 시도
        fallback = Path(
            "/PROJECT/0325120031_A/ghong/taketimes/llm-bang"
            "/tokenizer/korean_64k.model"
        )
        if fallback.exists():
            tokenizer_path = str(fallback)
        else:
            print(
                f"ERROR: 토크나이저 파일을 찾을 수 없습니다: {tokenizer_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=" * 60)
    print(" LLM-Bang tokenize_extra.py")
    print("=" * 60)
    print(f"  입력:        {args.input_dir}")
    print(f"  토크나이저:  {tokenizer_path}")
    print(f"  num_proc:    {args.num_proc}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  val_split:   {args.val_split}")
    print(f"  seed:        {args.seed}")
    print(f"  eos:         {not args.no_eos}")
    print()

    if args.auto_scan:
        stats_list = auto_scan_and_tokenize(
            root_dir       = Path(args.input_dir),
            output_dir     = Path(args.output_dir),
            tokenizer_path = tokenizer_path,
            text_col       = args.text_col,
            num_proc       = args.num_proc,
            batch_size     = args.batch_size,
            val_split      = args.val_split,
            seed           = args.seed,
        )
        print("=" * 60)
        print(" 전체 요약")
        print("=" * 60)
        grand_train = 0
        grand_val   = 0
        for s in stats_list:
            if "error" in s:
                print(f"  {s['source']:40s} ERROR: {s['error']}")
            else:
                t = s.get("train_tokens", 0)
                v = s.get("val_tokens", 0)
                grand_train += t
                grand_val   += v
                print(
                    f"  {s['source']:40s} "
                    f"train={t:>14,}  val={v:>12,}  "
                    f"({s['elapsed_s']:.0f}s)"
                )
        print("-" * 60)
        print(
            f"  {'합계':40s} "
            f"train={grand_train:>14,}  val={grand_val:>12,}"
        )
        print(
            f"\n  총 토큰: {grand_train + grand_val:,}  "
            f"({(grand_train + grand_val) * 2 / 1e9:.2f} GB)"
        )

    else:
        stats = tokenize_directory(
            input_dir      = Path(args.input_dir),
            output_path    = Path(args.output),
            tokenizer_path = tokenizer_path,
            text_col       = args.text_col,
            num_proc       = args.num_proc,
            batch_size     = args.batch_size,
            eos_between_docs = not args.no_eos,
            val_split      = args.val_split,
            seed           = args.seed,
        )
        print()
        print("=" * 60)
        print(" 결과 요약")
        print("=" * 60)
        print(f"  train .bin : {stats['train_path']}")
        if stats.get("val_path"):
            print(f"  val .bin   : {stats['val_path']}")
        print(f"  train 토큰 : {stats['train_tokens']:,}")
        print(f"  val 토큰   : {stats['val_tokens']:,}")
        print(f"  처리 문서  : {stats['total_docs']:,}")
        print(f"  소요 시간  : {stats['elapsed_s']:.1f}초")

        # 검증: memmap 로드 테스트
        print()
        print("  [검증] memmap 로드 테스트...")
        try:
            d = np.memmap(stats["train_path"], dtype="uint16", mode="r")
            print(f"  memmap shape: {d.shape}  dtype: {d.dtype}")
            print(f"  첫 10 토큰: {d[:10].tolist()}")
        except Exception as exc:
            print(f"  [경고] memmap 로드 실패: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

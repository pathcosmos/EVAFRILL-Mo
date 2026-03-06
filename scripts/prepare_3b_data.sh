#!/usr/bin/env bash
# =============================================================================
# prepare_3b_data.sh — 3B 모델 학습 데이터 전체 파이프라인
#
# 사용법:
#   bash scripts/prepare_3b_data.sh [--step N] [--jobs 72]
#
# 스텝:
#   1 = CulturaX 토큰화
#   2 = cc100 해제 + 토큰화
#   3 = OSCAR 토큰화
#   4 = korean_webtext 토큰화
#   5 = HPLT 한국어 추출 + 토큰화
#   6 = textbooks + finepdfs + kovast 토큰화
#   7 = 전체 병합
#   8 = train/val split 검증
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ─── 설정 ────────────────────────────────────────────────────────────────
DATA_DIR="data"
EXTRA_DIR="data/korean_extra"
TOKENIZER="tokenizer/tokenizer.json"
VAL_SPLIT=0.002
SEED=42
JOBS=72
FROM_STEP=0
LOG_FILE="data/prepare_3b.log"

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)   FROM_STEP="$2"; shift 2 ;;
        --jobs)   JOBS="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── 토큰화 헬퍼 (parquet → bin) ─────────────────────────────────────────
tokenize_parquet() {
    local name="$1"
    local input_pattern="$2"
    local text_col="$3"
    local output="${DATA_DIR}/${name}_train.bin"

    if [[ -f "$output" && $FROM_STEP -le 0 ]]; then
        log "[SKIP] $output already exists ($(du -h "$output" | cut -f1))"
        return
    fi

    log "[START] Tokenizing $name from parquet..."
    python3 - <<PYEOF
import glob, os, sys
import numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

tokenizer_path = "${TOKENIZER}"
input_pattern = "${input_pattern}"
text_col = "${text_col}"
output_train = "${output}"
output_val = output_train.replace("_train.bin", "_val.bin")
val_split = ${VAL_SPLIT}
seed = ${SEED}

files = sorted(glob.glob(input_pattern))
print(f"Found {len(files)} parquet files")

tokenizer = Tokenizer.from_file(tokenizer_path)

all_tokens = []
total_docs = 0

for f in tqdm(files, desc="${name}"):
    try:
        table = pq.read_table(f, columns=[text_col])
        for text in table.column(text_col):
            t = text.as_py()
            if t and len(t) > 50:
                ids = tokenizer.encode(t).ids
                all_tokens.extend(ids)
                total_docs += 1
    except Exception as e:
        print(f"Error processing {f}: {e}", file=sys.stderr)
        continue

print(f"Total: {total_docs:,} docs, {len(all_tokens):,} tokens")

# Split
import random
random.seed(seed)
random.shuffle(all_tokens)  # Not ideal but matches existing code
n_val = int(len(all_tokens) * val_split)
val_tokens = all_tokens[:n_val]
train_tokens = all_tokens[n_val:]

np.array(train_tokens, dtype=np.uint16).tofile(output_train)
np.array(val_tokens, dtype=np.uint16).tofile(output_val)
print(f"Saved: {output_train} ({len(train_tokens):,} tokens)")
print(f"Saved: {output_val} ({len(val_tokens):,} tokens)")
PYEOF
    log "[DONE] $name → $output"
}

# ─── Step 1: CulturaX ────────────────────────────────────────────────────
if [[ $FROM_STEP -le 1 ]]; then
    log "=== Step 1: CulturaX 토큰화 ==="
    tokenize_parquet "culturax" \
        "${EXTRA_DIR}/culturax_ko/ko/*.parquet" \
        "text"
fi

# ─── Step 2: cc100 해제 + 토큰화 ─────────────────────────────────────────
if [[ $FROM_STEP -le 2 ]]; then
    log "=== Step 2: cc100 해제 + 토큰화 ==="
    CC100_XZ="${EXTRA_DIR}/cc100_ko/ko.txt.xz"
    CC100_TXT="${EXTRA_DIR}/cc100_ko/ko.txt"
    CC100_OUT="${DATA_DIR}/cc100_train.bin"

    if [[ -f "$CC100_OUT" && $FROM_STEP -le 0 ]]; then
        log "[SKIP] cc100 already tokenized"
    else
        # 해제
        if [[ ! -f "$CC100_TXT" ]]; then
            log "Decompressing cc100 xz (14GB → 54GB)..."
            xz -dk "$CC100_XZ"
            log "Decompression done"
        fi

        # 토큰화 (대용량 → 스트리밍)
        log "Tokenizing cc100 (54GB text)..."
        python3 - <<'PYEOF'
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import random

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
input_file = "data/korean_extra/cc100_ko/ko.txt"
output_train = "data/cc100_train.bin"
output_val = "data/cc100_val.bin"

# Stream tokenize in chunks
all_tokens = []
doc_buffer = []
doc_count = 0

with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
    for line in tqdm(f, desc="cc100", unit=" lines"):
        line = line.strip()
        if not line:
            # Document boundary
            if doc_buffer:
                text = '\n'.join(doc_buffer)
                if len(text) > 50:
                    ids = tokenizer.encode(text).ids
                    all_tokens.extend(ids)
                    doc_count += 1
                doc_buffer = []
        else:
            doc_buffer.append(line)

    # Last doc
    if doc_buffer:
        text = '\n'.join(doc_buffer)
        if len(text) > 50:
            all_tokens.extend(tokenizer.encode(text).ids)
            doc_count += 1

print(f"Total: {doc_count:,} docs, {len(all_tokens):,} tokens")

# Split
n_val = int(len(all_tokens) * 0.002)
np.array(all_tokens[n_val:], dtype=np.uint16).tofile(output_train)
np.array(all_tokens[:n_val], dtype=np.uint16).tofile(output_val)
print(f"Saved train: {len(all_tokens)-n_val:,} tokens")
print(f"Saved val: {n_val:,} tokens")
PYEOF
        log "[DONE] cc100"
    fi
fi

# ─── Step 3: OSCAR ───────────────────────────────────────────────────────
if [[ $FROM_STEP -le 3 ]]; then
    log "=== Step 3: OSCAR 토큰화 ==="
    OSCAR_OUT="${DATA_DIR}/oscar_train.bin"

    if [[ -f "$OSCAR_OUT" && $FROM_STEP -le 0 ]]; then
        log "[SKIP] OSCAR already tokenized"
    else
        python3 - <<'PYEOF'
import glob, numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
files = sorted(glob.glob("data/korean_extra/oscar_ko/data/kor_Hang/*.parquet"))
all_tokens = []
doc_count = 0

for f in tqdm(files, desc="OSCAR"):
    table = pq.read_table(f, columns=['text'])
    for row in table.column('text'):
        if row is None:
            continue
        parts = row.as_py()
        if parts:
            text = '\n'.join(item['text'] for item in parts if item and item.get('text'))
            if len(text) > 50:
                all_tokens.extend(tokenizer.encode(text).ids)
                doc_count += 1

print(f"OSCAR: {doc_count:,} docs, {len(all_tokens):,} tokens")
n_val = int(len(all_tokens) * 0.002)
np.array(all_tokens[n_val:], dtype=np.uint16).tofile("data/oscar_train.bin")
np.array(all_tokens[:n_val], dtype=np.uint16).tofile("data/oscar_val.bin")
PYEOF
        log "[DONE] OSCAR"
    fi
fi

# ─── Step 4: korean_webtext ──────────────────────────────────────────────
if [[ $FROM_STEP -le 4 ]]; then
    log "=== Step 4: korean_webtext 토큰화 ==="
    tokenize_parquet "webtext" \
        "${EXTRA_DIR}/korean_webtext/data/*.parquet" \
        "text"
fi

# ─── Step 5: HPLT 한국어 추출 + 토큰화 ──────────────────────────────────
if [[ $FROM_STEP -le 5 ]]; then
    log "=== Step 5: HPLT 한국어 추출 + 토큰화 ==="
    HPLT_OUT="${DATA_DIR}/hplt_ko_train.bin"

    if [[ -f "$HPLT_OUT" && $FROM_STEP -le 0 ]]; then
        log "[SKIP] HPLT already tokenized"
    else
        python3 - <<'PYEOF'
import glob, numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
files = sorted(glob.glob("data/korean_extra/hplt_ko/en-ko/*.parquet"))
all_tokens = []
doc_count = 0

for f in tqdm(files, desc="HPLT"):
    table = pq.read_table(f, columns=['tgt_doc'])
    for row in table.column('tgt_doc'):
        d = row.as_py()
        if d and d.get('sentences'):
            text = '\n'.join(s for s in d['sentences'] if s)
            if len(text) > 50:
                all_tokens.extend(tokenizer.encode(text).ids)
                doc_count += 1

print(f"HPLT Korean: {doc_count:,} docs, {len(all_tokens):,} tokens")
n_val = int(len(all_tokens) * 0.002)
np.array(all_tokens[n_val:], dtype=np.uint16).tofile("data/hplt_ko_train.bin")
np.array(all_tokens[:n_val], dtype=np.uint16).tofile("data/hplt_ko_val.bin")
PYEOF
        log "[DONE] HPLT"
    fi
fi

# ─── Step 6: textbooks + finepdfs + kovast ───────────────────────────────
if [[ $FROM_STEP -le 6 ]]; then
    log "=== Step 6: 기타 소스 토큰화 ==="
    EXTRA_OUT="${DATA_DIR}/extra_misc_train.bin"

    if [[ -f "$EXTRA_OUT" && $FROM_STEP -le 0 ]]; then
        log "[SKIP] extra_misc already tokenized"
    else
        python3 - <<'PYEOF'
import glob, numpy as np, os
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
all_tokens = []
doc_count = 0

# korean_textbooks (MMLU-style: look for text columns)
tb_files = glob.glob("data/korean_extra/korean_textbooks/**/*.parquet", recursive=True)
for f in tqdm(tb_files, desc="textbooks"):
    try:
        table = pq.read_table(f)
        # Try common text columns
        for col in ['question', 'text', 'input', 'instruction']:
            if col in table.column_names:
                for val in table.column(col):
                    t = val.as_py()
                    if t and len(t) > 20:
                        all_tokens.extend(tokenizer.encode(t).ids)
                        doc_count += 1
                break
    except:
        continue

# finepdfs
pdf_files = glob.glob("data/korean_extra/finepdfs_edu_ko/*.parquet")
for f in tqdm(pdf_files, desc="finepdfs"):
    try:
        table = pq.read_table(f)
        for col in ['text', 'content']:
            if col in table.column_names:
                for val in table.column(col):
                    t = val.as_py()
                    if t and len(t) > 50:
                        all_tokens.extend(tokenizer.encode(t).ids)
                        doc_count += 1
                break
    except:
        continue

print(f"Extra: {doc_count:,} docs, {len(all_tokens):,} tokens")
n_val = int(len(all_tokens) * 0.002)
np.array(all_tokens[n_val:], dtype=np.uint16).tofile("data/extra_misc_train.bin")
np.array(all_tokens[:n_val], dtype=np.uint16).tofile("data/extra_misc_val.bin")
PYEOF
        log "[DONE] extra_misc"
    fi
fi

# ─── Step 7: 전체 병합 ──────────────────────────────────────────────────
if [[ $FROM_STEP -le 7 ]]; then
    log "=== Step 7: 전체 병합 ==="

    TRAIN_BINS=""
    for f in \
        "${DATA_DIR}/korean_train.bin" \
        "${DATA_DIR}/culturax_train.bin" \
        "${DATA_DIR}/cc100_train.bin" \
        "${DATA_DIR}/oscar_train.bin" \
        "${DATA_DIR}/webtext_train.bin" \
        "${DATA_DIR}/hplt_ko_train.bin" \
        "${DATA_DIR}/extra_misc_train.bin"; do
        if [[ -f "$f" ]]; then
            TRAIN_BINS="$TRAIN_BINS $f"
            log "  Including: $f ($(du -h "$f" | cut -f1))"
        else
            log "  [WARN] Missing: $f"
        fi
    done

    if [[ -n "$TRAIN_BINS" ]]; then
        python3 data/merge_bins.py $TRAIN_BINS "${DATA_DIR}/merged_3b_train.bin"
        log "[DONE] merged_3b_train.bin created"
    fi

    # Val 병합
    VAL_BINS=""
    for f in \
        "${DATA_DIR}/korean_val.bin" \
        "${DATA_DIR}/culturax_val.bin" \
        "${DATA_DIR}/cc100_val.bin" \
        "${DATA_DIR}/oscar_val.bin" \
        "${DATA_DIR}/webtext_val.bin" \
        "${DATA_DIR}/hplt_ko_val.bin" \
        "${DATA_DIR}/extra_misc_val.bin"; do
        if [[ -f "$f" ]]; then
            VAL_BINS="$VAL_BINS $f"
        fi
    done

    if [[ -n "$VAL_BINS" ]]; then
        python3 data/merge_bins.py $VAL_BINS "${DATA_DIR}/merged_3b_val.bin"
        log "[DONE] merged_3b_val.bin created"
    fi
fi

# ─── Step 8: 검증 ────────────────────────────────────────────────────────
if [[ $FROM_STEP -le 8 ]]; then
    log "=== Step 8: 최종 검증 ==="
    python3 - <<'PYEOF'
import os, glob
import numpy as np

print("=== 토큰화 결과 ===")
total_train = 0
total_val = 0
for f in sorted(glob.glob("data/*_train.bin") + glob.glob("data/train.bin")):
    n = os.path.getsize(f) // 2
    total_train += n
    print(f"  {os.path.basename(f):30s}: {n:>15,} tokens ({os.path.getsize(f)/1e9:.2f} GB)")

for f in sorted(glob.glob("data/*_val.bin") + glob.glob("data/val.bin")):
    n = os.path.getsize(f) // 2
    total_val += n

print(f"\n  Total train: {total_train:,} tokens ({total_train/1e9:.1f}B)")
print(f"  Total val:   {total_val:,} tokens ({total_val/1e6:.1f}M)")
print(f"\n  3B Chinchilla minimum: 60B tokens")
print(f"  Epochs needed for 60B: {60e9/total_train:.1f}")
print(f"  Epochs needed for 100B: {100e9/total_train:.1f}")
PYEOF
fi

log "=== 파이프라인 완료 ==="

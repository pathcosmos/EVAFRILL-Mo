#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="data"

echo "=================================================================="
echo "  3B 통합 데이터셋 빌드  |  시작: $(date)"
echo "=================================================================="

# 청크 병합 함수
merge_chunks() {
    PREFIX="$1"
    OUTPUT="$2"
    CHUNKS=$(ls "${PREFIX}".bin.chunk* 2>/dev/null | sort || true)
    if [[ -z "$CHUNKS" ]]; then return; fi
    if [[ -f "$OUTPUT" ]]; then echo "  [SKIP] $OUTPUT 이미 존재"; return; fi
    echo "  청크 병합: $(basename $PREFIX)"
    cat $CHUNKS > "$OUTPUT"
    echo "  완료: $(du -sh $OUTPUT | cut -f1)"
}

merge_chunks "$DATA/cosmo_auto_math_text_train" "$DATA/cosmo_auto_math_text_train.bin"
merge_chunks "$DATA/cosmo_auto_math_text_val"   "$DATA/cosmo_auto_math_text_val.bin"
merge_chunks "$DATA/cosmo_web_v2_train"         "$DATA/cosmo_web_v2_train.bin"
merge_chunks "$DATA/cosmo_web_v2_val"           "$DATA/cosmo_web_v2_val.bin"

TRAIN_FILES=""
for f in \
    "$DATA/korean_train.bin" \
    "$DATA/hplt_ko_train.bin" \
    "$DATA/korean_c4_train.bin" \
    "$DATA/cc100_ko_train.bin" \
    "$DATA/namuwiki_2023b_train.bin" \
    "$DATA/korean_namuwiki_train.bin" \
    "$DATA/wikipedia_ko_train.bin" \
    "$DATA/korean_wiki_train.bin" \
    "$DATA/open_web_math_train.bin" \
    "$DATA/mathpile_train.bin" \
    "$DATA/cosmo_auto_math_text_train.bin" \
    "$DATA/cosmo_stories_train.bin" \
    "$DATA/cosmo_web_v2_train.bin" \
    "$DATA/cosmo_stanford_train.bin" \
    "$DATA/cosmo_wikihow_train.bin" \
    "$DATA/cosmo_openstax_train.bin" \
    "$DATA/cosmo_khanacademy_train.bin"; do
    [[ -f "$f" ]] && TRAIN_FILES="$TRAIN_FILES $f"
done

VAL_FILES=""
for f in \
    "$DATA/korean_val.bin" \
    "$DATA/hplt_ko_val.bin" \
    "$DATA/korean_c4_val.bin" \
    "$DATA/cc100_ko_val.bin" \
    "$DATA/namuwiki_2023b_val.bin" \
    "$DATA/open_web_math_val.bin" \
    "$DATA/mathpile_val.bin" \
    "$DATA/cosmo_auto_math_text_val.bin" \
    "$DATA/cosmo_stories_val.bin" \
    "$DATA/cosmo_web_v2_val.bin"; do
    [[ -f "$f" ]] && VAL_FILES="$VAL_FILES $f"
done

echo ""
echo "train 파일 병합 → data/3b_train.bin ..."
python3 data/merge_bins.py $TRAIN_FILES data/3b_train.bin

echo ""
echo "val 파일 병합 → data/3b_val.bin ..."
python3 data/merge_bins.py $VAL_FILES data/3b_val.bin

echo ""
echo "=================================================================="
du -sh data/3b_train.bin data/3b_val.bin
python3 -c "
import os
sz = os.path.getsize('data/3b_train.bin')
tok = sz // 2
print(f'3b_train: {tok/1e9:.2f}B tokens')
print(f'60B 달성 에포크: {60/(tok/1e9):.1f}x 반복 필요')
"
echo "완료: $(date)"
echo "=================================================================="

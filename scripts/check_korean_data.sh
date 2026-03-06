#!/bin/bash

# 한국어 학습 데이터 현황 확인 스크립트
# 용도: 한국어 데이터셋 상태, 토크나이저, 원본 데이터 파일 확인

set -e

# 프로젝트 루트 (이 스크립트 실행 위치 기준)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== 한국어 학습 데이터 현황 ==="
echo ""

# ============================================================================
# 1. 학습용 바이너리 데이터 확인
# ============================================================================
echo "[ 학습 바이너리 데이터 ]"

check_binary_data() {
    local file=$1
    local name=$2

    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)

        # Python + numpy memmap으로 토큰 수 계산
        # 바이너리는 uint32 형태로 저장되어 있음 (4 bytes per token)
        local token_count=$(python3 -c "
import numpy as np
try:
    data = np.memmap('$file', dtype=np.uint32, mode='r')
    print(len(data))
except Exception as e:
    print('error')
" 2>/dev/null || echo "error")

        if [ "$token_count" != "error" ] && [ ! -z "$token_count" ]; then
            # 토큰 수를 포맷팅 (천 단위 쉼표)
            local formatted_tokens=$(printf "%'d" "$token_count")

            # 1B 모델 학습 스텝 계산
            # tokens_per_step = batch_size * grad_accum * seq_len * num_gpus
            #                 = 8 * 4 * 4096 * 8 = 1,048,576 tokens/step
            local tokens_per_step=1048576
            local estimated_steps=$((token_count / tokens_per_step))

            printf "  %-20s : 존재 (%s, %'d 토큰, ~%'d steps)\n" \
                "$name" "$size" "$token_count" "$estimated_steps"
        else
            printf "  %-20s : 존재 (%s, 토큰 계산 실패)\n" "$name" "$size"
        fi
    else
        printf "  %-20s : 없음\n" "$name"
    fi
}

check_binary_data "data/korean_train.bin" "korean_train.bin"
check_binary_data "data/korean_val.bin" "korean_val.bin"
check_binary_data "data/train.bin" "train.bin"
check_binary_data "data/val.bin" "val.bin"

echo ""

# ============================================================================
# 2. 토크나이저 확인
# ============================================================================
echo "[ 토크나이저 ]"

check_tokenizer() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        local files=$(find "$dir" -type f | wc -l)
        printf "  %-20s : 존재 (%d개 파일)\n" "$name" "$files"
    else
        printf "  %-20s : 없음\n" "$name"
    fi
}

check_tokenizer "tokenizer/korean_sp" "korean_sp"
check_tokenizer "tokenizer" "default tokenizer"

echo ""

# ============================================================================
# 3. 원본 데이터 디렉토리 확인
# ============================================================================
echo "[ 원본 데이터 ]"

check_raw_data() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        local file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
        local total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)

        if [ $file_count -eq 0 ]; then
            printf "  %-20s : 없음 (디렉토리만 존재, 0 파일)\n" "$name"
        else
            printf "  %-20s : %'d 파일 (%s)\n" "$name" "$file_count" "$total_size"
        fi
    else
        printf "  %-20s : 없음\n" "$name"
    fi
}

check_raw_data "data/raw/cc100_ko" "cc100_ko/"
check_raw_data "data/raw/c4_ko" "c4_ko/"
check_raw_data "data/raw/namuwiki_ko" "namuwiki_ko/"

# 위키 데이터는 raw/ 직접 하위
echo ""
echo "[ 위키피디아 데이터 ]"
ko_wiki_count=$(find "data/raw" -maxdepth 1 -name "ko_wiki_*.txt" | wc -l)
en_wiki_count=$(find "data/raw" -maxdepth 1 -name "en_wiki_*.txt" | wc -l)
ko_wiki_size=$(du -sh "data/raw" 2>/dev/null | cut -f1)

if [ $ko_wiki_count -gt 0 ]; then
    printf "  %-20s : %'d 파일\n" "ko_wiki" "$ko_wiki_count"
fi

if [ $en_wiki_count -gt 0 ]; then
    printf "  %-20s : %'d 파일\n" "en_wiki" "$en_wiki_count"
fi

echo ""

# ============================================================================
# 4. 종합 상태 요약
# ============================================================================
echo "[ 종합 상태 ]"

# 학습용 바이너리 데이터 확인
binary_ready=false
if [ -f "data/korean_train.bin" ] && [ -f "data/korean_val.bin" ]; then
    binary_ready=true
elif [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    binary_ready=true
fi

# 토크나이저 확인
tokenizer_ready=false
if [ -d "tokenizer/korean_sp" ] && [ -f "tokenizer/korean_sp/tokenizer.model" ]; then
    tokenizer_ready=true
fi

# 원본 데이터 확인
raw_ready=false
if [ -d "data/raw/c4_ko" ] || [ -d "data/raw/namuwiki_ko" ] || [ -d "data/raw/cc100_ko" ]; then
    count=$(find "data/raw/c4_ko" -maxdepth 1 -type f 2>/dev/null | wc -l)
    count=$((count + $(find "data/raw/namuwiki_ko" -maxdepth 1 -type f 2>/dev/null | wc -l)))
    count=$((count + $(find "data/raw/cc100_ko" -maxdepth 1 -type f 2>/dev/null | wc -l)))
    if [ $count -gt 0 ]; then
        raw_ready=true
    fi
fi

printf "  학습용 바이너리     : %s\n" "$([ "$binary_ready" = true ] && echo "✓ 준비됨" || echo "✗ 미준비")"
printf "  토크나이저          : %s\n" "$([ "$tokenizer_ready" = true ] && echo "✓ 준비됨" || echo "✗ 미준비")"
printf "  원본 데이터         : %s\n" "$([ "$raw_ready" = true ] && echo "✓ 준비됨" || echo "✗ 미준비")"

echo ""

# ============================================================================
# 5. 학습 설정 파라미터 정보
# ============================================================================
echo "[ 학습 설정 (1B 모델 기준) ]"
echo "  배치 사이즈         : 8"
echo "  시퀀스 길이         : 4096"
echo "  GPU 수              : 8"
echo "  그래디언트 누적     : 4"
echo "  토큰/스텝           : 8 × 4 × 4096 × 8 = 1,048,576"
echo ""

echo "=== 검사 완료 ==="

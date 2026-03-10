#!/usr/bin/env bash
# EVAFRILL-Mo 완전 백업 스크립트 (단일-tier 통합)
# 다른 기기에서 SFT를 그대로 재현할 수 있는 수준의 완전 백업
#
# 사용법: bash scripts/backup_evafrill_full.sh [날짜]
#   날짜: 백업 폴더명 (기본: 오늘 YYYYMMDD)
#
# 포함 대상 (~317GB raw):
#   checkpoints/3b_final (pretrain), checkpoints/3b_sft (SFT),
#   data/3b_train.bin, data/3b_val.bin, data/sft_combined, data/preference,
#   data/sft, data/*.py, data/*.sh (파이프라인 코드),
#   train/, model/, configs/, scripts/, eval/, docs/, reports/,
#   tokenizer/, benchmarks/, logs/, root files + git bundle
#
# 제외:
#   llm-bang-archive.tar.zst, checkpoints/1b_final, data/korean_extra,
#   data/sft_extra, data/raw, data/math, data/code, data/translation,
#   개별 *.bin (3b_train/val 제외), __pycache__, .git 등

set -euo pipefail

PROJECT_ROOT="/PROJECT/0325120031_A/ghong/taketimes"
LLM_STAR="${PROJECT_ROOT}/llm-star"
BACKUP_BASE="${PROJECT_ROOT}/llm-star-backup"
DATE_FOLDER="${1:-$(date +%Y%m%d)}"
BACKUP_DIR="${BACKUP_BASE}/${DATE_FOLDER}"
CHUNK_SIZE="990M"
ARCHIVE_NAME="evafrill-full"

mkdir -p "${BACKUP_DIR}"
cd "${PROJECT_ROOT}"

# ─────────────────────────────────────────────
# 제외 패턴
# ─────────────────────────────────────────────
EXCLUDE_COMMON=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='tensorboard'
    --exclude='nohup.out'
    --exclude='.git'
)

# ─────────────────────────────────────────────
# 백업 대상 수집
# ─────────────────────────────────────────────
collect_includes() {
    INCLUDES=()

    # 체크포인트 (pretrain + SFT)
    [ -d llm-star/checkpoints/3b_final ] && INCLUDES+=(llm-star/checkpoints/3b_final)
    [ -d llm-star/checkpoints/3b_sft ]   && INCLUDES+=(llm-star/checkpoints/3b_sft)

    # 학습 데이터
    [ -f llm-star/data/3b_train.bin ]    && INCLUDES+=(llm-star/data/3b_train.bin)
    [ -f llm-star/data/3b_val.bin ]      && INCLUDES+=(llm-star/data/3b_val.bin)
    [ -d llm-star/data/sft_combined ]    && INCLUDES+=(llm-star/data/sft_combined)
    [ -d llm-star/data/preference ]      && INCLUDES+=(llm-star/data/preference)
    [ -d llm-star/data/sft ]             && INCLUDES+=(llm-star/data/sft)

    # 데이터 파이프라인 코드 (data/*.py, data/*.sh)
    for f in llm-star/data/*.py llm-star/data/*.sh; do
        [ -f "$f" ] && INCLUDES+=("$f")
    done

    # 코드 디렉토리
    for dir in train model configs scripts eval docs reports tokenizer benchmarks logs; do
        [ -d "llm-star/${dir}" ] && INCLUDES+=("llm-star/${dir}")
    done

    # Root 파일
    for f in README.md CLAUDE.md train_3b_sft_resilient.sh train_3b_resilient.sh train_1b_resilient.sh requirements.txt PLAN_hybrid_3b_fixes.md; do
        [ -f "llm-star/${f}" ] && INCLUDES+=("llm-star/${f}")
    done
}

# ─────────────────────────────────────────────
# 메인 백업
# ─────────────────────────────────────────────
backup_full() {
    echo "════════════════════════════════════════════════"
    echo "[FULL] 완전 백업 → ${BACKUP_DIR}/${ARCHIVE_NAME}.tar.zst.*"
    echo "════════════════════════════════════════════════"

    collect_includes

    if [ ${#INCLUDES[@]} -eq 0 ]; then
        echo "[ERROR] 백업 대상 없음"
        return 1
    fi

    echo "[backup] 포함 대상 (${#INCLUDES[@]}개):"
    printf "  %s\n" "${INCLUDES[@]}"
    echo ""

    # 아카이브 생성
    tar cf - \
        "${EXCLUDE_COMMON[@]}" \
        -C "${PROJECT_ROOT}" \
        "${INCLUDES[@]}" \
        | zstd -3 -T0 \
        | split -b "${CHUNK_SIZE}" - "${BACKUP_DIR}/${ARCHIVE_NAME}.tar.zst."

    local CHUNK_COUNT
    CHUNK_COUNT=$(ls "${BACKUP_DIR}"/${ARCHIVE_NAME}.tar.zst.* 2>/dev/null | wc -l)
    echo "[FULL] 아카이브 완료: ${CHUNK_COUNT}개 청크"

    # MANIFEST
    {
        echo "EVAFRILL-FULL backup date=${DATE_FOLDER} chunks=${CHUNK_SIZE}"
        echo ""
        echo "=== 포함 경로 (${#INCLUDES[@]}개) ==="
        printf "%s\n" "${INCLUDES[@]}"
        echo ""
        echo "=== 청크 파일 ==="
        ls -lh "${BACKUP_DIR}"/${ARCHIVE_NAME}.tar.zst.* 2>/dev/null || true
    } > "${BACKUP_DIR}/MANIFEST_${ARCHIVE_NAME}.txt"

    # SHA256SUMS
    (cd "${BACKUP_DIR}" && sha256sum ${ARCHIVE_NAME}.tar.zst.* > "SHA256SUMS_${ARCHIVE_NAME}" 2>/dev/null || true)
    echo "[checksum] SHA256SUMS_${ARCHIVE_NAME} 생성 완료"

    # Git bundle
    echo ""
    echo "[git-bundle] 생성 중..."
    (cd "${LLM_STAR}" && git bundle create "${BACKUP_DIR}/llm-star.bundle" --all 2>/dev/null) || {
        echo "[git-bundle] 경고: git bundle 생성 실패 (git repo가 아닐 수 있음)"
    }
    [ -f "${BACKUP_DIR}/llm-star.bundle" ] && echo "[git-bundle] 완료: $(du -h "${BACKUP_DIR}/llm-star.bundle" | cut -f1)"
}

# ─────────────────────────────────────────────
# 복원 안내서
# ─────────────────────────────────────────────
generate_restore_guide() {
    cat > "${BACKUP_DIR}/RESTORE_EVAFRILL_FULL.md" << 'GUIDE_EOF'
# EVAFRILL-Mo 완전 백업 복원 안내서

## 아카이브 구성

| 파일 | 내용 |
|------|------|
| `evafrill-full.tar.zst.*` | Pretrain/SFT 체크포인트, 학습 데이터, 코드 전체 |
| `llm-star.bundle` | Git 전체 히스토리 (코드 이중 백업) |
| `SHA256SUMS_evafrill-full` | 청크별 체크섬 |
| `MANIFEST_evafrill-full.txt` | 포함 경로 및 청크 목록 |

## 복원 명령

### 전체 복원

```bash
# 상위 디렉터리에서 실행 (llm-star의 부모 디렉터리)
cd /path/to/parent

# 분할 압축 합치기 → zstd 해제 → tar 풀기
cat llm-star-backup/<날짜>/evafrill-full.tar.zst.* | zstd -d | tar xvf -
```

### Git bundle에서 코드만 복원 (대안)

```bash
git clone llm-star-backup/<날짜>/llm-star.bundle llm-star-from-bundle
```

## 체크섬 검증

```bash
cd llm-star-backup/<날짜>
sha256sum -c SHA256SUMS_evafrill-full
```

## 아카이브 내용 확인 (압축 풀지 않고)

```bash
# 파일 목록 미리보기
cat evafrill-full.tar.zst.aa evafrill-full.tar.zst.ab | zstd -d | tar tf - | head -50

# 전체 목록
cat evafrill-full.tar.zst.* | zstd -d | tar tf -
```

## SFT 재현 최소 셋업

1. 아카이브 복원 (위 명령)
2. `pip install -r requirements.txt`
3. `bash scripts/launch_3b_sft.sh` 또는 `bash train_3b_sft_resilient.sh`

## 포함 내용 상세

- **Pretrain 체크포인트**: `checkpoints/3b_final/` (~166GB)
- **SFT 체크포인트**: `checkpoints/3b_sft/` (~50GB)
- **Pretrain 데이터**: `data/3b_train.bin` + `data/3b_val.bin` (~77GB)
- **SFT 데이터**: `data/sft_combined/` (~16GB), `data/sft/` (~291MB)
- **선호도 데이터**: `data/preference/` (~7.9GB)
- **코드**: train/, model/, configs/, scripts/, eval/ 등 전체
- **데이터 파이프라인**: data/*.py, data/*.sh

## 제외 항목 (재다운로드 가능)

- `data/korean_extra/` (881GB) — 공개 HF 데이터셋
- `data/sft_extra/` (46GB) — 공개 HF 데이터셋
- `data/raw/` (38GB) — 다운로드 스크립트로 재생성
- `data/math/` (27GB), `data/code/` (6.6GB), `data/translation/` (115MB)
- 개별 `*.bin` 파일 (~79GB) — 3b_train.bin에 이미 merge됨
- `checkpoints/1b_final/` (45GB) — 1B 레거시
- `llm-bang-archive.tar.zst` (555GB) — 이전 프로젝트
GUIDE_EOF

    echo "[restore-guide] ${BACKUP_DIR}/RESTORE_EVAFRILL_FULL.md 생성 완료"
}

# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
echo "╔════════════════════════════════════════════════╗"
echo "║  EVAFRILL-Mo 완전 백업 (단일 아카이브)          ║"
echo "║  날짜: ${DATE_FOLDER}                          ║"
echo "║  출력: ${BACKUP_DIR}/                          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)

backup_full
generate_restore_guide

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS_REMAINING=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════"
echo "[완료] 총 소요: ${MINUTES}분 ${SECONDS_REMAINING}초"
echo "[출력] ${BACKUP_DIR}/"
ls -lh "${BACKUP_DIR}/" | tail -20
echo "════════════════════════════════════════════════"

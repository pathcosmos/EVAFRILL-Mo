#!/usr/bin/env bash
# llm-star-backup/<날짜> 폴더를 네이버 클라우드 오브젝트 스토리지에 업로드
# (S3 호환 API 사용, endpoint: https://kr.object.ncloudstorage.com)
#
# 사전 준비:
#   1. NCloud 콘솔에서 Object Storage 버킷 생성
#   2. 인증키(Access Key ID, Secret Key) 발급
#   3. aws configure 로 설정 (또는 환경변수 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
#   4. AWS CLI v2 설치 (https://aws.amazon.com/cli/ 또는 패키지 매니저)
#
# 사용:
#   bash scripts/upload_backup_to_ncloud.sh <버킷명> [날짜폴더] [s3_prefix]
#   예: bash scripts/upload_backup_to_ncloud.sh my-bucket 20260309
#   예: bash scripts/upload_backup_to_ncloud.sh my-bucket 20260309 backups/llm-star

set -euo pipefail

NCLOUD_ENDPOINT="https://kr.object.ncloudstorage.com"
PROJECT_ROOT="/PROJECT/0325120031_A/ghong/taketimes"
BACKUP_BASE="${PROJECT_ROOT}/llm-star-backup"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <bucket_name> [date_folder] [s3_prefix]"
  echo "  bucket_name  : 네이버 클라우드 오브젝트 스토리지 버킷 이름"
  echo "  date_folder  : 백업 날짜 폴더 (기본: 20260309)"
  echo "  s3_prefix    : 버킷 내 경로 접두사 (기본: llm-star-backup/20260309)"
  echo ""
  echo "Example: $0 my-bucket 20260309"
  echo "Example: $0 my-bucket 20260309 backups/llm-star"
  exit 1
fi

BUCKET="$1"
DATE_FOLDER="${2:-20260309}"
S3_PREFIX="${3:-llm-star-backup/${DATE_FOLDER}}"
LOCAL_DIR="${BACKUP_BASE}/${DATE_FOLDER}"

if ! command -v aws &>/dev/null; then
  echo "[error] AWS CLI v2가 필요합니다."
  echo "  설치: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  echo "  aws configure   # Access Key ID, Secret Key 입력"
  exit 1
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "[error] 로컬 백업 폴더가 없습니다: ${LOCAL_DIR}"
  exit 1
fi

echo "[upload] Local: ${LOCAL_DIR}"
echo "[upload] Target: s3://${BUCKET}/${S3_PREFIX}/"
echo "[upload] Endpoint: ${NCLOUD_ENDPOINT}"
echo ""

# 동기화: 로컬 디렉터리 → 오브젝트 스토리지 (기존 파일은 스킵, 새/변경만 업로드)
aws --endpoint-url="${NCLOUD_ENDPOINT}" s3 sync "${LOCAL_DIR}" "s3://${BUCKET}/${S3_PREFIX}/" \
  --no-progress \
  --only-show-errors

echo "[upload] Done. List: aws --endpoint-url=${NCLOUD_ENDPOINT} s3 ls s3://${BUCKET}/${S3_PREFIX}/ --summarize"

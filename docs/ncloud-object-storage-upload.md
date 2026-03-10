# 네이버 클라우드 오브젝트 스토리지 업로드

`llm-star-backup/20260309/` 를 네이버 클라우드 오브젝트 스토리지에 올리는 방법입니다.  
네이버 클라우드는 **S3 호환 API**를 쓰므로 **AWS CLI**로 업로드할 수 있습니다.

## 1. 사전 준비

### 1) 버킷 생성
- [네이버 클라우드 플랫폼 콘솔](https://console.ncloud.com) → Object Storage → 버킷 생성
- 버킷 이름 예: `llm-star-backups` (리전 예: Korea)

### 2) 인증키 발급
- 콘솔 **마이페이지 → 계정 관리 → 인증키 관리**에서 **Object Storage용 Access Key ID / Secret Key** 생성  
  (또는 Object Storage 메뉴 내 인증키 생성 안내 따라하기)

### 3) AWS CLI v2 설치 및 설정
- **AWS CLI v2** 사용 (필수).

```bash
# v2 설치 (Linux 예: 공식 번들)
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip && sudo ./aws/install

aws configure
# AWS Access Key ID [None]: <네이버 Access Key ID>
# AWS Secret Access Key [None]: <네이버 Secret Key>
# Default region name [None]: (엔터 또는 us-east-1)
# Default output format [None]: (엔터)
```

- 환경변수로만 쓰려면:
  ```bash
  export AWS_ACCESS_KEY_ID="<Access Key ID>"
  export AWS_SECRET_ACCESS_KEY="<Secret Key>"
  ```

- **버킷 생성** (v2에서는 `--region us-east-1` 필요할 수 있음):
  ```bash
  aws --endpoint-url=https://kr.object.ncloudstorage.com s3 mb s3://evafrillmo --region us-east-1
  ```

## 2. 업로드 실행

**엔드포인트**: 한국 리전 `https://kr.object.ncloudstorage.com`

### 스크립트 사용 (권장)
```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-star
bash scripts/upload_backup_to_ncloud.sh <버킷명> [날짜폴더] [s3_prefix]
```

예:
```bash
# 20260309 폴더를 버킷의 llm-star-backup/20260309/ 에 업로드
bash scripts/upload_backup_to_ncloud.sh llm-star-backups 20260309

# 버킷 내 경로를 다르게 쓰고 싶을 때
bash scripts/upload_backup_to_ncloud.sh my-bucket 20260309 backups/evafrill-mo
```

### 직접 AWS CLI 사용
```bash
export ENDPOINT="https://kr.object.ncloudstorage.com"
aws --endpoint-url="$ENDPOINT" s3 sync \
  /PROJECT/0325120031_A/ghong/taketimes/llm-star-backup/20260309/ \
  s3://<버킷명>/llm-star-backup/20260309/ \
  --human-readable
```

- **EVAFRILL 한정만** 올리려면 해당 날짜 폴더에서 `evafrill-mo.tar.zst.*` 만 있는 디렉터리를 sync 하거나,  
  업로드 후 버킷에서 `llm-star-repro.*` 객체를 삭제해도 됩니다.

## 3. 업로드 후 확인

```bash
aws --endpoint-url=https://kr.object.ncloudstorage.com s3 ls \
  s3://<버킷명>/llm-star-backup/20260309/ --human-readable --summarize
```

## 4. 다운로드(복원) 시

다른 서버에서 내려받을 때:
```bash
aws --endpoint-url=https://kr.object.ncloudstorage.com s3 sync \
  s3://<버킷명>/llm-star-backup/20260309/ \
  /path/to/local/llm-star-backup/20260309/ \
  --human-readable
```
이후 압축 해제는 `RESTORE_EVAFRILL.md` 참고.

## 참고

- [네이버 클라우드 Object Storage CLI 가이드](https://cli.ncloud-docs.com/docs/guide-objectstorage)
- 엔드포인트: 한국 `https://kr.object.ncloudstorage.com`

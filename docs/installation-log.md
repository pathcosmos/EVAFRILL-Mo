# Installation Log - Machine Setting

**Date**: 2026-03-06
**Machine**: DCTN-0306095349
**Operator**: ghong + Claude Opus 4.6

---

## Machine Specs

| Item | Value |
|------|-------|
| OS | Ubuntu 24.04.3 LTS (x86_64) |
| CPU | AMD EPYC 9365 36-Core (72 cores) |
| RAM | 2,263 GB |
| GPU | 8x NVIDIA B200 (183 GB VRAM each) |
| Driver | 580.95.05 |
| CUDA (nvcc) | 13.1 (V13.1.80) |
| CUDA (nvidia-smi) | 13.0 |
| Disk /home | 5.0 GB (GPFS) |
| Disk /PROJECT | 20 TB (12 TB available, GPFS) |

## Pre-existing Tools

| Tool | Status | Path |
|------|--------|------|
| uv | 0.9.18 | /usr/local/bin/uv |
| Python | 3.12.3 | /usr/bin/python3 |
| Node.js | v24.14.0 | (via NVM at non-standard path) |
| SDKMAN | Not installed | - |
| Java | Not installed | - |
| curl | 8.5.0 | /usr/bin/curl |

---

## Installation Steps

### [1/6] Hardware Detection - SUCCESS with ISSUE

**Command**: `bash scripts/detect-hardware.sh /home/ghong/.machine_setting_profile`

**Issue #1 - HOME 환경변수 비어있음 (MEDIUM)**
- **증상**: `$HOME`이 빈 문자열로 확장되어 프로필이 `/.machine_setting_profile`에 저장 시도 → 허가 거부
- **원인**: Claude Code 실행 환경에서 HOME이 다른 경로(`/data/rntierdata/...`)로 매핑됨
- **해결**: 명시적으로 `/home/ghong/.machine_setting_profile` 경로를 전달
- **프로젝트 반영 필요**: `setup.sh`에서 HOME 검증 로직 추가 권장

**Issue #2 - CUDA_SUFFIX cu131 미매칭 (HIGH)**
- **증상**: nvcc 13.1 → `cu131` 생성. 그러나 `gpu-index-urls.conf`에 cu131 항목 없음
- **원인**: gpu-index-urls.conf가 cu130까지만 정의
- **해결**: `~/.machine_setting_profile`의 CUDA_SUFFIX를 수동으로 `cu130`으로 변경
- **프로젝트 반영 필요**:
  - `gpu-index-urls.conf`에 `cu131` 항목 추가
  - `detect-hardware.sh`에 "매칭 안 될 경우 가장 가까운 하위 버전으로 fallback" 로직 추가

### [2/6] Python Setup - SUCCESS

**Command**: `bash scripts/install-python.sh 3.12`

- uv 0.9.18 이미 설치됨
- Python 3.12.12 다운로드 및 설치 (cpython-3.12.12-linux-x86_64-gnu)
- Warning: `~/.local/bin`이 PATH에 없음 (환경 특성, 정상)

### [3/6] AI Environment - SUCCESS with ISSUES

#### Venv 생성
- **경로**: `/home/ghong/project-ghong/taketimes/llm-star/ai-env` (커스텀 경로)
- **이유**: `/home/ghong`에 4.7GB밖에 없어서 기본 `~/ai-env`에 설치 불가
- Python 3.12.3 인터프리터 사용

**Issue #3 - 디스크 공간 부족 (BLOCKER)**
- **증상**: /home/ghong = 5.0GB, AI 패키지 전체 설치에 15-20GB+ 필요
- **해결**: 12TB 가용한 `/PROJECT` 심볼릭 링크 경로 사용
- **프로젝트 반영 필요**: `setup.sh`에 디스크 공간 사전 검증 로직 추가. 부족 시 대체 경로 안내.

#### Core 패키지 설치 - SUCCESS
- `requirements-core.txt` 전체 설치 성공 (약 220개 패키지)
- 특이사항 없음

#### Data 패키지 설치 - FAILED → RESOLVED

**Issue #4 - cx-Oracle 8.3.0 빌드 실패 (HIGH)**
- **증상**: `ModuleNotFoundError: No module named 'pkg_resources'`
- **원인**: cx-Oracle 8.3.0이 `pkg_resources` (setuptools)를 빌드 의존성으로 선언하지 않음. uv의 빌드 격리(build isolation)로 인해 venv의 setuptools가 빌드 환경에서 보이지 않음
- **1차 시도**: venv에 setuptools 설치 후 재시도 → 실패 (빌드 격리 문제 지속)
- **2차 시도 (성공)**: `uv pip install cx-Oracle==8.3.0 --no-build-isolation`
- **프로젝트 반영 필요**:
  - `setup-venv.sh`에서 cx-Oracle 설치 시 `--no-build-isolation` 옵션 적용
  - 또는 cx-Oracle을 oracledb로 대체 검토 (cx-Oracle은 레거시, oracledb가 이미 포함됨)

#### Web 패키지 설치 - SUCCESS
- `requirements-web.txt` 전체 설치 성공
- 일부 버전 다운그레이드 발생 (cryptography 46.0.5 → 41.0.7, jinja2 3.1.6 → 3.1.2) - 핀 버전 준수

#### GPU 패키지 설치 - FAILED → RESOLVED

**Issue #5 - nvidia-nccl-cu12 의존성 미해결 (HIGH)**
- **증상**: `No solution found: nvidia-nccl-cu12==2.29.7 not found`
- **원인**: cu130 인덱스에 nvidia-nccl-cu12 특정 버전이 없음. requirements-gpu.txt의 nvidia-* 패키지 핀이 cu130 인덱스와 불일치
- **1차 시도**: 전체 unpinned 설치 → bitsandbytes가 cu130 인덱스에 없어서 실패
- **2차 시도 (성공)**: `--index-url cu130 --extra-index-url pypi.org`로 핵심 패키지만 설치
  - torch, torchaudio, torchvision, triton → cu130 인덱스
  - bitsandbytes, onnxruntime-gpu → PyPI fallback
- **프로젝트 반영 필요**:
  - `setup-venv.sh`의 GPU 설치 로직에 `--extra-index-url https://pypi.org/simple` 추가
  - `requirements-gpu.txt`를 핵심 패키지와 nvidia 의존성으로 분리 검토
  - nvidia-* 패키지는 자동 의존성 해결에 맡기는 것이 안전

**Issue #6 - PyTorch CUDA 버전 불일치 (INFO)**
- **증상**: cu130 인덱스에서 설치했지만 실제 PyTorch는 `2.10.0+cu128` (CUDA 12.8 빌드)
- **원인**: PyTorch cu130 인덱스가 아직 완전한 CUDA 13.0 네이티브 빌드를 제공하지 않는 듯
- **영향**: 기능적으로는 정상 작동 (CUDA 하위호환), 하지만 CUDA 13.x 최적화 미활용
- **프로젝트 반영 필요**: 문서에 "cu130 인덱스는 반드시 CUDA 13.0 빌드를 보장하지 않음" 주석 추가

### [4/6] Node.js - SKIPPED

- Node.js v24.14.0 이미 설치됨 (NVM 경로: 비표준 위치)
- NVM 재설치 건너뜀

### [5/6] Java - SUCCESS

- SDKMAN 5.20.0 설치
- Java 21.0.10-tem (Eclipse Temurin) 설치 및 기본 설정

### [6/6] Shell Integration - SUCCESS with FIX

**Issue #7 - Shell 경로 하드코딩 (MEDIUM)**
- **증상**: `install-shell.sh` line 48에서 `$HOME/machine_setting/` 하드코딩
- **영향**: repo가 `~/machine_setting/` 이외의 경로에 있으면 shell 모듈 로딩 실패
- **수정 적용**:
  1. `install-shell.sh:48` → `$HOME/machine_setting/` → `$REPO_DIR/` (이미 line 7에서 계산됨)
  2. `50-ai-env.sh` → `$HOME/machine_setting` 참조 4곳을 `$_MS_REPO_DIR` 변수로 교체
- `.bashrc`에 올바른 경로로 블록 삽입 확인

---

## Final Verification

| Check | Result |
|-------|--------|
| 총 패키지 수 | 319개 |
| torch | 2.10.0+cu128 |
| CUDA available | True |
| GPU count | 7 (of 8 detected) |
| GPU name | NVIDIA B200 |
| transformers | 5.2.0 |
| anthropic | 0.84.0 |
| fastapi | 0.135.1 |
| pandas | 3.0.0 |
| langchain | 1.2.10 |

**Note**: GPU 7개만 인식됨 (8개 물리적 설치). 1개 GPU가 다른 프로세스에 의해 점유되었거나 드라이버 설정 차이 가능.

---

## Installed Paths Summary

| Component | Path |
|-----------|------|
| Repo | /home/ghong/taketimes/from_git/machine-setting/ |
| Venv | /home/ghong/project-ghong/taketimes/llm-star/ai-env/ |
| Python (uv) | /home/ghong/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/ |
| SDKMAN | /home/ghong/.sdkman/ |
| Java | /home/ghong/.sdkman/candidates/java/21.0.10-tem/ |
| Hardware Profile | /home/ghong/.machine_setting_profile |
| Shell Integration | /home/ghong/.bashrc (machine_setting block) |
| Secrets Template | /home/ghong/.bashrc.local |

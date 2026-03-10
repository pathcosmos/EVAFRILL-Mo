# ai-env 환경 가이드

llm-star 프로젝트의 Python 가상 환경 구성 및 관리 지침.

---

## 목차

1. [개요](#1-개요)
2. [사전 요구사항](#2-사전-요구사항)
3. [빠른 시작](#3-빠른-시작-quick-start)
4. [구성 요소 상세](#4-구성-요소-상세)
5. [패키지 카테고리별 설명](#5-패키지-카테고리별-설명)
6. [GPU 서버 특수 사항](#6-gpu-서버-특수-사항)
7. [트러블슈팅](#7-트러블슈팅)
8. [디렉토리 구조 참조](#8-디렉토리-구조-참조)

---

## 1. 개요

`ai-env`는 llm-star 프로젝트 전용 Python 가상 환경이다. 위치는 `ai-env/` (llm-star 루트 기준 상대 경로).

### 존재 이유: 시스템 Python과의 분리

| 환경 | Python | torch | CUDA |
|------|--------|-------|------|
| 시스템 Python (`/usr/bin/python3`) | 3.12.3 | 2.6.0 (CPU-only, pip 설치) | 미지원 |
| ai-env (`ai-env/bin/python`) | 3.12.3 | 2.10.0+cu128 (NV 커스텀) | CUDA 12.8 지원 |

시스템 Python에는 GPU 지원이 없는 표준 PyTorch가 설치되어 있다. 학습 스크립트를 시스템 Python으로 실행하면 CUDA를 인식하지 못해 CPU 학습이 시작되거나 오류가 발생한다. `ai-env`는 NVIDIA B200에 최적화된 NV 커스텀 빌드를 격리된 환경에 유지한다.

### llm-star 프로젝트와의 관계

모든 학습 스크립트(`scripts/launch_*.sh`, `train_*_resilient.sh`)는 상단에 `source ai-env/bin/activate`를 포함하며, ai-env를 자동 활성화한 후 학습을 시작한다. ai-env가 올바르게 구성되어 있지 않으면 학습 자체가 불가능하다.

---

## 2. 사전 요구사항

### 공통

| 항목 | 최소 버전 | 권장 |
|------|-----------|------|
| Python | 3.10 | 3.12.3 |
| uv (패키지 매니저) | 0.9.0 | 0.9.18 이상 |
| pip | 23.0 | uv가 있으면 불필요 |

uv가 없으면 표준 `python -m venv`와 `pip`로 대체된다. uv가 있으면 가상 환경 생성이 훨씬 빠르다.

### GPU 서버 (--gpu 플래그 사용 시)

| 항목 | 요구사항 |
|------|---------|
| GPU | NVIDIA (B200 권장) |
| CUDA Driver | 580.x 이상 |
| transformer_engine | 시스템 site-packages에 사전 설치 필요 |
| NV 커스텀 torch/flash_attn | 시스템 site-packages에 사전 설치 필요 |

GPU 서버에서 `--gpu` 없이 실행하면 NV 커스텀 패키지 연결이 생략되므로, GPU 서버에서는 반드시 `--gpu` 플래그를 사용해야 한다.

---

## 3. 빠른 시작 (Quick Start)

### Step 1: 저장소 클론

```bash
git clone <repo-url> llm-star
cd llm-star
```

### Step 2: 환경 설정

**CPU 환경 (로컬 개발 / Mac):**

```bash
bash scripts/setup_ai_env.sh
```

**GPU 서버 (NVIDIA B200 등):**

```bash
bash scripts/setup_ai_env.sh --gpu
```

`--gpu` 플래그를 붙이면 NV 커스텀 패키지(torch, flash_attn, transformer_engine) 심볼릭 링크 처리까지 수행한다.

### Step 3: 환경 검증

```bash
bash scripts/check_env.sh
```

정상 출력 예시 (GPU 서버):

```
torch 2.10.0a0+b4e4ee81d3.nv25.12 CUDA:True GPUs:8
TE 2.10.0
trl 0.8.6
tensorboard 2.16.2
peft 0.10.0
accelerate 0.30.1
wandb 0.16.6
datasets 4.4.1
transformers 4.40.2
```

---

## 4. 구성 요소 상세

### 4.1 requirements.txt

`requirements.txt`는 pip로 설치되는 패키지 목록이다. NV 커스텀 빌드(torch, flash_attn, transformer_engine)는 이 파일에 포함되지 않는다.

```
ai-env/bin/pip install -r requirements.txt
```

또는 uv 사용 시:

```
uv pip install -r requirements.txt
```

### 4.2 scripts/setup_ai_env.sh

자동화된 4단계 설치 흐름:

| 단계 | 내용 |
|------|------|
| 1. Python 탐지 | python3.12 → python3.11 → python3.10 → python3 순으로 자동 탐지 |
| 2. venv 생성 | uv가 있으면 `uv venv ai-env`, 없으면 `python -m venv ai-env` |
| 3. 패키지 설치 | `pip install -r requirements.txt` |
| 4. NV 심볼릭 링크 | `--gpu` 플래그 시, 시스템 site-packages의 NV 커스텀 패키지를 ai-env에 링크 |

### 4.3 scripts/check_env.sh

환경 무결성 검증 스크립트. 확인 항목:

- `ai-env/` 디렉토리 존재 여부
- torch 버전 및 `torch.cuda.is_available()` 결과
- GPU 수 (`torch.cuda.device_count()`)
- transformer_engine import 성공 여부
- 핵심 패키지 설치 여부: trl, tensorboard, peft, accelerate, wandb, datasets, transformers

학습 스크립트 실행 전 이 스크립트로 환경을 먼저 확인하는 것을 권장한다.

### 4.4 학습 스크립트 자동 활성화 패턴

모든 학습 실행 스크립트는 상단에 다음 코드를 포함한다:

```bash
source ai-env/bin/activate
```

이를 통해 스크립트를 어느 경로에서 실행하든 ai-env가 자동으로 활성화된다. 스크립트를 llm-star 루트 디렉토리에서 실행해야 상대 경로 `ai-env/`가 올바르게 해석된다.

---

## 5. 패키지 카테고리별 설명

### 5.1 HuggingFace 생태계

| 패키지 | 최소 버전 | 용도 |
|--------|-----------|------|
| transformers | 4.40.0 | 모델 아키텍처, 토크나이저 |
| tokenizers | 0.19.0 | 고성능 토크나이저 (Rust 기반) |
| datasets | 4.0.0 | 데이터셋 로딩 및 전처리 |
| accelerate | 0.30.0 | 분산 학습 추상화 |
| peft | 0.10.0 | LoRA 등 파라미터 효율적 파인튜닝 |

### 5.2 학습 도구

| 패키지 | 최소 버전 | 용도 |
|--------|-----------|------|
| trl | 0.8.6 | SFT, RLHF, DPO 학습 유틸리티 |
| sentencepiece | 0.1.99 | SentencePiece 토크나이저 |
| wandb | 0.16.0 | 학습 모니터링 (Weights & Biases) |
| tensorboard | 2.16.0 | 학습 시각화 |

### 5.3 유틸리티

| 패키지 | 최소 버전 | 용도 |
|--------|-----------|------|
| PyYAML | 6.0 | 설정 파일 파싱 |
| numpy | 1.26.0 | 수치 연산 |
| tqdm | 4.66.0 | 진행 표시 |

### 5.4 NV 커스텀 빌드 (pip install 금지)

> **경고**: 아래 패키지는 절대로 `pip install`로 재설치하지 말 것. NV 커스텀 빌드가 표준 패키지로 덮어씌워져 B200 최적화가 파괴된다.

| 패키지 | 버전 | 설치 방식 |
|--------|------|-----------|
| torch | 2.10.0+cu128 (nv25.12) | 시스템 사전 설치 + 심볼릭 링크 |
| flash_attn | 2.8.3 | 시스템 사전 설치 + 심볼릭 링크 |
| transformer_engine | 2.10.0 | 시스템 사전 설치 + 심볼릭 링크 |

재설치가 필요한 경우 반드시 NVIDIA 제공 빌드를 사용하거나, `scripts/setup_ai_env.sh --gpu`를 통해 심볼릭 링크를 재구성할 것.

---

## 6. GPU 서버 특수 사항

### 6.1 NV 커스텀 빌드 보존

시스템 site-packages에 설치된 NV 커스텀 torch는 일반 PyPI 패키지와 다르다. ai-env는 `include-system-site-packages = false` 설정으로 시스템 패키지를 자동 상속하지 않는다. 대신 `setup_ai_env.sh --gpu`가 필요한 패키지만 선택적으로 심볼릭 링크한다.

```
ai-env/lib/python3.12/site-packages/torch -> /usr/lib/python3/dist-packages/torch (심볼릭 링크)
```

이 방식으로 pip upgrade 등의 명령이 NV 빌드를 변경하지 못하도록 격리한다.

### 6.2 transformer_engine 심볼릭 링크

transformer_engine은 B200에서 FP8 학습을 활성화하는 핵심 패키지다. PyPI 버전이 없거나 호환되지 않으므로 반드시 시스템 사전 설치본을 링크해야 한다.

`setup_ai_env.sh --gpu` 실행 시 자동 처리된다. 수동 확인 방법:

```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

### 6.3 B200 정밀도 지원

| 정밀도 | 지원 여부 | 권장 용도 |
|--------|-----------|-----------|
| FP32 | O | 디버깅 |
| BF16 | O | 일반 학습 (권장) |
| FP16 | O | 주의: B200에서 BF16이 더 안정적 |
| FP8 (e4m3fn) | O (네이티브) | 대형 모델 학습, 메모리 절약 |

FP8 사용 예시:

```python
import torch
# B200은 FP8 네이티브 지원
dtype = torch.float8_e4m3fn
```

### 6.4 현재 서버 사양

| 항목 | 사양 |
|------|------|
| GPU | 8x NVIDIA B200, 183 GB VRAM each (~1.47 TB total) |
| RAM | 2.2 TB |
| CUDA (nvidia-smi) | 13.0 |
| CUDA (nvcc) | 13.1 |
| GPU Driver | 580.95.05 |
| PyTorch | 2.10.0+cu128 (nv25.12) |
| Python | 3.12.3 |
| uv | 0.9.18 |
| 작업 스토리지 | `/PROJECT/` — 3.5 TB, 여유 2.2 TB |
| 홈 스토리지 | `/home/ghong` — 5 GB (코드만) |

> **주의**: 체크포인트, 데이터셋 등 대용량 파일은 `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/` 하위에 저장할 것. 홈 디렉토리 용량 초과 주의.

---

## 7. 트러블슈팅

### 7.1 `torch.cuda.is_available()` 가 False를 반환

**원인 1**: ai-env가 활성화되지 않은 상태에서 시스템 Python을 사용 중.

```bash
# 확인
which python
# 출력이 /usr/bin/python3이면 문제
# 출력이 .../ai-env/bin/python이어야 정상

# 해결
source ai-env/bin/activate
```

**원인 2**: `setup_ai_env.sh --gpu`를 실행하지 않아 NV 커스텀 torch가 링크되지 않음.

```bash
bash scripts/setup_ai_env.sh --gpu
bash scripts/check_env.sh
```

**원인 3**: torch가 ai-env 내부에 CPU-only 버전으로 재설치됨 (pip install torch 실행 이력).

```bash
# ai-env 내 torch 버전 확인
ai-env/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 정상: 2.10.0+cu128  12.8
# 비정상: 2.x.x+cpu   None

# 해결: ai-env 삭제 후 재구성
rm -rf ai-env
bash scripts/setup_ai_env.sh --gpu
```

### 7.2 `No module named pip` 오류

uv로 생성된 venv에서 간혹 발생. uv pip을 직접 사용하거나 pip을 부트스트랩:

```bash
# uv pip 사용
uv pip install -r requirements.txt

# 또는 pip 부트스트랩
ai-env/bin/python -m ensurepip --upgrade
```

### 7.3 transformer_engine import 실패

```
ModuleNotFoundError: No module named 'transformer_engine'
```

`--gpu` 플래그 없이 설치했거나 심볼릭 링크가 깨진 경우:

```bash
# 심볼릭 링크 재구성
bash scripts/setup_ai_env.sh --gpu

# 시스템 설치 위치 수동 확인
python3 -c "import transformer_engine; print(transformer_engine.__file__)"
```

시스템에도 transformer_engine이 없다면 NVIDIA 제공 패키지 설치가 필요하다. 서버 관리자에게 문의할 것.

### 7.4 시스템 Python과 ai-env torch 혼동

학습 스크립트를 직접 `python train/sft.py`로 실행했을 때 시스템 Python이 사용되는 경우:

```bash
# 잘못된 실행
python train/sft.py --config configs/korean_3b_sft.yaml

# 올바른 실행 (ai-env 명시)
ai-env/bin/python train/sft.py --config configs/korean_3b_sft.yaml

# 또는 활성화 후 실행
source ai-env/bin/activate
python train/sft.py --config configs/korean_3b_sft.yaml
```

### 7.5 패키지 버전 충돌

requirements.txt 업데이트 후 충돌 발생 시:

```bash
# ai-env 완전 재생성
rm -rf ai-env
bash scripts/setup_ai_env.sh --gpu  # GPU 서버
# 또는
bash scripts/setup_ai_env.sh        # CPU 환경
```

---

## 8. 디렉토리 구조 참조

```
llm-star/                          # 프로젝트 루트
├── ai-env/                        # Python 가상 환경 (git 제외)
│   ├── bin/
│   │   ├── activate               # source로 활성화
│   │   ├── python -> python3.12
│   │   └── pip
│   ├── lib/python3.12/site-packages/
│   │   ├── torch -> (NV 심볼릭 링크, GPU 모드)
│   │   ├── flash_attn -> (NV 심볼릭 링크, GPU 모드)
│   │   ├── transformer_engine -> (NV 심볼릭 링크, GPU 모드)
│   │   ├── transformers/
│   │   ├── trl/
│   │   └── ...
│   └── pyvenv.cfg                 # venv 메타데이터
├── configs/
│   └── korean_3b_sft.yaml
├── docs/
│   └── ai-env-guide.md            # 이 문서
├── scripts/
│   ├── setup_ai_env.sh            # 환경 구성 자동화
│   ├── check_env.sh               # 환경 검증
│   ├── launch_3b_sft.sh           # 학습 실행 (ai-env 자동 활성화)
│   └── ...
├── train/
│   └── sft.py
├── eval/
├── requirements.txt               # pip 설치 패키지 목록
└── train_3b_sft_resilient.sh      # 학습 실행 (ai-env 자동 활성화)
```

### 관련 외부 경로

| 경로 | 용도 |
|------|------|
| `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/` | 체크포인트, 데이터셋 등 대용량 파일 |
| `/usr/lib/python3/dist-packages/` | 시스템 site-packages (NV 커스텀 빌드 위치) |
| `/home/ghong/` | 홈 디렉토리 (5 GB 제한, 코드만) |

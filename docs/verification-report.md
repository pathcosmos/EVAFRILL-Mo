# Verification Report - Blackwell B200 AI/ML Environment

**Date**: 2026-03-06
**Machine**: DCTN-0306095349

---

## 1. Hardware & Driver Stack

### NVIDIA Driver & CUDA

| Component | Version | Status |
|-----------|---------|--------|
| NVIDIA Driver | 580.95.05 | PASS |
| CUDA Toolkit (nvcc) | 13.1 (V13.1.80) | PASS |
| CUDA (nvidia-smi) | 13.0 | PASS |
| cuDNN | 9.17.0.29 (CUDA 13) | PASS |
| NCCL | 2.28.9 (CUDA 13.0) | PASS |
| HPC-X | Installed (Mellanox) | PASS |

### GPU Details

| GPU | Name | VRAM | Power | Temp | Compute Cap | NVLink |
|-----|------|------|-------|------|-------------|--------|
| 0 | B200 | 183,359 MiB | 143W / 1000W | 23°C | 10.0 | 18x NVLink |
| 1 | B200 | 183,359 MiB | 140W / 1000W | 22°C | 10.0 | 18x NVLink |
| 2 | B200 | 183,359 MiB | 139W / 1000W | 23°C | 10.0 | 18x NVLink |
| 3 | B200 | 183,359 MiB | 144W / 1000W | 24°C | 10.0 | 18x NVLink |
| 4 | B200 | 183,359 MiB | 141W / 1000W | 23°C | 10.0 | 18x NVLink |
| 5 | B200 | 183,359 MiB | 142W / 1000W | 23°C | 10.0 | 18x NVLink |
| 6 | B200 | 183,359 MiB | 142W / 1000W | 23°C | 10.0 | 18x NVLink |

**Note**: lspci detects 8 GPU slots but nvidia-smi only shows 7. 8번째 GPU는 드라이버 레벨에서 미인식.

### GPU Topology

- **NVLink**: 모든 GPU 간 NV18 (18x NVLink, 각 53.125 GB/s = 총 956 GB/s per GPU)
- **NUMA**: 2노드 구성
  - Node 0 (CPU 0-35): GPU 0-3, 1,160 GB RAM
  - Node 1 (CPU 36-71): GPU 4-6, 1,157 GB RAM
- **NIC**: 10x Mellanox (mlx5_0 ~ mlx5_9) — InfiniBand/RoCE 고속 네트워크

### System Resources

| Resource | Value |
|----------|-------|
| CPU | AMD EPYC 9365, 72 cores |
| RAM Total | 2.2 TiB (2,263 GB) |
| RAM Free | ~2.0 TiB |
| File descriptors limit | 650,000 |
| Max threads | 18,540,347 |
| Memory overcommit | 0 (conservative) |

---

## 2. PyTorch & GPU Framework

### Core Verification

| Check | Result | Status |
|-------|--------|--------|
| PyTorch version | 2.10.0+cu128 | PASS |
| CUDA available | True | PASS |
| CUDA version (runtime) | 12.8 | PASS |
| cuDNN version | 91002 (9.10.02) | PASS |
| cuDNN enabled | True | PASS |
| Device count | 7 | PASS (7/8) |
| Compute Capability | 10.0 (sm_100) | PASS - Blackwell |
| CUDA Arch List | sm_70~sm_120 | PASS - Blackwell(sm_100) 포함 |

### Precision Support (Blackwell Key Features)

| Feature | Status | Note |
|---------|--------|------|
| BF16 matmul | PASS | Blackwell 최적 정밀도 |
| FP8 (float8_e4m3fn) | PASS | Blackwell 신규 기능, 2x throughput |
| TF32 matmul | OFF (default) | 수동 활성화 필요 |
| TF32 cuDNN | ON | OK |

### Advanced Features

| Feature | Status |
|---------|--------|
| Flash Attention (SDPA) | PASS |
| torch.compile | PASS |
| Triton | 3.6.0 PASS |
| NCCL backend | PASS |
| Multi-GPU matmul (GPU 0) | PASS |
| Multi-GPU matmul (GPU 6) | PASS |
| 20GB VRAM allocation | PASS (all tested GPUs) |

### Quantization

| Feature | Status |
|---------|--------|
| bitsandbytes 4-bit | PASS (v0.49.2) |
| ONNX Runtime GPU | PASS |
| ONNX Providers | TensorRT, CUDA, CPU |

---

## 3. AI/ML Package Verification

### Import Test: 80/80 PASS

| Category | Packages | Status |
|----------|----------|--------|
| LLM Providers | anthropic, openai, google.generativeai | 3/3 PASS |
| LangChain | langchain, community, core, huggingface, langgraph, langsmith | 6/6 PASS |
| HuggingFace | transformers, datasets, tokenizers, hub, safetensors, accelerate, peft, sentence_transformers | 8/8 PASS |
| Classical ML | sklearn, scipy, xgboost, lightgbm, numba | 5/5 PASS |
| Vector/Embed | chromadb, faiss | 2/2 PASS |
| Visualization | matplotlib, seaborn | 2/2 PASS |
| Tracking | wandb | 1/1 PASS |
| Data | pandas, numpy, pyarrow, PIL, cv2 | 5/5 PASS |
| Database | sqlalchemy, psycopg2, pymysql, oracledb, cx_Oracle, clickhouse_connect | 6/6 PASS |
| Document | pdfplumber, pypdf, docx, pptx, openpyxl, lxml | 6/6 PASS |
| OCR | easyocr, pytesseract | 2/2 PASS |
| Web | fastapi, flask, gradio, starlette, uvicorn | 5/5 PASS |
| HTTP | httpx, requests, aiohttp | 3/3 PASS |
| Auth | jwt, cryptography, bcrypt | 3/3 PASS |
| Utilities | pydantic, loguru, structlog, tqdm, click, typer, rich | 7/7 PASS |
| Async | anyio, aiofiles | 2/2 PASS |
| Serialization | orjson, yaml, jsonschema | 3/3 PASS |
| GPU | torch, torchaudio, torchvision, bitsandbytes | 4/4 PASS |
| Messaging | aiokafka, paho.mqtt | 2/2 PASS |
| Cloud | boto3 | 1/1 PASS |
| Monitoring | opentelemetry, sentry_sdk, prometheus_client | 3/3 PASS |
| Infra | kubernetes | 1/1 PASS |

### Warnings

- `google.generativeai`: Deprecated → `google.genai` 마이그레이션 권장

---

## 4. Blackwell B200 최적화 평가

### Model Loading Feasibility

**Total VRAM**: 1,248 GB (7 x 178 GB) | **System RAM**: 2,263 GB

| Model | FP16 Size | Status | GPUs Needed |
|-------|-----------|--------|-------------|
| Llama-3.1-8B | 16 GB | VRAM 단독 | 1 |
| Llama-3.1-70B | 140 GB | VRAM 단독 | 1 |
| Qwen-2.5-72B | 144 GB | VRAM 단독 | 1 |
| Mixtral-8x22B | 176 GB | VRAM 단독 | 1 |
| Llama-3.1-405B | 810 GB | VRAM 단독 | 5 |
| DeepSeek-V3 (671B) | 1,342 GB | CPU offload 필요 | 7 + RAM |

| Model (4-bit) | Size | Status | GPUs Needed |
|---------------|------|--------|-------------|
| Llama-3.1-405B | 202 GB | VRAM 단독 | 2 |
| DeepSeek-V3 (671B) | 336 GB | VRAM 단독 | 2 |

### Optimization Recommendations

#### [ACTION NEEDED] TF32 활성화

TF32가 기본값 OFF. Blackwell에서 성능 향상을 위해 활성화 권장:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

또는 환경변수:
```bash
export NVIDIA_TF32_OVERRIDE=1
```

#### [INFO] PyTorch CUDA 버전 불일치

- 시스템 CUDA: 13.1
- PyTorch CUDA: 12.8
- 영향: Blackwell sm_100은 정상 인식되지만, CUDA 13.x 전용 최적화(FP4 등)는 미활용
- PyTorch가 CUDA 13 네이티브 빌드를 제공하면 업그레이드 권장

#### [INFO] 8번째 GPU 미인식

- lspci: 8개 NVIDIA 장치 감지
- nvidia-smi / PyTorch: 7개만 인식
- 가능한 원인: 하드웨어 결함, 드라이버 이슈, BIOS 설정, MIG 모드
- 확인 명령: `dmesg | grep -i nvidia | grep -i error`

#### [GOOD] NVLink 토폴로지 최적

- 모든 GPU 간 NV18 (18-way NVLink) 직접 연결
- GPU당 대역폭: ~956 GB/s (18 x 53.125 GB/s)
- 멀티GPU 학습/추론에 최적 구성

#### [GOOD] NUMA-aware 배치

- GPU 0-3: NUMA Node 0 (CPU 0-35)
- GPU 4-6: NUMA Node 1 (CPU 36-71)
- 데이터 로딩 시 NUMA affinity 고려하면 추가 성능 향상 가능

---

## 5. 종합 점수

| 영역 | 점수 | 비고 |
|------|------|------|
| Driver/CUDA Stack | 9/10 | 8번째 GPU 미인식 감점 |
| PyTorch GPU 통합 | 9/10 | cu128 빌드 (cu130 아님) |
| AI/ML 패키지 | 10/10 | 80/80 전체 PASS |
| Blackwell 최적화 | 8/10 | TF32 미활성화, CUDA 13 네이티브 아님 |
| 멀티GPU 인프라 | 10/10 | NVLink 18-way, NCCL, RDMA 완비 |
| **종합** | **9.2/10** | **프로덕션 AI 워크로드에 즉시 사용 가능** |

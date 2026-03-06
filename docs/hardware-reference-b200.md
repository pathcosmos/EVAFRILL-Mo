# NVIDIA B200 Blackwell Hardware Reference

**Machine**: DCTN-0306095349
**Last Updated**: 2026-03-06
**Purpose**: 다른 프로젝트에서 이 하드웨어의 특성을 참고할 수 있는 레퍼런스 문서

---

## 1. GPU: NVIDIA B200 (Blackwell Architecture)

### Architecture Overview

| Item | Value |
|------|-------|
| Architecture | Blackwell (GB202) |
| Compute Capability | **10.0 (sm_100)** |
| GPU Count | **7** |
| VRAM per GPU | 183,359 MiB (**~179 GB**, HBM3e) |
| Total VRAM | **~1,248 GB** |
| TDP per GPU | 1,000W |
| Idle Power per GPU | ~140W |
| VBIOS | 97.00.C5.00.2F |
| GPU Part | 2901-886-A1 |

### Blackwell 핵심 기능

| Feature | Status | 설명 |
|---------|--------|------|
| **FP8 (float8_e4m3fn)** | Supported | FP16 대비 2x throughput. Transformer Engine의 핵심. 대형 모델 학습/추론에 최적 |
| **BF16** | Supported | AI 학습의 표준 정밀도. 모든 주요 프레임워크에서 기본 지원 |
| **TF32** | Supported | FP32 연산에서 자동으로 텐서코어 활용. 환경변수로 활성화: `NVIDIA_TF32_OVERRIDE=1` |
| **FP4** | Hardware 지원 | CUDA 13.x 네이티브 PyTorch 빌드 필요 (현재 미제공) |
| **Flash Attention (SDPA)** | Supported | PyTorch 내장 `scaled_dot_product_attention` |
| **Structured Sparsity** | Supported | cuSPARSELt 0.8.1 설치됨. 2:4 sparsity로 2x 가속 |
| **MIG (Multi-Instance GPU)** | Disabled | 필요 시 활성화 가능. 각 B200을 독립 인스턴스로 분할 |

### NVLink 5.0 Interconnect

```
모든 GPU 간 NV18 (18-way NVLink 5.0) 직접 연결

     GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0  -    NV18  NV18  NV18  NV18  NV18  NV18
GPU1 NV18   -    NV18  NV18  NV18  NV18  NV18
GPU2 NV18  NV18   -    NV18  NV18  NV18  NV18
GPU3 NV18  NV18  NV18   -    NV18  NV18  NV18
GPU4 NV18  NV18  NV18  NV18   -    NV18  NV18
GPU5 NV18  NV18  NV18  NV18  NV18   -    NV18
GPU6 NV18  NV18  NV18  NV18  NV18  NV18   -
```

| NVLink Spec | Value |
|-------------|-------|
| Generation | NVLink 5.0 (Blackwell) |
| Links per GPU | 18 |
| Bandwidth per link | 53.125 GB/s |
| **Total per GPU** | **956.25 GB/s** |
| **All-to-all bandwidth** | **6,694 GB/s** |

이전 세대 대비:
- NVLink 4.0 (H100): 18 links × 25 GB/s = 450 GB/s → **B200은 2.1배**
- NVLink 3.0 (A100): 12 links × 25 GB/s = 300 GB/s → **B200은 3.2배**

---

## 2. CPU & Memory

| Item | Value |
|------|-------|
| CPU | AMD EPYC 9365 36-Core Processor |
| Cores | **72** (2 sockets × 36 cores) |
| Architecture | Zen 5 (Genoa/Turin) |
| Total RAM | **2,263 GB** (~2.2 TiB, DDR5) |

### NUMA Topology

| NUMA Node | CPUs | RAM | GPUs |
|-----------|------|-----|------|
| Node 0 | 0-35 (36 cores) | 1,160 GB | GPU 0, 1, 2, 3 |
| Node 1 | 36-71 (36 cores) | 1,157 GB | GPU 4, 5, 6 |

**NUMA distance**: Node 0↔1 = 32 (inter-socket)

**최적화 팁**: 데이터 로딩 프로세스를 GPU와 같은 NUMA 노드에 바인딩하면 메모리 접근 지연 감소.
```bash
# GPU 0-3 사용 시 NUMA Node 0에 바인딩
numactl --cpunodebind=0 --membind=0 python train.py

# GPU 4-6 사용 시 NUMA Node 1에 바인딩
numactl --cpunodebind=1 --membind=1 python train.py
```

---

## 3. Software Stack

### NVIDIA Driver & CUDA

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 580.95.05 (Open Kernel Module) |
| CUDA Toolkit (nvcc) | 13.1 (V13.1.80) |
| CUDA Version (nvidia-smi) | 13.0 |
| cuDNN | 9.17.0.29 |
| NCCL | 2.28.9 |
| TensorRT | 10.14.1.48 |
| cuBLAS | 13.2.0.9 |
| cuBLASMp | 0.7.0 (multi-process BLAS) |
| cuSPARSELt | 0.8.1 (structured sparsity) |
| NVSHMEM | 3.4.5 (symmetric memory) |
| Nsight Compute | 2025.4.0 |

### Kernel Modules

| Module | Purpose |
|--------|---------|
| nvidia | Core GPU driver |
| nvidia_uvm | Unified Virtual Memory (GPU-CPU 공유 메모리) |
| nvidia_peermem | GPUDirect RDMA (GPU↔NIC 직접 전송) |
| nvidia_drm | DRM subsystem |
| gdrdrv | GDRCopy (GPU Direct RDMA copy) |

### Networking

| Component | Details |
|-----------|---------|
| NIC | 10x Mellanox ConnectX (mlx5_0~mlx5_9) |
| InfiniBand | ibverbs 56.0, RDMA 지원 |
| HPC-X | NCCL RDMA/SHARP, Spectrum-X, MRC 플러그인 |
| AWS OFI | NCCL OFI 플러그인 (Libfabric 기반) |
| Libfabric | 2.1.0 (AWS 최적화) |

---

## 4. 대형 모델 적합성

### FP16 (Half Precision) — 1,248 GB VRAM

| Model | Parameter | FP16 Size | Fits? | GPUs |
|-------|-----------|-----------|-------|------|
| Llama-3.1-8B | 8B | 16 GB | 1 GPU | 1 |
| Qwen-2.5-72B | 72B | 144 GB | 1 GPU | 1 |
| Llama-3.1-70B | 70B | 140 GB | 1 GPU | 1 |
| Mixtral-8x22B | 141B | 176 GB | 1 GPU | 1 |
| Llama-3.1-405B | 405B | 810 GB | VRAM | 5 |
| DeepSeek-V3 | 671B | 1,342 GB | CPU offload | 7 + RAM |

### 4-bit Quantized (GPTQ/AWQ/GGUF)

| Model | 4-bit Size | Fits? | GPUs |
|-------|------------|-------|------|
| Llama-3.1-70B | ~35 GB | 1 GPU | 1 |
| Llama-3.1-405B | ~202 GB | VRAM | 2 |
| DeepSeek-V3 | ~336 GB | VRAM | 2 |
| DeepSeek-V3 (FP8) | ~671 GB | VRAM | 4 |

### 학습 (Training) 적합성

| Workload | Feasibility |
|----------|-------------|
| 8B 모델 full fine-tuning | 1 GPU, 여유로움 |
| 70B 모델 LoRA/QLoRA | 1-2 GPU |
| 70B 모델 full fine-tuning | 4-7 GPU (DeepSpeed ZeRO-3) |
| 405B 모델 LoRA/QLoRA | 4-5 GPU |
| 405B 모델 full fine-tuning | 불가 (VRAM 부족), CPU offload 필요 |

---

## 5. 성능 최적화 가이드

### 환경변수 (권장)

```bash
# TF32 활성화 — Ampere+ GPU에서 FP32 연산 가속
export NVIDIA_TF32_OVERRIDE=1

# NCCL 최적화
export NCCL_P2P_LEVEL=NVL           # NVLink 우선
export NCCL_IB_DISABLE=0            # InfiniBand 활성화
export NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA 최대 활용

# PyTorch 멀티GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Python 코드 (권장)

```python
import torch

# TF32 활성화 (FP32 matmul에 텐서코어 사용)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# BF16 학습
from torch.amp import autocast
with autocast('cuda', dtype=torch.bfloat16):
    output = model(input)

# FP8 추론 (Blackwell 최적)
# transformers 라이브러리 사용
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype=torch.float8_e4m3fn,  # Blackwell FP8
    device_map="auto"
)

# Multi-GPU 분산 추론
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    device_map="auto",  # 자동 GPU 분배
    torch_dtype=torch.bfloat16,
    max_memory={i: "170GiB" for i in range(7)}  # 7 GPUs
)
```

### DeepSpeed 설정 (멀티GPU 학습)

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "none"}
    },
    "gradient_accumulation_steps": 4,
    "train_micro_batch_size_per_gpu": 1
}
```

### vLLM 추론 서빙

```bash
# 7 GPU tensor parallel
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-405B \
    --tensor-parallel-size 7 \
    --dtype bfloat16 \
    --max-model-len 32768
```

---

## 6. 시스템 리소스 제한

| Parameter | Value | 의미 |
|-----------|-------|------|
| Open files (ulimit -n) | 650,000 | 대규모 데이터 로더에 충분 |
| Locked memory (ulimit -l) | unlimited | RDMA/GPUDirect에 필수 |
| Max threads | 18,540,347 | PyTorch DataLoader worker에 충분 |
| VM overcommit | 0 | Conservative (OOM killer 활성) |
| Disk /home | 5 GB (GPFS) | 작업 데이터는 /PROJECT 사용 |
| Disk /PROJECT | 20 TB (12 TB 가용, GPFS) | 모델/데이터 저장 |

---

## 7. 주의사항

1. **/home 용량 제한**: 5 GB만 사용 가능. venv, 모델 체크포인트, 데이터셋은 반드시 `/home/ghong/project-ghong/` (→ /PROJECT 심볼릭 링크) 경로 사용

2. **PyTorch CUDA 버전**: 시스템 CUDA는 13.1이지만 PyTorch는 CUDA 12.8 빌드 (`2.10.0+cu128`). Blackwell sm_100은 정상 인식되지만 CUDA 13.x 전용 최적화(FP4 등)는 미활용

3. **NUMA 경계**: GPU 0-3과 GPU 4-6은 다른 NUMA 노드. 크로스 노드 GPU 통신은 NVLink로 직접 연결되어 있어 성능 영향은 미미하지만, CPU 데이터 전처리는 NUMA 바인딩 권장

4. **전력**: 7 GPU × 1,000W TDP = 최대 7,000W GPU 전력 소비. 실제 학습 시 GPU당 300-700W 예상

5. **ECC 메모리**: 모든 GPU에서 ECC 활성화됨. 에러 없음 확인 (2026-03-06 기준)

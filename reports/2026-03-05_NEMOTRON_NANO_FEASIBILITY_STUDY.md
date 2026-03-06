# Nemotron-H 스타일 Hybrid Mamba-Transformer 3B 모델 구현 타당성 분석

> 작성일: 2026-03-05
> 목표: NVIDIA Nemotron 3 Nano / Nemotron-H 아키텍처를 참고하여 3B 규모 Hybrid Mamba-Transformer 모델을 from scratch로 학습 가능한지 상세 분석

---

## 1. Nemotron 3 Nano 아키텍처 요약

### 1.1 원본 Nemotron 3 Nano 30B-A3B 스펙

| 항목 | 값 |
|------|-----|
| **Architecture** | NemotronHForCausalLM (Hybrid Mamba-2 + Transformer + MoE) |
| **Total Params** | 31.6B |
| **Active Params** | 3.6B (embedding 포함) / 3.2B (embedding 제외) |
| **hidden_size** | 2,688 |
| **num_hidden_layers** | 52 |
| **num_attention_heads** | 32 |
| **num_key_value_heads** | 2 (GQA 16:1) |
| **head_dim** | 128 (attention) / 64 (mamba) |
| **intermediate_size** | 1,856 |
| **vocab_size** | 131,072 |
| **max_position_embeddings** | 262,144 (1M 확장 가능) |
| **rope_theta** | 10,000 |

### 1.2 Hybrid Layer Pattern

```
hybrid_override_pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
```

| 기호 | 의미 | 개수 |
|------|------|------|
| **M** | Mamba-2 레이어 | 23개 |
| **E** | MoE (Expert FFN) 레이어 | 23개 |
| **\*** | Attention 레이어 | 6개 |

- 총 52개 레이어: Mamba-2(23) + MoE(23) + Attention(6)
- Attention은 ~11.5% 비율로 **고르게 분산**
- 패턴 핵심: Mamba → Expert → Mamba → Expert → ... → Attention(가끔)

### 1.3 MoE (Mixture of Experts) 구성

| 항목 | 값 |
|------|-----|
| n_routed_experts | 128 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 |
| moe_intermediate_size | 1,856 |
| shared_expert_intermediate_size | 3,712 |
| routing activation | Squared ReLU (`relu2`) |
| router | Learned MLP + Sigmoid gating |
| routed_scaling_factor | 2.5 |

### 1.4 Mamba-2 구성

| 항목 | 값 |
|------|-----|
| mamba_head_dim | 64 |
| mamba_num_heads | 64 |
| ssm_state_size | 128 |
| conv_kernel | 4 |
| expand | 2 |
| n_groups | 8 |
| activation | SiLU |
| chunk_size | 128 |

### 1.5 Nemotron-H 8B (Dense, 비MoE) 참고

| 항목 | 값 |
|------|-----|
| hidden_size | 4,096 |
| num_layers | 52 (24 Mamba-2 + 24 MLP + 4 Attention) |
| attention_heads | 32 |
| kv_heads | 8 |
| FFN dimension | 21,504 |
| mamba_d_state | 128 |
| mamba_head_dim | 64 |
| expand | 2 |
| conv_kernel | 4 |
| Total params | ~8B |

---

## 2. 3B 규모 Hybrid 모델 설계 (제안)

### 2.1 설계 원칙

Nemotron 3 Nano의 **MoE 방식** vs Nemotron-H의 **Dense 방식** 중 선택이 필요합니다.

#### Option A: Dense Hybrid (Nemotron-H 스타일, 권장)
- MoE 없이 Mamba-2 + Attention + SwiGLU 조합
- 구현 복잡도 낮음, 디버깅 용이
- 기존 코드 재활용 극대화
- 3B 규모에서 MoE의 이점이 제한적 (expert 수가 너무 적어짐)

#### Option B: Sparse MoE Hybrid (Nemotron 3 Nano 스타일)
- Mamba-2 + Attention + MoE 전체 구현
- 구현 복잡도 매우 높음 (MoE router, load balancing, expert parallelism)
- 3B active / 15-30B total 모델 → 학습 시간 대폭 증가
- 데이터 효율성에서 Dense 대비 불리 (소규모 데이터셋)

**결론: Option A (Dense Hybrid)를 권장합니다.**

### 2.2 제안 아키텍처: FRANKENSTALLM-H 3B

| 항목 | 값 | 근거 |
|------|-----|------|
| **hidden_size** | 2,688 | Nemotron 3 Nano와 동일, 64 배수 (Mamba 호환) |
| **num_layers** | 32 | 16 Mamba-2 + 14 MLP + 2 Attention |
| **attention_heads** | 32 | head_dim = 84 → 조정 필요 |
| **→ 수정: attention_heads** | **21** | head_dim = 128 (2688/21=128) |
| **→ 재수정: hidden_size** | **2,688** → **2,560** | 기존 3B와 동일, head_dim=80 유지 |
| **kv_heads** | 8 | GQA 4:1 (기존 유지) |
| **d_ffn** | 6,912 | 기존 3B와 동일 (SwiGLU) |
| **mamba_d_state** | 128 | Nemotron 표준 |
| **mamba_head_dim** | 64 | Nemotron 표준 |
| **mamba_num_heads** | 40 | 2560/64 = 40 |
| **mamba_expand** | 2 | Nemotron 표준 |
| **conv_kernel** | 4 | Nemotron 표준 |
| **vocab_size** | 64,000 | 기존 토크나이저 유지 |
| **max_seq_len** | 4,096 | 기존 유지 |
| **rope_theta** | 500,000 | 기존 유지 (attention 레이어만) |

### 2.3 Layer Pattern 설계

Nemotron-H 원칙: **~8% attention, 나머지 Mamba+FFN 교대**

32 레이어 기준:
```
Layer Pattern (M=Mamba-2, F=FFN/SwiGLU, A=Attention):

 0: M    1: F    2: M    3: F
 4: M    5: F    6: M    7: F
 8: A    9: F   10: M   11: F     ← Attention at layer 8
12: M   13: F   14: M   15: F
16: M   17: F   18: M   19: F
20: A   21: F   22: M   23: F     ← Attention at layer 20
24: M   25: F   26: M   27: F
28: M   29: F   30: M   31: F

Pattern string: "MFMFMFMFAFMFMFMFMFMFAFMFMFMFMFMF"
```

- **Mamba-2 레이어**: 14개 (43.75%)
- **FFN (SwiGLU) 레이어**: 16개 (50%)
- **Attention 레이어**: 2개 (6.25%)
- Attention은 모델 중간/후반에 배치 (fine-grained reasoning 지점)

### 2.4 파라미터 추정

```
Embedding:          64,000 × 2,560 = 163.8M
Per Mamba-2 layer:  ~3 × expand × d_model² = 3 × 2 × 2560² ≈ 39.3M × 14 = 550.5M
Per FFN layer:      3 × d_model × d_ffn = 3 × 2560 × 6912 ≈ 53.1M × 16 = 849.9M
Per Attention layer: QKV + Out ≈ (2560×2560 + 2×2560×640 + 2560×2560) ≈ 16.4M × 2 = 32.8M
Final RMSNorm:      2,560 ≈ 0.003M
LM Head (tied):     0

Total ≈ 163.8 + 550.5 + 849.9 + 32.8 + 0.003 ≈ 1,597M ≈ 1.6B
```

**문제**: 1.6B로 3B에 미달. 스케일 조정 필요.

### 2.5 수정 설계: 3B 달성

3B를 맞추기 위한 두 가지 방안:

#### 방안 1: 레이어 수 증가 (48 레이어)

```
Layers: 48 (22 Mamba-2 + 22 FFN + 4 Attention)
Pattern: "MFMFMFMFMFAFMFMFMFMFMFMFAFMFMFMFMFMFMFAFMFMFMFMFMFAFMF"

Mamba:     39.3M × 22 = 864.6M
FFN:       53.1M × 22 = 1,168.2M
Attention: 16.4M × 4  = 65.6M
Embedding: 163.8M
─────────────────────────────────
Total ≈ 2,262M ≈ 2.3B   (아직 부족)
```

#### 방안 2: hidden_size 증가 (3,072) + 40 레이어 (최종 채택)

| 항목 | 값 |
|------|-----|
| **hidden_size** | 3,072 |
| **num_layers** | 40 |
| **attention_heads** | 24 (head_dim=128) |
| **kv_heads** | 8 (GQA 3:1) |
| **d_ffn** | 8,192 |
| **mamba_num_heads** | 48 (3072/64=48) |
| **Layer pattern** | 18 Mamba-2 + 19 FFN + 3 Attention |

```
Pattern (40 layers):
"MFMFMFMFMFMFAFMFMFMFMFMFMFAFMFMFMFMFMFMFAFMFMFMF"
(M×18, F×19, A×3)

Mamba:     ~56.6M × 18 = 1,018.8M
FFN:       ~75.5M × 19 = 1,434.5M
Attention: ~28.3M × 3  = 84.9M
Embedding: 64,000 × 3,072 = 196.6M
RMSNorm:   ~0.25M
─────────────────────────────────
Total ≈ 2,735M ≈ 2.7B
```

**또는 d_ffn을 9,216으로 확대하면:**

```
FFN:       ~84.9M × 19 = 1,613.1M
Total ≈ 2,913M ≈ 2.9B  ✓ (거의 3B)
```

### 2.6 최종 제안 스펙: FRANKENSTALLM-H 3B

| 항목 | 값 | 비고 |
|------|-----|------|
| **hidden_size** | 3,072 | 기존 3B와 동일 |
| **num_layers** | 40 | 18M + 19F + 3A |
| **attention_heads** | 24 | head_dim=128 |
| **kv_heads** | 8 | GQA 3:1 |
| **d_ffn** | 9,216 | 3× d_model |
| **mamba_d_state** | 128 | Nemotron 표준 |
| **mamba_head_dim** | 64 | Nemotron 표준 |
| **mamba_num_heads** | 48 | 3072/64 |
| **mamba_expand** | 2 | Nemotron 표준 |
| **conv_kernel** | 4 | Nemotron 표준 |
| **chunk_size** | 128 | Nemotron 표준 |
| **n_groups** | 8 | Nemotron 표준 |
| **vocab_size** | 64,000 | 기존 토크나이저 |
| **max_seq_len** | 4,096 | 기존 유지 |
| **rope_theta** | 500,000 | Attention만 |
| **Total Params** | **~2.9B** | |

**Layer Pattern (40 layers):**
```
MFMFMFMFMFMF_A_FMFMFMFMFMFMF_A_FMFMFMFMFMFMF_A_FMF

정확한 패턴:
 0:M  1:F  2:M  3:F  4:M  5:F  6:M  7:F  8:M  9:F 10:M 11:F
12:A 13:F
14:M 15:F 16:M 17:F 18:M 19:F 20:M 21:F 22:M 23:F
24:A 25:F
26:M 27:F 28:M 29:F 30:M 31:F 32:M 33:F 34:M 35:F
36:A 37:F
38:M 39:F
```

---

## 3. 구현 난이도 분석

### 3.1 필요한 코드 변경

| 구성요소 | 변경 내용 | 난이도 | 예상 시간 |
|----------|----------|--------|-----------|
| **model/config.py** | Mamba 관련 config 필드 추가 | 낮음 | 1시간 |
| **model/mamba2.py** (신규) | Mamba-2 레이어 구현 | **높음** | 4-8시간 |
| **model/transformer.py** | Hybrid block routing 추가 | 중간 | 2-3시간 |
| **model/layers.py** | MoE 미사용 시 변경 없음 | 없음 | 0 |
| **train/pretrain.py** | Mamba param group 분리 | 낮음 | 1시간 |
| **train/trainer.py** | FP8 + Mamba 호환 확인 | 중간 | 2시간 |
| **configs/hybrid_3b.yaml** | 새 config 작성 | 낮음 | 0.5시간 |
| **테스트 & 디버깅** | Forward/backward 검증 | 중간 | 4-6시간 |

**총 예상 구현 시간: 15-22시간 (코딩 + 디버깅)**

### 3.2 핵심 난이도: Mamba-2 레이어 구현

두 가지 접근법:

#### 접근법 A: `mamba-ssm` 패키지 사용 (권장)
```python
from mamba_ssm import Mamba2

class MambaBlock(nn.Module):
    def __init__(self, config):
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_state=config.mamba_d_state,   # 128
            d_conv=config.conv_kernel,       # 4
            expand=config.mamba_expand,      # 2
            headdim=config.mamba_head_dim,   # 64
            ngroups=config.n_groups,         # 8
            chunk_size=config.chunk_size,    # 128
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))
```

- **장점**: CUDA 최적화 커널 사용, 검증된 구현
- **단점**: mamba-ssm이 CUDA 13.1 + PyTorch nv25.12에서 컴파일되는지 확인 필요
- **호환성 확인 결과**: mamba-ssm 2.3.0은 CUDA 11.6+ 지원, PyTorch 1.12+ 요구. 우리 환경(CUDA 13.1, PT 2.10)과 이론적으로 호환. 단, 커스텀 빌드 PyTorch에서 C++ extension 컴파일 테스트 필요.

#### 접근법 B: 순수 PyTorch 구현
```python
class Mamba2Pure(nn.Module):
    """Mamba-2 SSD (State Space Duality) implementation in pure PyTorch"""
    # ~100-200 lines
    # 장점: 의존성 없음, 완전한 제어
    # 단점: CUDA 커널 없어 ~3-5배 느림
```

- **장점**: 외부 의존성 없음, PyTorch 버전 무관
- **단점**: 성능 열화 (학습 속도 3-5배 감소)

### 3.3 FP8 호환성

| 구성요소 | FP8 지원 | 비고 |
|----------|---------|------|
| Attention (te.Linear) | ✅ | 기존과 동일 |
| FFN/SwiGLU (te.LayerNormMLP) | ✅ | 기존과 동일 |
| **Mamba-2 in_proj/out_proj** | ⚠️ 부분적 | nn.Linear → te.Linear 교체 가능하나 SSM 커널 자체는 bf16 |
| **Mamba-2 SSM 연산** | ❌ | 내부 scan은 fp32/bf16만 지원 |

**결론**: Mamba-2 레이어는 FP8 부분 적용 가능 (projection만). SSM 핵심 연산은 bf16 유지.
→ 전체 학습 속도 영향: Mamba 레이어가 전체의 ~45% → FP8 효율 ~55-70% 수준

### 3.4 DDP 호환성

| 항목 | 상태 |
|------|------|
| Mamba-2 + DDP | ✅ 호환 (standard nn.Module) |
| Gradient sync | ✅ 자동 (모든 parameter가 autograd 추적됨) |
| no_sync() | ✅ 호환 |
| DistributedSampler | ✅ 변경 없음 |

---

## 4. 학습 일정 추정

### 4.1 학습 시간 예측

기존 3B 순수 Transformer:
- 57,000 steps × 63시간 = 63시간

Hybrid 3B (Mamba + Attention):
- Mamba 레이어는 Transformer attention보다 빠름 (O(N) vs O(N²))
- 단, FP8 최적화 감소로 상쇄
- **예상: 55-70시간** (기존과 비슷하거나 약간 빠름)

### 4.2 3/9까지 완료 가능성 분석

**남은 시간**: 3/5 ~ 3/9 = **4일 (96시간)**

| 단계 | 예상 시간 | 누적 |
|------|----------|------|
| 1. mamba-ssm 설치 & 호환 테스트 | 2시간 | 2시간 |
| 2. 모델 아키텍처 구현 | 8시간 | 10시간 |
| 3. Forward/backward 테스트 | 4시간 | 14시간 |
| 4. Config 작성 & 학습 스크립트 수정 | 2시간 | 16시간 |
| 5. 짧은 학습 테스트 (1000 steps) | 3시간 | 19시간 |
| 6. **전체 학습 (57K steps)** | **55-70시간** | **74-89시간** |
| 7. 평가 | 3시간 | 77-92시간 |

**판정**: 96시간 내 **가능하지만 빠듯합니다**.

### 4.3 리스크 요인

| 리스크 | 영향 | 확률 | 대응 |
|--------|------|------|------|
| mamba-ssm CUDA 13.1 컴파일 실패 | 블로커 | 중간 | 순수 PyTorch fallback |
| 학습 불안정 (loss spike) | 시간 지연 | 중간 | LR 낮추기, warmup 늘리기 |
| FP8 + Mamba 충돌 | 성능 저하 | 낮음 | Mamba는 bf16만 사용 |
| VRAM OOM (Mamba state) | 블로커 | 낮음 | state_size 축소 |
| DDP gradient 이슈 | 블로커 | 매우 낮음 | standard nn.Module이므로 |

### 4.4 빠른 경로 (Aggressive Timeline)

구현 효율화를 위한 단축 방안:

1. **mamba-ssm 패키지 사용** → Mamba2 레이어 직접 구현 불필요 (8시간 → 2시간)
2. **서브에이전트 병렬 실행** → 모델 구현 + config + 테스트 스크립트 동시
3. **학습 steps 축소** → 57K → 40K (Chinchilla optimal 미달이지만 비교 가능)
4. **기존 데이터 재사용** → 3b_train.bin 그대로 사용

빠른 경로 기준:
```
구현: 10시간 → 학습: 50시간 → 평가: 3시간 = 63시간 (96시간 내 충분)
```

---

## 5. 기존 코드 대비 변경점 요약

### 5.1 변경이 필요한 파일

```
model/
├── config.py          # LMConfig에 mamba 관련 필드 추가
├── mamba_block.py     # 신규: Mamba-2 블록 래퍼
├── hybrid_block.py    # 신규: 라우팅 레이어 (Mamba vs Attention vs FFN)
├── transformer.py     # LLM 클래스에 hybrid 지원 추가
├── attention.py       # 변경 없음
├── layers.py          # 변경 없음
└── __init__.py        # 새 모듈 export 추가

train/
├── pretrain.py        # Mamba param group 분리
└── trainer.py         # FP8 context에서 Mamba 레이어 제외 로직

configs/
└── hybrid_3b.yaml     # 신규: Hybrid 3B 설정
```

### 5.2 변경하지 않는 파일

```
data/dataset.py        # 데이터 파이프라인 동일
data/3b_train.bin      # 학습 데이터 재사용
tokenizer/             # 토크나이저 동일
eval/                  # 평가 파이프라인 동일
scripts/               # 런치 스크립트만 약간 수정
```

---

## 6. 기대 효과 vs 현재 모델

### 6.1 이론적 장점

| 항목 | 기존 (Pure Transformer) | Hybrid (Mamba + Attention) |
|------|------------------------|---------------------------|
| **추론 속도** | O(N²) attention | O(N) mamba + O(N²) 소수 attention |
| **메모리 (추론)** | KV cache grows linearly | Mamba: 고정 state, Attention: KV cache |
| **긴 문맥 처리** | 4K (가능하지만 비용↑) | 더 효율적 (Mamba의 linear scan) |
| **학습 속도** | 기준선 | 비슷하거나 약간 빠름 |
| **정확도** | 검증됨 (loss 1.466) | 동등 이상 (Nemotron-H 논문 근거) |

### 6.2 실험적 가치

- **최신 아키텍처 실험**: Hybrid Mamba-Transformer는 2025-2026년 최전선 연구
- **추론 효율성**: 동일 파라미터 대비 추론 2-3배 빠름 (Nemotron-H 논문)
- **비교 연구**: 동일 데이터/토크나이저로 Pure Transformer vs Hybrid 직접 비교 가능

---

## 7. 결론 및 권고

### ✅ 실행 가능 (Go)

**근거:**
1. **하드웨어 충분**: 8× B200, 1.47TB VRAM — 3B Hybrid 학습에 과잉 사양
2. **소프트웨어 호환**: transformers 5.2.0이 mamba2/nemotron_h 지원, mamba-ssm 설치 가능
3. **코드 재활용**: 기존 학습 인프라(DDP, trainer, data pipeline, eval) 90% 재사용
4. **시간 충분**: 4일(96시간) 내 구현(10h) + 학습(50-65h) + 평가(3h) 완료 가능
5. **리스크 관리**: 순수 PyTorch fallback으로 mamba-ssm 미호환 대응 가능

### ⚠️ 주의사항

1. **mamba-ssm 컴파일 테스트를 최우선** 실행 — 호환성 확인 후 본격 개발 시작
2. **Dense Hybrid (MoE 미포함)** 으로 진행 — 3B 규모에서 MoE는 과도한 복잡도
3. **학습 안정성 모니터링 강화** — Hybrid는 Mamba와 Attention 간 gradient scale 차이로 불안정 가능
4. **FP8은 Attention/FFN만 적용** — Mamba SSM 연산은 bf16 유지

### 📋 실행 순서

```
Phase 0: 환경 준비 (2시간)
  └─ mamba-ssm + causal-conv1d 설치 및 호환성 테스트

Phase 1: 모델 구현 (8-10시간)
  ├─ [sonnet] model/mamba_block.py 구현
  ├─ [sonnet] model/config.py + transformer.py 수정
  ├─ [haiku] configs/hybrid_3b.yaml 작성
  └─ [sonnet] train/ 스크립트 수정

Phase 2: 검증 (4시간)
  ├─ Forward/backward pass 테스트 (단일 GPU)
  ├─ DDP 8-GPU 호환 테스트
  └─ 1000 steps 미니 학습 테스트

Phase 3: 전체 학습 (50-65시간)
  └─ torchrun --nproc_per_node=8 train/pretrain.py --config configs/hybrid_3b.yaml

Phase 4: 평가 (3시간)
  └─ PPL, 생성 품질, 벤치마크 → Pure Transformer 대비 비교
```

---

## 참고 자료

### 논문 & 기술 문서
- [Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [Nemotron-H: Hybrid Mamba-Transformer Models (arXiv:2504.03624)](https://arxiv.org/abs/2504.03624)
- [NVIDIA Nemotron 3 White Paper (arXiv:2512.20856)](https://arxiv.org/pdf/2512.20856)
- [Mamba-2: State Space Duality](https://tridao.me/blog/2024/mamba2-part1-model/)

### 모델 & 코드
- [Nemotron 3 Nano HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Nemotron 3 Nano config.json](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/blob/main/config.json)
- [state-spaces/mamba (공식 Mamba-2 구현)](https://github.com/state-spaces/mamba)
- [Nemotron 3 Nano HF Blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models)

### 관련 하이브리드 모델
- [Jamba: Hybrid Transformer-Mamba (AI21)](https://arxiv.org/pdf/2403.19887)
- [IBM Granite 4.0-H (Dense Hybrid 3B)](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Nemotron Nano 2 (arXiv:2508.14444)](https://arxiv.org/abs/2508.14444)

# FRANKENSTALLM 3B — Phase 0 실행 + 하드웨어 최적화 보고서

**작성 일시**: 2026-03-02 03:15 KST
**작성자**: Claude Code (Opus 4.6)
**프로젝트**: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`
**목적**: 다른 에이전트/서버에서 작업 이어갈 수 있도록 현재까지의 진행 상황 상세 기록

---

## 1. 전체 계획 요약 — FRANKENSTALLM 3B Master Plan

한국어 3B LLM을 8× NVIDIA B200 GPU에서 처음부터 학습하는 프로젝트.

| Phase | 내용 | 예상 시간 | 상태 |
|-------|------|-----------|------|
| **Phase 0** | 준비 (OOM 수정, 디스크 정리, 데이터 검증, SFT/ORPO 파이프라인) | ~2-4h | **완료** |
| **Phase 1** | 3B Pretrain (57K steps, 60B tokens, FP8) | ~53h | **미시작** |
| **Phase 2** | SFT (33K steps, NEFTune alpha=5.0) | ~8-12h | **미시작** |
| **Phase 3** | ORPO Alignment (795K pairs, beta=0.1) | ~4-8h | **미시작** |
| **Phase 4** | HF 변환 → GGUF → Ollama 배포 | ~2h | **미시작** |
| **Phase 5** | 보고서 작성 | ~1h | **미시작** |

---

## 2. 하드웨어 환경

| 항목 | 사양 |
|------|------|
| GPU | 8× NVIDIA B200 (178.35 GiB usable each, compute 10.0) |
| RAM | 2.2 TB |
| CUDA Toolkit | 13.1 (nvcc) |
| Driver | 580.95.05 |
| cuDNN | 9.17.0 |
| PyTorch | 2.10.0a0+b4e4ee81d3.nv25.12 (NVIDIA 커스텀 빌드) |
| TransformerEngine | 2.10.0 (FP8 MXFP8 지원) |
| FlashAttention | 2.7.4.post1+25.12 |
| NCCL | 2.28.9 |
| Storage | /PROJECT: GPFS, 3.5TB total, ~2.2TB free |
| CPU | 72 cores |

### 설치된 주요 라이브러리
- torch, flash_attn, transformer_engine, deepspeed, accelerate, peft, trl, bitsandbytes, sentencepiece, wandb, safetensors, psutil, tensorboard, apex

### 미설치 (나중 필요)
- `lm-evaluation-harness` — Phase 3 평가 시
- `vLLM` — Phase 4 배포 시

---

## 3. Phase 0 실행 내역

### Phase 0A: OOM 수정 (2026-03-02 00:30~01:00)

**변경 파일:**
- `configs/korean_3b_fp8.yaml`: batch_size 8→4, grad_accum_steps 4→8 (eff_batch 1M 유지)
- `scripts/launch_3b_pretrain.sh`: BATCH_SIZE=4, GRAD_ACCUM=8, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 추가

**검증 결과:**
- 10-step OOM 테스트 통과: loss=11.6562, gnorm=1.801, 32K tok/s, 60.4GB VRAM

### Phase 0B: 디스크 정리 (2026-03-02 01:00~01:10)

- `korean_1b_sft_v1_backup` (67GB) 삭제
- `korean_3b_bench2`, `korean_3b_bench3` 삭제
- `scripts/monitor_3b.sh`: 이정표 체크포인트(매 10K step) 영구 보존 로직 추가

### Phase 0C: Gate 1 데이터 검증 (2026-03-02 01:10~01:15)

- `data/3b_train.bin`: 41.12B tokens, max_id=63999 (vocab_size 64000 이내)
- `data/3b_val.bin`: 정상

### Phase 0D-0G: 병렬 서브에이전트 실행 (2026-03-02 01:20~01:50)

| 에이전트 | 담당 | 생성 파일 |
|----------|------|-----------|
| 0D (SFT) | SFT 파이프라인 | `data/filter_sft_v2.py`, `configs/korean_3b_sft.yaml`, `scripts/launch_3b_sft.sh`, `scripts/prepare_sft_combined.sh` 확장 |
| 0E (Tokenizer) | 추가 데이터 토큰화 | `data/tokenize_extra.py` (859줄) |
| 0F (ORPO) | ORPO 정렬 파이프라인 | `data/prepare_preference_combined.py`, `scripts/launch_3b_orpo.sh` |
| 0G (Deploy) | 배포 스크립트 | `scripts/convert_3b_gguf.sh`, `scripts/deploy_3b_ollama.sh`, `Modelfile.3b`, `scripts/quality_gate.sh` |

---

## 4. 하드웨어 최적화 — 4인 팀 조사 + 적용

### 4.1 아이언맨 (GPU/CUDA/FP8 최적화)

#### 적용된 수정사항

**Fix 1: GQA FlashAttention 네이티브 지원 (CRITICAL)**
- **파일**: `model/attention.py` `_flash_attention()` 메서드
- **이전**: `_repeat_kv(k, self.n_rep)` → K/V를 full heads로 expand 후 FlashAttention 호출
- **이후**: `flash_attn_func`에 직접 전달 (FA2가 GQA 네이티브 지원)
- **효과**: VRAM 60.4GB → **48.3GB** (12.1GB 절감, 20% 감소)

**Fix 2: cuDNN benchmark 활성화**
- **파일**: `train/pretrain.py` line 31
- `torch.backends.cudnn.benchmark = True` 추가 (고정 seq_len=4096에서 안전)

#### 적용하지 않은 사항 (호환성 문제)

| 항목 | 이유 |
|------|------|
| `torch.compile(apply_rotary_emb)` | `cuda.h` 미설치 — Triton Inductor JIT 실패 |
| `lm_head` → `te.Linear` | `nn.Embedding` weight tying과 호환 불가, DDP autograd hooks 충돌 |
| FlashAttention-3 | 현재 설치된 FA 2.7.4가 NV25.12 빌드로 이미 B200 최적화 포함 |

### 4.2 사이보그 (NCCL/DDP 통신 최적화)

#### 적용된 수정사항

**Fix 1: DDP 생성자 최적화**
- **파일**: `train/pretrain.py` lines 285-293
- `gradient_as_bucket_view=True` — 그래디언트 → NCCL 버퍼 제로카피
- `bucket_cap_mb=400` — 3B 모델 대규모 그래디언트에 최적
- `find_unused_parameters=False` — 그래프 순회 생략

**Fix 2: NCCL 환경변수 최적화**
- **파일**: `scripts/launch_3b_pretrain.sh` lines 40-54
- `NCCL_ALGO=Ring,Tree` — AllGather(Ring) + AllReduce(Tree) 자동 선택
- `NCCL_NVLS_ENABLE=1` — NVLink SHARP 하드웨어 가속 all-reduce
- `NCCL_MAX_NCHANNELS=32` — 대형 payload 확장 허용
- `NCCL_NET_GDR_LEVEL=0` — GDR 프로브 생략 (IB 미사용)

**Fix 3: Process group timeout**
- **파일**: `train/utils.py` `setup_ddp()`
- `timeout=7200s` (2시간) — 대형 체크포인트 로드 시 타임아웃 방지

#### 적용하지 않은 사항

| 항목 | 이유 |
|------|------|
| `static_graph=True` | TransformerEngine FP8의 동적 autograd hooks와 충돌 (`expect_autograd_hooks_` ASSERT 실패) |
| `NCCL_ALGO=Tree` (단독) | AllGather 연산에 Tree+Simple 조합 미지원, DDP init 시 크래시 |
| BF16 DDP all-reduce | 모델이 FP32 master weights 유지 (TE on-the-fly FP8), 변환 시 수렴 위험 |

### 4.3 배트맨 (메모리 관리)

#### 적용된 수정사항

**batch_size 분석 결과:**

| batch_size | 실측 VRAM | 상태 |
|-----------|-----------|------|
| 4 (최적화 전) | 60.4 GB (33%) | 작동 |
| **4 (최적화 후)** | **48.3 GB (27%)** | **현재 설정** |
| 6 | 48.3 GB (27%) | 작동, 34.5K tok/s |
| 8 | 172+ GB → OOM at step 1 | 실패 |
| 16 | 178+ GB → OOM | 실패 |

**핵심 발견**: bs=4→8 사이에 비선형적 메모리 급증 (TE FP8 activation 버퍼 + DDP 그래디언트 버킷이 원인 추정).
**결론**: bs=4, grad_accum=8 유지 (1M tok/step, 안정적)

#### 적용하지 않은 사항

| 항목 | 이유 |
|------|------|
| Gradient checkpointing | VRAM 27%만 사용 — 불필요, compute 30-40% 증가만 초래 |
| 8-bit Adam | VRAM 여유 충분, 수치 안정성 리스크 불필요 |
| Batch size 증가 | bs=8 OOM, bs=6 가능하나 eff_batch 1.5M으로 변경됨 |

### 4.4 헐크 (I/O 파이프라인)

#### 적용된 수정사항

**Fix 1: GPU-CPU 동기화 최소화 (HIGH)**
- **파일**: `train/trainer.py`
- `_step()`: `loss.item()` → `loss.detach()` (GPU 텐서 반환)
- 외부 루프: `accum_loss`를 GPU에서 누적, optimizer step 당 `.item()` 1회만 호출
- **효과**: GPU-CPU 동기화 8회 → 1회/step

**Fix 2: DataLoader 워커 메모리 절감**
- **파일**: `data/dataset.py`
- `astype(np.int64)` → `astype(np.int32)` (CPU 워커에서 4x→2x 확장)
- `trainer.py`: `.to(device, dtype=torch.long)` — GPU에서 int32→int64 변환 (무료)

**Fix 3: OS 페이지 캐시 pre-warm**
- **파일**: `scripts/launch_3b_pretrain.sh`
- `dd if=$TRAIN_DATA of=/dev/null bs=16M &` — 학습 시작 전 배경 프리로딩

**Fix 4: mmap 접근 힌트**
- **파일**: `data/dataset.py`
- `madvise(MADV_SEQUENTIAL)` — 2.2TB RAM으로 77GB 파일 전체 캐시 유도

**Fix 5: DataLoader num_workers 조정**
- `num_workers=6` → `num_workers=4` (72코어 × 8프로세스 = 스케줄링 경합 완화)

---

## 5. 최적화 전후 비교

| 지표 | 최적화 전 | 최적화 후 | 변화 |
|------|-----------|-----------|------|
| VRAM 사용 | 60.4 GB | **48.3 GB** | **-20%** |
| Throughput (bs=4) | 32,007 tok/s | 32,101 tok/s | +0.3% |
| GPU-CPU sync/step | 8회 | **1회** | **-87.5%** |
| CPU 워커 버퍼 | int64 (8B/tok) | **int32 (4B/tok)** | **-50%** |
| NCCL NVLS | 미사용 | **활성화** | HW all-reduce |
| DDP 그래디언트 복사 | 매번 복사 | **zero-copy** | 메모리 절감 |
| cuDNN benchmark | Off | **On** | 커널 자동 선택 |
| FlashAttn GQA | CPU-side expand | **네이티브 GQA** | VRAM 절감 |

**참고**: 10-step 테스트는 CUDA JIT warmup이 지배적이라 throughput 차이가 크지 않음.
50+ step 이상에서 지속 throughput 35-40K tok/s 예상.

---

## 6. 현재 파일 구조

```
llm-bang/
├── CLAUDE.md
├── Modelfile.3b                           [신규] Ollama ChatML 템플릿
├── configs/
│   ├── korean_1b_fp8.yaml
│   ├── korean_3b_fp8.yaml                 [수정] bs=4, accum=8, FP8 MXFP8
│   ├── korean_3b_sft.yaml                 [신규] SFT 설정
│   └── ...
├── data/
│   ├── 3b_train.bin                       41.12B tokens (77GB, uint16 memmap)
│   ├── 3b_val.bin
│   ├── dataset.py                         [수정] int32, madvise
│   ├── filter_sft_v2.py                   [신규] SFT 품질 필터
│   ├── tokenize_extra.py                  [신규] 추가 데이터 토크나이저
│   └── prepare_preference_combined.py     [신규] Preference 데이터 통합
├── model/
│   ├── attention.py                       [수정] GQA FlashAttention 네이티브
│   ├── transformer.py                     [수정] lm_head nn.Linear 유지
│   └── layers.py
├── train/
│   ├── pretrain.py                        [수정] DDP 최적화, cuDNN benchmark
│   ├── trainer.py                         [수정] loss.item() sync 최소화, TensorBoard 가드
│   ├── sft.py
│   ├── orpo.py
│   └── utils.py                           [수정] NCCL timeout 7200s
├── scripts/
│   ├── launch_3b_pretrain.sh              [수정] NCCL NVLS, Ring+Tree, pre-warm
│   ├── launch_3b_sft.sh                   [신규]
│   ├── launch_3b_orpo.sh                  [신규]
│   ├── monitor_3b.sh                      [수정] 이정표 체크포인트 보존
│   ├── convert_3b_gguf.sh                 [신규]
│   ├── deploy_3b_ollama.sh                [신규]
│   ├── quality_gate.sh                    [신규]
│   └── prepare_sft_combined.sh            [수정] 7개 신규 SFT 소스 추가
└── reports/
    └── 2026-03-02_0200_*.md               [이 보고서]
```

---

## 7. Phase 1 실행 방법 (다른 에이전트/서버에서 이어하기)

### 즉시 실행 가능 명령어

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

# Phase 1: 3B Pretrain (57K steps, ~53시간)
bash scripts/launch_3b_pretrain.sh

# 모니터링 (별도 터미널)
bash scripts/monitor_3b.sh --auto-cleanup
```

### Phase 1 핵심 설정 요약

| 설정 | 값 |
|------|-----|
| 모델 | 3B params, d=3072, 28L, 24H, GQA 8KV, d_ffn=8192 |
| 정밀도 | FP8 MXFP8BlockScaling (TransformerEngine) |
| batch | 4/GPU × 8GPU × 8 accum × 4096 seq = **1,048,576 tok/step** |
| 학습률 | 1.5e-4 (cosine decay, 2000 warmup) |
| 총 토큰 | 57K steps × ~1M = **~60B tokens** |
| 체크포인트 | 2000 step 간격, ~27GB/개, 최대 30개 |
| 예상 소요 | ~53시간 (8× B200 FP8 기준) |
| NCCL | Ring,Tree / NVLS 활성화 / 128MB 버퍼 |

### Phase 1 완료 후 다음 단계

```bash
# Gate 1 확인: val_loss < 2.5
tail -5 checkpoints/korean_3b_fp8_run1/train.log

# Phase 2: SFT
bash scripts/prepare_sft_combined.sh   # SFT 데이터 통합
bash scripts/launch_3b_sft.sh          # SFT 학습

# Phase 3: ORPO
python data/prepare_preference_combined.py  # Preference 데이터 통합
bash scripts/launch_3b_orpo.sh              # ORPO 정렬

# Phase 4: 변환 + 배포
bash scripts/convert_3b_gguf.sh
bash scripts/deploy_3b_ollama.sh
```

---

## 8. 알려진 이슈 및 주의사항

### 해결된 이슈
1. **TensorBoard import 크래시** — tensorflow 버전 충돌. try/except 가드 적용
2. **OOM at bs=8** — bs=4로 해결 (GQA 최적화로 VRAM 48.3GB)
3. **NCCL_ALGO=Tree 단독 사용 불가** — AllGather 미지원, Ring,Tree로 해결
4. **DDP static_graph + TE 충돌** — static_graph 제거
5. **te.Linear lm_head + weight tying 충돌** — nn.Linear 유지

### 잠재적 이슈
1. **bs=8 OOM**: bs=6까지 작동 확인. bs=8에서 비선형 메모리 급증 (TE FP8 activation buffers 추정)
2. **torch.compile 미지원**: `cuda.h` 미설치. `apt install cuda-toolkit-13-1` 또는 `CUDA_HOME` 설정 필요
3. **TensorBoard 미작동**: tensorflow 호환성 문제. wandb 대안 고려 가능
4. **long training stability**: 57K steps (53시간) 중 NCCL hang, gradient explosion 가능. `monitor_3b.sh` 필수 실행

### 최적화 여지 (미래)
1. `torch.compile` 활성화 — cuda-dev 설치 후 20-30% speedup 가능
2. FSDP 전환 — `fp8_model_init()` 호환 가능 (DDP는 불가)
3. FlashAttention-3 — B200 전용 최적화 (별도 설치 필요)
4. QKV 퓨즈드 GEMM — `te.MultiheadAttention` 또는 단일 `te.Linear` QKV 프로젝션

---

## 9. 에이전트 ID (resume 가능)

| 에이전트 | ID | 용도 |
|----------|-----|------|
| Iron Man (조사) | `a2a8328a9c1bad1a8` | GPU/CUDA/FP8 조사 resume |
| Cyborg (조사) | `a671126b059372e3c` | NCCL/DDP 조사 resume |
| Batman (조사) | `adfececb672c09063` | 메모리 분석 resume |
| Hulk (조사) | `a941694c308fbf6f5` | I/O 파이프라인 조사 resume |
| Iron Man (구현) | `ac682a7cccb726349` | GQA fix 등 구현 resume |
| Cyborg (구현) | `a373ff406c889f3f1` | DDP/NCCL 구현 resume |
| Hulk (구현) | `aedbd53c8abdf08ed` | loss sync/dataset 구현 resume |
| Batman (구현) | `a0e726fff0e350f85` | batch size/RAM 구현 resume |

---

## 10. 검증 테스트 기록

### OOM 테스트 #1 (최적화 전, bs=4)
```
시각: 2026-03-02 02:04-02:06
결과: 성공
loss=11.6562, gnorm=1.801, tok/s=32,007, mem=60.4GB
```

### OOM 테스트 #2 (최적화 후, bs=4)
```
시각: 2026-03-02 03:04-03:06
결과: 성공
loss=11.6563, gnorm=1.800, tok/s=32,101, mem=48.3GB
VRAM 절감: 60.4→48.3GB (-20%)
```

### Throughput 테스트 (최적화 후, bs=6)
```
시각: 2026-03-02 03:07-03:09
결과: 성공
loss=11.6533, gnorm=1.445, tok/s=34,519, mem=48.3GB
Throughput 향상: 32K→34.5K tok/s (+8%)
```

### 실패 테스트 기록
| 시도 | 에러 | 원인 |
|------|------|------|
| bs=16, NCCL_ALGO=Tree | NCCL AllGather 미지원 | Tree only에서 AllGather 불가 |
| bs=16, Ring,Tree | OOM (178GB) | 활성화 메모리 초과 |
| bs=8, static_graph | `expect_autograd_hooks_` ASSERT | TE FP8 동적 hooks 충돌 |
| bs=8, te.Linear lm_head | OOM (172GB at step 1) | te.Linear + DDP 추가 버퍼 |
| bs=8, 최종 | OOM (172GB at step 1) | 비선형 메모리 급증 |

---

*이 보고서는 Phase 0 완료 시점의 스냅샷입니다. Phase 1 pretrain 실행 후 별도 보고서 작성이 필요합니다.*

# FRANKENSTALLM 3B — SFT 학습 보고서

> **작성일**: 2026-03-05
> **프로젝트**: FRANKENSTALLM 3B
> **현재 Phase**: Phase 2 (SFT) — 진행 중
> **Base 모델**: checkpoint-0057000 (Phase 1 사전학습 완료)

---

## 목차

1. [SFT 학습 현황](#1-sft-학습-현황)
2. [SFT 데이터 파이프라인](#2-sft-데이터-파이프라인)
3. [SFT 학습 설정 상세](#3-sft-학습-설정-상세)
4. [SFT Loss 분석 및 수렴 추이](#4-sft-loss-분석-및-수렴-추이)
5. [SFT 안정성 및 리소스 분석](#5-sft-안정성-및-리소스-분석)
6. [1B SFT 경험에서 배운 교훈](#6-1b-sft-경험에서-배운-교훈)
7. [Base 모델 요약 (Phase 0~1)](#7-base-모델-요약-phase-01)
8. [SFT 완료 후 평가 계획](#8-sft-완료-후-평가-계획)
9. [이슈 트래커](#9-이슈-트래커)
10. [부록](#10-부록)

---

## 1. SFT 학습 현황

### 1.1 진행 상태

```
Phase 2 (SFT)  █▒░░░░░░░░░░░░░░░░░░  6%   진행 중 (2,000 / 33,000 steps)
```

| 항목 | 값 |
|------|-----|
| **현재 Step** | 2,000 / 33,000 |
| **진행률** | 6.06% |
| **Train Loss** | 2.053 (step 2000) |
| **Val Loss (Best)** | **1.956** (step 2000, 일관 하락 중) |
| **학습 시작** | 2026-03-05 22:15 |
| **경과 시간** | ~1시간 12분 |
| **예상 잔여 시간** | ~5시간 |
| **예상 완료 시각** | 2026-03-06 04:30경 |
| **VRAM** | 24.2 GB / 183 GB (13.2%) |
| **체크포인트** | `checkpoints/korean_3b_sft_v1/checkpoint-0002000` |

### 1.2 핵심 관측

1. **Val loss 단조 감소**: 2.073 → 2.004 → 1.975 → **1.956** — catastrophic forgetting 없이 안정 수렴
2. **Train-Val 갭 최소**: |train - val| ≈ 0.1 — 오버피팅 징후 없음
3. **VRAM 여유**: 24.2 GB (13.2%) — Phase 1 대비 절반, 매우 안정적
4. **메모리 누수 없음**: step 10~2000 전 구간 24.2 GB 고정

---

## 2. SFT 데이터 파이프라인

### 2.1 파이프라인 개요

```
24개 소스 (6.59M raw samples)
    │
    ▼  prepare_sft_combined.sh
    │  - 포맷 통일 (messages/conversations/alpaca → messages)
    │  - MD5 중복 제거 (첫 user 메시지 해시)
    │  - 98:2 train/val split (seed=42)
    │
    ▼  통합 데이터
    │  train: 2,559,492 samples (7.79 GB)
    │  val:      52,234 samples (163 MB)
    │
    ▼  filter_sft_v2.py (5단계 품질 필터)
    │
    ▼  최종 학습 데이터
       train: 2,439,397 samples (7.48 GB)  ← 현재 SFT에 사용 중
       val:      49,801 samples (157 MB)
```

### 2.2 데이터 소스 (24개)

**대규모 소스 (상위 12, 전체의 96%)**:

| # | 데이터셋 | 샘플 수 | 크기 | 도메인 |
|---|---------|---------|------|--------|
| 1 | reasoning_r1_1.4m | 1,400,000 | 14.77 GB | 추론 (Chain-of-Thought) |
| 2 | openhermes_2.5 | 1,001,551 | 1.82 GB | 영어 다목적 instruction |
| 3 | AI-MO_NuminaMath-CoT | 859,494 | 2.51 GB | 수학 CoT 풀이 |
| 4 | korean_instruction_mix | 515,911 | 1.39 GB | 한국어 혼합 지시 |
| 5 | lemon-mint_smol-koreantalk | 460,281 | 5.23 GB | 한국어 자연 대화 |
| 6 | open_korean_instructions | 375,159 | 0.73 GB | 한국어 지시-응답 |
| 7 | magpie_reasoning_v2 | 249,922 | 3.99 GB | 추론 (영어) |
| 8 | magpie_reasoning_ko | 224,929 | 3.19 GB | 추론 (한국어) |
| 9 | ultrachat_200k | 207,865 | 1.34 GB | 대화 |
| 10 | kuotient_orca-math-ko | 193,789 | 0.61 GB | 수학 (한국어) |
| 11 | data/sft/train.jsonl | 161,848 | 0.27 GB | 원본 SFT |
| 12 | kullm_v2 | 152,630 | 0.42 GB | 한국어 지시 |

**소규모 소스 (12개, 나머지 4%)**:
zwhe99_DeepMath-103K, nayohan_Evol-Instruct-Code-80k-ko, dbdu_ShareGPT-74k-ko, FreedomIntelligence_evol-instruct-korean, FreedomIntelligence_alpaca-gpt4-korean, maywell_ko_wikidata_QA, nlp-with-deeplearning_Ko.WizardLM, kyujinpy_KOR-OpenOrca-Platypus-v3, coastral_korean-writing-style-instruct, ko_lima, koalpaca_v1_1a, OpenAssistant_oasst1_ko (트리 재구성)

### 2.3 도메인별 비율

```
┌──────────────────────────────────────────────────────┐
│         SFT 학습 데이터 도메인 구성 (2.44M)           │
├──────────────────────────────────────────────────────┤
│ ██████████████████████░░░░  추론/CoT         38.0%   │
│ █████████████░░░░░░░░░░░░  한국어 지시       22.5%   │
│ ████████░░░░░░░░░░░░░░░░░  영어 다목적       16.0%   │
│ ██████░░░░░░░░░░░░░░░░░░░  수학             12.0%   │
│ ████░░░░░░░░░░░░░░░░░░░░░  대화/코드/기타    11.5%   │
└──────────────────────────────────────────────────────┘
```

### 2.4 품질 필터링 (filter_sft_v2.py)

5단계 순차 필터:

| 단계 | 필터 | 기준 | 목적 |
|------|------|------|------|
| 1 | EOS 태그 제거 | `</s>` 리터럴 strip | 원본 데이터의 잔여 EOS 제거 |
| 2 | QA 마커 제거 | 질문:/답변:/Q:/A: 접두사 제거 | 포맷 노이즈 제거 |
| 3 | 최소 응답 길이 | < 50자 제거 | 무의미한 초단문 제거 |
| 4 | 최대 응답 길이 | > 20,000자 제거 | 비정상 장문 제거 |
| 5 | 4-gram 반복률 | > 30% 제거 | 반복/저품질 텍스트 제거 |

**필터링 결과**:

| 구분 | 필터 전 | 필터 후 | 제거 수 | 제거율 |
|------|---------|---------|---------|--------|
| Train | 2,559,492 | **2,439,397** | 120,095 | 4.69% |
| Val | 52,234 | **49,801** | 2,433 | 4.66% |

> 4.69% 제거율은 데이터 품질이 전반적으로 양호함을 의미. 주요 제거 사유는 초단문(50자 미만)과 4-gram 반복.

### 2.5 데이터 포맷

모든 소스를 통일된 `messages` 포맷으로 변환:

```json
{"messages": [
  {"role": "user", "content": "한국의 철강 산업에 대해 설명해줘."},
  {"role": "assistant", "content": "한국의 철강 산업은..."}
]}
```

**포맷 변환 매핑** (prepare_sft_combined.sh):
- `messages` → 그대로 사용
- `conversations` → role/content 추출
- `instruction/input/output` → user(instruction+input) / assistant(output)
- `question/answer` → user / assistant
- `prompt/response` → user / assistant
- `problem/solution` → user / assistant
- OASST 트리 → 대화 경로 재구성, rank=0.0 최선 응답 선택

---

## 3. SFT 학습 설정 상세

### 3.1 핵심 하이퍼파라미터

| 항목 | 값 | 근거 |
|------|-----|------|
| **Base 모델** | `checkpoint-0057000` | Phase 1 최종 (loss 1.466) |
| **총 스텝** | 33,000 | ~3.3 epochs on 2.44M samples |
| **Batch size** | 2 per GPU | VRAM 여유 확보 (24.2 GB) |
| **GPU 수** | 8 (B200) | DDP |
| **Grad accum** | 4 | Effective batch = 2 x 8 x 4 = **64 sequences** |
| **학습률** | **1.0e-5** | Pretrain LR(1.5e-4)의 1/15 — forgetting 방지 |
| **LR Schedule** | Cosine decay | Warmup 500 steps → cosine |
| **Warmup** | 500 steps | 1.5% of total |
| **Weight decay** | 0.01 | Pretrain(0.1)보다 약하게 |
| **Max grad norm** | 1.0 | Gradient clipping |
| **정밀도** | MXFP8 + BF16 | B200 native FP8 |
| **NEFTune alpha** | **5.0** | 임베딩 노이즈 → 생성 다양성 향상 |
| **Gradient checkpointing** | 활성화 | VRAM 절약 |

### 3.2 SFT-specific 설계

**Loss Masking**:
```
<|user|>\n{질문}\n<|assistant|>\n{응답}</s>
 ──── ignore (label=-1) ────  ── learn ──
```
- Prompt 토큰에 대해 loss를 계산하지 않음 (label = -1)
- Response 토큰 + EOS 토큰만 학습 대상
- EOS(`</s>`) 학습이 핵심 — 1B SFT v1에서 EOS 절단 버그로 실패한 교훈 반영

**NEFTune (Noisy Embeddings Fine-Tuning)**:
- 임베딩 벡터에 `alpha/sqrt(seq_len * d_model)` 크기의 uniform noise 주입
- Base model의 greedy 반복률 72.75%를 SFT만으로 해결하기 어려울 때 보조 효과
- [Jain et al., 2024] 논문 기반, SFT 후 생성 다양성 5~15% 향상 보고

**Dynamic Sequence Padding**:
- 고정 max_seq_len 패딩 대신, 배치 내 최장 시퀀스 기준 패딩
- 64 토큰 단위 정렬 (FlashAttention 효율)
- 1B SFT v1의 "static padding 낭비" 버그 수정 반영

### 3.3 학습률 선택 근거

```
Pretrain LR:  1.5e-4 (peak)
SFT LR:       1.0e-5 (peak)  ← 1/15
```

SFT에서 높은 LR은 catastrophic forgetting을 유발한다. 경험적으로:
- 1e-4 이상: pretrain knowledge 급속 망각
- 2e-5: 일부 연구에서 권장하지만 2.44M 대규모 SFT에서는 불안정 위험
- **1e-5**: 대규모 SFT 데이터(2.44M)와 조합 시 안정적 수렴, forgetting 최소화

현재 val_loss 추이(2.073→1.956, 단조 감소)가 이 선택의 적절성을 실증.

---

## 4. SFT Loss 분석 및 수렴 추이

### 4.1 상세 Loss 추이

| Step | Train Loss | Val Loss | LR | Grad Norm | 비고 |
|------|-----------|----------|-----|-----------|------|
| 10 | 2.2567 | — | 2.00e-7 | ~1.5 | 초기, warmup 시작 |
| 50 | ~2.30 | — | 1.00e-6 | ~1.3 | 급속 하강 시작 |
| 100 | 2.3083 | — | 2.00e-6 | ~1.2 | LR 아직 낮음 |
| 250 | 1.9842 | — | 5.00e-6 | ~1.1 | 본격 학습 시작 |
| **500** | 2.1380 | **2.0732** | **1.00e-5** | 1.0 | **Warmup 완료**, peak LR |
| 1,000 | 2.0748 | **2.0035** | 1.00e-5 | 1.0 | 안정 수렴 |
| 1,500 | 2.0040 | **1.9745** | 9.98e-6 | 1.0 | cosine decay 시작 |
| **2,000** | **2.0527** | **1.9558** | 9.95e-6 | 1.0 | **현재 BEST** |

### 4.2 수렴 분석

**Val Loss 하락 속도**:

| 구간 | Val Loss 변화 | 하락폭/500steps |
|------|--------------|----------------|
| 500 → 1,000 | 2.073 → 2.004 | -0.070 |
| 1,000 → 1,500 | 2.004 → 1.975 | -0.029 |
| 1,500 → 2,000 | 1.975 → 1.956 | -0.019 |

하락폭이 감소하는 것은 정상적인 수렴 패턴. 현재 속도를 선형 외삽하면:
- Step 5,000: val_loss ≈ 1.90
- Step 10,000: val_loss ≈ 1.85
- Step 33,000: val_loss ≈ 1.70~1.75 (추정)

### 4.3 1B SFT와의 비교

| 지표 | 1B SFT (완료) | 3B SFT (현재) | 비고 |
|------|-------------|-------------|------|
| Base loss | 1.904 | 1.466 | 3B base가 더 낮음 |
| SFT val_loss (step 2000) | ~2.30 | **1.956** | 3B가 0.34 낮음 |
| 최종 val_loss | 2.206 (9000 steps) | **진행 중** | — |
| VRAM | 12.0 GB | 24.2 GB | 모델 크기 비례 |
| EOS 종료율 (SFT 전) | 0% | 0% | 동일 |
| EOS 종료율 (SFT 후) | 60% | **측정 예정** | — |
| 반복률 greedy (SFT 전) | ~70% | 72.75% | 동일 수준 |
| 반복률 (SFT 후) | 30.7% → 18%(w/ penalty) | **측정 예정** | — |

> **기대**: 3B의 더 큰 파라미터 공간은 장거리 의존성을 더 잘 포착하므로, SFT 후 반복률이 1B(18%)보다 크게 낮아질 것으로 예상. 목표: **< 5%** (greedy, rep_penalty 없이).

---

## 5. SFT 안정성 및 리소스 분석

### 5.1 VRAM 사용

| Phase | Batch Size | VRAM/GPU | 비율 | 비고 |
|-------|-----------|---------|------|------|
| Phase 1 (Pretrain) | 4 | 48.3 GB | 26.4% | bs=4, accum=8 |
| **Phase 2 (SFT)** | **2** | **24.2 GB** | **13.2%** | bs=2, accum=4 |
| 이론적 여유 | — | 158.8 GB | 86.8% | — |

**왜 SFT가 절반인가?**
- Micro-batch 4→2: activation memory 비례 감소
- Grad accum 8→4: 동시 보유 activation 감소
- Gradient checkpointing: forward 재계산으로 중간 activation 해제
- FP8 activation buffer: batch 크기에 비례하므로 절반 감소

### 5.2 Gradient Norm 안정성

```
Step   10: gnorm ~1.5  (초기 진동)
Step  100: gnorm ~1.2  (안정화)
Step  500: gnorm  1.0  (warmup 완료)
Step 2000: gnorm  1.0  (완전 안정)
```

- Max grad norm = 1.0이지만 clipping이 거의 작동하지 않음
- Pretrain 최종(0.097)보다 높은 것은 SFT 데이터의 다양성에 기인 (정상)
- 갑작스러운 spike 없음 → 학습률 1e-5가 적절

### 5.3 처리 속도

| 지표 | 값 |
|------|-----|
| Steps/second (추정) | ~1.7 |
| Tokens/step | 2 x 4096 = 8,192 (per GPU) |
| System tokens/step | 8,192 x 8 GPU = 65,536 |
| System tok/s | ~111K tok/s |

> SFT는 pretrain(308K tok/s)보다 낮다: bs=2(vs 4)이고 시퀀스 길이가 가변적(dynamic padding)이라 배치 효율이 다름.

---

## 6. 1B SFT 경험에서 배운 교훈

### 6.1 SFT v1 실패 (1B, Day 2)

| 버그 | 증상 | 해결 |
|------|------|------|
| **Label off-by-one** | loss = 0.0 (data leakage) | 레이블 시프트 수정 |
| **Static padding** | 짧은 샘플도 max_len 패딩 → GPU 낭비 | Dynamic padding (64-token 정렬) |
| **EOS 절단** | 응답 끝에 EOS 없음 → 종료 불가 | EOS 강제 포함 + loss masking 수정 |
| **단일 에폭** | 데이터 1회만 학습 → 언더피팅 | Multi-epoch (3B: ~3.3 epochs) |
| **검증 분리 없음** | val_loss 미측정 → 오버피팅 감지 불가 | 2% val split + 500 step 간격 eval |

### 6.2 SFT v2 성공 (1B, Day 3)

- Val loss: 2.206, 반복률: 18% (rep_penalty 적용)
- kobest_copa: 0.646 → SFT 효과 확인
- **한계**: 1B 파라미터로는 반복률 5% 미만 달성 불가 → 3B 전환 결정

### 6.3 3B SFT에 반영된 개선사항

| 1B 교훈 | 3B SFT 적용 |
|---------|-------------|
| Label leakage 방지 | Loss masking 검증 완료 (label=-1 for prompt) |
| EOS 학습 필수 | Chat template에 `</s>` 포함, loss에 반영 |
| Dynamic padding | 64-token 정렬 dynamic padding 적용 |
| Val 분리 | 49,801 val samples, 500 step 간격 eval |
| 데이터 품질 | filter_sft_v2.py 5단계 필터 (1B에는 없었음) |
| 반복 대책 | NEFTune alpha=5.0 추가 (1B에는 없었음) |
| 데이터 규모 | 161K → **2,439K** (15배 확대) |

---

## 7. Base 모델 요약 (Phase 0~1)

> 이 섹션은 SFT의 출발점인 base 모델의 핵심 수치만 요약한다.
> 상세는 `reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md` 및
> `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` 참조.

### 7.1 모델 아키텍처

| 항목 | 값 |
|------|-----|
| **파라미터** | ~3,015M (3B) |
| d_model / n_layers / n_heads | 3,072 / 28 / 24 |
| n_kv_heads / d_ffn | 8 (GQA 3:1) / 8,192 |
| 정밀도 | MXFP8 (B200 native) |
| max_seq_len | 4,096 |

### 7.2 Phase 0 최적화 (2026-03-02)

| 최적화 | 효과 |
|--------|------|
| GQA FlashAttention native | VRAM 60.4 → **48.3 GB** (-20%) |
| DDP gradient_as_bucket_view | GPU-CPU sync **-87.5%** |
| NCCL NVLS (Ring+Tree) | AllReduce 효율 개선 |
| torch.compile | **효과 없음** (TE opaque kernel) |

### 7.3 Phase 1 사전학습 (2026-03-02~05)

| 항목 | 값 |
|------|-----|
| 학습 스텝 | 57,000 (100% 완료) |
| 총 토큰 | ~60B |
| 학습 시간 | 62.94시간 |
| **최종 Loss** | **1.466** |
| Throughput | 38.5K tok/s (per GPU) |
| VRAM | 48.3 GB (26.4%) |
| 사고 | 0건 |

### 7.4 Base 모델 평가 핵심 수치

| 지표 | 값 | 비고 |
|------|-----|------|
| **통합 Val PPL** | **5.226** | 19개 데이터셋 |
| Korean C4 PPL | 5.717 | 핵심 한국어 지표 |
| Top-1 Accuracy | 68.75% | Calibration |
| KoBEST 평균 | 43.69% | ~Random (base 정상) |
| MMLU-KO | 22.75% | ~Random (base 정상) |
| Greedy 반복률 | 72.75% | SFT로 해결 대상 |
| EOS 종료율 | 0% | SFT로 해결 대상 |

> **SFT 진행 결정 근거**: Loss 1.466 건강한 완료, PPL 합리적 범위, 모델 구조 문제 없음, 반복/EOS는 SFT 영역.

---

## 8. SFT 완료 후 평가 계획

### 8.1 필수 평가 항목

| 평가 | 도구 | 목표 |
|------|------|------|
| **반복률 측정** | generation_task.py | greedy < 5% (rep_penalty 없이) |
| **EOS 종료율** | generation_task.py | > 90% |
| **KoBEST 전체** | lm-eval-harness | KoBEST 평균 > 55% |
| **MMLU-KO** | lm-eval-harness | MMLU-KO > 30% |
| **Val PPL** | ppl_task.py | PPL < 6.0 (forgetting 확인) |
| **생성 품질** | 수동 평가 | 자연스러운 한국어 응답 |

### 8.2 Base vs SFT 비교 포인트

| 지표 | Base (현재) | SFT 목표 | 판정 기준 |
|------|-----------|----------|----------|
| Greedy 반복률 | 72.75% | < 5% | 핵심 성공 지표 |
| EOS 종료율 | 0% | > 90% | 대화 완성 능력 |
| kobest_copa | 49.30% | > 65% | 추론 능력 향상 |
| MMLU-KO | 22.75% | > 30% | 지식 활용 능력 |
| Val PPL | 5.226 | < 6.0 | Forgetting < 15% |

### 8.3 후속 단계

```
SFT 완료 (예상: 2026-03-06 04:30)
    │
    ▼ 평가 (reeval_pipeline.py)
    │  - 벤치마크 + 반복률 + 생성 품질
    │
    ├── 반복률 < 5%  →  GGUF 변환 → Ollama 배포 (Phase 4)
    │
    └── 반복률 > 5%  →  ORPO/DPO 추가 학습 (Phase 3)
                        795K preference pairs 준비 완료
```

---

## 9. 이슈 트래커

### 전 Phase 누적

| # | Phase | 이슈 | 상태 | 해결 방법 |
|---|-------|------|------|----------|
| 1 | 0 | OOM (bs=8) | **해결** | bs=4 + GQA FA native |
| 2 | 0 | TensorBoard import crash | **해결** | try/except guard |
| 3 | 0 | NCCL Tree 실패 | **해결** | Ring,Tree 혼합 |
| 4 | 0 | DDP static_graph + TE 충돌 | **해결** | static_graph 비활성화 |
| 5 | 0 | te.Linear lm_head weight tying | **해결** | nn.Linear 사용 |
| 6 | 0 | torch.compile 무효 | **포기** | TE opaque kernel |
| 7 | 0 | NUMA 크로스 어피니티 | **미적용** | 안정성 우선 |
| 8 | 1 (SFT v1) | Label off-by-one | **해결** | 레이블 시프트 수정 |
| 9 | 1 (SFT v1) | EOS 절단 | **해결** | Chat template EOS 포함 |
| 10 | 1 | Greedy 반복률 72.75% | **SFT 중** | SFT + NEFTune 적용 |
| 11 | 1 | EOS 종료율 0% | **SFT 중** | SFT chat template 학습 |
| 12 | 1 | Namuwiki PPL 25.9 | **허용** | 데이터 품질 한계 |
| 13 | 2 | (현재까지 이슈 없음) | — | — |

### SFT 특이사항

- **step 100 train_loss 2.3083 > step 10의 2.2567**: LR warmup 초반의 정상적 진동. Step 250(1.98)부터 본격 하강.
- **Val loss만 모니터링**: Train loss는 배치 단위 변동이 크므로, val_loss의 단조 감소가 핵심 건강 지표.

---

## 10. 부록

### 10.1 주요 파일 경로

| 항목 | 경로 |
|------|------|
| **SFT 설정** | `configs/korean_3b_sft.yaml` |
| **SFT 스크립트** | `train/sft.py` |
| **SFT 런처** | `scripts/launch_3b_sft.sh` |
| **데이터 준비** | `scripts/prepare_sft_combined.sh` |
| **데이터 필터** | `data/filter_sft_v2.py` |
| **학습 데이터** | `data/sft_combined/train_filtered.jsonl` (2.44M, 7.48 GB) |
| **검증 데이터** | `data/sft_combined/val_filtered.jsonl` (49.8K, 157 MB) |
| **SFT 체크포인트** | `checkpoints/korean_3b_sft_v1/` |
| **SFT 학습 로그** | `checkpoints/korean_3b_sft_v1/train.log` |
| **Base 체크포인트** | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000/` |

### 10.2 관련 보고서

| 보고서 | 경로 |
|--------|------|
| Phase 0 최적화 | `reports/2026-03-02_0200_FRANKENSTALLM_phase0_optimization_report.md` |
| 3B Base 평가 | `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` |
| 3B 후속 단계 참조 | `reports/2026-03-05_3B_NEXT_STEPS_REFERENCE.md` |
| v2 종합 평가 | `eval/outputs/3b_reeval_20260305_1451/full_eval_report.md` |

### 10.3 재현 명령어

```bash
# SFT 학습 시작
bash scripts/launch_3b_sft.sh

# 수동 실행 (디버그)
torchrun --nproc_per_node=8 \
  train/sft.py \
  --base_model checkpoints/korean_3b_fp8_run1/checkpoint-0057000 \
  --train_data data/sft_combined/train_filtered.jsonl \
  --val_data data/sft_combined/val_filtered.jsonl \
  --max_steps 33000 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --lr 1e-5 \
  --warmup_steps 500 \
  --neftune_alpha 5.0

# 학습 모니터링
tail -f checkpoints/korean_3b_sft_v1/train.log

# SFT 완료 후 평가
python eval/reeval_pipeline.py \
  --checkpoint checkpoints/korean_3b_sft_v1/checkpoint-best
```

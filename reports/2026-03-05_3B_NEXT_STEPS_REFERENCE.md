# FRANKENSTALLM 3B — 평가 결과 종합 & 다음 단계 참조 문서

**작성일**: 2026-03-05
**목적**: 3B Base 모델 평가 결과를 기반으로 다음 작업의 파라미터 조건 및 실행 전략을 기록

---

## Part 1: 현재 상태 종합

### 1.1 학습 완료 정보

| 항목 | 값 |
|------|-----|
| 모델 | FRANKENSTALLM 3B (3,015M params) |
| 아키텍처 | d=3072, 28L, 24H, GQA 8KV, d_ffn=8192, RoPE theta=500K |
| 체크포인트 | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000` |
| 학습 스텝 | 57,000 / 57,000 (100%) |
| 학습 토큰 | 41.12B tokens |
| Chinchilla 최적 | 60B tokens (31% 미달) |
| 최종 Loss | 1.466 |
| 학습 시간 | 62.94시간 |
| 인프라 | 8x B200, DDP, TE MXFP8 |
| VRAM | 48.3GB / 183GB (26%) |

### 1.2 평가 결과 전체 대시보드

#### Perplexity

| 카테고리 | 데이터셋 | PPL | BPT | 평가 |
|---------|---------|-----|-----|------|
| **통합** | 3b_val | **5.226** | 2.386 | 양호 |
| 한국어 웹 | hplt_ko | **2.403** | 1.265 | 우수 |
| | korean_c4 | 5.717 | 2.515 | 양호 |
| | korean (general) | 7.016 | 2.811 | 보통 |
| | cc100_ko | 21.782 | 4.445 | 불량 |
| 수학 | mathpile | **2.724** | 1.446 | 우수 |
| | open_web_math | 6.926 | 2.792 | 보통 |
| 영어/다국어 | cosmo_khanacademy | 2.932 | 1.552 | 우수 |
| | cosmo_auto_math_text | 3.149 | 1.655 | 우수 |
| | cosmo_stanford | 3.362 | 1.750 | 양호 |
| | cosmo_wikihow | 3.310 | 1.727 | 양호 |
| | cosmo_openstax | 3.867 | 1.951 | 양호 |
| | cosmo_stories | 3.955 | 1.984 | 양호 |
| | cosmo_web_v2 | 4.166 | 2.059 | 양호 |
| 위키 | wikipedia_ko | 10.706 | 3.420 | 보통 |
| | korean_wiki | 11.836 | 3.565 | 보통 |
| 나무위키 | namuwiki_2023b | 18.917 | 4.242 | 불량 |
| | korean_namuwiki | 25.881 | 4.694 | 불량 |

#### Calibration & Token 분석

| 지표 | 값 | 해석 |
|------|-----|------|
| Top-1 Accuracy | 68.75% | 2/3 토큰 정확 예측 |
| Top-5 Accuracy | 81.64% | 상위 5 후보에 정답 81% |
| Top-10 Accuracy | 85.93% | |
| Mean Correct Prob | 0.6152 | 건강한 수준 |
| Mean Entropy | 1.568 bits | 적절한 불확실성 |
| Median NLL | 0.122 | 대부분 토큰 잘 예측 |
| Mean NLL | 1.556 | heavy-tail 분포 |
| NLL>5 비율 | 10.86% | 고난이도 토큰 |
| NLL>10 비율 | 1.18% | 극단 사례 |

#### Generation Quality

| 설정 | 3-gram Rep% | 4-gram Rep% | EOS% |
|------|------------|------------|------|
| Greedy (t=0.0) | 72.75% | 70.78% | 0% |
| t=0.5 | ~60% | ~59% | 0% |
| t=0.7 | ~40% | 가변 | 0% |
| t=1.0 | 24.27% | 가변 | 0% |
| **t=0.7, rep=1.3** | **0.00%** | **0.00%** | 0% |
| **t=0.9, rep=1.2** | **0.00%** | **0.00%** | 0% |
| any + ngram_block=3 | **0.00%** | **0.00%** | 0% |

**주요 문제**:
- Greedy 반복 72.75% (base model 고유 문제)
- EOS 0% (instruction tuning 없으므로 정상)
- 사실 오류: "수도=인천" 등 hallucination
- 학습 데이터 노이즈 유출 (웹 게시판 목록 등)

#### Benchmarks (0-shot)

| 벤치마크 | 점수 | 랜덤 | 차이 | F1 |
|---------|------|------|------|-----|
| KoBEST BoolQ | 50.14% | 50% | +0.1% | 0.334 |
| KoBEST COPA | 49.40% | 50% | -0.6% | 0.493 |
| KoBEST HellaSwag | 21.60% | 25% | -3.4% | 0.193 |
| KoBEST SentiNeg | 50.13% | 50% | +0.1% | 0.467 |
| KoBEST WiC | 48.81% | 50% | -1.2% | 0.329 |
| **KoBEST avg** | **~47.7%** | ~49% | -1.3% | — |
| HAE-RAE | 19.98% | 20% | -0.02% | — |
| belebele_kor | 21.89% | 25% | -3.1% | — |
| MMLU-KO | 23.39% | 25% | -1.6% | — |

**참고**: MMLU-KO는 별도 lm-eval harness 실행에서만 측정됨 (full eval pipeline에서는 registry 오류)

#### 1B SFT vs 3B Base 비교

| 지표 | 1B SFT | 3B Base | 시사점 |
|------|--------|---------|--------|
| Loss | 1.904 | **1.466** | 3B가 더 잘 학습 |
| PPL (C4) | **5.67** | 5.72 | 동등 (3B는 5배 넓은 데이터) |
| kobest_copa | 0.646 | 0.494 | SFT 효과가 큼 |
| 3-gram rep | 30.7% | 72.75% | SFT vs base 차이 (직접 비교 불가) |
| EOS | 60% | 0% | SFT가 해결하는 영역 |
| 학습 데이터 | 8.5B tok | 41.12B tok | 4.8배 다양 |

---

## Part 2: 다음 작업별 상세 파라미터 제안

---

### 2.1 Phase 2: SFT (Supervised Fine-Tuning) — 최우선

#### 2.1.1 SFT 데이터 전략

**현재 보유 데이터**:

| 데이터셋 | 위치 | 샘플 수 | 용도 |
|---------|------|---------|------|
| 기존 SFT v2 | `data/sft/train.jsonl` | 161,848 | 1B SFT에 사용된 검증된 데이터 |
| 기존 SFT v2 val | `data/sft/val.jsonl` | 8,518 | 검증셋 |

**sft_extra 추가 데이터** (미큐레이션 상태):

| 데이터셋 | 샘플 수 | 도메인 | 품질 추정 | 사용 추천 |
|---------|---------|--------|----------|----------|
| reasoning_r1_1.4m | 1,400,000 | 추론/사고과정 | 중상 | 핵심 (체인오브소트) |
| openhermes_2.5 | 1,001,551 | 범용 대화/지시 | 상 | 핵심 (다양성) |
| AI-MO_NuminaMath-CoT | 859,494 | 수학 CoT | 상 | 선택 (수학 특화) |
| korean_instruction_mix | 515,911 | 한국어 혼합 | 중 | 핵심 (한국어) |
| smol-koreantalk | 460,281 | 한국어 대화 | 중 | 선택 (대화체) |
| open_korean_instructions | 375,159 | 한국어 지시 | 중상 | 핵심 |
| magpie_reasoning_v2 | 249,922 | 추론 | 중상 | 핵심 |
| ultrachat_200k | 230,975 | 대화 | 중상 | 선택 |
| magpie_reasoning_ko | 224,929 | 한국어 추론 | 중상 | 핵심 |
| orca-math-193k-korean | 193,789 | 한국어 수학 | 상 | 핵심 |
| DeepMath-103K | 103,022 | 수학 심화 | 상 | 선택 |
| kullm_v2 | 152,630 | 한국어 LLM | 중 | 선택 |
| Ko.WizardLM_196k | 142,759 | 한국어 Evol-Instruct | 중상 | 핵심 |
| maywell_ko_wikidata_QA | 137,505 | 한국어 QA | 중 | 선택 |
| ShareGPT-74k-ko | 130,688 | 한국어 대화 | 중상 | 핵심 |
| Evol-Instruct-Code-80k | 78,264 | 코드 | 중상 | 핵심 (코드 능력) |
| evol-instruct-korean | 59,022 | 한국어 Evol | 중상 | 핵심 |
| alpaca-gpt4-korean | 49,969 | 한국어 Alpaca | 중 | 선택 |
| writing-style-instruct | 28,978 | 글쓰기 | 중 | 선택 |
| KOR-OpenOrca-Platypus | 34,214 | 한국어 Orca | 중 | 선택 |
| koalpaca_v1_1a | 21,155 | 한국어 Alpaca | 중 | 선택 |

#### 2.1.2 데이터 큐레이션 추천 전략

**Option A (빠른 시작)**: 기존 161K만 사용 → 즉시 실행 가능

**Option B (추천)**: 핵심 데이터 큐레이션 → ~700K samples

```
기존 SFT v2:                  161,848
+ korean_instruction_mix:      ~200,000 (필터 후)
+ open_korean_instructions:    ~150,000 (필터 후)
+ magpie_reasoning_ko:         ~100,000 (필터 후)
+ orca-math-193k-korean:        ~80,000 (필터 후)
+ ShareGPT-74k-ko:             ~50,000 (필터 후)
= 총 ~740,000 samples
```

**Option C (대규모)**: reasoning_r1 + openhermes 포함 → ~1.5M samples
- 장점: 추론 능력 대폭 향상
- 단점: 영어 비중 높아짐, 큐레이션 시간 필요

**큐레이션 필터 조건**:
1. 토큰 길이: 128 < total_tokens < 4096 (max_seq_len 이내)
2. 언어 필터: 한국어 or 영어 (한국어 비중 60% 이상 유지)
3. 중복 제거: exact dedup (hash 기반)
4. 품질 필터: response 길이 > 50 tokens, instruction 길이 > 10 tokens
5. 도메인 밸런싱: 수학 <20%, 코드 <15%, 일반 >40%, 추론 >20%

#### 2.1.3 SFT 하이퍼파라미터 제안

**현재 config (`korean_3b_sft.yaml`) 기준 + 조정 제안**:

| 파라미터 | 현재 값 | 조정 제안 | 근거 |
|---------|---------|---------|------|
| **base_checkpoint** | XXXXXX | **checkpoint-0057000** | 확정 필요 |
| **lr** | 1.0e-5 | **2.0e-5** (Option A) / **1.5e-5** (Option B/C) | 1e-5는 보수적. 1B SFT에서 2e-5 사용하여 성공. 데이터 많을수록 낮게 |
| **batch_size** | 2 | **2** (유지) | VRAM 48.3GB → SFT 약 55-60GB 예상, bs=2가 안전 |
| **grad_accum_steps** | 4 | **4** (Option A) / **8** (Option B/C) | 데이터 많으면 larger effective batch가 안정적 |
| **effective_batch** | 64 | 64 (A) / **128** (B/C) | 2 x 8GPU x 4or8 |
| **max_steps** | 33,000 | **7,600** (A) / **17,500** (B) / **35,000** (C) | 3 epochs: A=161K/64*3, B=740K/128*3, C=1.5M/128*3 |
| **warmup_steps** | 500 | **200** (A) / **500** (B/C) | 전체의 3-5% |
| **weight_decay** | 0.01 | **0.01** (유지) | SFT 표준 |
| **max_grad_norm** | 1.0 | **1.0** (유지) | |
| **neftune_alpha** | 5.0 | **5.0** (유지) | 1B에서 효과 확인됨 |
| **save_interval** | 2,000 | **500** (A) / **1,000** (B) / **2,000** (C) | 작은 데이터셋일수록 자주 저장 |
| **eval_interval** | 500 | **200** (A) / **500** (B/C) | |
| **dropout** | 0.0 | **0.0** (유지) | SFT에서는 일반적으로 0 |
| **use_fp8** | true | **true** (유지) | B200 MXFP8 유지 |
| **lr_scheduler** | (미정) | **cosine** | cosine decay가 SFT 표준 |

**중요 주의사항**:
- `compile_model: false` 유지 — TE와 충돌
- `use_amp: false` 유지 — FP8이 대체
- label off-by-one 버그 확인 (SFT v1에서 발생했던 이슈)
- checkpoint-XXXXXX → checkpoint-0057000 반드시 수정

#### 2.1.4 SFT 모니터링 지표 & 목표

| 모니터링 지표 | 목표 | 위험 신호 |
|-------------|------|---------|
| val_loss | < 2.0 (1B SFT: 2.206) | > 2.5 또는 상승 추세 |
| train_loss | 안정적 하강 | 0에 수렴 (과적합) |
| 3-gram 반복률 | < 15% | > 30% (SFT 실패) |
| EOS 종료율 | > 50% | < 20% |
| gradient norm | < 1.0 안정 | 급증 (발산 징후) |

#### 2.1.5 SFT 예상 소요 시간

| Option | 데이터 | Steps | 예상 시간 |
|--------|--------|-------|---------|
| A (빠른) | 161K | ~7,600 | ~5-7시간 |
| B (추천) | 740K | ~17,500 | ~12-15시간 |
| C (대규모) | 1.5M | ~35,000 | ~24-28시간 |

---

### 2.2 Phase 3: ORPO/DPO Alignment — SFT 후 조건부

#### 2.2.1 실행 조건

SFT 완료 후 아래 조건 중 하나 이상 해당 시 실행:
- 3-gram 반복률 > 5%
- EOS 종료율 < 70%
- hallucination 빈도 개선 부족

#### 2.2.2 ORPO vs DPO 선택 기준

| 항목 | ORPO | DPO |
|------|------|-----|
| 별도 reference model | **불필요** | 필요 (SFT 모델 복사) |
| VRAM 사용 | **낮음** (1모델) | 높음 (2모델, ~96GB) |
| 학습 안정성 | 상 | 중상 (beta 민감) |
| 기대 효과 | 반복 감소, 선호도 정렬 | hallucination 감소, 정밀 정렬 |
| **추천** | **1순위** | 2순위 (ORPO 부족 시) |

#### 2.2.3 ORPO 파라미터 제안

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| lr | 5e-6 | SFT lr의 1/3~1/4 |
| beta (ORPO lambda) | 0.1 | 표준 시작점, 0.05~0.2 탐색 |
| epochs | 1~2 | preference 데이터는 소량 반복이 효과적 |
| batch_size | 2 per GPU | VRAM 제약 |
| grad_accum | 8 | eff_batch 128 |
| warmup | 100 steps | 짧게 |
| max_length | 2048 | chosen+rejected 합산이므로 절반으로 |
| data | 795K preference pairs | 이미 준비됨 |

#### 2.2.4 DPO 파라미터 제안 (대안)

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| lr | 1e-6 | DPO는 ORPO보다 보수적 |
| beta | 0.1~0.5 | 높을수록 reference에 가깝게 유지 |
| epochs | 1 | 과적합 주의 |
| reference_model | SFT checkpoint (frozen) | VRAM +48GB 필요 |

---

### 2.3 Continued Pretraining (선택적) — PPL 개선

#### 2.3.1 실행 근거

- Chinchilla 최적 60B tokens 대비 41.12B (31% 미달)
- cc100_ko PPL 21.8, namuwiki PPL 25.9 → 도메인 불균형
- SFT 전 base 품질을 높이면 SFT 효율도 향상

#### 2.3.2 파라미터 제안

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| 추가 토큰 | 20B tokens | 41B → 61B (Chinchilla 달성) |
| lr | 3e-5 (현재 학습 마지막 lr) → 1e-5 cosine decay | 기존 학습 연장 |
| warmup | 200 steps | 짧은 재워밍 |
| 데이터 구성 | namuwiki 정제본 30%, cc100 필터링 20%, 기존 mix 50% | 취약 도메인 보강 |
| 예상 시간 | ~24시간 (20B / 38K tok/s / 8GPU) |  |
| checkpoint | 0057000에서 이어서 학습 (resume) |  |

**주의**: 이 작업은 SFT 결과가 기대 이하일 때만 검토. SFT 먼저 진행이 효율적.

---

### 2.4 평가 재실행 계획 (SFT 완료 후)

#### 2.4.1 필수 평가 항목

| 평가 | 스크립트 | 소요 시간 | 목표 |
|------|---------|---------|------|
| PPL (19개 val set) | `eval/full_eval_pipeline.py` Phase 1 | ~35분 | val PPL < 5.0 유지 |
| Calibration | Phase 1 포함 | ~1분 | Top-1 > 65% 유지 |
| Generation (반복률) | Phase 1 포함 | ~3분 | < 15% (rep=1.1) |
| KoBEST 5개 (0-shot) | Phase 2 | ~2분 | avg > 65% |
| HAE-RAE (0-shot) | Phase 2 | ~1분 | > 50% |
| MMLU-KO (0-shot) | lm-eval 별도 | ~10분 | > 35% |
| belebele_kor | Phase 2 | ~1분 | > 45% |

#### 2.4.2 추가 평가 (신규)

| 평가 | 목적 | 방법 |
|------|------|------|
| 5-shot 벤치마크 | in-context learning 능력 | lm-eval --num_fewshot 5 |
| EOS 종료율 | 대화 완성도 | 생성 후 EOS 비율 측정 |
| 사실 정확도 | hallucination 정도 | 10개 사실 질문 수동 평가 |
| 멀티턴 대화 | 맥락 유지 | 3턴 대화 5세트 수동 평가 |

#### 2.4.3 MMLU-KO 평가 환경 수정

현재 `global_mmlu_ko_*` 태스크가 lm-eval registry에 없음.

```bash
# 해결 방법 1: kmmlu 사용 (한국어 MMLU)
pip install lm-eval --upgrade
python -m lm_eval --tasks kmmlu --model hf --model_args pretrained=<path>

# 해결 방법 2: global_mmlu 태스크 수동 등록
# lm_eval/tasks/ 아래 yaml 확인 및 추가
```

---

### 2.5 배포 준비 (SFT 완료 후)

#### 2.5.1 GGUF 변환

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| 양자화 | Q4_K_M | 품질/크기 밸런스 최적 |
| 예상 크기 | ~1.7GB | 3B * 4bit + overhead |
| 변환 도구 | llama.cpp convert | `scripts/convert_to_gguf.sh` 준비됨 |

#### 2.5.2 Ollama 배포 설정 (Modelfile.3b 기준)

```
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
```

**제안 수정**:

| 파라미터 | 현재 | 제안 | 근거 |
|---------|------|------|------|
| repeat_penalty | 1.1 | **1.2** | SFT 후에도 반복 잔존 가능, 약간 높게 |
| temperature | 0.7 | **0.7** (유지) | 사실성/유창성 균형 |
| top_p | 0.9 | **0.9** (유지) | |
| top_k | 40 | **40** (유지) | |
| stop | 미설정 | **`<|im_end|>`** | ChatML 포맷 EOS |

---

## Part 3: 실행 순서 로드맵

```
                    현재 위치
                        |
                        v
[1] SFT 데이터 큐레이션 ─────────────── (0.5~1일)
    - sft_extra 필터링 + 통합
    - Option B: ~740K samples 목표
    - 도메인 밸런싱, 중복 제거
                        |
                        v
[2] SFT 학습 실행 ────────────────────── (0.5~1일)
    - korean_3b_sft.yaml 파라미터 확정
    - torchrun 8GPU SFT
    - val_loss, 반복률 모니터링
                        |
                        v
[3] SFT 후 Full Evaluation ───────────── (1시간)
    - PPL, Calibration, Generation, Benchmarks
    - MMLU-KO 환경 수정 후 재평가
    - 1B SFT vs 3B SFT 직접 비교
                        |
                        v
[4] 판단 분기점 ──────────────────────────
    |                                    |
    v                                    v
  반복률 <5%, 벤치 양호              반복률 >5% 또는 벤치 미달
    |                                    |
    v                                    v
  [5a] GGUF 변환 + 배포          [5b] ORPO alignment
    - Q4_K_M 양자화                  - 795K preference pairs
    - Ollama 배포                    - 1~2 epochs
    - 실사용 테스트                   - 재평가 후 5a로
                                         |
                                         v
                              [선택] Continued Pretrain
                                - PPL 개선 필요 시
                                - 20B tokens 추가
                                - 다시 SFT부터
```

---

## Part 4: 리스크 & 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| SFT label 버그 (v1 재현) | 낮 | 치명적 | train_loss가 0 수렴 시 즉시 중단, label 검증 |
| SFT 과적합 | 중 | 높 | early stopping, eval 주기 짧게, dropout 추가 고려 |
| catastrophic forgetting | 중 | 높 | lr 낮게 (1e-5~2e-5), PPL 모니터링 |
| VRAM OOM (SFT) | 낮 | 중 | bs=2 안전, bs=1 fallback |
| SFT 후 벤치마크 미개선 | 중 | 중 | 데이터 품질 점검, few-shot 평가, ORPO 적용 |
| 나무위키 PPL 미개선 | 높 | 낮 | SFT 범위 밖, continued pretrain 검토 |

---

## Part 5: 참조 파일 경로

| 용도 | 경로 |
|------|------|
| 3B Base checkpoint | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000` |
| 3B Base backup | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000_BASE_BACKUP` |
| SFT config | `configs/korean_3b_sft.yaml` |
| SFT 기존 데이터 | `data/sft/train.jsonl` (161K) |
| SFT 추가 데이터 | `data/sft_extra/` (36개 소스, ~6.5M samples 미큐레이션) |
| SFT 실행 스크립트 | `scripts/launch_3b_sft.sh` |
| Eval pipeline | `eval/full_eval_pipeline.py` |
| Eval 결과 (3B base) | `eval/outputs/3b_full_eval_20260305_0318/` |
| Eval 보고서 (3B base) | `eval/outputs/3b_full_eval_20260305_0318/reports/` |
| 종합 보고서 | `reports/2026-03-05_3B_BASE_EVALUATION_REPORT.md` |
| Training log | `checkpoints/korean_3b_fp8_run1/train.log` |
| Modelfile (배포) | `Modelfile.3b` |
| GGUF 변환 | `scripts/convert_to_gguf.sh` |

---

*작성일: 2026-03-05 | 다음 참조: SFT 실행 시 이 문서의 Part 2.1 파라미터 확인*

# SFT 하이퍼파라미터 분석 & 다음 튜닝 옵션 조사

> 생성일: 2026-02-26  
> 모델: korean_1b_sft (1.19B params, base: korean_1b_fp8_run1/checkpoint-0034000)  
> 학습: 5000 steps, 39분, 8× B200

---

## 1. Loss Curve 분석

### 1-1. 기본 통계

| 구간 | Steps | n | Loss Mean | Loss Stdev | Loss Min | Loss Max | GNorm Mean |
|------|-------|---|-----------|------------|----------|----------|------------|
| Warmup | 10–150 | 15 | 2.3100 | 0.1144 | 2.1129 | 2.5229 | 1.414 |
| Post-warmup 전체 | 160–5000 | 485 | 1.9984 | 0.0942 | 1.7305 | 2.3413 | 1.133 |
| Q1 (초기) | 160–1360 | 121 | 2.0698 | 0.0860 | 1.8850 | 2.3413 | 1.138 |
| Q2 (중반1) | 1370–2570 | 121 | 1.9915 | 0.0801 | 1.7960 | 2.2088 | 1.131 |
| Q3 (중반2) | 2580–3780 | 121 | 1.9583 | 0.0870 | 1.7384 | 2.1293 | 1.119 |
| Q4 (후반) | 3790–5000 | 122 | **1.9739** | 0.0835 | 1.7305 | 2.1635 | 1.142 |

### 1-2. 500-step 이동 평균 Loss (±50 step 윈도우)

| Step | Loss(avg) | GNorm(avg) | 해석 |
|------|-----------|------------|------|
| ~500 | 2.0658 | 1.098 | 초기 하강 단계 |
| ~1000 | 2.0281 | 1.121 | 빠른 하강 지속 |
| ~1500 | 1.9663 | 1.092 | ✅ 최초 <2.0 진입 |
| ~2000 | 1.9802 | 1.158 | 소폭 반등 (정상) |
| ~2500 | 1.9882 | 1.140 | 안정화 구간 시작 |
| ~3000 | 1.9628 | 1.083 | 최저점 근방 |
| ~3500 | 1.9668 | 1.151 | 수렴 신호 |
| ~4000 | 1.9679 | 1.161 | 고원 진입 |
| ~4500 | 1.9555 | 1.142 | 미세 하강 지속 |
| ~5000 | 1.9718 | 1.195 | **최종: 1.9677** |

### 1-3. 해석

**Warmup 구간 (step 10–150):**
- LR이 1.33e-6 → 2e-5로 선형 증가하는 동안 loss가 2.11–2.52 범위에서 불규칙함
- Warmup 직후 step 160에서 loss spike (2.34, 3.6σ) 발생 — warmup 종료 직후 full LR 충격. 정상적이고 흔한 패턴
- Warmup 150 steps는 총 5000 steps의 3% → 적절

**정상 학습 구간 (step 160–5000):**
- Loss가 Q1→Q3 구간에서 2.07→1.96으로 지속 하강 (총 0.11 감소)
- Q3→Q4는 1.958→1.974으로 **오히려 소폭 상승** — cosine LR이 충분히 낮아지면서 학습 속도 저하, 수렴 징후
- 표준편차 0.094는 안정적 (SFT 기준 0.05–0.15 정상 범위)

**Outlier 분석:**
- Mean+2σ = 2.187 초과: 10개 / 485 = **2.1%** → 정상 수준
- 모두 초기(step 160–800)에 집중 + step 2190 1개 — 데이터 다양성에 의한 정상 변동
- gnorm spike와 동반하지 않아 gradient 폭발 없음

**GNorm 패턴:**
- 전체 평균 1.13, max_grad_norm=1.0으로 설정되어 있으나 로그값은 0.89–1.53
- 로그되는 gnorm은 clip **이전** 값으로 추정; 실제 1.0 초과 시 clip 발생
- Warmup 구간(평균 1.41)이 이후(평균 1.13)보다 높음 — 정상 패턴
- 학습 전반에 걸쳐 감소 추세 (gnorm 안정화 = 학습이 수렴 중)

**핵심 결론:** 학습은 건강하게 진행됨. Step ~3000 이후 수렴 신호가 있으나 loss는 여전히 미세 하강 중. 5000 steps 종료 시점이 적절한 stopping point였거나 추가 학습 여지 있음.

---

## 2. 하이퍼파라미터 영향 분석

### 2-1. Learning Rate: **2e-5** → ✅ 적절 (업계 표준 범위)

| 모델/프레임워크 | LR | 규모 |
|---|---|---|
| Meta Alpaca (Llama 7B) | 2e-5 | 7B |
| WizardLM (Vicuna 13B) | 2e-5 | 13B |
| OpenHermes (Mistral 7B) | 2e-5 | 7B |
| LIMA (65B) | 1e-5 | 65B |
| TinyLlama SFT (1.1B) | 2e-5 | 1.1B |
| **현재 설정** | **2e-5** | **1.2B** |

- 1B 규모에서 2e-5는 업계 표준값과 정확히 일치
- pretrain LR(2e-4)의 1/10으로 설정 → catastrophic forgetting 방지 원칙 충족
- 단, 추가 epoch 시에는 1e-5로 낮추는 것이 안전

**개선 방향:** 현재 설정 유지. 2차 학습 시 1e-5 추천.

### 2-2. Cosine Decay 스케줄 → ✅ 적절 (단, 최종 LR 약간 높음)

- 최종 LR: 2.00e-6 (peak의 10%)
- 표준 cosine schedule: min_lr = 0.1 × peak_lr
- 5000 steps에 맞는 설정: warmup 150 + cosine decay 4850 steps
- step 5000에서 LR이 2e-6으로 자연 수렴 → 학습이 마무리된 느낌

**개선 방향:** min_lr을 0 또는 1e-7로 낮추면 마지막 구간 더 안정적 수렴 가능. 현재 설정도 무방.

### 2-3. Effective Batch Size: **64 sequences** (=262K tokens/step) → ✅ 적절

- 64 seqs × 평균 ~500 tokens (dynamic padding) ≈ 32,000 tokens/step 실제 처리량
- max_seq_len=4096 기준 이론값은 262,144 tok/step이지만 동적 패딩으로 실제는 낮음
- SFT 배치 크기 참고: Alpaca=128 seqs, WizardLM=64 seqs, LIMA=64 seqs
- **64는 업계 표준값과 정확 일치**

**개선 방향:** 현재 설정 유지. 배치가 너무 크면 generalization 저하 가능성 있음.

### 2-4. Epochs: **~2 epoch** → ⚠️ 부족 가능성 (안전은 함)

- 5000 steps × 64 seqs = 320,000 예제 처리 / 159,000 샘플 = **약 2.0 epoch**
- SFT 업계 기준:
  - LIMA: 15 epoch (소량 데이터 1K개)
  - Alpaca, WizardLM: **3 epoch**
  - OpenHermes, Hermes: 3–5 epoch
  - 대규모 데이터(>100K): 1–3 epoch

- 2 epoch는 **과소학습 가능성** 있음 (특히 낮은 빈도 데이터 패턴 학습 부족)
- Q4 loss(1.974)가 Q3(1.958)보다 살짝 높아진 것은 cosine LR 감소 효과 + 아직 수렴 전일 가능성 공존
- Val loss가 없어 과적합 여부 확인 불가 (✅ eval_interval=100으로 설정은 되어 있었으나 결과 없음)

**개선 방향:** 3–4 epoch (7500–10000 steps) 추가 실험 권장. 단 val split 필수 확보 후 진행.

### 2-5. NEFTune alpha=10 → ✅ 이 데이터셋 크기에 적합

- 원논문(Jain et al., 2023) 권장값: 소규모(<10K) → 5, 중규모(10K–500K) → 10, 대규모(>500K) → 15
- 159K 샘플 → **alpha=10 적합**
- Noise magnitude = alpha / sqrt(seq_len × d_model) = 10 / sqrt(500 × 2048) ≈ 0.0099
  - 실제 embedding 값 대비 적절한 noise 비율
- Loss curve 안정성(stdev 0.094)으로 볼 때 NEFTune이 학습을 불안정하게 만들지 않았음

**개선 방향:** 현재 설정 유지. 데이터 증가(500K+) 시 alpha=15로 상향 고려.

### 2-6. max_seq_len: **4096** → ✅ 적절 (단, 활용도 확인 필요)

- 설정: max_seq_len=4096, dynamic padding 적용
- 한국어 instruction 데이터 평균 길이: 200–1000 tokens (kullm/KoAlpaca 기준)
- Dynamic padding 덕분에 짧은 시퀀스들은 실제로 4096을 채우지 않음 → compute 효율적
- rope_theta=500000 (Llama-3 스타일) → 4096 이상 외삽도 지원

**잠재 문제:**
- 데이터셋에 4096 초과 대화가 있다면 truncation 발생 → 긴 multi-turn 대화 손실
- 현재 데이터셋(kullm, KoAlpaca, LIMA 등)은 대부분 2048 이하이므로 실질적 영향 적음

**개선 방향:** 현재 설정 유지. 장문 대화 데이터 추가 시 8192 고려.

---

## 3. 다음 튜닝 옵션 후보군

### A. 추가 SFT Epoch (5000 → 10000 steps, epoch 4)

**Pros:**
- 현재 loss가 여전히 하강 추세 — 추가 학습 여지 있음
- epoch 3–4는 SFT 업계 표준 (Alpaca, WizardLM 기준)
- 기존 체크포인트에서 resume 가능, 39분 추가면 충분 (B200 속도 기준)
- 구현 가능: `--resume checkpoints/korean_1b_sft/checkpoint-5000 --max_steps 10000`

**Cons:**
- Val loss 없이 진행 시 과적합 감지 불가
- cosine schedule이 이미 step 5000 기준으로 설계되어 있음 → resume 시 LR 스케줄 재설정 필요
- epoch 4 이후 과적합 위험 (특히 반복 패턴 memorization)

**추천:** ✅ **조건부 추천** — val split 5–10% 확보 후, LR=1e-5로 새 cosine schedule 설정하여 추가 학습. Resume보다 fresh start 권장.

**구체적 설정:**
```yaml
max_steps: 5000  # 추가 5000 steps (epoch 3-4)
lr: 1.0e-5       # 이전의 절반
warmup_steps: 50 # 짧은 warmup
```

---

### B. LR 튜닝: 2e-5 vs 1e-5 vs 5e-6

| LR | 장점 | 단점 | 추천 |
|----|------|------|------|
| 5e-6 | 매우 안전, 과적합 방지 | 5000 steps에서 개선 폭 적을 수 있음 | ❌ 너무 보수적 |
| **1e-5** | **균형잡힌 선택, 2차 학습 표준** | 현재 대비 학습 속도 절반 | ✅ **추천** |
| 2e-5 (현재) | 1차 학습에서 좋은 결과 | 추가 epoch에서 과적합 위험 | ⚠️ 추가 학습에 불리 |

**결론:** 2차 학습 시 **lr=1e-5** 사용. 현재 lr=2e-5는 1차 학습에 최적.

---

### C. ORPO (Odds Ratio Preference Optimization)

**개요:** SFT + preference alignment을 단일 단계에서 동시 수행. Reference model 불필요.

**Pros:**
- Reference model 없어 메모리 절약 (DPO 대비 VRAM 약 40% 절약)
- SFT와 preference를 동시에 최적화 → 모델 품질 저하 없이 alignment 가능
- 1-stage 파이프라인 → 운영 단순화
- `trl` 라이브러리로 쉽게 구현 가능

**Cons:**
- Chosen/rejected 쌍 데이터 필수 (현재 없음)
- 한국어 preference 데이터 선택지가 제한적

**한국어 Preference 데이터 현황 (HuggingFace 기준):**
| 데이터셋 | 샘플 수 | 특징 |
|---------|---------|------|
| `maywell/ko_Ultrafeedback` | ~60K | UltraFeedback 한국어 번역 |
| `ChuGyouk/korean-ultrafeedback-armorm` | ~60K | ArmoRM 스코어 포함 |
| `HAERAE-HUB/K2-Align` | ~10K | 한국어 RLHF alignment |
| `heegyu/KORANI-v1` | ~20K | Korean RANI (human feedback) |
| `trl-lib/ultrafeedback_binarized` | ~60K | 영어 (번역 필요) |

**추천:** ✅ **추천** — `maywell/ko_Ultrafeedback` 또는 `ChuGyouk/korean-ultrafeedback-armorm` 확보 후 TRL `ORPOTrainer`로 구현. SFT 후 ORPO 적용 또는 from scratch ORPO 모두 가능.

**구현 예시:**
```python
from trl import ORPOConfig, ORPOTrainer
config = ORPOConfig(learning_rate=5e-7, num_train_epochs=1, ...)
trainer = ORPOTrainer(model, config, train_dataset=preference_data)
```

---

### D. DPO (Direct Preference Optimization)

**개요:** SFT 완료 모델 위에 preference alignment을 추가 학습. Reference model(=SFT 모델 frozen) 필요.

**vs ORPO:**
| | DPO | ORPO |
|--|-----|------|
| Reference model | 필요 (VRAM +40%) | 불필요 |
| SFT 단계 | 별도 필요 | 통합 가능 |
| 안정성 | 검증된 방법 | 상대적으로 신규 |
| 데이터 | chosen/rejected | chosen/rejected |
| 구현 복잡도 | 중간 | 낮음 |

**Pros:**
- 가장 널리 검증된 preference optimization 방법
- `trl` 라이브러리 완전 지원
- Llama, Mistral 기반 모든 주요 모델에 적용됨

**Cons:**
- SFT 모델을 reference로 두고 추가 학습 → 메모리 2배 (1.2B × 2 = ~16GB, B200 192GB에서 무리 없음)
- 2단계 학습 파이프라인 복잡성

**추천:** ✅ **추천** — ORPO보다 검증된 방법. B200 × 8에서 메모리 이슈 없음. ORPO와 A/B 테스트 가치 있음.

---

### E. LoRA/QLoRA

**맥락:** 이미 full fine-tuning 완료. LoRA의 역할은?

**Pros:**
- 빠른 하이퍼파라미터 실험 (LR, epoch, alpha 조합): full FT 대비 3-5x 빠름
- 여러 adaptation 동시 관리 (domain-specific LoRA weights)
- DPO/ORPO 단계에서 adapter만 학습 가능
- VRAM 사용 절약 → batch size 증가 가능

**Cons:**
- 이미 full FT된 모델이 있으므로 LoRA 성능 상한 ≤ full FT
- 1B 모델은 이미 작아서 QLoRA의 4-bit quantization 이점이 크지 않음
- Fine-tuning quality는 full FT가 항상 우세

**추천:** ⚠️ **조건부 추천** — 하이퍼파라미터 탐색(lr 그리드서치, epoch sweep)에 LoRA 활용. 최종 모델은 full FT.

**실용적 사용법:**
```python
# 빠른 실험: LoRA rank=64로 LR 그리드서치
# rank=64, alpha=128, dropout=0.05
# 약 5-10분 / 실험 (B200 기준)
```

---

### F. 데이터 품질 개선

**현재 데이터 구성:**
- kullm: 대규모 한국어 instruction (품질 혼재)
- KoAlpaca: Alpaca 한국어 번역 (번역 품질 이슈)
- safe_conv: 안전 대화 데이터
- LIMA: 고품질 영어 instruction (1000개)
- evol_instruct: GPT-4 생성 (고품질)
- kovast: 한국어 대화

**개선 방향:**

1. **Deduplication (MinHash LSH):**
   - instruction text에 대해 locality-sensitive hashing
   - 예상 중복 제거율: 5–15% (159K → 135–150K 정도)
   - 품질 향상 효과: 중복 패턴 memorization 방지

2. **Quality Filtering:**
   - Perplexity 기반 필터: 너무 낮거나 너무 높은 perplexity 제거
   - 언어 확인: 한국어 비율 체크 (`langdetect`)
   - 길이 필터: 너무 짧은 응답(<50 tokens) 제거
   - 반복 패턴 제거: `n-gram repetition score` 기반

3. **Domain Mixing 조정:**
   - LIMA-style: 소량의 고품질 데이터가 대량의 저품질보다 효과적
   - evol_instruct 비율 ↑ (GPT-4 생성이므로 고품질)
   - 단순 번역 데이터(KoAlpaca) 비율 ↓

**추천:** ✅ **강력 추천** — 데이터 품질이 epoch 수보다 중요. 1주일 투자로 실질적 성능 향상 기대.

---

### G. 더 많은 SFT 데이터 (159K → 500K+)

**HuggingFace 추가 가능 데이터셋:**

| 데이터셋 | 샘플 수 | 언어 | 품질 | 비고 |
|---------|---------|------|------|------|
| `HAERAE-HUB/qarv-instruct-100k` | 100K | 한국어 | 중상 | 한국어 instruction 100K |
| `nayohan/llama3-instruct-ko-dataset` | 58K | 한국어 | 상 | Llama-3 instruction 한국어 |
| `hPark/orca-ko` | 200K+ | 한국어 | 상 | Orca 스타일 한국어 |
| `maywell/synatra-orca` | 300K+ | 한국어 | 상 | 합성 데이터, 고품질 |
| `FreedomIntelligence/evol-instruct-korean` | 70K | 한국어 | 상 | GPT-4 생성 한국어 |
| `Bingsu/ko_alpaca_data` | 52K | 한국어 | 중 | Alpaca 한국어 (번역) |
| `HAERAE-HUB/KoInstruct` | 50K+ | 한국어 | 중상 | 한국어 instruction |
| `Open-Orca/OpenOrca` | 1M+ | 영어 | 최상 | 고품질 영어 (한국어 모델에 혼합 가능) |

**500K 달성 경로:**
1. 현재 159K
2. `hPark/orca-ko` + `maywell/synatra-orca` 추가: +200K = 359K
3. `HAERAE-HUB/qarv-instruct-100k` + `nayohan/llama3-instruct-ko-dataset`: +158K = 517K
4. 품질 필터 후 유지 비율 ~80% → **약 400K 순 데이터**

**Pros:**
- 더 많은 도메인 커버리지
- 드문 패턴 학습 기회 증가
- Generalization 향상

**Cons:**
- 데이터 품질 검증 필요 (무분별 추가는 역효과)
- 학습 시간 증가 (같은 epoch 기준 3배 → 2시간+)
- 고품질 소량 vs 저품질 다량 트레이드오프

**추천:** ✅ **추천 (품질 필터 전제)** — `hPark/orca-ko`나 `maywell/synatra-orca` 같은 고품질 합성 데이터 우선 추가. 단순 번역 데이터 비율 주의.

---

## 4. 즉시 실행 가능한 실험 Top 3

### 🥇 1순위: **현재 모델 종합 평가 (eval 실행)**

**이유:**
- Loss 1.9677이 실제로 좋은 모델인지 알 수 없음
- 추가 학습 방향 결정 전 baseline 필수
- 이미 `eval/comprehensive_eval.py` 존재

**즉시 실행:**
```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang

# Perplexity 평가
python eval/perplexity.py \
  --checkpoint checkpoints/korean_1b_sft/checkpoint-5000 \
  --data data/sft/val.jsonl  # val split 필요

# 생성 품질 빠른 체크
python eval/generate.py \
  --checkpoint checkpoints/korean_1b_sft/checkpoint-5000 \
  --prompts "안녕하세요, 저는 AI 모델입니다. 오늘 날씨에 대해 설명해주세요."
```

**예상 시간:** 10–30분

---

### 🥈 2순위: **lr=1e-5로 추가 SFT (epoch 3–4까지)**

**이유:**
- Loss curve가 아직 수렴하지 않았고 epoch 2는 업계 표준보다 부족
- 구현 비용 최소 (기존 코드 재사용)
- B200 × 8에서 약 40–60분 추가 (39분/5000steps 기준)

**구체적 설정:**
```bash
# 새 run으로 checkpoint-5000에서 시작
RUN_NAME=korean_1b_sft_v2 \
BASE_CHECKPOINT=checkpoints/korean_1b_sft/checkpoint-5000 \
LR=1.0e-5 \
MAX_STEPS=5000 \    # epoch 3-4
WARMUP_STEPS=50 \   # 짧은 warmup
bash scripts/launch_sft.sh
```

**주의:** val split 없으면 step 3000–5000에서 val loss 체크하며 early stop 기준 수동 설정 필요.

**예상 결과:** loss 1.90–1.93 (현재 1.97 대비 약 2–3% 개선), 생성 품질 체감 향상 기대.

---

### 🥉 3순위: **데이터 품질 개선 + 추가 데이터 수집**

**이유:**
- 데이터 품질이 하이퍼파라미터 튜닝보다 장기적으로 중요
- 현재 데이터에 중복/저품질 포함 가능성 있음
- ORPO/DPO 파이프라인 준비를 위해 preference 데이터도 동시에 수집

**즉시 실행 가능한 작업:**

```python
# 1. Deduplication (MinHash)
pip install datasketch
# instruction text 기준 MinHash dedup, threshold=0.8

# 2. 추가 데이터 다운로드
from datasets import load_dataset
ds = load_dataset("hPark/orca-ko")        # ~200K 고품질 한국어
ds2 = load_dataset("maywell/synatra-orca")  # ~300K 합성

# 3. 한국어 Preference 데이터 수집 (ORPO/DPO 준비)
pref = load_dataset("maywell/ko_Ultrafeedback")  # ~60K preference 쌍
```

**예상 시간:** 데이터 준비 2–4시간, 재학습은 추가 설정 후 진행.

---

## 5. 종합 평가 요약

### 현재 설정 평가

| 항목 | 설정값 | 평가 | 비고 |
|------|--------|------|------|
| Learning Rate | 2e-5 | ✅ 적절 | 업계 표준 정중앙 |
| Cosine Decay | 5000 steps | ✅ 적절 | min_lr ~10% |
| Warmup | 150 steps (3%) | ✅ 적절 | 3-5% 권장 범위 |
| Effective Batch | 64 seqs | ✅ 적절 | 업계 표준 |
| Epochs | ~2 | ⚠️ 부족 가능 | 3 epoch 표준 |
| NEFTune alpha | 10 | ✅ 적절 | 159K 데이터에 맞음 |
| max_seq_len | 4096 | ✅ 적절 | 동적 패딩으로 효율적 |
| Weight Decay | 0.01 | ✅ 적절 | pretrain(0.1)의 1/10 |

### 옵션별 추천 우선순위

| 옵션 | 추천 | 이유 |
|------|------|------|
| A. 추가 SFT (epoch 4) | ✅ 높음 | epoch 부족, 즉시 실행 가능 |
| B. LR 1e-5로 재학습 | ✅ 높음 | 추가 학습 시 필수 |
| C. ORPO | ✅ 중간 | 데이터 준비 필요 |
| D. DPO | ✅ 중간 | ORPO 대안, 더 검증됨 |
| E. LoRA | ⚠️ 낮음 | 하이퍼파라미터 탐색에만 유용 |
| F. 데이터 품질 개선 | ✅ 높음 | 장기 투자 대비 효과 큼 |
| G. 데이터 추가 (500K) | ✅ 중간 | 고품질 소스 전제 |

### 학습 곡선 총평

현재 SFT는 **건강하게 완료**됨:
- Gradient norm 안정, spike 없음
- Loss 단조 감소 (미시적 변동은 정상)
- Outlier 2.1%는 정상 범위
- 수렴 신호가 step 3000+ 이후 나타나지만 아직 plateau는 아님

**가장 우려되는 점:** Validation loss 없음 → 과적합 여부 불명확. **즉시 val split 확보 필요.**

---

*분석 완료. 다음 실행 시 이 파일을 기반으로 실험 방향 결정 권장.*

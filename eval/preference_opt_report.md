# Preference Optimization 조사 보고서

**작성일:** 2026-02-26
**목적:** SFT 이후 반복 퇴화(repetition degeneration) 해결을 위한 Preference Optimization 방법론 조사

---

## 1. 현재 환경

| 패키지 | 버전 | 비고 |
|---------|------|------|
| transformers | 5.2.0 | ✅ 설치됨 |
| accelerate | - | 확인 필요 |
| peft | - | 확인 필요 |
| **trl** | **미설치** | ⚠️ `pip install trl` 필요 |

**인프라:** 8× B200 183GB
**모델:** 커스텀 1B 파라미터 (Llama 계열 아키텍처, FP8 지원)
**최신 체크포인트:**
- Pretrain: `checkpoints/korean_1b_fp8_run1/checkpoint-0034000`
- SFT: `checkpoints/korean_1b_sft/` (최종 체크포인트는 log 확인 필요)

**HF 변환:** `scripts/convert_to_hf.py` 존재 ✅ — LlamaForCausalLM 포맷으로 변환 가능

---

## 2. ORPO vs DPO vs SimPO 비교

### ORPO (Odds Ratio Preference Optimization)
- **논문:** Hong et al. 2024 (arXiv:2403.07691)
- **Reference model:** 불필요 ✅
- **핵심 아이디어:** SFT loss + odds ratio 기반 preference loss를 단일 모델로 동시 학습
- **메모리:** SFT와 동일 (1× 모델만 필요)
- **1B 모델 적용:** 8× B200에서 매우 여유 (단일 GPU로도 가능)
- **구현:** TRL `ORPOTrainer` (trl >= 0.8.0)
- **장점:** 가장 간단, 메모리 효율적, SFT+preference 한 번에
- **단점:** DPO 대비 안정성 검증 사례 적음

### DPO (Direct Preference Optimization)
- **논문:** Rafailov et al. 2023 (arXiv:2305.18290)
- **Reference model:** 필요 (frozen copy, 2× 메모리)
- **메모리:** 1B 모델 × 2 ≈ 4GB (BF16) — 여전히 여유
- **1B 모델 적용:** 문제없음
- **구현:** TRL `DPOTrainer`
- **장점:** 가장 잘 검증됨, 안정적, 논문/사례 풍부
- **단점:** reference model 관리 필요

### SimPO (Simple Preference Optimization)
- **논문:** Meng et al. 2024 (arXiv:2405.14734)
- **Reference model:** 불필요
- **핵심:** Length-normalized implicit reward, margin 기반
- **구현:** TRL에 별도 Trainer 없음 → DPOTrainer의 `loss_type="simpo"` 로 사용 가능 (trl >= 0.9.0)
- **장점:** ORPO보다 성능 우수하다는 보고, reference-free
- **단점:** 상대적으로 새로운 방법

### PPO (Proximal Policy Optimization) — 참고용
- Reward model 별도 학습 필요 → 복잡도 높음
- 1B 모델에는 과도한 오버헤드
- **추천하지 않음** (데이터/인프라 대비 비효율)

---

## 3. 추천: **ORPO → DPO 순서**

### 1순위: ORPO
- Reference model 없음 → 메모리/구현 최소
- SFT 체크포인트에서 바로 시작 가능
- 반복 퇴화용 preference 데이터 제작이 간단

### 2순위: DPO
- ORPO로 부족하면 DPO로 전환
- 1B 모델이라 reference model 부담 없음
- 더 안정적이고 검증된 방법

### 근거
1B 모델 + 8× B200 환경에서는 DPO의 2× 메모리도 문제없지만,
**구현 속도와 단순성** 면에서 ORPO가 먼저 시도할 가치가 있음.

---

## 4. 한국어 Preference 데이터셋

### ✅ 접근 가능 (DPO/ORPO 형식 호환)

| 데이터셋 | 형식 | Downloads | 적합도 |
|----------|------|-----------|--------|
| **kuotient/orca-math-korean-dpo-pairs** | `{system, question, chosen, rejected}` | 111 | ⭐⭐⭐ DPO/ORPO 즉시 사용 가능 |
| **ChuGyouk/argilla-distilabel-math-preference-dpo-korean** | DPO 형식 | 10 | ⭐⭐⭐ 수학 도메인 |
| **nayohan/preference-collection-ko-full** | `{response_A, response_B, orig_score_A, orig_score_B, orig_preference}` | 30 | ⭐⭐⭐ 변환 필요하지만 풍부 |

### ✅ 접근 가능 (SFT 형식, preference 변환 필요)

| 데이터셋 | 형식 | Downloads |
|----------|------|-----------|
| jojo0217/korean_rlhf_dataset | `{instruction, input, output}` | 54 |
| FreedomIntelligence/alpaca-gpt4-korean | SFT 형식 | 158 |
| nlpai-lab/kullm-v2 | SFT 형식 | 730 |

### ❌ 접근 불가
maywell/ko_Ultrafeedback, HAERAE-HUB/KoRA, heegyu/OpenOrca-ko, Bongseok/ko-DPO-v0.1 — 모두 404

### 💡 자체 Preference 데이터 생성 전략 (반복 퇴화 특화)

가장 효과적인 방법: **현재 모델의 반복 출력을 rejected로 활용**

```
{
  "prompt": "서울의 유명한 관광지를 추천해주세요.",
  "chosen": "서울의 대표적인 관광지로는 경복궁, 북촌한옥마을, 남산타워...",
  "rejected": "서울의 관광지로는 경복궁이 있습니다. 경복궁이 있습니다. 경복궁이 있습니다..."
}
```

1. 현재 SFT 모델로 다양한 프롬프트에 대해 생성 (temperature 다양하게)
2. 반복이 발생한 응답 → rejected
3. 정상 응답 (또는 GPT-4로 생성) → chosen
4. 500~2000개만으로도 효과적

---

## 5. HF 변환

`scripts/convert_to_hf.py` 가 이미 존재하며 LlamaForCausalLM 포맷으로 변환:
- FP8 / BF16 체크포인트 모두 지원
- 출력: `config.json`, `model.safetensors`, `tokenizer.json` 등

**변환 명령:**
```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/korean_1b_sft/checkpoint-XXXXX \
    --output outputs/hf_for_orpo \
    --tokenizer tokenizer/korean_sp/tokenizer.json
```

변환 후 `AutoModelForCausalLM.from_pretrained("outputs/hf_for_orpo")` 로 로드 → TRL ORPOTrainer 사용 가능.

---

## 6. 반복 퇴화 해결에 ORPO가 효과적인 이유

### 메커니즘
ORPO의 odds ratio loss는 다음을 학습:
- **chosen 응답의 생성 확률 ↑** (정상적이고 다양한 응답)
- **rejected 응답의 생성 확률 ↓** (반복적인 응답)

반복 퇴화는 특정 토큰 시퀀스의 확률이 자기강화(self-reinforcing)되면서 발생.
ORPO는 이 패턴 자체를 직접적으로 페널티:

1. **반복 패턴 = rejected** → 모델이 반복 시퀀스에 높은 확률을 부여하는 것을 직접 억제
2. **다양한 정상 응답 = chosen** → 다양한 토큰 분포를 유도
3. **SFT loss와 동시 학습** → 일반 성능 유지하면서 반복 억제

### 왜 SFT만으로 부족한가
- SFT는 "좋은 응답을 따라하라"만 학습
- "나쁜 응답을 피하라"는 신호가 없음
- Preference optimization은 "이것은 하지 마라"를 명시적으로 학습

### 예상 효과
- 500~2000개의 반복-vs-정상 preference 쌍으로도 반복 퇴화 대폭 감소 가능
- repetition penalty 같은 디코딩 트릭보다 근본적 해결
- 일반 성능 저하 최소 (SFT loss가 함께 작용)

---

## 7. 실행 계획

```
1. TRL 설치: pip install trl --break-system-packages (또는 venv)
2. HF 변환: python scripts/convert_to_hf.py --checkpoint ... --output outputs/hf_for_orpo
3. Preference 데이터 준비:
   a. kuotient/orca-math-korean-dpo-pairs 다운로드 (즉시 사용 가능)
   b. 자체 반복 퇴화 데이터 생성 (eval/generate.py 활용)
4. ORPO 학습: python train/orpo.py (아래 스크립트)
5. 평가: 반복률 측정 + perplexity
```

ORPO 학습 스크립트: `train/orpo.py` 참조

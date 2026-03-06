# Korean LLM Benchmark Pipeline
> 작성: 2026-02-26 | 서버: 8× NVIDIA B200 183GB | PyTorch 2.10 (NV custom), CUDA 13.1

---

## 1. lm-eval 설치 상태

```
lm-eval 0.4.11 설치됨 (/usr/local/lib/python3.12/dist-packages/)
설치 명령: pip install lm-eval --break-system-packages
```

> ⚠️ `lm-eval[ko]` extra는 0.4.11에 없음. 기본 `lm-eval`로 설치하면 됨.
> Korean 관련 태스크는 기본 패키지에 모두 포함돼 있음.

---

## 2. Open Ko-LLM Leaderboard 9개 태스크 분석

### ❌ 결론: 로컬 실행 불가 (비공개 데이터셋)

Open Ko-LLM Leaderboard 2의 9개 태스크는 **전용 비공개 데이터셋** 사용:
- Ko-GPQA, Ko-WinoGrande, Ko-GSM8K, Ko-EQ-Bench → Flitto 제공 (비공개)
- KorNAT-CKA, KorNAT-SVA, Ko-Harmlessness, Ko-Helpfulness → SELECTSTAR + KAIST AI (비공개)
- Ko-IFEval → 비공개 번역본

leaderboard는 lm-evaluation-harness를 사용하지만, **데이터셋에 직접 접근 불가**.

### 각 태스크 상세 (메트릭 기준, 결과 데이터 분석)

| 태스크 | 레이블 | 메트릭 | Few-shot | 특징 |
|--------|--------|--------|----------|------|
| `ko_eqbench` | Ko-EQ Bench | `eqbench,none` | 0-shot | 감정지능 평가, 파싱 필요 |
| `ko_gpqa_diamond_zeroshot` | Ko-GPQA Diamond | `acc_norm,none` | 0-shot | 대학원 수준 과학 |
| `ko_gsm8k` | Ko-GSM8K | `exact_match,strict-match` | 0-shot | 초등 수학 추론 |
| `ko_ifeval` | Ko-IFEval | `prompt_level_strict_acc,none` + `inst_level_strict_acc,none` (평균) | 0-shot | 지시 따르기 |
| `ko_winogrande` | Ko-Winogrande | `acc,none` | 0-shot | 상식 추론 |
| `kornat_common` | KorNAT-CKA | `acc_norm,none` | 0-shot | 한국 문화·지식 |
| `kornat_harmless` | Ko-Harmlessness | `acc_norm,none` | 0-shot | 무해성 |
| `kornat_helpful` | Ko-Helpfulness | `acc_norm,none` | 0-shot | 유용성 |
| `kornat_social` | KorNAT-SVA | `A-SVA,none` | 0-shot | 사회적 가치 |

### 대안: 공개 유사 태스크로 간접 측정

| 원래 태스크 | 공개 대안 (lm-eval) |
|------------|-------------------|
| Ko-GSM8K | `global_mmlu_ko` + 수학 서브셋 |
| Ko-WinoGrande | `paws_ko` (유사 상식) |
| KorNAT-CKA | `haerae_general_knowledge`, `haerae_history` |
| Ko-IFEval | 별도 IFEval 스크립트 필요 |

---

## 3. 실제 사용 가능한 한국어 벤치마크

### 3-1. KoBEST ✅ (lm-eval 내장)
- **HF 데이터셋**: `skt/kobest_v1`
- **lm-eval 태스크 그룹**: `kobest`
- **5개 서브태스크**:
  - `kobest_boolq`: True/False 이진 분류 (~950 test)
  - `kobest_copa`: 원인·결과 추론 (~500 test)
  - `kobest_hellaswag`: 문장 완성 상식 (~500 test)
  - `kobest_sentineg`: 감성 분석 부정문 (~500 test)
  - `kobest_wic`: 단어 의미 파악 (~638 test)
- **실행 명령**:
  ```bash
  lm_eval --model hf --model_args pretrained=<HF_MODEL_PATH> \
    --tasks kobest --num_fewshot 0 --batch_size auto
  ```
- **예상 소요**: 1B 모델 기준 GPU 1장 ~15-30분

### 3-2. HAE-RAE Bench ✅ (lm-eval 내장)
- **HF 데이터셋**: `HAERAE-HUB/HAE_RAE_BENCH_1.0`
- **lm-eval 태스크 그룹**: `haerae`
- **6개 서브태스크**: (reading_comprehension 제외 5개 lm-eval에서 지원)
  - `haerae_general_knowledge`: 한국 상식 (~430 test)
  - `haerae_history`: 역사 (~100 test)
  - `haerae_loan_word`: 외래어 (~200 test)
  - `haerae_rare_word`: 희귀어 (~200 test)
  - `haerae_standard_nomenclature`: 표준어 표기 (~200 test)
- **실행 명령**:
  ```bash
  lm_eval --model hf --model_args pretrained=<HF_MODEL_PATH> \
    --tasks haerae --num_fewshot 0 --batch_size auto
  ```
- **예상 소요**: ~5-10분

### 3-3. Global MMLU (Korean) ✅ (lm-eval 내장)
- **HF 데이터셋**: `CohereForAI/Global-MMLU`
- **lm-eval 태스크 그룹**: `global_mmlu_ko`
- **57개 도메인** 한국어 번역본
- **실행 명령**:
  ```bash
  lm_eval --model hf --model_args pretrained=<HF_MODEL_PATH> \
    --tasks global_mmlu_ko --num_fewshot 0 --batch_size auto
  ```
- **예상 소요**: 1B 모델 기준 ~60-90분

### 3-4. K2-Eval ⚠️ (별도 평가 필요)
- **HF 데이터셋**: `HAERAE-HUB/K2-Eval` ✅ (공개 접근 가능)
- **형태**: 개방형 지시 따르기 (Open-ended instructions)
- **카테고리**: Korean History, Geography, Social Issues, Numerical Estimation, Creative Writing 등
- **lm-eval 지원**: ❌ — LLM-as-a-Judge 방식 필요 (GPT-4 또는 Claude)
- **대안**: vLLM으로 생성 후 별도 judge 스크립트

### 3-5. LogiKor ❌ (HuggingFace에서 미확인)
- 공개된 LogiKor 데이터셋을 HF에서 찾지 못함
- 논문/GitHub 경로 직접 확인 필요
- 추후 발견 시 추가 예정

### 3-6. PAWS-Ko ✅ (lm-eval 내장)
- **태스크**: `paws_ko` — 패러프레이즈 탐지
- 빠르게 언어 이해 측정 가능

---

## 4. 빠른 체크 vs 전체 평가 태스크셋

### ⚡ 빠른 체크 (목표: 30분 이내)
```
kobest_boolq, kobest_copa, haerae_general_knowledge, haerae_history, paws_ko
```
- 총 샘플 수: ~2,000개 이하
- 1B 모델 + 8×B200 → **약 10-20분** 예상
- 다양성: 분류, 추론, 상식, 패러프레이즈

### 📊 전체 평가 (목표: 2-4시간)
```
kobest (5) + haerae (5) + global_mmlu_ko (전체) + paws_ko
```
- 총 샘플 수: ~15,000개
- 1B 모델 + 8×B200 → **약 1.5-3시간** 예상
- tensor_parallel 미지원 시 단일 GPU 사용 → 더 길어질 수 있음

---

## 5. 모델 서빙 방법 결론

### 현황
- 체크포인트: `checkpoints/korean_1b_sft/checkpoint-0005000/`
- 내용: `model.pt`, `config.yaml`, `optimizer.pt`, `scheduler.pt`, `train_state.pt`
- 모델 아키텍처: 커스텀 LLaMA-like (FP8, d_model=2048, n_layers=24, n_heads=16)
- **lm-eval 기본 포맷**: HuggingFace `AutoModelForCausalLM`

### ✅ 추천 방법: HF 변환 후 평가

`scripts/convert_to_hf.py`가 이미 구현되어 있음. LlamaForCausalLM으로 변환.

```bash
# Step 1: HF 포맷으로 변환
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
python scripts/convert_to_hf.py \
    --checkpoint checkpoints/korean_1b_sft/checkpoint-0005000 \
    --output outputs/hf_korean_1b_sft_5000 \
    --tokenizer tokenizer/korean_sp/tokenizer.json

# Step 2: lm-eval 실행
lm_eval --model hf \
    --model_args pretrained=outputs/hf_korean_1b_sft_5000 \
    --tasks kobest \
    --device cuda:0
```

**주의사항**:
- FP8 가중치를 float32로 변환하는 과정 포함 (convert_to_hf.py 내부 처리)
- 커스텀 어휘(vocab_size=64000) → `sentencepiece_unigram` 방식
- lm-eval이 tokenizer를 인식하려면 `tokenizer_config.json`에 `"model_type": "llama"` 필요 (스크립트에 이미 포함)

### 대안 방법 B: API 서빙 + local-completions

```bash
# vLLM으로 변환된 모델 서빙
python -m vllm.entrypoints.openai.api_server \
    --model outputs/hf_korean_1b_sft_5000 --port 8000

# lm-eval API 평가
lm_eval --model local-completions \
    --model_args model=outputs/hf_korean_1b_sft_5000,base_url=http://localhost:8000/v1,num_concurrent=8 \
    --tasks kobest
```

### ❌ 방법 C: 커스텀 래퍼 (권장 안 함)
lm-eval ModelWrapper 작성 필요 → 복잡도 높음, 유지보수 어려움.

---

## 6. 설치 가이드

```bash
# 현재 환경 (Python 3.12, externally managed)
pip install lm-eval --break-system-packages

# 또는 가상환경 사용 (권장)
python3 -m venv /PROJECT/0325120031_A/ghong/taketimes/llm-bang/venv
source /PROJECT/0325120031_A/ghong/taketimes/llm-bang/venv/bin/activate
pip install lm-eval

# 추가 의존성
pip install safetensors transformers torch accelerate
```

---

## 7. 스크립트 위치

| 스크립트 | 용도 |
|---------|------|
| `scripts/run_eval_quick.sh` | 빠른 체크 (10-20분) |
| `scripts/run_eval_full.sh` | 전체 평가 (1.5-3시간) |
| `scripts/convert_to_hf.py` | 커스텀 체크포인트 → HF 변환 |

---

## 8. 참고 자료

- Open Ko-LLM Leaderboard: https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- KoBEST: https://huggingface.co/datasets/skt/kobest_v1
- HAE-RAE Bench: https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0
- K2-Eval: https://huggingface.co/datasets/HAERAE-HUB/K2-Eval
- KorNAT 논문: Lee et al. (2024) — KorNAT: LLM Alignment Benchmark for Korean Social Values

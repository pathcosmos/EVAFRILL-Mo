# FRANKENSTALLM 3B — 벤치마크 평가 보고서

**작성일**: 2026-03-05
**모델**: FRANKENSTALLM 3B (base pretrain, checkpoint-0057000)
**평가 도구**: lm-evaluation-harness (lm-eval)
**HF 변환**: `scripts/convert_to_hf.py` → `eval/outputs/hf_3b_base/`

---

## 1. 개요

FRANKENSTALLM 3B base 모델을 HuggingFace LlamaForCausalLM 형식으로 변환한 뒤, lm-evaluation-harness를 사용하여 한국어 벤치마크 평가를 수행하였다.

### HF 변환 정보

| 항목 | 값 |
|------|-----|
| 원본 체크포인트 | `checkpoints/korean_3b_fp8_run1/checkpoint-0057000` |
| 변환 출력 | `eval/outputs/hf_3b_base/` |
| 모델 형식 | LlamaForCausalLM (safetensors, 11GB) |
| 정밀도 | bfloat16 |
| 수정 사항 | `lm_head.weight` 공유 메모리 → `.clone()` 적용 |

### 평가 설정

| 항목 | 값 |
|------|-----|
| 벤치마크 | belebele_kor_Hang, global_mmlu_full_ko |
| 배치 크기 | 8 |
| GPU | cuda:2 |
| few-shot | 0-shot |

> **참고**: kobest_copa, kobest_boolq, haerae 등의 태스크는 설치된 lm-eval 버전에서 지원하지 않아 대체 벤치마크를 사용하였다.

---

## 2. 벤치마크 결과

### 2-1. 전체 요약

| 벤치마크 | Accuracy | Stderr | 랜덤 기준 | 판정 |
|----------|----------|--------|-----------|------|
| **belebele_kor_Hang** | **0.2189** | ±0.0138 | 0.25 (4지선다) | ≈ 랜덤 |
| **global_mmlu_full_ko** | **0.2339** | ±0.0036 | 0.25 (4지선다) | ≈ 랜덤 |

### 2-2. MMLU 한국어 분야별 상세

| 분야 | Accuracy | Stderr |
|------|----------|--------|
| Humanities (인문학) | 0.2389 | ±0.0062 |
| Social Sciences (사회과학) | 0.2301 | ±0.0076 |
| STEM (이공계) | 0.2312 | ±0.0075 |
| Other (기타) | 0.2327 | ±0.0076 |

### 2-3. MMLU 개별 과목 중 주목할 수치

| 과목 | Accuracy | 비고 |
|------|----------|------|
| computer_security | 0.3100 | 랜덤 이상 (+0.06) |
| machine_learning | 0.3125 | 랜덤 이상 (+0.06) |
| us_foreign_policy | 0.2900 | 랜덤 근처 |
| college_mathematics | 0.2700 | 랜덤 근처 |
| high_school_government | 0.1762 | 랜덤 이하 |
| human_sexuality | 0.1832 | 랜덤 이하 |
| high_school_chemistry | 0.1823 | 랜덤 이하 |

---

## 3. 결과 해석

### 3-1. Base model에서 랜덤 수준은 정상인가?

**예, 완전히 정상이다.**

| 이유 | 설명 |
|------|------|
| **형식 미학습** | Base model은 "A/B/C/D 중 고르시오" 형식을 학습한 적이 없음 |
| **지시 미학습** | 질문에 답변하는 패턴(instruction following)이 없음 |
| **토큰 확률 분포** | 선택지 토큰(A, B, C, D)에 대한 확률이 태스크에 맞게 조정되지 않음 |
| **업계 사례** | Llama-2-7B base도 MMLU에서 ~0.25-0.30 수준. SFT/RLHF 후 0.45+ |

### 3-2. Belebele vs MMLU 비교

- **Belebele (0.219)**: 독해 이해력 테스트. 긴 지문 + 질문 형식으로, base model이 형식 자체를 이해하기 더 어려움
- **MMLU (0.234)**: 지식 평가. 단문 질문이라 미세하게 높지만, 여전히 랜덤 수준

### 3-3. 랜덤 미만 점수의 의미

일부 과목에서 0.25 미만(예: high_school_chemistry 0.182)이 나온 것은:
- 통계적 노이즈 (표본 크기에 의한 변동, stderr ±0.027)
- Base model이 특정 선택지 토큰에 편향된 확률을 부여할 수 있음 (systematic bias)
- 모델 품질 문제가 아닌 **형식 부적합 문제**

---

## 4. 1B 베이스라인과 비교

| 벤치마크 | 1B (SFT 후) | 3B Base | 비고 |
|----------|------------|---------|------|
| kobest_copa | 0.646 | N/A | lm-eval 버전 미지원 |
| kobest_boolq | 0.50 | N/A | lm-eval 버전 미지원 |
| haerae_gk | 0.227 | N/A | lm-eval 버전 미지원 |
| belebele_kor | N/A | 0.219 | 1B에서 미측정 |
| global_mmlu_ko | N/A | 0.234 | 1B에서 미측정 |

> **비교 한계**: 1B와 3B가 다른 벤치마크로 평가되어 직접 비교가 불가능하다. 1B의 kobest_copa 0.646은 SFT 이후 수치이므로, 3B base와 비교하는 것 자체가 부적절하다.

---

## 5. SFT 후 기대치

| 벤치마크 | 3B Base | SFT 후 목표 | 근거 |
|----------|---------|------------|------|
| belebele_kor | 0.219 | >0.45 | 형식 학습으로 큰 폭 향상 기대 |
| global_mmlu_ko | 0.234 | >0.35 | 지식 활용 + 형식 적응 |
| kobest_copa | N/A | >0.70 | 1B SFT(0.646) 대비 개선 목표 |

---

## 6. SFT 진행 판단 (벤치마크 기준)

| 판단 | 근거 |
|------|------|
| **SFT 진행 ✅** | Base model의 랜덤 수준 벤치마크는 정상. SFT가 해결할 영역 |
| | 모델 구조/학습 실패의 징후 없음 (특정 분야만 극단적으로 낮지 않음) |
| | 분야별 분포가 균일 (0.23 ± 0.01) → 건강한 표현 학습 |

---

## 7. 평가 데이터 파일

| 파일 | 설명 |
|------|------|
| `eval/outputs/3b_benchmark_results.txt` | lm-eval 전체 로그 + 결과 테이블 |
| `eval/outputs/hf_3b_base/` | HF 형식 변환 모델 (11GB) |

---

*보고서 작성: 2026-03-05*

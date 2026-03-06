# SFT 데이터 품질 감사 보고서

**날짜:** 2026-02-26  
**데이터:** `data/sft/train.jsonl` (159,125 샘플)  
**소스:** 6개 HuggingFace 데이터셋 (KOR-OpenOrca-Platypus-v3, kullm-v2, ko-alpaca-12k, korean_safe_conversation, evol-instruct-korean, kovast)

---

## 1. 데이터 기본 통계

| 항목 | 값 |
|------|-----|
| 총 샘플 수 | 159,125 |
| Output 평균 길이 | 608 chars |
| Output 중앙값 | 468 chars |
| Output 최소/최대 | 10 / 7,393 chars |
| 중복 (instruction+output) | 0 (dedup 적용됨) |
| 중복 (instruction only) | 0 |

### Output 길이 분포

| 구간 | 수량 | 비율 |
|------|------|------|
| < 50 chars | 16,519 | 10.4% |
| 50-100 | 11,112 | 7.0% |
| 100-500 | 55,550 | 34.9% |
| 500-1000 | 47,023 | 29.6% |
| 1000-2000 | 23,731 | 14.9% |
| 2000-4000 | 5,049 | 3.2% |
| > 4000 | 141 | 0.1% |

---

## 2. 발견된 품질 문제

### 🔴 심각 (반복 루프 직접 원인 가능성)

#### 문제 1: 특수 토큰 오염 — `</s>` 113건
- Output 텍스트 안에 `</s>` 문자열이 리터럴로 포함된 샘플 113건
- **영향:** 학습 시 chat template이 `{output}</s>`를 붙이므로, output 내부의 `</s>`는 premature EOS를 학습시킴. 이후 모델이 EOS를 제대로 생성하지 못하거나, EOS 이후에도 계속 생성하는 패턴을 학습
- 기타: `<|endoftext|>` 1건, `EOS` 44건, `[PAD]` 3건

#### 문제 2: Output 내 질문/답변 마커 — 약 550건
- `"질문:"` 503건, `"답변:"` 430건 (output 내부)
- `"### 답변:"` 141건, `"### 질문:"` 10건
- `"### Instruction:"` 4건, `"### Response:"` 2건
- **영향:** 모델이 답변 중에 "질문:" → "답변:" 패턴을 학습하여 자체적으로 Q/A 루프를 생성

#### 문제 3: Self-repetition 패턴 — 57건
- 10-gram 기준 50% 이상 반복되는 output 57건
- **영향:** 반복 생성 패턴을 직접 학습

### 🟡 중간 (품질 저하)

#### 문제 4: 짧은 Output — 16,519건 (10.4%)
- 50자 미만 output이 전체의 10.4%
- 30자 미만은 8,833건
- **영향:** 모델이 충분히 긴 답변을 생성하는 능력 저하. 짧게 끝내야 할 곳에서 EOS를 배우지만, 대부분의 질문에서는 너무 짧은 답변 → EOS 미생성 → 계속 생성 → 루프

#### 문제 5: 낮은 한국어 비율 — 21,774건 (13.7%)
- 한글 문자 비율 30% 미만인 샘플 (코드, 영어, 중국어 등 혼재)
- `prepare_sft_data.py`의 필터가 이미 30% 기준을 적용하지만, 가중치 샘플링 이후 적용 순서 문제 가능성
- **영향:** 한국어 LLM으로서의 일관성 저하

---

## 3. 가설 검증 결과

### 가설 A: Output에 Q/A 루프 패턴 존재 → ⚠️ 부분 확인
- `### 질문: ... ### 답변:` 정확한 패턴: **4건** (0.003%)
- `질문: ... 답변:` 비공식 패턴: **119건** (0.07%)
- 단순 "질문:" 또는 "답변:" 포함: **~550건**
- **결론:** 정확한 루프 패턴은 극소수이나, "질문/답변" 키워드가 output에 포함된 샘플이 수백 건 존재. 이것만으로 루프의 주 원인이라 보기 어려움.

### 가설 B: 짧은 Output → ✅ 유력 원인
- 50자 미만 16,519건 (10.4%)이 output 분포의 상당 부분
- 모델이 짧은 답변 후 EOS를 생성하지 못하고 계속 토큰을 생성할 가능성
- **특히 `</s>` 토큰 오염(113건)과 결합하면:** 모델이 EOS 경계를 정확히 학습하지 못함

### 가설 C: 소스별 품질 편차 → ✅ 확인 (간접)
- `prepare_sft_data.py` 기준: KOR-OpenOrca-Platypus-v3 **5배 업샘플링**, kovast **0.8배 다운샘플링**
- 가중치가 매우 공격적 (5.0배는 동일 데이터 5회 반복 = 과적합 위험)
- kovast는 멀티턴 대화에서 첫 턴만 추출 → 문맥 부족으로 이상한 output 가능
- **결론:** 5배 업샘플링된 OpenOrca-Platypus가 주 학습 데이터를 지배. 해당 소스에 문제가 있으면 전체 모델에 직접 영향.

### 🔍 추가 발견: 반복 루프의 진짜 원인 추정
**EOS 학습 실패가 핵심.** 원인 조합:
1. Output 내 `</s>` 리터럴 (113건) → EOS 경계 혼란
2. 짧은 output 10.4% → EOS 타이밍 학습 불안정
3. 5000 steps로 159K 데이터 학습 → 각 샘플 평균 1.6 epoch도 안 됨 → underfitting 가능
4. **inference 시 repetition_penalty 미적용** (eval 코드에는 top_p/top_k만 있고 repetition_penalty 없음)

---

## 4. 즉시 적용 가능한 데이터 필터링 코드

```python
"""
enhanced_quality_filter.py — SFT 데이터 품질 강화 필터
Usage: python enhanced_quality_filter.py data/sft/train.jsonl data/sft/train_cleaned.jsonl
"""
import json
import re
import sys

def enhanced_filter(sample: dict) -> bool:
    instruction = sample.get("instruction", "").strip()
    output = sample.get("output", "").strip()
    
    # 1. 기본 길이 필터 (강화)
    if len(output) < 80:  # 50 → 80으로 상향
        return False
    if len(output) > 3000:  # 4000 → 3000으로 하향
        return False
    if len(instruction) < 15:
        return False
    
    # 2. 특수 토큰 제거
    BAD_TOKENS = ["</s>", "<|endoftext|>", "<|end|>", "<s>", "<pad>", "[PAD]", "<unk>"]
    for tok in BAD_TOKENS:
        if tok in output:
            return False
    
    # 3. Q/A 마커 오염 제거
    QA_PATTERNS = [
        r"###\s*(질문|답변|Instruction|Response|Input|Output)\s*:",
        r"^(질문|답변)\s*:",  # 줄 시작에서 "질문:" "답변:"
    ]
    for pat in QA_PATTERNS:
        if re.search(pat, output, re.MULTILINE):
            return False
    
    # 4. 한국어 비율 강화 (30% → 40%)
    ko_chars = sum(1 for c in output if '\uac00' <= c <= '\ud7a3')
    if len(output) > 0 and ko_chars / len(output) < 0.4:
        return False
    
    # 5. N-gram 반복 필터 (강화)
    words = output.split()
    if len(words) > 15:
        # 5-gram 반복 체크
        fivegrams = [tuple(words[i:i+5]) for i in range(len(words) - 4)]
        if fivegrams:
            unique_ratio = len(set(fivegrams)) / len(fivegrams)
            if unique_ratio < 0.7:  # 30% 이상 반복이면 제거
                return False
    
    # 6. "EOS" 리터럴 제거
    if re.search(r'\bEOS\b', output):
        return False
    
    return True


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    kept, dropped = 0, 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            if enhanced_filter(sample):
                fout.write(line)
                kept += 1
            else:
                dropped += 1
    
    print(f"Kept: {kept:,} | Dropped: {dropped:,} | Drop rate: {dropped/(kept+dropped)*100:.1f}%")


if __name__ == "__main__":
    main()
```

---

## 5. 데이터 파이프라인 개선 권장사항

### 5.1 가중치 재조정

현재 가중치가 너무 공격적. 권장 변경:

```python
DATASET_WEIGHTS = {
    "KOR-OpenOrca-Platypus-v3": 2.0,   # 5.0 → 2.0 (과적합 방지)
    "kullm-v2":                 1.0,
    "ko-alpaca-12k":            1.5,   # 2.0 → 1.5
    "korean_safe_conversation": 1.0,   # 1.5 → 1.0
    "evol-instruct-korean":     1.5,
    "kovast":                   0.5,   # 0.8 → 0.5 (품질 이슈)
}
```

### 5.2 학습 설정 수정

```bash
# 현재: 5000 steps, batch 4×8×2 = 64
# 159K samples / 64 = 2,486 steps/epoch → 현재 약 2 epochs

# 권장: 필터링 후 ~120K 데이터로 3 epochs
MAX_STEPS=6000
```

### 5.3 Inference 시 repetition_penalty 추가

```python
# eval/comprehensive_eval.py 수정
repetition_penalty = 1.2  # 반복 억제
```

---

## 6. 추천 고품질 데이터셋 (HuggingFace)

| 데이터셋 | URL | 설명 | 예상 크기 |
|----------|-----|------|-----------|
| Open-Orca Korean | `kyujinpy/KOR-OpenOrca-Platypus-v3` | 이미 사용 중 | - |
| ShareGPT Korean | `junelee/sharegpt_deepl_ko` | ShareGPT 한국어 번역 | ~90K |
| KoAlpaca v1.1 | `beomi/KoAlpaca-v1.1a` | 고품질 한국어 Alpaca | ~21K |
| LIMA Korean | `HAERAE-HUB/KMMLU` | 한국어 벤치마크 (평가용) | - |
| Korean HC3 | `heegyu/korean_chatgpt_corpus` | ChatGPT 한국어 대화 | ~12K |
| Orca DPO Korean | `kyujinpy/orca_dpo_pairs_ko` | DPO 페어 (SFT+DPO 가능) | ~12K |
| OpenHermes 2.5 Ko | `maywell/ko_Ultrafeedback_binarized` | 한국어 Ultrafeedback | ~60K |
| KOpen-platypus | `kyujinpy/KOpen-platypus` | 한국어 Platypus | ~25K |

**가장 추천하는 추가 데이터:**
1. `junelee/sharegpt_deepl_ko` — 다양한 주제의 멀티턴 대화, 충분히 긴 output
2. `heegyu/korean_chatgpt_corpus` — ChatGPT 품질 한국어 답변
3. `beomi/KoAlpaca-v1.1a` — 검증된 한국어 instruction 데이터

---

## 7. 요약: 즉시 조치 사항

| 우선순위 | 조치 | 예상 효과 |
|----------|------|-----------|
| 🔴 P0 | `</s>`, `<|endoftext|>`, `EOS` 포함 샘플 제거 (161건) | EOS 학습 혼란 해소 |
| 🔴 P0 | Output 최소 길이 80자로 상향 | 짧은 답변으로 인한 EOS 미학습 방지 |
| 🔴 P0 | Inference에 `repetition_penalty=1.2` 추가 | 즉시 반복 루프 완화 |
| 🟡 P1 | Q/A 마커 포함 샘플 제거 (~550건) | 자체 Q/A 루프 패턴 학습 방지 |
| 🟡 P1 | OpenOrca 가중치 5.0 → 2.0 | 과적합 방지, 다양성 확보 |
| 🟡 P1 | 한국어 비율 필터 40%로 강화 | 한국어 일관성 향상 |
| 🟢 P2 | 추가 고품질 데이터셋 수집 | 전반적 품질 향상 |
| 🟢 P2 | Self-repetition 필터 강화 (5-gram, 70% threshold) | 반복 패턴 원천 차단 |

**예상 필터링 후 데이터:** ~120,000-130,000 샘플 (현재 대비 18-25% 제거)

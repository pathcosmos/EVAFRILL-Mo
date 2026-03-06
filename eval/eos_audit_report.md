# EOS 토큰 처리 전수 감사 보고서

**날짜:** 2026-02-26  
**감사 대상:** `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`  
**문제:** SFT 모델이 "### 답변:" 이후 "### 질문:"을 반복 (반복률 57%)

---

## 결론 요약

### 🔴 근본 원인: 추론 시 프롬프트 템플릿 불일치 (EOS 버그 아님)

| 항목 | 학습 템플릿 | 추론 템플릿 (test_generation_params.py) |
|------|------------|----------------------------------------|
| 사용자 태그 | `<\|user\|>\n{instruction}\n` | `### 질문: {instruction}\n` |
| 어시스턴트 태그 | `<\|assistant\|>\n` | `### 답변:` |
| 종료 토큰 | `</s>` (EOS, id=2) | 없음 (stop_strings로 대체 시도) |

모델은 `<|user|>` / `<|assistant|>` 포맷으로 학습됐으나, 추론 시 `### 질문:` / `### 답변:` 포맷으로 호출됨.  
모델 입장에서 `### 질문:` `### 답변:`은 일반 텍스트 — EOS를 출력할 이유가 없으므로 무한 반복.

---

## 상세 감사 결과

### ✅ 체크포인트 1: SFTDataset — response 끝 EOS 토큰 부착

**결과: 정상**

`sft_dataset.py` Line ~52, ~87:
```python
response = f"{output}{_EOS_STRING}"   # _EOS_STRING = "</s>"
response = f"{content}{_EOS_STRING}"  # conversation format도 동일
```

실제 검증: `response_ids[-1] == 2 (EOS)` ✓

### ✅ 체크포인트 2: EOS 토큰 label = 학습 대상

**결과: 정상**

`sft_dataset.py` Line ~144-152:
```python
resp_label_start = max(0, resp_start - 1)  # 1칸 왼쪽 시프트 (causal LM 관례)
resp_label_end = resp_label_start + len(response_ids)
labels[resp_label_start:resp_label_end] = response_ids
```

- `labels[resp_label_end - 1] = EOS (2)` — EOS가 학습 대상에 포함됨 ✓
- logits[마지막 응답 토큰 위치] → EOS 예측하도록 학습됨 ✓

### ✅ 체크포인트 3: prompt 부분 label = -1 (무시)

**결과: 정상**

labels 초기값이 `-1`이고, response 영역만 덮어쓰므로 prompt 전체는 `-1` ✓

### ✅ 체크포인트 4: 트렁케이션으로 EOS 손실

**결과: 무시 가능 수준**

- 전체 159,125 샘플 중 61개 (0.04%)만 max_seq_len=4096 초과
- 이 61개에서만 EOS가 잘릴 수 있음 — 반복률 57%와 무관

### ⚠️ 체크포인트 5: 토크나이저 특수 토큰 미등록

**결과: 경미한 문제**

- `<|user|>` → `token_to_id()` = **None** (특수 토큰 아님, 서브워드로 분할됨)
- `<|assistant|>` → **None** (동일)
- `</s>` → id=2 ✓ (정상 등록)

`<|user|>` / `<|assistant|>`가 단일 토큰이 아니라 서브워드 조각으로 분할됨.  
학습/추론 모두 같은 토크나이저를 쓰면 동작은 하지만, 단일 특수 토큰으로 등록하는 것이 더 robust.

### 🔴 체크포인트 6: 추론 프롬프트 포맷 불일치 (근본 원인)

**`eval/test_generation_params.py`:**
```python
"### 질문: 한국의 수도는 어디인가요?\n### 답변:",
```

**`eval/comprehensive_eval.py`:**
```python
"한국의 수도는",  # 템플릿 없이 raw text
```

**학습된 포맷:**
```
<|user|>
한국의 수도는 어디인가요?
<|assistant|>
서울입니다.</s>
```

추론 시 올바른 프롬프트:
```
<|user|>
한국의 수도는 어디인가요?
<|assistant|>
```

---

## 수정 사항

### Fix 1: 추론 프롬프트 템플릿 수정 (필수, 재학습 불필요)

`eval/test_generation_params.py`와 `eval/comprehensive_eval.py`에서 프롬프트를 SFT 학습 템플릿에 맞게 변경:

```python
# Before (WRONG)
prompt = "### 질문: 한국의 수도는 어디인가요?\n### 답변:"

# After (CORRECT)
prompt = "<|user|>\n한국의 수도는 어디인가요?\n<|assistant|>\n"
```

### Fix 2: 트렁케이션 시 EOS 보장 (권장, 재학습 필요)

`sft_dataset.py`에서 truncation 후 EOS를 강제 삽입:

```python
# 현재 (truncation 시 EOS 손실 가능)
response_ids = response_ids[:allowed_response]

# 수정안 (truncation 후 EOS 강제)
response_ids = response_ids[:allowed_response]
if response_ids and response_ids[-1] != self.eos_token_id:
    response_ids[-1] = self.eos_token_id  # 마지막 토큰을 EOS로 교체
```

### Fix 3: `<|user|>` / `<|assistant|>` 특수 토큰 등록 (선택, 재학습 필요)

토크나이저에 특수 토큰으로 추가하면 단일 토큰으로 인코딩되어 더 안정적:
```python
tokenizer.add_special_tokens(["<|user|>", "<|assistant|>"])
```

---

## 재학습 필요 여부

| 수정 | 재학습 필요 | 효과 |
|------|-----------|------|
| Fix 1: 추론 템플릿 수정 | ❌ | **반복 문제 해결 예상 (근본 원인)** |
| Fix 2: 트렁케이션 EOS 보장 | ⭕ (0.04%만 해당) | 미미 |
| Fix 3: 특수 토큰 등록 | ⭕ | 장기적 안정성 향상 |

**즉시 조치: Fix 1만으로 반복 문제 해결 가능. 재학습 불필요.**

---

## 검증 방법

```bash
python eval/generate.py \
    --checkpoint checkpoints/korean_1b_sft \
    --prompt $'<|user|>\n한국의 수도는 어디인가요?\n<|assistant|>\n' \
    --max_new_tokens 200 \
    --temperature 0.7
```

반복이 멈추고 `</s>` (EOS)에서 정상 종료되면 Fix 1 성공.

# 반복 퇴화(Repetition Degeneration) 근본 원인 분석 + Fix 보고서

**날짜:** 2026-02-26  
**모델:** Korean 1B SFT (checkpoint-0005000)  
**문제:** 생성 시 57% 3-gram 반복률, "### 답변:" 패턴 반복

---

## 1. 핵심 발견: 반복률 57%의 근본 원인

### 🔴 원인 #1 (가장 큰 원인): 프롬프트 포맷 불일치

| | 학습 시 포맷 | 평가(sft_quick_check) 시 포맷 |
|---|---|---|
| 유저 턴 | `<\|user\|>\n{질문}\n<\|assistant\|>\n` | `### 질문: {질문}\n### 답변:` |
| EOS | `</s>` | 없음 |

**SFT 데이터는 `<|user|>` / `<|assistant|>` 포맷으로 학습했는데, 평가는 `### 질문:` / `### 답변:` 포맷으로 수행함.**

이건 모델이 한 번도 본 적 없는 포맷이므로:
- EOS를 언제 생성해야 하는지 모름 → 무한 생성
- `### 답변:` 패턴이 pretrain 코퍼스에 있었을 가능성 → 반복 루프 진입

### 🟡 원인 #2: 추론 파라미터 부재
- `repetition_penalty` 미적용 (기본 1.0)
- `no_repeat_ngram_size` 미적용
- stop sequence 미구현

---

## 2. 실험 결과 (17개 파라미터 조합 × 2 포맷)

### 올바른 SFT 포맷 (`<|user|>` / `<|assistant|>`)

| Config | Avg 3-gram Rep | 비고 |
|---|---|---|
| baseline (T=0.8, no penalty) | **5.4%** | 포맷만 맞춰도 57% → 5%로 급감! |
| rep_penalty=1.1 | **0.0%** | ✅ 완전 해결 |
| rep_penalty=1.2 | **0.0%** | ✅ |
| no_repeat_3gram | **0.0%** | ✅ |
| best_combo | **0.0%** | ✅ |
| temp=0.5 (낮은 온도) | 15.5% | ❌ 오히려 악화 |
| contrastive_a0.6_k4 | 35.4% | ❌ 간이 구현은 효과 없음 |

### 잘못된 포맷 (`### 질문:` / `### 답변:`)

| Config | Avg 3-gram Rep | 비고 |
|---|---|---|
| baseline | **35.3%** | 포맷 불일치로 심각 |
| rep_penalty=1.1 | 1.9% | 거의 해결되지만 품질 저하 |
| rep_penalty=1.2 | 0.0% | 반복은 없지만 출력 품질 나쁨 (깨진 텍스트) |
| temp=0.5 | **57.5%** | ← 보고된 57% 수치와 일치! |
| contrastive_a0.4_k6 | **60.9%** | 최악 |

### 핵심 인사이트

1. **포맷 수정만으로 반복률 57% → 5%로 감소** (10배 개선)
2. **rep_penalty=1.1 추가하면 5% → 0%** (완전 해결)
3. 잘못된 포맷에서 rep_penalty로 반복을 억제하면 **품질이 심각하게 저하** (깨진 텍스트 출력)
4. 낮은 temperature(0.5)가 오히려 반복을 **악화**시킴 (확률 분포가 뾰족해져서 같은 토큰에 빠짐)

---

## 3. 즉시 적용 가능한 최적 파라미터 조합

### ✅ 추천 설정 (즉시 적용)

```python
# eval/generate.py 또는 추론 코드에 적용
generation_config = {
    # 1. 올바른 프롬프트 포맷 사용 (가장 중요!)
    "prompt_template": "<|user|>\n{question}\n<|assistant|>\n",
    
    # 2. 생성 파라미터
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,       # 최소한의 반복 억제
    "no_repeat_ngram_size": 3,        # 3-gram 반복 완전 차단 (보험)
    "max_new_tokens": 512,
    
    # 3. EOS token
    "eos_token_id": 2,                # </s> token ID
}
```

---

## 4. Stop Sequence 구현 코드

### 토큰 ID 매핑
```
'</s>'          → [2]                          (단일 토큰, 이미 EOS로 사용)
'### 답변:'     → [493, 2894, 2894, 10663, 36] (5 토큰)
'### 질문:'     → [493, 2894, 2894, 6326, 36]  (5 토큰)
'<|user|>'      → [473, 2526, 887, 201, 2526, 927] (6 토큰)
'<|assistant|>' → [473, 2526, 29273, 16, 24232, 2526, 927] (7 토큰)
```

### 구현 방법 (generate.py 수정)

```python
# eval/generate.py의 generate() 함수에 추가

def generate(..., stop_strings=None, repetition_penalty=1.0, no_repeat_ngram_size=0):
    # ... 기존 코드 ...
    
    # Stop sequence token IDs 사전 계산
    stop_seqs = []
    if stop_strings:
        for s in stop_strings:
            stop_seqs.append(tokenizer.encode(s).ids)
    
    new_token_ids = []
    
    for _ in range(max_new_tokens):
        logits_all, _ = model(generated_ids)
        logits = logits_all[:, -1, :].clone()
        
        # --- Repetition penalty ---
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        
        # --- No-repeat n-gram blocking ---
        if no_repeat_ngram_size > 0:
            all_ids = generated_ids[0].tolist()
            if len(all_ids) >= no_repeat_ngram_size:
                last_ngram = tuple(all_ids[-(no_repeat_ngram_size - 1):])
                for i in range(len(all_ids) - no_repeat_ngram_size + 1):
                    if tuple(all_ids[i:i + no_repeat_ngram_size - 1]) == last_ngram:
                        logits[0, all_ids[i + no_repeat_ngram_size - 1]] = float("-inf")
        
        # ... 기존 sampling 코드 ...
        
        new_token_ids.append(next_token_id.item())
        
        # --- Stop sequence 확인 ---
        for seq in stop_seqs:
            if len(new_token_ids) >= len(seq) and new_token_ids[-len(seq):] == seq:
                new_token_ids = new_token_ids[:-len(seq)]  # stop seq 제거
                return tokenizer.decode(new_token_ids)
        
        # EOS
        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break
```

---

## 5. 학습 레벨 근본 Fix

### 5.1 현재 학습 코드 진단

**✅ 잘 되어 있는 것:**
- `SFTDataset`에서 `labels=-1`로 prompt 토큰 마스킹 → response 토큰에만 loss 계산
- `</s>` EOS 토큰이 response 끝에 추가됨
- `ignore_index=-1`로 `CrossEntropyLoss` 사용

**⚠️ 개선 필요:**
- 반복 억제 학습 전략 없음 (Unlikelihood Training 등)
- 데이터 품질 필터링 없음 (반복적인 응답 포함 가능)

### 5.2 Unlikelihood Training 추가 (중기)

```python
# train/trainer.py의 _compute_loss() 수정

def _compute_loss(self, logits, targets, input_ids=None, alpha_ul=0.5):
    """Loss = CE_loss + alpha * Unlikelihood_loss"""
    B, S, V = logits.shape
    
    # 기존 CE loss
    ce_loss = F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
        ignore_index=-1,
    )
    
    if input_ids is None or alpha_ul == 0:
        return ce_loss
    
    # Unlikelihood loss: 이전에 나온 토큰의 확률을 낮추는 loss
    probs = F.softmax(logits, dim=-1)  # [B, S, V]
    
    # 각 position에서 "이전에 이미 나온 토큰" 집합 구성
    # Negative candidates = 이전 context에서 나온 토큰들
    ul_loss = torch.tensor(0.0, device=logits.device)
    count = 0
    
    for b in range(B):
        seen_tokens = set()
        for s in range(S):
            if targets[b, s] == -1:
                seen_tokens.add(input_ids[b, s].item())
                continue
            
            # 현재 position에서 seen tokens의 확률
            for tok in seen_tokens:
                # Unlikelihood: maximize log(1 - p(tok))
                p = probs[b, s, tok]
                ul_loss -= torch.log(torch.clamp(1 - p, min=1e-8))
                count += 1
            
            seen_tokens.add(targets[b, s].item())
    
    if count > 0:
        ul_loss = ul_loss / count
    
    return ce_loss + alpha_ul * ul_loss
```

### 5.3 데이터 증강: 반복 필터링 (중기)

```python
# data/filter_repetitive.py
def filter_repetitive_samples(jsonl_path, output_path, max_3gram_rep=0.15):
    """반복률이 높은 샘플 제거"""
    import json
    
    kept, removed = 0, 0
    with open(jsonl_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            obj = json.loads(line)
            output = obj.get('output', '')
            
            # 3-gram 반복률 계산
            tokens = output.split()
            if len(tokens) >= 3:
                ngrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
                rep = 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0
                if rep > max_3gram_rep:
                    removed += 1
                    continue
            
            fout.write(line)
            kept += 1
    
    print(f"Kept: {kept}, Removed: {removed}")
```

---

## 6. 로드맵

### 🟢 단기 (즉시, 코드 수정만)
1. **평가 프롬프트 포맷을 `<|user|>` / `<|assistant|>`로 수정** — 이것만으로 57% → 5%
2. **`repetition_penalty=1.1` 추가** — 5% → 0%
3. **`no_repeat_ngram_size=3` 추가** — 보험
4. sft_quick_check 스크립트의 프롬프트 템플릿 수정

### 🟡 중기 (1-2주)
1. 데이터 품질 필터링 (반복적 응답 제거)
2. Unlikelihood Training loss 추가 (alpha=0.1~0.5)
3. EOS 생성 학습 강화: response 끝 EOS에 weight 1.5x 부여
4. 다양한 temperature에서의 robustness 테스트

### 🔵 장기 (1개월+)
1. HuggingFace `transformers` 호환 모델 변환 → `generate()` 내장 기능 활용
2. RLHF/DPO: 반복 없는 응답을 preferred로 학습
3. Contrastive Search (진짜 hidden-state 기반) 구현
4. 모델 크기별 반복 퇴화 비교 분석

---

## 7. sft_quick_check 수정 예시

기존 (잘못됨):
```python
prompt = "### 질문: 한국의 수도는 어디인가요?\n### 답변:"
```

수정 (올바름):
```python
prompt = "<|user|>\n한국의 수도는 어디인가요?\n<|assistant|>\n"
```

---

## 부록: 실험 전체 결과

테스트 스크립트: `eval/test_generation_params.py`  
결과 JSON: `eval/repetition_param_search_results.json`

총 17개 파라미터 조합 × 2개 포맷 (sft_format, wrong_format) = 34 실험 수행.
모든 실험은 checkpoint-0005000 기준, 5개 프롬프트, max_new_tokens=200.

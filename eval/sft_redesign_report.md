# SFT 학습 파이프라인 근본 재설계 보고서

**작성일**: 2026-02-26  
**대상 프로젝트**: `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/`  
**현재 상태**: 반복 퇴화 57%, EOS 생성 불안정

---

## 1. 현재 구현의 근본적 문제점

### 🔴 Critical #1: Dynamic Padding이 작동하지 않음 (가장 큰 성능 낭비)

**파일**: `data/sft_dataset.py` L139-146, `train/sft.py` L198-230

`SFTDataset.__init__`에서 모든 샘플을 `max_seq_len`(4096)으로 **미리 패딩**한다:

```python
# sft_dataset.py L139-141
input_ids = torch.full(
    (max_seq_len,), fill_value=pad_token_id, dtype=torch.long
)
```

그런데 `dynamic_collate_fn`은 배치 내 최대 길이에 맞춰 패딩하도록 설계됐지만, 이미 모든 텐서가 `max_seq_len` 길이이므로 **raw_max = max_seq_len 항상 고정**. Dynamic padding이 사실상 무효화된 상태.

**영향**: 평균 시퀀스 길이가 ~500 토큰이라면, 매 스텝마다 ~3600개의 패딩 토큰을 불필요하게 처리. **학습 속도 ~3-8x 저하**, GPU FLOPs 낭비.

**수정**: `__getitem__`에서 패딩하지 말고, 실제 길이의 텐서를 반환. `dynamic_collate_fn`이 배치별로 패딩하도록.

### 🔴 Critical #2: 트렁케이션 시 EOS 토큰 손실

**파일**: `data/sft_dataset.py` L130-134

```python
# truncation이 발생하면
response_ids = response_ids[:allowed_response]
```

`response_ids`는 `"{output}</s>"`를 인코딩한 것인데, 잘리면 마지막의 `</s>` 토큰이 제거된다. 해당 샘플은 EOS를 학습할 수 없다.

**영향**: `truncated_count`개의 샘플이 EOS 없이 학습됨. 긴 응답 → 잘림 → EOS 미학습 → 생성 시 끝없이 반복 생성의 직접적 원인.

**수정**:
```python
response_ids = response_ids[:allowed_response - 1] + [self.eos_token_id]
```

### 🟡 Important #3: Label Shift 로직 미묘한 버그 가능성

**파일**: `data/sft_dataset.py` L148-157, `train/trainer.py` L263-274

현재 구현:
- `labels[resp_start-1 : resp_start-1+len(response_ids)] = response_ids`
- `_compute_loss`에서 **별도의 shift 없이** `cross_entropy(logits[i], labels[i])` 수행

이 패턴은 "logits[i]가 position i+1을 예측한다"는 가정 하에 올바르지만, **마지막 response 토큰 이후 position에 대한 예측이 학습되지 않는다**. 즉, EOS를 출력한 후 다음 토큰을 패딩(또는 아무것도 아닌 것)으로 예측하도록 학습하지 않아서, EOS 이후에도 계속 토큰을 생성하는 경향이 있을 수 있다.

**권장**: response_ids의 마지막 토큰(EOS) 다음 position의 label도 EOS로 설정하여 "EOS 이후에도 EOS"를 학습시킴.

### 🟡 Important #4: Validation Split 없음

**파일**: `scripts/launch_sft.sh` — `--val_data` 인자 미전달

Trainer에 validation 루프가 구현돼 있고 (`trainer.py` L200-220), best checkpoint 저장도 되지만, 실제로 val_data를 전달하지 않아 과적합 모니터링이 불가능.

### 🟡 Important #5: 학습 에포크 부족

- Effective batch = 4 × 2 × 8 = **64 seqs/step**
- 5000 steps × 64 = **320,000 sample-steps**
- 159k 샘플 → **~2 epochs**

SFT는 보통 **3-5 epochs**가 적절. 2 epochs는 데이터를 충분히 학습하지 못함.

### 🟢 Minor #6: Warmup 비율

150/5000 = 3%. SFT에서는 적절한 범위 (3-10%).

### 🟢 Minor #7: NEFTune 사용 중

`noise_alpha=10.0`으로 NEFTune 적용 중. 이건 좋은 설정. 다만 반복 퇴화가 심한 경우 효과가 제한적.

---

## 2. 업계 최고 수준 SFT 구현과의 비교

### LLaMA-Factory
- **completion_only_loss**: prompt 토큰 masking + response만 학습 (현재 구현과 동일)
- **EOS 보장**: 트렁케이션 시 반드시 EOS 토큰 유지
- **Packing**: 여러 짧은 샘플을 하나의 시퀀스에 패킹하여 GPU 활용률 극대화
- **데이터 필터링**: 너무 짧거나 품질 낮은 샘플 자동 제거

### TRL SFTTrainer
- `packing=True`: 여러 샘플을 하나의 시퀀스에 패킹 (패딩 낭비 제거)
- `DataCollatorForCompletionOnlyLM`: response 토큰만 학습하는 collator
- **max_seq_length에 맞춰 동적 패딩** (배치별)

### Axolotl
- **Sample packing**: 긴 시퀀스 내에 여러 대화를 패킹
- **Sequence length warmup**: 짧은 시퀀스부터 시작해서 점진적으로 길이 증가
- **Repetition penalty during training**: 학습 중 반복 패널티 적용 옵션

### OpenInstruct (Allen AI)
- **데이터 품질 우선**: 데이터 필터링 파이프라인 강조
- **길이 기반 정규화**: 긴 응답에 대한 loss normalization
- **다단계 학습**: 일반 SFT → 도메인 SFT → DPO/RLHF

### 반복 퇴화 방지를 위한 업계 공통 패턴
1. **EOS 토큰 확실한 학습**: 모든 샘플에서 EOS가 label에 포함되도록 보장
2. **Repetition penalty loss**: 동일 n-gram 반복 시 추가 페널티
3. **Length normalization**: 긴 응답의 loss가 과도하게 커지지 않도록 정규화
4. **데이터 품질 필터링**: 반복이 있는 학습 데이터 자체를 제거
5. **KL divergence regularization**: base model과의 KL divergence 제약

---

## 3. Curriculum Learning 방법론

### IFD (Instruction Following Difficulty) Score
```python
# 각 샘플에 대해:
# 1. base model로 response의 perplexity 계산
# 2. instruction 없이 response만의 perplexity 계산
# 3. IFD = conditional_ppl / unconditional_ppl
# IFD가 낮을수록 "쉬운" 샘플

def compute_ifd(model, tokenizer, instruction, response):
    cond_ppl = compute_perplexity(model, instruction + response)
    uncond_ppl = compute_perplexity(model, response)
    return cond_ppl / uncond_ppl
```

### KL-Divergence 기반 데이터 선택
- Base model 대비 응답 분포가 크게 다른 샘플 = 어려운 샘플
- 처음에는 KL이 작은 (쉬운) 샘플, 나중에 큰 샘플

### 실용적 접근: 길이 기반 Curriculum
- Phase 1 (epoch 1): 응답 길이 < 256 토큰 샘플만
- Phase 2 (epoch 2): 응답 길이 < 1024 토큰
- Phase 3 (epoch 3+): 전체 데이터

---

## 4. 즉시 적용 가능한 수정 Top 5

### Fix #1: Dynamic Padding 실제 작동하도록 수정 (성능 3-8x 개선)

```python
# data/sft_dataset.py — __getitem__ 수정
# 기존: 고정 max_seq_len 패딩된 텐서 반환
# 수정: 실제 길이의 텐서만 반환

class SFTDataset(Dataset):
    def __init__(self, ...):
        # ...기존 코드...
        # samples 저장 시 패딩 제거
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        
        for prompt_text, response_text in raw_samples:
            # ...기존 인코딩 코드...
            
            seq_len = len(full_ids)
            # 패딩 없이 실제 길이만 저장
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            
            labels = torch.full((seq_len,), fill_value=-1, dtype=torch.long)
            resp_start = len(prompt_ids)
            resp_label_start = max(0, resp_start - 1)
            resp_label_end = resp_label_start + len(response_ids)
            labels[resp_label_start:resp_label_end] = torch.tensor(
                response_ids, dtype=torch.long
            )
            
            self.samples.append((input_ids, labels))
```

### Fix #2: 트렁케이션 시 EOS 보존

```python
# data/sft_dataset.py L130-134 수정
# 기존:
#   response_ids = response_ids[:allowed_response]
# 수정:
if len(full_ids) > max_seq_len:
    truncated_count += 1
    allowed_response = max_seq_len - len(prompt_ids)
    if allowed_response <= 1:
        skipped_too_long += 1
        continue
    # EOS를 반드시 마지막에 유지
    response_ids = response_ids[:allowed_response - 1] + [self.eos_token_id]
    full_ids = prompt_ids + response_ids
```

### Fix #3: Validation Split 추가

```bash
# 데이터 분리 스크립트
python -c "
import json, random
random.seed(42)
with open('data/sft/train.jsonl') as f:
    lines = f.readlines()
random.shuffle(lines)
split = int(len(lines) * 0.9)
with open('data/sft/train_split.jsonl', 'w') as f:
    f.writelines(lines[:split])
with open('data/sft/val_split.jsonl', 'w') as f:
    f.writelines(lines[split:])
print(f'Train: {split}, Val: {len(lines)-split}')
"

# launch_sft.sh에 추가:
# --val_data data/sft/val_split.jsonl
```

### Fix #4: Max Steps 증가 (3-5 epochs)

```bash
# launch_sft.sh 수정
# 159k samples / 64 effective_batch ≈ 2484 steps/epoch
# 4 epochs → ~10,000 steps
MAX_STEPS=10000
WARMUP_STEPS=300  # 3% of 10000
```

### Fix #5: 학습 데이터에서 반복 패턴 필터링

```python
# data/filter_repetitive.py
import json, re

def has_repetition(text, n=3, threshold=0.3):
    """n-gram 반복 비율이 threshold 이상이면 True"""
    words = text.split()
    if len(words) < n * 2:
        return False
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < (1 - threshold)

filtered = 0
with open('data/sft/train.jsonl') as fin, \
     open('data/sft/train_clean.jsonl', 'w') as fout:
    for line in fin:
        obj = json.loads(line)
        text = obj.get('output', '') or ''
        if 'conversations' in obj:
            text = ' '.join(t['content'] for t in obj['conversations'] 
                          if t['role'] == 'assistant')
        if not has_repetition(text):
            fout.write(line)
        else:
            filtered += 1

print(f"Filtered {filtered} repetitive samples")
```

---

## 5. 근본적 재학습 시나리오

### Phase 1: 데이터 준비 (1-2시간)

1. 반복 패턴이 있는 데이터 필터링
2. Train/Val split (90/10)
3. 데이터 통계 확인 (길이 분포, 카테고리 분포)

### Phase 2: 코드 수정 (1-2시간)

1. ✅ `sft_dataset.py`: 패딩 제거, EOS 보존, 가변 길이 반환
2. ✅ `sft.py`의 `dynamic_collate_fn`: 이미 구현됨 (dataset 수정만 하면 작동)
3. ✅ `launch_sft.sh`: val_data 추가, max_steps 10000

### Phase 3: 학습 (8-12시간, 8×B200 기준)

```bash
# 수정된 launch_sft.sh
MAX_STEPS=10000
BATCH_SIZE=4
GRAD_ACCUM=2
LR="2.0e-5"
WARMUP_STEPS=300

torchrun --nproc_per_node=8 train/sft.py \
    --base_checkpoint checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \
    --sft_data data/sft/train_clean.jsonl \
    --val_data data/sft/val_split.jsonl \
    --max_steps 10000 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 2.0e-5 \
    --warmup_steps 300 \
    --use_fp8
```

### Phase 4: 평가 (1시간)

1. Val loss 모니터링 (TensorBoard)
2. 반복 퇴화율 측정 (기존 eval 스크립트)
3. Best checkpoint 선택 (val_loss 기준)

### 예상 소요시간
- Dynamic padding 수정 후 학습 속도: **3-5x 향상** (4096→~600 avg 기준)
- 10000 steps: 기존 대비 비슷하거나 더 빠를 수 있음
- 총 12-16시간 (데이터 준비 ~ 평가 완료)

---

## 6. 예상 개선 효과

| 수정 항목 | 현재 | 예상 개선 | 근거 |
|-----------|------|----------|------|
| Dynamic padding | 4096 고정 | 학습속도 3-8x↑ | 평균 ~600 토큰 시 패딩 85% 감소 |
| EOS 보존 | 트렁케이션 시 손실 | 반복퇴화 57%→~20% | EOS 학습 누락이 반복의 직접 원인 |
| 데이터 필터링 | 반복 데이터 포함 | 반복퇴화 → ~10% | 반복 학습데이터 제거 |
| Val split | 없음 | 과적합 조기 감지 | best checkpoint 선택 가능 |
| Epoch 증가 | ~2 epochs | 수렴 개선 | 3-5 epochs이 SFT 표준 |
| **종합** | **반복 57%** | **반복 <10%** | 위 수정 모두 적용 시 |

---

## 7. 권장 아키텍처 (장기)

현재 직접 구현한 SFT 파이프라인을 **TRL SFTTrainer** 또는 **LLaMA-Factory**로 교체 권장:

1. **Packing 지원**: 패딩 낭비 완전 제거
2. **검증된 EOS 처리**: 수백 개 프로젝트에서 검증
3. **DPO/RLHF 파이프라인 연계**: SFT 후 alignment 학습으로 자연스럽게 이행
4. **Sample packing + Flash Attention**: 최적화된 메모리/속도

단, 현재 커스텀 모델(`LLM` 클래스)과의 호환성 확인 필요. HuggingFace 포맷으로 변환 후 사용하면 바로 적용 가능.

---

## 요약: 즉시 해야 할 일 (우선순위)

1. 🔴 **sft_dataset.py**: 패딩 제거 + EOS 보존 (30분)
2. 🔴 **데이터 필터링**: 반복 패턴 있는 샘플 제거 (30분)
3. 🟡 **Val split 생성 + launch_sft.sh 수정** (15분)
4. 🟡 **Max steps 10000으로 증가** (설정만 변경)
5. 🟢 **재학습 시작** (수정 완료 후)

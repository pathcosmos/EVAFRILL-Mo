# FRANKENSTALLM-H 3B Hybrid Model — 점검 결과 및 수정 실행 가이드

> **작성일**: 2026-03-05
> **목적**: Phase 2 검증 전, 발견된 이슈 6건을 수정하고 바로 실행 가능한 상태로 만든다.
> **다음 세션에서 이 문서를 참조하여 바로 실행할 것.**

---

## 이슈 요약 (6건)

| # | 심각도 | 이슈 | 파일 | 영향 |
|---|--------|------|------|------|
| 1 | **CRITICAL** | Mamba 블록에 FFN(channel mixer) 없음 | `model/mamba_block.py` | 37/40 레이어 capacity 부족 |
| 2 | **HIGH** | `n_groups=1` (Nemotron 표준은 8) | `configs/hybrid_3b.yaml` | B/C projection 표현력 저하 |
| 3 | **HIGH** | Hybrid 아키텍처 startup 로그 없음 | `train/pretrain.py` | 디버깅·모니터링 곤란 |
| 4 | **MEDIUM** | 체크포인트 resume 시 아키텍처 검증 없음 | `train/utils.py` | 잘못된 가중치 로드 가능 |
| 5 | **MEDIUM** | selective_scan에 NaN/Inf 감지 없음 | `model/mamba_block.py` | 수치 불안정 진단 불가 |
| 6 | **LOW** | selective_scan 입력 shape 검증 없음 | `model/mamba_block.py` | 모호한 에러 메시지 |

---

## 구현 순서 및 의존성

```
Step 1 (FFN 추가) ← 가장 먼저, 아키텍처 변경
  ├── 1a. model/config.py: mamba_d_ffn 필드 추가
  ├── 1b. model/mamba_block.py: FFN sublayer 추가
  ├── 1c. model/transformer.py: 생성자 인자 전달 + _init_weights 수정
  └── 1d. configs/hybrid_3b.yaml: mamba_d_ffn=4608 추가

Step 2 (n_groups) ← Step 1과 독립, 병렬 가능
  └── configs/hybrid_3b.yaml: n_groups=8

Step 3 (로그) ← Step 1 완료 후 (파라미터 수 정확해야)
  └── train/pretrain.py: startup 배너에 hybrid 정보 추가

Step 4 (체크포인트 검증) ← 독립
  └── train/utils.py: load_checkpoint에 config 비교 로직

Step 5-6 (NaN 감지 + shape 검증) ← 독립
  └── model/mamba_block.py: selective_scan 함수
```

**병렬 가능**: Step 1 + Step 2는 YAML만 겹침 (마지막에 합치면 됨).
Step 4, Step 5-6도 독립적으로 병렬 실행 가능.

---

## Step 1: Mamba2Block에 FFN 추가 (CRITICAL)

### 배경

- Mamba2Block은 SSM(sequence mixer)만 있고 FFN(channel mixer)이 없음
- Nemotron-H에서는 모든 Mamba 레이어 뒤에 MLP가 따라옴
- 현재 37/40 레이어에 FFN이 없어 feature mixing이 불가능
- **확정**: `mamba_d_ffn = 4608` (d_model × 1.5), 총 파라미터 ~4.5B, VRAM ~80GB/GPU

### 1a. `model/config.py` 수정

**위치**: LMConfig dataclass 내부 (line 61 이후)

**추가할 필드** (기존 `mamba_chunk_size` 뒤에):
```python
    mamba_d_ffn: Optional[int] = None  # FFN dim for Mamba blocks (None → d_ffn)
```

**`__post_init__` 추가** (line 86, hybrid validation 블록 뒤에):
```python
        # Mamba FFN dimension: default to d_ffn if not specified
        if self.mamba_d_ffn is None:
            self.mamba_d_ffn = self.d_ffn
```

**`to_dict()` 추가** (기존 mamba_chunk_size 뒤에):
```python
            "mamba_d_ffn": self.mamba_d_ffn,
```

### 1b. `model/mamba_block.py` 수정

**Import 변경** (line 19):
```python
# 변경 전:
from .layers import RMSNorm

# 변경 후:
from .layers import RMSNorm, SwiGLU
```

**`Mamba2Block.__init__` 시그니처 변경** (line 128-137):
```python
# 변경 전:
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        head_dim: int = 64,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 1,
        chunk_size: int = 256,
    ) -> None:

# 변경 후:
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        head_dim: int = 64,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 1,
        chunk_size: int = 256,
        d_ffn: int = 0,
        bias: bool = False,
    ) -> None:
```

**FFN 서브레이어 추가** (line 192, `self.out_proj` 뒤에):
```python
        # --- FFN sublayer (channel mixer) ---
        if d_ffn > 0:
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = SwiGLU(d_model, d_ffn, bias=bias)
        else:
            self.ffn_norm = None
            self.ffn = None
```

**`forward()` 수정** (line 280):
```python
# 변경 전:
        return residual + self.out_proj(y)

# 변경 후:
        x = residual + self.out_proj(y)
        # FFN sublayer (channel mixer)
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))
        return x
```

### 1c. `model/transformer.py` 수정

**Mamba2Block 생성자 호출 변경** (line 124-132):
```python
# 변경 전:
                    layers.append(Mamba2Block(
                        d_model=config.d_model,
                        d_state=config.mamba_d_state,
                        head_dim=config.mamba_head_dim,
                        expand=config.mamba_expand,
                        conv_kernel=config.mamba_conv_kernel,
                        n_groups=config.mamba_n_groups,
                        chunk_size=config.mamba_chunk_size,
                    ))

# 변경 후:
                    layers.append(Mamba2Block(
                        d_model=config.d_model,
                        d_state=config.mamba_d_state,
                        head_dim=config.mamba_head_dim,
                        expand=config.mamba_expand,
                        conv_kernel=config.mamba_conv_kernel,
                        n_groups=config.mamba_n_groups,
                        chunk_size=config.mamba_chunk_size,
                        d_ffn=config.mamba_d_ffn,
                        bias=config.bias,
                    ))
```

**`_init_weights` 수정** (line 180-182):
```python
# 변경 전:
        # Mamba2Block handles its own parameter init (A_log, D, dt_bias, etc.)
        if isinstance(module, Mamba2Block):
            return

# 변경 후 (이 3줄을 삭제):
# 삭제 이유: FFN 추가 후 내부 SwiGLU의 nn.Linear가 init 필요.
# A_log, D, dt_bias는 nn.Parameter이므로 isinstance(nn.Linear) 체크에 걸리지 않아
# 자동으로 스킵됨 (Mamba2Block.__init__에서 직접 초기화됨).
```

### 1d. `configs/hybrid_3b.yaml` 수정

```yaml
# mamba_chunk_size: 256 뒤에 추가:
  mamba_d_ffn: 4608
```

### Step 1 검증

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, sys
sys.path.insert(0, '.')
from model import LLM, LMConfig

config = LMConfig.from_yaml('configs/hybrid_3b.yaml')
print(f'mamba_d_ffn = {config.mamba_d_ffn}')

model = LLM(config)
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total:,} ({total/1e9:.2f}B)')

# Forward test
x = torch.randint(0, 64000, (1, 128))
logits, loss = model(x, targets=x)
print(f'Forward OK: logits shape={logits.shape}, loss={loss.item():.4f}')

# Backward test
loss.backward()
grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f'Backward OK: all grads exist = {grads_ok}')
"
# 예상 출력: Total params ~4.5B, Forward/Backward OK
```

---

## Step 2: n_groups 수정

### `configs/hybrid_3b.yaml`

```yaml
# 변경 전:
  mamba_n_groups: 1

# 변경 후:
  mamba_n_groups: 8
```

### 검증

n_heads(= d_inner / head_dim = 6144 / 64 = 96) % 8 == 0 ✓
Step 1 검증 스크립트에서 함께 확인됨 (assertion이 `__init__`에 있음).

---

## Step 3: 하이브리드 아키텍처 startup 로그 추가

### `train/pretrain.py` 수정

**위치**: line 296-297 (모델 파라미터 출력 부분) 뒤에 추가

```python
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"LMConfig: {lm_config}")

        # --- 여기부터 추가 ---
        if lm_config.use_hybrid:
            pattern = lm_config.hybrid_pattern.split()
            m_count = sum(1 for p in pattern if p == 'M')
            a_count = sum(1 for p in pattern if p == 'A')
            mamba_params = sum(
                p.numel() for n, p in model.named_parameters()
                if 'layers.' in n and pattern[int(n.split('.')[1])] == 'M'
            )
            attn_params = sum(
                p.numel() for n, p in model.named_parameters()
                if 'layers.' in n and pattern[int(n.split('.')[1])] == 'A'
            )
            other_params = total_params - mamba_params - attn_params
            print(
                f"  arch     : Hybrid Mamba-Transformer\n"
                f"  layers   : {m_count} Mamba + {a_count} Attention = {len(pattern)} total\n"
                f"  params   : Mamba {mamba_params/1e6:.0f}M + "
                f"Attn {attn_params/1e6:.0f}M + Other {other_params/1e6:.0f}M\n"
                f"  mamba cfg: d_state={lm_config.mamba_d_state}, "
                f"head_dim={lm_config.mamba_head_dim}, "
                f"expand={lm_config.mamba_expand}, "
                f"n_groups={lm_config.mamba_n_groups}, "
                f"d_ffn={lm_config.mamba_d_ffn}"
            )
        # --- 추가 끝 ---
```

### 검증

Step 1 검증 실행 시 로그에 hybrid 정보가 출력되는지 확인.

---

## Step 4: 체크포인트 resume 아키텍처 검증

### `train/utils.py` — `load_checkpoint()` 수정

**위치**: line 179 (`raw_model.load_state_dict(...)`) 직전에 추가

```python
    # --- Architecture validation ---
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists() and hasattr(raw_model, "config"):
        with open(config_path, "r", encoding="utf-8") as f:
            saved_cfg = yaml.safe_load(f)
        current_cfg = raw_model.config.to_dict()
        critical_keys = [
            "d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
            "use_hybrid", "hybrid_pattern",
        ]
        mismatches = []
        for key in critical_keys:
            saved_val = saved_cfg.get(key)
            current_val = current_cfg.get(key)
            if saved_val is not None and saved_val != current_val:
                mismatches.append(
                    f"  {key}: checkpoint={saved_val} vs current={current_val}"
                )
        if mismatches:
            raise ValueError(
                f"Checkpoint architecture mismatch!\n"
                f"Checkpoint dir: {ckpt_dir}\n"
                + "\n".join(mismatches)
                + "\nUse --config matching the checkpoint, or start fresh."
            )
    # --- End architecture validation ---
```

**참고**: `yaml`은 이미 `train/utils.py` line 23에서 import 되어 있음.

### 검증

```bash
# 의도적으로 다른 config로 resume 시도
CUDA_VISIBLE_DEVICES=0 python train/pretrain.py \
    --config configs/small.yaml \
    --train_data data/3b_train.bin \
    --resume checkpoints/hybrid_3b_run1/checkpoint-0001000
# 예상: ValueError "Checkpoint architecture mismatch!" 출력
```

---

## Step 5: selective_scan NaN/Inf 감지

### `model/mamba_block.py` — `selective_scan()` 수정

**위치**: line 94 (`y[:, t, :, :] = y_t.to(x.dtype)`) 뒤에 추가

```python
        # Periodic NaN/Inf check (every 512 steps, < 1% overhead)
        if t % 512 == 511:
            if not torch.isfinite(h).all():
                raise RuntimeError(
                    f"NaN/Inf in Mamba SSM state at timestep {t}/{seq_len}. "
                    f"h stats: min={h.min().item():.4e}, max={h.max().item():.4e}, "
                    f"A_log range=[{A_log.min().item():.4f}, {A_log.max().item():.4f}]"
                )
```

### 검증

```bash
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, sys
sys.path.insert(0, '.')
from model.mamba_block import Mamba2Block

block = Mamba2Block(d_model=256, d_state=64, head_dim=32, d_ffn=384)
x = torch.randn(1, 1024, 256)

# 정상 케이스
y = block(x)
print(f'Normal: output shape={y.shape}, finite={torch.isfinite(y).all()}')

# NaN 주입 테스트
block.A_log.data.fill_(100.0)  # 매우 큰 값 → exp(100) overflow
try:
    y = block(x)
    print('WARNING: NaN not detected!')
except RuntimeError as e:
    print(f'NaN correctly detected: {e}')
"
```

---

## Step 6: selective_scan 입력 shape 검증

### `model/mamba_block.py` — `selective_scan()` 수정

**위치**: line 49 (`batch, seq_len, n_heads, head_dim = x.shape`) 직전에 추가

```python
    # Input shape validation
    assert x.ndim == 4, f"x expected 4D (B,L,n_heads,head_dim), got {x.shape}"
    assert dt.ndim == 3, f"dt expected 3D (B,L,n_heads), got {dt.shape}"
    assert B.ndim == 4, f"B expected 4D (B,L,n_groups,d_state), got {B.shape}"
    assert C.ndim == 4, f"C expected 4D (B,L,n_groups,d_state), got {C.shape}"
```

---

## 최종 검증 절차 (모든 Step 완료 후)

### 1. 모델 생성 + Forward/Backward (단일 GPU)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, sys
sys.path.insert(0, '.')
from model import LLM, LMConfig

config = LMConfig.from_yaml('configs/hybrid_3b.yaml')
model = LLM(config).cuda()

total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total:,} ({total/1e9:.2f}B)')
assert 4.0e9 < total < 5.0e9, f'Expected ~4.5B params, got {total/1e9:.2f}B'

# Forward
x = torch.randint(0, 64000, (2, 512)).cuda()
logits, loss = model(x, targets=x)
print(f'Forward: logits={logits.shape}, loss={loss.item():.4f}')

# Backward
loss.backward()
no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
assert len(no_grad) == 0, f'Missing gradients: {no_grad}'
print(f'Backward: all {sum(1 for p in model.parameters() if p.requires_grad)} params have grad')

# VRAM
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated')
"
```

### 2. DDP 8-GPU 테스트 (10 steps)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
torchrun --nproc_per_node=8 --master_port=29501 train/pretrain.py \
    --config configs/hybrid_3b.yaml \
    --train_data data/3b_train.bin \
    --batch_size 2 \
    --lr 1e-4 \
    --warmup_steps 5 \
    --grad_accum 1 \
    --max_steps 10 \
    --checkpoint_dir /tmp/hybrid_test_ckpt \
    --use_fp8
# 예상: 10 steps 완료, 체크포인트 저장, startup 배너에 hybrid 정보 출력
```

### 3. 체크포인트 Resume 테스트

```bash
# Step 2 체크포인트에서 resume
torchrun --nproc_per_node=8 --master_port=29501 train/pretrain.py \
    --config configs/hybrid_3b.yaml \
    --train_data data/3b_train.bin \
    --batch_size 2 \
    --lr 1e-4 \
    --warmup_steps 5 \
    --grad_accum 1 \
    --max_steps 20 \
    --checkpoint_dir /tmp/hybrid_test_ckpt \
    --resume /tmp/hybrid_test_ckpt/checkpoint-0000010 \
    --use_fp8
# 예상: step 10에서 이어서 step 20까지 학습
```

---

## 수정하지 않는 것들 (의도적 제외)

- **sequential scan 성능**: Python for-loop는 느리지만 구조 변경이 큼. 별도 태스크로 chunked SSD 구현
- **FP8 + Mamba 혼합**: 현재 설계(Mamba=bf16, Attention=FP8)가 올바름. te.fp8_autocast는 te 모듈만 영향
- **DDP 설정**: find_unused_parameters=False, gradient_as_bucket_view=True 모두 정상
- **pure Transformer 모드**: use_hybrid=False면 기존 동작 유지 (하위 호환)

---

## 수정 대상 파일 요약

| 파일 | Step | 변경 내용 |
|------|------|----------|
| `model/config.py` | 1a | `mamba_d_ffn` 필드 + `__post_init__` + `to_dict()` |
| `model/mamba_block.py` | 1b, 5, 6 | SwiGLU import, FFN sublayer, NaN 감지, shape 검증 |
| `model/transformer.py` | 1c | Mamba2Block 생성자에 d_ffn/bias 전달, `_init_weights` 수정 |
| `configs/hybrid_3b.yaml` | 1d, 2 | `mamba_d_ffn: 4608`, `mamba_n_groups: 8` |
| `train/pretrain.py` | 3 | Hybrid startup 로그 |
| `train/utils.py` | 4 | `load_checkpoint()` 아키텍처 검증 |

---

## 실행 지시 (다음 세션용)

이 문서를 참조하여 다음 명령을 내리면 됩니다:

> "이 문서(hashed-drifting-harp.md)의 Step 1~6을 순서대로 실행해 줘.
> Step 1+2는 병렬로, Step 3~6은 독립적으로 진행.
> 각 Step 완료 후 해당 검증을 실행하고,
> 전체 완료 후 최종 검증 절차 3단계를 모두 실행해 줘."

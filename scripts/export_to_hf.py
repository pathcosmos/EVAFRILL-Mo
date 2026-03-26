"""
Export EVAFRILL-Mo checkpoint to HuggingFace Hub format.

Converts a native EVAFRILL-Mo checkpoint (model.pt + config.yaml) into the
standard HuggingFace layout:
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  README.md  (model card)

Usage:
    python scripts/export_to_hf.py
    python scripts/export_to_hf.py --checkpoint checkpoints/3b_dpo/checkpoint-slerp
    python scripts/export_to_hf.py --output hf_export/ --push
    python scripts/export_to_hf.py --repo-id pathcosmos/EVAFRILL-Mo-3B --push
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import model.config if needed
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------
MODEL_CARD = """\
---
language:
- ko
- en
license: mit
tags:
- mamba2
- hybrid
- korean
- causal-lm
pipeline_tag: text-generation
---

# EVAFRILL-Mo-3B

EVAFRILL-Mo-3B is a 2.94B-parameter **hybrid Mamba-2 + Transformer** language
model optimised for Korean, trained from scratch on 55 billion tokens of
Korean-dominant multilingual text.

> **EVAFRILL-Mo** stands for *Efficient Variably-Architected Fusion of
> Recurrent and Integrated Linear Layers for Language Model-based Output* — a
> custom architecture inspired by [Nemotron-H](https://arxiv.org/abs/2501.14587)
> that replaces most self-attention layers with Mamba-2 SSM blocks, achieving
> linear-time inference without sacrificing generation quality.

---

## Architecture

| Property | Value |
|---|---|
| Total parameters | ~2.94 B |
| Layers | 26 (24 × Mamba-2 + 2 × Attention) |
| Hidden size | 3072 |
| Attention heads | 24 (GQA, 8 KV heads) |
| FFN size | 9 216 |
| Mamba-2 state dim | 128 |
| Mamba-2 head dim | 64 |
| Vocab size | 64 000 |
| Max sequence length | 4 096 |
| RoPE theta | 500 000 |

The layer pattern places attention blocks at positions 9 and 18 (zero-indexed),
mirroring the Nemotron-H 8B dense design scaled to 3B parameters. All other
layers use Mamba-2 with SwiGLU FFN (mamba_d_ffn = 4 608). Attention layers use
full SwiGLU FFN (d_ffn = 9 216).

---

## Training

### Pretraining
- **Tokens**: 55 B (319 772 steps, effective batch ≈ 172 K tokens)
- **Hardware**: 8× NVIDIA B200 (183 GB each), ~62 hours
- **Optimizer**: AdamW, lr=2e-4, cosine decay, warmup 2 000 steps
- **Precision**: FP8 (TransformerEngine MXFP8) + BF16 embedding
- **Data**: Korean web corpus, Wikipedia, books, code (Korean-dominant)

### Supervised Fine-Tuning (SFT)
- **Steps**: 65 000 (≈ 1 epoch on 2.44M instruction samples)
- **Effective batch**: 56 (2 per GPU × 7 GPU × 4 grad_accum)
- **LR**: 1e-5 (pretrain/30, catastrophic-forgetting guard)
- **NEFTune alpha**: 5.0 (repetition degeneracy mitigation)
- **Data**: Combined Korean instruction set (filtered, 2.44M samples)

### Direct Preference Optimisation (DPO)
- **Rounds**: 2-round DPO (Nemotron-H style)
  - Round 1: 3 000 steps, beta=0.1, lr=5e-7, LoRA rank=32
  - Round 2: 2 000 steps, beta=0.05, lr=1e-7, LoRA rank=32
- **Hardware**: 1× NVIDIA H100 MIG 3g.40gb (~42 GB VRAM)
- **Method**: Native LoRA DPO (no TRL dependency)

### SLERP Merge
The final checkpoint is produced by **spherical linear interpolation (SLERP)**
between the SFT-v2 and DPO-round-2 checkpoints (ratio 0.5), combining the
instruction-following strengths of both stages.

---

## Evaluation (SLERP checkpoint, lm-eval-harness)

| Benchmark | Metric | Score |
|---|---|---|
| HellaSwag | acc_norm | 0.42 |
| ARC-Challenge | acc_norm | 0.22 |
| ARC-Easy | acc_norm | 0.28 |
| Belebele (kor_Hang) | acc | 0.30 |
| Global-MMLU-ko (full) | acc | 0.233 |
|  — Humanities | acc | 0.242 |
|  — STEM | acc | 0.237 |
|  — Social Sciences | acc | 0.221 |
|  — Other | acc | 0.229 |

*Evaluated on 100-sample subsets per task. Numbers reflect the final
SLERP-merged checkpoint.*

---

## Usage

```python
# Requires: pip install transformers tokenizers safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "pathcosmos/EVAFRILL-Mo-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Chat-style prompt
prompt = "<|user|>\\n안녕하세요! 자기소개를 해 주세요.\\n<|assistant|>\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

---

## Limitations

- This model is an **experimental research checkpoint**, not a production system.
- Korean is the dominant language; English and other languages are secondary.
- The custom architecture (`evafrill-mo`) requires either
  (a) the original project code for full inference, or
  (b) a compatible HuggingFace integration that understands Mamba-2 hybrid layers.
  The exported `model.safetensors` preserves the native weight layout.
- Benchmark numbers were evaluated on small (100-sample) subsets and should be
  treated as rough estimates.

---

## Citation

```bibtex
@misc{evafrill-mo-3b-2026,
  title  = {EVAFRILL-Mo-3B: A Hybrid Mamba-2 + Transformer LLM for Korean},
  author = {pathcosmos},
  year   = {2026},
  url    = {https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B},
}
```

---

## Acknowledgment / 감사의 글

이 프로젝트는 **과학기술정보통신부**의 **「첨단 GPU 활용 지원 사업」** (과학기술정보통신부 공고 제2025-1068호)을 통해 제공된 GPU 컴퓨팅 자원을 활용하여 수행되었습니다.

> **국가 AI컴퓨팅자원 지원포털**: [https://aiinfrahub.kr](https://aiinfrahub.kr)
>
> - 주관: 과학기술정보통신부 (MSIT), 정보통신산업진흥원 (NIPA)
> - 운영: 한국정보통신진흥협회 (KAIT)

대한민국 정부의 AI 인프라 지원 사업 덕분에 7× NVIDIA B200 GPU 환경에서 한국어 3B 하이브리드 Mamba-Transformer 모델을 처음부터 학습할 수 있었습니다. 국가 차원의 AI 컴퓨팅 자원 지원에 깊이 감사드립니다.

This project was conducted using GPU computing resources provided through the **"Advanced GPU Utilization Support Program"** (MSIT Notice No. 2025-1068) by the **Ministry of Science and ICT (MSIT)** of the Republic of Korea.

> **National AI Computing Resource Support Portal**: [https://aiinfrahub.kr](https://aiinfrahub.kr)
>
> - Organized by: Ministry of Science and ICT (MSIT), National IT Industry Promotion Agency (NIPA)
> - Operated by: Korea Association of Information & Telecommunication (KAIT)

We are deeply grateful for the national-level AI computing infrastructure support from the Korean government, which made it possible to train a Korean 3B hybrid Mamba-Transformer model from scratch on 7× NVIDIA B200 GPUs.

---

## License

[MIT](LICENSE)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state_dict(checkpoint: Path) -> dict:
    """Load model.pt as a pure state_dict (weights_only=True for safety)."""
    model_pt = checkpoint / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found: {model_pt}")
    print(f"  Loading {model_pt} ...")
    state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
    print(f"  {len(state_dict)} tensors loaded")
    return state_dict


def save_safetensors(state_dict: dict, output_dir: Path) -> None:
    """Write state_dict to model.safetensors; fallback to pytorch_model.bin."""
    try:
        from safetensors.torch import save_file
        out = output_dir / "model.safetensors"
        print(f"  Saving {out} ...")
        save_file(state_dict, out)
        size_gb = out.stat().st_size / 1e9
        print(f"  Saved model.safetensors ({size_gb:.2f} GB)")
    except ImportError:
        print("  [WARN] safetensors not installed; falling back to pytorch_model.bin")
        print("         Install with: pip install safetensors")
        out = output_dir / "pytorch_model.bin"
        torch.save(state_dict, out)
        size_gb = out.stat().st_size / 1e9
        print(f"  Saved pytorch_model.bin ({size_gb:.2f} GB)")


def load_and_write_config(checkpoint: Path, output_dir: Path) -> dict:
    """
    Load config.yaml, inject HF compatibility fields, write config.json.
    Returns the raw config dict.
    """
    config_yaml = checkpoint / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_yaml}")

    with open(config_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Flatten nested YAML (e.g. config["model"] -> top-level) if present
    if "model" in config and isinstance(config["model"], dict):
        model_cfg = config["model"]
    else:
        model_cfg = config

    # Add HuggingFace-required fields
    model_cfg["model_type"] = "evafrill-mo"
    model_cfg["architectures"] = ["EvafrillMoForCausalLM"]

    # Add torch_dtype hint for from_pretrained
    model_cfg.setdefault("torch_dtype", "bfloat16")

    out = output_dir / "config.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2, ensure_ascii=False)
    print(f"  Saved config.json")
    return model_cfg


def copy_tokenizer(tokenizer_src: Path, output_dir: Path) -> None:
    """Copy tokenizer.json and write tokenizer_config.json."""
    if not tokenizer_src.exists():
        print(f"  [WARN] Tokenizer not found at {tokenizer_src}. Copy manually.")
        return

    dst = output_dir / "tokenizer.json"
    shutil.copy(tokenizer_src, dst)
    print(f"  Copied tokenizer.json from {tokenizer_src}")

    # Minimal tokenizer_config.json for HuggingFace compatibility.
    # EOS/PAD/UNK IDs match the standard SentencePiece convention used in training.
    tok_cfg = {
        "model_type": "evafrill-mo",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "chat_template": "<|user|>\n{{ message }}\n<|assistant|>\n",
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    print("  Saved tokenizer_config.json")


def write_generation_config(output_dir: Path) -> None:
    """Write generation_config.json with sensible defaults for chat use."""
    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, ensure_ascii=False)
    print("  Saved generation_config.json")


def write_model_card(output_dir: Path) -> None:
    """Write README.md (HuggingFace model card)."""
    out = output_dir / "README.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD)
    print("  Saved README.md (model card)")


def push_to_hub(output_dir: Path, repo_id: str) -> None:
    """Upload the export directory to a HuggingFace Hub repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("        Install with: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    print(f"  Creating repo '{repo_id}' (exist_ok=True) ...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"  Uploading {output_dir} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Export EVAFRILL-Mo-3B (SLERP checkpoint)",
    )
    print(f"  Pushed to: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EVAFRILL-Mo checkpoint to HuggingFace format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/3b_dpo/checkpoint-slerp"),
        help="Path to checkpoint directory (must contain model.pt and config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hf_export"),
        help="Output directory for the HuggingFace-format files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="pathcosmos/EVAFRILL-Mo-3B",
        help="HuggingFace Hub repository ID (user/model-name).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/korean_sp/tokenizer.json"),
        help="Path to the source tokenizer.json.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the exported files to the HuggingFace Hub after export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint: Path = args.checkpoint
    output_dir: Path = args.output_dir
    repo_id: str = args.repo_id
    tokenizer_src: Path = args.tokenizer

    # Resolve relative paths from project root for ergonomic CLI use
    if not checkpoint.is_absolute():
        checkpoint = _PROJECT_ROOT / checkpoint
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    if not tokenizer_src.is_absolute():
        tokenizer_src = _PROJECT_ROOT / tokenizer_src

    print("=" * 60)
    print("EVAFRILL-Mo -> HuggingFace Export")
    print("=" * 60)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Output dir : {output_dir}")
    print(f"  Repo ID    : {repo_id}")
    print(f"  Tokenizer  : {tokenizer_src}")
    print(f"  Push       : {args.push}")
    print()

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Weights ---
    print("[1/5] Converting weights to safetensors ...")
    state_dict = load_state_dict(checkpoint)
    save_safetensors(state_dict, output_dir)
    # Free memory immediately — the state dict can be several GB
    del state_dict
    print()

    # --- Step 2: Config ---
    print("[2/5] Writing config.json ...")
    load_and_write_config(checkpoint, output_dir)
    print()

    # --- Step 3: Tokenizer ---
    print("[3/5] Copying tokenizer ...")
    copy_tokenizer(tokenizer_src, output_dir)
    print()

    # --- Step 4: Generation config ---
    print("[4/5] Writing generation_config.json ...")
    write_generation_config(output_dir)
    print()

    # --- Step 5: Model card ---
    print("[5/5] Writing model card (README.md) ...")
    write_model_card(output_dir)
    print()

    print("Export complete.")
    print(f"  Files in {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        size = p.stat().st_size
        if size >= 1_000_000_000:
            size_str = f"{size / 1e9:.2f} GB"
        elif size >= 1_000_000:
            size_str = f"{size / 1e6:.1f} MB"
        else:
            size_str = f"{size / 1e3:.1f} KB"
        print(f"    {p.name:<30} {size_str:>10}")

    # --- Optional: Push to Hub ---
    if args.push:
        print()
        print("[Push] Uploading to HuggingFace Hub ...")
        push_to_hub(output_dir, repo_id)

    print()
    print("Done.")


if __name__ == "__main__":
    main()

"""
Convert custom LLM checkpoint to HuggingFace LlamaForCausalLM format.

Usage:
    python scripts/convert_to_hf.py \\
        --checkpoint checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \\
        --output outputs/hf \\
        [--tokenizer tokenizer/korean_sp/tokenizer.json]

Outputs (in --output directory):
    config.json          — LlamaConfig
    model.safetensors    — converted weights
    tokenizer.json       — tokenizer (copied)
    tokenizer_config.json
    generation_config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.config import LMConfig


def remap_weights(
    src_state_dict: dict,
    config: LMConfig,
) -> dict:
    """
    Remap custom LLM weight names to HuggingFace LlamaForCausalLM names.

    Handles both FP8 (te.LayerNormMLP / te.Linear) and BF16 (SwiGLU / nn.Linear)
    checkpoints transparently.
    """
    dst = {}
    is_fp8 = config.use_fp8

    # --- Token embedding ---
    dst["model.embed_tokens.weight"] = src_state_dict["embedding.weight"].float()

    for i in range(config.n_layers):
        pfx = f"layers.{i}"
        hpfx = f"model.layers.{i}"

        # Attention norm (always RMSNorm)
        dst[f"{hpfx}.input_layernorm.weight"] = (
            src_state_dict[f"{pfx}.attn_norm.weight"].float()
        )

        # Attention projections (te.Linear or nn.Linear — same key names)
        for src_name, dst_name in [
            ("q_proj", "self_attn.q_proj"),
            ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"),
            ("out_proj", "self_attn.o_proj"),
        ]:
            w_key = f"{pfx}.attn.{src_name}.weight"
            if w_key in src_state_dict:
                dst[f"{hpfx}.{dst_name}.weight"] = src_state_dict[w_key].float()

        # FFN — FP8 (te.LayerNormMLP) vs BF16 (SwiGLU)
        if is_fp8 and f"{pfx}.ffn.layer_norm_weight" in src_state_dict:
            # te.LayerNormMLP: RMSNorm is fused inside
            dst[f"{hpfx}.post_attention_layernorm.weight"] = (
                src_state_dict[f"{pfx}.ffn.layer_norm_weight"].float()
            )
            # fc1_weight: [2*d_ffn, d_model] — gate and up are concatenated
            fc1 = src_state_dict[f"{pfx}.ffn.fc1_weight"].float()
            half = fc1.shape[0] // 2
            dst[f"{hpfx}.mlp.gate_proj.weight"] = fc1[:half]
            dst[f"{hpfx}.mlp.up_proj.weight"] = fc1[half:]
            # fc2_weight: [d_model, d_ffn]
            dst[f"{hpfx}.mlp.down_proj.weight"] = (
                src_state_dict[f"{pfx}.ffn.fc2_weight"].float()
            )
        else:
            # Standard SwiGLU (BF16 checkpoint)
            dst[f"{hpfx}.post_attention_layernorm.weight"] = (
                src_state_dict[f"{pfx}.ffn_norm.weight"].float()
            )
            dst[f"{hpfx}.mlp.gate_proj.weight"] = (
                src_state_dict[f"{pfx}.ffn.gate_proj.weight"].float()
            )
            dst[f"{hpfx}.mlp.up_proj.weight"] = (
                src_state_dict[f"{pfx}.ffn.up_proj.weight"].float()
            )
            dst[f"{hpfx}.mlp.down_proj.weight"] = (
                src_state_dict[f"{pfx}.ffn.down_proj.weight"].float()
            )

    # --- Final norm and LM head ---
    dst["model.norm.weight"] = src_state_dict["norm.weight"].float()
    # Weight tying: embedding.weight == lm_head.weight in our model.
    # HF LlamaForCausalLM expects lm_head.weight explicitly.
    dst["lm_head.weight"] = src_state_dict["embedding.weight"].float().clone()

    return dst


def build_llama_config(config: LMConfig) -> dict:
    """Map LMConfig fields to HuggingFace LlamaConfig dict."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": config.d_model,
        "intermediate_size": config.d_ffn,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_seq_len,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "vocab_size": config.vocab_size,
        "rope_theta": config.rope_theta,
        "rope_scaling": None,
        "attention_bias": config.bias,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "transformers_version": "4.40.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert custom LLM checkpoint to HuggingFace LlamaForCausalLM format."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to checkpoint directory (must contain model.pt + config.yaml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for HF-format files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/korean_sp/tokenizer.json"),
        help="Path to tokenizer.json (default: tokenizer/korean_sp/tokenizer.json).",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    out_path = args.output

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {out_path}")

    # Load config
    config = LMConfig.from_yaml(ckpt_path / "config.yaml")
    print(f"Model      : d_model={config.d_model}, n_layers={config.n_layers}, "
          f"vocab_size={config.vocab_size}, use_fp8={config.use_fp8}")

    # Load weights
    print("Loading model.pt ...")
    state_dict = torch.load(
        ckpt_path / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    print(f"  Source keys: {len(state_dict)}")

    # Remap
    print("Remapping weight names ...")
    hf_state_dict = remap_weights(state_dict, config)
    print(f"  Destination keys: {len(hf_state_dict)}")

    # Save safetensors
    print("Saving model.safetensors ...")
    try:
        from safetensors.torch import save_file
        save_file(hf_state_dict, out_path / "model.safetensors")
    except ImportError:
        print("  [WARN] safetensors not installed; falling back to pytorch_model.bin")
        torch.save(hf_state_dict, out_path / "pytorch_model.bin")

    # Save config.json
    llama_cfg = build_llama_config(config)
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(llama_cfg, f, indent=2, ensure_ascii=False)
    print("Saved config.json")

    # Save generation_config.json
    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "do_sample": True,
    }
    with open(out_path / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, ensure_ascii=False)

    # Copy tokenizer
    tok_src = args.tokenizer
    if tok_src.exists():
        shutil.copy(tok_src, out_path / "tokenizer.json")
        # Minimal tokenizer_config.json for HF compatibility
        tok_cfg = {
            "model_type": "llama",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "clean_up_tokenization_spaces": False,
        }
        with open(out_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
        print(f"Copied tokenizer: {tok_src} -> {out_path / 'tokenizer.json'}")
    else:
        print(f"[WARN] Tokenizer not found at {tok_src}. Copy manually.")

    print(f"\nDone! HF model saved to: {out_path}")
    print("Verify: ls -lh", out_path)


if __name__ == "__main__":
    main()

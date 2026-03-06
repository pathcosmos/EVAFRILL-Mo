"""
Text generation (inference) script with temperature + top-p / top-k sampling.

Usage:
    python eval/generate.py \
        --checkpoint checkpoints/checkpoint-0100000 \
        --prompt "Once upon a time" \
        --max_new_tokens 200 \
        --temperature 0.8 \
        --top_p 0.9 \
        --top_k 50 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Generator

import torch
import torch.nn.functional as F
from model.transformer import LLM
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_p_filtering(
    logits: torch.Tensor,
    top_p: float = 0.9,
    top_k: int = 0,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Apply top-k and / or top-p (nucleus) filtering to a logits tensor.

    Args:
        logits:       1-D or 2-D tensor of raw (un-normalised) logits.
                      Shape: [vocab_size] or [batch, vocab_size].
        top_k:        Keep only the top-k tokens (0 = disabled).
        top_p:        Keep the smallest set of tokens whose cumulative
                      probability is >= top_p (1.0 = disabled).
        filter_value: Value assigned to filtered positions (−inf by default).

    Returns:
        Filtered logits with the same shape as input.
    """
    # Work on a 2-D tensor [batch, vocab].
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # --- Top-K ---
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        # Find the k-th largest value for each row.
        kth_values = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_values, filter_value)

    # --- Top-P (nucleus) ---
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens once cumulative probability exceeds top_p.
        # Shift right by one so that the token that *pushes* the cumulative
        # probability over the threshold is kept.
        sorted_indices_to_remove = cumulative_probs - F.softmax(
            sorted_logits, dim=-1
        ) >= top_p
        sorted_logits = sorted_logits.masked_fill(
            sorted_indices_to_remove, filter_value
        )
        # Scatter filtered sorted_logits back to the original ordering.
        logits = torch.zeros_like(logits).scatter_(
            -1, sorted_indices, sorted_logits
        )

    if squeeze_output:
        logits = logits.squeeze(0)

    return logits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cuda:0",
) -> Generator[str, None, None]:
    """
    Auto-regressive token generation with streaming output.

    Yields decoded string fragments (one token at a time) so callers can
    stream output to stdout without waiting for the full sequence.

    Args:
        model:          A causal LM whose forward pass returns logits
                        (last dim = vocab_size).
        tokenizer:      Matching tokenizer; must expose encode / decode.
        prompt:         Text prompt to condition on.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Softmax temperature (1.0 = neutral, <1 = sharper).
        top_p:          Nucleus sampling probability threshold.
        top_k:          Top-K token candidates (0 = disabled).
        device:         Torch device string.

    Yields:
        Decoded string for each newly generated token.
    """
    model.eval()

    # Encode prompt.
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    eos_token_id: int | None = tokenizer.token_to_id("</s>")

    # Incremental generation.
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        # Full-sequence forward (no KV cache) — each step re-runs all tokens.
        logits_all, _ = model(generated_ids)
        logits: torch.Tensor = logits_all[:, -1, :]  # [1, vocab]

        # --- Temperature scaling ---
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        # --- Top-k / Top-p filtering ---
        logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)

        # --- Sample ---
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # Decode and yield the new token.
        token_str: str = tokenizer.decode([next_token_id.item()])
        yield token_str

        # Stop at EOS.
        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    checkpoint_dir: str, device: str
) -> tuple[torch.nn.Module, Tokenizer]:
    """
    Load a model and tokenizer from a checkpoint directory.

    Expects:
      - <checkpoint_dir>/model.pt     — model weights
      - <checkpoint_dir>/config.yaml  — LMConfig
      - <checkpoint_dir>/tokenizer.json — HuggingFace tokenizers format
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")
    model = LLM.from_pretrained(str(ckpt_path)).to(device=device, dtype=torch.float16)
    model.eval()

    tokenizer_path = ckpt_path / "tokenizer.json"
    if not tokenizer_path.exists():
        # Fallback: try project-level tokenizer
        tokenizer_path = Path("tokenizer/korean_sp/tokenizer.json")
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return model, tokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained LLM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Input prompt text.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 200).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold (default: 0.9).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k token candidates; 0 disables top-k (default: 50).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to run inference on (default: cuda:0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")
    print(f"\nPrompt: {args.prompt!r}")
    print("-" * 60)
    print(args.prompt, end="", flush=True)

    generated_tokens = 0
    for token_str in generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    ):
        print(token_str, end="", flush=True)
        generated_tokens += 1

    print()  # newline after generation
    print("-" * 60)
    print(f"Generated {generated_tokens} token(s).")


if __name__ == "__main__":
    main()

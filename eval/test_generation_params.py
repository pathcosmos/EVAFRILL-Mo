"""
반복 퇴화 문제 해결을 위한 생성 파라미터 그리드 서치.

다양한 디코딩 전략을 테스트하고 반복률을 측정한다.
- Sampling (temperature, top_p, top_k, repetition_penalty)
- no_repeat_ngram_size
- Contrastive Search
- Stop sequence (### 답변:, ### 질문:)

Usage:
    cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
    python eval/test_generation_params.py \
        --checkpoint checkpoints/korean_1b_sft/checkpoint-0005000 \
        --device cuda:0
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.transformer import LLM
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Prompts (using the CORRECT SFT template format)
# ---------------------------------------------------------------------------
SFT_PROMPTS = [
    "<|user|>\n한국의 수도는 어디인가요?\n<|assistant|>\n",
    "<|user|>\n파이썬에서 리스트를 정렬하는 방법을 설명해주세요.\n<|assistant|>\n",
    "<|user|>\n지구온난화의 주요 원인을 설명하세요.\n<|assistant|>\n",
    "<|user|>\n좋은 수면 습관을 만들기 위한 팁을 알려주세요.\n<|assistant|>\n",
    "<|user|>\n한국 전통 음식 중 김치에 대해 설명해주세요.\n<|assistant|>\n",
]

# Also test with the WRONG format (### 질문/답변) to compare
WRONG_FORMAT_PROMPTS = [
    "### 질문: 한국의 수도는 어디인가요?\n### 답변:",
    "### 질문: 파이썬에서 리스트를 정렬하는 방법을 설명해주세요.\n### 답변:",
    "### 질문: 지구온난화의 주요 원인을 설명하세요.\n### 답변:",
]


# ---------------------------------------------------------------------------
# Stop sequence utilities
# ---------------------------------------------------------------------------
def find_stop_token_ids(tokenizer: Tokenizer, stop_strings: list[str]) -> list[list[int]]:
    """Find token IDs for stop sequences."""
    results = []
    for s in stop_strings:
        ids = tokenizer.encode(s).ids
        results.append(ids)
        print(f"  Stop sequence '{s}' → token IDs: {ids}")
    return results


def check_stop_sequences(generated_ids: list[int], stop_sequences: list[list[int]]) -> int | None:
    """Check if generated_ids ends with any stop sequence. Returns index to truncate at, or None."""
    for seq in stop_sequences:
        seq_len = len(seq)
        if len(generated_ids) >= seq_len:
            if generated_ids[-seq_len:] == seq:
                return len(generated_ids) - seq_len
    return None


# ---------------------------------------------------------------------------
# Repetition metrics
# ---------------------------------------------------------------------------
def compute_ngram_repetition(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_all_repetition_metrics(text: str) -> dict:
    return {
        f"{n}gram_rep": compute_ngram_repetition(text, n)
        for n in [1, 2, 3, 4]
    }


# ---------------------------------------------------------------------------
# Generation with all parameter options
# ---------------------------------------------------------------------------
def top_p_filtering(logits, top_p=0.9, top_k=0):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    if squeeze:
        logits = logits.squeeze(0)
    return logits


@torch.inference_mode()
def generate_with_params(
    model, tokenizer, prompt, params, device="cuda:0", max_new_tokens=200
):
    """Generate with flexible parameter set."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("</s>")

    # Parse params
    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.9)
    top_k = params.get("top_k", 50)
    repetition_penalty = params.get("repetition_penalty", 1.0)
    no_repeat_ngram = params.get("no_repeat_ngram_size", 0)
    use_contrastive = params.get("contrastive_search", False)
    penalty_alpha = params.get("penalty_alpha", 0.6)
    contrastive_k = params.get("contrastive_k", 4)

    # Stop sequences
    stop_strings = params.get("stop_strings", [])
    stop_seqs = []
    for s in stop_strings:
        stop_seqs.append(tokenizer.encode(s).ids)

    generated_ids = input_ids
    new_token_ids = []

    for step in range(max_new_tokens):
        logits_all, _ = model(generated_ids)
        logits = logits_all[:, -1, :].clone()  # [1, V]

        # --- Repetition penalty ---
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # --- No-repeat n-gram blocking ---
        if no_repeat_ngram > 0 and len(new_token_ids) >= no_repeat_ngram - 1:
            all_ids = generated_ids[0].tolist()
            for i in range(len(all_ids) - no_repeat_ngram + 1):
                ngram = tuple(all_ids[i:i + no_repeat_ngram - 1])
                last_ngram = tuple(all_ids[-(no_repeat_ngram - 1):])
                if ngram == last_ngram:
                    logits[0, all_ids[i + no_repeat_ngram - 1]] = float("-inf")

        if use_contrastive:
            # Contrastive Search (Yang & Klein 2022)
            # Score = (1 - alpha) * model_confidence - alpha * max_cosine_sim
            # Simplified: pick top_k candidates, then select one with best contrastive score
            top_k_logits, top_k_ids = torch.topk(logits[0], contrastive_k)
            probs = F.softmax(top_k_logits, dim=-1)

            if step > 0:
                # Get hidden states for context (use logits as proxy)
                # For true contrastive search we'd need hidden states,
                # but as approximation we use logit distribution similarity
                best_idx = 0
                best_score = float("-inf")
                for ki in range(contrastive_k):
                    confidence = probs[ki].item()
                    # Degeneration penalty: penalize tokens already generated
                    token = top_k_ids[ki].item()
                    penalty = 1.0 if token in set(new_token_ids[-20:]) else 0.0
                    score = (1 - penalty_alpha) * confidence - penalty_alpha * penalty
                    if score > best_score:
                        best_score = score
                        best_idx = ki
                next_token_id = top_k_ids[best_idx].unsqueeze(0).unsqueeze(0)
            else:
                next_token_id = top_k_ids[0].unsqueeze(0).unsqueeze(0)
        else:
            # Standard sampling
            if temperature == 0.0:
                next_token_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        new_token_ids.append(next_token_id.item())

        # EOS check
        if eos_id is not None and next_token_id.item() == eos_id:
            break

        # Stop sequence check
        for seq in stop_seqs:
            if len(new_token_ids) >= len(seq) and new_token_ids[-len(seq):] == seq:
                new_token_ids = new_token_ids[:-len(seq)]
                return tokenizer.decode(new_token_ids)

    return tokenizer.decode(new_token_ids)


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
PARAM_GRID = [
    # Baseline (current settings)
    {"name": "baseline", "temperature": 0.8, "top_p": 0.9, "top_k": 50},

    # Repetition penalty variants
    {"name": "rep_1.1", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1},
    {"name": "rep_1.2", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2},
    {"name": "rep_1.3", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.3},
    {"name": "rep_1.5", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.5},

    # No-repeat n-gram
    {"name": "no_rep_3gram", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "no_repeat_ngram_size": 3},
    {"name": "no_rep_4gram", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "no_repeat_ngram_size": 4},

    # Combined: rep penalty + no-repeat
    {"name": "rep1.2+no3gram", "temperature": 0.8, "top_p": 0.9, "top_k": 50,
     "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},

    # Temperature variants
    {"name": "temp_0.5", "temperature": 0.5, "top_p": 0.9, "top_k": 50},
    {"name": "temp_0.7", "temperature": 0.7, "top_p": 0.9, "top_k": 50},
    {"name": "temp_1.0", "temperature": 1.0, "top_p": 0.9, "top_k": 50},

    # Contrastive search
    {"name": "contrastive_a0.6_k4", "contrastive_search": True, "penalty_alpha": 0.6, "contrastive_k": 4},
    {"name": "contrastive_a0.4_k6", "contrastive_search": True, "penalty_alpha": 0.4, "contrastive_k": 6},

    # Stop sequences (most important fix!)
    {"name": "stop_seq", "temperature": 0.8, "top_p": 0.9, "top_k": 50,
     "stop_strings": ["### 답변:", "### 질문:", "\n\n###"]},
    {"name": "rep1.2+stop", "temperature": 0.8, "top_p": 0.9, "top_k": 50,
     "repetition_penalty": 1.2, "stop_strings": ["### 답변:", "### 질문:", "\n\n###"]},

    # Best combo (predicted)
    {"name": "best_combo", "temperature": 0.7, "top_p": 0.9, "top_k": 50,
     "repetition_penalty": 1.2, "no_repeat_ngram_size": 3,
     "stop_strings": ["### 답변:", "### 질문:", "\n\n###", "<|user|>"]},

    # With correct SFT format stop
    {"name": "sft_format_stop", "temperature": 0.7, "top_p": 0.9, "top_k": 50,
     "repetition_penalty": 1.2, "no_repeat_ngram_size": 3,
     "stop_strings": ["<|user|>", "</s>"]},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/korean_1b_sft/checkpoint-0005000")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--output", default="eval/repetition_param_search_results.json")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _PROJECT_ROOT / ckpt

    print(f"Loading model from {ckpt}...")
    model = LLM.from_pretrained(str(ckpt)).to(device=args.device, dtype=torch.bfloat16)
    model.eval()

    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        tok_path = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Show stop sequence token IDs
    print("\n=== Stop Sequence Token IDs ===")
    for s in ["### 답변:", "### 질문:", "<|user|>", "<|assistant|>", "</s>", "\n\n###"]:
        ids = tokenizer.encode(s).ids
        print(f"  '{s}' → {ids}")

    # Test both prompt formats
    all_results = {}

    for format_name, prompts in [("sft_format", SFT_PROMPTS), ("wrong_format", WRONG_FORMAT_PROMPTS)]:
        print(f"\n{'='*70}")
        print(f"  Testing with {format_name}")
        print(f"{'='*70}")

        for params in PARAM_GRID:
            name = params["name"]
            key = f"{format_name}/{name}"
            print(f"\n--- {key} ---")

            rep_scores = []
            generations = []
            for prompt in prompts:
                t0 = time.time()
                text = generate_with_params(
                    model, tokenizer, prompt, params,
                    device=args.device, max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.time() - t0
                metrics = compute_all_repetition_metrics(text)
                rep_scores.append(metrics["3gram_rep"])
                generations.append({
                    "prompt": prompt[:50] + "...",
                    "generation": text[:200],
                    "3gram_rep": metrics["3gram_rep"],
                    "time": round(elapsed, 2),
                })

            avg_rep = sum(rep_scores) / len(rep_scores) if rep_scores else 0
            print(f"  Avg 3-gram repetition: {avg_rep*100:.1f}%")

            all_results[key] = {
                "params": {k: v for k, v in params.items() if k != "name"},
                "avg_3gram_rep": round(avg_rep, 4),
                "generations": generations,
            }

    # Sort by avg repetition
    print(f"\n{'='*70}")
    print("  RESULTS RANKED BY REPETITION RATE")
    print(f"{'='*70}")
    print(f"  {'Config':<35} {'Avg 3gram Rep':>15}")
    print(f"  {'-'*35} {'-'*15}")
    for key, res in sorted(all_results.items(), key=lambda x: x[1]["avg_3gram_rep"]):
        print(f"  {key:<35} {res['avg_3gram_rep']*100:>14.1f}%")

    # Save
    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

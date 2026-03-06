"""
3B BASE 모델 생성 품질 + 반복률 종합 분석 스크립트.

Part 1: 10개 프롬프트 × 3 온도 → 자유 생성 텍스트 저장
Part 2: 파라미터 그리드 서치 → 반복률 분석 JSON 저장

BASE 모델용 completion-style 프롬프트 사용.

Usage:
    cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
    python eval/analyze_3b_generation.py \
        --checkpoint checkpoints/korean_3b_fp8_run1/checkpoint-0057000 \
        --device cuda:1
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

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    HAS_TE = True
except ImportError:
    te = None
    HAS_TE = False


def fp8_inference_context():
    """Return the appropriate inference context manager for FP8 models."""
    if HAS_TE:
        return te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling())
    import contextlib
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# BASE model completion-style prompts (10 prompts)
# ---------------------------------------------------------------------------
BASE_PROMPTS = [
    "대한민국의 수도는",
    "인공지능이란",
    "한국의 전통 음식 중에서",
    "지구 온난화의 주요 원인은",
    "프로그래밍을 배우려면",
    "조선시대에는",
    "물리학에서 에너지란",
    "한국어는 세계에서",
    "경제 성장을 위해서는",
    "우주 탐사의 역사를 보면",
]

# Subset for repetition grid (3 prompts to keep runtime reasonable)
GRID_PROMPTS = BASE_PROMPTS[:3]


# ---------------------------------------------------------------------------
# Sampling utilities
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


# ---------------------------------------------------------------------------
# Repetition metrics
# ---------------------------------------------------------------------------
def compute_ngram_repetition(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_all_repetition_metrics(text: str) -> dict:
    tokens = text.split()
    return {
        f"{n}gram_rep": compute_ngram_repetition(tokens, n)
        for n in [1, 2, 3, 4]
    }


# ---------------------------------------------------------------------------
# Generation (greedy or sampling, with optional rep penalty + no_repeat_ngram)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    device: str = "cuda:1",
) -> tuple[str, int, bool]:
    """
    Returns: (generated_text, num_new_tokens, hit_eos)
    MXFP8 requires sequence length divisible by 32; we right-pad before each
    forward pass but use the logit at the true last real position.
    """
    model.eval()
    raw_ids = tokenizer.encode(prompt).ids
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>") or 0

    # Keep an unpadded running sequence; pad only for the forward pass
    real_ids: list[int] = list(raw_ids)
    new_token_ids: list[int] = []
    hit_eos = False

    ctx = fp8_inference_context()
    with ctx:
        for _ in range(max_new_tokens):
            real_len = len(real_ids)
            # Pad to next multiple of 32 for MXFP8
            pad_to = ((real_len + 31) // 32) * 32
            padded = real_ids + [pad_id] * (pad_to - real_len)
            x = torch.tensor([padded], dtype=torch.long, device=device)

            logits_all, _ = model(x)
            # Logit at the last REAL token (index real_len - 1)
            logits = logits_all[:, real_len - 1, :].clone()  # [1, V]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(real_ids):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and real_len >= no_repeat_ngram_size:
                for i in range(real_len - no_repeat_ngram_size + 1):
                    ngram = tuple(real_ids[i:i + no_repeat_ngram_size - 1])
                    last_ngram = tuple(real_ids[-(no_repeat_ngram_size - 1):])
                    if ngram == last_ngram:
                        logits[0, real_ids[i + no_repeat_ngram_size - 1]] = float("-inf")

            # Decode strategy
            if temperature == 0.0:
                next_token_id = int(logits.argmax(dim=-1).item())
            else:
                logits = logits / max(temperature, 1e-8)
                logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)
                probs = F.softmax(logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            real_ids.append(next_token_id)
            new_token_ids.append(next_token_id)

            if eos_id is not None and next_token_id == eos_id:
                hit_eos = True
                break

    generated_text = tokenizer.decode(new_token_ids)
    return generated_text, len(new_token_ids), hit_eos


# ---------------------------------------------------------------------------
# Part 1: Free generation (10 prompts × 3 temps)
# ---------------------------------------------------------------------------
def run_free_generation(model, tokenizer, device, output_path: Path):
    temperatures = [0.0, 0.7, 1.0]
    results = []

    print("\n" + "=" * 70)
    print("  PART 1: FREE GENERATION (10 prompts × 3 temperatures)")
    print("=" * 70)

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        for prompt in BASE_PROMPTS:
            t0 = time.time()
            gen_text, n_tokens, hit_eos = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=256,
                temperature=temp,
                top_p=0.9,
                top_k=50,
                device=device,
            )
            elapsed = time.time() - t0
            metrics = compute_all_repetition_metrics(gen_text)

            entry = {
                "prompt": prompt,
                "temperature": temp,
                "generation": gen_text,
                "n_new_tokens": n_tokens,
                "hit_eos": hit_eos,
                "elapsed_sec": round(elapsed, 2),
                **metrics,
            }
            results.append(entry)

            # Print summary
            preview = gen_text[:120].replace("\n", "\\n")
            print(f"  [{temp}] {prompt!r}")
            print(f"    → {preview}...")
            print(f"    tokens={n_tokens}, eos={hit_eos}, 3gram_rep={metrics['3gram_rep']*100:.1f}%")

    # Save text version for easy reading
    txt_path = output_path.parent / "3b_generation_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"Temperature: {r['temperature']}\n")
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Generated ({r['n_new_tokens']} tokens, eos={r['hit_eos']}):\n")
            f.write(r["generation"] + "\n")
            f.write(f"3gram_rep={r['3gram_rep']*100:.1f}% | 4gram_rep={r['4gram_rep']*100:.1f}%\n")

    print(f"\n[Part 1] Saved text to: {txt_path}")
    return results


# ---------------------------------------------------------------------------
# Part 2: Repetition parameter grid search
# ---------------------------------------------------------------------------
PARAM_GRID = []

# Generate grid: temp × rep_penalty × no_repeat_ngram × top_p
for temp in [0.7, 0.9, 1.0]:
    for rep in [1.0, 1.1, 1.2, 1.3]:
        for ngram in [0, 3, 4]:
            for top_p in [0.9, 0.95]:
                name = f"t{temp}_r{rep}_ng{ngram}_tp{top_p}"
                PARAM_GRID.append({
                    "name": name,
                    "temperature": temp,
                    "repetition_penalty": rep,
                    "no_repeat_ngram_size": ngram,
                    "top_p": top_p,
                    "top_k": 50,
                })


def run_repetition_analysis(model, tokenizer, device, output_path: Path):
    print("\n" + "=" * 70)
    print(f"  PART 2: REPETITION ANALYSIS ({len(PARAM_GRID)} configs × {len(GRID_PROMPTS)} prompts)")
    print("=" * 70)

    all_results = {}
    eos_counts = {}

    for params in PARAM_GRID:
        name = params["name"]
        rep_scores = {n: [] for n in [1, 2, 3, 4]}
        eos_hits = 0
        token_counts = []
        generations = []

        for prompt in GRID_PROMPTS:
            gen_text, n_tokens, hit_eos = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=256,
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                repetition_penalty=params["repetition_penalty"],
                no_repeat_ngram_size=params["no_repeat_ngram_size"],
                device=device,
            )
            metrics = compute_all_repetition_metrics(gen_text)
            for n in [1, 2, 3, 4]:
                rep_scores[n].append(metrics[f"{n}gram_rep"])
            if hit_eos:
                eos_hits += 1
            token_counts.append(n_tokens)
            generations.append({
                "prompt": prompt,
                "generation": gen_text[:300],
                "n_tokens": n_tokens,
                "hit_eos": hit_eos,
                **{f"{n}gram_rep": round(metrics[f"{n}gram_rep"], 4) for n in [1, 2, 3, 4]},
            })

        n_prompts = len(GRID_PROMPTS)
        avg_reps = {f"avg_{n}gram_rep": round(sum(rep_scores[n]) / n_prompts, 4) for n in [1, 2, 3, 4]}
        eos_rate = eos_hits / n_prompts
        avg_tokens = sum(token_counts) / n_prompts

        all_results[name] = {
            "params": {k: v for k, v in params.items() if k != "name"},
            **avg_reps,
            "eos_rate": round(eos_rate, 4),
            "avg_tokens": round(avg_tokens, 1),
            "generations": generations,
        }

        print(f"  {name:<45} 3g={avg_reps['avg_3gram_rep']*100:.1f}% eos={eos_rate:.0%} tok={avg_tokens:.0f}")

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print ranked summary
    print(f"\n{'='*70}")
    print("  RANKED BY 3-GRAM REPETITION RATE")
    print(f"{'='*70}")
    print(f"  {'Config':<45} {'3gram':>7} {'eos':>6} {'tokens':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*6} {'-'*7}")
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["avg_3gram_rep"])
    for name, res in sorted_results[:20]:  # top 20
        print(
            f"  {name:<45} {res['avg_3gram_rep']*100:>6.1f}%"
            f" {res['eos_rate']:>5.0%} {res['avg_tokens']:>7.0f}"
        )

    print(f"\n[Part 2] Saved JSON to: {output_path}")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/korean_3b_fp8_run1/checkpoint-0057000",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output_dir", default="eval/outputs")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _PROJECT_ROOT / ckpt

    # Set default CUDA device BEFORE loading — required for TE MXFP8 device routing
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    torch.cuda.set_device(device_id)

    print(f"Loading model from: {ckpt}")
    model = LLM.from_pretrained(str(ckpt)).cuda(device_id).to(dtype=torch.bfloat16)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Params: {n_params / 1e9:.2f}B")

    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        tok_path = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: free generation
    free_gen_results = run_free_generation(
        model, tokenizer, args.device, output_dir / "3b_generation_results.txt"
    )

    # Save Part 1 JSON
    gen_json_path = output_dir / "3b_generation_results.json"
    with open(gen_json_path, "w", encoding="utf-8") as f:
        json.dump(free_gen_results, f, ensure_ascii=False, indent=2)
    print(f"[Part 1] JSON saved: {gen_json_path}")

    # Part 2: repetition analysis
    rep_json_path = output_dir / "3b_repetition_analysis.json"
    run_repetition_analysis(model, tokenizer, args.device, rep_json_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

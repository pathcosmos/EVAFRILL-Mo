"""
FRANKENSTALLM 3B — 6-GPU 병렬 종합 평가 스크립트.

GPU 배분:
  cuda:0  PPL — 3b_val.bin (145MB)
  cuda:1  PPL — korean_c4_val.bin (29MB)
  cuda:2  PPL — korean_namuwiki_val.bin (4.2MB) + korean_wiki_val.bin (1.1MB)
  cuda:3  Calibration (top-1/5/10 accuracy, entropy) on 3b_val.bin
  cuda:4  생성 품질 (10 프롬프트 × 3 온도)
  cuda:5  반복률 파라미터 그리드 탐색

Usage:
    cd /PROJECT/0325120031_A/ghong/taketimes/llm-bang
    python eval/parallel_eval_3b.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000")
TOKENIZER_PATH = str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json")
DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_DIR = _PROJECT_ROOT / "eval" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
STRIDE = 512
BATCH_SIZE = 32  # 183GB VRAM이므로 충분


# ===========================================================================
# Shared utilities
# ===========================================================================

class SlidingWindowDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = max(0, (len(tokens) - seq_len + stride - 1) // stride)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        actual_end = min(end, len(self.tokens))
        chunk_len = actual_end - start

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets = torch.full((self.seq_len,), fill_value=-100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        if chunk_len > 1:
            toks = torch.from_numpy(self.tokens[start:actual_end].astype(np.int64))
            input_ids[:chunk_len] = toks
            targets[:chunk_len - 1] = toks[1:]

        new_start = 0 if idx == 0 else self.stride
        if chunk_len > 1:
            for pos in range(new_start, chunk_len - 1):
                loss_mask[pos] = True

        return input_ids, targets, loss_mask


def load_model(device: str):
    from model.transformer import LLM
    model = LLM.from_pretrained(CHECKPOINT)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_PATH)


# ===========================================================================
# Task 1: Perplexity (runs on cuda:0, cuda:1, cuda:2)
# ===========================================================================

def eval_ppl(val_file: str, device: str) -> dict:
    """Compute sliding-window PPL for one val set."""
    torch.cuda.set_device(int(device.split(":")[-1]))
    data_path = DATA_DIR / val_file
    name = val_file.replace("_val.bin", "").replace(".bin", "")

    print(f"[PPL {device}] Loading model for {name}...")
    model = load_model(device)
    tokens = np.fromfile(str(data_path), dtype=np.uint16)
    n_tokens = len(tokens)
    print(f"[PPL {device}] {name}: {n_tokens:,} tokens, {n_tokens*2/1e6:.1f}MB")

    ds = SlidingWindowDataset(tokens, SEQ_LEN, STRIDE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    total_nll = 0.0
    total_count = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_idx, (inp, tgt, mask) in enumerate(dl):
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                reduction="none",
            )
            loss_flat = loss_flat.view(mask.shape)
            nll = (loss_flat * mask.float()).sum().item()
            cnt = mask.sum().item()
            total_nll += nll
            total_count += cnt

            if (batch_idx + 1) % 50 == 0:
                running_ppl = math.exp(total_nll / total_count) if total_count > 0 else float("inf")
                elapsed = time.time() - t0
                print(f"[PPL {device}] {name}: batch {batch_idx+1}/{len(dl)}, "
                      f"running PPL={running_ppl:.4f}, {elapsed:.0f}s")

    avg_nll = total_nll / total_count if total_count > 0 else 0
    ppl = math.exp(avg_nll)
    bpt = avg_nll / math.log(2)
    elapsed = time.time() - t0

    result = {
        "name": name,
        "file": val_file,
        "n_tokens": int(n_tokens),
        "n_eval_tokens": int(total_count),
        "ppl": round(ppl, 4),
        "bits_per_token": round(bpt, 4),
        "avg_nll": round(avg_nll, 6),
        "elapsed_sec": round(elapsed, 1),
        "device": device,
    }
    print(f"[PPL {device}] ✓ {name}: PPL={ppl:.4f}, BPT={bpt:.4f}, {elapsed:.1f}s")
    return result


def eval_ppl_multi(val_files: list[str], device: str) -> list[dict]:
    """Compute PPL for multiple small val sets on one GPU."""
    results = []
    for f in val_files:
        results.append(eval_ppl(f, device))
    return results


# ===========================================================================
# Task 2: Calibration (cuda:3)
# ===========================================================================

def eval_calibration(device: str = "cuda:3", n_tokens: int = 50000) -> dict:
    """Top-k accuracy and entropy calibration."""
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[CALIB {device}] Loading model...")
    model = load_model(device)
    tokenizer = load_tokenizer()

    tokens = np.fromfile(str(DATA_DIR / "3b_val.bin"), dtype=np.uint16)
    tokens = tokens[:min(n_tokens, len(tokens))]

    ds = SlidingWindowDataset(tokens, SEQ_LEN, STRIDE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_entropy = 0.0
    total_prob = 0.0
    total_count = 0
    t0 = time.time()

    with torch.inference_mode():
        for inp, tgt, mask in dl:
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)
            probs = F.softmax(logits, dim=-1)

            valid = mask & (tgt != -100)
            if valid.sum() == 0:
                continue

            flat_logits = logits[valid]
            flat_tgt = tgt[valid]
            flat_probs = probs[valid]

            # Top-k accuracy
            _, top1_pred = flat_logits.topk(1, dim=-1)
            _, top5_pred = flat_logits.topk(5, dim=-1)
            _, top10_pred = flat_logits.topk(10, dim=-1)

            top1_correct += (top1_pred.squeeze(-1) == flat_tgt).sum().item()
            top5_correct += (top5_pred == flat_tgt.unsqueeze(-1)).any(dim=-1).sum().item()
            top10_correct += (top10_pred == flat_tgt.unsqueeze(-1)).any(dim=-1).sum().item()

            # Mean probability of correct token
            correct_probs = flat_probs[torch.arange(len(flat_tgt)), flat_tgt]
            total_prob += correct_probs.sum().item()

            # Entropy
            log_probs = torch.log(flat_probs + 1e-10)
            entropy = -(flat_probs * log_probs).sum(dim=-1)
            total_entropy += entropy.sum().item()

            total_count += valid.sum().item()

    elapsed = time.time() - t0
    result = {
        "n_eval_tokens": int(total_count),
        "top1_accuracy": round(top1_correct / total_count, 4) if total_count > 0 else 0,
        "top5_accuracy": round(top5_correct / total_count, 4) if total_count > 0 else 0,
        "top10_accuracy": round(top10_correct / total_count, 4) if total_count > 0 else 0,
        "mean_correct_prob": round(total_prob / total_count, 4) if total_count > 0 else 0,
        "mean_entropy": round(total_entropy / total_count, 4) if total_count > 0 else 0,
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"[CALIB {device}] ✓ top1={result['top1_accuracy']:.4f}, "
          f"top5={result['top5_accuracy']:.4f}, entropy={result['mean_entropy']:.4f}, {elapsed:.1f}s")
    return result


# ===========================================================================
# Task 3: Generation quality (cuda:4)
# ===========================================================================

PROMPTS = [
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

TEMPERATURES = [0.0, 0.7, 1.0]


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


def generate_one(model, tokenizer, prompt, temperature, top_p=0.9, top_k=50,
                 max_new_tokens=256, device="cuda:4", repetition_penalty=1.0):
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("</s>")
    generated = input_ids
    new_ids = []
    hit_eos = False

    for _ in range(max_new_tokens):
        logits_all, _ = model(generated)
        logits = logits_all[:, -1, :].clone()

        if repetition_penalty != 1.0:
            for tid in set(generated[0].tolist()):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=-1)
        new_ids.append(next_id.item())

        if eos_id is not None and next_id.item() == eos_id:
            hit_eos = True
            break

    text = tokenizer.decode(new_ids)
    return text, len(new_ids), hit_eos


def compute_ngram_rep(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def eval_generation(device: str = "cuda:4") -> dict:
    """Generate text with 10 prompts × 3 temperatures."""
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[GEN {device}] Loading model...")
    model = load_model(device)
    tokenizer = load_tokenizer()
    t0 = time.time()

    results = []
    for prompt in PROMPTS:
        for temp in TEMPERATURES:
            with torch.inference_mode():
                text, n_tokens, hit_eos = generate_one(
                    model, tokenizer, prompt, temp, device=device
                )
            rep1 = compute_ngram_rep(text, 1)
            rep2 = compute_ngram_rep(text, 2)
            rep3 = compute_ngram_rep(text, 3)
            rep4 = compute_ngram_rep(text, 4)

            entry = {
                "prompt": prompt,
                "temperature": temp,
                "generated_tokens": n_tokens,
                "hit_eos": hit_eos,
                "1gram_rep": round(rep1, 4),
                "2gram_rep": round(rep2, 4),
                "3gram_rep": round(rep3, 4),
                "4gram_rep": round(rep4, 4),
                "text": text[:500],  # truncate for readability
            }
            results.append(entry)
            label = "greedy" if temp == 0.0 else f"t={temp}"
            print(f"[GEN {device}] {prompt[:10]}... ({label}): "
                  f"{n_tokens}tok, 3gram_rep={rep3:.2%}, eos={hit_eos}")

    elapsed = time.time() - t0

    # Aggregate stats
    greedy = [r for r in results if r["temperature"] == 0.0]
    sampled = [r for r in results if r["temperature"] > 0.0]

    summary = {
        "total_generations": len(results),
        "greedy_avg_3gram_rep": round(np.mean([r["3gram_rep"] for r in greedy]), 4) if greedy else 0,
        "greedy_eos_rate": round(np.mean([r["hit_eos"] for r in greedy]), 4) if greedy else 0,
        "sampled_avg_3gram_rep": round(np.mean([r["3gram_rep"] for r in sampled]), 4) if sampled else 0,
        "sampled_eos_rate": round(np.mean([r["hit_eos"] for r in sampled]), 4) if sampled else 0,
        "greedy_avg_tokens": round(np.mean([r["generated_tokens"] for r in greedy]), 1) if greedy else 0,
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"[GEN {device}] ✓ greedy 3gram_rep={summary['greedy_avg_3gram_rep']:.4f}, "
          f"eos_rate={summary['greedy_eos_rate']:.2%}, {elapsed:.1f}s")

    return {"summary": summary, "samples": results}


# ===========================================================================
# Task 4: Repetition parameter grid (cuda:5)
# ===========================================================================

REP_GRID = [
    {"name": "greedy", "temperature": 0.0, "repetition_penalty": 1.0},
    {"name": "t0.7", "temperature": 0.7, "repetition_penalty": 1.0},
    {"name": "t0.7_rep1.1", "temperature": 0.7, "repetition_penalty": 1.1},
    {"name": "t0.7_rep1.2", "temperature": 0.7, "repetition_penalty": 1.2},
    {"name": "t0.7_rep1.3", "temperature": 0.7, "repetition_penalty": 1.3},
    {"name": "t0.9", "temperature": 0.9, "repetition_penalty": 1.0},
    {"name": "t0.9_rep1.1", "temperature": 0.9, "repetition_penalty": 1.1},
    {"name": "t0.9_rep1.2", "temperature": 0.9, "repetition_penalty": 1.2},
    {"name": "t1.0", "temperature": 1.0, "repetition_penalty": 1.0},
    {"name": "t1.0_rep1.1", "temperature": 1.0, "repetition_penalty": 1.1},
]

REP_PROMPTS = [
    "대한민국의 수도는",
    "인공지능이란",
    "한국의 전통 음식 중에서",
    "지구 온난화의 주요 원인은",
    "프로그래밍을 배우려면",
]


def eval_repetition_grid(device: str = "cuda:5") -> dict:
    """Grid search over generation parameters to find lowest repetition."""
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[REP {device}] Loading model...")
    model = load_model(device)
    tokenizer = load_tokenizer()
    t0 = time.time()

    results = []
    for params in REP_GRID:
        combo_results = []
        for prompt in REP_PROMPTS:
            with torch.inference_mode():
                text, n_tokens, hit_eos = generate_one(
                    model, tokenizer, prompt,
                    temperature=params["temperature"],
                    repetition_penalty=params["repetition_penalty"],
                    device=device, max_new_tokens=256,
                )
            combo_results.append({
                "prompt": prompt,
                "n_tokens": n_tokens,
                "hit_eos": hit_eos,
                "3gram_rep": compute_ngram_rep(text, 3),
                "4gram_rep": compute_ngram_rep(text, 4),
            })

        avg_3gram = np.mean([r["3gram_rep"] for r in combo_results])
        avg_4gram = np.mean([r["4gram_rep"] for r in combo_results])
        eos_rate = np.mean([r["hit_eos"] for r in combo_results])
        avg_tokens = np.mean([r["n_tokens"] for r in combo_results])

        entry = {
            "params": params["name"],
            "temperature": params["temperature"],
            "repetition_penalty": params["repetition_penalty"],
            "avg_3gram_rep": round(avg_3gram, 4),
            "avg_4gram_rep": round(avg_4gram, 4),
            "eos_rate": round(eos_rate, 4),
            "avg_tokens": round(avg_tokens, 1),
        }
        results.append(entry)
        print(f"[REP {device}] {params['name']}: 3gram={avg_3gram:.2%}, "
              f"4gram={avg_4gram:.2%}, eos={eos_rate:.0%}, {avg_tokens:.0f}tok")

    elapsed = time.time() - t0

    # Find best combo
    best = min(results, key=lambda r: r["avg_3gram_rep"])
    print(f"[REP {device}] ✓ Best: {best['params']} (3gram={best['avg_3gram_rep']:.2%}), {elapsed:.1f}s")

    return {"grid_results": results, "best": best, "elapsed_sec": round(elapsed, 1)}


# ===========================================================================
# Main: parallel orchestration
# ===========================================================================

def run_ppl_0():
    return eval_ppl("3b_val.bin", "cuda:0")

def run_ppl_1():
    return eval_ppl("korean_c4_val.bin", "cuda:1")

def run_ppl_2():
    return eval_ppl_multi(["korean_namuwiki_val.bin", "korean_wiki_val.bin"], "cuda:2")

def run_calib():
    return eval_calibration("cuda:3")

def run_gen():
    return eval_generation("cuda:4")

def run_rep():
    return eval_repetition_grid("cuda:5")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("=" * 70)
    print("FRANKENSTALLM 3B — 6-GPU 병렬 종합 평가")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, Stride: {STRIDE}")
    print("=" * 70)

    t_start = time.time()
    all_results = {}

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(run_ppl_0): "ppl_3b_val",
            executor.submit(run_ppl_1): "ppl_c4_ko",
            executor.submit(run_ppl_2): "ppl_namuwiki_wiki",
            executor.submit(run_calib): "calibration",
            executor.submit(run_gen): "generation",
            executor.submit(run_rep): "repetition",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                all_results[key] = result
                print(f"\n{'='*50}")
                print(f"✓ {key} COMPLETED")
                print(f"{'='*50}\n")
            except Exception as e:
                print(f"\n✗ {key} FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = {"error": str(e)}

    total_elapsed = time.time() - t_start

    # Assemble final output
    output = {
        "model": "FRANKENSTALLM 3B",
        "checkpoint": "checkpoint-0057000",
        "total_elapsed_sec": round(total_elapsed, 1),
        "perplexity": {},
        "calibration": all_results.get("calibration", {}),
        "generation": all_results.get("generation", {}),
        "repetition": all_results.get("repetition", {}),
    }

    # Merge PPL results
    if "ppl_3b_val" in all_results and not isinstance(all_results["ppl_3b_val"], list):
        output["perplexity"]["3b_val"] = all_results["ppl_3b_val"]
    if "ppl_c4_ko" in all_results and not isinstance(all_results["ppl_c4_ko"], list):
        output["perplexity"]["korean_c4"] = all_results["ppl_c4_ko"]
    if "ppl_namuwiki_wiki" in all_results:
        for item in (all_results["ppl_namuwiki_wiki"] if isinstance(all_results["ppl_namuwiki_wiki"], list) else [all_results["ppl_namuwiki_wiki"]]):
            if isinstance(item, dict) and "name" in item:
                output["perplexity"][item["name"]] = item

    # Save
    out_path = OUTPUT_DIR / "3b_parallel_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FRANKENSTALLM 3B 종합 평가 결과 요약")
    print("=" * 70)
    print(f"총 소요 시간: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    print("\n--- Perplexity ---")
    for name, data in output["perplexity"].items():
        if isinstance(data, dict) and "ppl" in data:
            print(f"  {name}: PPL={data['ppl']:.4f}, BPT={data['bits_per_token']:.4f}")

    calib = output.get("calibration", {})
    if "top1_accuracy" in calib:
        print(f"\n--- Calibration ---")
        print(f"  Top-1 Acc: {calib['top1_accuracy']:.4f}")
        print(f"  Top-5 Acc: {calib['top5_accuracy']:.4f}")
        print(f"  Top-10 Acc: {calib['top10_accuracy']:.4f}")
        print(f"  Mean Entropy: {calib['mean_entropy']:.4f}")

    gen = output.get("generation", {}).get("summary", {})
    if gen:
        print(f"\n--- Generation Quality ---")
        print(f"  Greedy 3-gram rep: {gen.get('greedy_avg_3gram_rep', 0):.2%}")
        print(f"  Greedy EOS rate: {gen.get('greedy_eos_rate', 0):.2%}")
        print(f"  Sampled 3-gram rep: {gen.get('sampled_avg_3gram_rep', 0):.2%}")
        print(f"  Sampled EOS rate: {gen.get('sampled_eos_rate', 0):.2%}")

    rep = output.get("repetition", {}).get("best", {})
    if rep:
        print(f"\n--- Best Repetition Params ---")
        print(f"  Config: {rep.get('params', 'N/A')}")
        print(f"  3-gram rep: {rep.get('avg_3gram_rep', 0):.2%}")

    print(f"\n결과 저장: {out_path}")
    print("=" * 70)

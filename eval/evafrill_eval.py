"""
EVAFRILL-Mo 3B — 종합 평가 파이프라인
======================================

Phase 1: PPL (7-GPU 병렬, 14개 val 셋)
Phase 2: 생성 품질 + 반복률 분석 (cuda:0)
Phase 3: Calibration (cuda:0)
Phase 4: lm-eval 벤치마크 — 커스텀 래퍼 사용
          (belebele_kor_Hang, global_mmlu_full_ko, hellaswag, arc_easy, arc_challenge)

Usage:
    cd /home/ghong/project-ghong/taketimes/llm-star
    python eval/evafrill_eval.py
    python eval/evafrill_eval.py --skip-phase4
    python eval/evafrill_eval.py --checkpoint checkpoints/3b_final/checkpoint-0319772
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.transformer import LLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "3b_final" / "checkpoint-0319772")
TOKENIZER_PATH = str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json")
DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_DIR = _PROJECT_ROOT / "eval" / "outputs"

# GPUs available
N_GPUS = 7
GPU_IDS = list(range(N_GPUS))

# 한국어 생성 프롬프트 (15개)
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
    "머신러닝과 딥러닝의 차이는",
    "한국 문학의 대표적인 작품으로는",
    "양자 컴퓨터란",
    "건강한 식습관을 위해서는",
    "세계 2차 대전 이후",
]

# PPL 태스크: GPU → val 파일 리스트
PPL_TASKS: Dict[int, List[str]] = {
    0: ["3b_val.bin"],
    1: ["korean_c4_val.bin", "korean_val.bin"],
    2: ["hplt_ko_val.bin", "cc100_ko_val.bin"],
    3: ["korean_wiki_val.bin", "korean_namuwiki_val.bin"],
    4: ["cosmo_auto_math_text_val.bin", "cosmo_stories_val.bin", "cosmo_web_v2_val.bin"],
    5: ["cosmo_stanford_val.bin", "cosmo_khanacademy_val.bin", "cosmo_openstax_val.bin", "cosmo_wikihow_val.bin"],
    6: ["mathpile_val.bin", "open_web_math_val.bin"],
}


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVAFRILL-Mo 종합 평가")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-phase4", action="store_true")
    return parser.parse_args()


# ===========================================================================
# Sliding-window PPL dataset
# ===========================================================================

class BinDataset(Dataset):
    def __init__(self, path: str, seq_len: int, stride: int):
        data = np.fromfile(path, dtype=np.uint16)
        self.data = torch.from_numpy(data.astype(np.int64))
        self.seq_len = seq_len
        self.stride = stride
        self.indices = list(range(0, max(1, len(self.data) - seq_len), stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.data[start: start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        return chunk[:-1], chunk[1:]


# ===========================================================================
# PPL worker (runs in separate process)
# ===========================================================================

def _ppl_worker(
    checkpoint: str,
    gpu_id: int,
    val_files: List[str],
    data_dir: str,
    seq_len: int,
    stride: int,
    batch_size: int,
) -> Dict[str, float]:
    """각 GPU에서 여러 val 파일의 PPL을 계산."""
    import torch
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(checkpoint).parent.parent.parent))  # project root

    from model.transformer import LLM  # noqa

    device = f"cuda:{gpu_id}"
    model = LLM.from_pretrained(checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    results = {}
    for fname in val_files:
        fpath = Path(data_dir) / fname
        if not fpath.exists():
            results[fname.replace("_val.bin", "")] = None
            continue

        ds = BinDataset(str(fpath), seq_len, stride)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="sum",
                    ignore_index=0,
                )
                valid = (y != 0).sum().item()
                total_nll += loss.item()
                total_tokens += valid

        ppl = math.exp(total_nll / max(total_tokens, 1))
        key = fname.replace("_val.bin", "")
        results[key] = round(ppl, 4)
        print(f"[GPU {gpu_id}] {key}: PPL={ppl:.4f}", flush=True)

    return results


# ===========================================================================
# Phase 1: PPL (병렬)
# ===========================================================================

def run_phase1(checkpoint: str, seq_len: int, stride: int, batch_size: int) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print("Phase 1: PPL 평가 (7-GPU 병렬)")
    print("=" * 60)
    t0 = time.time()

    futures_map = {}
    all_results = {}
    ctx = torch.multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=N_GPUS, mp_context=ctx) as executor:
        for gpu_id, val_files in PPL_TASKS.items():
            # filter only existing files
            existing = [f for f in val_files if (DATA_DIR / f).exists()]
            if not existing:
                continue
            fut = executor.submit(
                _ppl_worker,
                checkpoint,
                gpu_id,
                existing,
                str(DATA_DIR),
                seq_len,
                stride,
                batch_size,
            )
            futures_map[fut] = gpu_id

        for fut in as_completed(futures_map):
            gpu_id = futures_map[fut]
            try:
                res = fut.result()
                all_results.update(res)
            except Exception as e:
                print(f"[GPU {gpu_id}] PPL worker 오류: {e}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 완료 ({elapsed:.1f}s)")
    return all_results


# ===========================================================================
# Phase 2: 생성 품질 + 반복률
# ===========================================================================

def _ngram_repetition(tokens: List[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return round(1.0 - unique / total, 4) if total > 0 else 0.0


def run_phase2(checkpoint: str, max_new_tokens: int) -> List[Dict]:
    print("\n" + "=" * 60)
    print("Phase 2: 생성 품질 + 반복률")
    print("=" * 60)

    device = "cuda:0"
    model = LLM.from_pretrained(checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    tok = Tokenizer.from_file(TOKENIZER_PATH)

    results = []
    configs = [
        ("greedy",    0.0, 1.0),
        ("t0.7",      0.7, 1.0),
        ("t0.7_r1.2", 0.7, 1.2),
        ("t0.9_r1.1", 0.9, 1.1),
    ]

    for prompt in PROMPTS:
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)

        row = {"prompt": prompt, "configs": {}}
        for cfg_name, temp, rep_pen in configs:
            with torch.no_grad():
                generated = list(ids)
                for _ in range(max_new_tokens):
                    inp = torch.tensor([generated[-2048:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    logits = logits[:, -1, :]

                    # Repetition penalty
                    if rep_pen != 1.0:
                        for tok_id in set(generated[-64:]):
                            logits[0, tok_id] /= rep_pen

                    if temp == 0.0:
                        next_tok = logits.argmax(dim=-1).item()
                    else:
                        probs = torch.softmax(logits / temp, dim=-1)
                        next_tok = torch.multinomial(probs[0], 1).item()

                    generated.append(next_tok)
                    if next_tok in (tok.token_to_id("</s>"), tok.token_to_id("<eos>"), 2):
                        break

            new_ids = generated[len(ids):]
            text = tok.decode(new_ids)
            rep3 = _ngram_repetition(new_ids, 3)
            rep4 = _ngram_repetition(new_ids, 4)
            eos_hit = new_ids[-1] in (2,) if new_ids else False

            row["configs"][cfg_name] = {
                "text": text,
                "tokens": len(new_ids),
                "3gram_rep": rep3,
                "4gram_rep": rep4,
                "eos": eos_hit,
            }

        results.append(row)
        greedy = row["configs"]["greedy"]
        print(f"\n[{prompt}]")
        print(f"  greedy({greedy['tokens']}tok, rep3={greedy['3gram_rep']:.2%}): {greedy['text'][:120]}")

    del model
    torch.cuda.empty_cache()
    return results


# ===========================================================================
# Phase 3: Calibration
# ===========================================================================

def run_phase3(checkpoint: str) -> Dict:
    print("\n" + "=" * 60)
    print("Phase 3: Calibration 체크")
    print("=" * 60)

    device = "cuda:0"
    model = LLM.from_pretrained(checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    val_path = DATA_DIR / "3b_val.bin"
    if not val_path.exists():
        print("  3b_val.bin 없음 — 스킵")
        return {}

    ds = BinDataset(str(val_path), seq_len=512, stride=256)
    loader = DataLoader(ds, batch_size=8, num_workers=0)

    top1 = top5 = top10 = total = 0
    mean_probs, mean_entropies = [], []

    CALIB_TOKENS = 50_000
    token_count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)

            mask = (y != 0)
            labels = y[mask]
            p = probs[mask]

            ranks = (p > p.gather(1, labels.unsqueeze(1))).sum(dim=1)
            top1 += (ranks < 1).sum().item()
            top5 += (ranks < 5).sum().item()
            top10 += (ranks < 10).sum().item()

            chosen_p = p.gather(1, labels.unsqueeze(1)).squeeze(1)
            mean_probs.append(chosen_p.mean().item())

            ent = -(p * (p + 1e-10).log()).sum(dim=-1)  # p already masked → 1D
            mean_entropies.append(ent.mean().item())

            total += labels.size(0)
            token_count += labels.size(0)
            if token_count >= CALIB_TOKENS:
                break

    result = {
        "top1_acc":  round(top1 / total, 4),
        "top5_acc":  round(top5 / total, 4),
        "top10_acc": round(top10 / total, 4),
        "mean_prob": round(float(np.mean(mean_probs)), 4),
        "mean_entropy": round(float(np.mean(mean_entropies)), 4),
        "total_tokens": total,
    }
    print(f"  Top-1: {result['top1_acc']:.2%}  Top-5: {result['top5_acc']:.2%}  Top-10: {result['top10_acc']:.2%}")
    print(f"  Mean prob: {result['mean_prob']:.4f}  Entropy: {result['mean_entropy']:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


# ===========================================================================
# Phase 4: lm-eval 벤치마크 (커스텀 래퍼)
# ===========================================================================

def run_phase4(checkpoint: str) -> Dict:
    print("\n" + "=" * 60)
    print("Phase 4: lm-eval 벤치마크")
    print("=" * 60)

    try:
        import lm_eval
        from lm_eval.api.model import LM as BaseLM
        from lm_eval.api.instance import Instance
        from lm_eval import evaluator
    except ImportError:
        print("  lm-eval 미설치 — 스킵 (pip install lm-eval)")
        return {}

    device = "cuda:0"

    class EvafrillLM(BaseLM):
        """EVAFRILL-Mo를 lm-eval-harness에 연결하는 래퍼."""

        def __init__(self, checkpoint: str, device: str, batch_size: int = 8):
            super().__init__()
            self._model = LLM.from_pretrained(checkpoint)
            self._model = self._model.to(device=device, dtype=torch.bfloat16)
            self._model.eval()
            self._tok = Tokenizer.from_file(TOKENIZER_PATH)
            self._device = device
            self._batch_size = batch_size
            self._max_len = 4096

        @property
        def eot_token_id(self) -> int:
            return 2  # </s>

        @property
        def max_length(self) -> int:
            return self._max_len

        @property
        def max_gen_toks(self) -> int:
            return 256

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str) -> List[int]:
            return self._tok.encode(string).ids

        def tok_decode(self, tokens) -> str:
            return self._tok.decode(list(tokens))

        def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits, _ = self._model(inps.to(self._device))
                return logits

        def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
            results = []
            for req in requests:
                ctx, cont = req.args[0], req.args[1]
                ctx_ids = self.tok_encode(ctx)
                cont_ids = self.tok_encode(cont)

                all_ids = ctx_ids + cont_ids
                if len(all_ids) > self._max_len:
                    all_ids = all_ids[-self._max_len:]
                    # adjust cont boundary
                    cont_start = len(all_ids) - len(cont_ids)
                else:
                    cont_start = len(ctx_ids)

                inp = torch.tensor([all_ids[:-1]], dtype=torch.long)
                tgt = torch.tensor([all_ids[1:]], dtype=torch.long)

                logits = self._model_call(inp)
                log_probs = F.log_softmax(logits, dim=-1)

                # sum log-probs over continuation tokens
                cont_log_prob = 0.0
                is_greedy = True
                for i, t in enumerate(cont_ids):
                    pos = cont_start - 1 + i
                    if pos >= log_probs.size(1):
                        break
                    cont_log_prob += log_probs[0, pos, t].item()
                    pred = log_probs[0, pos].argmax().item()
                    if pred != t:
                        is_greedy = False

                results.append((cont_log_prob, is_greedy))
            return results

        def loglikelihood_rolling(self, requests) -> List[float]:
            results = []
            for req in requests:
                text = req.args[0]
                ids = self.tok_encode(text)
                total_nll = 0.0
                for start in range(0, len(ids) - 1, self._max_len - 1):
                    chunk = ids[start: start + self._max_len]
                    if len(chunk) < 2:
                        break
                    inp = torch.tensor([chunk[:-1]], dtype=torch.long)
                    tgt = torch.tensor([chunk[1:]], dtype=torch.long)
                    logits = self._model_call(inp)
                    nll = F.cross_entropy(
                        logits[0], tgt[0].to(self._device), reduction="sum"
                    ).item()
                    total_nll += nll
                results.append(-total_nll)
            return results

        def generate_until(self, requests) -> List[str]:
            results = []
            for req in requests:
                ctx = req.args[0]
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}
                until = gen_kwargs.get("until", [])
                max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
                temp = gen_kwargs.get("temperature", 0.0)

                ids = self.tok_encode(ctx)
                generated = list(ids)

                with torch.no_grad():
                    for _ in range(max_gen):
                        inp = torch.tensor(
                            [generated[-self._max_len:]], dtype=torch.long
                        )
                        logits = self._model_call(inp)[:, -1:, :].squeeze(1)
                        if temp == 0.0:
                            next_tok = logits.argmax(dim=-1).item()
                        else:
                            probs = torch.softmax(logits / temp, dim=-1)
                            next_tok = torch.multinomial(probs[0], 1).item()
                        generated.append(next_tok)
                        if next_tok == self.eot_token_id:
                            break
                        decoded_new = self.tok_decode(generated[len(ids):])
                        if any(stop in decoded_new for stop in until):
                            break

                new_text = self.tok_decode(generated[len(ids):])
                for stop in until:
                    if stop in new_text:
                        new_text = new_text[:new_text.index(stop)]
                results.append(new_text)
            return results

    lm = EvafrillLM(checkpoint, device=device, batch_size=8)

    tasks = [
        "belebele_kor_Hang",
        "global_mmlu_full_ko",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
    ]

    print(f"  태스크: {', '.join(tasks)}")
    print("  (belebele/mmlu: 한국어, hellaswag/arc: 영어)")

    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=8,
            log_samples=False,
        )
        return results.get("results", {})
    except Exception as e:
        print(f"  lm-eval 오류: {e}")
        import traceback; traceback.print_exc()
        return {}


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    checkpoint: str,
    output_dir: Path,
    ppl: Dict,
    gen: List[Dict],
    calib: Dict,
    bench: Dict,
    elapsed: float,
) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = _PROJECT_ROOT / "reports" / f"{run_tag}_EVAFRILL_EVAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EVAFRILL-Mo 3B — 종합 평가 보고서",
        "",
        f"- **평가 일시**: {now}",
        f"- **체크포인트**: `{Path(checkpoint).name}`",
        f"- **총 소요 시간**: {elapsed/60:.1f}분",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    # PPL summary
    if ppl:
        avg_ko = np.mean([v for k, v in ppl.items() if v and "korean" in k or "hplt" in k or "cc100" in k])
        lines += [
            "### PPL (주요 셋)",
            "",
            "| 데이터셋 | PPL |",
            "|---------|-----|",
        ]
        for k, v in sorted(ppl.items()):
            if v is not None:
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    # Generation summary
    if gen:
        greedy_reps = [r["configs"]["greedy"]["3gram_rep"] for r in gen if "greedy" in r["configs"]]
        greedy_eos = [r["configs"]["greedy"]["eos"] for r in gen if "greedy" in r["configs"]]
        t07r12_reps = [r["configs"].get("t0.7_r1.2", {}).get("3gram_rep", None) for r in gen]
        t07r12_reps = [x for x in t07r12_reps if x is not None]

        lines += [
            "### 생성 품질 요약",
            "",
            f"| 설정 | 평균 3-gram 반복률 | EOS 종료율 |",
            f"|------|-------------------|-----------|",
            f"| greedy | {np.mean(greedy_reps):.2%} | {np.mean(greedy_eos):.0%} |",
        ]
        if t07r12_reps:
            t07r12_eos = [r["configs"].get("t0.7_r1.2", {}).get("eos", False) for r in gen]
            lines.append(f"| temp=0.7 rep=1.2 | {np.mean(t07r12_reps):.2%} | {np.mean(t07r12_eos):.0%} |")
        lines.append("")

    # Calibration
    if calib:
        lines += [
            "### Calibration",
            "",
            f"| Top-1 | Top-5 | Top-10 |",
            f"|-------|-------|--------|",
            f"| {calib['top1_acc']:.2%} | {calib['top5_acc']:.2%} | {calib['top10_acc']:.2%} |",
            "",
        ]

    # Benchmarks
    if bench:
        lines += [
            "### lm-eval 벤치마크",
            "",
            "| 태스크 | Accuracy | 랜덤 기준 |",
            "|--------|----------|----------|",
        ]
        random_baseline = {
            "belebele_kor_Hang": 0.25,
            "global_mmlu_full_ko": 0.25,
            "hellaswag": 0.25,
            "arc_easy": 0.25,
            "arc_challenge": 0.25,
        }
        for task, res in bench.items():
            acc = res.get("acc,none", res.get("acc", "N/A"))
            rb = random_baseline.get(task, "?")
            lines.append(f"| {task} | {acc:.4f} | {rb} |")
        lines.append("")

    # Generation samples
    if gen:
        lines += ["## 2. 생성 샘플 (Greedy)", ""]
        for r in gen:
            gcfg = r["configs"].get("greedy", {})
            lines += [
                f"**[{r['prompt']}]**",
                f"> {gcfg.get('text', '')[:200]}",
                f"> *EOS={gcfg.get('eos')}, 3gram_rep={gcfg.get('3gram_rep', 0):.2%}, tokens={gcfg.get('tokens')}*",
                "",
            ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  보고서 저장: {report_path}")

    # JSON 결과도 저장
    json_path = output_dir / "evafrill_eval_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ppl": ppl, "calib": calib, "bench": bench}, f, ensure_ascii=False, indent=2)

    return report_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    t_start = time.time()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "eval" / "outputs" / f"evafrill_eval_{run_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EVAFRILL-Mo 3B 종합 평가 시작")
    print(f"체크포인트: {args.checkpoint}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60)

    ppl_results = {}
    gen_results = []
    calib_results = {}
    bench_results = {}

    if not args.skip_phase1:
        ppl_results = run_phase1(
            args.checkpoint, args.seq_len, args.stride, args.batch_size
        )

    if not args.skip_phase2:
        gen_results = run_phase2(args.checkpoint, args.max_new_tokens)

    if not args.skip_phase3:
        calib_results = run_phase3(args.checkpoint)

    if not args.skip_phase4:
        bench_results = run_phase4(args.checkpoint)

    elapsed = time.time() - t_start
    report_path = generate_report(
        args.checkpoint, output_dir,
        ppl_results, gen_results, calib_results, bench_results,
        elapsed,
    )

    print("\n" + "=" * 60)
    print(f"평가 완료! 총 {elapsed/60:.1f}분")
    print(f"보고서: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Markdown report generator for FRANKENSTALLM 3B evaluation pipeline.

Generates comprehensive evaluation reports with sections for:
- Perplexity metrics across datasets
- Calibration statistics
- Token NLL distribution
- Generation quality samples
- Repetition parameter search results
- Standard benchmark results (lm-eval) — Korean + English
- 0-shot vs 5-shot comparison
- Comparison with reference models
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json


# =========================================================================
# Normalization helpers — map GPU-label keys to logical sections
# =========================================================================

def _normalize_phase1_results(raw: dict) -> dict:
    """Convert GPU-labelled phase1_results into logical sections.

    Returns dict with keys: perplexity, calibration, token_nll, generation, repetition.
    """
    normalized: Dict[str, Any] = {
        "perplexity": {},
        "calibration": {},
        "token_nll": {},
        "generation": {},
        "repetition": {},
    }

    for label, data in raw.items():
        if not isinstance(data, (dict, list)):
            continue

        if "PPL" in label:
            # PPL entries: single dict or list of dicts
            if isinstance(data, dict) and "ppl" in data:
                name = data.get("name", label)
                normalized["perplexity"][name] = data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "ppl" in item:
                        name = item.get("name", f"unknown_{len(normalized['perplexity'])}")
                        normalized["perplexity"][name] = item
            elif isinstance(data, dict) and "error" in data:
                # Task failed — skip
                pass
        elif "Calibration" in label:
            if isinstance(data, dict):
                if "calibration" in data:
                    normalized["calibration"] = data["calibration"]
                if "token_nll" in data:
                    normalized["token_nll"] = data["token_nll"]
        elif "Generation" in label:
            if isinstance(data, dict):
                normalized["generation"] = data
        elif "Repetition" in label:
            if isinstance(data, dict):
                normalized["repetition"] = data

    return normalized


def _normalize_phase2_results(raw: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert GPU-labelled phase2_results into flat task dicts for 0-shot and 5-shot.

    Returns (zero_shot_metrics, five_shot_metrics) where each is:
      {"kobest_boolq": {"acc,none": 0.50, ...}, "haerae": {...}, ...}
    """
    zero_shot: Dict[str, Any] = {}
    five_shot: Dict[str, Any] = {}

    for label, data in raw.items():
        if label == "5shot":
            # Recurse into 5-shot sub-dict
            if isinstance(data, dict):
                for sub_label, sub_data in data.items():
                    if isinstance(sub_data, dict) and "per_task_metrics" in sub_data:
                        for task_name, metrics in sub_data["per_task_metrics"].items():
                            five_shot[task_name] = metrics
            continue

        if isinstance(data, dict) and "per_task_metrics" in data:
            for task_name, metrics in data["per_task_metrics"].items():
                zero_shot[task_name] = metrics

    return zero_shot, five_shot


def _get_acc(metrics: dict, prefer_norm: bool = False) -> Optional[float]:
    """Extract accuracy from lm-eval metrics dict."""
    if prefer_norm and "acc_norm,none" in metrics:
        val = metrics["acc_norm,none"]
        if isinstance(val, (int, float)):
            return float(val)
    if "acc,none" in metrics:
        val = metrics["acc,none"]
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _fmt_pct(val: Optional[float]) -> str:
    """Format as percentage string or N/A."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def _fmt_f(val, decimals: int = 4) -> str:
    """Format float or return N/A."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val) if val is not None else "N/A"


# =========================================================================
# Main report generator
# =========================================================================

def generate_report(
    phase1_results: dict,
    phase2_results: dict,
    generation_samples: list,
    output_dir: Path,
    checkpoint_name: str = "checkpoint-0057000",
    total_elapsed_sec: float = 0.0,
) -> str:
    """Generate a comprehensive markdown evaluation report.

    Handles the GPU-labelled key structure from full_eval_pipeline.py
    and produces multiple report files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Normalize data
    p1 = _normalize_phase1_results(phase1_results)
    zero_shot, five_shot = _normalize_phase2_results(phase2_results)

    eval_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ===== Generate individual reports =====
    ppl_report = _generate_perplexity_report(p1["perplexity"])
    cal_report = _generate_calibration_report(p1["calibration"], p1["token_nll"])
    gen_report = _generate_generation_report(p1["generation"], generation_samples)
    bench_report = _generate_benchmark_report(zero_shot, five_shot, p1["repetition"])
    exec_summary = _generate_executive_summary(
        p1, zero_shot, five_shot, checkpoint_name, eval_datetime, total_elapsed_sec,
    )

    # Write individual reports
    (reports_dir / "00_executive_summary.md").write_text(exec_summary, encoding="utf-8")
    (reports_dir / "01_perplexity_report.md").write_text(ppl_report, encoding="utf-8")
    (reports_dir / "02_calibration_report.md").write_text(cal_report, encoding="utf-8")
    (reports_dir / "03_generation_quality.md").write_text(gen_report, encoding="utf-8")
    (reports_dir / "04_benchmark_report.md").write_text(bench_report, encoding="utf-8")

    # Combined full report
    full_report = "\n\n---\n\n".join([
        exec_summary, ppl_report, cal_report, gen_report, bench_report,
    ])

    report_path = output_dir / "full_eval_report.md"
    report_path.write_text(full_report, encoding="utf-8")

    return full_report


# =========================================================================
# Individual report sections
# =========================================================================

def _generate_executive_summary(
    p1: dict,
    zero_shot: dict,
    five_shot: dict,
    checkpoint_name: str,
    eval_datetime: str,
    total_elapsed_sec: float,
) -> str:
    lines = [
        "# FRANKENSTALLM 3B 종합 평가 리포트\n",
        f"- **모델**: FRANKENSTALLM 3B",
        f"- **체크포인트**: {checkpoint_name}",
        f"- **평가 일시**: {eval_datetime}",
        f"- **총 소요 시간**: {total_elapsed_sec:.1f}초\n",
        "## Executive Summary\n",
    ]

    # Main PPL
    main_ppl = "N/A"
    ppl_data = p1.get("perplexity", {})
    for name in ["3b", "3b_val"]:
        if name in ppl_data and isinstance(ppl_data[name], dict):
            main_ppl = _fmt_f(ppl_data[name].get("ppl"))
            break

    # KoBEST average
    kobest_tasks = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    kobest_accs = []
    for t in kobest_tasks:
        if t in zero_shot:
            a = _get_acc(zero_shot[t])
            if a is not None:
                kobest_accs.append(a)
    kobest_avg = _fmt_pct(sum(kobest_accs) / len(kobest_accs)) if kobest_accs else "N/A"

    # MMLU-KO — prefer group-level weighted average from lm-eval
    mmlu_ko_avg = "N/A"
    mmlu_ko_count = 0
    if "global_mmlu_ko" in zero_shot:
        a = _get_acc(zero_shot["global_mmlu_ko"])
        if a is not None:
            mmlu_ko_avg = _fmt_pct(a)
            # Count subtasks for display
            mmlu_ko_count = sum(
                1 for t in zero_shot
                if t.startswith("global_mmlu_ko_") and _get_acc(zero_shot[t]) is not None
            )
            if mmlu_ko_count == 0:
                mmlu_ko_count = 1  # group-level only
    else:
        # Fallback: average subtask-level metrics
        mmlu_ko_accs = []
        for t, m in zero_shot.items():
            if t.startswith("global_mmlu_ko_"):
                a = _get_acc(m)
                if a is not None:
                    mmlu_ko_accs.append(a)
        if mmlu_ko_accs:
            mmlu_ko_avg = _fmt_pct(sum(mmlu_ko_accs) / len(mmlu_ko_accs))
            mmlu_ko_count = len(mmlu_ko_accs)

    # MMLU-EN — exclude group-level keys to avoid double-counting
    _MMLU_EN_GROUPS = {"mmlu", "mmlu_humanities", "mmlu_social_sciences", "mmlu_stem", "mmlu_other"}
    mmlu_en_accs = []
    for t, m in zero_shot.items():
        if (t.startswith("mmlu_") or t == "mmlu") and t not in _MMLU_EN_GROUPS:
            a = _get_acc(m)
            if a is not None:
                mmlu_en_accs.append(a)
    if not mmlu_en_accs:
        # Fallback to group-level if no subtasks
        for t in _MMLU_EN_GROUPS:
            if t in zero_shot:
                a = _get_acc(zero_shot[t])
                if a is not None:
                    mmlu_en_accs.append(a)
    mmlu_en_avg = _fmt_pct(sum(mmlu_en_accs) / len(mmlu_en_accs)) if mmlu_en_accs else "N/A"

    # HAE-RAE
    haerae_acc = "N/A"
    if "haerae" in zero_shot:
        a = _get_acc(zero_shot["haerae"])
        if a is not None:
            haerae_acc = _fmt_pct(a)

    # English benchmarks
    en_benchmarks = {}
    for t in ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]:
        if t in zero_shot:
            a = _get_acc(zero_shot[t], prefer_norm=(t in ["hellaswag", "arc_challenge"]))
            if a is not None:
                en_benchmarks[t] = a

    # Top-1 accuracy
    top1 = _fmt_f(p1.get("calibration", {}).get("top1_accuracy"))

    lines.append("| 메트릭 | 값 |")
    lines.append("|--------|-----|")
    lines.append(f"| 주요 PPL (3b_val) | {main_ppl} |")
    lines.append(f"| MMLU-KO 평균 ({mmlu_ko_count}과목) | {mmlu_ko_avg} |")
    lines.append(f"| MMLU-EN 평균 | {mmlu_en_avg} |")
    lines.append(f"| KoBEST 평균 ({len(kobest_accs)}태스크) | {kobest_avg} |")
    lines.append(f"| HAE-RAE | {haerae_acc} |")
    for t, a in en_benchmarks.items():
        lines.append(f"| {t} (0-shot) | {_fmt_pct(a)} |")
    lines.append(f"| Top-1 정확도 (Calibration) | {top1} |")
    lines.append("")

    # Reference comparison
    lines.append("## 참고 모델 비교\n")
    lines.append("| 모델 | 파라미터 | MMLU-KO | MMLU-EN | KoBEST 평균 | PPL |")
    lines.append("|------|---------|---------|---------|------------|-----|")
    lines.append(f"| **FRANKENSTALLM 3B** | 3B | {mmlu_ko_avg} | {mmlu_en_avg} | {kobest_avg} | {main_ppl} |")
    lines.append("| Llama-3.2-3B | 3B | ~42% | ~58% | ~55% | — |")
    lines.append("| Qwen2.5-3B | 3B | ~48% | ~65% | ~60% | — |")
    lines.append("| EXAONE-3.5-2.4B | 2.4B | ~35% | ~50% | ~50% | — |")
    lines.append("")

    return "\n".join(lines)


def _generate_perplexity_report(ppl_data: dict) -> str:
    lines = ["# Perplexity 평가\n"]

    if not ppl_data:
        lines.append("데이터 없음\n")
        return "\n".join(lines)

    rows = []
    for name, metrics in ppl_data.items():
        if isinstance(metrics, dict) and "ppl" in metrics:
            rows.append({
                "name": name,
                "ppl": metrics.get("ppl"),
                "bits": metrics.get("bits_per_token"),
                "n_tokens": metrics.get("n_tokens"),
                "n_eval": metrics.get("n_eval_tokens"),
                "elapsed": metrics.get("elapsed_sec"),
            })

    rows.sort(key=lambda x: x["ppl"] if isinstance(x["ppl"], (int, float)) else float("inf"),
              reverse=True)

    lines.append("| 데이터셋 | PPL | Bits/Token | 전체 토큰 | 평가 토큰 | 소요 시간 |")
    lines.append("|---------|-----|-----------|---------|---------|---------|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt_f(r['ppl'])} | {_fmt_f(r['bits'])} | "
            f"{r['n_tokens']:,} | {r['n_eval']:,} | {_fmt_f(r['elapsed'], 1)}s |"
            if isinstance(r['n_tokens'], (int, float)) and isinstance(r['n_eval'], (int, float))
            else f"| {r['name']} | {_fmt_f(r['ppl'])} | {_fmt_f(r['bits'])} | "
                 f"{r['n_tokens']} | {r['n_eval']} | {_fmt_f(r['elapsed'], 1)}s |"
        )
    lines.append("")
    return "\n".join(lines)


def _generate_calibration_report(cal_data: dict, nll_data: dict) -> str:
    lines = ["# Calibration 및 Token NLL 분석\n"]

    # Calibration
    lines.append("## Calibration 결과\n")
    if cal_data:
        lines.append("| 메트릭 | 값 |")
        lines.append("|--------|-----|")
        metrics_map = {
            "top1_accuracy": "Top-1 Accuracy",
            "top5_accuracy": "Top-5 Accuracy",
            "top10_accuracy": "Top-10 Accuracy",
            "mean_correct_prob": "Mean Correct Prob",
            "mean_entropy": "Mean Entropy",
        }
        for key, label in metrics_map.items():
            lines.append(f"| {label} | {_fmt_f(cal_data.get(key))} |")
        lines.append("")
    else:
        lines.append("데이터 없음\n")

    # Token NLL
    lines.append("## Token NLL 분포\n")
    if nll_data:
        # Keys may be "mean"/"std" or "nll_mean"/"nll_std"
        stats_map = [
            (["nll_mean", "mean"], "평균"),
            (["nll_std", "std"], "표준편차"),
            (["nll_median", "median"], "중앙값"),
            (["nll_min", "min"], "최솟값"),
            (["nll_max", "max"], "최댓값"),
        ]
        lines.append("| 통계 | 값 |")
        lines.append("|------|-----|")
        for candidates, label in stats_map:
            val = None
            for c in candidates:
                if c in nll_data:
                    val = nll_data[c]
                    break
            lines.append(f"| {label} | {_fmt_f(val)} |")
        lines.append("")

        # Percentiles: "nll_percentiles" (dict) or "percentiles" (dict)
        pct_data = nll_data.get("nll_percentiles", nll_data.get("percentiles"))
        if pct_data and isinstance(pct_data, dict):
            lines.append("### Percentiles\n")
            lines.append("| Percentile | 값 |")
            lines.append("|------------|-----|")
            for pct, value in pct_data.items():
                lines.append(f"| {pct}th | {_fmt_f(value)} |")
            lines.append("")

        # High loss: "high_loss_fractions" (dict) or flat "high_loss_fraction_N" keys
        hlf = nll_data.get("high_loss_fractions")
        if hlf and isinstance(hlf, dict):
            lines.append("### 고손실 토큰 비율\n")
            lines.append("| 임계값 | 비율 |")
            lines.append("|--------|-----|")
            for threshold, fraction in hlf.items():
                lines.append(f"| NLL > {threshold} | {_fmt_f(fraction)} |")
            lines.append("")
        else:
            # Check flat keys: high_loss_fraction_5, high_loss_fraction_10, ...
            hlf_flat = {k.replace("high_loss_fraction_", ""): v
                        for k, v in nll_data.items()
                        if k.startswith("high_loss_fraction_")}
            if hlf_flat:
                lines.append("### 고손실 토큰 비율\n")
                lines.append("| 임계값 | 비율 |")
                lines.append("|--------|-----|")
                for threshold, fraction in sorted(hlf_flat.items()):
                    lines.append(f"| NLL > {threshold} | {_fmt_f(fraction)} |")
                lines.append("")
    else:
        lines.append("데이터 없음\n")

    return "\n".join(lines)


def _generate_generation_report(gen_data: dict, samples: list) -> str:
    lines = ["# 생성 품질 분석\n"]

    if gen_data and "summary" in gen_data:
        lines.append("## 요약 통계\n")
        lines.append("| 메트릭 | 값 |")
        lines.append("|--------|-----|")
        for key, value in gen_data["summary"].items():
            display = key.replace("_", " ").title()
            lines.append(f"| {display} | {_fmt_f(value)} |")
        lines.append("")

    if samples:
        lines.append("## 생성 샘플 (Greedy)\n")
        for i, sample in enumerate(samples[:5], 1):
            if isinstance(sample, dict):
                prompt = sample.get("prompt", "")
                generated = sample.get("generated_text", "")
                if len(generated) > 300:
                    generated = generated[:300] + "..."
                lines.append(f"### 샘플 {i}\n")
                lines.append(f"**Prompt**: {prompt}\n")
                lines.append(f"**Generated**: {generated}\n")
        lines.append("")
    elif not gen_data:
        lines.append("데이터 없음\n")

    return "\n".join(lines)


def _generate_benchmark_report(
    zero_shot: dict,
    five_shot: dict,
    repetition: dict,
) -> str:
    lines = ["# 표준 벤치마크 결과\n"]

    if not zero_shot and not five_shot:
        lines.append("데이터 없음\n")
        return "\n".join(lines)

    # --- Korean Benchmarks ---
    lines.append("## 한국어 벤치마크\n")

    # KoBEST
    kobest_names = ["kobest_boolq", "kobest_copa", "kobest_hellaswag",
                    "kobest_sentineg", "kobest_wic"]
    kobest_0 = {t: zero_shot[t] for t in kobest_names if t in zero_shot}
    if kobest_0:
        lines.append("### KoBEST (0-shot)\n")
        lines.append("| 태스크 | Accuracy | F1 |")
        lines.append("|--------|----------|-----|")
        for t in kobest_names:
            if t in kobest_0:
                m = kobest_0[t]
                acc = _fmt_pct(_get_acc(m))
                f1 = _fmt_f(m.get("f1,none"))
                lines.append(f"| {t} | {acc} | {f1} |")
        kobest_accs = [_get_acc(kobest_0[t]) for t in kobest_names
                       if t in kobest_0 and _get_acc(kobest_0[t]) is not None]
        if kobest_accs:
            lines.append(f"| **평균** | **{_fmt_pct(sum(kobest_accs)/len(kobest_accs))}** | |")
        lines.append("")

    # HAE-RAE
    if "haerae" in zero_shot:
        lines.append("### HAE-RAE (0-shot)\n")
        m = zero_shot["haerae"]
        lines.append(f"- Accuracy: {_fmt_pct(_get_acc(m))}")
        # Check for sub-tasks
        haerae_subs = {t: zero_shot[t] for t in zero_shot if t.startswith("haerae_") and t != "haerae"}
        if haerae_subs:
            lines.append("\n| 서브태스크 | Accuracy |")
            lines.append("|-----------|----------|")
            for t, sm in sorted(haerae_subs.items()):
                lines.append(f"| {t} | {_fmt_pct(_get_acc(sm))} |")
        lines.append("")

    # MMLU-KO
    mmlu_ko_tasks = {t: zero_shot[t] for t in zero_shot
                     if t.startswith("global_mmlu_ko") and t != "global_mmlu_ko"}
    if mmlu_ko_tasks or "global_mmlu_ko" in zero_shot:
        lines.append("### MMLU-KO (0-shot)\n")
        if mmlu_ko_tasks:
            lines.append(f"평가된 과목 수: **{len(mmlu_ko_tasks)}**\n")
            accs = [(t, _get_acc(m)) for t, m in sorted(mmlu_ko_tasks.items())
                    if _get_acc(m) is not None]
            if accs:
                # Prefer group-level weighted average from lm-eval
                group_acc = _get_acc(zero_shot["global_mmlu_ko"]) if "global_mmlu_ko" in zero_shot else None
                avg_acc = group_acc if group_acc is not None else sum(a for _, a in accs) / len(accs)
                lines.append(f"전체 평균: **{_fmt_pct(avg_acc)}**\n")

                # Top 10
                accs_sorted = sorted(accs, key=lambda x: x[1], reverse=True)
                lines.append("**상위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[:10]:
                    subject = t.replace("global_mmlu_ko_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")

                lines.append("**하위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[-10:]:
                    subject = t.replace("global_mmlu_ko_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")
        elif "global_mmlu_ko" in zero_shot:
            a = _get_acc(zero_shot["global_mmlu_ko"])
            lines.append(f"전체 정확도: {_fmt_pct(a)}\n")

    # --- English Benchmarks ---
    lines.append("## 영어 벤치마크\n")

    en_tasks = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
    en_found = {t: zero_shot[t] for t in en_tasks if t in zero_shot}
    if en_found:
        lines.append("### 주요 벤치마크 (0-shot)\n")
        lines.append("| 태스크 | Accuracy | Acc (norm) |")
        lines.append("|--------|----------|-----------|")
        for t in en_tasks:
            if t in en_found:
                m = en_found[t]
                acc = _fmt_pct(_get_acc(m))
                acc_norm = _fmt_pct(_get_acc(m, prefer_norm=True) if "acc_norm,none" in m else None)
                lines.append(f"| {t} | {acc} | {acc_norm} |")
        lines.append("")

    # MMLU-EN
    mmlu_en_tasks = {t: zero_shot[t] for t in zero_shot
                     if (t.startswith("mmlu_") or t == "mmlu") and not t.startswith("mmlu_ko")}
    if mmlu_en_tasks:
        lines.append("### MMLU-EN (0-shot)\n")
        # Filter out the group-level "mmlu" if sub-tasks exist
        subtasks = {t: m for t, m in mmlu_en_tasks.items() if t != "mmlu"}
        if subtasks:
            lines.append(f"평가된 과목 수: **{len(subtasks)}**\n")
            accs = [(t, _get_acc(m)) for t, m in sorted(subtasks.items())
                    if _get_acc(m) is not None]
            if accs:
                avg_acc = sum(a for _, a in accs) / len(accs)
                lines.append(f"전체 평균: **{_fmt_pct(avg_acc)}**\n")

                accs_sorted = sorted(accs, key=lambda x: x[1], reverse=True)
                lines.append("**상위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[:10]:
                    subject = t.replace("mmlu_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")

                lines.append("**하위 10개 과목**:\n")
                lines.append("| 과목 | Accuracy |")
                lines.append("|------|----------|")
                for t, a in accs_sorted[-10:]:
                    subject = t.replace("mmlu_", "")
                    lines.append(f"| {subject} | {_fmt_pct(a)} |")
                lines.append("")
        elif "mmlu" in mmlu_en_tasks:
            a = _get_acc(mmlu_en_tasks["mmlu"])
            lines.append(f"전체 정확도: {_fmt_pct(a)}\n")

    # --- 0-shot vs 5-shot Comparison ---
    if five_shot:
        lines.append("## 0-shot vs 5-shot 비교\n")

        # Collect all tasks that have both 0-shot and 5-shot results
        common_tasks = sorted(set(zero_shot.keys()) & set(five_shot.keys()))
        if common_tasks:
            lines.append("| 태스크 | 0-shot Acc | 5-shot Acc | 변화 |")
            lines.append("|--------|-----------|-----------|------|")
            for t in common_tasks:
                a0 = _get_acc(zero_shot[t])
                a5 = _get_acc(five_shot[t])
                if a0 is not None and a5 is not None:
                    diff = a5 - a0
                    sign = "+" if diff >= 0 else ""
                    lines.append(
                        f"| {t} | {_fmt_pct(a0)} | {_fmt_pct(a5)} | {sign}{diff*100:.2f}pp |"
                    )
                else:
                    lines.append(f"| {t} | {_fmt_pct(a0)} | {_fmt_pct(a5)} | — |")
            lines.append("")

            # Summary
            diffs = []
            for t in common_tasks:
                a0 = _get_acc(zero_shot[t])
                a5 = _get_acc(five_shot[t])
                if a0 is not None and a5 is not None:
                    diffs.append(a5 - a0)
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                improved = sum(1 for d in diffs if d > 0)
                degraded = sum(1 for d in diffs if d < 0)
                lines.append(
                    f"평균 변화: {'+' if avg_diff >= 0 else ''}{avg_diff*100:.2f}pp | "
                    f"개선: {improved} | 하락: {degraded} | 동일: {len(diffs) - improved - degraded}\n"
                )

    # --- Repetition ---
    if repetition and repetition.get("grid_results"):
        lines.append("## Repetition 파라미터 검색\n")
        rep_data = repetition["grid_results"]
        rep_rows = []
        # grid_results can be a list of dicts or a dict of dicts
        items = rep_data.items() if isinstance(rep_data, dict) else enumerate(rep_data)
        for key, metrics in items:
            if isinstance(metrics, dict):
                rep_rows.append({
                    "config": metrics.get("params", str(key)),
                    "temp": metrics.get("temperature"),
                    "rep_pen": metrics.get("repetition_penalty"),
                    "3gram": metrics.get("avg_3gram_rep", metrics.get("3gram_repetition", float("inf"))),
                    "4gram": metrics.get("avg_4gram_rep", metrics.get("4gram_repetition")),
                    "eos_rate": metrics.get("eos_rate"),
                    "avg_tokens": metrics.get("avg_tokens"),
                })
        rep_rows.sort(key=lambda x: x["3gram"] if isinstance(x["3gram"], (int, float)) else float("inf"))

        lines.append("| 설정 | Temp | Rep Pen | 3-gram | 4-gram | EOS Rate | Avg Tokens |")
        lines.append("|------|------|---------|--------|--------|----------|-----------|")
        for i, r in enumerate(rep_rows):
            marker = " **← best**" if i == 0 else ""
            lines.append(
                f"| {r['config']} | {_fmt_f(r['temp'], 2)} | {_fmt_f(r['rep_pen'], 2)} | "
                f"{_fmt_f(r['3gram'])} | {_fmt_f(r['4gram'])} | "
                f"{_fmt_f(r['eos_rate'])} | {_fmt_f(r['avg_tokens'], 1)} |{marker}"
            )
        lines.append("")

    lines.append("---\n")
    lines.append("*이 리포트는 자동으로 생성되었습니다.*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("report_generator.py — use via full_eval_pipeline.py or reeval_pipeline.py")

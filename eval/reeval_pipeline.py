"""
FRANKENSTALLM 3B — Re-evaluation Pipeline
==========================================

Re-runs Phase 2 benchmarks with corrected task names and English benchmarks,
then regenerates reports with the fixed report_generator.

Reuses:
  - HF checkpoint from previous eval run (Phase 0 skip)
  - phase1_results.json from previous eval run (Phase 1 skip)

Usage:
    python eval/reeval_pipeline.py
    python eval/reeval_pipeline.py --dry-run
    python eval/reeval_pipeline.py --skip-phase2  # regenerate reports only
    python eval/reeval_pipeline.py --prev-run eval/outputs/3b_full_eval_20260305_0318
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.full_eval_pipeline import (
    _bar,
    _build_phase2_tasks,
    _fmt_seconds,
    _print_banner,
    _print_phase_header,
    _save_json,
    _spawn_phase2_batch,
    _spawn_task,
    _wait_and_collect,
    CHECKPOINT,
    TOKENIZER_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reeval")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_PREV_RUN = _PROJECT_ROOT / "eval" / "outputs" / "3b_full_eval_20260305_0318"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRANKENSTALLM 3B — Re-evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prev-run",
        type=str,
        default=str(_DEFAULT_PREV_RUN),
        help="Previous eval run directory (for HF checkpoint and phase1 results).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: eval/outputs/3b_reeval_YYYYMMDD_HHMM/",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (re-run reports only from existing phase2 results).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task distribution without running.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs. Default: all 8 GPUs (0-7).",
    )
    return parser.parse_args()


def _find_hf_checkpoint(prev_run: Path) -> Optional[Path]:
    """Locate HF checkpoint in previous run directory."""
    candidates = list(prev_run.glob("hf_3b_*"))
    if candidates:
        return candidates[0]
    # Search one level up
    candidates = list(prev_run.parent.glob("**/hf_3b_*"))
    return candidates[0] if candidates else None


def _copy_phase1_artifacts(prev_run: Path, output_dir: Path) -> Dict[str, Any]:
    """Copy phase1_results.json and generation_samples.json from previous run."""
    phase1_src = prev_run / "phase1_results.json"
    phase1_dst = output_dir / "phase1_results.json"

    if not phase1_src.exists():
        raise FileNotFoundError(f"phase1_results.json not found in {prev_run}")

    shutil.copy2(phase1_src, phase1_dst)
    logger.info("  Copied phase1_results.json from previous run")

    # Also copy generation_samples if available
    gen_src = prev_run / "generation_samples.json"
    if gen_src.exists():
        shutil.copy2(gen_src, output_dir / "generation_samples.json")
        logger.info("  Copied generation_samples.json")

    with open(phase1_dst, encoding="utf-8") as f:
        return json.load(f)


def _copy_existing_reports(prev_run: Path, output_dir: Path) -> None:
    """Copy existing report files that won't change (perplexity, calibration, generation)."""
    prev_reports = prev_run / "reports"
    if not prev_reports.exists():
        return

    dst_reports = output_dir / "reports"
    dst_reports.mkdir(parents=True, exist_ok=True)

    # These are unchanged — copy as backup reference
    for fname in ["01_perplexity_report.md", "02_calibration_report.md",
                   "03_generation_quality_report.md"]:
        src = prev_reports / fname
        if src.exists():
            shutil.copy2(src, dst_reports / fname)


def run_phase2_reeval(
    hf_model_path: Path,
    output_dir: Path,
    gpu_ids: List[int],
) -> Dict[str, Any]:
    """Run Phase 2 benchmarks with per-GPU pipelining.

    Korean GPUs run 0-shot then 5-shot in the SAME process (model loaded
    once).  English GPUs run 0-shot only.  All GPUs are spawned in a single
    batch so there is no barrier between 0-shot and 5-shot — early-finishing
    Korean GPUs start 5-shot immediately while slow GPUs (e.g. MMLU-EN) are
    still running their 0-shot.
    """
    gpu_task_list = _build_phase2_tasks(gpu_ids)

    _KO_PREFIXES = ("kobest", "haerae", "global_mmlu_ko")

    processes: list = []
    for gpu_id, task_names, label in gpu_task_list:
        is_korean = any(t.startswith(_KO_PREFIXES) for t in task_names)

        out_path = output_dir / f"phase2_gpu{gpu_id}_pipeline_reeval.json"
        if is_korean:
            extra_args = {
                "--hf-model-path": str(hf_model_path),
                "--lm-eval-tasks": ",".join(task_names),
                "--fewshot-list": "0,5",
            }
            spawn_label = f"[pipeline 0+5shot] {label}"
        else:
            extra_args = {
                "--hf-model-path": str(hf_model_path),
                "--lm-eval-tasks": ",".join(task_names),
                "--num-fewshot": "0",
            }
            spawn_label = f"[0-shot] {label}"

        proc_info = _spawn_task(
            task_name="lm_eval",
            gpu_id=gpu_id,
            output_path=out_path,
            label=spawn_label,
            extra_args=extra_args,
        )
        processes.append(proc_info)

    logger.info(
        "  Spawned %d GPUs (Korean GPUs run 0+5-shot pipeline, EN GPUs 0-shot only).",
        len(gpu_task_list),
    )
    raw_results = _wait_and_collect(processes)

    # --- Reorganise into the expected output format ---
    # Pipeline results come as {"0shot": {...}, "5shot": {...}}
    # Non-pipeline results come as a flat dict (single fewshot).
    all_results: Dict[str, Any] = {}
    five_shot_bucket: Dict[str, Any] = {}

    for label, data in raw_results.items():
        if isinstance(data, dict) and "error" not in data and "0shot" in data:
            # Pipeline result — split into 0-shot and 5-shot
            zero_label = label.replace("[pipeline 0+5shot]", "[0-shot]")
            all_results[zero_label] = data["0shot"]
            if "5shot" in data and "error" not in data.get("5shot", {}):
                five_label = label.replace("[pipeline 0+5shot]", "[5-shot]")
                five_shot_bucket[five_label] = data["5shot"]
        else:
            all_results[label] = data

    if five_shot_bucket:
        all_results["5shot"] = five_shot_bucket

    _save_json(all_results, output_dir / "phase2_results.json")
    logger.info("  Phase 2 results saved: %s", output_dir / "phase2_results.json")

    return all_results


def run_phase3_reeval(
    phase1_results: Dict[str, Any],
    phase2_results: Dict[str, Any],
    output_dir: Path,
    total_elapsed_sec: float = 0.0,
) -> Optional[Path]:
    """Generate reports using the fixed report_generator."""
    try:
        from eval.report_generator import generate_report

        # Extract generation samples from phase1_results
        gen_samples = []
        for label, result in phase1_results.items():
            if isinstance(result, dict) and "Generation" in label:
                if "samples" in result:
                    gen_samples = result["samples"]
                    break

        generate_report(
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            generation_samples=gen_samples,
            output_dir=output_dir,
            checkpoint_name=Path(CHECKPOINT).name,
            total_elapsed_sec=total_elapsed_sec,
        )

        report_path = output_dir / "full_eval_report.md"
        logger.info("  Report saved: %s", report_path)
        logger.info("  Individual reports: %s", output_dir / "reports")
        return report_path
    except Exception:
        logger.error("  Report generation failed:\n%s", traceback.format_exc())
        return None


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    # Parse GPU IDs
    if args.gpus:
        gpu_ids = sorted([int(g.strip()) for g in args.gpus.split(",")])
    else:
        gpu_ids = list(range(8))

    # Resolve paths
    prev_run = Path(args.prev_run)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = _PROJECT_ROOT / "eval" / "outputs" / f"3b_reeval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find HF checkpoint
    hf_model_path = _find_hf_checkpoint(prev_run)

    # ---------------------------------------------------------------------------
    # Dry run
    # ---------------------------------------------------------------------------
    if args.dry_run:
        _print_banner("DRY RUN — FRANKENSTALLM 3B Re-evaluation Pipeline")
        logger.info("  Previous run : %s", prev_run)
        logger.info("  HF checkpoint: %s", hf_model_path or "NOT FOUND")
        logger.info("  Output dir   : %s", output_dir)
        logger.info("  GPUs         : %s", gpu_ids)

        _print_phase_header("Phase 2", f"Corrected Benchmarks ({len(gpu_ids)} GPUs)")
        gpu_task_list = _build_phase2_tasks(gpu_ids)
        logger.info("  %-6s  %-60s", "GPU", "Tasks")
        logger.info("  %s  %s", "-" * 6, "-" * 60)
        for gpu_id, tasks, label in gpu_task_list:
            logger.info("  cuda:%-2d  %s  (%d tasks)", gpu_id, label, len(tasks))
            for t in tasks:
                logger.info("           - %s", t)

        total_tasks = sum(len(tasks) for _, tasks, _ in gpu_task_list)
        logger.info("")
        logger.info("  Total benchmark tasks: %d", total_tasks)
        logger.info("  Estimated time: ~80 min (0-shot ~40 min + 5-shot ~40 min)")
        logger.info("")
        logger.info("  Dry run complete.")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Banner
    # ---------------------------------------------------------------------------
    _print_banner("FRANKENSTALLM 3B — Re-evaluation Pipeline")
    logger.info("  Previous run : %s", prev_run)
    logger.info("  HF checkpoint: %s", hf_model_path)
    logger.info("  Output dir   : %s", output_dir)
    logger.info("  GPUs         : %s", gpu_ids)
    logger.info("  Skip Phase 2 : %s", args.skip_phase2)

    pipeline_start = time.time()

    # ---------------------------------------------------------------------------
    # Phase 1 — Copy from previous run
    # ---------------------------------------------------------------------------
    _print_phase_header("PHASE 1", "Copy from Previous Run")
    try:
        phase1_results = _copy_phase1_artifacts(prev_run, output_dir)
    except FileNotFoundError as exc:
        logger.error("  %s", exc)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Phase 2 — Corrected Benchmarks
    # ---------------------------------------------------------------------------
    phase2_results: Dict[str, Any] = {}

    _print_phase_header("PHASE 2", f"Corrected Benchmarks — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase2:
        logger.info("  Skipping Phase 2.")
        phase2_out = output_dir / "phase2_results.json"
        if phase2_out.exists():
            with open(phase2_out, encoding="utf-8") as f:
                phase2_results = json.load(f)
            logger.info("  Loaded existing Phase 2 results: %s", phase2_out)
        else:
            # Try previous run
            prev_p2 = prev_run / "phase2_results.json"
            if prev_p2.exists():
                shutil.copy2(prev_p2, phase2_out)
                with open(phase2_out, encoding="utf-8") as f:
                    phase2_results = json.load(f)
                logger.info("  Copied Phase 2 results from previous run")
    elif hf_model_path is None:
        logger.error("  HF checkpoint not found in %s — cannot run Phase 2", prev_run)
        sys.exit(1)
    else:
        t0 = time.time()
        try:
            phase2_results = run_phase2_reeval(hf_model_path, output_dir, gpu_ids)
            logger.info("  Phase 2 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 2 FAILED:\n%s", traceback.format_exc())

    # ---------------------------------------------------------------------------
    # Phase 3 — Report Generation
    # ---------------------------------------------------------------------------
    _print_phase_header("PHASE 3", "Report Generation (Fixed)")
    t0 = time.time()
    report_path = run_phase3_reeval(
        phase1_results, phase2_results, output_dir,
        total_elapsed_sec=time.time() - pipeline_start,
    )
    logger.info("  Phase 3 complete in %s.", _fmt_seconds(time.time() - t0))

    # ---------------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    _print_banner("RE-EVALUATION COMPLETE")
    logger.info("  Total time      : %s", _fmt_seconds(total_elapsed))
    logger.info("  Output dir      : %s", output_dir)
    logger.info("  Phase 1 results : %s", output_dir / "phase1_results.json")
    logger.info("  Phase 2 results : %s", output_dir / "phase2_results.json")
    logger.info("  Report          : %s", report_path or "N/A")
    logger.info("  Reports dir     : %s", output_dir / "reports")

    if phase2_results:
        p2_entries = {k: v for k, v in phase2_results.items() if k != "5shot"}
        p2_ok = sum(1 for v in p2_entries.values()
                    if not (isinstance(v, dict) and "error" in v))
        p2_fail = len(p2_entries) - p2_ok
        logger.info("  Phase 2 (0-shot): %d OK / %d failed", p2_ok, p2_fail)

        five_shot = phase2_results.get("5shot", {})
        if isinstance(five_shot, dict) and "error" not in five_shot:
            fs_ok = sum(1 for v in five_shot.values()
                        if not (isinstance(v, dict) and "error" in v))
            logger.info("  Phase 2 (5-shot): %d OK", fs_ok)

    logger.info(_bar())


if __name__ == "__main__":
    main()

"""
FRANKENSTALLM 3B — Full Evaluation Pipeline Orchestrator
=========================================================

Runs 4 phases sequentially:
  Phase 0  — Convert checkpoint to HuggingFace format (convert_to_hf.py)
  Phase 1  — Internal evaluation across 8 GPUs (subprocess.Popen, isolated)
  Phase 2  — Standard benchmarks via lm-eval-harness (8 GPU parallel)
  Phase 3  — Report generation (eval/report_generator.py)

Usage:
    python eval/full_eval_pipeline.py
    python eval/full_eval_pipeline.py --dry-run
    python eval/full_eval_pipeline.py --skip-phase0 --skip-phase2
    python eval/full_eval_pipeline.py --checkpoint checkpoints/.../checkpoint-NNNNNNN
    python eval/full_eval_pipeline.py --output-dir eval/outputs/my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root — add to sys.path so sub-imports resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------
CHECKPOINT = str(
    _PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000"
)
TOKENIZER_PATH = str(
    _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
)
DATA_DIR = _PROJECT_ROOT / "data"
SEQ_LEN = 2048
STRIDE = 512
BATCH_SIZE = 32

# NUMA affinity: GPU 0-3 → cores 0-35 (NUMA node 0)
#                GPU 4-7 → cores 36-71 (NUMA node 1)
_NUMA_CORES: Dict[int, List[int]] = {
    0: list(range(0, 36)),
    1: list(range(0, 36)),
    2: list(range(0, 36)),
    3: list(range(0, 36)),
    4: list(range(36, 72)),
    5: list(range(36, 72)),
    6: list(range(36, 72)),
    7: list(range(36, 72)),
}

# Phase 1 val files distributed across GPUs 0-4
_PHASE1_PPL_FILES: Dict[int, List[str]] = {
    0: ["3b_val.bin"],
    1: ["korean_c4_val.bin", "korean_val.bin"],
    2: ["hplt_ko_val.bin", "cc100_ko_val.bin"],
    3: [
        "cosmo_auto_math_text_val.bin",
        "cosmo_stories_val.bin",
        "cosmo_web_v2_val.bin",
        "cosmo_stanford_val.bin",
        "cosmo_khanacademy_val.bin",
        "cosmo_openstax_val.bin",
        "cosmo_wikihow_val.bin",
    ],
    4: [
        "korean_namuwiki_val.bin",
        "korean_wiki_val.bin",
        "namuwiki_2023b_val.bin",
        "wikipedia_ko_val.bin",
        "mathpile_val.bin",
        "open_web_math_val.bin",
        "val.bin",
    ],
}

# Phase 2 lm-eval benchmark task assignment per GPU
_PHASE2_GPU_TASKS: Dict[int, List[str]] = {
    0: ["kobest_boolq", "kobest_copa"],
    1: ["kobest_hellaswag", "kobest_sentineg"],
    2: ["kobest_wic"],
    3: ["haerae"],
}
# global_mmlu_ko split across 4 GPUs (quarters) — populated at runtime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("full_eval")


# ===========================================================================
# NUMA Affinity Helper
# ===========================================================================

def set_numa_affinity(gpu_id: int) -> None:
    """Set CPU affinity of the calling process based on GPU NUMA node.

    GPU 0-3 → cores 0-35  (NUMA node 0)
    GPU 4-7 → cores 36-71 (NUMA node 1)
    """
    cores = _NUMA_CORES.get(gpu_id, list(range(72)))
    try:
        os.sched_setaffinity(0, cores)
    except AttributeError:
        # os.sched_setaffinity not available on non-Linux platforms
        pass
    except OSError as exc:
        # Non-fatal: log and continue
        print(f"[WARN] NUMA affinity set failed for GPU {gpu_id}: {exc}", flush=True)


# ===========================================================================
# Phase 1/2 — Subprocess helpers (Popen-based, fully isolated per task)
# ===========================================================================

def _isolate_gpu(gpu_id: int) -> None:
    """Set CUDA_VISIBLE_DEVICES and NUMA affinity for subprocess GPU isolation.

    After this call, the process only sees one GPU as cuda:0.
    Used in dry-run display only; actual isolation is done by _spawn_task().
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_numa_affinity(gpu_id)


def _spawn_task(
    task_name: str,
    gpu_id: int,
    output_path: Path,
    label: str,
    extra_args: Optional[Dict[str, str]] = None,
) -> Tuple[subprocess.Popen, str, Path, Any]:
    """Spawn a completely isolated subprocess for a single evaluation task.

    Each task runs as:
        CUDA_VISIBLE_DEVICES=<gpu_id> python eval/tasks/task_runner.py
            --task <task_name> --gpu-id <gpu_id> --output <output_path> [extra_args...]

    Returns (process, label, output_path, log_file).
    The caller must close log_file after the process finishes.
    """
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "eval" / "tasks" / "task_runner.py"),
        "--task", task_name,
        "--gpu-id", str(gpu_id),
        "--output", str(output_path),
    ]
    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([k, v])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    log_file = open(log_path, "w")  # noqa: WPS515 (resource managed by _wait_and_collect)

    logger.info("  Spawning: %s (GPU %d)", label, gpu_id)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(_PROJECT_ROOT),
    )
    return proc, label, output_path, log_file


def _wait_and_collect(
    processes: List[Tuple[subprocess.Popen, str, Path, Any]],
    max_timeout_sec: float = 3600.0,
) -> Dict[str, Any]:
    """Poll all spawned processes until completion and collect their JSON results.

    Each task_runner.py writes its result to output_path as JSON on success.
    On failure, the error and last 2000 chars of log are captured.
    Processes still running after *max_timeout_sec* are terminated.
    """
    results: Dict[str, Any] = {}
    success_count = 0
    failure_count = 0
    start_time = time.time()

    remaining = list(processes)
    while remaining:
        still_running = []
        for proc, label, out_path, log_file in remaining:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, label, out_path, log_file))
                continue

            log_file.close()
            log_path = out_path.with_suffix(".log")

            if ret == 0 and out_path.exists():
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    results[label] = result
                    success_count += 1
                    logger.info("  [DONE] %s", label)
                except Exception as exc:
                    results[label] = {"error": f"JSON parse failed: {exc}"}
                    failure_count += 1
                    logger.error("  [FAILED] %s — JSON parse error: %s", label, exc)
            else:
                error_msg = f"Process exited with code {ret}"
                try:
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    error_msg += f"\n--- Last log output ---\n{log_text}"
                except Exception:
                    pass
                results[label] = {"error": error_msg}
                failure_count += 1
                logger.error("  [FAILED] %s — exit code %d", label, ret)

        remaining = still_running

        # Timeout guard — terminate hung processes
        if remaining and (time.time() - start_time) > max_timeout_sec:
            logger.error(
                "  Timeout reached (%.0fs). Terminating %d remaining processes.",
                max_timeout_sec, len(remaining),
            )
            for proc, label, out_path, log_file in remaining:
                proc.terminate()
                log_file.close()
                results[label] = {"error": f"Timeout after {max_timeout_sec:.0f}s"}
                failure_count += 1
                logger.error("  [TIMEOUT] %s", label)
            remaining = []
            break

        if remaining:
            time.sleep(2)  # poll every 2 seconds

    logger.info("  Complete: %d succeeded, %d failed", success_count, failure_count)
    return results


# ---------------------------------------------------------------------------
# Phase 1 task distribution builder (adapts to available GPUs)
# ---------------------------------------------------------------------------

# All PPL val files grouped by workload size (descending)
_PPL_GROUPS = [
    (["3b_val.bin"], "PPL: 3b_val.bin"),
    (["korean_c4_val.bin", "korean_val.bin"], "PPL: korean_c4 + korean_val"),
    (["hplt_ko_val.bin", "cc100_ko_val.bin"], "PPL: hplt_ko + cc100_ko"),
    ([
        "cosmo_auto_math_text_val.bin", "cosmo_stories_val.bin",
        "cosmo_web_v2_val.bin", "cosmo_stanford_val.bin",
        "cosmo_khanacademy_val.bin", "cosmo_openstax_val.bin",
        "cosmo_wikihow_val.bin",
    ], "PPL: 7 cosmo files"),
    ([
        "korean_namuwiki_val.bin", "korean_wiki_val.bin",
        "namuwiki_2023b_val.bin", "wikipedia_ko_val.bin",
        "mathpile_val.bin", "open_web_math_val.bin", "val.bin",
    ], "PPL: 7 remaining files"),
]


def _build_phase1_tasks(gpu_ids: List[int]) -> List[Dict[str, Any]]:
    """Build Phase 1 task descriptors adapted to available GPUs.

    Returns a list of dicts with keys:
      - task     : task_runner.py --task value
      - gpu_id   : GPU to assign
      - label    : human-readable description
      - extra_args: dict of additional CLI flags (--val-file, --val-files, etc.)

    Strategy:
    - Reserve last 2-3 GPUs for non-PPL tasks (calib+NLL, generation, repetition)
    - Distribute PPL groups across remaining GPUs, merging if necessary
    """
    n = len(gpu_ids)
    tasks: List[Dict[str, Any]] = []

    if n < 3:
        raise ValueError(f"Need at least 3 GPUs, got {n}: {gpu_ids}")

    # Last GPU: repetition grid
    rep_gpu = gpu_ids[-1]
    # Second-to-last GPU: generation
    gen_gpu = gpu_ids[-2]

    # If we have >= 4 GPUs, give calibration+NLL its own GPU (third-to-last)
    if n >= 4:
        calib_gpu = gpu_ids[-3]
        ppl_gpus = gpu_ids[:-3]
        tasks.append({
            "task": "calib_nll",
            "gpu_id": calib_gpu,
            "label": f"GPU {calib_gpu} — Calibration + Token NLL",
            "extra_args": {},
        })
        tasks.append({
            "task": "generation",
            "gpu_id": gen_gpu,
            "label": f"GPU {gen_gpu} — Generation (15 prompts × 4 temps)",
            "extra_args": {},
        })
    else:
        # Tight on GPUs: combine calib+NLL+generation on second-to-last GPU
        ppl_gpus = gpu_ids[:-2]
        tasks.append({
            "task": "calib_nll_and_gen",
            "gpu_id": gen_gpu,
            "label": f"GPU {gen_gpu} — Calibration + NLL + Generation",
            "extra_args": {},
        })

    tasks.append({
        "task": "repetition_grid",
        "gpu_id": rep_gpu,
        "label": f"GPU {rep_gpu} — Repetition grid (12 × 5)",
        "extra_args": {},
    })

    # Distribute PPL groups across available PPL GPUs
    if len(ppl_gpus) == 0:
        # No dedicated PPL GPUs — merge all PPL into first available GPU
        all_files = []
        for files, _ in _PPL_GROUPS:
            all_files.extend(files)
        tasks.insert(0, {
            "task": "ppl_multi",
            "gpu_id": gpu_ids[0],
            "label": f"GPU {gpu_ids[0]} — PPL: all {len(all_files)} val files",
            "extra_args": {"--val-files": ",".join(all_files)},
        })
    elif len(ppl_gpus) >= len(_PPL_GROUPS):
        # One group per GPU (possibly some GPUs idle)
        for i, (files, label) in enumerate(_PPL_GROUPS):
            gpu = ppl_gpus[i]
            if len(files) == 1:
                tasks.append({
                    "task": "ppl_single",
                    "gpu_id": gpu,
                    "label": f"GPU {gpu} — {label}",
                    "extra_args": {"--val-file": files[0]},
                })
            else:
                tasks.append({
                    "task": "ppl_multi",
                    "gpu_id": gpu,
                    "label": f"GPU {gpu} — {label}",
                    "extra_args": {"--val-files": ",".join(files)},
                })
    else:
        # Fewer GPUs than groups — merge smallest groups
        merged: List[Tuple[List[str], str]] = list(_PPL_GROUPS)
        while len(merged) > len(ppl_gpus):
            a_files, a_label = merged.pop()
            b_files, b_label = merged.pop()
            merged.append((b_files + a_files, f"{b_label} + {a_label}"))
        for i, (files, label) in enumerate(merged):
            gpu = ppl_gpus[i]
            if len(files) == 1:
                tasks.append({
                    "task": "ppl_single",
                    "gpu_id": gpu,
                    "label": f"GPU {gpu} — {label}",
                    "extra_args": {"--val-file": files[0]},
                })
            else:
                tasks.append({
                    "task": "ppl_multi",
                    "gpu_id": gpu,
                    "label": f"GPU {gpu} — {label}",
                    "extra_args": {"--val-files": ",".join(files)},
                })

    return tasks


# ===========================================================================
# Banner / formatting helpers
# ===========================================================================

def _bar(char: str = "=", width: int = 72) -> str:
    return char * width


def _print_banner(title: str) -> None:
    logger.info(_bar())
    logger.info("  %s", title)
    logger.info(_bar())


def _print_phase_header(phase: str, description: str) -> None:
    logger.info("")
    logger.info(_bar("-"))
    logger.info("  %s — %s", phase, description)
    logger.info(_bar("-"))


def _fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ===========================================================================
# Dry-run helpers
# ===========================================================================

_ESTIMATED_TIMES = {
    "GPU 0 — PPL: 3b_val.bin":                        "~10 min",
    "GPU 1 — PPL: korean_c4_val + korean_val":         "~15 min",
    "GPU 2 — PPL: hplt_ko_val + cc100_ko_val":        "~15 min",
    "GPU 3 — PPL: 7 cosmo files":                      "~25 min",
    "GPU 4 — PPL: 7 remaining files":                  "~25 min",
    "GPU 5 — Calibration + Token NLL":                 "~20 min",
    "GPU 6 — Generation (15 prompts × 4 temps)":       "~20 min",
    "GPU 7 — Repetition grid (12 settings × 5 prompts)": "~15 min",
}


def _dry_run(args: argparse.Namespace, checkpoint: str, output_dir: Path,
             gpu_ids: Optional[List[int]] = None) -> None:
    """Validate configuration and print distribution tables without loading models."""
    _print_banner("DRY RUN — FRANKENSTALLM 3B Full Eval Pipeline")

    # Config summary
    logger.info("  Checkpoint  : %s", checkpoint)
    logger.info("  Tokenizer   : %s", TOKENIZER_PATH)
    logger.info("  Data dir    : %s", DATA_DIR)
    logger.info("  Output dir  : %s", output_dir)
    logger.info("  SEQ_LEN     : %d", SEQ_LEN)
    logger.info("  STRIDE      : %d", STRIDE)
    logger.info("  BATCH_SIZE  : %d", BATCH_SIZE)

    if gpu_ids is None:
        gpu_ids = list(range(8))

    # Phase 1 task distribution
    _print_phase_header("Phase 1", f"Internal Eval — {len(gpu_ids)} GPU Task Distribution")
    phase1_tasks = _build_phase1_tasks(gpu_ids)
    col_w = 60
    logger.info("  %-6s  %-*s  %s", "GPU", col_w, "Task", "NUMA")
    logger.info("  %s  %s  %s", "-" * 6, "-" * col_w, "-" * 20)
    for desc in phase1_tasks:
        gpu_id = desc["gpu_id"]
        label = desc["label"]
        numa_node = 0 if gpu_id < 4 else 1
        cores = _NUMA_CORES.get(gpu_id, [])
        core_range = f"cores {cores[0]}-{cores[-1]}" if cores else "?"
        logger.info("  cuda:%-2d  %-*s  [NUMA %d, %s]",
                    gpu_id, col_w, label, numa_node, core_range)

    # Phase 1 val file existence check
    _print_phase_header("Phase 1", "Val File Existence Check")
    all_files: List[str] = []
    for files in _PHASE1_PPL_FILES.values():
        all_files.extend(files)
    missing = []
    for fname in all_files:
        fpath = DATA_DIR / fname
        status = "OK" if fpath.exists() else "MISSING"
        logger.info("  [%s] %s", status, fpath)
        if status == "MISSING":
            missing.append(fname)

    if missing:
        logger.warning("  %d val file(s) missing — those tasks will be skipped at runtime.", len(missing))
    else:
        logger.info("  All %d val files present.", len(all_files))

    # Checkpoint existence
    _print_phase_header("Phase 0", "Checkpoint Existence Check")
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        logger.info("  [OK] Checkpoint found: %s", ckpt_path)
    else:
        logger.warning("  [MISSING] Checkpoint not found: %s", ckpt_path)

    hf_output = output_dir / f"hf_3b_{ckpt_path.name}"
    logger.info("  HF output will be: %s", hf_output)

    # Phase 2 task distribution
    _print_phase_header("Phase 2", f"lm-eval Benchmark Distribution (0-shot, {len(gpu_ids)} GPUs)")
    phase2_tasks = _build_phase2_tasks(gpu_ids)
    logger.info("  %-6s  %-60s", "GPU", "Tasks")
    logger.info("  %s  %s", "-" * 6, "-" * 60)
    for gpu_id, tasks, label in phase2_tasks:
        logger.info("  cuda:%-2d  %s", gpu_id, label)

    # NUMA summary
    _print_phase_header("NUMA Affinity", "GPU → Core Mapping")
    logger.info("  %-6s  %-10s  %-12s  %s", "GPU", "NUMA node", "Core range", "Cores")
    logger.info("  %s  %s  %s  %s", "-" * 6, "-" * 10, "-" * 12, "-" * 12)
    for gpu_id in gpu_ids:
        cores = _NUMA_CORES[gpu_id]
        numa = 0 if gpu_id < 4 else 1
        logger.info("  cuda:%-2d  node %-5d  %3d - %-5d  (%d cores)",
                    gpu_id, numa, cores[0], cores[-1], len(cores))

    logger.info("")
    logger.info("  Dry run complete. No models were loaded.")
    sys.exit(0)


# ===========================================================================
# Phase 0 — HF Checkpoint Conversion
# ===========================================================================

def run_phase0(checkpoint: str, output_dir: Path) -> Path:
    """Convert custom checkpoint to HuggingFace format via subprocess."""
    ckpt_name = Path(checkpoint).name
    hf_output = output_dir / f"hf_3b_{ckpt_name}"
    hf_output.mkdir(parents=True, exist_ok=True)

    convert_script = _PROJECT_ROOT / "scripts" / "convert_to_hf.py"
    cmd = [
        sys.executable,
        str(convert_script),
        "--checkpoint", checkpoint,
        "--output", str(hf_output),
        "--tokenizer", TOKENIZER_PATH,
    ]
    logger.info("  Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Phase 0 failed: convert_to_hf.py exited with {exc.returncode}") from exc

    logger.info("  HF checkpoint saved to: %s", hf_output)
    return hf_output


# ===========================================================================
# Phase 1 — Internal Evaluation (8 GPU, subprocess.Popen isolated)
# ===========================================================================

def run_phase1(output_dir: Path, gpu_ids: List[int]) -> Dict[str, Any]:
    """Run internal eval tasks in parallel across the given GPUs.

    Each task is launched as a completely isolated subprocess via task_runner.py.
    Results are collected by polling until all processes finish.

    Returns merged results dict.
    """
    task_descriptors = _build_phase1_tasks(gpu_ids)
    processes: List[Tuple[subprocess.Popen, str, Path, Any]] = []

    for desc in task_descriptors:
        out_path = output_dir / f"phase1_{desc['task']}_gpu{desc['gpu_id']}.json"
        proc_info = _spawn_task(
            task_name=desc["task"],
            gpu_id=desc["gpu_id"],
            output_path=out_path,
            label=desc["label"],
            extra_args=desc.get("extra_args"),
        )
        processes.append(proc_info)

    results = _wait_and_collect(processes)

    # Persist combined results
    phase1_out = output_dir / "phase1_results.json"
    _save_json(results, phase1_out)
    logger.info("  Phase 1 results saved: %s", phase1_out)

    # Save generation samples separately if present — scan by label content
    gen_samples: Dict[str, Any] = {}
    for label, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            if "Generation" in label:
                gen_samples["generation"] = result
            elif "Repetition" in label:
                gen_samples["repetition_grid"] = result
    if gen_samples:
        gen_out = output_dir / "generation_samples.json"
        _save_json(gen_samples, gen_out)
        logger.info("  Generation samples saved: %s", gen_out)

    return results


# ===========================================================================
# Phase 2 — lm-eval Benchmarks (8 GPU, subprocess.Popen isolated)
# ===========================================================================

# Benchmark task groups — balanced for 8 GPU parallel execution.
# MMLU-EN is split into 2 category groups to avoid a single GPU bottleneck
# (previously: 1 GPU took 210s for all 57 MMLU subtasks while others finished in 83-108s).
# lm-eval 0.4.x provides mmlu_humanities, mmlu_social_sciences, mmlu_stem, mmlu_other.
_BENCHMARK_GROUPS = [
    (["kobest_boolq", "kobest_copa", "kobest_wic"], "KoBEST: boolq + copa + wic"),
    (["kobest_hellaswag", "kobest_sentineg"], "KoBEST: hellaswag + sentineg"),
    (["haerae"], "HAE-RAE (all subtasks)"),
    (["global_mmlu_ko"], "MMLU-KO (57 subtasks)"),
    (["hellaswag", "arc_easy", "arc_challenge"], "EN: hellaswag + arc_easy + arc_challenge"),
    (["winogrande", "piqa"], "EN: winogrande + piqa"),
    (["mmlu_humanities", "mmlu_social_sciences"], "MMLU-EN: humanities + social_sciences"),
    (["mmlu_stem", "mmlu_other"], "MMLU-EN: stem + other"),
]


def _build_phase2_tasks(gpu_ids: List[int]) -> List[Tuple[int, List[str], str]]:
    """Distribute lm-eval benchmark tasks across available GPUs."""
    n = len(gpu_ids)
    task_list: List[Tuple[int, List[str], str]] = []

    if n <= 0:
        return task_list

    # Assign benchmark groups to GPUs (round-robin if fewer GPUs than groups)
    for i, (tasks, label) in enumerate(_BENCHMARK_GROUPS):
        gpu_id = gpu_ids[i % n]
        # If GPU already has tasks assigned (round-robin wrap), merge
        existing = None
        for j, (gid, existing_tasks, existing_label) in enumerate(task_list):
            if gid == gpu_id:
                existing = j
                break
        if existing is not None:
            gid, existing_tasks, existing_label = task_list[existing]
            task_list[existing] = (gid, existing_tasks + tasks,
                                   f"{existing_label} + {label}")
        else:
            task_list.append((gpu_id, tasks, f"GPU {gpu_id} — {label}"))

    return task_list


def _spawn_phase2_batch(
    hf_model_path: Path,
    output_dir: Path,
    gpu_task_list: List[Tuple[int, List[str], str]],
    num_fewshot: int,
    label_suffix: str,
) -> Dict[str, Any]:
    """Spawn all Phase 2 lm_eval subprocesses for one fewshot setting and collect results."""
    processes: List[Tuple[subprocess.Popen, str, Path, Any]] = []

    for gpu_id, task_names, label in gpu_task_list:
        fewshot_label = f"[{num_fewshot}-shot] {label}"
        out_path = output_dir / f"phase2_gpu{gpu_id}_{num_fewshot}shot{label_suffix}.json"
        proc_info = _spawn_task(
            task_name="lm_eval",
            gpu_id=gpu_id,
            output_path=out_path,
            label=fewshot_label,
            extra_args={
                "--hf-model-path": str(hf_model_path),
                "--lm-eval-tasks": ",".join(task_names),
                "--num-fewshot": str(num_fewshot),
            },
        )
        processes.append(proc_info)

    return _wait_and_collect(processes)


def run_phase2(
    hf_model_path: Path,
    output_dir: Path,
    gpu_ids: Optional[List[int]] = None,
    num_fewshot: int = 0,
) -> Dict[str, Any]:
    """Run lm-eval benchmarks across available GPUs in parallel.

    Each GPU runs its benchmark group as a completely isolated subprocess
    via task_runner.py. After 0-shot completes, attempts 5-shot (best-effort).
    """
    if gpu_ids is None:
        gpu_ids = list(range(8))

    gpu_task_list = _build_phase2_tasks(gpu_ids)

    logger.info("  Running %d-shot benchmarks on %d GPUs ...", num_fewshot, len(gpu_ids))
    results = _spawn_phase2_batch(hf_model_path, output_dir, gpu_task_list, num_fewshot, "")

    logger.info("  Phase 2 (%d-shot) complete.", num_fewshot)

    # Attempt 5-shot if we ran 0-shot
    if num_fewshot == 0:
        logger.info("  Attempting 5-shot benchmarks ...")
        try:
            five_shot_results = _spawn_phase2_batch(
                hf_model_path, output_dir, gpu_task_list, 5, "_5shot"
            )
            logger.info("  Phase 2 (5-shot) complete.")
        except Exception:
            logger.warning("  5-shot benchmarks failed (non-fatal): %s",
                           traceback.format_exc())
            five_shot_results = {"error": traceback.format_exc()}
        results["5shot"] = five_shot_results

    phase2_out = output_dir / "phase2_results.json"
    _save_json(results, phase2_out)
    logger.info("  Phase 2 results saved: %s", phase2_out)

    return results


# ===========================================================================
# Phase 3 — Report Generation
# ===========================================================================

def run_phase3(
    phase1_results: Dict[str, Any],
    phase2_results: Dict[str, Any],
    output_dir: Path,
    total_elapsed_sec: float = 0.0,
) -> Optional[Path]:
    """Generate markdown report from all collected results."""
    report_path = output_dir / "full_eval_report.md"
    try:
        from eval.report_generator import generate_report  # type: ignore[import]

        # Extract generation samples from phase1_results
        gen_samples = []
        gen_label = "GPU 6 — Generation (15 prompts × 4 temps)"
        if gen_label in phase1_results and isinstance(phase1_results[gen_label], dict):
            gen_data = phase1_results[gen_label]
            if "samples" in gen_data:
                gen_samples = gen_data["samples"]

        generate_report(
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            generation_samples=gen_samples,
            output_dir=report_path.parent,
            checkpoint_name=Path(CHECKPOINT).name,
            total_elapsed_sec=total_elapsed_sec,
        )
        logger.info("  Report saved: %s", report_path)
        return report_path
    except ImportError:
        logger.warning(
            "  eval.report_generator not found — generating minimal fallback report."
        )
        _write_fallback_report(phase1_results, phase2_results, report_path)
        return report_path
    except Exception:
        logger.error("  Phase 3 report generation failed:\n%s", traceback.format_exc())
        return None


def _write_fallback_report(
    phase1_results: Dict[str, Any],
    phase2_results: Dict[str, Any],
    report_path: Path,
) -> None:
    """Write a simple markdown report when report_generator is unavailable."""
    lines: List[str] = [
        "# FRANKENSTALLM 3B — Full Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Phase 1 Results",
        "",
    ]
    for label, result in phase1_results.items():
        lines.append(f"### {label}")
        if isinstance(result, dict) and "error" in result:
            lines.append(f"**FAILED**: {result['error'][:200]}")
        else:
            lines.append(f"```json\n{json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000]}\n```")
        lines.append("")

    lines += [
        "## Phase 2 Results",
        "",
    ]
    for label, result in phase2_results.items():
        lines.append(f"### {label}")
        if isinstance(result, dict) and "error" in result:
            lines.append(f"**FAILED**: {result['error'][:200]}")
        else:
            lines.append(f"```json\n{json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000]}\n```")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# Utilities
# ===========================================================================

def _save_json(data: Any, path: Path) -> None:
    """Save data as JSON, converting non-serialisable objects to strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _make_output_dir(output_dir_override: Optional[str]) -> Path:
    if output_dir_override:
        out = Path(output_dir_override)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out = _PROJECT_ROOT / "eval" / "outputs" / f"3b_full_eval_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ===========================================================================
# CLI Argument Parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRANKENSTALLM 3B — Full Evaluation Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate task distribution without loading models, then exit.",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip HF conversion (reuse existing checkpoint in outputs/).",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip internal 8-GPU evaluation.",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip lm-eval-harness benchmarks.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Override checkpoint path (default: {CHECKPOINT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: eval/outputs/3b_full_eval_YYYYMMDD_HHMM/)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use, e.g. '2,3,4,5,6,7'. Default: all 8 GPUs (0-7).",
    )
    return parser.parse_args()


# ===========================================================================
# Main Orchestrator
# ===========================================================================

def main() -> None:
    # Use "spawn" start method to avoid CUDA fork issues
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set in some environments

    args = parse_args()

    # Resolve checkpoint
    checkpoint = args.checkpoint if args.checkpoint else CHECKPOINT

    # Create output directory
    output_dir = _make_output_dir(args.output_dir)

    # Parse GPU IDs
    if args.gpus:
        gpu_ids = sorted([int(g.strip()) for g in args.gpus.split(",")])
    else:
        gpu_ids = list(range(8))

    # Dry run — validate and exit
    if args.dry_run:
        _dry_run(args, checkpoint, output_dir, gpu_ids)
        return  # unreachable (dry_run calls sys.exit), but for clarity

    # ---------------------------------------------------------------------------
    # Banner
    # ---------------------------------------------------------------------------
    _print_banner("FRANKENSTALLM 3B — Full Evaluation Pipeline")
    logger.info("  Checkpoint  : %s", checkpoint)
    logger.info("  Tokenizer   : %s", TOKENIZER_PATH)
    logger.info("  Data dir    : %s", DATA_DIR)
    logger.info("  Output dir  : %s", output_dir)
    logger.info("  GPUs        : %s", gpu_ids)
    logger.info("  SEQ_LEN     : %d   STRIDE: %d   BATCH_SIZE: %d",
                SEQ_LEN, STRIDE, BATCH_SIZE)
    logger.info("  Phases      : phase0=%s  phase1=%s  phase2=%s",
                "skip" if args.skip_phase0 else "run",
                "skip" if args.skip_phase1 else "run",
                "skip" if args.skip_phase2 else "run")

    pipeline_start = time.time()
    phase1_results: Dict[str, Any] = {}
    phase2_results: Dict[str, Any] = {}
    hf_model_path: Optional[Path] = None

    # -----------------------------------------------------------------------
    # Phase 0 — HF Conversion
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 0", "HF Checkpoint Conversion")
    if args.skip_phase0:
        # Try to locate an existing hf checkpoint in outputs/
        ckpt_name = Path(checkpoint).name
        candidate = output_dir / f"hf_3b_{ckpt_name}"
        if candidate.exists():
            hf_model_path = candidate
            logger.info("  Skipping Phase 0 — reusing: %s", hf_model_path)
        else:
            # Search any parent of output_dir
            candidates = list(output_dir.parent.glob(f"**/hf_3b_{ckpt_name}"))
            if candidates:
                hf_model_path = candidates[0]
                logger.info("  Skipping Phase 0 — reusing found: %s", hf_model_path)
            else:
                logger.warning(
                    "  --skip-phase0 set but no HF checkpoint found for %s. "
                    "Phase 2 will be skipped unless you specify --skip-phase2 "
                    "or set --output-dir to a directory containing the HF checkpoint.",
                    ckpt_name,
                )
    else:
        t0 = time.time()
        try:
            hf_model_path = run_phase0(checkpoint, output_dir)
            logger.info("  Phase 0 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 0 FAILED:\n%s", traceback.format_exc())
            logger.warning("  Continuing without HF conversion — Phase 2 will be skipped.")

    # -----------------------------------------------------------------------
    # Phase 1 — Internal Evaluation (8 GPU parallel)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 1", f"Internal Evaluation — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase1:
        logger.info("  Skipping Phase 1.")
        # Try to load existing results
        phase1_out = output_dir / "phase1_results.json"
        if phase1_out.exists():
            with open(phase1_out, encoding="utf-8") as f:
                phase1_results = json.load(f)
            logger.info("  Loaded existing Phase 1 results from: %s", phase1_out)
    else:
        t0 = time.time()
        try:
            phase1_results = run_phase1(output_dir, gpu_ids)
            logger.info("  Phase 1 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 1 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 2 — lm-eval Benchmarks (8 GPU parallel)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 2", f"lm-eval Benchmarks — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase2:
        logger.info("  Skipping Phase 2.")
        phase2_out = output_dir / "phase2_results.json"
        if phase2_out.exists():
            with open(phase2_out, encoding="utf-8") as f:
                phase2_results = json.load(f)
            logger.info("  Loaded existing Phase 2 results from: %s", phase2_out)
    elif hf_model_path is None:
        logger.warning("  Phase 2 skipped — HF model path unavailable (Phase 0 failed or skipped).")
    else:
        t0 = time.time()
        try:
            phase2_results = run_phase2(hf_model_path, output_dir, gpu_ids=gpu_ids,
                                               num_fewshot=0)
            logger.info("  Phase 2 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 2 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 3 — Report Generation
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 3", "Report Generation")
    t0 = time.time()
    report_path = run_phase3(phase1_results, phase2_results, output_dir,
                              total_elapsed_sec=time.time() - pipeline_start)
    logger.info("  Phase 3 complete in %s.", _fmt_seconds(time.time() - t0))

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    _print_banner("PIPELINE COMPLETE")
    logger.info("  Total time      : %s", _fmt_seconds(total_elapsed))
    logger.info("  Output dir      : %s", output_dir)
    logger.info("  Phase 1 results : %s", output_dir / "phase1_results.json")
    logger.info("  Phase 2 results : %s", output_dir / "phase2_results.json")
    logger.info("  Gen samples     : %s", output_dir / "generation_samples.json")
    logger.info("  Report          : %s", report_path or "N/A (generation failed)")

    # Success / failure summary for Phase 1
    if phase1_results:
        p1_ok = sum(1 for v in phase1_results.values()
                    if not (isinstance(v, dict) and "error" in v))
        p1_fail = len(phase1_results) - p1_ok
        logger.info("  Phase 1 tasks   : %d OK / %d failed", p1_ok, p1_fail)

    # Success / failure summary for Phase 2
    if phase2_results:
        p2_entries = {k: v for k, v in phase2_results.items() if k != "5shot"}
        p2_ok = sum(1 for v in p2_entries.values()
                    if not (isinstance(v, dict) and "error" in v))
        p2_fail = len(p2_entries) - p2_ok
        logger.info("  Phase 2 tasks   : %d OK / %d failed", p2_ok, p2_fail)

    logger.info(_bar())


if __name__ == "__main__":
    main()

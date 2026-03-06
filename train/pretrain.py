"""
train/pretrain.py — Main pretraining entry point.

Launch single-GPU:
    python train/pretrain.py --config configs/small.yaml --train_data data/train.bin

Launch multi-GPU with torchrun:
    torchrun --nproc_per_node=8 train/pretrain.py --config configs/small.yaml \
        --train_data data/train.bin

The script auto-detects whether it is running inside a torchrun launch by
checking for the RANK environment variable.
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

# B200 Tensor Core 최대 활용: TF32 matmul + cuDNN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # fixed seq_len=4096 → safe to auto-tune
torch.set_float32_matmul_precision("high")  # TF32 precision for fp32 matmul

# Allow imports from the project root regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import PackedDataset
from model import LLM, LMConfig
from train.trainer import TrainConfig, Trainer
from train.utils import (
    cleanup_ddp,
    get_cosine_schedule_with_warmup,
    is_main_process,
    load_checkpoint,
    setup_ddp,
)

# ---------------------------------------------------------------------------
# Optional TransformerEngine import (FP8 support)
# ---------------------------------------------------------------------------
try:
    import transformer_engine.pytorch as te  # type: ignore[import]
    HAS_TE = True
except ImportError:
    te = None  # type: ignore[assignment]
    HAS_TE = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a decoder-only LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/small.yaml"),
        help="Path to the LMConfig YAML file.",
    )
    parser.add_argument(
        "--train_data",
        type=Path,
        required=True,
        help="Path to the training data .bin file (numpy uint16 memmap).",
    )
    parser.add_argument(
        "--val_data",
        type=Path,
        default=None,
        help="Optional path to validation data .bin file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for saving checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )

    # Training hyper-parameters
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override the number of optimiser steps (default: TrainConfig.max_steps).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-GPU micro-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW weight decay coefficient.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of linear warmup steps.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (rank offset is added automatically).",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to a text file for structured training logs (rank-0 only). "
             "If omitted, logs go only to stdout.",
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        default=False,
        help="Enable TransformerEngine FP8 training (overrides config; requires B200/H100).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer parameter groups
# ---------------------------------------------------------------------------


def build_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Split parameters into two groups:
      - decay group  : weight tensors with ndim >= 2
      - no-decay group: bias, LayerNorm/RMSNorm weights, and embedding weights

    This follows standard practice (e.g. GPT-style training).
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    # Names of module types whose parameters should never be decayed.
    no_decay_module_types = (
        torch.nn.Embedding,
        torch.nn.LayerNorm,
    )
    # Also skip any parameter whose name ends with '.bias' or 'norm'.
    # Mamba-2 SSM parameters that should never be decayed
    no_decay_name_suffixes = ("bias", "A_log", "D", "dt_bias")

    # Collect module-level exclusions.
    no_decay_module_params: set[int] = set()
    for module in model.modules():
        if isinstance(module, no_decay_module_types):
            for param in module.parameters(recurse=False):
                no_decay_module_params.add(id(param))

    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        if (
            id(param) in no_decay_module_params
            or any(name.endswith(sfx) for sfx in no_decay_name_suffixes)
            or param.ndim < 2
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ---- Distributed setup -------------------------------------------------
    is_ddp = "RANK" in os.environ
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_ddp:
        rank, local_rank, world_size, device = setup_ddp()

    # Per-rank seed so data shuffling differs across replicas.
    set_seed(args.seed + rank)

    # ---- NUMA affinity for optimal GPU↔CPU memory locality ---------------
    # B200 topology: GPU 0-3 → NUMA node 0 (cores 0-35)
    #                GPU 4-6 → NUMA node 1 (cores 36-71)
    # Without pinning, ranks may end up on wrong NUMA → 3.2x memory latency.
    try:
        if local_rank < 4:
            os.sched_setaffinity(0, set(range(0, 36)))   # NUMA node 0
        else:
            os.sched_setaffinity(0, set(range(36, 72)))   # NUMA node 1
        if is_main_process():
            print(f"NUMA affinity: rank {rank} (GPU {local_rank}) → "
                  f"{'NUMA0 cores 0-35' if local_rank < 4 else 'NUMA1 cores 36-71'}")
    except (AttributeError, OSError) as e:
        if is_main_process():
            print(f"[WARN] NUMA affinity failed: {e}")

    # ---- Model -------------------------------------------------------------
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    lm_config = LMConfig.from_yaml(args.config)

    # CLI --use_fp8 flag overrides whatever the config file says.
    if args.use_fp8:
        lm_config.use_fp8 = True

    # FP8 alignment check: (batch_size × seq_len) must be divisible by 8.
    if lm_config.use_fp8 and (args.batch_size * lm_config.max_seq_len) % 8 != 0:
        raise ValueError(
            f"FP8: batch_size × max_seq_len = {args.batch_size} × {lm_config.max_seq_len} "
            f"= {args.batch_size * lm_config.max_seq_len} must be divisible by 8."
        )

    # Note: fp8_model_init() is intentionally omitted — MXFP8Tensor weights are
    # incompatible with DDP's _broadcast_coalesced during multi-GPU init.
    # Weights remain in float32; TE quantizes on-the-fly inside fp8_autocast.
    model = LLM(lm_config).to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"LMConfig: {lm_config}")
        if lm_config.use_hybrid:
            m_count = sum(1 for lt in model._layer_types if lt == "M")
            a_count = sum(1 for lt in model._layer_types if lt == "A")
            a_pos = [i for i, lt in enumerate(model._layer_types) if lt == "A"]
            mamba_p = sum(
                p.numel() for l, lt in zip(model.layers, model._layer_types)
                if lt == "M" for p in l.parameters()
            )
            attn_p = sum(
                p.numel() for l, lt in zip(model.layers, model._layer_types)
                if lt == "A" for p in l.parameters()
            )
            print(f"  Hybrid: {m_count}M + {a_count}A layers")
            print(f"  Mamba params: {mamba_p:,} | Attn params: {attn_p:,}")
            print(f"  Attn positions: {a_pos}")
            print(f"  mamba_d_ffn={lm_config.mamba_d_ffn}, n_groups={lm_config.mamba_n_groups}")

    # ---- Wrap in DDP -------------------------------------------------------
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,   # zero-copy gradient → NCCL buffer
            bucket_cap_mb=800,              # larger buckets for NVLS (was 400)
            find_unused_parameters=False,   # fixed graph, no traversal overhead
            # NOTE: static_graph=True 제거 — TE FP8 레이어의 동적 autograd hooks와 충돌
        )

    # ---- Dataset & DataLoader ----------------------------------------------
    # PackedDataset: non-overlapping stride=seq_len windows.
    # Avoids 600M random-index mmap accesses from stride-1 TextDataset.
    train_dataset = PackedDataset(args.train_data, seq_len=lm_config.max_seq_len)

    if is_ddp:
        train_sampler: DistributedSampler | RandomSampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        shuffle = False
    else:
        train_sampler = RandomSampler(train_dataset)
        shuffle = False  # Sampler is provided; DataLoader must not also shuffle.

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=6,           # 6×8=48 workers, fits 72-core budget with OMP=4
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,       # deeper pipeline for larger worker pool
        persistent_workers=True, # keep workers alive across epochs — eliminates respawn stall
    )

    # ---- Optimizer ---------------------------------------------------------
    param_groups = build_optimizer_param_groups(
        getattr(model, "module", model), args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),  # Use fused kernel when on CUDA.
    )

    # ---- LR Scheduler ------------------------------------------------------
    train_config = TrainConfig(
        checkpoint_dir=str(args.checkpoint_dir),
        grad_accum_steps=args.grad_accum,
        use_fp8=lm_config.use_fp8,
        log_file=str(args.log_file) if args.log_file is not None else None,
    )
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=train_config.max_steps,
    )

    # ---- Resume from checkpoint --------------------------------------------
    start_step = 0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {args.resume}")
        start_step, resume_loss = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if is_main_process():
            print(f"Resumed from {args.resume} at step {start_step} (loss={resume_loss:.4f})")

    # ---- Checkpoint directory ----------------------------------------------
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Trainer -----------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config,
        device=device,
        rank=rank,
        sampler=train_sampler if is_ddp else None,
    )

    # ---- Signal handlers for graceful shutdown ----------------------------
    # SIGHUP: SSH 세션 끊김 시 발생 → 이전에 학습을 죽인 주범
    # SIGTERM: kill 명령 또는 시스템 종료 시 발생
    # 핸들러가 trainer.request_shutdown()을 호출하면, 학습 루프가
    # 현재 step 완료 후 비상 체크포인트를 저장하고 깨끗하게 종료합니다.
    _trainer_ref = trainer

    def _graceful_shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        if is_main_process():
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"[{ts}] [SIGNAL] Received {sig_name} (signum={signum}). "
                f"Initiating graceful shutdown..."
            )
            print(f"\n{msg}")
            # 로그 파일에도 즉시 기록 (시그널 핸들러 내에서 안전하게)
            if args.log_file is not None:
                try:
                    with open(args.log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass  # 시그널 핸들러 내에서는 예외 무시
        _trainer_ref.request_shutdown(sig_name)

    for _sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(_sig, _graceful_shutdown_handler)

    if is_main_process():
        import datetime
        eff_tokens_per_step = args.batch_size * lm_config.max_seq_len * args.grad_accum * world_size
        nccl_debug = os.environ.get("NCCL_DEBUG", "not set")
        omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
        print(
            f"\n{'='*70}\n"
            f"  LLM Pretraining — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}\n"
            f"  model     : {lm_config.num_params:,} params  |  "
            f"d_model={lm_config.d_model}  n_layers={lm_config.n_layers}\n"
            f"  precision : {'FP8 (MXFP8BlockScaling)' if lm_config.use_fp8 else 'BF16'}\n"
            f"  GPUs      : {world_size}  |  batch/GPU={args.batch_size}  "
            f"grad_accum={args.grad_accum}\n"
            f"  eff_batch : {args.batch_size * args.grad_accum * world_size} seqs  "
            f"= {eff_tokens_per_step:,} tok/step\n"
            f"  max_steps : {train_config.max_steps:,}  "
            f"({train_config.max_steps * eff_tokens_per_step / 1e9:.1f}B tokens total)\n"
            f"  data      : {args.train_data}\n"
            f"  ckpt_dir  : {args.checkpoint_dir}\n"
            f"  env       : OMP_NUM_THREADS={omp_threads}  NCCL_DEBUG={nccl_debug}\n"
            f"{'='*70}\n"
        )

    try:
        trainer.train(start_step=start_step)
        # 학습 완료 또는 graceful shutdown 후 상태 출력
        if is_main_process():
            if trainer._shutdown_requested:
                print(
                    f"\n[INFO] Training gracefully shut down via {trainer._shutdown_signal}. "
                    f"Emergency checkpoint saved. Resume with same command."
                )
            else:
                print("\n[INFO] Training completed successfully.")
    except KeyboardInterrupt:
        if is_main_process():
            print("\n[INFO] Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        import traceback
        if is_main_process():
            tb = traceback.format_exc()
            print(f"\n[ERROR] Training failed at rank {rank}:\n{tb}")
            # log_file에도 기록
            if args.log_file is not None:
                with open(args.log_file, "a", encoding="utf-8") as f:
                    import datetime
                    f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [FATAL] {tb}\n")
        raise
    finally:
        if is_ddp:
            cleanup_ddp()

    # Note: DDP cleanup is handled in the try/finally block above.


if __name__ == "__main__":
    main()

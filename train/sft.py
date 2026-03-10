"""
train/sft.py — Supervised Fine-Tuning (SFT) entry point.

Loads a pretrained checkpoint and fine-tunes it on instruction/conversation
data using SFTDataset, which masks prompt tokens with ignore_index=-1 so only
the assistant response tokens contribute to the loss.

Launch single-GPU:
    python train/sft.py \\
        --base_checkpoint checkpoints/korean_1b_fp8_run1/checkpoint-0034000 \\
        --sft_data data/sft/train.jsonl \\
        --device cuda:0

Launch multi-GPU (DDP via torchrun, 7 GPU):
    torchrun --nproc_per_node=7 train/sft.py \\
        --base_checkpoint checkpoints/3b_final/checkpoint-0319772 \\
        --sft_data data/sft_combined/train_filtered.jsonl

KEY DIFFERENCES from pretrain.py:
  - Loads weights from a pretrained checkpoint via LLM.from_pretrained()
  - Uses SFTDataset (JSONL instruction data) instead of PackedDataset
  - Lower default learning rate (2e-5 vs 2e-4)
  - Fewer default steps (3000 vs 100000)
  - Copies tokenizer.json to checkpoint_dir for easy deployment
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

# B200 Tensor Core 최대 활용: TF32 matmul + cuDNN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # TF32 precision for fp32 matmul

# Allow imports from the project root regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import LLM
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
        description="Supervised Fine-Tuning (SFT) of a pretrained decoder-only LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required paths -----------------------------------------------------
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the pretrained checkpoint directory. "
            "Must contain model.pt and config.yaml (produced by save_checkpoint)."
        ),
    )
    parser.add_argument(
        "--sft_data",
        type=Path,
        required=True,
        help="Path to the JSONL SFT training data file.",
    )

    # --- Optional paths -----------------------------------------------------
    parser.add_argument(
        "--val_data",
        type=Path,
        default=None,
        help="Optional path to JSONL SFT validation data file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/korean_1b_sft"),
        help="Root directory for saving SFT checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an SFT checkpoint directory to resume fine-tuning from.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help=(
            "Override path to tokenizer.json. "
            "Defaults to <base_checkpoint>/tokenizer.json, "
            "then falls back to tokenizer/korean_sp/tokenizer.json."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help=(
            "Path to a text file for structured training logs (rank-0 only). "
            "If omitted, logs go only to stdout."
        ),
    )

    # --- Training hyper-parameters ------------------------------------------
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3000,
        help="Total number of optimiser steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-GPU micro-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help=(
            "Peak learning rate. "
            "SFT uses a much lower lr than pretraining (2e-5 vs 2e-4) "
            "to preserve pretrained representations."
        ),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay. Lower than pretrain (0.01 vs 0.1).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of linear LR warmup steps.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (rank offset is added automatically in DDP).",
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        default=False,
        help=(
            "Enable TransformerEngine FP8 training "
            "(requires B200/H100, uses MXFP8BlockScaling)."
        ),
    )

    # --- Single-GPU device override (ignored when using torchrun) -----------
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Explicit device string (e.g. 'cuda:0'). "
            "Ignored when running under torchrun (DDP auto-assigns devices)."
        ),
    )

    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file. Values under 'train:' section are used as CLI defaults.",
    )
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval (steps).")
    parser.add_argument("--eval_interval", type=int, default=250, help="Validation eval interval (steps).")
    parser.add_argument("--neftune_alpha", type=float, default=5.0, help="NEFTune noise magnitude (0 to disable).")

    # First pass: just get --config
    args, remaining = parser.parse_known_args()

    # Load YAML config and apply values as defaults
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        import yaml
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        train_section = yaml_cfg.get("train", {})
        yaml_to_arg = {
            "max_steps": "max_steps",
            "batch_size": "batch_size",
            "lr": "lr",
            "weight_decay": "weight_decay",
            "warmup_steps": "warmup_steps",
            "grad_accum_steps": "grad_accum",
            "save_interval": "save_interval",
            "eval_interval": "eval_interval",
            "neftune_alpha": "neftune_alpha",
        }
        new_defaults = {}
        for yaml_key, arg_name in yaml_to_arg.items():
            if yaml_key in train_section:
                new_defaults[arg_name] = train_section[yaml_key]
        if new_defaults:
            parser.set_defaults(**new_defaults)

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
# (Copied from pretrain.py to avoid circular import; identical logic)
# ---------------------------------------------------------------------------


def build_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Split parameters into two groups:
      - decay group   : weight tensors with ndim >= 2 (Linear, etc.)
      - no-decay group: bias, LayerNorm/RMSNorm weights, and embedding weights

    This follows standard practice (e.g. GPT-style training).
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    # Module types whose parameters should never be decayed.
    no_decay_module_types = (
        torch.nn.Embedding,
        torch.nn.LayerNorm,
    )
    # Also skip any parameter whose name ends with '.bias'.
    no_decay_name_suffixes = ("bias",)

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
# Tokenizer resolution helper
# ---------------------------------------------------------------------------


def _resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    """
    Determine the tokenizer path in priority order:
      1. Explicit --tokenizer argument
      2. tokenizer.json inside the base_checkpoint directory
      3. Project default: tokenizer/korean_sp/tokenizer.json
    """
    if args.tokenizer is not None:
        p = Path(args.tokenizer)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer not found at --tokenizer path: {p}")
        return p

    ckpt_tok = args.base_checkpoint / "tokenizer.json"
    if ckpt_tok.exists():
        return ckpt_tok

    default_tok = _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
    if default_tok.exists():
        return default_tok

    raise FileNotFoundError(
        "Could not locate tokenizer.json. Tried:\n"
        f"  1. {ckpt_tok}\n"
        f"  2. {default_tok}\n"
        "Use --tokenizer to specify an explicit path."
    )


# ---------------------------------------------------------------------------
# Dynamic padding collate function
# ---------------------------------------------------------------------------


def dynamic_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function that pads each batch to its own maximum sequence length
    instead of a fixed global max_seq_len.  This reduces wasted FLOPs on
    short sequences and speeds up SFT which tends to have highly variable
    response lengths.

    Pads to the batch-local max, aligned to 64 tokens (for Flash Attention
    efficiency), with a floor of 512 tokens so micro-batches are not too short.

    Args:
        batch: List of ``(input_ids, labels)`` tuples from SFTDataset.

    Returns:
        Tuple of ``(input_ids, labels, attention_mask)`` tensors shaped
        ``[B, max_len]``.
        ``input_ids``      is right-padded with 0 (pad token).
        ``labels``         is right-padded with -1 (cross-entropy ignore_index).
        ``attention_mask`` is 1 for real tokens, 0 for padding.
    """
    # 64-token alignment + minimum 512 floor
    raw_max = max(item[0].size(0) for item in batch)
    max_len = max(512, ((raw_max + 63) // 64) * 64)

    input_ids_list, labels_list, mask_list = [], [], []
    for ids, labs in batch:
        pad_len = max_len - ids.size(0)
        input_ids_list.append(F.pad(ids, (0, pad_len), value=0))
        labels_list.append(F.pad(labs, (0, pad_len), value=-1))
        mask_list.append(
            F.pad(torch.ones(ids.size(0), dtype=torch.long), (0, pad_len), value=0)
        )

    return (
        torch.stack(input_ids_list),
        torch.stack(labels_list),
        torch.stack(mask_list),
    )


# ---------------------------------------------------------------------------
# NEFTune helper
# ---------------------------------------------------------------------------


def add_neftune_hook(model: torch.nn.Module, noise_alpha: float = 10.0):
    """
    Register a forward hook on the model's input embedding layer that adds
    uniform noise scaled by noise_alpha during training (NEFTune).

    Reference: "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
    (Jain et al., 2023). https://arxiv.org/abs/2310.05914

    Args:
        model:       Raw (non-DDP) model instance.
        noise_alpha: Noise magnitude parameter (paper default: 10).

    Returns:
        The hook handle (call ``handle.remove()`` to deactivate), or None if
        the embedding layer could not be located.
    """
    # Unwrap DDP if needed
    raw = model.module if hasattr(model, "module") else model

    # 1) Try the standard HuggingFace accessor first.
    embedding: torch.nn.Embedding | None = None
    if hasattr(raw, "get_input_embeddings"):
        try:
            emb = raw.get_input_embeddings()
            if isinstance(emb, torch.nn.Embedding):
                embedding = emb
        except Exception:
            pass

    # 2) Fallback: walk common attribute paths found in open-source LLMs.
    if embedding is None:
        for attr_path in [
            "embedding",
            "embed_tokens",
            "token_embedding",
            "wte",
            "word_embeddings",
            "tok_embeddings",
            "transformer.wte",
            "model.embed_tokens",
            "model.embedding",
        ]:
            obj = raw
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, torch.nn.Embedding):
                embedding = obj
                break

    if embedding is None:
        print("[WARN] NEFTune: embedding layer을 찾지 못함, NEFTune 비활성화")
        return None

    print(
        f"[INFO] NEFTune: {type(embedding).__name__} hook 등록 "
        f"(shape={tuple(embedding.weight.shape)}, alpha={noise_alpha})"
    )

    def _hook(
        module: torch.nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if module.training:
            # out shape: [B, seq_len, d_model]
            mag = noise_alpha / ((out.size(1) * out.size(2)) ** 0.5)
            out = out + torch.empty_like(out).uniform_(-mag, mag)
        return out

    return embedding.register_forward_hook(_hook)


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

    if is_ddp:
        rank, local_rank, world_size, device = setup_ddp()
    else:
        # Single-GPU: honour --device flag, else pick cuda:0 or cpu.
        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # Per-rank seed so data shuffling differs across replicas.
    set_seed(args.seed + rank)

    # ---- NUMA affinity for optimal GPU↔CPU memory locality ---------------
    # B200 topology: GPU 0-3 → NUMA node 0 (cores 0-35)
    #                GPU 4-6 → NUMA node 1 (cores 36-71)  [7 GPU 환경]
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

    # ---- Validate base checkpoint ------------------------------------------
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint directory not found: {args.base_checkpoint}"
        )
    for required_file in ("model.pt", "config.yaml"):
        if not (args.base_checkpoint / required_file).exists():
            raise FileNotFoundError(
                f"Expected {required_file} inside base checkpoint: {args.base_checkpoint}"
            )

    # ---- Load pretrained model ---------------------------------------------
    # LLM.from_pretrained() reads config.yaml + model.pt and returns the model on CPU.
    # We move it to the target device immediately after loading.
    #
    # NOTE: fp8_model_init() is intentionally NOT used here (same as pretrain.py).
    # MXFP8Tensor weights are incompatible with DDP's _broadcast_coalesced.
    # Weights stay in float32; TransformerEngine quantizes on-the-fly inside fp8_autocast.
    model = LLM.from_pretrained(args.base_checkpoint)

    # When FP8 flag is passed at SFT time, enable it on the loaded config.
    # This is useful if the pretrained model was trained without FP8 but you
    # want to fine-tune with FP8 precision (the TE layers must exist in the model).
    if args.use_fp8:
        model.config.use_fp8 = True

    # Move model to target device in bfloat16 (more memory-efficient than fp32
    # for fine-tuning, and required when BF16 autocast + TE are active).
    model = model.to(device=device, dtype=torch.bfloat16)

    # ---- Gradient checkpointing ----------------------------------------
    # Trades activation memory for recomputation during backward pass.
    # Especially useful for large models / long sequences in SFT.
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("[INFO] Gradient checkpointing enabled")

    # FP8 alignment check: (batch_size × seq_len) must be divisible by 8.
    if model.config.use_fp8:
        seq_len = model.config.max_seq_len
        if (args.batch_size * seq_len) % 8 != 0:
            raise ValueError(
                f"FP8: batch_size × max_seq_len = {args.batch_size} × {seq_len} "
                f"= {args.batch_size * seq_len} must be divisible by 8."
            )

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Pretrained model loaded: {total_params:,} parameters")
        print(f"LMConfig: {model.config}")

    # ---- Wrap in DDP -------------------------------------------------------
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            bucket_cap_mb=800,
            find_unused_parameters=False,
        )

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer_path = _resolve_tokenizer_path(args)
    if is_main_process():
        print(f"Loading tokenizer from: {tokenizer_path}")

    # Use the fast tokenizers library (same as the rest of the project).
    from tokenizers import Tokenizer  # type: ignore[import]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ---- Dataset & DataLoader ----------------------------------------------
    # Import SFTDataset (created separately alongside this file).
    # SFTDataset returns (input_ids, targets) where prompt token positions in
    # targets are filled with -1.  The Trainer._compute_loss already uses
    # ignore_index=-1, so only response tokens contribute to the gradient.
    from data.sft_dataset import SFTDataset  # type: ignore[import]

    train_dataset = SFTDataset(
        data_path=args.sft_data,
        tokenizer=tokenizer,
        max_seq_len=model.config.max_seq_len
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.module.config.max_seq_len,
    )

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
        # SFT datasets are typically small enough that 2–4 workers suffice.
        # We use 4 to balance I/O with CPU parsing overhead from JSONL.
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=dynamic_collate_fn,
    )

    # Optional validation loader.
    # NOTE: The current Trainer implementation does not yet accept a val_loader
    # argument; the eval_interval config field is reserved for future use.
    # We construct the loader here so that once Trainer gains eval support,
    # wiring it in requires only passing val_loader=val_loader below.
    val_loader: DataLoader | None = None
    if args.val_data is not None:
        if not args.val_data.exists():
            raise FileNotFoundError(f"Validation data not found: {args.val_data}")
        val_dataset = SFTDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=train_dataset.max_seq_len,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=dynamic_collate_fn,
        )
        if is_main_process():
            print(f"Validation dataset: {len(val_dataset):,} samples")

    # ---- Optimizer ---------------------------------------------------------
    # Use the same two-group split (weight_decay / no weight_decay) as pretrain.
    # Unwrap DDP to get the raw model's parameters.
    raw_model = getattr(model, "module", model)
    param_groups = build_optimizer_param_groups(raw_model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),  # Use fused kernel when on CUDA.
    )

    # ---- TrainConfig -------------------------------------------------------
    # Set use_fp8 from the (possibly overridden) model config so Trainer builds
    # the correct FP8 recipe and wraps forward passes in fp8_autocast.
    use_fp8 = raw_model.config.use_fp8

    train_config = TrainConfig(
        max_steps=args.max_steps,
        checkpoint_dir=str(args.checkpoint_dir),
        grad_accum_steps=args.grad_accum,
        use_fp8=use_fp8,
        log_file=str(args.log_file) if args.log_file is not None else None,
        save_interval=args.save_interval,
        log_interval=10,
        eval_interval=args.eval_interval,
    )

    # ---- LR Scheduler ------------------------------------------------------
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=train_config.max_steps,
    )

    # ---- Resume from SFT checkpoint ----------------------------------------
    # When --resume is given we restore the SFT optimizer/scheduler state as
    # well so learning rate, momentum buffers, etc. are correctly restored.
    # NOTE: This resumes SFT training, NOT the pretrain checkpoint.
    #       The pretrain weights were already loaded above via from_pretrained().
    start_step = 0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_step, resume_loss = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if is_main_process():
            print(f"Resumed SFT from {args.resume} at step {start_step} (loss={resume_loss:.4f})")

    if args.resume is not None and isinstance(train_sampler, DistributedSampler):
        steps_per_epoch = len(train_loader)
        approx_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
        train_sampler.set_epoch(approx_epoch)
        if is_main_process():
            print(f"[INFO] Resume: sampler epoch set to {approx_epoch}")

    # ---- Checkpoint directory ----------------------------------------------
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Copy tokenizer to checkpoint dir for easy deployment later --------
    # This mirrors the tokenizer into the SFT checkpoint root so that the
    # final checkpoint directory is self-contained for convert_to_hf.py, etc.
    if is_main_process():
        dest_tok = args.checkpoint_dir / "tokenizer.json"
        if not dest_tok.exists():
            shutil.copy2(str(tokenizer_path), str(dest_tok))
            print(f"Tokenizer copied to {dest_tok}")

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
        val_loader=val_loader,
    )

    # ---- Signal handlers for graceful shutdown ----------------------------
    import signal as _signal_mod

    _trainer_ref = trainer

    def _graceful_shutdown_handler(signum, frame):
        sig_name = _signal_mod.Signals(signum).name
        if is_main_process():
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"[{ts}] [SIGNAL] Received {sig_name} (signum={signum}). "
                f"Initiating graceful shutdown..."
            )
            print(f"\n{msg}")
            if args.log_file is not None:
                try:
                    with open(args.log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass
        _trainer_ref.request_shutdown(sig_name)

    for _sig in (_signal_mod.SIGHUP, _signal_mod.SIGTERM):
        _signal_mod.signal(_sig, _graceful_shutdown_handler)

    # ---- SFT banner --------------------------------------------------------
    if is_main_process():
        import datetime

        inner_config = raw_model.config
        eff_batch_seqs = args.batch_size * args.grad_accum * world_size
        eff_tokens_per_step = eff_batch_seqs * inner_config.max_seq_len
        train_samples = len(train_dataset)
        precision_label = "FP8 (MXFP8BlockScaling)" if use_fp8 else "BF16"
        nccl_debug = os.environ.get("NCCL_DEBUG", "not set")
        omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")

        print(
            f"\n{'='*70}\n"
            f"  LLM Supervised Fine-Tuning — "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}\n"
            f"  base ckpt : {args.base_checkpoint}\n"
            f"  sft data  : {args.sft_data} ({train_samples:,} samples)\n"
            f"  model     : {inner_config.num_params:,} params  |  "
            f"d_model={inner_config.d_model}  n_layers={inner_config.n_layers}\n"
            f"  precision : {precision_label}\n"
            f"  GPUs      : {world_size}  |  batch/GPU={args.batch_size}  "
            f"grad_accum={args.grad_accum}\n"
            f"  eff_batch : {eff_batch_seqs} seqs  "
            f"= {eff_tokens_per_step:,} tok/step\n"
            f"  max_steps : {train_config.max_steps:,}\n"
            f"  lr        : {args.lr:.2e}  "
            f"warmup={args.warmup_steps}  weight_decay={args.weight_decay}\n"
            f"  ckpt_dir  : {args.checkpoint_dir}\n"
            f"  env       : OMP_NUM_THREADS={omp_threads}  NCCL_DEBUG={nccl_debug}\n"
            f"{'='*70}\n"
        )

    # ---- NEFTune -----------------------------------------------------------
    # Add uniform noise to embeddings during training to improve instruction
    # following (Jain et al., 2023).  Hook is registered on the raw (non-DDP)
    # model so it survives DDP's internal module wrapping.
    neftune_alpha = getattr(args, 'neftune_alpha', 5.0)
    neftune_handle = add_neftune_hook(raw_model, noise_alpha=neftune_alpha)
    if rank == 0:
        if neftune_handle is not None:
            print(f"[INFO] NEFTune enabled (noise_alpha={neftune_alpha})")
        else:
            print("[WARN] NEFTune disabled - embedding layer not found")

    # ---- Train -------------------------------------------------------------
    try:
        trainer.train(start_step=start_step)
    except KeyboardInterrupt:
        if is_main_process():
            print("\n[INFO] SFT interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        import traceback
        if is_main_process():
            tb = traceback.format_exc()
            print(f"\n[ERROR] SFT failed at rank {rank}:\n{tb}")
            if args.log_file is not None:
                with open(args.log_file, "a", encoding="utf-8") as f:
                    import datetime
                    f.write(
                        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[FATAL] {tb}\n"
                    )
        raise
    finally:
        # Remove NEFTune hook so the model is clean for inference/saving.
        if neftune_handle is not None:
            neftune_handle.remove()
        if is_ddp:
            cleanup_ddp()


if __name__ == "__main__":
    main()

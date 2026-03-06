"""
train/trainer.py — Core training loop.

Provides:
    TrainConfig : Dataclass of all training hyper-parameters.
    Trainer     : Orchestrates gradient accumulation, AMP, gradient clipping,
                  tensorboard logging, and checkpoint saving.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, AttributeError):
    SummaryWriter = None  # type: ignore[misc,assignment]
    HAS_TENSORBOARD = False

from train.utils import get_grad_norm, is_main_process, save_checkpoint


# ---------------------------------------------------------------------------
# Optional TransformerEngine import (FP8 support)
# ---------------------------------------------------------------------------
try:
    import transformer_engine.pytorch as te  # type: ignore[import]
    from transformer_engine.common.recipe import DelayedScaling, Format  # type: ignore[import]
    HAS_TE = True
except ImportError:
    te = None  # type: ignore[assignment]
    HAS_TE = False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyper-parameters that control the training loop."""

    # Total number of optimiser update steps.
    max_steps: int = 100_000

    # Number of forward passes accumulated before each optimiser step.
    grad_accum_steps: int = 1

    # Maximum global gradient L2 norm; clips if exceeded (0 = disabled).
    max_grad_norm: float = 1.0

    # Log training metrics every this many *optimiser* steps.
    log_interval: int = 10

    # Save a checkpoint every this many optimiser steps.
    save_interval: int = 1000

    # Run validation (if val_loader provided) every this many optimiser steps.
    eval_interval: int = 500

    # Root directory where checkpoint sub-folders are written.
    checkpoint_dir: str = "checkpoints"

    # Use bf16 autocast during the forward pass (no GradScaler needed for bf16).
    use_amp: bool = True

    # Pass model through torch.compile() before training.
    compile_model: bool = False

    # FP8 (TransformerEngine) settings — only relevant when use_fp8=True.
    use_fp8: bool = False
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"   # "max" | "most_recent"
    fp8_format: str = "MXFP8"            # "MXFP8" (B200 block scaling) | "HYBRID" (E4M3+E5M2)

    # Path to a text log file (rank-0 only). None = stdout only.
    log_file: Optional[str] = None

    # grad_norm을 파일에 기록하는 간격 (0=비활성)
    log_grad_norm_interval: int = 100
    # GPU 메모리를 파일에 기록하는 간격 (0=비활성)
    log_memory_interval: int = 100


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Manages the full pretraining loop for a decoder-only LLM.

    Supports:
        - Gradient accumulation over ``config.grad_accum_steps`` micro-batches.
        - bf16 mixed-precision via ``torch.autocast`` (no GradScaler required).
        - Global gradient norm clipping.
        - Tensorboard logging on the main process.
        - Periodic checkpoint saving via :func:`train.utils.save_checkpoint`.
        - Optional ``torch.compile`` acceleration.

    Args:
        model:        The LLM (plain ``nn.Module`` or DDP-wrapped).
        train_loader: DataLoader yielding ``(input_ids, targets)`` batches.
        optimizer:    AdamW (or any ``Optimizer``) configured externally.
        scheduler:    LR scheduler produced by the caller.
        config:       ``TrainConfig`` instance controlling all loop behaviour.
        device:       Target device for data and model.
        rank:         Process rank (used to suppress logging on non-main ranks).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        config: TrainConfig,
        device: torch.device,
        rank: int = 0,
        sampler: Optional[DistributedSampler] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.rank = rank
        self._is_main = is_main_process()
        self._sampler = sampler   # for set_epoch() on each data pass
        self._epoch = 0
        self._val_loader = val_loader
        self._best_val_loss: float = float("inf")
        self._val_patience_counter: int = 0
        self._val_patience_limit: int = 5  # early stopping patience

        # Graceful shutdown support — signal handler에서 flag 설정,
        # 학습 루프가 각 step 완료 후 확인하여 비상 체크포인트 저장 후 종료
        self._shutdown_requested = False
        self._shutdown_signal = ""

        # Build FP8 recipe once (reused every step) ----------------------
        self._fp8_recipe = None
        if config.use_fp8 and HAS_TE:
            if config.fp8_format == "MXFP8":
                from transformer_engine.common.recipe import MXFP8BlockScaling  # type: ignore[import]
                self._fp8_recipe = MXFP8BlockScaling()
            else:
                self._fp8_recipe = DelayedScaling(
                    fp8_format=getattr(Format, config.fp8_format),
                    amax_history_len=config.fp8_amax_history_len,
                    amax_compute_algo=config.fp8_amax_compute_algo,
                )

        # Optionally compile the model (unwrap DDP first to compile the inner module).
        if config.compile_model:
            inner: nn.Module = getattr(self.model, "module", self.model)
            compiled = torch.compile(inner)
            if hasattr(self.model, "module"):
                self.model.module = compiled  # type: ignore[assignment]
            else:
                self.model = compiled  # type: ignore[assignment]

        # Tensorboard writer — only on rank 0.
        self._writer: Optional[SummaryWriter] = None
        self._log_fh = None  # optional file handle for structured text log
        if self._is_main:
            if HAS_TENSORBOARD:
                log_dir = Path(config.checkpoint_dir) / "tensorboard"
                self._writer = SummaryWriter(log_dir=str(log_dir))
            if config.log_file is not None:
                Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_fh = open(config.log_file, "a", encoding="utf-8", buffering=1)

        # 학습 시작 시각 기록 (통계 요약 로그에 사용)
        import datetime
        self._train_start_time = datetime.datetime.now()

        # Infinite iterator over the DataLoader.
        self._loader_iter = iter(self.train_loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_shutdown(self, signal_name: str = "UNKNOWN") -> None:
        """Request graceful shutdown after the current training step.

        Called from signal handlers (SIGHUP, SIGTERM). Sets a flag
        that the training loop checks after each optimizer step.
        The loop will save an emergency checkpoint and exit cleanly.
        """
        self._shutdown_requested = True
        self._shutdown_signal = signal_name

    def train(self, start_step: int = 0) -> None:
        """
        Run the main training loop from ``start_step`` to ``config.max_steps``.

        Args:
            start_step: First optimiser step index (non-zero when resuming).
        """
        cfg = self.config
        model = self.model

        model.train()

        # Timing state for tokens/sec estimation.
        t0 = time.perf_counter()
        running_loss = 0.0
        log_step_count = 0
        accum_loss = torch.tensor(0.0, device=self.device)  # initialise so end-of-training save is safe on empty loops

        for step in range(start_step, cfg.max_steps):
            # ---- Gradient accumulation loop --------------------------------
            self.optimizer.zero_grad(set_to_none=True)
            # Accumulate loss on GPU to avoid one GPU-CPU sync per micro-step.
            accum_loss = torch.zeros(1, device=self.device)

            for micro_step in range(cfg.grad_accum_steps):
                batch = self._next_batch()
                # Suppress DDP all-reduce on all but the last micro-step (Bug 3).
                is_last_micro = micro_step == cfg.grad_accum_steps - 1
                sync_ctx = (
                    contextlib.nullcontext()
                    if not isinstance(model, DDP) or is_last_micro
                    else model.no_sync()
                )
                try:
                    with sync_ctx:
                        micro_loss = self._step(batch)  # returns detached GPU tensor
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    raise RuntimeError(
                        f"CUDA OOM at step {step}, micro_step {micro_step}. "
                        f"GPU mem: {mem_alloc:.1f}/{mem_total:.1f} GB. "
                        f"Try reducing batch_size or grad_accum_steps."
                    ) from e
                except RuntimeError as e:
                    self._log(f"RuntimeError at step {step}, micro_step {micro_step}: {e}", level="ERROR")
                    raise
                accum_loss += micro_loss  # GPU-side accumulation, no CPU sync

            # Single GPU-CPU sync per optimizer step (was one sync per micro-step).
            avg_loss = accum_loss.item() / cfg.grad_accum_steps

            # Detect NaN/Inf loss — indicates numerical instability.
            if not math.isfinite(avg_loss):
                mem_gb = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                raise RuntimeError(
                    f"Non-finite loss detected: {avg_loss}. "
                    f"GPU mem: {mem_gb:.1f}/{mem_total:.1f} GB. "
                    f"Check lr, grad clipping, FP8 amax history. "
                    f"Try: lower lr, increase fp8_amax_history_len, or switch to BF16."
                )

            # ---- Gradient clipping -----------------------------------------
            # clip_grad_norm_ already computes the global norm internally.
            # Reuse its return value to avoid a second pass of ~50 GPU-CPU syncs.
            if cfg.max_grad_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                ).item()
            else:
                grad_norm = get_grad_norm(model)

            # ---- Optimiser + scheduler step ---------------------------------
            self.optimizer.step()
            self.scheduler.step()

            # ---- Graceful shutdown check -----------------------------------
            # Signal handler가 request_shutdown()을 호출하면 이 flag가 True.
            # 현재 step의 optimizer 업데이트가 완료된 시점에서 체크하므로
            # 모델 가중치는 항상 일관된 상태로 저장됩니다.
            if self._shutdown_requested:
                self._log(
                    f"Graceful shutdown initiated (signal: {self._shutdown_signal}) "
                    f"at step {step + 1}, loss={avg_loss:.4f}",
                    level="WARN",
                )
                if self._is_main:
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step + 1,
                        loss=avg_loss,
                        path=cfg.checkpoint_dir,
                    )
                    self._log(f"Emergency checkpoint saved → {ckpt_path}", level="WARN")
                # DDP 동기화: 모든 rank가 함께 종료하도록 barrier
                try:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                except Exception:
                    pass  # DDP 미사용 또는 이미 해체된 경우 무시
                self._log("Shutdown complete. Exiting training loop.", level="WARN")
                if self._writer is not None:
                    self._writer.close()
                if self._log_fh is not None:
                    self._log_fh.flush()
                return

            running_loss += avg_loss
            log_step_count += 1

            # ---- Logging ---------------------------------------------------
            if (step + 1) % cfg.log_interval == 0 and self._is_main:
                t1 = time.perf_counter()
                elapsed = t1 - t0

                avg_loss = running_loss / log_step_count

                # Estimate throughput: tokens processed during this log window.
                batch_size, seq_len = self._last_batch_shape
                tokens_per_sec = (
                    batch_size * seq_len * cfg.grad_accum_steps * cfg.log_interval
                ) / max(elapsed, 1e-9)

                current_lr = self.scheduler.get_last_lr()[0]
                global_step = step + 1

                mem_gb = torch.cuda.memory_allocated() / 1e9
                self._log(
                    f"step {global_step:>7d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"gnorm {grad_norm:.3f} | "
                    f"tok/s {tokens_per_sec:,.0f} | "
                    f"mem {mem_gb:.1f}GB | "
                    f"epoch {self._epoch}"
                )

                if self._writer is not None:
                    self._writer.add_scalar("train/loss", avg_loss, global_step)
                    self._writer.add_scalar("train/lr", current_lr, global_step)
                    self._writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    self._writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step)

                # Reset accumulators.
                running_loss = 0.0
                log_step_count = 0
                t0 = t1

            # ---- Validation ------------------------------------------------
            if (step + 1) % cfg.eval_interval == 0 and self._val_loader is not None:
                val_loss = self._run_validation()
                if self._is_main:
                    self._log(f"step {step + 1:>7d} | val_loss {val_loss:.4f}")
                    if self._writer is not None:
                        self._writer.add_scalar("val/loss", val_loss, step + 1)
                    # Save best checkpoint when val loss improves.
                    if val_loss < self._best_val_loss:
                        self._best_val_loss = val_loss
                        self._val_patience_counter = 0
                        best_path = save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=step + 1,
                            loss=val_loss,
                            path=cfg.checkpoint_dir,
                            suffix="best",
                        )
                        self._log(
                            f"New best val_loss={val_loss:.4f} → {best_path}"
                        )
                    else:
                        self._val_patience_counter += 1
                        self._log(
                            f"val_loss {val_loss:.4f} did not improve "
                            f"(best={self._best_val_loss:.4f}, "
                            f"patience={self._val_patience_counter}/{self._val_patience_limit})"
                        )
                        if self._val_patience_counter >= self._val_patience_limit:
                            self._log(
                                f"Early stopping triggered at step {step + 1} "
                                f"(patience {self._val_patience_limit} exhausted)"
                            )
                            return

            # ---- Checkpoint save -------------------------------------------
            if (step + 1) % cfg.save_interval == 0 and self._is_main:
                ckpt_path = save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step + 1,
                    loss=avg_loss,
                    path=cfg.checkpoint_dir,
                )
                self._log(f"Checkpoint saved → {ckpt_path}")

        # ---- End of training cleanup ---------------------------------------
        if self._is_main:
            # Save final checkpoint.
            final_path = save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=cfg.max_steps,
                loss=avg_loss,
                path=cfg.checkpoint_dir,
            )
            self._log(f"Training complete. Final checkpoint → {final_path}")

            import datetime
            elapsed = (datetime.datetime.now() - self._train_start_time).total_seconds()
            total_steps_done = cfg.max_steps - start_step
            self._log(
                f"Training summary: {total_steps_done} steps, "
                f"{elapsed/3600:.2f}h elapsed, "
                f"avg {total_steps_done/elapsed:.1f} steps/s"
            )

            if self._writer is not None:
                self._writer.close()
            if self._log_fh is not None:
                self._log_fh.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_validation(self) -> float:
        """
        Evaluate the model on the entire validation set and return the mean loss.

        Temporarily switches the model to eval mode and back to train mode
        afterwards so that dropout / NEFTune hooks are inactive during eval.
        """
        model = self.model
        model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self._val_loader:  # type: ignore[union-attr]
            input_ids = batch[0].to(self.device, dtype=torch.long, non_blocking=True)
            targets   = batch[1].to(self.device, dtype=torch.long, non_blocking=True)
            # Consume attention_mask if provided (model does not use it yet).
            _attn_mask = batch[2].to(self.device, non_blocking=True) if len(batch) > 2 else None  # noqa: F841

            device_type = self.device.type
            with contextlib.ExitStack() as stack:
                if self.config.use_fp8 and self._fp8_recipe is not None:
                    stack.enter_context(
                        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                    )
                    stack.enter_context(
                        te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe)
                    )
                elif self.config.use_amp:
                    stack.enter_context(
                        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                    )
                logits, _ = model(input_ids)
                loss = self._compute_loss(logits, targets)

            total_loss += loss.item()
            total_batches += 1

        model.train()
        if total_batches == 0:
            self._log("Validation set is empty — returning inf", level="WARN")
            return float("inf")
        return total_loss / total_batches

    def _log(self, msg: str, level: str = "INFO") -> None:
        """Print to stdout and optionally write to the log file."""
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")

    def _step(self, batch: tuple) -> torch.Tensor:
        """
        Execute one forward + backward pass for a single micro-batch.

        The loss is divided by ``grad_accum_steps`` so that gradients
        accumulated over multiple micro-batches sum to the correct scale.

        Args:
            batch: ``(input_ids, targets)`` or ``(input_ids, targets, attention_mask)``
                   tensors on CPU; moved to device here.

        Returns:
            Raw (un-scaled) loss as a detached GPU tensor (no CPU sync).
            The caller is responsible for calling .item() once per optimizer step.
        """
        input_ids = batch[0].to(self.device, dtype=torch.long, non_blocking=True)
        targets   = batch[1].to(self.device, dtype=torch.long, non_blocking=True)
        # Consume attention_mask if the dataset provides it (future-proof).
        # Current model forward(input_ids, targets=None) does not accept
        # attention_mask, so we read it but do not forward it yet.
        _attn_mask = batch[2].to(self.device, non_blocking=True) if len(batch) > 2 else None  # noqa: F841

        # Store for tokens/sec calculation.
        self._last_batch_shape = (input_ids.shape[0], input_ids.shape[1])

        device_type = self.device.type
        # te.fp8_autocast must be combined with torch.autocast(bfloat16) so that
        # all tensors entering TE modules are in BF16 (not FP32 master weights).
        # te.fp8_autocast only affects TE modules (te.Linear, te.LayerNormMLP).
        # Hybrid Mamba-2 layers use nn.Linear → stay in bf16 under torch.autocast.
        with contextlib.ExitStack() as stack:
            if self.config.use_fp8 and self._fp8_recipe is not None:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                )
                stack.enter_context(
                    te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe)
                )
            elif self.config.use_amp:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                )
            _, loss = self.model(input_ids, targets=targets)

        # Scale loss for gradient accumulation before backward.
        scaled_loss = loss / self.config.grad_accum_steps
        scaled_loss.backward()

        # Return detached GPU tensor — no CPU sync here.
        # Caller accumulates on GPU and calls .item() once per optimizer step.
        return loss.detach()

    @staticmethod
    def _compute_loss(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss, ignoring target positions equal to -1.

        Args:
            logits:  ``[B, T, vocab_size]`` float tensor.
            targets: ``[B, T]`` long tensor (may contain -1 as ignore index).

        Returns:
            Scalar loss tensor.
        """
        B, T, V = logits.shape
        return nn.functional.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=-1,
        )

    def _next_batch(self) -> tuple:
        """Return the next batch, restarting the DataLoader iterator if exhausted."""
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._epoch += 1
            # Advance DistributedSampler epoch so each pass has a fresh shuffle.
            if self._sampler is not None:
                self._sampler.set_epoch(self._epoch)
            self._loader_iter = iter(self.train_loader)
            return next(self._loader_iter)

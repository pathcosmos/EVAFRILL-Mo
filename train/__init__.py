"""
train — LLM pretraining package.

Public API:
    TrainConfig   : Dataclass of training hyper-parameters.
    Trainer       : Core training loop with gradient accumulation, AMP, and logging.

Utility functions (re-exported from train.utils):
    get_cosine_schedule_with_warmup
    save_checkpoint
    load_checkpoint
    get_grad_norm
    setup_ddp
    cleanup_ddp
    is_main_process
"""

from train.trainer import TrainConfig, Trainer
from train.utils import (
    cleanup_ddp,
    get_cosine_schedule_with_warmup,
    get_grad_norm,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
    setup_ddp,
)

__all__ = [
    # Core classes
    "TrainConfig",
    "Trainer",
    # Utility functions
    "get_cosine_schedule_with_warmup",
    "save_checkpoint",
    "load_checkpoint",
    "get_grad_norm",
    "setup_ddp",
    "cleanup_ddp",
    "is_main_process",
]

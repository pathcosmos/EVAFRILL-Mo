"""
model — LLM architecture package.

Public API:
    LLM        : top-level decoder-only transformer/hybrid language model
    LMConfig   : configuration dataclass
    Mamba2Block: Mamba-2 SSD block (used internally by LLM in hybrid mode)
"""

from .config import LMConfig
from .mamba_block import Mamba2Block
from .transformer import LLM

__all__ = [
    "LLM",
    "LMConfig",
    "Mamba2Block",
]

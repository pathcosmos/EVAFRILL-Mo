"""
Reusable building-block layers: RMSNorm, RotaryEmbedding, SwiGLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# RMS Layer Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019).

    Computation is promoted to float32 for numerical stability and cast back
    to the input dtype before returning.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D) — compute in fp32
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32, normalise, scale, then restore original dtype.
        out = self._norm(x.float()).to(x.dtype)
        return out * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputed rotary positional embeddings (Su et al., RoFormer 2021).

    Cos/sin tables are stored as buffers (shape: max_seq_len × D//2) so they
    move with the module to the correct device automatically.
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute and register
        cos, sin = self._build_tables(dim, max_seq_len, theta)
        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    @staticmethod
    def _build_tables(
        dim: int, max_seq_len: int, theta: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables with shape (max_seq_len, dim // 2)."""
        half_dim = dim // 2
        # Inverse frequencies: shape (half_dim,)
        freqs = 1.0 / (
            theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )
        # Positions: shape (max_seq_len,)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product → (max_seq_len, half_dim)
        emb = torch.outer(t, freqs)
        cos = emb.cos()  # (T, D//2)
        sin = emb.sin()  # (T, D//2)
        return cos, sin

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices of shape (seq_len, D//2) on *device*.

        If *seq_len* exceeds the precomputed length the tables are recomputed
        on-the-fly (rare, but graceful fallback).
        """
        if seq_len > self.max_seq_len:
            cos, sin = self._build_tables(self.dim, seq_len, self.theta)
            cos = cos.to(device)
            sin = sin.to(device)
        else:
            cos = self._cos_cached[:seq_len].to(device)
            sin = self._sin_cached[:seq_len].to(device)
        return cos, sin


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (Shazeer, 2020).

    Architecture:
        out = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    The gate and up projections are separate linear layers so that the gating
    mechanism can learn an independent representation.
    """

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.up_proj   = nn.Linear(d_model, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: element-wise product of SiLU(gate) and up projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

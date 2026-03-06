"""
Multi-Head (and Grouped-Query) Attention with optional FlashAttention-2 backend.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig

# ---------------------------------------------------------------------------
# Optional FlashAttention import
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func  # type: ignore[import]
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

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
# Rotary embedding helper
# ---------------------------------------------------------------------------

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to query or key tensor.

    Args:
        x:   (B, T, H, D_head)
        cos: (T, D_head // 2)  — from RotaryEmbedding.forward
        sin: (T, D_head // 2)  — from RotaryEmbedding.forward

    Returns:
        Tensor with the same shape as *x*, rotated.
    """
    d = x.shape[-1]
    half_d = d // 2

    x1 = x[..., :half_d]   # (B, T, H, D//2)
    x2 = x[..., half_d:]   # (B, T, H, D//2)

    # Broadcast cos/sin from (T, D//2) → (1, T, 1, D//2)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)

    rotated = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    return rotated.to(x.dtype)



# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head (or grouped-query) causal self-attention.

    Supports:
    - Standard MHA: n_kv_heads == n_heads
    - GQA / MQA:    n_kv_heads < n_heads  (must evenly divide n_heads)

    Attention backend:
    - FlashAttention-2 when available and config.use_flash_attn is True
    - Vanilla scaled dot-product otherwise (causal mask via upper-triangular)
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads        # resolved in __post_init__
        self.head_dim   = config.d_model // config.n_heads
        self.d_model    = config.d_model
        self.dropout    = config.dropout
        self.use_flash  = config.use_flash_attn

        # Number of query-head groups per KV head
        self.n_rep = self.n_heads // self.n_kv_heads

        # Projections ----------------------------------------------------
        # Select Linear implementation: te.Linear (FP8) or nn.Linear (BF16)
        _Linear = te.Linear if (config.use_fp8 and HAS_TE) else nn.Linear

        # Fused QKV projection: single GEMM (d_model → q_dim + k_dim + v_dim)
        # For GQA 24:8 with head_dim=128: 3072 + 1024 + 1024 = 5120
        self._q_dim  = self.n_heads    * self.head_dim  # e.g. 24 * 128 = 3072
        self._kv_dim = self.n_kv_heads * self.head_dim  # e.g.  8 * 128 = 1024
        self.qkv_proj = _Linear(
            config.d_model,
            self._q_dim + 2 * self._kv_dim,  # 3072 + 2*1024 = 5120
            bias=config.bias,
        )
        self.out_proj = _Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

    # ------------------------------------------------------------------
    # KV-head expansion for GQA
    # ------------------------------------------------------------------

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand KV heads to match the number of query heads.

        Args:
            x:     (B, T, n_kv_heads, head_dim)
            n_rep: repetition factor

        Returns:
            (B, T, n_kv_heads * n_rep, head_dim)
        """
        if n_rep == 1:
            return x
        B, T, n_kv, D = x.shape
        return x.repeat_interleave(n_rep, dim=2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, T, C)
            cos: (T, head_dim // 2) — from RotaryEmbedding
            sin: (T, head_dim // 2) — from RotaryEmbedding

        Returns:
            (B, T, C)
        """
        B, T, C = x.shape

        # --- Fused QKV projection (single GEMM) --------------------------------
        qkv = self.qkv_proj(x)  # (B, T, q_dim + 2*kv_dim)
        q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)
        q = q.view(B, T, self.n_heads,    self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # FlashAttention-2 and rotary embedding require bf16/fp16.
        # te.Linear with MXFP8 may emit FP8-format output tensors; cast if needed.
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        # --- Rotary embeddings -----------------------------------------------
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # --- Attention -------------------------------------------------------
        if self.use_flash and HAS_FLASH_ATTN and x.is_cuda:
            attn_out = self._flash_attention(q, k, v, B, T)
        else:
            attn_out = self._standard_attention(q, k, v, B, T)

        # --- Output projection -----------------------------------------------
        # attn_out: (B, T, C)
        return self.out_proj(attn_out)

    # ------------------------------------------------------------------
    # FlashAttention-2 path
    # ------------------------------------------------------------------

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Run FlashAttention-2.

        flash_attn_func expects inputs in (B, T, H, D) layout and returns
        (B, T, H, D).  FlashAttention-2 natively supports GQA via head count
        mismatch (q has n_heads, k/v have n_kv_heads) — no KV expansion needed.
        """
        dropout_p = self.dropout if self.training else 0.0

        # flash_attn_func: (B, T, H, D) → (B, T, H, D)
        # GQA is handled natively: q=(B,T,n_heads,D), k/v=(B,T,n_kv_heads,D)
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)

        # Reshape (B, T, n_heads, head_dim) → (B, T, C)
        return out.reshape(B, T, self.n_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Standard (fallback) attention path
    # ------------------------------------------------------------------

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Vanilla scaled dot-product causal attention.

        Softmax is computed in float32 for numerical stability.
        """
        # Expand KV heads for GQA
        k = self._repeat_kv(k, self.n_rep)  # (B, T, n_heads, head_dim)
        v = self._repeat_kv(v, self.n_rep)  # (B, T, n_heads, head_dim)

        # (B, T, H, D) → (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)

        # Scaled dot-product: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: fill upper triangle (excluding diagonal) with -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Softmax in fp32, then cast back
        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)

        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Weighted sum: (B, H, T, D)
        out = torch.matmul(attn_weights, v)

        # (B, H, T, D) → (B, T, H, D) → (B, T, C)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.d_model)
        return out

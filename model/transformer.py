"""
Full transformer: TransformerBlock and top-level LLM model.
Supports pure Transformer and hybrid Mamba-2 + Transformer architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig
from .layers import RMSNorm, RotaryEmbedding, SwiGLU
from .attention import MultiHeadAttention
from .mamba_block import Mamba2Block

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
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single pre-norm transformer decoder block.

    Layout:
        x = x + Attention( RMSNorm(x) )
        x = x + FFN( RMSNorm(x) )
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn      = MultiHeadAttention(config)
        self._use_fp8  = config.use_fp8 and HAS_TE

        if self._use_fp8:
            # te.LayerNormMLP fuses RMSNorm + gate/up/down projections into one kernel.
            # It applies normalisation internally, so ffn_norm is not needed.
            self.ffn_norm = None
            self.ffn = te.LayerNormMLP(
                hidden_size=config.d_model,
                ffn_hidden_size=config.d_ffn,
                bias=config.bias,
                activation="swiglu",
                normalization="RMSNorm",
            )
        else:
            self.ffn_norm = RMSNorm(config.d_model)
            self.ffn      = SwiGLU(config.d_model, config.d_ffn, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, T, C)
            cos: (T, head_dim // 2)
            sin: (T, head_dim // 2)

        Returns:
            (B, T, C)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # FFN with residual — te.LayerNormMLP applies norm internally
        if self._use_fp8:
            x = x + self.ffn(x)
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class LLM(nn.Module):
    """Decoder-only transformer language model.

    Features:
    - Learned token embeddings with weight tying to the LM head
    - Rotary positional embeddings (no learned position embeddings)
    - Stack of pre-norm TransformerBlocks
    - Final RMSNorm before the LM head
    - Optional cross-entropy loss computation (for training)
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config

        # --- Embedding -------------------------------------------------------
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # --- Layers (pure Transformer or hybrid Mamba-Transformer) -----------
        if config.use_hybrid and config.hybrid_pattern:
            pattern = config.hybrid_pattern.strip().split()
            if len(pattern) != config.n_layers:
                raise ValueError(
                    f"hybrid_pattern has {len(pattern)} entries but "
                    f"n_layers={config.n_layers}"
                )
            layers: list[nn.Module] = []
            # Track which layers are Mamba vs Attention for forward dispatch
            self._layer_types: list[str] = pattern
            for layer_type in pattern:
                if layer_type == "M":
                    layers.append(Mamba2Block(
                        d_model=config.d_model,
                        d_state=config.mamba_d_state,
                        head_dim=config.mamba_head_dim,
                        expand=config.mamba_expand,
                        conv_kernel=config.mamba_conv_kernel,
                        n_groups=config.mamba_n_groups,
                        chunk_size=config.mamba_chunk_size,
                        d_ffn=config.mamba_d_ffn,
                        use_fp8=config.use_fp8,
                    ))
                elif layer_type == "A":
                    layers.append(TransformerBlock(config))
                else:
                    raise ValueError(
                        f"Unknown layer type '{layer_type}' in hybrid_pattern. "
                        f"Use 'M' (Mamba) or 'A' (Attention)."
                    )
            self.layers = nn.ModuleList(layers)
        else:
            self._layer_types = ["A"] * config.n_layers
            self.layers = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layers)]
            )

        # --- Final normalisation and LM head ---------------------------------
        self.norm    = RMSNorm(config.d_model)
        # NOTE: lm_head는 nn.Linear 유지 — embedding weight tying + TE FP8 호환성
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and LM-head weight matrices
        self.lm_head.weight = self.embedding.weight

        # --- Rotary embeddings -----------------------------------------------
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # --- Initialise weights ----------------------------------------------
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply standard initialisation:
        - Linear / Embedding weights: N(0, 0.02)
        - Bias parameters: zeros
        - te.Linear / te.LayerNormMLP: skipped (TE manages its own init)
        - Mamba2Block: skipped (manages its own init)
        """
        # TE modules handle their own weight initialisation.
        if HAS_TE and isinstance(module, (te.Linear, te.LayerNormMLP)):
            return
        # Mamba2Block handles its own parameter init (A_log, D, dt_bias, etc.)
        if isinstance(module, Mamba2Block):
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) long tensor of token indices
            targets:   (B, T) long tensor of target token indices, or None.
                       Use -1 (ignore_index) to mask positions.

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets is None
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings: (B, T, C)
        x = self.embedding(input_ids)

        # Rotary cos/sin for this sequence length: (T, head_dim // 2)
        # Only needed for Attention layers, but precomputed once for all.
        cos, sin = self.rope(T, device)

        # Run through blocks — Mamba blocks ignore cos/sin
        for layer, ltype in zip(self.layers, self._layer_types):
            if ltype == "M":
                x = layer(x)
            else:
                x = layer(x, cos, sin)

        # Final normalisation
        x = self.norm(x)

        # Compute loss if targets are provided
        if targets is not None:
            # Chunked cross-entropy: avoid materializing full (B, T, V) logits
            # at once.  Process `_ce_chunk` tokens at a time so that only
            # (B, chunk, V) logits live on the GPU simultaneously.
            # Memory saving: chunk/T of the naive approach  (e.g. 512/4096 = 1/8).
            chunk = getattr(self.config, 'ce_chunk_size', 512)
            total_loss: torch.Tensor = torch.zeros(
                1, device=x.device, dtype=torch.float32,
            )
            total_count: torch.Tensor = torch.zeros(
                1, device=x.device, dtype=torch.long,
            )
            for i in range(0, T, chunk):
                end = min(i + chunk, T)
                logits_chunk = self.lm_head(x[:, i:end])       # (B, chunk, V)
                tgt_chunk = targets[:, i:end].reshape(-1)       # (B*chunk,)
                loss_chunk = F.cross_entropy(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    tgt_chunk,
                    ignore_index=-1,
                    reduction='sum',
                )
                total_loss = total_loss + loss_chunk
                total_count = total_count + (tgt_chunk != -1).sum()
            loss = total_loss / total_count.clamp(min=1)
            return None, loss

        # Inference / generation: return full logits
        logits = self.lm_head(x)
        return logits, None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> nn.Embedding:
        """HuggingFace-compatible accessor for the token embedding layer."""
        return self.embedding

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: LMConfig) -> "LLM":
        """Construct an LLM from an LMConfig instance."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LLM":
        """Load model from a checkpoint directory.

        Expects:
            <path>/config.yaml  — serialised LMConfig
            <path>/model.pt     — state dict produced by save_pretrained
        """
        path = Path(path)
        config = LMConfig.from_yaml(path / "config.yaml")
        model = cls(config)
        state_dict = torch.load(
            path / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Save config and model weights to a directory.

        Creates:
            <path>/config.yaml
            <path>/model.pt
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(path / "config.yaml")
        torch.save(self.state_dict(), path / "model.pt")

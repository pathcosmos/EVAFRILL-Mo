"""
model/lora.py — LoRA (Low-Rank Adaptation) for EVAFRILL-Mo hybrid models.

Injects trainable low-rank adapters into:
  - Attention layers: qkv_proj, out_proj
  - Mamba-2 layers: in_proj, out_proj

Usage:
    model = LLM.from_pretrained(checkpoint)
    apply_lora(model, rank=32, alpha=64)
    # Only LoRA params are trainable; base model is frozen

    # After training, merge LoRA weights back:
    merge_lora(model)

    # Or save/load LoRA weights separately:
    save_lora(model, path)
    load_lora(model, path)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .mamba_block import Mamba2Block


class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear layer.

    Computes: output = original_linear(x) + (alpha/rank) * x @ A^T @ B^T
    where A: (rank, in_features), B: (out_features, rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # A: down-projection (in_features → rank)
        # Create on same device/dtype as original weights
        _dev = original.weight.device
        _dt = original.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=_dev, dtype=_dt))
        # B: up-projection (rank → out_features)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=_dev, dtype=_dt))

        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros → initial LoRA output is zero

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        # LoRA path: x → dropout → A → B → scale
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
        return result + lora_out * self.scaling

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original linear layer permanently."""
        with torch.no_grad():
            # W' = W + scaling * B @ A
            self.original.weight.add_(
                (self.lora_B @ self.lora_A) * self.scaling
            )

    @property
    def weight(self) -> torch.Tensor:
        """Access original weight for compatibility."""
        return self.original.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original.bias


def apply_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> int:
    """Apply LoRA adapters to a model, freeze base weights.

    Args:
        model: The LLM model (raw, not DDP-wrapped).
        rank: LoRA rank (default 32).
        alpha: LoRA scaling factor (default 64).
        dropout: Dropout on LoRA path (default 0).
        target_modules: List of module attribute names to adapt.
            Default: ["qkv_proj", "out_proj", "in_proj"]
            (covers both Attention and Mamba layers).

    Returns:
        Number of LoRA parameters added.
    """
    if target_modules is None:
        target_modules = ["qkv_proj", "out_proj", "in_proj"]

    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        # Check Attention layers
        if isinstance(module, MultiHeadAttention):
            for attr in target_modules:
                if hasattr(module, attr):
                    original = getattr(module, attr)
                    if isinstance(original, nn.Linear):
                        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                        setattr(module, attr, lora_layer)
                        params = rank * original.in_features + original.out_features * rank
                        total_lora_params += params
                        lora_count += 1

        # Check Mamba layers
        elif isinstance(module, Mamba2Block):
            for attr in target_modules:
                if hasattr(module, attr):
                    original = getattr(module, attr)
                    if isinstance(original, nn.Linear):
                        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                        setattr(module, attr, lora_layer)
                        params = rank * original.in_features + original.out_features * rank
                        total_lora_params += params
                        lora_count += 1

    print(f"[LoRA] Applied {lora_count} adapters, {total_lora_params:,} trainable params "
          f"(rank={rank}, alpha={alpha})")
    return total_lora_params


def merge_lora(model: nn.Module) -> int:
    """Merge all LoRA weights back into base model and remove LoRA layers.

    Returns:
        Number of LoRA layers merged.
    """
    merged = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            # Check nn.Module children
            pass

        if isinstance(module, (MultiHeadAttention, Mamba2Block)):
            for attr in ["qkv_proj", "out_proj", "in_proj"]:
                if hasattr(module, attr):
                    layer = getattr(module, attr)
                    if isinstance(layer, LoRALinear):
                        layer.merge_weights()
                        setattr(module, attr, layer.original)
                        merged += 1

    # Unfreeze all parameters after merging
    for param in model.parameters():
        param.requires_grad = True

    print(f"[LoRA] Merged {merged} adapters back into base model")
    return merged


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all LoRA trainable parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def save_lora(model: nn.Module, path: str | Path) -> Path:
    """Save only the LoRA adapter weights."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()

    save_path = path / "lora_weights.pt"
    torch.save(lora_state, save_path)
    n_params = sum(v.numel() for v in lora_state.values())
    size_mb = save_path.stat().st_size / 1e6
    print(f"[LoRA] Saved {len(lora_state)} tensors ({n_params:,} params, {size_mb:.1f} MB) → {save_path}")
    return save_path


def load_lora(model: nn.Module, path: str | Path) -> int:
    """Load LoRA adapter weights. LoRA layers must already be applied."""
    path = Path(path)
    lora_file = path / "lora_weights.pt" if path.is_dir() else path
    lora_state = torch.load(lora_file, map_location="cpu", weights_only=True)

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state and b_key in lora_state:
                module.lora_A.data.copy_(lora_state[a_key])
                module.lora_B.data.copy_(lora_state[b_key])
                loaded += 1

    print(f"[LoRA] Loaded {loaded} adapter weight pairs from {lora_file}")
    return loaded

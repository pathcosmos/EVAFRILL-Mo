#!/usr/bin/env python3
"""
scripts/merge_checkpoints.py — Slerp (Spherical Linear Interpolation) checkpoint merge.

Merges two model checkpoints (e.g., SFT + DPO) using SLERP interpolation
to balance knowledge retention and alignment improvement.

Reference: Nemotron-H paper — SLERP merging reduces alignment tax.

Usage:
    python scripts/merge_checkpoints.py \
        --ckpt_a checkpoints/3b_sft_v2/checkpoint-best \
        --ckpt_b checkpoints/3b_dpo/checkpoint-merged \
        --output checkpoints/3b_dpo/checkpoint-slerp \
        --alpha 0.5

    alpha=0.0 → pure ckpt_a (SFT)
    alpha=1.0 → pure ckpt_b (DPO)
    alpha=0.5 → equal blend (recommended starting point)
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch
import yaml


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    Args:
        t: Interpolation factor in [0, 1]. 0 → v0, 1 → v1.
        v0: First tensor (flattened internally).
        v1: Second tensor (same shape as v0).
        eps: Small value to avoid division by zero.

    Returns:
        Interpolated tensor with the same shape as v0.
    """
    original_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # Normalize
    v0_norm = v0_flat / (v0_flat.norm() + eps)
    v1_norm = v1_flat / (v1_flat.norm() + eps)

    # Cosine of angle between vectors
    cos_omega = torch.dot(v0_norm, v1_norm).clamp(-1.0, 1.0)

    # If vectors are very similar, fall back to linear interpolation
    if abs(cos_omega.item()) > 0.9995:
        result = (1.0 - t) * v0_flat + t * v1_flat
        return result.reshape(original_shape).to(v0.dtype)

    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    s0 = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega

    # Interpolate using original (non-normalized) vectors to preserve scale
    result = s0 * v0_flat + s1 * v1_flat
    return result.reshape(original_shape).to(v0.dtype)


def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Simple linear interpolation."""
    return ((1.0 - t) * v0.float() + t * v1.float()).to(v0.dtype)


def merge_state_dicts(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    alpha: float = 0.5,
    method: str = "slerp",
) -> dict[str, torch.Tensor]:
    """Merge two state dicts using SLERP or LERP.

    Args:
        sd_a: State dict A (e.g., SFT model).
        sd_b: State dict B (e.g., DPO model).
        alpha: Interpolation factor. 0 → A, 1 → B.
        method: "slerp" or "lerp".

    Returns:
        Merged state dict.
    """
    interp_fn = slerp if method == "slerp" else lerp

    merged = {}
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    if only_a:
        print(f"[WARN] {len(only_a)} keys only in ckpt_a (kept as-is)")
    if only_b:
        print(f"[WARN] {len(only_b)} keys only in ckpt_b (kept as-is)")

    for key in sorted(common):
        va = sd_a[key]
        vb = sd_b[key]

        if va.shape != vb.shape:
            print(f"[WARN] Shape mismatch for {key}: {va.shape} vs {vb.shape}, keeping ckpt_a")
            merged[key] = va
            continue

        # Only interpolate float parameters (skip int buffers, etc.)
        if va.is_floating_point() and va.numel() > 1:
            merged[key] = interp_fn(alpha, va, vb)
        else:
            merged[key] = va  # Keep from ckpt_a for non-float/scalar

    # Include keys unique to each
    for key in only_a:
        merged[key] = sd_a[key]
    for key in only_b:
        merged[key] = sd_b[key]

    return merged


def main():
    parser = argparse.ArgumentParser(description="SLERP checkpoint merge")
    parser.add_argument("--ckpt_a", type=Path, required=True,
                        help="Path to checkpoint A (e.g., SFT)")
    parser.add_argument("--ckpt_b", type=Path, required=True,
                        help="Path to checkpoint B (e.g., DPO)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output checkpoint directory")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation factor (0=A, 1=B, default 0.5)")
    parser.add_argument("--method", choices=["slerp", "lerp"], default="slerp",
                        help="Interpolation method (default: slerp)")
    args = parser.parse_args()

    print(f"Merge: {args.ckpt_a.name} ←({1-args.alpha:.1%})— ({args.alpha:.1%})→ {args.ckpt_b.name}")
    print(f"Method: {args.method}, alpha={args.alpha}")

    # Load state dicts
    print("Loading checkpoint A...")
    sd_a = torch.load(args.ckpt_a / "model.pt", map_location="cpu", weights_only=True)
    print(f"  {len(sd_a)} keys loaded")

    print("Loading checkpoint B...")
    sd_b = torch.load(args.ckpt_b / "model.pt", map_location="cpu", weights_only=True)
    print(f"  {len(sd_b)} keys loaded")

    # Merge
    print("Merging...")
    merged_sd = merge_state_dicts(sd_a, sd_b, alpha=args.alpha, method=args.method)
    print(f"  {len(merged_sd)} keys in merged state dict")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(merged_sd, args.output / "model.pt")

    # Copy config from ckpt_a
    config_src = args.ckpt_a / "config.yaml"
    if config_src.exists():
        shutil.copy2(str(config_src), str(args.output / "config.yaml"))

    # Copy tokenizer if available
    for tok_name in ["tokenizer.json", "tokenizer.model"]:
        tok_src = args.ckpt_a / tok_name
        if tok_src.exists():
            shutil.copy2(str(tok_src), str(args.output / tok_name))

    # Write merge metadata
    meta = {
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
        "alpha": args.alpha,
        "method": args.method,
    }
    with open(args.output / "merge_info.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    size_mb = (args.output / "model.pt").stat().st_size / 1e6
    print(f"\nMerged checkpoint saved → {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()

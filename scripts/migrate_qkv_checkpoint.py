#!/usr/bin/env python3
"""Migrate checkpoint from separate Q/K/V projections to fused QKV.

Usage:
    python3 scripts/migrate_qkv_checkpoint.py <checkpoint_dir>

Migrates both model.pt AND optimizer.pt:
  - model.pt:     q_proj/k_proj/v_proj weights → qkv_proj weight
  - optimizer.pt: exp_avg/exp_avg_sq states fused, param indices re-mapped

The concatenation order is [Q ; K ; V] along the output (dim-0) axis,
which matches the split in MultiHeadAttention.forward:
    q, k, v = qkv.split([_q_dim, _kv_dim, _kv_dim], dim=-1)

Optimizer layout (group 0 = weight_decay, per layer × 28):
  [i*6+0] q_proj.weight  [3072, 3072]
  [i*6+1] k_proj.weight  [1024, 3072]
  [i*6+2] v_proj.weight  [1024, 3072]
  [i*6+3] out_proj.weight [3072, 3072]
  [i*6+4] fc1_weight     [16384, 3072]
  [i*6+5] fc2_weight     [3072, 8192]
After fusion: indices 0,1,2 → single qkv_proj → 4 params per layer.
"""
import sys
import torch
from pathlib import Path

N_LAYERS = 28
OLD_PARAMS_PER_LAYER = 6  # q, k, v, out, fc1, fc2
NEW_PARAMS_PER_LAYER = 4  # qkv, out, fc1, fc2


def migrate_model(state: dict) -> dict:
    """Fuse Q/K/V projection weights into QKV in model state dict."""
    new_state: dict = {}
    layers_done: set = set()

    for key, val in state.items():
        if ".q_proj." not in key and ".k_proj." not in key and ".v_proj." not in key:
            new_state[key] = val
            continue

        if ".q_proj." not in key:
            continue

        prefix = key.rsplit(".", 2)[0]
        suffix = key.rsplit(".", 1)[-1]

        tag = (prefix, suffix)
        if tag in layers_done:
            continue
        layers_done.add(tag)

        q_key = f"{prefix}.q_proj.{suffix}"
        k_key = f"{prefix}.k_proj.{suffix}"
        v_key = f"{prefix}.v_proj.{suffix}"

        missing = [k for k in (q_key, k_key, v_key) if k not in state]
        if missing:
            raise KeyError(f"Expected keys not found in checkpoint: {missing}")

        q_w, k_w, v_w = state[q_key], state[k_key], state[v_key]
        fused = torch.cat([q_w, k_w, v_w], dim=0)
        fused_key = f"{prefix}.qkv_proj.{suffix}"
        new_state[fused_key] = fused
        print(f"  Fused  {fused_key}: {list(fused.shape)}"
              f"  (q={list(q_w.shape)}, k={list(k_w.shape)}, v={list(v_w.shape)})")

    leaked = [k for k in new_state if ".q_proj." in k or ".k_proj." in k or ".v_proj." in k]
    if leaked:
        raise RuntimeError(f"BUG: old projection keys still present: {leaked}")

    return new_state


def migrate_optimizer(opt_state: dict) -> dict:
    """Fuse optimizer states for Q/K/V → QKV and re-index parameters.

    The optimizer has 2 param groups:
      Group 0 (weight_decay): 168 = 28 layers × 6 (q,k,v,out,fc1,fc2)
      Group 1 (no weight_decay): 58 = norms + embedding

    We fuse q,k,v entries in group 0 (indices i*6+0,1,2 → one entry per layer).
    Group 0 shrinks from 168 to 112 (28 layers × 4 params).
    Group 1 stays at 58. Total: 170.
    """
    old_state = opt_state["state"]
    old_groups = opt_state["param_groups"]

    group0_count = len(old_groups[0]["params"])
    expected_g0 = N_LAYERS * OLD_PARAMS_PER_LAYER
    if group0_count != expected_g0:
        raise ValueError(
            f"Group 0 has {group0_count} params, expected {expected_g0}. "
            f"Cannot auto-detect QKV layout."
        )

    # Validate shapes for first layer
    shapes = []
    for j in range(OLD_PARAMS_PER_LAYER):
        idx = old_groups[0]["params"][j]
        shapes.append(list(old_state[idx]["exp_avg"].shape))
    expected_shapes = [[3072, 3072], [1024, 3072], [1024, 3072],
                       [3072, 3072], [16384, 3072], [3072, 8192]]
    if shapes != expected_shapes:
        raise ValueError(
            f"Layer 0 shapes {shapes} don't match expected {expected_shapes}. "
            f"Cannot auto-detect QKV layout."
        )
    print(f"  Shape validation passed for layer 0.")

    new_state_entries = {}
    new_idx = 0

    # --- Group 0: fuse q/k/v per layer ---
    for layer_i in range(N_LAYERS):
        base = layer_i * OLD_PARAMS_PER_LAYER
        q_opt_idx = old_groups[0]["params"][base + 0]
        k_opt_idx = old_groups[0]["params"][base + 1]
        v_opt_idx = old_groups[0]["params"][base + 2]

        q_entry = old_state[q_opt_idx]
        k_entry = old_state[k_opt_idx]
        v_entry = old_state[v_opt_idx]

        # Fuse QKV
        fused_entry = {"step": q_entry["step"]}
        for field in ["exp_avg", "exp_avg_sq"]:
            if field in q_entry:
                fused_entry[field] = torch.cat(
                    [q_entry[field], k_entry[field], v_entry[field]], dim=0
                )
        new_state_entries[new_idx] = fused_entry
        if layer_i == 0:
            print(f"  Layer 0 QKV fused: exp_avg {list(fused_entry['exp_avg'].shape)}")
        new_idx += 1

        # Copy remaining params (out, fc1, fc2)
        for offset in [3, 4, 5]:
            opt_idx = old_groups[0]["params"][base + offset]
            new_state_entries[new_idx] = old_state[opt_idx]
            new_idx += 1

    new_group0_count = new_idx  # should be N_LAYERS * NEW_PARAMS_PER_LAYER = 112
    print(f"  Group 0: {group0_count} → {new_group0_count} params")

    # --- Group 1: copy as-is (norms, embedding — no QKV) ---
    group1_count = len(old_groups[1]["params"])
    for j in range(group1_count):
        opt_idx = old_groups[1]["params"][j]
        if opt_idx in old_state:
            new_state_entries[new_idx] = old_state[opt_idx]
        new_idx += 1
    print(f"  Group 1: {group1_count} → {group1_count} params (unchanged)")

    # Build new param_groups
    new_groups = []
    g0 = {k: v for k, v in old_groups[0].items() if k != "params"}
    g0["params"] = list(range(0, new_group0_count))
    new_groups.append(g0)

    g1 = {k: v for k, v in old_groups[1].items() if k != "params"}
    g1["params"] = list(range(new_group0_count, new_group0_count + group1_count))
    new_groups.append(g1)

    total = new_group0_count + group1_count
    print(f"  Total: {len(old_state)} → {total} optimizer params")

    return {"state": new_state_entries, "param_groups": new_groups}


def migrate(ckpt_dir: Path) -> None:
    model_path = ckpt_dir / "model.pt"
    opt_path = ckpt_dir / "optimizer.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

    # --- Model migration ---
    print(f"[1/2] Migrating model weights from {model_path} ...")
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    has_old = any(".q_proj." in k for k in state)
    has_new = any(".qkv_proj." in k for k in state)

    if has_new and not has_old:
        print("  Model already migrated. Skipping.")
    elif has_old:
        new_model_state = migrate_model(state)
        torch.save(new_model_state, model_path)
        print(f"  Model saved.")
    else:
        raise RuntimeError("Model state has neither q_proj nor qkv_proj keys!")

    # --- Optimizer migration ---
    if opt_path.exists():
        print(f"\n[2/2] Migrating optimizer states from {opt_path} ...")
        opt = torch.load(opt_path, map_location="cpu", weights_only=True)

        # Check if already migrated
        total_params = sum(len(pg["params"]) for pg in opt["param_groups"])
        expected_old = N_LAYERS * OLD_PARAMS_PER_LAYER + 58  # 168 + 58 = 226
        expected_new = N_LAYERS * NEW_PARAMS_PER_LAYER + 58  # 112 + 58 = 170

        if total_params == expected_old:
            opt_backup = ckpt_dir / "optimizer.pt.backup_pre_qkv"
            if not opt_backup.exists():
                torch.save(opt, opt_backup)
                print(f"  Backup: {opt_backup}")
            new_opt = migrate_optimizer(opt)
            torch.save(new_opt, opt_path)
            print(f"  Optimizer saved.")
        elif total_params == expected_new:
            print(f"  Optimizer already migrated ({total_params} params). Skipping.")
        else:
            print(f"  [WARN] Unexpected param count {total_params} "
                  f"(expected old={expected_old} or new={expected_new}). "
                  f"Deleting optimizer.pt — optimizer will restart fresh.")
            opt_path.unlink()
    else:
        print("\n[2/2] No optimizer.pt found. Optimizer will restart fresh.")

    print("\nMigration complete!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    migrate(Path(sys.argv[1]))

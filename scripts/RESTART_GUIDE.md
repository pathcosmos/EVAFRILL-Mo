# FRANKENSTALLM 3B — Optimization Restart Guide

## Quick restart (all optimizations applied automatically):
```bash
bash scripts/apply_optimizations.sh
```

## Validate only (no restart):
```bash
bash scripts/apply_optimizations.sh --test-only
```

## Manual steps if auto-migration fails:
1. Stop: `kill $(cat checkpoints/korean_3b_fp8_run1/train.pid)`
2. Migrate: `python3 scripts/migrate_qkv_checkpoint.py checkpoints/korean_3b_fp8_run1/checkpoint-XXXXX`
3. Restart: `bash scripts/launch_3b_pretrain.sh`

## Rollback (undo QKV fusion):
```bash
CKPT=checkpoints/korean_3b_fp8_run1/checkpoint-XXXXX
cp ${CKPT}/model.pt.backup_pre_qkv ${CKPT}/model.pt
git checkout model/attention.py  # restore original attention code
```

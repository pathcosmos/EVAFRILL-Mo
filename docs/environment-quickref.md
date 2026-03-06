# Environment Quick Reference

**Machine**: DCTN-0306095349 | **Updated**: 2026-03-06

---

## Activation

```bash
# 방법 1: aienv 함수 (커스텀 경로)
aienv /home/ghong/project-ghong/taketimes/llm-star/ai-env

# 방법 2: 직접 activate
source /home/ghong/project-ghong/taketimes/llm-star/ai-env/bin/activate

# 비활성화
aienv-off
# 또는
deactivate
```

## Key Paths

| Item | Path |
|------|------|
| Repo | `/home/ghong/taketimes/from_git/machine-setting/` |
| Venv | `/home/ghong/project-ghong/taketimes/llm-star/ai-env/` |
| Logs | `/home/ghong/project-ghong/taketimes/llm-star/logs/` |
| Docs | `/home/ghong/project-ghong/taketimes/llm-star/docs/` |
| HW Profile | `/home/ghong/.machine_setting_profile` |
| Secrets | `/home/ghong/.bashrc.local` |

## GPU Info

- 8x NVIDIA B200 (183 GB VRAM each, 7 visible to PyTorch)
- Driver: 580.95.05
- CUDA: 13.1 (nvcc) / 13.0 (nvidia-smi)
- PyTorch: 2.10.0+cu128

## Package Summary

| Category | Count | Key Packages |
|----------|-------|-------------|
| Core | ~220 | torch, transformers, langchain, anthropic, openai |
| Data | ~35 | pandas, numpy, opencv, SQLAlchemy, boto3 |
| Web | ~15 | fastapi, gradio, flask, httpx, aiohttp |
| GPU | ~6 | torch+cu128, bitsandbytes, onnxruntime-gpu |
| **Total** | **~319** | |

## Daily Commands

```bash
# Sync
make -C /home/ghong/taketimes/from_git/machine-setting push    # Export + commit + push
make -C /home/ghong/taketimes/from_git/machine-setting update  # Pull + notify
make -C /home/ghong/taketimes/from_git/machine-setting status  # Check sync status

# Package management
make -C /home/ghong/taketimes/from_git/machine-setting export  # Export current packages
```

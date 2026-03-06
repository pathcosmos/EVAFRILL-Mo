#!/bin/bash
# lm-eval environment setup script
# Sets up Python path and environment variables for lm-eval evaluation

export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"
export LM_EVAL_BIN="${HOME}/.local/bin/lm_eval"

# Verify installation
python -c "import lm_eval; print('✓ lm_eval', lm_eval.__version__)" || { echo "✗ lm_eval import failed"; exit 1; }
python -c "import sacrebleu; print('✓ sacrebleu installed')" || { echo "✗ sacrebleu import failed"; exit 1; }
python -c "import rouge_score; print('✓ rouge_score installed')" || { echo "✗ rouge_score import failed"; exit 1; }

echo ""
echo "Available Korean tasks:"
$LM_EVAL_BIN ls tasks 2>&1 | grep -E "kobest|haerae|pawsx|global_mmlu_full_ko|mmmlu_ko"

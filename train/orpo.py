"""
ORPO (Odds Ratio Preference Optimization) training script.

Usage:
    python train/orpo.py \
        --model_path outputs/hf_for_orpo \
        --dataset kuotient/orca-math-korean-dpo-pairs \
        --output_dir outputs/orpo_1b \
        [--custom_data_path data/preference_pairs.jsonl] \
        [--epochs 3] [--lr 5e-6] [--beta 0.1] [--batch_size 4]

Prerequisites:
    pip install trl>=0.8.0 transformers accelerate peft datasets
    python scripts/convert_to_hf.py --checkpoint <sft_ckpt> --output outputs/hf_for_orpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# TRL imports
try:
    from trl import ORPOConfig, ORPOTrainer
except ImportError:
    print("ERROR: trl not installed. Run: pip install trl>=0.8.0")
    sys.exit(1)


def load_hf_preference_dataset(dataset_name: str, token: str | None = None) -> Dataset:
    """Load and normalize a HuggingFace preference dataset to {prompt, chosen, rejected}."""
    ds = load_dataset(dataset_name, split="train", token=token)

    # kuotient/orca-math-korean-dpo-pairs format: {system, question, chosen, rejected}
    if "question" in ds.column_names and "chosen" in ds.column_names:
        def normalize(example):
            prompt = example.get("system", "") + "\n" + example["question"]
            return {"prompt": prompt.strip(), "chosen": example["chosen"], "rejected": example["rejected"]}
        return ds.map(normalize, remove_columns=ds.column_names)

    # nayohan/preference-collection-ko-full format: {response_A, response_B, orig_preference}
    if "orig_preference" in ds.column_names:
        def normalize_pref(example):
            prompt = example.get("orig_instruction", example.get("instruction", ""))
            if example["orig_preference"] == "B":
                return {"prompt": prompt, "chosen": example["orig_response_B"], "rejected": example["orig_response_A"]}
            else:
                return {"prompt": prompt, "chosen": example["orig_response_A"], "rejected": example["orig_response_B"]}
        return ds.map(normalize_pref, remove_columns=ds.column_names)

    # Already in {prompt, chosen, rejected} format
    if all(c in ds.column_names for c in ["prompt", "chosen", "rejected"]):
        return ds

    raise ValueError(f"Unknown dataset format. Columns: {ds.column_names}")


def load_custom_jsonl(path: str) -> Dataset:
    """Load custom JSONL with {prompt, chosen, rejected} fields."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="ORPO Training")
    parser.add_argument("--model_path", type=str, required=True, help="HF format model path")
    parser.add_argument("--dataset", type=str, default="kuotient/orca-math-korean-dpo-pairs")
    parser.add_argument("--custom_data_path", type=str, default=None, help="Custom JSONL preference data")
    parser.add_argument("--output_dir", type=str, default="outputs/orpo_1b")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO beta (odds ratio weight)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.custom_data_path:
        print(f"Loading custom data from {args.custom_data_path}...")
        dataset = load_custom_jsonl(args.custom_data_path)
    else:
        print(f"Loading dataset {args.dataset}...")
        dataset = load_hf_preference_dataset(args.dataset, token=args.hf_token)

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")

    # ORPO config
    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        beta=args.beta,  # odds ratio loss weight
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        gradient_checkpointing=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting ORPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

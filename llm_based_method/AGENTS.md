# AGENTS.md

This project fine-tunes `Qwen/Qwen3-8B` with LoRA for Chinese e-commerce review opinion quadruple extraction.

## Start Here

Before doing anything else, read:

- `/root/.codex/memories/llm_based_method.md`

That memory file contains the current project state, key paths, known issues, and the expected workflow.

## Working Directory

Always work from:

```bash
cd /root/autodl-tmp/.autodl/nlp/llm_based_method
```

## Environment

Always activate the conda env before running Python:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm
export HF_ENDPOINT=https://hf-mirror.com
```

## Main Workflow

Run tasks in this order unless the user asks otherwise:

1. `python build_sft_dataset.py`
2. `python finetune_lora.py`
3. `python merge.py`
4. `python infer.py`
5. `python fix_result_format.py`

## Important Project Facts

- This project uses LoRA, not full fine-tuning
- Base model: `Qwen/Qwen3-8B`
- Best adapter checkpoint: `qwen3_8b_opinion_lora_fp16/checkpoint-1212`
- Merged model path: `/root/autodl-tmp/huggingface/models/qwen3_8b_opinion_merged`
- Final result path: `/root/autodl-tmp/.autodl/nlp/llm_based_method/Result.csv`

## Output Rules

- Submission file must be a headerless UTF-8 CSV
- Columns: `ID, AspectTerms, OpinionTerms, Categories, Polarities`
- Keep all test IDs
- Use `_` for missing fields
- Preserve ID ascending order

## Guardrails

- Do not switch to `Qwen/Qwen3-8B-Instruct`
- If changing training or inference behavior, inspect the current scripts first
- Prefer plain filesystem paths in notes and commands

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "llama3_8b_opinion_lora_fp16/checkpoint-404"

# ★★★ 改这里：保存到系统盘 ★★★
MERGED_DIR = "/root/autodl-tmp/huggingface/models/llama3_8b_opinion_merged"

os.makedirs(MERGED_DIR, exist_ok=True)

def main():
    print("==> Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,     # 新写法
        device_map="auto",
    )

    print("==> Loading LoRA adapter from", ADAPTER_DIR)
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("==> Merging LoRA into base weights...")
    merged_model = lora_model.merge_and_unload()

    print("==> Moving model to CPU for saving...")
    merged_model = merged_model.to("cpu")

    print("==> Saving merged model to", MERGED_DIR)
    merged_model.save_pretrained(MERGED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Done. Merged model saved at:", MERGED_DIR)

if __name__ == "__main__":
    main()

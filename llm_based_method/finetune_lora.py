# finetune_lora_fp16.py  （你可以仍然叫 f.py）

import json
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "train_sft.jsonl"      # 上一步 build_sft_dataset.py 生成的文件
OUTPUT_DIR = "llama3_8b_opinion_lora_fp16"

# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. LoRA 配置（不变）
peft_config = LoraConfig(
    r=64, #r越大可能效果越好
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 3. 数据集
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def formatting_prompts_func(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(formatting_prompts_func)

# 4. 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # 为了保险一点，先用 1，看显存再调大 4/8/16
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,                         # V100 不支持 bf16
    fp16=True,                          # 用 fp16
    optim="adamw_torch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
)

# 5. SFTTrainer：不再传 BitsAndBytesConfig、quantization_config
trainer = SFTTrainer(
    model=MODEL_ID,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,
    args=args,
    peft_config=peft_config,
    model_init_kwargs={
        "torch_dtype": "float16",
        "device_map": "auto",
        "use_cache": False,
    },
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("LoRA fp16 微调完成，保存到：", OUTPUT_DIR)

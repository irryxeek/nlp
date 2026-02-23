import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 配置路径
MODEL_PATH = "/root/autodl-tmp/huggingface/models/llama3_8b_opinion_merged"
TEST_DATA_PATH = "test/Test_reviews.csv" 
OUTPUT_CSV_PATH = "Result.csv"

# 系统提示词（保持一致）
SYSTEM_PROMPT = """你是一个电商评论分析助手。
对于给定的中文商品评论，抽取其中所有的观点四元组：
{属性特征词 AspectTerm, 观点词 OpinionTerm, 情感极性 Polarity, 属性种类 Category}。
要求：
1. AspectTerm 和 OpinionTerm 必须从原始评论中逐字拷贝，不能改写；如果没有明确的属性特征词或观点词，用 "_"。
2. Polarity 使用标注中的中文极性标签（例如：正面、负面、中性等）。
3. Category 使用标注中的属性种类标签（例如：整体、价格、功效、包装等）。
4. 一条评论中可能有多个四元组。
5. 如果没有可抽取的四元组，则返回空列表。
6. 最终输出必须是一个 JSON 对象：
   {"tuples": [{"aspect": "...","opinion": "...","polarity": "...","category": "..."}, ...]}
   不要输出任何说明文字。
"""

def main():
    print("==> Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # df_test = pd.read_csv(TEST_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    results = []

    print(f"==> Running inference on {len(df_test)} samples...")
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        rid = row["id"]
        review = row["Reviews"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"评论：{review}\n请按要求输出 JSON。"}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            clean_json = response[start_idx:end_idx]
            
            data = json.loads(clean_json)
            tuples = data.get("tuples", [])
            
            if not tuples:
                results.append([rid, "_", "_", "_", "_"])
            else:
                for t in tuples:
                    # 严格按照你要求的顺序排列字段：AspectTerm, OpinionTerm, Category, Polarity
                    results.append([
                        rid,
                        t.get("aspect", "_"),
                        t.get("opinion", "_"),
                        t.get("category", "_"),
                        t.get("polarity", "_")
                    ])
        except Exception:
            results.append([rid, "_", "_", "_", "_"])

    # 保存为无表头的 CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False, header=False, encoding="utf-8")
    print(f"完成！预测结果已保存至（无表头）：{OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
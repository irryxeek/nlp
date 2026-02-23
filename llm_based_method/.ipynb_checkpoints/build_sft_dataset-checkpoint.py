# build_sft_dataset.py
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

TRAIN_REVIEWS = "train/TRAIN/Train_reviews.csv"
TRAIN_LABELS = "train/TRAIN/Train_labels.csv"
OUT_PATH = "train_sft.jsonl"

# 1. 读入数据
df_r = pd.read_csv(TRAIN_REVIEWS)   # id, Reviews
df_l = pd.read_csv(TRAIN_LABELS)    # id, AspectTerms, OpinionTerms, Categories, Polarities, ...

# 2. 按 id 聚合 label
labels_by_id = defaultdict(list)
for _, row in df_l.iterrows():
    rid = int(row["id"])
    aspect = str(row["AspectTerms"]).strip()
    opinion = str(row["OpinionTerms"]).strip()
    category = str(row["Categories"]).strip()
    polarity = str(row["Polarities"]).strip()
    # 统一用 "_" 表示空（和官方一致）
    if aspect == "" or aspect.lower() == "nan":
        aspect = "_"
    if opinion == "" or opinion.lower() == "nan":
        opinion = "_"
    labels_by_id[rid].append({
        "aspect": aspect,
        "opinion": opinion,
        "category": category,
        "polarity": polarity,
    })

# 3. 构造 chat 样本
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

def build_answer_tuples(rid: int):
    tuples = labels_by_id.get(rid, [])
    return {"tuples": tuples}

out_path = Path(OUT_PATH)
with out_path.open("w", encoding="utf-8") as f_out:
    for _, row in df_r.iterrows():
        rid = int(row["id"])
        review = str(row["Reviews"])

        # user 提示
        user_content = f"评论：{review}\n请按要求输出 JSON。"

        # assistant 标准答案（JSON 字符串）
        label_json = build_answer_tuples(rid)
        assistant_content = json.dumps(label_json, ensure_ascii=False)

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"已写出微调数据到 {OUT_PATH}")

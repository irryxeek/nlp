import os
import json
import pandas as pd
from tqdm.auto import tqdm

from google import genai  # 新增
from google.genai import types  # 目前没用到高级配置，也可以先不导入

# ---- Gemini Client 初始化 ----
# 建议通过环境变量 GEMINI_API_KEY 传 key，而不是写死在代码里
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY is None:
    raise RuntimeError("请先在环境变量中设置 GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"  # 使用最新的 Gemini 2.5 Flash 模型


CATEGORIES = [
    "物流", "尺寸", "价格", "质量", "包装", "外观", "材质", "做工",
    "服务", "使用体验", "气味", "功能", "其他"
]

SYSTEM_PROMPT = f"""你是一个电商评论分析助手。
对于给定的中文商品评论，抽取其中所有的观点四元组：
{{属性特征词 AspectTerm, 观点词 OpinionTerm, 情感极性 Polarity, 属性种类 Category}}。
要求：
1. AspectTerm 和 OpinionTerm 必须从原始评论中逐字拷贝，不能改写。
2. Polarity 只能取“正面”“负面”“中性”三类。
3. Category 必须从下面的集合中选择一个：
{{{", ".join(CATEGORIES)}}}。
4. 一条评论中可能有多个四元组。
5. 如果没有可抽取的四元组，则返回空列表。
6. 最终输出必须是一个 JSON 对象：
   {{"tuples": [{{"aspect": "...","opinion": "...","polarity": "...","category": "..."}}, ...]}}
   不要输出任何说明文字。
"""

# few-shot 示例仍然保留，只是不用 chat 模式，而是串到一个大 prompt 里
FEWSHOT = [
    {
        "role": "user",
        "content": "评论：这家店快递很快，尺寸刚好，给 5 分好评！\n请按要求输出 JSON。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "tuples": [
                {"aspect": "快递", "opinion": "很快", "polarity": "正面评价", "category": "物流"},
                {"aspect": "尺寸", "opinion": "刚好", "polarity": "正面评价", "category": "尺寸"},
            ]
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "评论：这个大小不太合适，勉强用。\n请按要求输出 JSON。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "tuples": [
                {"aspect": "大小", "opinion": "不太合适", "polarity": "负面评价", "category": "尺寸"},
            ]
        }, ensure_ascii=False)
    },
]


def build_prompt(review: str) -> str:
    """
    把 system 指令 + few-shot 示例 + 当前评论
    拼成一个给 Gemini 的大文本 prompt。
    """
    parts = [SYSTEM_PROMPT.strip(), ""]
    for turn in FEWSHOT:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            parts.append(f"用户：{content}")
        else:
            parts.append(f"助手：{content}")
    parts.append(f"用户：评论：{review}\n请按要求输出 JSON。")
    return "\n\n".join(parts)


def call_llm(review: str) -> str:
    """
    调用 Gemini 2.5 Flash，对单条评论进行抽取。
    """
    prompt = build_prompt(review)

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        # 如果你想调温度等，可以加 config 参数：
        # config=types.GenerateContentConfig(
        #     temperature=0.1,
        # )
    )
    # Gemini SDK 提供 resp.text 汇总文本
    return (resp.text or "").strip()


def parse_output(raw: str):
    """
    解析模型输出的 JSON。
    出错时返回空列表。
    同时兼容：
    - polarity: 「正面评价/负面评价/中性评价」
    - 或简写：「正面/负面/中性」
    """
    try:
        data = json.loads(raw)
        tuples = data.get("tuples", [])
        if not isinstance(tuples, list):
            return []

        cleaned = []
        for t in tuples:
            a = str(t.get("aspect", "")).strip()
            o = str(t.get("opinion", "")).strip()
            p = str(t.get("polarity", "")).strip()
            c = str(t.get("category", "")).strip()
            if not a or not o or not p or not c:
                continue

            # 统一极性写法
            polarity_map = {
                "正面": "正面评价",
                "负面": "负面评价",
                "中性": "中性评价",
                "正面评价": "正面评价",
                "负面评价": "负面评价",
                "中性评价": "中性评价",
            }
            if p not in polarity_map:
                continue
            p_std = polarity_map[p]

            if c not in CATEGORIES:
                continue
            cleaned.append((a, o, p_std, c))
        return cleaned
    except Exception:
        # JSON 解析失败直接当作没抽到
        return []


def run_inference(test_reviews_path: str, out_path: str):
    df = pd.read_csv(test_reviews_path)

    # 目前你脚本只跑前 100 条
    df = df.iloc[:100]

    ids = df["id"].tolist()
    reviews = df["Reviews"].astype(str).tolist()

    results = []

    for rid, text in tqdm(zip(ids, reviews), total=len(ids), desc="Running Gemini 2.5 Flash"):
        raw = call_llm(text)
        tuples = parse_output(raw)

        if not tuples:
            results.append([rid, "_", "_", "_", "_"])
        else:
            for (a, o, p, c) in tuples:
                # 注意你之前列名是：
                # ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"]
                results.append([rid, a, o, c, p])

    res_df = pd.DataFrame(
        results,
        columns=["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"],
    )
    res_df = res_df.sort_values(by=["ID"]).reset_index(drop=True)
    res_df.to_csv(out_path, index=False, header=False, encoding="utf-8")


if __name__ == "__main__":
    run_inference("TEST/Test_reviews.csv", "Result.csv")

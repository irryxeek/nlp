import pandas as pd

# ===== 修改这里为你的 result 路径 =====
input_csv = "Result.csv"
output_csv = "Result_fixed.csv"   # 你也可以写成 input_csv 覆盖原文件

# Result.csv 无表头，所以手动指定列名
df = pd.read_csv(input_csv, header=None,
                 names=["ID", "AspectTerms", "OpinionTerms", "Polarities", "Categories"])

# 交换列顺序：先 Categories 后 Polarities
df = df[["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"]]

# 保存（无表头，UTF-8）
df.to_csv(output_csv, index=False, header=False, encoding="utf-8")

print("列顺序调整完毕，输出文件：", output_csv)

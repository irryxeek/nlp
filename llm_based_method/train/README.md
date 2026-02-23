下面是**整理后的 Markdown（md）格式文档**，结构清晰、可直接用于 README / 竞赛说明文档。



# 电商评论观点挖掘（Aspect-Based Sentiment Analysis）

## 原链接

https://tianchi.aliyun.com/competition/entrance/532421/information



## Baseline

* GitHub 地址：
  👉 [https://github.com/eguilg/OpinioNet](https://github.com/eguilg/OpinioNet)



## 一、竞赛题目背景

随着电商行业的迅速发展，商品评论数据成为影响消费者购买决策的重要因素。消费者在购买商品时，往往会参考其他用户的评论内容，从而调整自己的消费意愿。这使得**电商评论数据对商品销量和商家利益产生直接影响**。

电商平台沉淀了大量评论文本，其中蕴含着消费者对商品属性的真实观点。这些信息可用于：

* 舆情分析
* 用户需求理解
* 产品优化
* 营销决策

### 任务目标

本次竞赛任务为**品牌评论观点挖掘**，需要从商品评论中：

* 抽取 **商品属性特征**
* 抽取 **消费者观点**
* 判断 **情感极性**
* 确定 **属性种类**

最终对测试数据输出以下 **4 个字段**：

1. 属性特征词（AspectTerm）
2. 观点词（OpinionTerm）
3. 观点极性（Polarity）
4. 属性种类（Category）

> 当前赛季仅开放 **初赛测评**。



## 二、数据说明

### 1. 样例说明

（示例略，具体见数据集）



### 2. 字段说明

#### （1）评论 ID（ID）

* 每条用户评论的唯一标识

#### （2）用户评论（Reviews）

* 用户对商品的原始评论文本

#### （3）属性特征词（AspectTerms）

* 评论中提及的商品属性
* 示例：

  * “价格很便宜” → 属性特征词为 **价格**
* 要求：

  * 必须与原文表述保持一致

#### （4）观点词（OpinionTerms）

* 用户对某一属性的观点描述
* 示例：

  * “价格很便宜” → 观点词为 **很便宜**
* 要求：

  * 必须与原文表述保持一致

#### （5）观点极性（Polarity）

* 属性观点所体现的情感倾向：

  * 负面
  * 中性
  * 正面

#### （6）属性种类（Category）

* 将相似或同类的属性特征归为一个属性种类
* 示例：

  * “快递”“物流” → 归入 **物流** 类
* 属性种类集合已给定（详见训练数据 `README.md`）



## 三、训练数据说明

### 1. 数据目录结构

```text
TRAIN/
├── README.md
├── Train_reviews.csv
└── Train_labels.csv
```



### 2. 文件说明

#### A. `Train_reviews.csv`

* 包含：

  * 评论 ID
  * 评论原文

#### B. `Train_labels.csv`

* 包含：

  * 评论 ID
  * 标注结果

##### 标注规则说明

1. **ID 字段**

   * 唯一标识一条评论及其标注数据

2. **四元组表示**

   * 使用以下四元组表示一个观点：

     ```
     (AspectTerm, OpinionTerm, Polarity, Category)
     ```
   * 若某字段为空，统一使用下划线 `_`

3. **多观点情况**

   * 同一条评论中可能存在多个四元组
   * 各四元组相互独立

4. **数据噪声**

   * 评论中可能存在：

     * 不规范符号
     * 错别字
     * 非标准表达



## 四、结果提交说明

### 1. 测试数据目录结构

```text
TEST/
├── README.md
├── Test_reviews.csv
└── Result(example).csv
```



### 2. 文件说明

#### A. `Test_reviews.csv`

* 包含：

  * 评论 ID
  * 评论原文
* 格式参考训练数据

#### B. `Result(example).csv`

* 提交结果示例文件
* 参赛者需按该格式生成：

  * **Result.csv**



### 3. 提交要求（重要）

* 必须提交 **无 BOM 的 UTF-8 编码 `.csv` 文件**
* **不需要表头**
* **不能遗漏任何 Test_reviews.csv 中的 ID**
* ID 必须 **升序排列**
* 若某 ID 的预测结果全部为空：

  * 仍需保留该 ID
  * 其他字段用 `_` 填充



## 五、评分标准

### 1. 四元组匹配规则

在同一 ID 内：

* 若 **AspectTerm、OpinionTerm、Category、Polarity** 四个字段均正确
* 则该四元组判为 **正确**



### 2. 评价指标定义

* 预测四元组总数：
  **P**
* 真实四元组总数：
  **G**
* 正确预测四元组数：
  **S**

#### （1）精确率（Precision）

$$
Precision = \frac{S}{P}
$$

#### （2）召回率（Recall）

$$
Recall = \frac{S}{G}
$$

#### （3）F1 值（最终评分指标）

$$
F1\text{-}score = \frac{2 \times Precision \times Recall}{Precision + Recall}
$$




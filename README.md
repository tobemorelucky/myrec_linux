# myrectest

本仓库是一个面向**多兴趣序列推荐（Multi-Interest Sequential Recommendation）**的 PyTorch 实验代码库。代码以 PoMRec 为基础框架，并在此基础上加入了大语言模型语义嵌入、语义-协同空间对齐、残差式语义融合、目标兴趣一致性约束以及意图引导式历史行为软去噪等模块。
> 注意：本 README 按当前仓库代码重新整理，旧 README 中的说明可能与当前代码不完全一致。
---
## 1. 项目简介
| 模型 | 说明 |
|---|---|
| `PoMRec` | PoMRec 主干模型，并支持可选的大语言模型语义嵌入对齐与融合 |
| `MyModel` | 在 PoMRec + LLM 语义增强基础上，进一步加入目标兴趣一致性约束与意图引导式历史行为软去噪 |
统一入口文件为：
```bash
python main.py --model_name <MODEL_NAME> --dataset <DATASET_NAME>
```
其中 `<MODEL_NAME>` 可选：
```text
PoMRec         — baseline with optional LLM semantic enhancement
MyModel        — paper version: PoMRec + LLM + IPD + LGD
MyModelV2      — PoMRec + LLM + IU-SCBR + UGC-TIC + old IPD
MyModelV4      — PoMRec + LLM + DSIP + old IPD
MyModelV5      — PoMRec + LLM + SSID (auxiliary loss) + old IPD
SIERec         — PoMRec + LLM + MVTC (clean skeleton)
```
---
## 2. 代码结构
```text
.
├── main.py
├── helpers/
│   ├── BaseReader.py
│   ├── SeqReader.py
│   └── BaseRunner.py
├── models/
│   ├── BaseModel.py
│   └── sequential/
│       ├── PoMRec.py
│       └── MyModel.py
└── utils/
    └── utils.py
```
主要文件说明如下：
| 文件 | 作用 |
|---|---|
| `main.py` | 项目统一训练、验证、测试入口 |
| `helpers/BaseReader.py` | 读取 `train.csv`、`dev.csv`、`test.csv` 数据文件 |
| `helpers/SeqReader.py` | 根据时间顺序构建用户历史序列，并补充交互位置 |
| `helpers/BaseRunner.py` | 控制训练、验证、测试、Early Stop 和指标计算 |
| `models/BaseModel.py` | 定义基础模型、序列数据集构造、负采样和 BPR 损失 |
| `models/sequential/PoMRec.py` | PoMRec 主干模型，支持 LLM 语义对齐与融合 |
| `models/sequential/MyModel.py` | 扩展模型，支持 IPD 约束与意图引导式历史行为软去噪 |
| `utils/utils.py` | 随机种子、日志格式化、DataFrame 处理、GPU 数据转移等工具函数 |
---
## 3. 模型说明

### 3.1 PoMRec 主干

`PoMRec` 是基础多兴趣序列推荐模型，主要包含：

1. 多兴趣提取器（Multi-Interest Extractor）
2. 兴趣分布预测器（Interest Distribution Predictor）
3. 提取器与聚合器对应的 Prompt Embeddings（提示向量）
4. 中心性-离散性结合的兴趣表示
5. 基于 BPR 的 pairwise ranking loss（成对排序损失）

模型会从用户历史序列中提取多个兴趣向量，再预测当前上下文下各兴趣的重要性权重，最终聚合成用户表示并对候选物品打分。
---
### 3.2 LLM 语义对齐与语义融合
当前 `PoMRec.py` 已经支持使用离线构建的大语言模型物品语义嵌入表。
该部分包含：
1. 从 `.pkl` 文件加载 LLM item embedding table（物品语义嵌入表）
2. 使用 Adapter（适配器）将 LLM 语义向量映射到推荐嵌入空间
3. 使用 InfoNCEAlign（对比对齐损失）约束语义表示与协同表示对齐
4. 可选加载预训练协同 item embedding 作为 anchor（锚点）
5. 使用残差式融合方式注入语义信息：

```text
item_embedding = collaborative_embedding + gamma * llm_embedding
```
相关参数如下：
| 参数 | 说明 |
|---|---|
| `--use_llmemb` | 是否启用 LLM 语义嵌入分支 |
| `--llm_emb_path` | LLM 物品语义嵌入 `.pkl` 文件路径 |
| `--srs_emb_path` | 预训练协同物品嵌入 `.pkl` 文件路径 |
| `--alpha` | 语义-协同对齐损失权重 |
| `--tau` | InfoNCE 对比损失温度系数 |
| `--rat_alpha_warmup_steps` | 对齐损失 warmup 步数 |
| `--llm_fuse` | 是否将 LLM 语义向量融合进 item embedding |
| `--gamma_init` | 语义融合系数 gamma 初始值 |
| `--gamma_trainable` | gamma 是否可训练 |
| `--init_ckpt` | 可选的 warm-start checkpoint |
| `--init_strict` | 是否严格加载 checkpoint |

---

### 3.3 MyModel 扩展模型

`MyModel` 在 LLM 增强版 PoMRec 基础上加入了两个主要扩展。

#### 3.3.1 EMILE-style IPD 目标兴趣约束

该模块用于增强目标物品与多兴趣表示之间的匹配关系，主要约束以下几类表示：

1. 聚合后的用户兴趣表示
2. 与目标物品最匹配的兴趣表示
3. 正样本目标物品
4. 负样本物品

相关参数：

| 参数 | 说明 |
|---|---|
| `--use_emile` | 是否启用 IPD 目标兴趣约束 |
| `--lambda_ipd` | IPD 损失权重 |
| `--ipd_margin` | IPD 排序约束 margin |
| `--emile_use_fused_itememb` | IPD 是否使用融合后的 item embedding |
| `--emile_warmup_steps` | IPD 损失 warmup 步数 |

---

#### 3.3.2 意图引导式历史行为软去噪

该模块用于对历史序列中的弱相关行为进行软重加权。

训练阶段流程如下：

1. 第一遍不使用去噪，先通过多兴趣提取器得到初步兴趣表示；
2. 根据初步兴趣权重聚合得到当前序列的 intent query（意图查询向量）；
3. 第二遍使用该 intent query 对历史行为进行相似度估计；
4. 根据相似度生成 gate（门控权重），对历史行为进行软去噪；
5. 再用去噪后的历史序列进行最终兴趣建模和推荐预测。

该设计不直接使用目标 item 构造去噪查询，因此可以减少 label leakage（标签泄露）风险。

相关参数：

| 参数 | 说明 |
|---|---|
| `--use_logic_denoise` | 是否启用意图引导式历史行为软去噪 |
| `--logic_denoise_alpha` | 去噪 gate 的斜率系数 |
| `--logic_denoise_warmup_steps` | 去噪强度 warmup 步数 |
| `--logic_denoise_topk` | 可选 top-k gate 增强 |
| `--logic_denoise_r` | 去噪残差比例 |
| `--logic_denoise_b` | gate 偏置阈值 |

> 说明：`logic_aggr` 相关参数在代码中仍然保留，用于兼容旧脚本和旧日志，但当前 `MyModel` 的 forward 路径中已经不再使用 logic aggregation，最终预测始终使用基础预测分数。

---

## 4. 数据格式

数据集应放在：

```text
data/<dataset_name>/
```

每个数据集目录下至少需要包含：

```text
train.csv
dev.csv
test.csv
```

每个文件至少需要包含以下字段：

```text
user_id
item_id
time
```

验证集和测试集可以额外包含：

```text
neg_items
```

默认分隔符为 tab：

```bash
--sep "\t"
```

如果你的数据文件是逗号分隔，则需要指定：

```bash
--sep ","
```

`SeqReader` 会合并 train/dev/test 中的交互记录，按照时间顺序构建用户历史序列，并为每条交互补充 `position` 信息。

## 6. 运行示例

### 6.1 训练基础 PoMRec

```bash
python main.py \
  --model_name PoMRec \
  --dataset ml-1m \
  --path ./data/ \
  --emb_size 64 \
  --attn_size 8 \
  --K 2 \
  --prompt_num 3 \
  --lamb 1 \
  --history_max 20 \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --epoch 200 \
  --early_stop 10
```

---

### 6.2 训练带 LLM 语义增强的 PoMRec

```bash
python main.py \
  --model_name PoMRec \
  --dataset beauty \
  --path ./data/ \
  --emb_size 64 \
  --attn_size 8 \
  --K 4 \
  --prompt_num 3 \
  --lamb 4.0 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_emb_path ./data/beauty/llm_emb.pkl \
  --srs_emb_path ./data/beauty/srs_emb.pkl \
  --alpha 0.001 \
  --tau 0.2 \
  --rat_alpha_warmup_steps 5000 \
  --llm_fuse 1 \
  --gamma_init 0.1 \
  --gamma_trainable 0 \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --epoch 200 \
  --early_stop 10
```

---

### 6.3 训练 MyModel

```bash
python main.py \
  --model_name MyModel \
  --dataset beauty \
  --path ./data/ \
  --emb_size 64 \
  --attn_size 8 \
  --K 4 \
  --prompt_num 3 \
  --lamb 4.0 \
  --history_max 20 \
  --use_llmemb 1 \
  --llm_emb_path ./data/beauty/llm_emb.pkl \
  --srs_emb_path ./data/beauty/srs_emb.pkl \
  --alpha 0.001 \
  --tau 0.2 \
  --rat_alpha_warmup_steps 5000 \
  --llm_fuse 1 \
  --gamma_init 0.1 \
  --gamma_trainable 0 \
  --use_emile 1 \
  --lambda_ipd 0.05 \
  --ipd_margin 0.2 \
  --emile_warmup_steps 5000 \
  --use_logic_denoise 1 \
  --logic_denoise_alpha 1.0 \
  --logic_denoise_warmup_steps 5000 \
  --logic_denoise_topk 0 \
  --logic_denoise_r 0.15 \
  --logic_denoise_b 0.0 \
  --lr 0.001 \
  --l2 1e-6 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --epoch 200 \
  --early_stop 10
```

---

## 7. 常用参数说明

### 7.1 全局参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model_name` | `PoMRec` | 选择运行的模型 |
| `--gpu` | `0` | 指定 CUDA_VISIBLE_DEVICES |
| `--random_seed` | `1` | 随机种子 |
| `--load` | `0` | 是否加载已有模型继续训练 |
| `--train` | `1` | 是否训练模型 |
| `--regenerate` | `0` | 是否重新生成中间缓存文件 |
| `--log_file` | 空 | 日志保存路径 |
| `--model_path` | 空 | 模型保存路径 |

### 7.2 数据参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--path` | `./data/` | 数据根目录 |
| `--dataset` | `ml-1m` | 数据集名称 |
| `--sep` | `\t` | CSV 文件分隔符 |

### 7.3 训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--epoch` | `200` | 最大训练轮数 |
| `--early_stop` | `10` | Early stopping 参数 |
| `--lr` | `1e-3` | 学习率 |
| `--l2` | `1e-6` | 权重衰减 |
| `--batch_size` | `256` | 训练 batch size |
| `--eval_batch_size` | `256` | 验证/测试 batch size |
| `--optimizer` | `Adam` | 优化器 |
| `--num_workers` | `5` | DataLoader worker 数 |
| `--topk` | `5,10,20,50` | 评估 Top-K |
| `--metric` | `NDCG,HR` | 评估指标 |

### 7.4 PoMRec 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--emb_size` | `64` | 物品嵌入维度 |
| `--attn_size` | `8` | 注意力隐层维度 |
| `--K` | `3` | 兴趣数量 |
| `--prompt_num` | `4` | prompt 向量数量 |
| `--n_layers` | `1` | 兴趣分布投影层数 |
| `--lamb` | `3` | 离散性项权重 |
| `--history_max` | `20` | 最大历史序列长度 |

### 7.5 LLM 语义增强参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use_llmemb` | `0` | 是否启用 LLM 语义分支 |
| `--llm_emb_path` | 空 | LLM item embedding `.pkl` 路径 |
| `--srs_emb_path` | 空 | 预训练协同 item embedding `.pkl` 路径 |
| `--alpha` | `0.001` 或 `0.01` | 对齐损失权重，不同模型默认值略有不同 |
| `--tau` | `0.2` | InfoNCE 温度 |
| `--rat_alpha_warmup_steps` | `0` 或 `5000` | 对齐损失 warmup 步数 |
| `--llm_fuse` | `1` | 是否融合 LLM 表示 |
| `--gamma_init` | `0.05` 或 `0.1` | 语义融合系数初始值 |
| `--gamma_trainable` | `0/1` | gamma 是否可训练 |
| `--init_ckpt` | 空 | warm-start checkpoint |
| `--init_strict` | `0` | 是否严格加载 checkpoint |

### 7.6 MyModel 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use_emile` | `0` | 是否启用 IPD 目标兴趣约束 |
| `--lambda_ipd` | `0.05` | IPD 损失权重 |
| `--ipd_margin` | `0.2` | IPD 排序 margin |
| `--emile_use_fused_itememb` | `0` | IPD 是否使用融合 item embedding |
| `--emile_warmup_steps` | `5000` | IPD warmup 步数 |
| `--use_logic_denoise` | `0` | 是否启用意图引导式软去噪 |
| `--logic_denoise_alpha` | `1.0` | gate 斜率 |
| `--logic_denoise_warmup_steps` | `5000` | 去噪 warmup 步数 |
| `--logic_denoise_topk` | `0` | top-k gate 增强 |
| `--logic_denoise_r` | `0.15` | 去噪残差比例 |
| `--logic_denoise_b` | `0.0` | gate 阈值偏置 |

---

## 8. 评估指标

当前代码支持：

```text
HR@K
NDCG@K
```

默认评估：

```text
HR@5, HR@10, HR@20, HR@50
NDCG@5, NDCG@10, NDCG@20, NDCG@50
```

---

## 9. 日志与模型保存

如果没有手动指定 `--log_file`，日志会自动保存到：
```text
./log/<model_name>/
```
如果没有手动指定 `--model_path`，模型会自动保存到：
```text
./model/<model_name>/
```
训练过程中会根据验证集主指标保存最优模型，并在最终测试前重新加载最优 checkpoint。

---
## 10. 注意事项

1. 使用 `--use_llmemb 1` 时，必须提供 `--llm_emb_path`。
2. LLM embedding 文件应为二维 `.pkl` 数组。
3. LLM embedding 表可以包含 padding 行，也可以不包含；代码会自动处理 index 0。
4. `--srs_emb_path` 不是必须的，但如果使用语义对齐，建议提供预训练协同 embedding 作为稳定 anchor。
5. `MyModel` 中的 `logic_aggr` 相关参数当前只用于兼容旧脚本，不参与 forward 预测。
6. 如果使用 GPU，确保 PyTorch 与 CUDA 版本匹配。
7. 如果修改了数据文件，建议使用 `--regenerate 1` 重新生成缓存 corpus。
8. 如果日志文件名过长，`main.py` 会自动截断并添加 hash，避免系统路径长度问题。
## 运行实验

本项目的主要实验脚本均为 `.sh` 文件，建议在 Linux 服务器、WSL 或 Git Bash 环境下运行。

### 1. 完整三模块多种子主实验

完整模型包含以下三个模块：

1. LLM Semantic Alignment and Fusion（大语言模型语义对齐与融合）
2. IPD Target-Interest Constraint（目标兴趣一致性约束）
3. Logic-Guided Denoising, LGD（意图引导式历史行为软去噪）

三个数据集对应的最终多种子运行脚本如下：

```bash
# Beauty dataset
bash run_beauty_multiseed_final.sh

# ML-1M dataset
bash run_ml1m_full3_multiseed.sh

# Toys dataset
bash run_toys_final_multiseed_best.sh
```

其中，三个脚本均会运行多个随机种子：

```text
0, 1, 2, 3, 41, 42, 43
```

推荐将这三个脚本作为论文中完整模型的主实验结果来源。

### 2. 三数据集 all-off 消融实验

如果需要运行三个数据集的 all-off 消融实验，可以使用：

```bash
bash run_alloff_s42_3datasets.sh
```

该脚本会依次在以下三个数据集上运行：

```text
beauty
ml-1m
toys
```

但需要注意，该脚本只使用：

```text
random_seed = 42
```

并且关闭所有增强模块：

```text
use_llmemb = 0
use_emile = 0
use_logic_denoise = 0
```

因此它更适合作为 all-off 消融或基础对照实验，而不是最终完整模型的多种子主实验。

### 2.1 论文版主实验最佳超参数

以下超参数从 `bash脚本/` 中的多种子脚本提取，对应 **MyModel（论文版）**：PoMRec + LLM语义对齐 + IPD目标兴趣约束 + LGD意图引导式去噪。

三个脚本均使用 seeds: `0 1 2 3 41 42 43`，warm-start 自 PoMRec checkpoint。

| 参数 | Beauty | ML-1M | Toys |
|------|--------|-------|------|
| `--lr` | 0.002 | 0.001 | 0.001 |
| `--l2` | 1e-6 | 1e-6 | 1e-6 |
| `--lamb` | 3.0 | 3.0 | 3.8 |
| `--K` / `--prompt_num` | 3 / 4 | 3 / 4 | 3 / 4 |
| `--emb_size` / `--attn_size` | 64 / 8 | 64 / 8 | 64 / 8 |
| `--history_max` | 20 | 20 | 20 |
| **LLM** | | | |
| `--use_llmemb` / `--llm_fuse` | 1 / 1 | 1 / 1 | 1 / 1 |
| `--gamma_init` / `--gamma_trainable` | 0.1 / 0 | 0.08 / 0 | 0.05 / 0 |
| `--alpha` (align weight) | 0.001 | 0.001 | 0.001 |
| `--tau` (align temp) | 0.2 | 0.3 | 0.5 |
| `--rat_alpha_warmup_steps` | 5000 | 5000 | 5000 |
| **IPD** | | | |
| `--use_emile` | 1 | 1 | 1 |
| `--lambda_ipd` | 0.05 | 0.02 | 0.05 |
| `--ipd_margin` | 0.2 | 0.10 | 0.10 |
| `--emile_warmup_steps` | 5000 | 20000 | 20000 |
| **LGD** | | | |
| `--use_logic_denoise` | 1 | 1 | 1 |
| `--logic_denoise_alpha` | 8.0 | 8.0 | 10 |
| `--logic_denoise_b` | 0.3 | 0.40 | 0.3 |
| `--logic_denoise_topk` | 5 | 5 | 10 |
| `--logic_denoise_r` | 0.15 | 0.08 | 0.10 |
| `--logic_denoise_warmup_steps` | 20000 | 50000 | 50000 |
| **其他** | | | |
| `--use_logic_aggr` | 0 | 0 | 0 |
| `--lambda_logic_aggr` | 0.0 | 0.0 | 0.0 |
| LLM emb path | `./data/beauty/handled/llm_table_pca1536.pkl` | `./data/ml-1m/handled/llm_table_pca1536.pkl` | `./data/toys/handled/llm_table_pca1536.pkl` |
| SRS emb path | `./data/beauty/handled/itm_emb_pomrec.pkl` | `./data/ml-1m/handled/itm_emb_pomrec.pkl` | `./data/toys/handled/itm_emb_pomrec.pkl` |

### 3. GPU 注意事项

不同脚本中已经手动指定了 `CUDA_VISIBLE_DEVICES`。

当前推荐脚本中：
[README_core.md](README_core.md)
```text
run_beauty_multiseed_final.sh      使用 GPU 0
run_ml1m_full3_multiseed.sh        使用 GPU 1
run_toys_final_multiseed_best.sh   使用 GPU 1[README_core.md](README_core.md)
```

因此，如果只有一张 GPU，建议按顺序运行三个脚本。

如果有多张 GPU，可以同时运行 Beauty 和 ML-1M；但 ML-1M 和 Toys 默认都使用 GPU 1，不建议直接同时运行，除非手动修改其中一个脚本的 `CUDA_VISIBLE_DEVICES`。

### 4. 运行前检查

运行脚本前，建议确认以下内容：

```bash
# 检查数据目录
ls ./data/

# 检查模型保存目录
ls ./model/

# 检查日志目录
ls ./log/
```


然后再执行对应脚本。

### 5. 日志与模型输出

实验日志会保存到：

```text
./log/MyModel/<dataset_name>/
```

模型 checkpoint 会保存到：

```text
./model/MyModel/<dataset_name>/
```
不同脚本会自动创建对应的子目录，用于区分不同数据集、不同模块组合和不同随机种子。

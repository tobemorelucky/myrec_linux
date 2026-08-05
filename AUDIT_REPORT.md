# 仓库只读审计报告

审计日期: 2026-08-05
仓库: /home/yyx/hzgcode/pom2.0
Git commit: dac9aa3 ("大改前最后一次提交")
分支: main (clean working tree)

---

## 一、训练框架

### 1.1 main.py 模型加载机制

**文件**: `main.py:125-130`

```python
model_name = eval('{0}.{0}'.format(init_args.model_name))
reader_name = eval('{0}.{0}'.format(model_name.reader))
runner_name = eval('{0}.{0}'.format(model_name.runner))
```

- `--model_name` 通过 `eval('PoMRec.PoMRec')` 动态获取类对象。
- 每个模型类必须定义 `reader`, `runner`, `extra_log_args` 三个类属性。
- 所有模型在 `main.py:20-37` 硬编码 import。当前 import 的模型（17个）：PoMRec, PoMRecLLMEmb, PoMRecLLMEmbLinear, MyModel, SIERec, MyModelV2, MyModelV4, MyModelV5, MyModelLLM, MyModelLLMIPD, MyModelSCIL, MyModelHMIF, MyModelTIRL, MyModelCTIRL, MyModelITIRL, MyModelCIRF, MyModelSHNC。
- `models/sequential/MyModel（论文版）.py` 存在但未在 main.py 中 import。

**日志和模型路径** (main.py:141-155): 由 `log_args` 拼接后 MD5 截断，自动创建 `./log/<ModelName>/` 和 `./model/<ModelName>/` 目录。

---

### 1.2 模型继承链 (`models/BaseModel.py`)

```
BaseModel (nn.Module)
  └─ GeneralModel: 添加 BPR loss, num_neg, item_num/user_num, 负采样
       └─ SequentialModel: 使用 SeqReader, 添加 history_max, 筛选 position>0 的样本
            └─ PoMRec / MyModel / SIERec / MyModelHMIF / ...  (所有具体模型)
```

**关键实现细节**:

1. **GeneralModel.Dataset.actions_before_epoch** (`BaseModel.py:207-215`):
   - 每个 epoch 前重新随机采样负样本: `np.random.randint(1, n_items, ...)`
   - 排除 `train_clicked_set[u]`，但**不排除 dev/test 中的已点击项**
   - 这是训练期数据泄漏的一个潜在来源（论文中常见做法）

2. **GeneralModel.loss** (`BaseModel.py:176-189`): BPR 损失含 softmax-based negative sampling:
   ```python
   pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
   neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
   loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
   ```

3. **SequentialModel.Dataset.__init__** (`BaseModel.py:232-236`):
   - 过滤 `position == 0` 的样本（历史为空时无法构建兴趣），这会改变 len(dataset)

4. **SequentialModel.Dataset._get_feed_dict** (`BaseModel.py:238-247`):
   - 取 `user_his[uid][:pos]` 作为历史序列
   - 截断到 `history_max`

5. **BaseModel.Dataset.collate_batch** (`BaseModel.py:135-152`):
   - 变长序列用 `pad_sequence` 处理
   - 等长序列直接 `np.stack`

**在 Dataset 内增加 future_items 的可行性**: 可以在新模型的嵌套 `Dataset` 类中覆盖 `_get_feed_dict`，通过 `self.corpus.user_his[uid][pos+1:pos+1+H]` 获取未来的 H 个交互，**不需要修改 Reader 或 Runner**。但需注意：
   - `SequentialModel.Dataset.__init__` 过滤了 position==0，所以过滤后的 dataset 不会包含无历史样本，但 future_items 为空的样本（position 接近 user_his 末尾）仍会存在，需要在 loss 里正确处理 mask
   - 不需要修改 `collate_batch` 因为 future_items 是等长的（H 个）

---

### 1.3 BaseRunner 训练/验证/测试流程 (`helpers/BaseRunner.py`)

**train()** (`BaseRunner.py:114-169`):
1. 每 epoch 调用 `dataset.actions_before_epoch()` 重采样负样本
2. 调用 `model.actions_before_epoch()` (如果存在) 调整冻结策略
3. `self.fit()` 训练
4. `self.evaluate(data_dict['dev'], topk[:1], metrics)` 验证
5. 保存 best model (`main_metric` 最佳时保存)
6. Early stop 条件: 连续 `early_stop` 次 dev 结果不增长，或最佳 epoch 距今超过 20

**fit()** (`BaseRunner.py:171-194`):
- 首次调用时 `_build_optimizer`
- 使用 DataLoader 批量训练
- 支持 grad_clip

**evaluate()** (`BaseRunner.py:203-209`): 调用 predict 后计算 metrics。

**predict()** (`BaseRunner.py:211-237`):
- `torch.inference_mode()` 预测
- 如果 `test_all` 模式, 对已交互 item 赋 `-inf`

---

### 1.4 数据组织 (`helpers/SeqReader.py`, `helpers/BaseReader.py`)

**BaseReader** (`BaseReader.py`):
- 读取 `train.csv` / `dev.csv` / `test.csv`（TSV 格式）
- `n_users = max(user_id) + 1`, `n_items = max(item_id) + 1`
- 维护 `train_clicked_set` 和 `residual_clicked_set`（dev/test 的 item_id）

**SeqReader** (`SeqReader.py`):
- 合并所有 split → 按 time 排序 → 构建 `user_his[uid] = [(iid, t), ...]`
- 为每条交互分配 `position`（该 uid 历史中的位置索引）
- 通过 pd.merge 将 position 合并回 train/dev/test 的 data_df

**关键数据结构** (在 SeqReader 中):
```
corpus.user_his[uid] = [(iid1, t1), (iid2, t2), ...]   # 按时间排序的完整历史
corpus.data_df['train']['position']   # 每条 train 交互在 user_his 中的位置
```

---

## 二、现有模型逐个分析

### 2.1 PoMRec (`models/sequential/PoMRec.py`)

**类**: `PoMRec(SequentialModel)` (reader=SeqReader, runner=BaseRunner)

**真实功能**:
- MultiInterestExtractor: 双 prompt + self-attention → K 个兴趣向量 + 1 个分布向量
- LLM 分支（可选）: 加载 LLM 表 → adapter(GELU+LN) → 融合 `cf + gamma * llm` → InfoNCE 对齐损失
- 预测: `user_vector = sum(interest_k * softmax(proj(distri))_k)` → dot product with item_emb

**可复用**:
- `_load_llm_table_pkl` / `_load_srs_emb_pkl` (lines 22-66): 安全的 embedding loader，支持自动 padding row0
- `InfoNCEAlign` (lines 72-88): 对称 InfoNCE 对齐损失
- `load_model(strict=False)` (lines 411-431): 按 shape 匹配的部分权重加载
- `MultiInterestExtractor.__init__` 的 adapter 构造模式 (GELU+LN)
- `_alpha_t()` warmup (line 433-437)
- `get_item_emb` = `cf + gamma * llm` 的 residual 融合模式 (line 199-212)

**问题**:
- PoMRec 自身已有 LLM fusion，但不是 replace 模式
- gamma debug 打印 (lines 462-488) 会污染日志

---

### 2.2 PoMRecLLMEmb (`models/sequential/PoMRecLLMEmb.py`)

**类**: `PoMRecLLMEmb(PoMRec)` — 继承 PoMRec

**真实功能**:
- 三个融合模式: `none` (纯 CF), `replace` (只用 LLM), `residual` (cf + gamma * llm)
- 四种 adapter 架构: `ours` (GELU+LN), `llmemb` (纯 Linear), `noln` (GELU 无 LN), `linear` (单层 Linear)
- 三种注入范围: `both`, `history_only`, `candidate_only`
- TIC: target-interest consistency (cosine BPR)
- MVTC: CF-teacher → intent KL
- Candidate-aware scoring: candidate/residual/mix 三种模式

**关键实现**: line 200, **monkey patch**:
```python
self.interest_extractor.get_item_emb = self._fused_get_item_emb
```
- 这替换了 extractor 的 get_item_emb 方法，同时影响历史编码和候选打分
- 当 `llm_inject_scope == "history_only"` 时: forward 中候选显式用 `i_embeddings` (纯 CF)
- 当 `llm_inject_scope == "both"` 时: monkey patch 同时覆盖历史和候选

**replace 模式** (line 295-296): `return e_llm` — 完全丢弃 CF 嵌入。

**问题**:
- Monkey patch 依赖运行时替换，不透明
- llm_fuse_mode 的实际生效取决于 self._llm_inject_scope 的设置组合
- 继承 PoMRec 但通过 `args.use_llmemb=0` 禁用父类 LLM 逻辑

---

### 2.3 PoMRecLLMEmbLinear (`models/sequential/PoMRecLLMEmbLinear.py`)

**类**: `PoMRecLLMEmbLinear(PoMRec)` — 继承 PoMRec

**真实功能**:
- **LLM embedding + Linear adapter (无 GELU, 无 LN) + 直接替换物品嵌入**
- 这是整个仓库中最接近纯 "replace" 实验的文件
- Adapter: Linear(d_llm, d_llm//2) → Linear(d_llm//2, emb_size)，无激活，无 LayerNorm
- Line 132: `self.interest_extractor.get_item_emb = self._get_item_emb` — monkey patch
- `_get_item_emb` (line 142-144): `return self._get_adapted_llm_emb(item_ids)` — **纯 replace，无 CF**
- 自动发现 LLM 路径: `_find_llm_emb_path` (lines 51-64)
- 内置 NaN/Inf 诊断 (lines 149-230)

**回答关键问题**:
1. ✅ 这是实现了"LLM embedding + Linear(无激活) + 直接替换"的文件
2. ✅ 历史物品和候选物品都使用替换后的嵌入（因为 monkey patch 了 `get_item_emb`，同时被 extractor.forward 和候选打分使用）
3. ✅ 继承 PoMRec
4. ✅ 训练和推理路径一致（同一个 `_get_item_emb` 被所有路径使用）
5. ✅ 存在 monkey patch (line 132)

---

### 2.4 MyModel (`models/sequential/MyModel.py`)

**类**: `MyModel(SequentialModel)` — 直接继承 SequentialModel（不继承 PoMRec）

**真实功能** (论文版):
- 独立的 MultiInterestExtractor（拷贝自 PoMRec，有 LGD 支持）
- LLM fusion: residual 模式 `cf + gamma * llm`，GELU+LN adapter
- EMILE(IPD): 三次 cosine-distance BPR 约束
- LGD: 自条件双轮去噪（首轮无去噪提取 query → 次轮门控重加权）
- logic_aggr 参数读入但在 forward 中已废弃

**可复用**:
- LGD self-conditioned two-pass 架构 (lines 529-554)
- EMILE IPD 三约束 loss (lines 648-660)
- `_emile_w()`, `_logic_denoise_w()` 分段 warmup (lines 485-495)

**问题**:
- 与 PoMRec 各自维护独立的 `MultiInterestExtractor`，代码分支已分叉
- logic_aggr 参数仍保留但不生效（line 566 注释"已移除"）

---

### 2.5 SIERec (`models/sequential/SIERec.py`)

**类**: `SIERec(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- **Phase 1 clean skeleton**: PoMRec + LLM residual fusion + InfoNCE alignment
- **Phase 2 MVTC**: CF-only target interest distillation — 用正样本 CF 嵌入构建 teacher 分布，KL 蒸馏监督 intent 分布（line 449-503）
- 设计了 SPIP / semantic teacher / SIDE 预留位置

**可复用**:
- ✅ **已实现目标兴趣蒸馏** (MVTC): lines 479-502, `get_cf_emb(pos)` 作 teacher → softmax → KL(p_teacher || q_student)
- `_mvtc_w()` warmup (lines 406-410)
- Clean 代码架构，没有 monkey patch

**问题**: 无。这是新模型的最佳模板。

---

### 2.6 MyModelHMIF (`models/sequential/MyModelHMIF.py`)

**类**: `MyModelHMIF(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- HMIF: Hybrid Multi-Interest Fusion scoring
- 在 forward 中计算两种预测：aggregated（标准 PoMRec） 和 multi-interest matching（`einsum("bkd,bnd->bkn")`）
- 最终预测 = `(1-eta)*agg + eta*mi` (line 405)
- 支持 IPD + LLM fusion

**可复用**:
- Multi-interest matching scoring 模式 (lines 396-405)
- `logsumexp` 和 `max` 两种兴趣聚合

---

### 2.7 MyModelTIRL (`models/sequential/MyModelTIRL.py`)

**类**: `MyModelTIRL(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- TIRL: Target-routed Interest-level Ranking Learning
- **辅助损失**: 对 positive item 选择最相关兴趣 → 对该兴趣的 pos/neg 分数做 BPR（line 384-424）
- 不改变前向预测路径

**可复用**:
- ✅ **已实现目标路由的兴趣级排序损失** (lines 384-424)
- `compute_tirl_loss`: argmax 路由 + BPR
- `tirl_mode="selected"` 做 argmax 路由，`tirl_mode="max"` 做 max 路由

---

### 2.8 MyModelCTIRL (`models/sequential/MyModelCTIRL.py`)

**类**: `MyModelCTIRL(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- C-TIRL: Confidence-Gated TIRL
- Softmax 路由 + 置信度门控（只对高置信度样本施加损失）

**可复用**: 置信度门控机制 (lines 389-393)

---

### 2.9 MyModelITIRL (`models/sequential/MyModelITIRL.py`)

**类**: `MyModelITIRL(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- ITIRL: IPD-Guided TIRL — 复用 IPD 的距离信号作为路由依据
- 支持 `itirl_route_source="ipd"` / `"weight"`
- 置信度或 margin 门控

---

### 2.10 MyModelCIRF (`models/sequential/MyModelCIRF.py`)

**类**: `MyModelCIRF(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- CIRF: Candidate-aware Interest Residual Fusion
- Forward 中计算 base + gate * (route - base) (line 284)
- 不改变损失函数

---

### 2.11 MyModelSHNC (`models/sequential/MyModelSHNC.py`)

**类**: `MyModelSHNC(SequentialModel)` — 直接继承 SequentialModel

**真实功能**:
- SHNC: Semantic Hard Negative Calibration
- 加载离线预计算的语义最邻表 → 训练时从语义相似物品中采样 hard negatives
- 额外 BPR 损失

---

### 汇总表

| 文件 | 继承自 | Monkey Patch | LLM replace | 目标兴趣蒸馏 | 层次兴趣 | 原型路由 | 局部兴趣保持 | 最优传输 | 未来窗口 |
|------|-------|-------------|------------|------------|---------|---------|------------|--------|---------|
| PoMRec | SequentialModel | ❌ | ❌ (only residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PoMRecLLMEmb | PoMRec | ✅ L200 | ✅ replace/residual | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PoMRecLLMEmbLinear | PoMRec | ✅ L132 | ✅ replace only | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModel | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SIERec | SequentialModel | ❌ | ❌ (residual) | ✅ MVTC | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelHMIF | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelTIRL | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelCTIRL | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelITIRL | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelCIRF | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MyModelSHNC | SequentialModel | ❌ | ❌ (residual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**关于三个方向的已有实现**:
1. **SPCF (语义保持的子空间互补融合)**: ❌ 无现成实现。PoMRecLLMEmb 的 `replace` 是全部替换而非互补（消融中优于 residual +12.4%，但训练不稳定），`residual` 是简单加法而非子空间互补。InfoNCE 对齐损失可以复用为语义保持约束的一部分。SPCF 应保留 CF 嵌入作为稳定锚点，同时引入 LLM 语义作为互补信号——既避免纯 replace 的不稳定，又超越简单 residual 加法。
2. **CHIR (协同校准的层次语义兴趣路由)**: ⚠️ 部分组件存在。TIRL 有目标路由兴趣选择，SIERec 有 CF-teacher 蒸馏。但"层次语义兴趣"（显式多级抽象路由）和"协同校准"（双视角校准框架）均未实现。
3. **LFBT (局部因子保持与均衡目标学习)**: ❌ 无现成实现。没有局部兴趣保持、因子均衡或最优传输相关代码。

---

## 三、本地嵌入文件状态

### Beauty

```
路径: ./data/beauty/handled/llm_table_pca1536.pkl
shape: (12102, 1536)
dtype: float32
row0 norm: 0.000000     ← 已包含零填充 padding 行
other rows: min=6.433  mean=8.723  max=15.107  std=0.913
NaN: 0  Inf: 0
corpus.n_items: 12102  → rows 匹配 (arr.shape[0] == n_items) ✅
```

### ML-1M

```
路径: ./data/ml-1m/handled/llm_table_pca1536.pkl
shape: (3707, 1536)
dtype: float32
row0 norm: 0.000000     ← 已包含零填充 padding 行
other rows: min=0.000010  mean=9.706  max=14.966  std=2.134
⚠️ 注意: 有 norm≈0 的行 (min=0.000010)，可能是某些 item 的 LLM 嵌入为噪声
NaN: 0  Inf: 0
corpus.n_items: 3707  → rows 匹配 ✅
```

### Toys

```
路径: ./data/toys/handled/llm_table_pca1536.pkl
shape: (78772, 1536)
dtype: float32
row0 norm: 0.000000     ← 已包含零填充 padding 行
other rows: min=6.131  mean=9.395  max=15.566  std=0.995
NaN: 0  Inf: 0
corpus.n_items: 78772  → rows 匹配 ✅
```

### 结论

三个数据集都存在完整的 PCA 1536 维 LLM 嵌入文件，均已包含 padding row0，无 NaN/Inf，与 corpus.n_items 完全匹配。可以直接用于训练。

---

## 四、LLMemb 直接替换实验的真实结论

### 4.1 哪些文件实现了"LLM 直接替换"

全仓库搜索 `return e_llm` / `return.*llm_emb` / `replace` 模式，**只有两个文件**实现了 LLM embedding 完全替换 CF embedding：

| 文件 | 类 | 继承自 | 替换方式 |
|------|---|-------|---------|
| `models/sequential/PoMRecLLMEmb.py` | `PoMRecLLMEmb(PoMRec)` | PoMRec | `--llm_fuse_mode replace` 启用 |
| `models/sequential/PoMRecLLMEmbLinear.py` | `PoMRecLLMEmbLinear(PoMRec)` | PoMRec | 硬编码 replace |

其他所有文件（PoMRec、MyModel、SIERec、MyModelHMIF、MyModelTIRL 等）的 `get_item_emb` 实现全部是 `e_cf + gamma * e_llm`（residual 融合），**不存在 replace 模式**。

### 4.2 干净消融实验：replace vs residual vs none

**框架**: PoMRecLLMEmb（同一套代码，仅改变 `--llm_fuse_mode`）
**配置**: Beauty, seed=42, lr=0.002, K=4, prompt_num=3, lamb=4.0, adapter=ours(GELU+LN)

| llm_fuse_mode | 含义 | Dev NDCG@5 | Test NDCG@5 | 训练时间 |
|--------------|------|-----------|------------|---------|
| **replace** | 纯 LLM: `e_final = adapter(llm)` | **0.1344** | **0.1076** | 1105s |
| none | 纯 PoMRec CF: `e_final = e_cf` | 0.1286 | 0.0986 | 2679s |
| residual | 残差融合: `e_final = e_cf + gamma * adapter(llm)` | 0.1262 | 0.0957 | 1087s |

**结论: replace > none > residual**。replace 的 Test NDCG@5 (0.1076) 比 residual (0.0957) 高出 **+12.4%**。

> residual 实验使用 gamma_init=0.1, gamma_trainable=0（固定 gamma）。replace 模式不使用 gamma（直接返回 e_llm），因此 gamma 参数对 replace 无影响。三个实验的代码框架、adapter 架构、训练超参数完全一致，构成干净消融。

### 4.3 多 seed 验证

PoMRecLLMEmb replace + ours adapter, Beauty, lr=0.002:

| seed | Dev NDCG@5 | Test NDCG@5 |
|------|-----------|------------|
| 0 | 0.1319 | 0.1055 |
| 1 | 0.1366 | 0.1116 |
| 42 | 0.1344 | 0.1076 |
| **Mean** | **0.1343** | **0.1082** |

三个 seed 全部系统性高于 residual (Test=0.0957) 和 none (Test=0.0986)。

### 4.4 与用户残差方法（PoMRec + use_llmemb + llm_fuse=1 + alignment）对比

用户的残差连接方法在 PoMRec 框架中实现，除 residual fusion 外还包含 InfoNCE alignment loss。Beauty seed=42 最佳结果：

| 配置 | Dev NDCG@5 | Test NDCG@5 |
|------|-----------|------------|
| gamma=0.1 fixed, alpha=0.001 | 0.1339 | 0.1043 |
| gamma=0.01 trainable, alpha=0.0005 | 0.1337 | 0.1083 |

**PoMRecLLMEmb replace (test=0.1076) 对比 PoMRec residual+align 最佳 (test=0.1083): 两者基本持平。**
但 PoMRecLLMEmb replace **没有使用 alignment loss**，且训练时间更短（1105s vs 2087-2958s）。

### 4.5 其他数据集

**ML-1M** (seed=42, replace+ours):
- Test NDCG@5=0.2046
- PoMRec 标准基线: 0.2074, MyModelLLM residual: 0.2188
- replace 略低于 residual，但差距很小（-1.3% vs PoMRec, -6.5% vs MyModelLLM）

**Toys** (仅 PoMRecLLMEmbLinear, lr=0.0001):
- replace Test NDCG@5=0.146
- PoMRec 标准基线: 0.1134
- **replace 大幅领先 +28.7%**

### 4.6 不同 adapter 的影响

PoMRecLLMEmb, Beauty seed=42, replace 模式:

| Adapter | lr | Dev NDCG@5 | Test NDCG@5 | 稳定性 |
|---------|-----|-----------|------------|--------|
| ours (GELU+LN) | 0.002 | 0.1344 | 0.1076 | ✅ 稳定 |
| ours (GELU+LN) | 0.001 | 0.1284 | 0.1041 | ✅ 稳定 |
| noln (GELU, 无LN) | 0.001 | 0.1370 | **0.1121** | ✅ 稳定 |
| noln (GELU, 无LN) | 0.002 | 0.1205 | 0.1000 | ❌ NaN 崩溃 |
| llmemb (纯Linear) | 0.001 | 0.1054 | 0.0871 | ✅ 稳定 |
| llmemb (纯Linear) | 0.002 | 0.1022 | 0.0865 | ❌ NaN 崩溃 |

- **最佳 adapter**: noln (GELU, 无 LN) + lr=0.001 → Test NDCG@5=**0.1121**，进一步超过 residual
- LayerNorm 在 replace 模式下可能过度约束了 LLM embedding 的表达能力
- 纯 Linear adapter 无论 lr 高低都表现最差，说明非线性激活（GELU）对 replace 模式至关重要

### 4.7 PoMRecLLMEmbLinear 的结果

PoMRecLLMEmbLinear 使用纯 Linear adapter（无 GELU, 无 LN），硬编码 replace：

| 数据集 | 最佳 lr | Test NDCG@5 | 稳定性 |
|--------|--------|------------|--------|
| Beauty | 0.0005 | 0.085-0.088 | ⚠️ 部分 NaN |
| ML-1M | 0.0001 | 0.193 | ⚠️ 高 lr 全部 NaN |
| Toys | 0.0001 | 0.146 | ⚠️ 高 lr 全部 NaN |

PoMRecLLMEmbLinear 在极低 lr (0.0001) 下可以获得可接受的结果，但训练极慢（Toys 需要 7.5h），且稍高 lr 即 NaN 崩溃。

### 4.8 总结

1. **只有两个文件实现了 replace**: PoMRecLLMEmb 和 PoMRecLLMEmbLinear
2. **replace 在干净消融中优于 residual**：Test NDCG@5 0.1076 vs 0.0957 (+12.4%)
3. **replace 与用户的 PoMRec residual+align 方法持平**（0.1076 vs 0.1083），但不需要 alignment loss
4. **replace 的最大问题是训练稳定性**：仅 GELU-based adapter 能收敛，纯 Linear adapter 普遍 NaN。最佳 performer（noln, lr=0.001）在 lr=0.002 时也崩溃
5. **仓库中不存在 "ReLU + 直接替换" 的实验**。所有 replace 实验使用 GELU 或纯 Linear。ReLU 替代 GELU 的变体未被探索
6. **对后续 SPCF 的启示**：完全丢弃 CF 嵌入虽然在消融中优于 residual，但训练不稳定。SPCF 应采用"子空间互补融合"而非简单替换，保留 CF 嵌入作为稳定锚点，同时引入 LLM 语义作为互补信号

---

## 五、未来窗口可行性

### 5.1 各数据集可构造 future 窗口的样本比例

基于 `train.csv` 和 position 字段的真实统计:

| 数据集 | 训练样本 | 平均历史长度 | ≥2 future | ≥3 future | ≥4 future |
|--------|---------|------------|-----------|-----------|-----------|
| Beauty | 153,776 | 7.8 | 100% | 85.5% | 70.9% |
| ML-1M | 988,129 | 194.8 | 100% | 99.4% | 98.8% |
| Toys | 1,343,463 | 7.8 | 100% | 84.5% | 69.2% |

### 5.2 如何只用 train 内后续交互

在 `SequentialModel.Dataset._get_feed_dict` (BaseModel.py:238) 中，`self.corpus.user_his[uid][:pos]` 是按时间排序的完整历史。因为 SeqReader 已将 train/dev/test 合并后排序，`user_his[uid]` 包含所有交互（不分 split）。要在训练中只使用 train 内交互，有两种策略:

1. **保守策略**: 只取 `position+1` 到 `position+H` 的 future items，这必然来自后续时间（因为 user_his 按时间排序）。这些后续 items 可能是 train、dev 或 test。需要额外过滤只保留 train 中的 items。过滤方法：检查 `item_id in train_clicked_set[uid]`。

2. **纯 train 策略**: 对每个 uid，只取其 train 内部的时间连续子序列。这需要重建 train-only 的时间序列，但不需要修改 Reader——可以在 Dataset 内 `prepare()` 时构建 `train_only_his[uid]`。

### 5.3 建议默认值

- **future_window=3**: ML-1M 有 99.4% 覆盖，Beauty/Toys 约 85% 覆盖
- **future_window=2**: 三个数据集均 100% 覆盖，但太短
- **推荐 future_window=3**，对于 Beauty/Toys 中 H<3 的样本可以用 mask 跳过

---

## 六、最终输出

### 6.1 当前仓库真实结构

```
训练框架: main.py → eval(model_name) → model(reader, runner)
继承链: BaseModel → GeneralModel → SequentialModel → 具体模型
数据流: CSV → BaseReader → SeqReader → corpus.pkl → Dataset._get_feed_dict → DataLoader
模型文件: 17 个 import + 1 个未 import（论文版备份）
日志: ./log/<ModelName>/<encoded_args>.txt
模型: ./model/<ModelName>/<encoded_args>.pt
```

### 6.2 可以直接复用的代码

| 来源 | 组件 | 用途 |
|------|-----|------|
| `PoMRec.py:22-66` | `_load_llm_table_pkl`, `_load_srs_emb_pkl` | 安全加载 LLM/SRS 嵌入表 |
| `PoMRec.py:72-88` | `InfoNCEAlign` | 语义对齐损失 |
| `PoMRec.py:411-431` | `load_model(strict=False)` | 按 shape 的部分权重加载 |
| `PoMRec.py:199-212` | `get_item_emb` residual 融合 | `cf + gamma * llm` 模式 |
| `PoMRec.py:433-437` | `_alpha_t()` warmup | 分段线性 warmup |
| `SIERec.py:98-249` | MultiInterestExtractor (clean) | 最干净的 extractor 骨架 |
| `SIERec.py:449-503` | MVTC CF-only distillation | 可改编为语义 teacher |
| `MyModelTIRL.py:384-424` | `compute_tirl_loss` | 目标路由兴趣级排序 |
| `MyModel.py:529-554` | LGD self-conditioned two-pass | 可改编为多层次提取 |
| `MyModel.py:648-660` | IPD 三约束 BPR | target-interest consistency |
| `MyModelCTIRL.py:361-409` | 置信度门控 | 过滤低置信度路由 |

### 6.3 必须废弃或隔离的旧逻辑

| 来源 | 问题 | 建议 |
|------|-----|------|
| PoMRecLLMEmb.py:200 | `get_item_emb` monkey patch | 新模型不使用 monkey patch，改为显式参数控制 |
| PoMRecLLMEmbLinear.py:132 | 同上 | 同上 |
| PoMRec.py (整个类) | PoMRec 自己维护了一个独立的 MultiInterestExtractor 副本 | 新模型直接继承 SequentialModel，自定义 extractor |
| MyModel.py | MultiInterestExtractor 与 PoMRec 分叉 | 不再继承此版本，从 SIERec 骨架开始 |
| MyModelV2/V4/V5 | 未审计但大概率是废弃版本 | 忽略 |
| MyModelLLM, MyModelLLMIPD, MyModelSCIL | 同上 | 忽略 |

### 6.4 LLMemb 直接替换实验的真实结论（已更正）

**只有两个文件实现了 LLM 直接替换**: PoMRecLLMEmb.py (`--llm_fuse_mode replace`) 和 PoMRecLLMEmbLinear.py（硬编码）。

**干净消融（PoMRecLLMEmb, Beauty seed=42, lr=0.002, K=4, prompt=3, lamb=4.0, adapter=GELU+LN）**:

| llm_fuse_mode | Test NDCG@5 |
|--------------|------------|
| **replace** | **0.1076** |
| none (纯CF) | 0.0986 |
| residual (cf+γ×llm) | 0.0957 |

**replace > none > residual，replace 比 residual 高 +12.4%。**

多 seed 均值 (0/1/42): replace Test NDCG@5 = 0.1082，系统性地高于 residual。

**与用户 PoMRec residual+alignment 方法对比**: replace (0.1076) ≈ PoMRec residual+align 最佳 (0.1083)，但 replace 不需要 alignment loss，训练更快。

**稳定性问题**: 纯 Linear adapter (PoMRecLLMEmbLinear) 频繁 NaN，仅 GELU-based adapter 在低 lr 下稳定。

**关键教训**: 完全替换在消融中优于残差，但需要 GELU 非线性 + 低 lr。SPCF 应采用子空间互补融合（保留 CF 稳定锚点 + LLM 互补信号），而非完全丢弃 CF。

### 6.5 三个数据集嵌入文件状态

全部就绪：Beauty (12102, 1536), ML-1M (3707, 1536), Toys (78772, 1536)。均已 padding row0，无 NaN/Inf，与 corpus 匹配。

### 6.6 未来窗口监督的可行性

✅ **完全可行。** 三个数据集均有充足的 train 内后续交互可用于未来窗口监督：
- H=3 推荐默认值（ML-1M 99.4%, Beauty 85.5%, Toys 84.5%）
- 不需要修改 Reader/Runner
- 在模型 Dataset 内覆盖 `_get_feed_dict` 和 `collate_batch` 即可

### 6.7 建议新增和修改的文件列表

**新增**:
```
models/sequential/MyModel2.py          # 新模型主干，继承 SequentialModel
models/sequential/layers_spcf.py       # SPCF: 子空间互补融合模块
models/sequential/layers_chir.py       # CHIR: 层次语义兴趣路由模块
models/sequential/layers_lfbt.py       # LFBT: 局部因子保持与均衡目标
models/sequential/utils_emb.py         # 统一的 embedding loader 和 adapter 工具
```

**修改**:
```
main.py                                # 添加 import models.sequential.MyModel2
```

**不需要修改**:
```
helpers/BaseReader.py, helpers/SeqReader.py, helpers/BaseRunner.py
models/BaseModel.py
```

### 6.8 在正式编码前仍缺少的信息

1. **SRS 嵌入文件** (`srs_emb_path`): 需要确认以下文件是否存在以及 shape:
   - `./data/beauty/handled/itm_emb_pomrec.pkl`
   - `./data/ml-1m/handled/` 下对应文件
   - `./data/toys/handled/` 下对应文件
   这些在 CLAUDE.md 中提及但未在此审计中验证

2. **PoMRec 预训练 checkpoint**: 需要确认可用的 `--init_ckpt` 路径。CLAUDE.md 中提到需要 `<PoMRec_checkpoint.pt>` 但没有给出具体路径

3. **SPCF 的设计细节**: 需要明确子空间互补融合的具体数学形式（正交投影？CCA？互信息最大化？），当前仓库的 InfoNCE 对齐只能作为语义保持约束的基础组件

4. **CHIR 的设计细节**: 需要明确"层次"是指多层抽象还是多粒度路由。当前仓库的 TIRL 和 MVTC 可以分别提供路由和蒸馏的基础组件，但缺乏层次化架构

5. **LFBT 的设计细节**: 仓库中完全没有相关实现。需要明确是使用 Optimal Transport (Sinkhorn) 还是其他均衡约束方法

6. **LLM 嵌入的 PCA 降维**: 当前 1536 维 PCA 嵌入的来源和降维方法需要确认（是否还有更高维的原始嵌入可用）。`tools/convert_llmemb_jsonl_to_pkl_table.py` 可能包含答案

7. **实验基线**: 需要明确新模型的对标基线（PoMRec baseline? PoMRec+LLM residual? MyModel 论文版?），当前仓库中 PoMRec __none mode 的测试结果是 NDCG@5=0.0986 (Beauty)

# CHAPTER4_PHASE3_CONTEXT.md

> Chapter 4 Phase 3 — History Contextualization Axis Test
> 日期：2026-10-07
> 性质：**结构性 control experiment，不是 Chapter 4 的最终方法**
> 实现：`models/sequential/LLMMIRecContextControl.py`
> 测试：`tools/test_llmmirec_context_control.py`（22/22 通过）
> 脚本：`new_bash/run_llmmirec_context_phase3_beauty.sh`
> 诊断：`tools/diagnose_context_block.py`

---

## 0. 本轮回答的唯一问题

> **在进入 multi-interest extractor 之前，让 history 行为彼此 contextualize，
> 是否比当前"逐 item 独立表示"更有效？**

由 `CHAPTER4_DIAGNOSIS.md` §9 的事实 **F7** 导出：
当前每个 history position 进入 extractor 前，
其表示**完全没有被其他 history position 修改过**。

**这不是最终方法。** 它是一次 axis test，用来判断
"history item contextualization" 这条轴是否值得继续投入。

---

## 1. F7：结构性前提

```
history_emb_raw  = item_encoder(history)     ← 逐 item 独立
                 + position_emb(position)    ← 逐位置独立
                 -> dropout
                 -> QueryMultiInterestExtractor
                      Q = Wq(learned K queries)   ← 与 history 无关
                      K = Wk(history_position)    ← 逐位置独立
                      V = Wv(history_position)    ← 逐位置独立
```

唯一的跨位置操作是 extractor 内部"**K 个固定可学习 query 读取 L 个位置**"，
**不是 history 位置之间的相互 contextualize**。
`ItemEncoder` 亦为纯逐 item 模块（padding 项恒输出零向量）。

⇒ **F7 成立**（详见 `CHAPTER4_DIAGNOSIS.md` §9.1）。

---

## 2. 唯一插入点

只改**喂给 extractor 的那个张量**：

```
history_emb_pos -> [contextualizer] -> QueryMultiInterestExtractor
```

下游**逐字不变**：

| 环节 | 是否改动 |
|---|---|
| `ItemEncoder` | ❌ 未改（导入原模块） |
| `position_emb` / dropout | ❌ 未改 |
| `QueryMultiInterestExtractor` | ❌ 未改（导入原模块） |
| `InterestAggregator(history_emb_raw, lengths)` | ❌ 未改，**仍读 pre-context 的 `history_emb_raw`** |
| `user_vector = Σ_k w_k V_k` | ❌ 未改 |
| `prediction = <user_vector, e_candidate>` | ❌ 未改 |
| relation loss | ❌ 未改 |
| **LOSS** | **BPR + 0.01 × Chapter3 relation loss，本轮不新增任何 Chapter 4 loss** |

> 注：`InterestAggregator` 读的是 **pre-context** 的 `history_emb_raw`，
> 这是有意为之——保持聚合器的输入分布在 control 与 baseline 之间一致。

---

## 3. 三种 `context_mode`

| mode | 结构 | 有跨位置交互？ |
|---|---|---|
| `baseline` | **不构造任何模块** | — （严格复现 ASPCF） |
| `ffn_control` | `H + dropout(MLP(LN(H)))`，`Linear(64,256)-GELU-Linear(256,64)` | ❌ **无**（逐位置独立） |
| `self_attn` | 1 层 Pre-LN self-attention（§4） | ✅ 有 |

### 3.1 self-attn block（严格按指定公式）

```
X   = H
QKV = LN1(X)
A   = MHSA(QKV, QKV, QKV, key_padding_mask)
Y   = X + dropout(A)
H'  = Y + dropout(FFN(LN2(Y)))
```

参数：`num_heads=4`，`D=64`（每头 16 维），`FFN hidden=128`，
`context dropout=0.1`，**层数=1**。**不堆多层，不扫 heads/layers/hidden。**

### 3.2 padding 策略（明确）

1. **key 必须 mask**：`key_padding_mask` 把 pad 位置的 key 置为 `-inf`。
2. **block 输出后 padded positions 显式清零**，避免无效 query 位置
   产生垃圾表示进入 extractor。

> 为什么必须清零：`item_encoder(0) = 0`（padding 项恒为零向量），
> 但 `position_emb(0)` 非零，故 padded 位置的 `history_emb_pos` **不是零**。

### 3.3 为什么**不用** causal mask（明确说明）

当前任务是用**完整历史**预测下一个 item。
所有 history 位置都**严格发生在 target 之前**，
因此**任何位置都不可能泄漏 target 信息**。
故第一版**不使用下三角 causal mask，只用 padding mask**。

（若将来序列中出现"未来 item"，才需要因果掩码。）

**不使用 candidate / target 的任何信息** ⇒ **不存在 leakage**。

---

## 4. 等参数量 FFN control

| 模块 | 参数量 | 其中 context block |
|---|---|---|
| ASPCF / `baseline` | **943,174** | 0 |
| `ffn_control` | **976,390** | **+33,216** |
| `self_attn` | **976,646** | **+33,472** |
| **两者 context block 差距** | **256** | **占 block 的 0.77%** |

`self_attn` 的 block = LN1(128) + QKV(3×4160) + out_proj(4160) + LN2(128)
+ FFN(8320+8256) = 33,472。
`ffn_control` 用 `hidden=256` 恰好给出 33,216，差距仅 256（0.77%）。

**目的**：让 `ffn_control` 拥有几乎相同的**新增非线性容量**，
但**没有任何跨位置交互**。于是
`self_attn − ffn_control` 才是"history interaction"的独立效应。

---

## 5. 单元测试（22/22 通过）

`python tools/test_llmmirec_context_control.py`

| # | 检查 | 结果 |
|---|---|---|
| 1 | baseline 加载真实 ASPCF checkpoint，prediction **逐位一致** | ✅ `max|diff| = 0.000e+00` |
| 1b | baseline 参数量 == ASPCF 参数量 | ✅ 943,174 == 943,174 |
| 2 | shape `[B,L,64] -> [B,L,64]` | ✅ |
| 3 | padded 位置输出**严格为 0** | ✅ `max = 0.000e+00` |
| 4a | attention 落在 padded **key** 上的质量 == 0 | ✅ `max = 0.000e+00` |
| 4b | 有效 query 行 attention 和 == 1 | ✅ `1.2e-07` |
| 4c | 全 mask 的 query 行恰好为 0（非 NaN，length=0 边界） | ✅ |
| 5 | **ffn_control**：扰动位置 5，**只有位置 5 变化** | ✅ `changed=[5]` |
| 6 | **self_attn**：扰动位置 5，**其他有效位置也变化** | ✅ 全部 20 个位置变化 |
| 7 | 三种 mode 均无 NaN/Inf | ✅ |
| 8 | backward：context block / ItemEncoder / extractor **均有梯度** | ✅ 三者 > 0 |
| 9 | `length=1` 与 `length=20` 边界 | ✅ |

**测试 4 的语义澄清**（写下来避免误读）：
规范只要求 mask **key**。一个 **padded query 行**仍然会合法地
attend 到有效 key（和=1），其输出随后被**显式清零**（测试 3）。
所以正确的断言是"**没有 attention 质量落在 padded key 上**"，
而不是"padded query 行的和为零"。首轮测试写错了这一点，已修正。

---

## 6. 训练配置与运行时 diff

**配置完全继承 frozen Beauty ASPCF**：`K=4`、`lr=0.004`、`batch_size=1024`、
`eval_batch_size=256`、`history_max=20`、`dropout=0.1`、`lambda_relation=0.01`、
`num_neg=1`、`epoch=200`、`early_stop=10`、`topk=5,10,20,50`。

**因为新增层而修改 lr？——没有。**

### 6.1 启动前静态 diff

逐 flag 比对两个 bash 脚本：除 5 个新增 `context_*` flag 外，
`${EPOCH}` / `${EARLY_STOP}` / `${NUM_WORKERS}` / `${LLM_PATH}` 均为 shell 变量，
展开值相同（200 / 10 / 5 / 同一路径）。

### 6.2 启动后真实 Arguments 表 diff

从运行日志解析 Arguments 表（ASPCF 41 行；两个 control 各 46 行）：

| 配置 | 与 frozen ASPCF 不同的键 |
|---|---|
| `ffn_control` | **5 个**：`context_mode`, `context_heads`, `context_ffn_hidden`, `context_ffn_control_hidden`, `context_dropout` |
| `self_attn` | 同上 5 个 + `gpu`（0→1，两卡分跑，预期内） |

**其余 41 项逐项一致。** 启动日志确认：
`First-batch NaN/Inf check passed`，无 NaN/Inf，epoch 正常推进。

---

## 7. 诊断脚本（`tools/diagnose_context_block.py`）

`return_intermediate=True` 额外返回：

| 键 | 形状 | 含义 |
|---|---|---|
| `history_pre_context` | `[B,L,D]` | 进入 context block 前 |
| `history_post_context` | `[B,L,D]` | context block 输出（已清零 padding） |
| `context_attention_maps` | `[B,h,L,L]` | self-attn 权重（`ffn_control` 为 `None`） |

可计算：

| 量 | 定义 | 目的 |
|---|---|---|
| **A. contextual shift** | `‖H_ctx − H‖ / ‖H‖`（仅有效位置） | 表示被改动了多少 |
| **B. attention entropy** | `H(A)` per (query, head)，仅在 ≥2 个有效 key 的行上 | 是否接近均匀 |
| **C. diagonal vs off-diagonal mass** | query i 对 key i 的质量 vs 对其他有效 key | **是否退化为 self-copy** |
| **D. head-to-head similarity** | 各头 attention 分布两两余弦 | 4 个头是否冗余 |
| **E. shift by history length** | 按 `lengths` 分组 | 长历史是否被改动更多 |

**核心目的：确认 self-attention 不是只学到 identity / self-copy。**
若 C 的 diagonal share ≈ 1，则"contextualization"是假的。

---

## 8. 判定标准（预先声明，先于结果）

重点看 **SelfAttn vs FFNControl vs ASPCF** 三者。

| 情形 | 条件 | 判定 |
|---|---|---|
| **A** | SelfAttn > ASPCF **且** SelfAttn > FFNControl | **history-history interaction 得到强支持** |
| **B** | SelfAttn 与 ASPCF 基本持平（NDCG@5 下降 ≤ ~0.5%），**且明显优于 FFNControl**，**且诊断证明存在真实 off-diagonal interaction** | 仍视为 **promising**，值得下一步跑 ML-1M |
| **C** | SelfAttn 与 FFNControl **同方向同幅度**变化 | 更像**参数量/非线性容量效应**，history interaction 无独立证据 |
| **D** | SelfAttn NDCG@5 相对 ASPCF 下降 **> ~1~1.5%**，且无其他稳定收益 | **停止该轴，不跑 ML-1M** |

**本轮不机械要求 Beauty 显著提升**——该方向主要可能帮助
**collaborative-rich 的 ML-1M**。

**即便 promising 也绝不自动跑 ML-1M**，只报告，由你决定。

---

## 9. 结果

### 9.1 训练概况

| config | params | best epoch | train time | best dev |
|---|---|---|---|---|
| ASPCF (frozen) | 943,174 | 61 | 869.4 s | 0.1891 / 0.1300 |
| `ffn_control` | 976,390 | 43 | 752.1 s | 0.1901 / 0.1300 |
| `self_attn` | 976,646 | 63 | 1036.8 s | 0.1856 / 0.1281 |

三者均正常收敛，无 NaN/Inf。

### 9.2 Ranking（Beauty test）

| config | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| **ASPCF (frozen)** | **0.1592** | 0.2292 | 0.3171 | **0.1088** | 0.1313 | 0.1535 |
| `ffn_control` | **0.1601** | **0.2341** | **0.3240** | 0.1078 | **0.1316** | **0.1543** |
| `self_attn` | 0.1539 | 0.2232 | 0.3138 | 0.1035 | 0.1258 | 0.1487 |

**Δ vs ASPCF（%）**

| config | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| `ffn_control` | **+0.57** | +2.14 | +2.18 | **−0.92** | +0.23 | +0.52 |
| `self_attn` | **−3.33** | −2.62 | −1.04 | **−4.87** | −4.19 | −3.13 |

**`self_attn` 相对 `ffn_control`**：HR@5 **−3.87%**，NDCG@5 **−3.99%**。

`ffn_control` 的 +0.57% HR@5 折合 **≈0.8σ**（ASPCF 5-seed std = 0.0011），
**在种子噪声内**，不能算增益；其 NDCG@5 反而 −0.92%。
⇒ **`ffn_control` ≈ ASPCF（无实质变化）。**

### 9.3 上下文诊断（test）

| 量 | `self_attn` | `ffn_control` |
|---|---|---|
| **A. contextual shift** `‖H_ctx−H‖/‖H‖` | mean **0.7757**（p05 0.37 / p95 1.67） | mean **1.0603**（p05 0.26 / p95 2.60） |
| **B. attention entropy** | mean **1.4178**（p50 1.35 / p95 2.44；上限 ln20=2.9957） | — （无注意力） |
| **C. diagonal / off-diagonal mass** | diag **0.0416** / off-diag **0.9584**，**diag share = 4.16%** | — |
| **D. head-to-head similarity** | mean **0.7041**（p05 0.33 / p95 0.97） | — |
| **E. shift 随 history 长度** | len 4 → 0.679，len 9 → 0.862，len 17 → 0.885，len 20 → 0.746 | len 4 → 0.890，len 15 → 1.253，len 20 → 1.218 |

**核心读数（最重要的一条）**：

> **self-attention 并没有退化成 identity / self-copy。**
> - 对角质量只有 **4.16%**，**95.8% 的注意力质量落在其他位置上**；
> - 注意力熵 1.42，远低于均匀上限 2.996；
> - contextual shift 0.78，表示确实被大幅改写。

**⇒ 跨位置交互是真实发生的，而且很强。**

但是——`ffn_control` 的 shift（1.0603）**比 `self_attn`（0.7757）还大**，
说明"表示被改写多少"本身不解释性能差异：
`ffn_control` 改写更多但 ≈ 不变，`self_attn` 改写更少但 −4.87%。

⇒ **造成损失的不是"改写的幅度"，而是"跨位置混合"这一操作本身。**

---

## 10. 判定

| 情形 | 条件 | 是否成立 |
|---|---|---|
| **A** | SelfAttn > ASPCF **且** > FFNControl | ❌ 两者都不成立（SelfAttn 全面最差） |
| **B** | SelfAttn ≈ ASPCF（≤0.5%）**且** > FFNControl **且** 有真实 off-diagonal 交互 | ❌ 前两条不成立（NDCG@5 −4.87%，且**劣于** FFNControl）。**第三条成立**——交互确实真实 |
| **C** | SelfAttn 与 FFNControl 同方向同幅度 ⇒ 容量效应 | ❌ 方向同为负，但**幅度差 5 倍**（−0.92% vs −4.87%），不构成容量效应 |
| **D** | SelfAttn NDCG@5 相对 ASPCF 下降 **> ~1~1.5%**，无其他稳定收益 | ✅ **−4.87%，远超阈值** |

> ### 判定：**情形 D 成立 —— 停止该轴，不跑 ML-1M。**

### 10.1 这个否定结果的性质

与 PPCIM Round 1 不同，**这次不能归因于"机制未激活"**：

| | PPCIM Round 1 | Phase 3 self-attn |
|---|---|---|
| 机制是否激活 | ❌ 否（cos 0.99，熵降 0.5%） | ✅ **是**（off-diag 95.8%，熵 1.42 ≪ 3.00） |
| 排序 | 未改善 | **−4.87% NDCG@5** |
| 结论强度 | 待定，需补探针 | **机制真实发生且真实有害** |

**这是一个干净的负面结果**：`ffn_control` 用几乎相同的参数量、
**更大的表示改写幅度**，却保持中性；唯一差别是**有无跨位置混合**。
因此"cross-position mixing 本身在 Beauty 上是负贡献"这一结论，
**被等参数量 control 隔离出来了**。

### 10.2 可能的解释（**假设，本轮未验证**）

1. **ASPCF 的 value space 本就是"逐 item 语义"空间**（`Wv` 作用在
   ASPCF 融合后的 item 表示上），位置间混合会把**已经足够好的 item 语义**
   平均掉。`ffn_control` 不混合，所以不受伤。
2. **Beauty 的历史很短**（7162/22363 用户长度 <5，10959 长度 5~10），
   长距离上下文信息量有限。
3. 与 §9.2 呼应：ASPCF 在 Beauty 的收益集中在**长尾 item**（F5），
   而注意力混合倾向于把表示拉向**高频共现**的方向。

**这些都只是假设，未经任何实验支持。本轮不做归因。**

### 10.3 与 `CHAPTER4_DIAGNOSIS.md` F6 的关系

F6 说"K 个 interest 在打分前被压成单一向量"。
Phase 3 表明：**把 upstream 表示做得更"上下文相关"也无效。**
结合 PPCIM（改打分路径无效）与 Dispersion（改统计量无效），
当前已有**四条独立负面证据**：

> 瓶颈既不在 interest 的组织方式、不在打分的 candidate 条件化、
> 不在兴趣的统计阶数、也不在 **history 进入 extractor 前的上下文补全**。

**Beauty 上的问题更可能在 item 表示与 candidate 打分之间的几何**（F4/F5）。

### 10.4 处置

- **不跑 ML-1M。**（判定 D 明确要求；且用户 §10 明确禁止自动进入）
- 保留 `LLMMIRecContextControl.py` 与两个 checkpoint 作为 **negative evidence**，
  不再修改、不再训练。
- `LLMMIRecASPCF.py` 全程未被修改。

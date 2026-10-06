# CHAPTER4_DESIGN.md

> Chapter 4 方向设计（**只给方案，不写代码、不训练**）
> 日期：2026-10-06
> 前置约束：Chapter 3 = frozen ASPCF（`CHAPTER3_FINAL.md`），ASPCF `ItemEncoder` 固定不动。
>
> ## ⚠️ 状态说明（2026-10-06 更新）
>
> - **§3 的 SCIR（Semantic-Competitive Interest Routing）标记为 `DRAFT`**，
>   **不作为最终方案实现**，保留为备选。
> - **§8 的 RCIR（Residual Competitive Interest Routing）为当前主候选**。
> - 在 `CHAPTER4_PHASE0_DIAGNOSIS` 的结论出来之前，**两者都不实现**。
> - 诊断决定分支：**A** interest routing 是瓶颈 → 实现 RCIR；
>   **B** aggregation/scoring 是瓶颈 → 优先转 candidate-aware scoring；
>   **C** 两者都有问题 → 再规划两阶段结构。

---

## 1. 失败复盘：Chapter 4 已有探索为何全部没有稳定收益

### 1.1 已有方法与其真实结果

| 方法 | 核心机制 | 作用位置 | Beauty NDCG@5 | ML-1M NDCG@5 | seeds |
|---|---|---|---|---|---|
| **ASPCF**（Ch3 冻结基线） | — | — | **0.1075 ± 0.0014** | **0.2140 ± 0.0023** | 5 |
| HSDIR（best: hier+agreement） | 行为共属图蒸馏 | 训练期 loss | 0.1098 † | 0.2156 † | 1 |
| HSDIR（同配置 λ=0 对照） | — | — | 0.1053 † | 0.2144 † | 1 |
| CHIR（dual_routing） | prototype query + 双视角路由 | 前向（query 来源） | 0.1069 | — | 1 |
| CASIR（complement_coherence） | 语义残差注入打分路径 | 前向（残差加性） | 0.1017 | 0.2141 | 1 |
| CAISD（5-seed, responsibility） | 兴趣语义 profile 蒸馏 | 训练期 loss | **0.10738** | **0.21042** | 5 |
| TASID（llm_only / asymmetric） | target 条件蒸馏 | 训练期 loss | 0.1082 | 0.2135 | 1 |

† HSDIR 在其**自己的**超参配置（Beauty lr=0.008/bs=2048；ML-1M lr=0.002/bs=2048）下运行，
与冻结 ASPCF（lr=0.004 或 0.001 / bs=1024）**不同配置，不可直接比较**。

**唯一有 5-seed 公平对比的是 CAISD：Beauty 打平、ML-1M 落后 ASPCF 1.68%。**
其余全部是单 seed，且没有一条稳定超过冻结 ASPCF。

### 1.2 HSDIR 是最好的诊断样本

HSDIR 是唯一在**结构指标**上有大幅、方向一致的改变、却仍未能稳定转化为收益的方法：

| 指标 | Beauty baseline → HSDIR | ML-1M baseline → HSDIR |
|---|---|---|
| 兴趣间余弦 | 0.9457 → **0.8691** | 0.7620 → **0.7377** |
| 有效兴趣数 | 1.714 → **2.280** | 2.762 → **3.028** |
| route membership entropy | 0.913 → **0.543** | — |
| effective active K | 3.833 → **2.632** | — |
| route 与语义结构相关性 | ~0.03 → **~0.36 / 0.38** | — |

### 1.3 共同的结构性原因

上表所有方法，无论机制如何，都落入以下三类之一：

| 类型 | 方法 | 问题 |
|---|---|---|
| **A. 训练期监督** | HSDIR、CAISD、TASID | 前向路由算子完全不变。teacher 只改变了参数的落点，没有给模型任何**新能力**。测试期模型做的事与 baseline 逐字相同 |
| **B. 只改 query 来源** | CHIR | 路由算子仍是 `softmax over L`，只是 query 换成 prototype 派生。表达能力未变，只换了输入 |
| **C. 在打分路径上加残差** | CASIR | 改的是 `V_k` 的数值而不是 `history → V_k` 的**映射**，且与 ASPCF 的 item 表示耦合，容易互相干扰 |

**结论：Chapter 4 要有效，必须改变 `history → K interests` 的算子本身（类型 D），
而不是在它外面套监督或残差。**

---

## 2. 设计原则（从失败中推导）

| # | 原则 | 来自 |
|---|---|---|
| **P1** | 必须改变**前向算子**。测试期 `V_k` 的计算方式要与 baseline 不同，而不只是参数不同 | HSDIR 的核心教训 |
| **P2** | 核心创新不能是"再加一个 auxiliary loss"。辅助 loss 只能作为可选增益项，默认关闭 | 用户约束 + HSDIR 教训 |
| **P3** | 语义信息必须进入**前向路由计算**，而不是只出现在训练期 teacher 里 | 用户约束 |
| **P4** | K 个 interest 之间要有**显式竞争** | 用户约束 |
| **P5** | 不许碰 ASPCF `ItemEncoder` | 用户约束（Ch3 已冻结） |
| **P6** | 必须在 Beauty **和** ML-1M 同时验证，且先看无辅助 loss 的版本 | `CHAPTER3_FINAL.md` §6.3 的结构性风险 |

---

## 3. 方案：Semantic-Competitive Interest Routing (SCIR)

### 3.1 当前算子（ASPCF backbone）

```
Q = Wq(query_k)                    # [K, d]  K 个可学习 query
Km = Wk(h)                         # [B, L, d]
scores = Q · Kmᵀ / √d              # [B, K, L]
attn   = softmax_L(scores)         ← ★ softmax 在 HISTORY 上
V_k    = Σ_l attn[k,l] · Wv(h_l)   # [B, K, D]
```

**关键**：softmax 在 `L` 维上。每个 interest **独立地**在历史上分配质量，
K 个 interest 之间**没有任何竞争**——同一个 position 可以同时以 1.0 的权重进入所有 K 个 interest。

### 3.2 新算子

```
# ── 竞争性分配：softmax 在 K 上，不在 L 上 ──
S[b,l,k] = collab(b,l,k) + β · sem(b,l,k)
R        = softmax_K( S / τ_r )          # ★ [B, L, K]  每个 item 在 K 个 interest 间分配归属

# ── item 级质量（防止丢失"重要性"信息）──
m[b,l]   = σ( w_m · h_{b,l} + b_m )      # [B, L]

# ── interest = 其归属 item 的加权平均 ──
V_k = Σ_l m[b,l] · R[b,l,k] · Wv(h_l)
      ──────────────────────────────────      # [B, K, D]
        Σ_l m[b,l] · R[b,l,k] + ε
```

其中两个打分通道：

| 通道 | 定义 | 来源 |
|---|---|---|
| `collab(b,l,k)` | `Wk(h_{b,l}) · Wq(q_k) / √d` | **完全复用现有 `QueryMultiInterestExtractor` 的投影** |
| `sem(b,l,k)` | `cos( s_{b,l} , A_k ) / τ_sem` | `s_{b,l}` = ASPCF item 表示的**前 32 维**；`A_k ∈ R³²` 为 K 个**可学习语义锚点** |

> **`s_{b,l}` 的取法（关键复用点）**：ASPCF 的 item 表示为
> `e = concat[√α_s·s , √α_c·c]`，因此 `e[:, :32] ∝ s`。
> **余弦相似度对正标量不敏感**，所以 `cos(e[:, :32], A_k) = cos(s, A_k)`。
> → **`ItemEncoder` 一行都不用改**，直接从已有表示切片即可。

### 3.3 为什么这个改动对应 P1–P4

| 原则 | 如何满足 |
|---|---|
| **P1 改前向算子** | `softmax_L` → `softmax_K` + 归一化加权平均。即便 `β=0`（不加语义），测试期 `V_k` 的**计算方式**也已不同于 baseline。这是新的表达能力，不是新的参数落点 |
| **P2 非辅助 loss** | 核心是算子本身。`LOSS = 原 BPR`。HSDIR 的共属图蒸馏可作为**可选增益项**（默认 `λ=0`），用于消融而非主干 |
| **P3 语义进前向** | `sem(b,l,k)` 直接进入路由打分。语义相干的历史 item 会**自然地**被分配到对应 interest——不需要 loss 去"教" |
| **P4 显式竞争** | `softmax_K`：每个 item 的归属是一个在 K 个 interest 上的概率分布。一个 item 强烈归属 interest 1 时，就不可能同时强烈归属 interest 2。这是**定义上的竞争** |

### 3.4 为什么这个设计能绕开 HSDIR 的失败

HSDIR 监督的是**行为共属图** `G_route = R @ Rᵀ`——一个**二阶、置换不变、按用户平均**的统计量。
它可以被大量不同的路由配置满足，其中绝大多数对排序没有帮助；而且它只在训练期起作用，
测试期的前向仍然是 `softmax_L`。

SCIR 把路由**本身**变成机制：任何结构性的路由行为都是前向算子的**结果**，
因此**在测试期必然存在**，不需要靠 loss 去维持。
若之后要叠加 HSDIR 的共属图蒸馏，它作用的对象也变成了新的 `R`，是锦上添花而非支柱。

### 3.5 与已有工作的关系（论文定位需要）

该算子在形式上接近 **slot attention / soft k-means 聚类路由**：
K 个 slot（interest）对输入集合（history）做竞争性分配，再用分配权重做加权平均。
本章的贡献点是：

1. 把该算子引入**多兴趣序列推荐**的 `history → interest` 环节，并论证它比 attention 路由更适合该任务；
2. 在分配打分中引入**语义通道** `sem(b,l,k)`，且该通道**完全复用已冻结的 ASPCF item 表示**，零额外表示学习成本；
3. **P4 的竞争性是显式的**，可直接用分配分布 `R` 度量（熵、active K），诊断指标与 Ch3 的诊断工具链通用。

---

## 4. 与现有代码的复用关系

### 4.1 直接复用（不改动）

| 组件 | 位置 | 用途 |
|---|---|---|
| `ItemEncoder(mode="aspcf")` | `llmmi_components.py` | item 表示。**完全不动**；`s` 从 `e[:, :32]` 切片 |
| `InterestAggregator` | `llmmi_components.py:352` | `w_k`（history-only），**不变** |
| `LLMMIRecASPCF.py` 的 forward step 1–3、5–7 | `LLMMIRecASPCF.py:156-238` | 模型骨架 |
| 位置编码 / dropout / NaN 检查 / `return_intermediate` 契约 | 同上 | — |
| `main.py` 模型注册 | `main.py` | 加一行 import |
| `new_bash/run_llmmirec_aspcf_phase2_{beauty,ml1m}.sh` | `new_bash/` | 超参配置模板 |
| HSDIR 的 `return_route_scores` 管线 | `llmmi_components.py:274-345` | 暴露 `S` 供诊断 |
| `tools/analyze_cgscd_itemrep.py` | `tools/` | 表示诊断（可小改复用） |

### 4.2 需要新增

| 组件 | 说明 | 预计行数 |
|---|---|---|
| `CompetitiveInterestRouter` | 新类，放进 `llmmi_components.py`。复用现有 `Wq/Wk/Wv` 的**参数形状与 mask 约定**，只替换算子；新增 `A_k`（K×32）与 `m` 的头 | ~130 |
| `LLMMIRecSCIR.py` | 模型，以 `LLMMIRecASPCF.py` 为骨架 | ~290 |
| `tools/test_llmmirec_scir.py` | CPU 单测：分配归一化、padding 零、`β=0` 退化路径、梯度覆盖、与 ASPCF 回归 | ~230 |
| `tools/analyze_scir_routing.py` | 诊断：`R` 的熵 / active-K / 与语义的一致性；对比 attention 路由 | ~260 |
| `new_bash/run_llmmirec_scir_phase1_beauty.sh` | 实验脚本 | ~110 |

### 4.3 明确不复用

| 不复用 | 原因 |
|---|---|
| `QueryMultiInterestExtractor` 作为路由算子 | 正是要被替换的对象。但复用其参数初始化与 mask 逻辑 |
| CAISD / TASID 的 teacher 与 KL | 属"训练期监督"类型（§1.3 类型 A），不作为主干 |
| CASIR 的残差注入 | 属类型 C，与 ASPCF 表示耦合 |
| HSDIR 的 loss 作为主干 | 降级为**可选增益项**，默认 `λ_hsr=0` |

---

## 5. 最小实验矩阵

**第一阶段只在 Beauty、seed 42 上跑，4 个新配置，2 张卡并行约 40 分钟。**
全部 `LOSS = 原 BPR`（无任何辅助 loss）。

| # | 配置 | 隔离的变量 | 关键 flag |
|---|---|---|---|
| **0** | ASPCF（冻结基线） | — | 已有：0.1592 / 0.1088 |
| **1** | SCIR，`β=0`，含质量项 `m` | **路由算子变更本身**（无任何语义参与） | `--scir_sem_beta 0 --scir_use_mass 1` |
| **2** | SCIR，`β=0`，无质量项 `m` | 质量项是否必要 | `--scir_sem_beta 0 --scir_use_mass 0` |
| **3** | SCIR，`β>0`，含质量项 `m` | **语义进入前向路由的增量** | `--scir_sem_beta 1.0 --scir_use_mass 1` |
| **4** | 容量对照：ASPCF + 与 SCIR 等量的额外参数 | 排除"多参数带来的收益" | 新增约 `K×32 + 65` 个参数加在 ASPCF 上 |

**判读逻辑**：
- 若 ① ≈ ⓪ → 路由算子变更本身无效 → **停止**，不做 ②③
- 若 ① > ⓪ 且 ③ ≈ ① → 竞争路由有效，但语义通道无增量 → 保留算子，去掉语义通道
- 若 ③ > ① → 语义进入前向路由确有增量 → 这是本章的核心证据
- 若 ① ≈ ④ → 收益来自参数量而非算子 → **停止**

**成功判据**：Beauty seed 42 上 NDCG@5 > 0.1088（ASPCF 同 seed），
且最终目标是 `FINAL_EXPERIMENT_TARGETS.md` 的 SATCRec Beauty N@5 = 0.1107。

**通过后才进入**：ML-1M 同配置同 seed（**硬性要求，见 §6**）。

### 5.1 参数（预计）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scir_sem_beta` | 1.0 | 语义通道权重 |
| `--scir_sem_tau` | 0.1 | 语义通道温度 |
| `--scir_route_tau` | 1.0 | 分配温度 `τ_r` |
| `--scir_use_mass` | 1 | 是否使用质量项 `m` |
| `--scir_route_mode` | `competitive` | `competitive` / `attention`（后者= 复现 baseline 用于对照） |

`--scir_route_mode attention` 让新模型文件能**逐字复现 ASPCF 路由**，作为同一代码路径内的对照，
避免"两个模型文件配置漂移"这类历史问题。

---

## 6. 风险与停止条件

| # | 风险 | 监测指标 | 缓解 / 停止条件 |
|---|---|---|---|
| **R1** | `softmax_K` 在 K=4 上是硬瓶颈，分配可能塌缩（所有 item 挤到 1 个 interest） | 分配熵、active-K（`R` 的列质量分布） | 调 `τ_r`；若 active-K ≈ 1 且无法通过温度缓解 → 停止 |
| **R2** | 丢掉 attention 的"按重要性加权"能力会伤性能 | ① vs ⓪ | 这正是质量项 `m` 的作用；若 ① 与 ② 都差于 ⓪ → 停止 |
| **R3** | 语义通道 `s = e[:, :32]` 是**与另一半共适应**的表示，训练早期不稳定 | `sem` 分数的方差、`A_k` 的范数 | 对 `s` 做 L2 normalize；必要时 detach；若 `β>0` 明显差于 `β=0` 且 `A_k` 发散 → 说明该表示不适合做路由信号 |
| **R4** | **数据集依赖**（`CHAPTER3_FINAL.md` §6.3 已第四次出现） | Beauty vs ML-1M | **ML-1M 是硬性判据**，Beauty 通过不构成成功 |
| **R5** | 参数量增加带来的混淆 | 配置 ④ | 若 ① ≈ ④ → 判定为容量效应，停止 |
| **R6** | 与 ASPCF 的超参不匹配（历史教训：HSDIR 用 lr=0.008/bs=2048 与冻结基线不可比） | — | 严格复用 `run_llmmirec_aspcf_phase2_beauty.sh` 的 lr/batch/K/emb/history_max/dropout/l2 |

---

## 7. 本阶段不做的事

- ❌ 不写任何代码
- ❌ 不训练
- ❌ 不改 ASPCF `ItemEncoder` 或 `LLMMIRecASPCF`
- ❌ 不以增加 auxiliary loss 作为本章核心创新
- ❌ 不启动 Chapter 5

---

## 8. 主候选：Residual Competitive Interest Routing (RCIR)

> 状态：**当前主候选**。记录备用，**本阶段不实现**。
> 替代 §3 的 SCIR 作为首选方案，因为它是 §3 的**严格嵌套特例**（见 §8.4），
> 工程成本更低、风险更可控、且失败时可完全归零。

### 8.1 定义

保留原 collaborative logits：

```
C[k,l] = (Wq(q_k) · Wk(h_l)) / √d          # [B, K, L]  与现 ASPCF 完全一致
```

增加 **behavior-to-interest competition**（在 K 维上归一化）：

```
R_cf[k,l] = softmax_K( C[k,l] / τ )        # [B, K, L]  每个 history item 在 K 个 interest 间的归属分布
```

**不直接用 `R_cf` 替换原 attention**，而是作为 logit 空间的残差：

```
C_new[k,l] = C[k,l] + λ_comp · log( K · R_cf[k,l] + ε )
```

最终仍使用原有的 history 维 softmax：

```
A[k,l] = softmax_L( C_new[k,l] )           # ★ 仍然是 softmax over L
V_k    = Σ_l A[k,l] · value_l
```

### 8.2 数学简化（实现时的重要性质）

`log(K · R_cf + ε) = log K + log(R_cf + ε/K)`。

由于后续 `softmax_L` 是**在 l 维上**做的，而 `log K` 对固定的 k 是一个常数，
在 `softmax_L` 中被完全消去。因此等价地：

```
C_new[k,l] = C[k,l] + λ_comp · log( R_cf[k,l] + ε )
```

即：**只有 `log softmax_K(C/τ)` 这一项真正起作用**，`K·` 系数不影响结果。

### 8.3 与现有代码的关系（重要复用点）

`QueryMultiInterestExtractor` **已经实现了这个残差机制**：

```python
# models/sequential/llmmi_components.py
if attention_prior is not None and prior_strength > 0:
    scores = scores + prior_strength * torch.log(attention_prior + 1e-8)
```

该 `attention_prior` / `prior_strength` 钩子由 HSDIR/CHIR 引入并已在 CHIR 中使用过。

**因此 RCIR 的实现只需要**：把 `attention_prior` 传成 `softmax_K(C/τ)`（自竞争分布），
`prior_strength = λ_comp`。**算子、mask、value 投影、聚合路径全部不变。**

已知细节：padding 位置的 `R_cf` 为 0，`log(1e-8) ≈ −18.4`，
但由于 padding 随后被 mask 成 `−inf`，**无实际影响**（与 CHIR/HSDIR 的既有行为一致）。

### 8.4 关键性质

| 性质 | 说明 |
|---|---|
| **严格嵌套** | `λ_comp = 0` 时 `C_new = C`，**逐字退化为当前 ASPCF extractor**。这消除了"新模型文件与基线配置漂移"的历史风险 |
| **非辅助 loss** | 它改变的是**前向路由 logits**，不是训练期监督，满足设计原则 P1 |
| **显式竞争** | `R_cf` 是 item 在 K 个 interest 上的归属分布；某 interest 占优时其余被压低，构成竞争（P4） |
| **不引入语义** | 第一版**不含** semantic routing。语义通道留待诊断确认 routing 是瓶颈后再考虑 |
| **改动极小** | 新增仅：一个 `τ`、一个 `λ_comp`，以及 `R_cf` 的计算（一次 softmax）。参数量 **+0** |

### 8.5 第一版明确不加入

按用户约束，第一版**不包含**：

- ❌ semantic routing（语义通道）
- ❌ quality gate（质量项 `m`）
- ❌ auxiliary loss（任何形式）
- ❌ HSDIR loss（共属图蒸馏）
- ❌ coverage / focus

即第一版是**纯结构、零新增参数、零新增 loss**的最小干预。

### 8.6 最小实验矩阵（待诊断结论后启动）

| # | 配置 | 说明 |
|---|---|---|
| 0 | ASPCF 基线 | 已有：Beauty 0.1592/0.1088；ML-1M 0.3068/0.2134（seed 42） |
| 1 | RCIR，`λ_comp = 0` | **必须逐位复现基线**——验证实现的正确性（sanity check） |
| 2 | RCIR，`λ_comp > 0`（单个合理默认值） | 竞争是否带来增益 |
| 3 | RCIR，`τ` 敏感性（仅当 ② 有希望） | 0.1 / 0.5 / 1.0 |

**判据**：(1) 必须与基线数值完全一致（否则实现有误）；
(2) 若 ② 在 Beauty 上无增益 → 结合诊断结论判断是否转向 candidate-aware scoring；
(3) 任何结论都必须先在 ML-1M 上复验（`CHAPTER3_FINAL.md` §7.4 硬性要求）。

---

## 9. 三方案关系总结

| | SCIR（§3，DRAFT） | **RCIR（§8，主候选）** | Candidate-aware scoring（备选） |
|---|---|---|---|
| 改变的位置 | 路由算子（`softmax_L` → `softmax_K`） | 路由 **logits**（加竞争残差） | 打分路径 |
| 与基线的关系 | 替换算子，非嵌套 | **严格嵌套**（λ=0 即基线） | 新增分支 |
| 新增参数 | `A_k`(K×32) + `m` 头 | **0** | 待定 |
| 破坏性 | 高（丢弃 attention 的加权能力） | **低** | 中 |
| 与 P1/P4 的符合度 | 强 | 中（改 logits 而非算子） | 弱（P1 不满足） |
| 当前状态 | 保留为备选 | **待诊断后决定** | 待诊断后决定 |

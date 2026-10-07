# CHAPTER4_DESIGN.md

> Chapter 4 方向设计（**只给方案，不写代码、不训练**）
> 更新日期：2026-10-06（第二次修订）
> 前置约束：Chapter 3 = frozen ASPCF（`CHAPTER3_FINAL.md`），ASPCF `ItemEncoder` 完全冻结。

---

## 0. 修订说明与明确排除项

### 0.1 本版取代的内容

| 内容 | 处置 |
|---|---|
| 旧 §3 SCIR（Semantic-Competitive Interest Routing，用 `softmax_K` **替换** `softmax_L`） | ❌ **superseded**，不再作为候选。它丢弃了 attention 在 history 上的加权能力，破坏性过大 |
| 旧 §8 RCIR（Residual Competitive Interest Routing） | ✅ 保留并**升级为 Module 1**，纳入核心结构 |
| Adaptive Interest Cardinality / dynamic-K | ❌ **明确排除，不是 Chapter 4 的创新**（见 §0.2） |

### 0.2 明确排除：K 的选择属于公平调参，不是创新

**PoMRec 的标准配置本来就按数据集调 K**：Beauty `K=4`，ML-1M `K=2`。
因此 ASPCF 也必须被允许做同样的 K 选择——**这是公平调参，不是方法贡献**。

据此：
- ❌ 不把 "Adaptive / Dynamic Interest Cardinality" 写成 Chapter 4 的核心创新；
- ❌ 不做 K 敏感性扫描、不做 learnable K；
- ✅ 只补一个**最小 K baseline**（K=2），用于
  1. 量化 ML-1M 当前约 2% 差距中有多少只是 K 造成的；
  2. 给 Chapter 4 建立**公平的 dataset-specific ASPCF baseline**。

K=2 baseline 的实验设置见 §6，其结果只作为**对照基线**出现在论文中，不作为 contribution。

---

## 1. 失败复盘：已有探索为何全部没有稳定收益

### 1.1 已有方法与真实结果

| 方法 | 核心机制 | 作用位置 | Beauty NDCG@5 | ML-1M NDCG@5 | seeds |
|---|---|---|---|---|---|
| **ASPCF**（Ch3 冻结基线） | — | — | **0.1075 ± 0.0014** | **0.2140 ± 0.0023** | 5 |
| HSDIR（best: hier+agreement） | 行为共属图蒸馏 | 训练期 loss | 0.1098 † | 0.2156 † | 1 |
| HSDIR（同配置 λ=0 对照） | — | — | 0.1053 † | 0.2144 † | 1 |
| CHIR（dual_routing） | prototype query + 双视角路由 | 前向（query 来源） | 0.1069 | — | 1 |
| CASIR（complement_coherence） | 语义残差注入打分路径 | 前向（残差加性） | 0.1017 | 0.2141 | 1 |
| CAISD（5-seed, responsibility） | 兴趣语义 profile 蒸馏 | 训练期 loss | **0.10738** | **0.21042** | 5 |
| TASID（llm_only / asymmetric） | target 条件蒸馏 | 训练期 loss | 0.1082 | 0.2135 | 1 |

† HSDIR 在其**自己的**超参配置下运行，与冻结 ASPCF **不同配置，不可直接比较**。

**唯一有 5-seed 公平对比的是 CAISD：Beauty 打平、ML-1M 落后 ASPCF 1.68%。**

### 1.2 失败共性

| 类型 | 方法 | 问题 |
|---|---|---|
| **A. 训练期监督** | HSDIR、CAISD、TASID | 前向路由算子完全不变。teacher 只改变参数落点，测试期模型做的事与 baseline 逐字相同 |
| **B. 只改 query 来源** | CHIR | 路由算子仍是 `softmax over L`，表达能力未变 |
| **C. 打分路径加残差** | CASIR | 改 `V_k` 的数值而非 `history → V_k` 的映射 |

**结论：必须改变 `history → K interests` 与 `interests → user vector` 的前向算子本身。**

---

## 2. 设计原则

| # | 原则 | 来源 |
|---|---|---|
| **P1** | 改变**前向算子**，测试期计算方式必须与 baseline 不同 | HSDIR 教训 |
| **P2** | 核心创新不能是"再加一个 auxiliary loss" | 用户约束 |
| **P3** | 不依赖新增 auxiliary loss 才成立（`LOSS = 原 BPR`） | 用户约束 |
| **P4** | K 个 interest 之间要有**显式竞争** | 用户约束 |
| **P5** | ASPCF `ItemEncoder` 完全冻结 | 用户约束 |
| **P6** | 必须在 Beauty **和** ML-1M 同时验证 | `CHAPTER3_FINAL.md` §6.3 |
| **P7** | 与基线的退化关系必须严格：某个超参置零时应**逐字**回到 baseline | 工程安全性 |

---

## 3. 核心问题（本版新定义）

> ### 当前 `QueryMultiInterestExtractor` 对每个 interest **独立**沿 history 做 softmax，
> ### 不同 interests 之间**没有显式竞争**，
> ### 因此多个 interest 可能重复关注相同的历史行为，并形成冗余表示。

### 3.1 为什么这是问题：当前算子的结构

```
C[k,l] = (Wq(q_k) · Wk(h_l)) / √d          # [B, K, L]
A[k,l] = softmax over L( C[k,l] )          # ★ 逐 k 独立，softmax 只在 l 维
V_k    = Σ_l A[k,l] · value_l
```

`softmax` 在 `l` 维上且**逐 interest 独立**：
同一个历史位置可以同时以接近 1.0 的权重进入**所有** K 个 interest，
**没有任何机制阻止两个 interest 学到相同的注意力分布**。

### 3.2 实测证据（Chapter 4 Phase 0 诊断 + HSDIR 诊断）

| 指标 | Beauty ASPCF | ML-1M ASPCF | 上界 |
|---|---|---|---|
| interest 两两余弦（mean） | **0.8699** | **0.7876** | 1.0 |
| interest 有效秩 effR（mean） | **1.110** | **1.615** | K−1 = 3 |
| attention entropy | 1.5051 | 2.4691 | — |
| 聚合权重熵 wEnt | **1.3854** | 1.2865 | ln4 = 1.386 |

- **effR 仅为上界（K−1=3）的 37% / 54%**：4 个 interest 高度共线。
- **Beauty 上聚合权重熵 = 1.3854 ≈ ln4，且 p05 = p50 = p95 = 1.386**：
  聚合权重**在所有样本上完全相同且恰好均匀**——`InterestAggregator` 实际退化为"对 4 个兴趣取平均"，
  没有在做选择。
- 对比：HSDIR 诊断中 baseline 的 interest 余弦为 0.9457、有效秩 1.714。

**结论：兴趣冗余（模块 1 要解决的）与聚合失效（模块 2 要解决的）都是实测存在的，不是假设。**

---

## 4. 两个核心前向模块

Chapter 4 的核心由**两个前向模块**构成，不依赖任何新增 auxiliary loss。
`LOSS = 原 BPR`。

### 4.1 Module 1 — Competitive Diversified Interest Routing (CDIR)

在**不替换**原 attention 的前提下，向 logit 空间注入 behavior-to-interest 竞争。

```
# ── 原 collaborative logits（不变）──
C[k,l] = (Wq(q_k) · Wk(h_l)) / √d                    # [B, K, L]

# ── 原 attention（不变）──
A[k,l] = softmax over L( C[k,l] )

# ── 新增：behavior-to-interest ownership ──
R[k,l] = softmax over K( C[k,l] / τ )                # [B, K, L]  每个行为在 K 个兴趣间的归属

# ── 新增：residual competitive routing（★ 不直接用 R 替换 A）──
C_new[k,l] = C[k,l] + λ_comp · log( K · R[k,l] + ε )

# ── 最终仍用 history 维 softmax ──
A[k,l] = softmax over L( C_new[k,l] )
V_k    = Σ_l A[k,l] · value_l
```

**数学简化**（实现时的重要性质）：`log(K·R + ε) = log K + log(R + ε/K)`。
`log K` 对固定的 k 是常数，在 `softmax_L` 中被消去，因此等价于

```
C_new[k,l] = C[k,l] + λ_comp · log( R[k,l] + ε )
```

**关键性质**

| 性质 | 说明 |
|---|---|
| **严格退化（P7）** | `λ_comp = 0` 时 `C_new = C`，**逐字回到当前 extractor** |
| **改前向（P1）** | 改变的是前向 logits 与 `V_k` 的取值，测试期生效 |
| **显式竞争（P4）** | `R` 在 K 维归一化：某 interest 占优时其余被压低 |
| **零新增参数** | 只多 `τ`、`λ_comp` 两个标量 |
| **与已有代码兼容** | `QueryMultiInterestExtractor` 已有 `attention_prior` / `prior_strength` 钩子（HSDIR/CHIR 引入并已使用），本模块只需把 prior 传为 `softmax_K(C/τ)` |

> 注：padding 位置 `R=0` → `log(ε)` 很小，但随后被 mask 成 `−inf`，无影响（与 CHIR/HSDIR 既有行为一致）。

### 4.2 Module 2 — Evidence-Aware Interest Aggregation (EAIA)

**问题**：当前 `InterestAggregator` 计算 `w_k` 时**看不到 interest 本身**。

```
# 现状
context = LayerNorm( mean_his + last_his )      # [B, D]
logits  = MLP(context)                          # [B, K]  ← 与 V_k 无关
w       = softmax_k(logits)
```

权重只由 history 摘要决定，与"第 k 个 interest 实际是什么"无关。
这解释了 §3.2 的实测：Beauty 上 `w` 在所有样本上恒为均匀分布。

**新设计**：让 `w_k` 同时看到 interest 向量、其**占用度/证据量**、以及 history 摘要。

```
# occupancy / evidence（来自 Module 1 的 R）
o_k = mean over l( R[k,l] )                     # [B, K]  第 k 个兴趣占用了多少行为

# history summary（复用现有 InterestAggregator 的构造）
h_sum = LayerNorm( mean_his + last_his )        # [B, D]

# 融合
u_k = MLP_a( [ V_k ; o_k ; h_sum ] )            # [B, K, H]
w   = softmax over K( Linear(u_k) )             # [B, K]
```

**必须保持的性质**

- **history-only，无 target 泄露**：`V_k`、`o_k`、`h_sum` 全部只由 history 计算，不含 candidate/target 信息。
  这一点与现状一致，必须保持。
- `o_k` 在 `λ_comp = 0` 时**依然有定义**（`R = softmax_K(C/τ)` 与 `λ_comp` 无关），
  因此 Module 2 可以**单独**开启/关闭，支持 §5 的 2×2 归因。

**为什么这能修正 §3.2 的退化**：`w_k` 现在能看到 `V_k` 的实际内容与 `o_k`
——冗余或"空"的 interest（`o_k` 小）可以被自动降权，而不是被迫均分。

### 4.3 两个模块与原则的对应

| 原则 | Module 1 | Module 2 |
|---|---|---|
| P1 改前向 | ✅ 改 logits 与 `V_k` | ✅ 改 `w_k` 的生成 |
| P3 无 aux loss | ✅ 纯结构 | ✅ 纯结构 |
| P4 显式竞争 | ✅ `softmax_K` | ✅ `o_k` 反映竞争结果 |
| P7 严格退化 | ✅ `λ_comp=0` | ✅ 可单独关闭，回到原 `InterestAggregator` |

---

## 5. 最小实验矩阵（2×2 归因）

**先只跑 Beauty、seed 42。全部 `LOSS = 原 BPR`，无任何 auxiliary loss。**

| # | 配置 | Module 1 | Module 2 | 隔离的变量 |
|---|---|---|---|---|
| **0** | ASPCF（冻结基线） | — | — | 已有 |
| **0'** | **ASPCF K=2**（公平 baseline） | — | — | K 的影响（§6） |
| **1** | 仅 Module 1 | `λ_comp>0` | 关 | 竞争路由本身的增量 |
| **2** | 仅 Module 2 | `λ_comp=0` | 开 | 证据感知聚合本身的增量 |
| **3** | Module 1 + 2 | `λ_comp>0` | 开 | 两者的联合（核心配置） |
| **4** | sanity：`λ_comp=0` + Module 2 关 | `=0` | 关 | **必须逐位复现 baseline**，否则实现有误 |

**判读逻辑**

- 配置 ④ 若与 baseline 不完全一致 → 实现有 bug，先修再谈其他
- 若 ① ≈ ⓪ 且 ② ≈ ⓪ → 两个模块都无效 → **停止**，转向 candidate-aware scoring
- 若 ② > ⓪ 而 ① ≈ ⓪ → 瓶颈在 aggregation，不在 routing → 与诊断结论交叉验证
- 若 ① > ⓪ 而 ② ≈ ⓪ → 瓶颈在 routing
- 若 ③ > max(①, ②) → 两模块互补，这是本章的核心证据

**结构指标（次要但重要，HSDIR 的教训）**：必须同时报告
`interest 两两余弦`、`effR/(K−1)`、`active_K`、`wEnt`，
并确认结构改变**出现在测试期前向**（两个模块都在 forward 中，因此必然满足）。

**参数**（预计）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cdir_lambda_comp` | 0.0 | `λ_comp`；0 = 严格退化 |
| `--cdir_tau` | 1.0 | ownership 温度 `τ` |
| `--eaia_enable` | 0 | Module 2 开关 |
| `--eaia_hidden` | 64 | Module 2 隐层 |

---

## 6. K=2 公平 baseline 的实验设置（非创新，仅对照）

| 项 | 值 |
|---|---|
| 目的 | ① 量化 ML-1M 约 2% 差距中 K 的贡献；② 建立 dataset-specific ASPCF baseline |
| 配置 | **除 `--K` 外，与 frozen ASPCF 稳定配置逐字相同**（含各数据集 lr：beauty 0.004 / ml-1m 0.001） |
| 新增运行 | Beauty `K=2` seed 42；ML-1M `K=2` seed 42 |
| 不重跑 | K=4（已有 5 seeds） |
| 不扫描 | 不测其他 K 值，不做 learnable K |
| 脚本 | `new_bash/run_llmmirec_aspcf_K.sh`、`new_bash/run_aspcf_k2_baseline.sh` |
| 输出 | `new_log/llmmirec_aspcf_K2/<ds>/` |
| 论文定位 | **对照基线**，不作为 contribution |

**当前状态**：Beauty K=2 已启动运行；ML-1M K=2 待 Beauty 结果出来后再决定是否启动。

---

## 7. 与现有代码的复用关系

### 7.1 直接复用（不改动）

| 组件 | 位置 | 用途 |
|---|---|---|
| `ItemEncoder(mode="aspcf")` | `llmmi_components.py` | **完全冻结，不碰** |
| `QueryMultiInterestExtractor` 的 `Wq/Wk/Wv` 与 mask 约定 | `llmmi_components.py:250-345` | Module 1 复用，且其 `attention_prior`/`prior_strength` 钩子已存在 |
| `InterestAggregator` 的 `h_sum = LayerNorm(mean + last)` | `llmmi_components.py:352-409` | Module 2 复用该 history 摘要构造 |
| `LLMMIRecASPCF.py` forward step 1–3、5–7 | `LLMMIRecASPCF.py:156-238` | 模型骨架 |
| 「`λ=0` 严格退化」的既有先例 | TASID `tasid_mode=none` | 设计模式一致 |
| 诊断工具 | `tools/diagnose_ch4_phase0.py` | 结构指标可直接复用 |

### 7.2 需要新增

| 组件 | 说明 | 预计行数 |
|---|---|---|
| Module 1 实现 | 在 `QueryMultiInterestExtractor` 内计算 `R = softmax_K(C/τ)` 并作为 prior 注入；**或**新增 `CompetitiveInterestRouter` 类 | ~90 |
| Module 2 实现 | `EvidenceAwareAggregator`（`V_k` + `o_k` + `h_sum` → `w`） | ~70 |
| `LLMMIRecCDIR.py` | 模型，以 `LLMMIRecASPCF.py` 为骨架 | ~300 |
| `tools/test_llmmirec_cdir.py` | CPU 单测：`λ_comp=0` 逐位复现、padding、梯度、EAIA 开关、ASPCF 回归 | ~250 |
| 实验脚本 | `new_bash/run_llmmirec_cdir_phase1_beauty.sh` | ~110 |

### 7.3 第一版明确不使用（用户约束）

- ❌ semantic routing（语义通道）
- ❌ HSDIR / CAISD / TASID loss
- ❌ coverage / focus loss
- ❌ 额外的 quality gate
- ❌ 任何 auxiliary loss

---

## 8. 风险与停止条件

| # | 风险 | 监测指标 | 缓解 / 停止条件 |
|---|---|---|---|
| **R1** | `softmax_K` 在 K=4 上过尖，`R` 退化为 one-hot，`log R` 变成极端项 | `R` 的熵、`λ_comp` 的有效幅值 | 调 `τ`；若 `R` 熵 ≈ 0 且无法通过温度缓解 → 停止 |
| **R2** | Module 1 与 attention 重复，无增量 | 配置 ① vs ⓪ | 若 ① ≈ ⓪ → 瓶颈不在此 |
| **R3** | Module 2 参数少、表达力不足 | 配置 ② vs ⓪ | 若 ② ≈ ⓪ 但诊断显示 aggregation 确实退化 → 增大 `--eaia_hidden` 前先确认不是实现问题 |
| **R4** | **数据集依赖**（`CHAPTER3_FINAL.md` §6.3 已四次出现） | Beauty vs ML-1M | **ML-1M 是硬性判据** |
| **R5** | K 的选择混淆 | 配置 ⓪ vs ⓪' | 所有 Chapter 4 比较必须对照 **K 匹配**的 baseline |
| **R6** | 超参与 baseline 不匹配（HSDIR 的历史教训） | — | 严格复用稳定脚本的 lr/batch/emb/history_max/dropout/l2 |

---

## 9. 后续 TODO（仅记录，本阶段不执行）

### TODO-1：ML-1M 训练时间 profiling 与 fast-screening 配置

**背景**：ML-1M 单次正式训练约 **80 分钟**（seed 42 为 4279 s；5 seeds 均值约 4970 s），
Beauty 约 18 分钟。Chapter 4 要做的是 **2×2 结构归因 + 多个结构变体**，
若全部按 batch=1024 跑，实验轮次成本会显著拖慢迭代。

**待办**：

1. 单独 profile ML-1M 的训练时间构成（数据加载 / 前向 / 反向 / 评测各占多少），
   确认瓶颈在算力还是 DataLoader；
2. 在此基础上考虑建立 **batch=2048 的 fast-screening 配置**，用于结构探索阶段的快速筛选。

**硬性约束（必须写进任何使用该配置的实验）**：

> ⚠️ **fast-screening 配置必须有它自己的 ASPCF baseline，不得与 batch=1024 的正式结果直接比较。**
>
> 理由：batch=2048 属于**不同的训练 regime**（已有 HSDIR 的历史教训——
> 它在 lr=0.008/bs=2048 下运行，与冻结 ASPCF 的 lr=0.004/bs=1024 不可比，
> 导致其增益无法与基线对照）。
>
> 因此 batch=2048 只能用于**同一配置内部**的相对比较（结构 A vs 结构 B），
> 任何要写进论文的绝对数字必须在 batch=1024 的正式配置下重跑。

**当前状态**：❌ 不执行 profiling，不建立 fast-screening 配置。

---

## 10. 本阶段不做的事

- ❌ 不实现 Chapter 4 的任何模块
- ❌ 不训练除 K=2 baseline 之外的任何模型
- ❌ 不把 K 选择写成方法贡献
- ❌ 不启动 Chapter 5
- ❌ 不修改 ASPCF `ItemEncoder` / `LLMMIRecASPCF`

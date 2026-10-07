# CHAPTER4_DESIGN.md

> Chapter 4 方向设计
> 更新日期：2026-10-07（第三次修订）
> 前置约束：Chapter 3 ASPCF **结构**冻结（`CHAPTER3_FINAL.md`）。
> **注意：「冻结」指结构冻结，不是冻结梯度** —— 新模型仍按现有方式 end-to-end 训练，
> 并保留 Chapter 3 的 `lambda_relation = 0.01`。

---

## 0. 修订说明

| 内容 | 本版处置 |
|---|---|
| **ECTIR**（Evidence-Constrained Transport Interest Routing） | ✅ **主候选**（§4） |
| CDIR + EAIA（Residual Competitive Routing + Evidence-Aware Aggregation） | 🔸 降为 **simple candidate / ablation**（§5），不再是首选 |
| SCIR（用 `softmax_K` 替换 `softmax_L`） | ❌ superseded |
| Adaptive Interest Cardinality / dynamic-K | ❌ **不是创新**；K 属公平调参（§6） |

---

## 1. 失败复盘

| 方法 | 作用位置 | Beauty NDCG@5 | ML-1M NDCG@5 | seeds |
|---|---|---|---|---|
| **ASPCF**（Ch3 冻结基线） | — | **0.1075 ± 0.0014** | **0.2140 ± 0.0023** | 5 |
| HSDIR | 训练期 loss | 0.1098 † | 0.2156 † | 1 |
| CHIR | 前向 query 来源 | 0.1069 | — | 1 |
| CASIR | 打分路径残差 | 0.1017 | 0.2141 | 1 |
| CAISD | 训练期 loss | 0.10738 | 0.21042 | 5 |
| TASID | 训练期 loss | 0.1082 | 0.2135 | 1 |

† 配置不同，不可直接比较。**唯一 5-seed 公平对比（CAISD）Beauty 打平、ML-1M 落后 1.68%。**

**失败共性**：训练期监督（A 类）前向算子不变；只改 query 来源（B 类）表达能力不变；
打分路径加残差（C 类）改的是数值而非映射。**必须改变前向算子。**

---

## 2. 设计原则

| # | 原则 |
|---|---|
| **P1** | 改变**前向算子**，测试期计算方式与 baseline 不同 |
| **P2** | 核心创新不能是"再加一个 auxiliary loss" |
| **P3** | 训练目标保持 **BPR + Chapter 3 现有 relation loss**，不新增 Chapter 4 辅助损失 |
| **P4** | K 个 interest 之间要有**显式竞争 + 全局联合分配** |
| **P5** | ASPCF `ItemEncoder` **结构与代码完全不动** |
| **P6** | 必须在 Beauty **和** ML-1M 同时验证 |
| **P7** | 必须有可验证的退化路径（能严格复现原 extractor） |
| **P8** | 复杂度可控：K=4、history_max=20，Sinkhorn ≤5 次、refinement ≤2 次 |

---

## 3. 核心问题

> ### 当前 `QueryMultiInterestExtractor` 对每个 interest **独立**沿 history 做 softmax
> ### （`A[k,l] = softmax_l(C[k,l])`），K 个 interest 之间没有**全局联合分配**，
> ### 因此多个 interest 可能重复关注相同历史行为，形成冗余表示。

### 3.1 实测证据（`CHAPTER4_DIAGNOSIS.md`）

| 事实 | 数值 |
|---|---|
| F1 ASPCF interest 冗余 | 两两余弦 **0.8699**（Beauty）/ **0.7876**（ML-1M）；effR **1.110 / 1.615**，仅为上界 (K−1=3) 的 **37% / 54%** |
| F2 Beauty 上聚合权重恒为均匀 | wEnt = **1.3854 ≈ ln4**，且 **p05 = p50 = p95** |
| F3 Beauty 上 PoMRec 塌缩为单兴趣 | wEnt = 0，actK = 1.00 |
| F4 ML-1M 差距不在头部/长历史 | 在中流行度段 [186,423) |
| F5 Beauty 增益集中在长尾 | [1,4) 3.13× vs 头部 1.03× |

**判定：属情形 C（routing 与 aggregation 都有问题）** → Chapter 4 核心由两个前向环节共同构成。

### 3.2 为什么现有竞争机制不够（即使在 CDIR 中）

`softmax_K` 型的竞争有一个隐含约束：
**对每个 history 位置 l，`Σ_k R[k,l] = 1`。**
即每个位置把一份单位质量摊给 K 个 interest，**各 interest 的总容量因而被隐式地拉向均匀**。
这让"竞争"停留在**逐位置的局部归一化**层面，没有对**全局的 interest 容量分配**做任何约束。

**ECTIR 的出发点**：把 `history → interests`  explicitly 建模为**带边缘约束的最优传输**，
同时约束 **history 侧质量**与 **interest 侧容量**，且容量**不强制均匀**，而是由当前用户的
affinity evidence 产生。

---

## 4. 主候选：ECTIR

### 4.0 记号与 tensor shape

| 符号 | 含义 | shape |
|---|---|---|
| `B` | batch size | 标量 |
| `L` | `history_max`（padded 长度）= 20 | 标量 |
| `K` | interest 数 = 4 | 标量 |
| `D` | `emb_size` = 64 | 标量 |
| `d` | `attn_size` = 64 | 标量 |
| `h` | 加位置编码后的 history 表示 | `[B, L, D]` |
| `m` | valid mask（1=有效） | `[B, L]` |
| `q` | 可学习 interest query（复用原 `self.query`） | `[K, D]` |
| `Q^t` | 第 t 步的 interest query（投影后） | `[B, K, d]` |
| `Kmat` | `Wk(h)` | `[B, L, d]` |
| `S^t` | 第 t 步 affinity | `[B, L, K]` |
| `T^t` | 第 t 步 transport plan | `[B, L, K]` |
| `a` | history 侧边缘（行和） | `[B, L]` |
| `b` | interest 侧容量（列和） | `[B, K]` |
| `V^t` | 第 t 步 interest 表示 | `[B, K, D]` |
| `w` | 聚合权重 | `[B, K]` |

> **注意 shape 约定**：现有 `QueryMultiInterestExtractor` 内部是 `[B, K, L]`，
> ECTIR 的 transport 用 **`[B, L, K]`（行为为行、兴趣为列）**，最后转置回 `[B, K, ...]`。
> 这是与现有代码最主要的接口差异，实现时必须显式 `transpose(1, 2)`。

### 4.1 Module 1 — Evidence-Constrained Transport Routing

```
# ---- 1. affinity（与基线同源，仅转置）----
Q       = Wq(q).unsqueeze(0).expand(B, K, d)              # [B, K, d]
Kmat    = Wk(h)                                            # [B, L, d]
S       = torch.bmm(Kmat, Q.transpose(1,2)) / sqrt(d)      # [B, L, K]
S       = S.masked_fill(~m[:,:,None].bool(), NEG)          # padding 置 NEG（有限大负数）

# ---- 2. history 侧质量（行边缘）：有效位置均匀 ----
a       = m / m.sum(dim=1, keepdim=True).clamp(min=1)      # [B, L]  Σ_l a = 1

# ---- 3. interest 侧容量（列边缘）：由 affinity evidence 产生，不强制均匀 ----
e       = torch.logsumexp(S, dim=1)                        # [B, K]  每个 interest 对 history 的总证据
b       = softmax(e / tau_c, dim=-1)                       # [B, K]  Σ_k b = 1

# ---- 4. entropic transport（log-domain Sinkhorn，数值稳定）----
M       = S / eps                                          # [B, L, K]  cost = -S ⇒ kernel = exp(S/eps)
f       = torch.zeros(B, L);  g = torch.zeros(B, K)
for _ in range(n_sinkhorn):                                # n_sinkhorn ∈ [3,5]
    f   = torch.where(m.bool(), log_a - torch.logsumexp(M + g[:,None,:], dim=2), NEG)
    g   = log_b - torch.logsumexp(M + f[:,:,None], dim=1)
T       = torch.exp(M + f[:,:,None] + g[:,None,:]) * m[:,:,None]   # [B, L, K]，padding 严格为 0

# ---- 5. interest 表示（凸组合，与基线同形）----
Tm      = T.transpose(1, 2)                                # [B, K, L]
denom   = Tm.sum(dim=-1, keepdim=True).clamp(min=eps)      # [B, K, 1]
A       = Tm / denom                                       # [B, K, L]  行和为 1
V       = torch.bmm(A, Wv(h))                              # [B, K, D]
```

其中 `log_a = log(a + eps)`，`NEG = -1e4`（`exp(NEG)` 下溢为 0，不产生 NaN）。

**为什么这满足"同时体现竞争与证据获取"**

| 需求 | 实现 |
|---|---|
| behavior → interest 竞争 | 行和为 `a`：每个位置的单位质量必须在 K 个 interest 间分配（传输计划的行约束） |
| interest 对历史证据的获取 | 列和为 `b`：每个 interest 按其 affinity 证据 `e_k` 获得容量。证据强的兴趣占用更多历史质量 |
| **不是 K 个独立 softmax** | `T` 由**行/列边缘联合约束**求解得到，任一元素的变化会影响整行与整列 |

**关键性质**

- **容量非均匀**：`b` 由 `softmax(e/τ_c)` 产生，不是 `1/K`。
  `τ_c → ∞` 时退化为均匀容量（可作为一个消融点）。
- **padding 严格 mask**：`S` 在 padding 处为 `NEG` ⇒ `T` 在 padding 处精确为 0（乘 `m` 保证）。
- **数值稳定**：全程 log-domain；`m` 遮蔽的行的 `f` 显式取 `NEG` 而非 `-inf`，避免 `-inf - (-inf) = NaN`。
- ⚠️ **没有连续退化到 baseline**：transport 与 `softmax_l` 是不同的算子，
  不存在一个标量把前者连续变回后者（`ε→∞` 时 `T → a ⊗ b`，是外积而非 attention）。
  **因此 Module 1 的"退化"只能通过 `routing_mode=baseline` 的模式切换实现**（见 §4.5）。
  这一点与 CDIR 不同，必须如实记录。

### 4.2 Module 2 — Sequence-Specific Iterative Interest Refinement

```
V_cur = V^{(0)}                                    # 来自 Module 1
for t in 1..n_refine:                              # n_refine ≤ 2
    q_t   = q.unsqueeze(0) + W_r(V_cur)            # [B, K, D]  q 为原可学习 query（复用）
    Q_t   = Wq(q_t)                                # [B, K, d]  复用同一 Wq
    S_t   = bmm(Kmat, Q_t.transpose(1,2)) / sqrt(d)                    # [B, L, K]
    S_t   = S_t.masked_fill(~m[:,:,None].bool(), NEG)
    T_t   = sinkhorn(S_t, a, b)                                        # 同 Module 1
    V_cur = bmm(normalize(T_t.transpose(1,2)), Wv(h))                  # [B, K, D]
V = V_cur
```

- `W_r`：`nn.Linear(D, D)`，**新增参数 4,160 个**（D=64）。
- **初始化即恒等**：仓库统一 `N(0, 0.01)` 初始化 ⇒ 训练初始时 `W_r(V) ≈ 0` ⇒ `q_t ≈ q`，
  第 1 次 refinement 与 Module 1 等价。修正量只在训练发现有用时增长。
- **sequence-specific**：`q_t` 依赖 `V_cur`，而 `V_cur` 由**该用户自己的 history** 产生，
  因此 refinement 是 per-sequence 的，不是全局固定的。

### 4.3 Module 3 — Transport-Evidence Interest Aggregation

从**最终** transport 矩阵 `T` 提取每个 interest 的证据。至少三项：

```
mass_k     = T[:,:,k].sum(dim=1)                                   # [B, K]  传输质量（= 列边缘 b_k）
conc_k     = 1 - H( T[:,:,k] / mass_k ) / log(L_valid)             # [B, K]  分配集中度/置信度 ∈ [0,1]
rec_k      = Σ_l T[:,l,k] * pos_l / mass_k                         # [B, K]  时间质心（pos_l：越大越新）

h_sum      = LayerNorm(mean_his + last_his)                        # [B, D]  复用原 InterestAggregator 构造

u_k        = MLP_e( [ V_k ; mass_k ; conc_k ; rec_k ; h_sum ] )    # [B, K, H_e]
w          = softmax_k( Linear(u_k) )                              # [B, K]
```

`pos_l` 沿用现有位置编码约定：`position = (lengths - i) * valid`，即**越大越新**；
`rec_k` 越大表示该 interest 越偏向最近行为。

**必须保持**：`V_k`、`mass_k`、`conc_k`、`rec_k`、`h_sum` **全部只由 history 计算**，
不含 candidate/target 信息 → 无标签泄露，与现状一致。

> **定位说明**：`MLP_e` / `Linear` 只是把上述证据**参数化**的局部手段。
> **核心是证据特征本身**（transport mass / concentration / temporal centroid），
> 不是 gate 或 residual 结构。论文中不得把"加了一个 gate"写成贡献。

### 4.4 模块与原则的对应

| 原则 | Module 1 | Module 2 | Module 3 |
|---|---|---|---|
| P1 改前向 | ✅ 换算子 | ✅ 多步重路由 | ✅ 换 `w` 的生成 |
| P3 无 aux loss | ✅ 纯结构 | ✅ 纯结构 | ✅ 纯结构 |
| P4 竞争+全局分配 | ✅ 边缘约束 transport | ✅ 重分配 | ✅ 证据驱动 |
| P7 可退化 | ✅ `routing_mode=baseline` | ✅ `n_refine=0` | ✅ `mode` 不含 full 时关闭 |

### 4.5 三个可独立开关的配置

| 配置 | `ectir_mode` | Module 1 | Module 2 | Module 3 |
|---|---|---|---|---|
| 基线 | `baseline` | — （调用原 extractor，逐位复现） | — | — |
| **ECTIR-1** | `transport` | ✅ | ✗ | ✗ |
| **ECTIR-2** | `transport_refine` | ✅ | ✅ | ✗ |
| **ECTIR-Full** | `full` | ✅ | ✅ | ✅ |

`ectir_mode=baseline` 时**必须逐位复现 ASPCF**（单测断言）。

### 4.6 相对当前 ASPCF forward 的最小修改范围

| ASPCF forward 步骤 | 是否修改 |
|---|---|
| 1. item encoder（history + candidate） | ❌ 不动 |
| 2. 位置编码 + dropout | ❌ 不动 |
| **3. multi-interest extraction** | ✅ **替换**（Module 1/2） |
| 4. `interest_vectors` dropout | ❌ 不动 |
| **5. `InterestAggregator`** | ✅ **仅在 `full` 模式下替换**（Module 3） |
| 6. `user_vector = Σ w_k V_k` | ❌ 不动 |
| 7. `prediction = <u, e_cand>` | ❌ 不动 |
| 8. relation loss stash | ❌ 不动 |
| 9. NaN 检查 / 10. `return_intermediate` | ❌ 不动（新增键，不改既有键） |

`ItemEncoder`、`LLMMIRecASPCF` 本身**不改动**；新增独立模型文件。

### 4.7 复杂度与显存

| 项 | 计算量 | 说明 |
|---|---|---|
| affinity | `O(B·L·K·d)` = 1024×20×4×64 ≈ **5.2 MFLOP** | 与基线相同 |
| Sinkhorn | `O(n_sinkhorn · B·L·K)` = 5×1024×20×4 ≈ **0.4 M** | 可忽略 |
| Module 2 额外 | × n_refine 倍的 affinity + sinkhorn | ≤2× |
| Module 3 | `O(B·K·(2D+3)·H_e)` | 小 MLP |
| **`T` 显存** | `B·L·K` floats = 1024×20×4×4B = **328 KB** | 可忽略 |
| **总开销** | 约 **+15% ~ +25%** forward 时间（n_sinkhorn=5, n_refine=1） | 需实测确认 |

对比：ML-1M 单次训练 4279 s，Beauty 869 s。若开销 20%，Beauty 约 +3 分钟，可接受。

### 4.8 与现有 `QueryMultiInterestExtractor` 的复用点

| 复用 | 说明 |
|---|---|
| `self.query` `[K, D]` | 直接复用为 `q`；Module 2 在其上加 `W_r(V)` |
| `Wq` / `Wk` / `Wv` | **参数形状与语义完全复用**，不新增投影 |
| mask 约定 | `valid_mask = (arange(L) < lengths)`，与原实现一致 |
| `Wv(h)` 的 value 投影 | 复用，`V_k` 仍是 value 的凸组合 |
| `attention_prior` / `prior_strength` 钩子 | **不需要**（ECTIR 不用 logit 残差） |
| `return_route_scores` 管线 | 可扩展返回 `T` 供诊断 |
| 聚合侧 `LayerNorm(mean+last)` | Module 3 的 `h_sum` 直接复用 `InterestAggregator` 的构造 |

### 4.9 与 CDIR / EAIA 的区别

| 维度 | CDIR + EAIA（simple candidate） | **ECTIR（主候选）** |
|---|---|---|
| 竞争机制 | 逐位置在 K 上 `softmax_K`，作为 **logit 残差** | **全局最优传输**，同时约束行/列边缘 |
| interest 容量 | **隐式均匀**（`Σ_k R[k,l]=1` ⇒ 各兴趣总容量被拉平） | **显式非均匀**，由 affinity evidence `softmax(e/τ_c)` 产生 |
| 是否联合分配 | 否，仍是逐位置独立归一化 | **是**，任一元素变化影响整行整列 |
| 与 baseline 退化 | `λ_comp=0` **连续**退化 | **无连续退化**，靠 `routing_mode` 切换 |
| 聚合证据 | 仅 `o_k = mean_l R[k,l]` | mass + concentration + temporal centroid |
| 新增参数 | 0（CDIR） | `W_r` 4,160 + Module 3 MLP |
| 结构强度 | 弱（局部） | 强（全局） |

**两者不是互斥的**：CDIR 可作为 ECTIR 的消融对照（"局部竞争是否已足够"）。

---

## 5. Simple candidate（降级为 ablation）

### 5.1 CDIR — Residual Competitive Routing

```
R      = softmax_K(C / τ)
C_new  = C + λ_comp · log(K·R + ε)          # 数学上 log K 在 softmax_L 中被消去
A      = softmax_L(C_new)
V_k    = Σ_l A[k,l]·value_l
```
`λ_comp = 0` 严格退化为原 extractor。零新增参数。已记录于旧版 §8。

### 5.2 EAIA — Evidence-Aware Aggregation

`w = softmax_k(MLP([V_k ; o_k ; h_sum]))`，`o_k = mean_l R[k,l]`。

**用途**：作为 ECTIR 的**下界对照**。若 ECTIR 与 CDIR/EAIA 无差别，
说明"全局传输"相对"局部竞争"没有增量，应重新审视设计。

---

## 6. K=2 公平 baseline（非创新，仅对照）

| 项 | 结果 |
|---|---|
| Beauty K=2 vs K=4 | **实质持平**（差异全在 ±0.8%，1.2σ） |
| ML-1M K=2 vs K=4 | **明显更差**（HR@5 −3.13%，NDCG@5 −2.53%，−4.20σ / −2.57σ） |
| 结论 | K 是普通超参数，不是创新；**两数据集统一使用 K=4 baseline**；ML-1M 差距不能靠匹配 K=2 解决 |

详见 `CHAPTER4_DIAGNOSIS.md` §5。

---

## 7. 第一版工程约束

### 7.1 训练目标（不变）

```
LOSS = BPR + λ_relation · L_relation        # λ_relation = 0.01（Chapter 3 现有）
```

### 7.2 第一版明确不加入

- ❌ 任何新的 Chapter 4 auxiliary loss
- ❌ HSDIR loss、CAISD/TASID
- ❌ semantic routing teacher
- ❌ coverage / focus
- ❌ diversity regularizer
- ❌ orthogonality loss
- ❌ 额外的 quality gate

**不得用额外 loss 抢救结构。**

### 7.3 复杂度上限

| 项 | 上限 |
|---|---|
| Sinkhorn 迭代 | 3 ~ 5 |
| interest refinement | ≤ 2 |
| K | 4 |
| history_max | 20 |

### 7.4 默认超参（数值合理即可，首轮不扫）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ectir_eps` | 0.1 | entropic 正则（越小越接近硬分配） |
| `--ectir_tau_c` | 1.0 | 容量温度；→∞ 退化为均匀容量 |
| `--ectir_n_sinkhorn` | 5 | |
| `--ectir_n_refine` | 1 | Module 2 步数 |
| `--ectir_mode` | `full` | `baseline`/`transport`/`transport_refine`/`full` |

---

## 8. 风险与停止条件

| # | 风险 | 监测 | 缓解 / 停止 |
|---|---|---|---|
| **R1** | Sinkhorn 数值不稳（`f/g` 发散或 `T` 全零） | 单测：row/col mass 误差、NaN | log-domain + 有限 `NEG`；单测必须通过才开训 |
| **R2** | `b` 退化为 one-hot（容量塌缩到单个兴趣） | `b` 的熵 | 调大 `τ_c`；若熵 ≈ 0 且无法缓解 → 停止 |
| **R3** | transport 无增量（≈ CDIR） | ECTIR-1 vs CDIR | 若两者无差别 → 全局分配的假设不成立，回到 CDIR |
| **R4** | Module 2/3 无增量 | ECTIR-2 vs ECTIR-1；Full vs ECTIR-2 | 逐级判定，无增量的模块从核心移除 |
| **R5** | **数据集依赖**（已出现五次） | Beauty vs ML-1M | ML-1M 是硬性判据 |
| **R6** | 超参与 baseline 不匹配 | — | 严格复用 frozen ASPCF beauty 配置（K=4, lr=0.004, bs=1024, λ_rel=0.01） |
| **R7** | 训练变慢影响迭代 | 单 epoch 时间 | 实测；若 >1.5× 则减少 sinkhorn 迭代 |

---

## 9. 后续 TODO（仅记录，本阶段不执行）

### TODO-1：ML-1M 训练时间 profiling 与 fast-screening 配置

1. profile ML-1M 训练时间构成（数据加载 / 前向 / 反向 / 评测）；
2. 考虑建立 batch=2048 的 fast-screening 配置。

> ⚠️ **硬性约束**：fast-screening 配置必须有**它自己的 ASPCF baseline**，
> 不得与 batch=1024 的正式结果直接比较（batch=2048 属不同训练 regime；
> HSDIR 曾因 lr=0.008/bs=2048 与冻结基线不可比而无法归因）。
> 任何写进论文的绝对数字必须在 batch=1024 正式配置下重跑。

**当前状态**：❌ 不执行。

---

## 10. 本阶段不做的事

- ❌ 不改 ASPCF `ItemEncoder` / `LLMMIRecASPCF`
- ❌ 不新增 Chapter 4 auxiliary loss
- ❌ 不做大规模 transport 超参搜索
- ❌ 不跑 ML-1M / 3 seeds / Toys
- ❌ 不启动 Chapter 5

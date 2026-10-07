# CHAPTER4_DESIGN.md

> Chapter 4 方向设计
> 更新日期：2026-10-07（第四次修订）
> 前置约束：Chapter 3 ASPCF **结构**冻结（`CHAPTER3_FINAL.md`）。
> **注意：「冻结」指结构冻结，不是冻结梯度** —— 新模型仍按现有方式 end-to-end 训练，
> 并保留 Chapter 3 的 `lambda_relation = 0.01`。

---

## 0. 修订说明

| 内容 | 本版处置 |
|---|---|
| **PPCIM**（Prior–Posterior Candidate Interest Matching） | ✅ **主候选**（§11） |
| **ECTIR**（Evidence-Constrained Transport Interest Routing） | ❌ **已停止**（Round 1 失败，`CHAPTER4_ECTIR_ROUND1.md`），降为 negative evidence（§4） |
| CDIR + EAIA | 🔸 simple candidate / ablation（§5） |
| SCIR（用 `softmax_K` 替换 `softmax_L`） | ❌ superseded |
| Adaptive Interest Cardinality / dynamic-K | ❌ **不是创新**；K 属公平调参（§6） |

### 0.1 核心问题的两次修订

| 版本 | 核心问题 | 状态 |
|---|---|---|
| v1 | ~~如何让多个 interests 更分散~~ | ❌ 被 ECTIR Round 1 否证（结构改善 ≠ 排序改善） |
| **v2（当前）** | **如何让多个 interests 真正参与 candidate-specific ranking，而不是在打分前重新压缩成单一 user vector** | ✅ 见 §11.0，依据为 `CHAPTER4_DIAGNOSIS.md` §6.4 的事实 F6 |

**关键依据**：全部 7 个模型的打分路径逐字相同 ——
`u = Σ_k w_k V_k`（`w` 只由 history 决定）→ `score_j = uᵀ e_j`。
**K 个 interest 在打分前被压成一个与 candidate 无关的向量。**
这解释了为何 HSDIR / CAISD / ECTIR 都显著改变了 `V_k` 的结构却不改善排序。

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

---

## 11. 主候选（2026-10-07 修订）：PPCIM

> **ECTIR 已停止**（见 `CHAPTER4_ECTIR_ROUND1.md`）。CDIR/EAIA 仍为 simple candidate。
> PPCIM = **Prior–Posterior Candidate Interest Matching**，当前主候选。
>
> **本节只做设计与审计，不训练。**

### 11.0 核心问题（由 `CHAPTER4_DIAGNOSIS.md` §6.4 修订而来）

> ~~"如何让多个 interests 更分散"~~ ← 已被 ECTIR Round 1 否证
>
> ### **"如何让多个 interests 真正参与 candidate-specific ranking，
> ### 而不是在打分前重新压缩成单一 user vector"**

**依据（事实 F6）**：全部 7 个模型（ASPCF/HSDIR/CAISD/CASIR/CGSCD/CHIR/ECTIR）的打分路径
逐字相同：`u = Σ_k w_k V_k` → `score_j = uᵀ e_j`，`w` 只由 history 决定。
**K 个 interest 在打分前被压成一个 candidate-independent 向量。**

ECTIR 证明：只改 `V_k` 的结构（cosine 0.8823 → 0.7182，effR 1.103 → 1.437）
而打分路径不变，排序**不会改善**（−0.88%）。

### 11.1 约束

| # | 约束 |
|---|---|
| C1 | Chapter 3 ASPCF `ItemEncoder` 不改 |
| C2 | 原 `QueryMultiInterestExtractor` **第一版不改** |
| C3 | K = 4 |
| C4 | history-only aggregator **仅作为 interest prior**，不再是最终打分权重 |

### 11.2 Module 1 — history interest prior

```
p_k = P(z = k | H)                                  # [B, K]
```

**第一版直接复用现有 `InterestAggregator` 的输出**（`llmmi_components.py:590`）：

```
h_sum   = LayerNorm(mean_his + last_his)            # [B, D]
p       = softmax(MLP(h_sum))                       # [B, K]
```

`p` **只依赖 history**，不含任何 candidate / target 信息 → 无泄露。
它从"最终打分权重"降级为"兴趣先验"。

### 11.3 Module 2 — candidate–interest matching

**统一的 shape 约定（本文档仅使用这一套，不得混用）**：

| 符号 | shape | 含义 |
|---|---|---|
| `V` | `[B, K, D]` | `interest_vectors`，来自**未修改**的 `QueryMultiInterestExtractor` |
| `E` | `[B, C, D]` | `candidate_emb`，`C = 1 + N`，第 0 列恒为 positive |
| **`m`** | **`[B, K, C]`** | match logits（**不是 `[B,C,K]`**） |
| `p` | `[B, K]` | history prior |
| `pi` | `[B, K, C]` | posterior（与 `m` 同布局） |
| `z` | `[B, K, C]` | `log p + m/τ`，logsumexp 前的 logits |

```
sqrtD = sqrt(D)
m     = torch.bmm(V, E.transpose(1, 2)) / sqrtD     # [B, K, C]
```

- `D = emb_size = 64`，`K = 4`，训练 `C = 2`，dev/test `C = 1001`
- **第一版不引入任何额外参数**（无 bilinear projection）

> **为什么选 `[B,K,C]` 而不是 `[B,C,K]`**：
> `softmax`/`logsumexp` 沿 **兴趣维 K** 做（Module 3/4），
> 放在 `[B,K,C]` 布局下即 `dim=1`，避免转置；且 `bmm(V, Eᵀ)` 天然给出该布局，
> 无需额外的 `transpose/contiguous`。
> **代价**：`return_intermediate` 输出的 `m`/`pi` 是 `[B,K,C]`，诊断脚本需注意。
> 本文档早期版本混用过 `[B,C,K]`，**已统一修正**。

### 11.4 Module 3 — candidate-specific posterior

```
log_p = torch.log(p + eps).unsqueeze(-1)                    # [B, K, 1]
z     = log_p + m / tau                                     # [B, K, C]
pi    = softmax(z, dim=1)                                   # [B, K, C]
```

- **每个 candidate 拥有独立的 `pi[:, :, j]`**（区别于 F6 中所有 candidate 共用一个 `u`）。
- **positive 与所有 negative 使用完全相同的公式**，不允许按正负号分支 → 无 target leakage。
- 先验 `p_k` 以 log 形式加性进入 → 自然的先验-似然分解。
- `pi` **只在需要时（`return_intermediate` 或 `posterior_mean` 模式）才显式保留**；
  主 `marginal` 模式可直接从 `z` 做 `logsumexp`，**无需长期保存 `pi`**。

### 11.5 Module 4 — latent-interest marginal scoring

**主 scoring（PPCIM Round 1 采用）**：

```
score_marginal = sqrtD * tau * torch.logsumexp(z, dim=1)     # [B, C]
```

> #### ⚠️ `sqrtD` 因子是必需的（不是可选的缩放）
>
> `m = <V_k, e_j> / √D`，因此
> ```
> τ · log Σ_k p_k exp(m_k/τ)  --(τ→∞)-->  Σ_k p_k m_k = (1/√D) · Σ_k p_k <V_k, e_j>
> ```
> 而 ASPCF 的打分是 `score_ASPCF = uᵀe_j = Σ_k p_k <V_k, e_j>`。
>
> **因此必须乘回 `√D`**：
> ```
> √D · (τ · logsumexp)  --(τ→∞)-->  Σ_k p_k <V_k, e_j>  =  score_ASPCF   ✅
> ```
> 若不乘 `√D`，虽然**排序**在 `τ→∞` 时仍等价（全局正标量不影响 argsort），
> 但 **BPR 的分数尺度会被缩小约 `√D = 8` 倍**，
> 而 BPR 的 `sigmoid(pos − neg)` 对尺度敏感 ⇒ 梯度行为改变 ⇒
> **无法做到"只改 scoring operator、其余完全公平"**。
>
> 这一条必须写进单元测试（测试 J）。

**同时实现但仅作为 ablation**：

```
score_posterior_mean = sqrtD * Σ_k pi[:, k, :] * m[:, k, :]      # [B, C]
```

**不需要显式构造 `u_j`**（由关系 1 直接算，省一个 `[B,C,D]` 张量）。

#### 两种 scoring 的数学关系（必须写清）

**关系 1 —— 精确恒等式**：由 `m[j,k] = <V_k, e_j>/√D` 得

```
score_posterior_mean_j
  = Σ_k pi[j,k] <V_k, e_j>
  = sqrt(D) * Σ_k pi[j,k] * m[j,k]          ← 不需要构造 u_j
```

即它是 **posterior 加权平均的匹配分**，与 `u_j = Σ_k π[j,k]V_k` 的 `<u_j, e_j>` **精确相等**。

**关系 2 —— 泛函形式**：

| | `score_marginal`（主） | `score_posterior_mean`（ablation） |
|---|---|---|
| 形式 | `√D·τ·log Σ_k p_k exp(m/τ)` | `√D · Σ_k π_k m` |
| 加权用 | **先验 `p_k`** | **后验 `π_k`** |
| 算子 | log-sum-exp（**soft-max**） | 加权平均 |
| 数学身份 | **log-partition（潜在变量边际似然）** | **后验期望匹配分** |
| `τ→0` | → `√D·max_k m[j,k]`（硬最大） | 不变（无 τ） |
| `τ→∞` | → `√D·Σ_k p_k m[j,k]` = **ASPCF** | 不变 |
| 对 `m` 的梯度 | `∂score/∂m[:,k,j] = √D·π[:,k,j]`（**不穿过 softmax Jacobian**） | `√D·π + √D·Σ m·∂π/∂m`（**多一项**） |

**关键差别**：`score_marginal` 的梯度**恰好是 `√D·π`**，不包含对归一化的反传；
`score_posterior_mean` 则要穿过 `softmax` 的 Jacobian。

**τ 的作用**：`score_marginal` 在 "最佳匹配兴趣"（τ→0）与 "先验平均 = ASPCF"（τ→∞）之间连续插值。
默认 `τ = 1.0`。

**注意**：`score_posterior_mean` **即使 `p` 均匀且 `τ→∞` 也不等于 ASPCF**
（它用**后验** `π`，ASPCF 用**先验** `p`）；二者仅在 `m` 对所有 k 相等时一致。
消融解读时必须注意这一点。

### 11.6 新增 Chapter 4 loss（**只写入文档，第一轮不启用**）

**PPCIM-1（第一轮）**：只改前向 scoring。
```
LOSS = L_BPR + lambda_relation * L_relation        # 与 Chapter 3 完全相同
```

**PPCIM-2（仅当 PPCIM-1 有希望时才考虑）**：加入与 latent-interest 结构匹配的 pairwise objective。

对 `(positive, negative)` 对，定义逐兴趣的边际差：

```
delta_k = m_pos,k - m_neg,k                          # [B, K]
L_LI    = -log( Σ_k p_k * sigmoid( delta_k / tau_r ) )
L       = L_BPR + lambda_relation * L_relation + lambda_LI * L_LI
```

**含义**：`L_LI` 是"**至少有一个（按先验加权的）latent interest 能把正样本与负样本区分开**"的
软最大化目标。`sigmoid(delta_k/τ_r)` ∈ (0,1) 是兴趣 k 的判别成功率，
`Σ_k p_k ·` 是按先验的期望，取 `-log` 即最大化它。

**约束**：`lambda_LI` 默认 **0**，第一轮**不打开**。理由：先验证结构本身。

### 11.7 代码审计

#### (a) Tensor shape（`C` = candidate 数，`K` = 4，`D` = 64）

| 量 | shape | 备注 |
|---|---|---|
| `V` | `[B, K, D]` | 来自未修改的 `QueryMultiInterestExtractor` |
| `E` | `[B, C, D]` | 现有 `candidate_emb`，已存在 |
| `p` | `[B, K]` | 来自未修改的 `InterestAggregator` |
| `z` / `m` | **`[B, K, C]`** | `bmm(V, Eᵀ)/√D`；`softmax`/`logsumexp` 走 `dim=1` |
| `π` | **`[B, K, C]`** | 与 `m` 同布局 |
| `score` | `[B, C]` | 替换现有 `prediction` |

#### (b) `C` 的真实取值（**实测**）

| phase | C | 来源 |
|---|---|---|
| train | **2** | 无 `neg_items` 列；`num_neg=1` → 1 正 + 1 采样负 |
| dev / test | **1001** | `neg_items` 列首行长度 **1000** → 1 正 + 1000 负 |

（注：是 **1000** 个负样本，不是 999。）

#### (c) 显存与计算量（`B = eval_batch_size = 256`，`C = 1001`）

| 项 | 大小 | 说明 |
|---|---|---|
| `e` | 256×1001×64×4B = **65.5 MB** | **已存在**，非新增 |
| `m` | 256×4×1001×4B = **4.1 MB** | 新增 |
| `π` | 256×1001×4×4B = **4.1 MB** | 仅 ablation |
| `score` | 256×1001×4B = **1.0 MB** | 替换原 `prediction` |
| **新增合计** | **< 10 MB** | 相对现有 65.5 MB 可忽略 |
| `m` 的计算 | `B·K·C·D` = **65.6 MFLOP** / batch | 可忽略 |
| **不构造 `u_j`** | 省 **65.5 MB** | 由关系 1 直接从 `m, π` 算 `score_mean_j` |

**结论：999/1000 负样本评测不构成显存或算力障碍。**

#### (d) 全向量化

```
m      = torch.bmm(V, e.transpose(1, 2)) / math.sqrt(D)          # [B, K, C]
logits = torch.log(p + eps).unsqueeze(-1) + m / tau              # [B, K, C]
score  = tau * torch.logsumexp(logits, dim=1)                     # [B, C]
```
全部为 `bmm` + 逐元素 + `logsumexp`，**无 Python 循环、无 `[B,C,D]` 中间张量**。
ablation：`score_mean = math.sqrt(D) * (pi * m.transpose(1,2)).sum(-1)`（同样是 `bmm` 级别）。

#### (e) train / dev / test 打分一致性

**`score_j` 是逐 candidate 独立的（pointwise）**：
它只依赖 `p`（history）与 `m[:, :, j]`（candidate j 自身），
**不含任何跨 candidate 的归一化**（如 `softmax over C`）。

⇒ train（C=2）与 dev/test（C=1001）**使用完全相同的公式**，
`C` 变化不影响任何单个 `score_j`。**这是本设计的关键正确性属性。**
（若采用"对 candidate 做 softmax"那类写法则会破坏这一点。）

#### (f) 与 TiMiRec / MIND greedy matching 的区别

> ### ⚠️ 未核验 —— 不得作为已确认的 novelty claim
>
> 以下仅为**结构性区分**，基于对这两类方法的**通行描述**，**本轮未回原文核对**。
> **正式写论文前必须回原文核验确切公式**。
> 在核验之前，**禁止**把"与 MIND/TiMiRec 不同"写成已成立的 novelty 声明。

| 维度 | MIND 式 greedy / max | TiMiRec 式 target-interest matching | **PPCIM** |
|---|---|---|---|
| candidate 是否参与 | 是 | 是 | 是 |
| 兴趣选择方式 | **硬 argmax / 取最大**（单兴趣） | 以 target 为 query 做注意力 | **温度控制的 soft-max**（`logsumexp`） |
| 是否使用 history 先验 | 通常不使用 | — | **显式 `p_k` 加性进入 logit** |
| `τ` 可调 | 无 | — | **有**（`τ→0` 退化为硬最大） |
| 是否可退化到 baseline | 否 | 否 | **是**：见 §11.7(g) |
| 与 loss 的关系 | — | — | 后验 `π` 可直接用于 `L_LI`（§11.6） |

**设计层面的实质差别**：PPCIM 把"选哪个兴趣"表述为一个**带先验的潜在变量边际化**
（log-partition），而不是一次性的 argmax；因此它天然保留了"当没有任何兴趣匹配时，
退回先验"的行为，并且 `τ` 提供一个从硬选择到软平均的连续谱。

#### (g) 与 baseline 的退化关系（重要）

`PPCIM` **不是**严格嵌套 baseline 的。原因：baseline 的
`score_j = <Σ_k w_k V_k, e_j> = Σ_k w_k <V_k, e_j> = √D Σ_k p_k m[j,k]`
是**先验 `p` 的加权平均**（一次平均），而 PPCIM 主 scoring 是 `log-partition`（soft-max）。

但存在一条**精确对应**：

```
tau → infinity 时,  score_j → Σ_k p_k m[j,k] = score_baseline_j / sqrt(D)
```

即 **PPCIM 在 `τ→∞` 时逐点等于 baseline 打分的 `1/√D` 倍**。
由于 BPR 只依赖分数之差、且 `1/√D` 是全局正常数（对所有 candidate 相同），
**`τ→∞` 时 PPCIM 的排序与 baseline 完全一致**。

⇒ 这提供了一个**可验证的退化路径**（虽然不是严格恒等，但排序等价），
应写入单元测试：`τ = 1e6` 时 `argsort(score_PPCIM) == argsort(score_baseline)`。

同时说明：**PPCIM 的 ablation `score_mean_j` 在 `p` 为均匀且 `τ→∞` 时也不等于 baseline**
（`score_mean_j` 用的是**后验** `π`，而 baseline 用**先验** `p`）；
两者只有在 `m` 对所有 k 相等时才一致。这一点必须在消融解读中注意。

#### (h) 与现有 TASID 的关系（关键区别）

**代码核实**（`models/sequential/LLMMIRecCAISD.py`）：

| | TASID | **PPCIM** |
|---|---|---|
| target/candidate 信息进入的位置 | `distill_info["_tasid_loss"]`（`:359`）→ `out_dict`（`:431`）→ `total += λ·L` (`:484-486`) | **`prediction` 本身** |
| 是否改变 `prediction` | ❌ **否**。`prediction = <Σ_k w_k V_k, e_cand>` 完全未变 | ✅ **是**。`score_j` 每个 candidate 独立计算 |
| 测试期是否生效 | ❌ 否（`if self.training` 门控） | ✅ 是 |
| 本质 | 训练期正则 | **前向打分算子** |

> **TASID 的 target teacher 以前只进入 loss；
> PPCIM 的 candidate information 必须进入真实 prediction path。**
> 这是两者最根本的差别，也是 PPCIM 满足 P1（改前向算子）而 TASID 不满足的地方。

#### (i) Candidate leakage 检查

| 风险 | 检查结论 |
|---|---|
| 先验 `p_k` 是否用到 candidate | ❌ 否，只由 history 计算（`InterestAggregator`） |
| positive 与 negative 是否走不同分支 | ❌ 否，**同一个公式** `pi[j,k]`、`score_j` |
| 是否依赖 candidate 集合 | ❌ 否，`score_j` **逐点独立**（§11.7(e)） |
| dev/test 是否泄露 target | ❌ 否。target 只作为 candidate 之一参与打分，与所有 negative 同权 |
| `test_all` 模式下是否仍成立 | ✅ 成立（pointwise，`C = n_items` 亦可） |
| 与 baseline 相比是否引入新的信息 | 无新信息源；**只是改变了 candidate 如何与 K 个 interest 交互** |

### 11.8 相对现有 ASPCF forward 的最小代码 diff

只改 forward step 5–7 的 **2 行**：

```python
# ---- ASPCF 现状（LLMMIRecASPCF.py:199-201）----
interest_weights = self.aggregator(history_emb_raw, lengths)          # [B, K]
user_vector = (interest_vectors * interest_weights[:, :, None]).sum(dim=1)   # [B, D]
prediction = (user_vector[:, None, :] * candidate_emb).sum(dim=-1)     # [B, C]

# ---- PPCIM（marginal）----
p   = self.aggregator(history_emb_raw, lengths)                        # [B, K]   先验
m   = torch.bmm(interest_vectors, candidate_emb.transpose(1, 2)) / math.sqrt(self.emb_size)
z   = torch.log(p + 1e-8).unsqueeze(-1) + m / self.ppcim_tau           # [B, K, C]
prediction = math.sqrt(self.emb_size) * self.ppcim_tau * torch.logsumexp(z, dim=1)   # [B, C]
```

**其余全部不变**：item encoder、位置编码、`extractor`、relation loss、`return_intermediate` 契约。

### 11.8.1 Round 1 硬约束（不可越界）

> **PPCIM Round 1 只能改变 scoring path。** 以下全部保持 ASPCF 原样：

| 保持不动 | 说明 |
|---|---|
| `ItemEncoder`（`mode="aspcf"`） | 一行不改 |
| position encoding | 不动 |
| `QueryMultiInterestExtractor` | **结构与代码均不改** |
| `InterestAggregator` | **不改**，仅把其输出从"最终权重"改称"先验" |
| relation loss | 不动，`lambda_relation = 0.01` |
| BPR | 不动（含其 `neg_softmax` 形式） |
| `K = 4` | 固定 |
| `dropout` | 0.1，固定 |
| 所有 dataset-specific 超参 | 逐项继承 frozen Beauty ASPCF |

**新增可训练参数 = 0**（v1 无任何 projection）。

### 11.8.2 关于 BPR 的一处澄清（避免混淆）

`GeneralModel.loss`（`models/BaseModel.py:176`）对**负样本**做了 softmax：

```python
pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
```

这是 **loss 层**对候选的聚合，**不是 scoring 层的耦合**。
PPCIM 的 `score_j` 仍是**逐候选独立**的，因此：

- 训练（`C=2`）与评测（`C=1001`）的**打分函数完全一致**；
- 评测用 `argsort` 对候选排序，属于排序本身，与 scoring 的逐点性不冲突。

**结论：无 candidate-set coupling 问题。**

### 11.9 第一轮最小实验矩阵（Beauty, seed 42）

**不重跑 ASPCF baseline**（0.1592 / 0.1088）。`LOSS = BPR + relation`，`λ_LI = 0`。

| # | 配置 | 隔离变量 | 关键 flag |
|---|---|---|---|
| 0 | ASPCF（已有） | — | — |
| **1** | **PPCIM-1**：主 scoring（logsumexp） | **candidate-specific 打分本身** | `--ppcim_score marginal --ppcim_tau 1.0` |
| **2** | ablation：`score_mean_j`（posterior 均值） | 两种 scoring 的差异 | `--ppcim_score posterior_mean` |
| **3** | ablation：先验置均匀（`p = 1/K`） | history 先验是否有用 | `--ppcim_uniform_prior 1` |

判读：
- 配置 1 若 ≤ 0 → candidate-specific 打分本身无效 → **停止**，重新审视 F6
- 配置 1 > 0 且 配置 2 ≈ 配置 1 → 两种 scoring 等价，取更简单的
- 配置 1 > 配置 3 → **history 先验确实贡献**，否则先验可去掉

**通过后才考虑**：ML-1M 同配置同 seed；之后才考虑 `λ_LI > 0`。

### 11.10 新增参数（预计）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ppcim_score` | `marginal` | `marginal`（主）/ `posterior_mean`（ablation） |
| `--ppcim_tau` | 1.0 | 温度；`τ→∞` 排序等价 baseline |
| `--ppcim_uniform_prior` | 0 | 1 = 先验置均匀（消融） |
| `--ppcim_eps` | 1e-8 | `log(p+eps)` 的数值保护 |

### 11.11 第一版不做

- ❌ 不改 `QueryMultiInterestExtractor`
- ❌ 不改 ASPCF `ItemEncoder`
- ❌ 不加 bilinear projection（Module 2 保持无参数）
- ❌ 不打开 `L_LI`
- ❌ 不做大规模 `τ` 扫描

# THESIS_MASTER_PLAN.md

> THESIS PHASE 0 — Research Freeze & Engineering Decomposition
> 建立日期：2026-10-03
> 基线 commit：`bf008e2`（main 分支，工作区干净）
>
> **本文件性质：技术主线冻结文档。只记录设计，不含任何实现。**
> 本阶段不编码、不训练、不修改 evaluation protocol、不修改数据 split。

---

## 0. 边界声明

本文件固定以下内容，后续阶段不得在未显式修订本文件的前提下偏离：

1. 论文的三个递进研究工作（Chapter 3 / 4 / 5）的问题定义与核心模块构成。
2. 三章之间的继承关系（数据流与监督信号流向）。
3. 每章依赖的预计算资产及其生成方式。
4. 每章已识别的主要技术风险。

本文件**不**固定：具体超参数取值、损失函数的具体形式细节、消融实验的最终清单。这些在实现阶段确定。

评估方式：**完全沿用现有 PoMRec-compatible protocol**（`helpers/BaseRunner.py`，HR@K / NDCG@K，early stop on NDCG@5）。Phase 0 不讨论、不修改、不优化 evaluation。

---

## 0.5 Architecture-First 修订说明

> 修订日期：2026-10-03（THESIS PHASE 1 起始）
> 修订原因：Phase 0 的工作量评估（`THESIS_WORKLOAD_REVIEW.md` §问题 2）暴露出一个结构性倾向——三章的规划都在"增加 auxiliary loss"上堆叠模块。这与实际目标不符。

### 修订内容

**优先级反转：核心结构必须先证明有效，辅助 loss 后加。**

三个章节最终希望分别真正改变的是**推荐链路中的三个不同环节**，而不是三个监督信号：

| 章节 | 真正要改变的东西 | 在链路中的位置 |
|---|---|---|
| **Chapter 3** | **item representation** | `item_id → e_item`，被 history 与 candidate 共享 |
| **Chapter 4** | **history → multi-interest routing** | `history items → K interests`，即**信息如何被分配到兴趣** |
| **Chapter 5** | **candidate → interest → prediction** | `candidate × interests → score`，即**打分路径本身** |

三章由此分别对应推荐链路的**输入表示层、路由层、决策层**，且每章只改一层。

### 对 Chapter 3 的直接约束

Chapter 3 的唯一核心主张必须是：**改变 item representation 的结构本身就有收益**。

因此：

- ✅ **保留**：shared projection、shared branch、private residual、complementary branch、adaptive gate、fused embedding。
- ✅ **允许**：现有 ASPCF relation loss 作为一个**可开关的已有基础项**（`--lambda_relation 0` vs `0.01`），用于判断"收益来自新 encoder 还是旧辅助监督"。
- ❌ **本轮不新增**：shared relation loss、private relation loss、redundancy loss、orthogonality loss、contrastive loss。
- ❌ **不以增加 loss 数量的方式制造工作量。**

`L_relation` 在 `THESIS_ENGINEERING_MAP.md` 中原本被规划为"重写为双教师形式"，**该规划在此修订下推迟**：先验证 encoder，再决定是否需要更精细的关系约束。

### 对 Chapter 4 的直接约束（规划层面）

Chapter 4 的核心主张改为：**语义信息必须进入 `history → interest` 的 routing forward 路径**，而不是只通过 profile/coverage 这类训练期 loss 影响兴趣。

- 这否定了"用 KL 蒸馏一个 teacher profile 来间接塑造兴趣"的路线（CAISD 的做法）。
- Phase 0 的证据支持这一转向：HSDIR 大幅改变了兴趣结构（cosine 0.9457→0.8691）却未转化为收益，说明**训练期结构监督**与**前向路由行为**之间存在缺口。

### 对 Chapter 5 的直接约束（规划层面）

Chapter 5 的核心主张改为：**每个 candidate 根据其与 K 个 interests 的协同/语义匹配，产生自己的 interest distribution 和 candidate-specific user representation，直接改变 prediction path**。

- 这否定了"target-conditioned 只用于训练期 KL"的路线（TASID 的做法）。
- Phase 0 的证据支持这一转向：现有 TASID 的 `prediction` 路径完全不受 target 影响，target 知识在打分时已被丢弃。

### 对辅助机制的重新定位

CAISD / TASID / confidence calibration / semantic hard-negative 等既有机制**全部保留为后续可选辅助项**，不再默认作为任何章节的核心创新点。它们的作用改为：

1. 作为 baseline 与对照；
2. 作为核心结构验证通过之后，用于进一步提升的候选手段；
3. 作为论文中"我们尝试过但收益不稳定"的诚实记录。

**判定顺序**：核心结构 → 收益 → 才考虑辅助 loss。

---

## 1. 论文总体结构

三个研究工作沿推荐系统的三个层级逐级递进：

```
┌─────────────────────────────────────────────────────────────────┐
│ Chapter 3 — Item Representation                                  │
│ Collaborative-Guided Shared–Complementary Semantic Decomposition │
│ 问题：LLM item 语义中，哪些部分与协同信号共享，哪些是私有的？      │
│ 产物：item 表示 e = fuse(shared_semantic, complementary_semantic)│
└─────────────────────────────────────────────────────────────────┘
                              │ 提供语义空间与 item 表示
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Chapter 4 — Interest Representation                              │
│ Prototype-Grounded Interest Semantic Profile + Coverage          │
│ 问题：如何让 K 个 latent interest 具有明确且完整的语义组织，       │
│       而不是仅仅彼此分离？                                       │
│ 产物：每个 interest 的语义 profile P_k，且 Σ_k w_k P_k 覆盖 q_seq │
└─────────────────────────────────────────────────────────────────┘
                              │ 提供语义清晰的多个 interests
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Chapter 5 — Interest Decision                                    │
│ Target-Conditioned Interest Selection                            │
│ 问题：给定 target item，如何选择真正相关的 interest，              │
│       并让这种 target-interest 知识真正改善 ranking？             │
│ 产物：target-conditioned interest distribution + discriminative  │
│       ranking objective                                          │
└─────────────────────────────────────────────────────────────────┘
```

**递进性的本质**（三章不是三个拼接模型）：

| | 上一章的输出 | 下一章如何消费 |
|---|---|---|
| Ch3 → Ch4 | shared/complementary 语义子空间 + 分解后的 item 表示 | prototype 建立在 Ch3 语义表示上（不再依赖 PCA 人工切片）；interest profile 的语义坐标由该空间定义 |
| Ch4 → Ch5 | 每个 interest 的语义 profile P_k 与历史覆盖关系 | target 语义与 P_k 构造 target-conditioned teacher 分布；coverage 提供 w_k 的先验 |
| Ch5 反馈 | target-discriminative ranking 的压力 | 通过 interest 表示回传，间接要求 Ch4 的 P_k 具有目标可判别性 |

---

## 2. Chapter 3 — Item Representation

### 2.1 旧基础与其局限

**旧基础：LLMMIRec + ASPCF**

现有 ASPCF（`models/sequential/llmmi_components.py::ItemEncoder(mode="aspcf")`）：

```
z = llm_table[item]                     # 1536-dim PCA
z_high = z[:, :512]                     # 高方差主子空间 → "semantic"
z_low  = z[:, 512:]                     # 尾部分量         → 喂给 complement
s = semantic_branch(z_high)             # 32-dim
c = complement_mlp([complement_id_emb, complement_tail(z_low)])  # 32-dim
e = concat[sqrt(α_s)·s, sqrt(α_c)·c]    # 64-dim, α = softmax(gate([s;c]))
L_relation = KL(item-item 余弦关系 teacher=z_high ‖ student=s)
```

**局限**：semantic / complement 的划分完全由 **PCA 方差顺序**人工确定（"前 512 维是语义，其余是互补"）。PCA 方差与"是否被协同信号解释"没有因果关系。

**Phase 0 实测证据**（见 §6 与 `THESIS_WORKING_NOTES` 中的探针结果）：

| 量 | Beauty | ML-1M |
|---|---|---|
| 前 32 维 PCA 解释 LLM 方差占比 | 28.1% | 48.7% |
| 前 512 维 PCA 解释 LLM 方差占比 | 78.4% | 84.5% |
| 用 CF 视图线性回归预测 LLM（全体 1536 维）R² | **0.041** | **0.017** |
| 用 CF 视图预测 `z[:, :512]`（ASPCF 所谓 semantic 切片）R² | 0.050 | 0.017 |
| 用 CF 视图预测 `z[:, 512:1024]` R² | 0.005 | 0.015 |
| 用 CF 视图预测 `z[:, 1024:]` R² | 0.004 | 0.014 |

结论：**高方差 PCA 方向并不是与协同信号关联最强的方向**。前 512 维携带了 78% 的方差，却只有 5% 可被 CF 解释。这直接否证了"按 PCA variance 人工定义 semantic/complement"的合理性，构成本章动机。

### 2.2 新研究方向

**Collaborative-Guided Shared–Complementary Semantic Decomposition (CGSCD)**

记号：
- `Z = frozen LLM item embedding`  `[N, d_llm]`（d_llm = 1536）
- `C = collaborative item embedding` `[N, d_cf]`（d_cf = 64），**只从推荐交互学习**
- `Z_shared`：LLM 中与协同空间强关联的方向
- `Z_private`：LLM 中不能被协同空间解释的信息

方法构成（四个模块）：

1. **Cross-view decomposition**：由 `Z` 与 `C` 求共享子空间基 `U_r ∈ R^{d_llm × r}`，
   `Z_shared = (Z - μ_z) U_r`，`Z_private = (Z - μ_z) - Z_shared U_rᵀ`。
2. **Shared branch**：`s = f_shared(Z_shared)`，编码"LLM 与协同一致认可的语义"。
3. **Complementary branch**：`c = f_compl([Z_private, C])`，编码"LLM 独有语义 + 协同锚点"。
4. **Adaptive fusion**：沿用 ASPCF 的 gate 机制 `e = concat[√α_s·s, √α_c·c]`。

**语义关系/冗余约束**（替代并推广现有 `L_relation`）：
- shared 侧：student 的 `s` 应保持 `Z_shared` 中的 item-item 关系；
- private 侧：student 的 `c` 应保持 `Z_private` 中的 item-item 关系；
- **跨侧冗余惩罚**：`s` 与 `c` 之间不应重复编码同一信息（互信息/相关性上界约束）。

### 2.3 技术方案（不编码）

#### 2.3.1 Collaborative embedding 选择

现有三个候选（已完成溯源，见 `THESIS_ENGINEERING_MAP.md` §Ch3-E）：

| 候选 | 文件 | 来源 | 形状 | 是否只来自推荐交互 |
|---|---|---|---|---|
| **A. PoMRec pretrained** | `data/<ds>/handled/itm_emb_pomrec.pkl` | PoMRec checkpoint 的 `interest_extractor.i_embeddings.weight[1:]` | (N, 64) | ✅ 是 |
| B. LLMMIRec `complement_id_emb` | 无独立导出文件 | ASPCF 内部子模块 | (n_items, 64) | ⚠️ 与 LLM 语义分支联合训练，非纯协同 |
| C. 交互矩阵 SVD | 无现成文件 | `train.csv` 的 user-item 矩阵截断 SVD | (N, k) | ✅ 是 |

**推荐 A（PoMRec pretrained）**，理由：

- **独立性**：PoMRec 是独立训练的模型，其 item embedding 不含任何 LLM 语义信号，与 `Z` 在训练过程上正交。候选 B 的 `complement_id_emb` 与 `semantic_branch` 共享 gate 与梯度，用它构造"协同视图"会造成循环论证。
- **工程复杂度**：A 已有现成文件，三个数据集全覆盖，且已 100% 验证与 checkpoint 一致（maxdiff = 0.0）。B 需要从 LLMMIRec checkpoint 中导出，且导出的是一个"半协同"张量，还需额外论证。C 需要重新实现 builder 并决定归一化/加权方案。
- **与主线一致性**：A 已经是历史脚本中 `--srs_emb_path` 的实际指向（所有 `--srs_emb_path` 均指向 `itm_emb_pomrec.pkl`；`srs_emb.pkl` 从未存在）。复用它保持与前期实验的可追溯性。

**但必须记录 A 的已知缺陷**，并在实验中给出候选 C 的对照：
- A 来自 PoMRec（`lamb=4.0`，K=4，`emb_size=64`），其几何结构是为 PoMRec 自身的多兴趣打分服务的，不是通用协同表示。
- **数据集依赖性**（实测）：用 C 视图预测 LLM 的 R² 在 Beauty 上 PoMRec(4.1%) > SVD128(1.6%)，在 ML-1M 上 PoMRec(1.7%) < SVD128(24.8%)。即"哪个协同视图更好"本身是数据集相关的，这是 Chapter 3 的一个真实变量，应当作为分析而非隐藏。

#### 2.3.2 分解的数学形式

```
# 离线，一次性
Z̄ = Z[1:]                      # 去掉 padding row 0，[N, d_llm]
C̄ = C                          # [N, d_cf]
μ_z = mean(Z̄, axis=0)          # [d_llm]
μ_c = mean(C̄, axis=0)          # [d_cf]
Zc = Z̄ - μ_z ;  Cc = C̄ - μ_c
M  = Zcᵀ Cc / N                 # [d_llm, d_cf]
[U, Σ, Vᵀ] = svd(M, full_matrices=False)
U_r = U[:, :r]                  # [d_llm, r]  共享方向基
```

模型内：
```
z_c  = llm_table[ids] - μ_z          # [*, d_llm]
z_sh = z_c @ U_r                     # [*, r]
z_pv = z_c - z_sh @ U_rᵀ             # [*, d_llm]
s = f_shared(z_sh)                   # [*, s_dim]
c = f_compl([z_pv, c_table[ids]])    # [*, c_dim]
```

**注意**：`M` 的秩 ≤ min(d_llm, d_cf) = 64。因此 `r` 的合法上界是 64，这与 `d_cf` 直接耦合——若将来换用更高维的协同视图，r 的上界随之提高。

#### 2.3.3 逐项回答工程问题

**(a) 是否必须显式构造 d_llm × d_llm projection？**

**不需要，且不应构造。**
- 需要的是把 `z_c` 投到 r 维共享子空间：`z_sh = z_c @ U_r`。这只需 `[d_llm, r]` 一个矩阵。
- 裁剪投影 `P = U_r U_rᵀ` 作用到 private 分支时写作 `z_c - (z_c @ U_r) @ U_rᵀ`，全程只出现 `U_r`。
- 显式的 `[1536, 1536]` 矩阵（2.36M 元素，9.4 MB float32）在数学上等价但完全冗余，且会掩盖 rank 这个超参数。

**(b) 如何避免大矩阵浪费？**

- 存储：只保存 `U_r`（`r=32` → 1536×32 = 49K float32 = 196 KB/数据集），不保存 `P`。
- 计算：每个 item 的分解成本 `O(d_llm · r)` = 1536×32×2 ≈ 98K flops。batch 内 item 数上界 `B × (L + 1 + N_neg) = 1024 × 22 ≈ 22.5K`，总计 ≈ 2.2 GFLOP 前向（含反向约 3 倍）。相对 ASPCF 现有的 `Linear(1536→128)`（每 item 393K flops）**更便宜**。
- 缓存：`Z_shared` 与 `Z_private` 对全表是固定的，可离线预计算并作为 buffer 存入（`[N, r]` + `[N, d_llm]` = 12K×(1536+32)×4B ≈ 75 MB / 数据集，可接受）；也可在模型内实时算。**建议实时算**，因为这样 `U_r` 可切换为可学习参数做消融。

**(c) 推荐 rank r？**

来自 Phase 0 实测的 `M` 奇异值能量累积（`Σσ²` 归一化）：

| r | Beauty 累积能量 | ML-1M 累积能量 |
|---|---|---|
| 8 | 0.462 | 0.159 |
| 16 | 0.814 | 0.465 |
| 32 | 0.909 | 0.626 |
| 64 | 1.000 | 1.000 |

**推荐默认 `r = 32`**，扫描范围 `{8, 16, 32, 64}`。
理由：r=32 在 Beauty 上已捕获 91% 的跨视图关联能量，在 ML-1M 上 63%（该数据集能量谱更平坦，本身是一个可写入论文的观察）；同时 r 远小于 d_llm，private 分支仍保留 1504 维，"互补"在维度上占绝对主导。

**(d) 离线预计算还是 end-to-end？**

**推荐混合：basis 离线冻结，占用方式 end-to-end。**

- **离线**：`svd(M)` 一次算完，保存 `U_r`、`μ_z`、`μ_c`、`σ`。理由：
  1. 可复现——分解结果不随随机种子漂移；
  2. 分解本身可以作为一个独立的分析图（奇异值谱、共享/私有方差占比），不依赖训练；
  3. 避免 end-to-end 训练时 `U_r` 与 `C` 同时漂移导致的退化（例如 `U_r` 塌缩到平凡方向使 shared 分支恒为常数）。
- **end-to-end**：`f_shared`、`f_compl`、gate 全部正常训练。冻结的是"哪些方向算共享"这个定义，学习的是"如何使用它们"。
- **必须提供的消融**：`--cgscd_basis_mode {frozen, learnable}`，`learnable` 时把 `U_r` 设为 `nn.Parameter` 并由 SVD 初始化。这是回答"分解是否必须来自 SVD"的关键对照。

**(e) 保存什么文件？**

`data/<dataset>/handled/cgscd_basis_r<r>.pkl`：

```
{
  'U_r':             float32 [d_llm, r],
  'singular_values': float32 [r],
  'z_mean':          float32 [d_llm],
  'c_mean':          float32 [d_cf],
  'r': int, 'd_llm': int, 'd_cf': int,
  'source': {'z_path': str, 'z_sha1': str, 'c_path': str, 'c_sha1': str},
  'diagnostics': {'sigma_energy_cum': float32 [r],
                  'z_shared_var_share': float,
                  'predictability_r2': float}
}
```

`source` 中的 sha1 用于防止 basis 与 embedding 表版本错配——这是一个已知的真实风险（`itm_emb_pomrec.pkl` 在 toys 上就存在多个同名/近似 checkpoint，溯源时曾出现 2.5 的 maxdiff 不匹配）。

**(f) ItemEncoder 最小改动位置？**

`models/sequential/llmmi_components.py::ItemEncoder`：

| 位置 | 改动 |
|---|---|
| `mode` 校验列表 | 增加 `"cgscd"` |
| `__init__` 签名 | 增加 `shared_basis`, `z_mean`, `c_mean`, `cf_table`, `shared_hidden`, `shared_dim`, `compl_hidden`, `compl_dim`, `cgscd_use_cf_input` |
| `__init__` 主体 | 注册 `shared_basis`/`z_mean`/`c_mean`/`cf_table` 为 **non-persistent buffer**；新建 `shared_branch`、`compl_mlp`；gate 复用现有 `basic` 结构（`gate_in_dim = shared_dim + compl_dim`） |
| `forward` | 增加 `elif self.mode == "cgscd": return self._forward_cgscd(...)` |
| 新方法 | `_forward_cgscd(item_ids, return_components)`，与 `_forward_aspcf` 结构同构，区别仅在 `z_sh`/`z_pv` 的构造 |
| `return_components` | 复用现有契约，额外返回 `z_shared`、`z_private` 供 Ch4 的 prototype 与诊断使用 |

**不需要改动**：`QueryMultiInterestExtractor`、`InterestAggregator`、`position_emb`、`BaseRunner`、`main.py`（只需新增 model 文件后 import）。

#### 2.3.4 与旧 ASPCF 的关系

- `--item_encoder aspcf` **保留不动**，作为 Chapter 3 的 baseline。
- 新方法新增为独立 model 文件（建议 `LLMMIRecCGSCD.py`）+ 独立 encoder mode。
- 递进叙事：ASPCF = 「按 PCA 方差人工切片」；CGSCD = 「按跨视图关联自动分解」。前者是后者的特例/对照组。

---

## 3. Chapter 4 — Interest Representation

### 3.1 已有探索与结论

已探索：`LLMMIRecHSDIR`（层次语义路由蒸馏）、`LLMMIRecCHIR`（prototype query + dual-view routing）、`LLMMIRecCASIR`（协同锚定语义兴趣精炼）、`LLMMIRecCAISD`（兴趣级语义分布蒸馏）。

**已有结论（保留，不推翻）**：
- semantic teacher **可以显著改变**兴趣结构：Beauty 兴趣间余弦 0.9457 → 0.8691，有效秩 1.714 → 2.280，route membership entropy 0.913 → 0.543。
- 但**兴趣分化并不会稳定转化为 recommendation gain**：fair 5-seed 对比中 CAISD 并未稳定超过 ASPCF（详见 `THESIS_PROGRESS.md` §3）。

### 3.2 最终问题定义

> **如何让多个 latent interests 具有明确并且完整的 semantic organization，而不是仅仅彼此分离。**

关键词"完整"（completeness）是新引入的、HSDIR/CAISD 都没有约束的维度：HSDIR 约束的是"哪些行为应归为同一 interest"（分化），CAISD 约束的是"单个 interest 的语义是什么"（profile），但**没有任何机制要求 K 个 interest 合起来覆盖用户历史的全部语义**。

### 3.3 核心模块

#### 模块 1：Prototype-Grounded Interest Semantic Profile

对历史 item 的 semantic prototype assignment（`Q ∈ [B, L, P]`），结合 interest attention（`A ∈ [B, K, L]`），生成每个 interest 的 semantic teacher profile：

```
T_k = Σ_l Ā_{k,l} Q_{b,l,:}        # [B, K, P]，归一化到单形
```

student 从 interest vector 预测 semantic profile：

```
P_k = softmax(g_predictor(V_k))     # [B, K, P]
L_profile = KL(T_k ‖ P_k)
```

**与现有 CAISD 的关系**：机制相同，直接复用 `LLMMIRecCAISD.py:248-297`。差别在：
1. `Q` 的来源——Chapter 4 的 prototype 建立在 Chapter 3 的语义表示上（见 §5.2），而非 PCA-512 切片；
2. `L_profile` 不再是本章唯一的 loss（见模块 2）。

#### 模块 2：Multi-Interest Semantic Coverage

构造整个 history 的 semantic distribution：

```
q_seq = normalize( Σ_{l valid} Q_{b,l,:} )        # [B, P]
```

由 K 个 interest semantic profile `P_k` 与 **history-only** interest weights `w_k`：

```
q_hat_seq = Σ_k w_k · P_k                          # [B, P]
```

约束 `q_hat_seq` 覆盖/重构 `q_seq`：

```
L_coverage = D( q_seq ‖ q_hat_seq )
```

**`w_k` 必须只用 history 计算**（复用 `InterestAggregator(history_emb_raw, lengths)`，`LLMMIRecCAISD.py:408`），不得使用 candidate/target 信息，否则构成 label leakage。

**⚠️ 本模块的头号设计风险（必须处理，否则 Chapter 4 退化）**：

`L_coverage` 存在**平凡解**：若所有 `P_k` 都预测同一个分布，则 `q_hat_seq = Σ_k w_k P_k = P` 与 `w` 无关；只要 `P ≈ q_seq` 即可使 loss 归零。此时 coverage 完全满足，但 K 个 interest **没有任何语义分工**——恰好是本章要避免的。

**处理方案**：coverage 必须与**专业化约束**成对出现：
- 对 `P_k` 加**低熵/稀疏先验**（每个 interest 应聚焦于少数 prototype）：
  `L_focus = -Σ_k w_k · H(P_k)` 或 `L_focus = Σ_k w_k · ‖P_k‖₂ 的负值`；
- 可选：跨 interest 的 profile 去重（`Σ_{k≠j} sim(P_k, P_j)` 惩罚），但需谨慎——HSDIR 的教训说明"强行分离"本身不产生收益，因此**建议以 focus 为主、去重为辅，并把去重作为消融而非默认开启**。

这一对目标合起来才构成完整的表述：**`q_seq ≈ Σ_k w_k P_k`，其中每个 `P_k` 稀疏，`w_k` 由 history 决定**——这与 topic model（LDA）的分解形式同构，是一个有理论支撑、可解释、且非平凡的表述。

#### 模块 3：HSDIR 的定位

`LLMMIRecHSDIR` 保留为 **preliminary / motivation experiment**，不构成 Chapter 4 最终方法的组成部分。它的作用是在论文中建立"语义结构可被 teacher 改变"这一前提，以及"改变结构 ≠ 获得收益"这一动机。

### 3.4 损失构成（Ch4 章节内）

```
L_total = L_BPR
        + λ_profile  · L_profile      # 模块 1（继承 CAISD）
        + λ_coverage · L_coverage     # 模块 2（新）
        + λ_focus    · L_focus        # 模块 2 的必要配套（新）
        + λ_relation · L_relation     # 继承 Ch3（Ch3 后为 CGSCD 版关系约束）
```

**注意**：Chapter 4 不再是"一个 KL loss"。这是对 §8 问题 2 的正面回答——见 `THESIS_WORKLOAD_REVIEW.md`。

---

## 4. Chapter 5 — Interest Decision

### 4.1 已有基础

`CAISD` 的 `--semantic_distill_mode`（静态 profile 蒸馏）、`TASID llm_only`、`TASID asymmetric`、`tools/analyze_caisd_teacher_benefit.py`（teacher benefit 分析）。

**已有结论**：teacher 需要与当前预测目标动态匹配；静态 teacher 收益不稳定。但 fair 5-seed 对比显示现有 TASID 未稳定超越 ASPCF（见 `THESIS_PROGRESS.md` §3），因此 Chapter 5 必须**重新定义问题**而不是继续调参。

### 4.2 最终问题定义

> 即使已经得到语义清晰的多个 interests，**如何针对当前 target item 选择真正相关的 interest，并让这种 target-interest knowledge 真正改善 ranking？**

关键在最后半句"真正改善 ranking"——现有 TASID 只在**训练期**通过 KL 影响兴趣向量，ranking 打分路径 `prediction = <Σ_k w_k V_k, e_cand>` 完全不变，`w_k` 也不受 target 影响。这解释了为什么收益不稳定：**target-interest 知识从未进入打分**。

### 4.3 核心模块

#### 模块 1：Target-Conditioned Semantic Interest Teacher

基于当前 positive target 的语义 `q_target` 与 Chapter 4 的 interest semantic profile `T_k`（或 `P_k`），构造 target-conditioned interest distribution：

```
p_teacher[k] = softmax_k( cos(q_target, T_k) / τ )      # [B, K]
```

**与现有 TASID 的差异**：现有 TASID 的 teacher 由 `q_target = semantic_branch(z_target[:, :512])` 与 `T = bmm(A_detach, Q)` 构成——语义坐标是 **PCA-512 切片**。Chapter 5 的 teacher 必须建立在 **Chapter 3 的共享语义子空间**与 **Chapter 4 的 P_k** 上，从而形成三章的真实继承。

#### 模块 2：Teacher Confidence Calibration

```
c = 1 - H(p_teacher) / log(K)
```

teacher 越确定，distillation 权重越大。

**⚠️ 双面风险（Phase 0 实测，必须在实现前先测 runtime 熵）**：

Phase 0 用 prototype 空间（P=32）代理测得：

| τ | Beauty 平均 H (max ln32=3.466) | 平均 `1-H/ln32` | ML-1M 平均 H | 平均 `1-H/ln32` |
|---|---|---|---|---|
| 0.05 | 0.483 | 0.861 | 0.534 | 0.846 |
| 0.10 | 1.617 | 0.533 | 1.458 | 0.580 |
| 0.20 | 2.977 | 0.141 | 2.760 | 0.204 |
| 0.50 | 3.413 | 0.015 | 3.369 | 0.028 |

- **代理空间结论**：τ=0.1 时 confidence 落在 0.53 附近，有良好区分度，校准**不会失效**。
- **但真实 teacher 分布在 K=4 个 interest 上，且 `T_k = bmm(A, Q)` 是 prototype 的加权平均**。若某个 interest 覆盖语义多样的 item，`T_k` 会趋于均匀 → `cos(q_target, T_k)` 对所有 k 接近 → `p_teacher` 接近均匀 → `H → log 4` → `c → 0` → **校准会把 loss 整体关闭**。
- 这是一个与代理测量**方向相反**的风险。

**处理方案**：
1. 实现前先加一个 runtime 探针，测量真实 `p_teacher` 的熵分布（几秒级，一个 batch 即可）；
2. 若熵接近 log K，改用 **τ-independent 的置信度**：由 `cos_teacher` 的 top-1/top-2 margin 归一化构造，例如 `c = sigmoid(β · (cos_top1 - cos_top2))`，绕开温度饱和问题；
3. 无论采用哪种，都必须在日志中记录 `c` 的均值与方差，并把它作为消融（`confidence_mode ∈ {none, entropy, margin}`）而非固定实现。

#### 模块 3：Target-Discriminative Semantic Ranking

针对 positive target 使用 **semantic hard negative item**，基于 target-conditioned interest representation 增加判别式 ranking 目标。

**关键设计：让 target-interest 知识真正进入打分。**

- target-conditioned interest representation：
  ```
  π = student target-conditioned distribution  (由 TASID 的 student 侧产生, [B, K])
  z_t = Σ_k π_k V_k                            [B, D]
  ```
- positive score：`s⁺ = <z_t, e_pos>`
- hard-negative score：`s⁻ = <z_t, e_hn>`，`e_hn` 来自 semantic hard-negative bank
- discriminative ranking loss：
  ```
  L_disc = max(0, m - s⁺ + s⁻)
  ```

这条 loss 与 BPR 的区别在于：BPR 用 `user_vector = Σ_k w_k V_k`（history-only 权重），而 `L_disc` 用 **target-conditioned 的 `π`**。因此梯度会同时推动 (a) `V_k` 本身更有判别性，(b) student 分布 `π` 更准确。这是"target-interest 知识进入 ranking"的具体落点。

**与现有 TASID 的最小差异**：现有 TASID 的 student 分布 `q_tasid` 只用于计算 KL，不参与任何打分。`L_disc` 是新增的第三个 loss 项。

### 4.4 损失构成（Ch5 章节内）

```
L_total = L_BPR
        + λ_profile   · L_profile
        + λ_coverage  · L_coverage
        + λ_focus     · L_focus
        + λ_tasid     · [ c · KL(p_teacher ‖ π) ]      # 增加 confidence 加权
        + λ_disc      · L_disc                          # 新增，进入 ranking
        + λ_relation  · L_relation
```

`L_disc` 的存在是 Chapter 5 区别于"TASID 小修"的核心证据——见 `THESIS_WORKLOAD_REVIEW.md` §问题 3。

---

## 5. 三章递进关系

### 5.1 监督信号流

```
Chapter 3
  输入: Z (frozen LLM), C (frozen CF)
  学习: U_r (离线), f_shared, f_compl, gate
  产出: e_item = concat[√α_s·s, √α_c·c]
        + 语义空间定义: Z_shared / Z_private

        │  e_item 送入 extractor/aggregator（三章共享）
        │  Z_shared 坐标定义 prototype（Ch4）
        ▼
Chapter 4
  输入: e_item, prototype 空间
  学习: semantic_predictor g, (可选) focus 权重
  产出: P_k  [B,K,P]  每个 interest 的语义 profile
        w_k  [B,K]    由 InterestAggregator 给出的 history-only 权重

        │  T_k / P_k 送入 teacher 构造（Ch5）
        │  V_k 送入 target-conditioned 表示（Ch5）
        ▼
Chapter 5
  输入: q_target, T_k, V_k, w_k
  学习: (可选) confidence 参数
  产出: π  [B,K]  target-conditioned student 分布
        z_t = Σ_k π_k V_k   （进入 ranking 打分）
        L_disc
```

### 5.2 Chapter 3 → Chapter 4 的 prototype 依赖（关键工程决策）

现状：`llmmi_proto32_sr512.pkl` 的生成链为

```
llm_table_pca1536.pkl
  → z[:, :512]                       (PCA 方差切片)
  → MiniBatchKMeans(n_clusters=32)
  → centers [32, 512]
  → softmax(cos(z_shared, centers)/0.1)
  → soft_assignments [n_items, 32]   (row 0 = zeros)
```

**问题**：若 Chapter 3 不再使用 PCA first-512 作为 semantic representation，Chapter 4 的 prototype 是否也应重建？

**答案：应该重建，但必须以"分阶段 + 可对照"的方式重建，不能静默替换。**

推荐方案（三档）：

| 档 | prototype 建立在 | 是否需要训练 | 用途 |
|---|---|---|---|
| A（保留） | `z[:, :512]`（现状） | 离线 | HSDIR preliminary 实验的复现；"LLM-only 语义空间"对照 |
| **B（主方法）** | `Z_shared = (Z̄ - μ_z) U_r`，即 Chapter 3 的共享子空间 | 离线 | Chapter 4 主实验 |
| C（仅诊断） | Chapter 3 的 `semantic_branch` 输出 `s`（已训练） | 需模型，且随训练漂移 | **不建议作为主方法**：teacher 会随 student 漂移，失去"冻结 teacher"的性质 |

**明确排除 C**：把 prototype 建立在 student 自己学出的表示上，会使 teacher 不再独立，Chapter 4 的核心论证（"语义组织来自 LLM 而非自举"）将不成立。

需要注意的一个取舍：档 B 的共享子空间只有 r=32 维，而档 A 是 512 维。在 32 维上做 KMeans 聚类，簇的可分性可能与 512 维不同。**必须同时提供 r=64 的档 B 变体**，否则"prototype 质量下降"会与"表示更好"混淆。

**需要修改的 build script**：
- `tools/build_llmmi_semantic_prototypes.py`：增加 `--input_rep {pca512, cgscd_shared}` 与 `--basis_path`，并按 `input_rep`/`r` 编码输出文件名（当前输出名 `llmmi_proto{num}_sr{rank}.pkl` 无法区分表示来源）。预计改动 ~30 行。
- `tools/build_llmmi_hierarchical_teacher.py`：其输入是 fine prototype 文件路径，**只需路径参数化**，逻辑无需改动。但输出命名同样需要带上 `input_rep`。

### 5.3 Chapter 4 → Chapter 5 的依赖

Chapter 5 的 `T_k` 直接来自 Chapter 4 的 `P_k`。若 Chapter 5 用 Chapter 4 的 student `P_k` 而非 teacher `T_k` 构造 teacher 分布，会造成自举循环。**必须明确规定**：teacher 侧的 `T_k` 由 detached `Q` 与 detached `A` 构成（沿用 CAISD 的做法），student 侧才是可学习的 `P_k`。

---

## 6. 预计算资产清单

### 6.1 已存在（Phase 0 已验证）

| 文件 | 形状 | 来源 | 三数据集 |
|---|---|---|---|
| `data/<ds>/handled/llm_table.pkl` | (n_items, 4096) | 原始 LLM item embedding，row 0 = 0 | ✅ |
| `data/<ds>/handled/llm_table_pca1536.pkl` | (n_items, 1536) | 4096→1536 PCA，按方差降序 | ✅ |
| `data/<ds>/handled/itm_emb_pomrec.pkl` | (n_items−1, 64) | PoMRec ckpt `i_embeddings.weight[1:]` | ✅ 已逐元素验证 |
| `data/<ds>/handled/llmmi_proto32_sr512.pkl` | centers (32,512), soft_assign (n_items,32) | KMeans on `z[:, :512]` | ✅ |
| `data/<ds>/handled/llmmi_hier_proto32_8_sr512.pkl` | fine 32 + coarse 8 | KMeans on fine centers | ✅ |
| `data/<ds>/handled/semantic_hardneg_top100.pkl` | (n_items, 100) int64 | LLM 空间 L2-归一化 + 余弦 top-100，已排除自身 | ✅ |
| `data/<ds>/SeqReader.pkl` | — | corpus 缓存 | ✅ |

**`semantic_hardneg_top100.pkl` 是 Chapter 5 的重大复用点**——它已经存在、已覆盖三数据集、且质量已验证：row 0 全零、无自引用、每行 100 个唯一候选、余弦难度梯度干净（beauty: rank-1 0.602 / rank-2 0.528 / rank-10 0.391 / rank-50 0.281）。

### 6.2 需要新建

| 文件 | 生成工具（新建） | 服务于 |
|---|---|---|
| `data/<ds>/handled/cgscd_basis_r32.pkl` | `tools/build_cgscd_basis.py` | Ch3 |
| `data/<ds>/handled/cgscd_basis_r64.pkl` | 同上（`--rank 64`） | Ch3 消融 |
| `data/<ds>/handled/cgscd_proto32_r32.pkl` | `tools/build_llmmi_semantic_prototypes.py`（扩展） | Ch4 |
| `data/<ds>/handled/cgscd_hier_proto32_8_r32.pkl` | `tools/build_llmmi_hierarchical_teacher.py`（路径参数化） | Ch4 消融 |

### 6.3 溯源风险（已发现，必须记录）

- `srs_emb.pkl` / `llm_emb.pkl`（CLAUDE.md 中提及）**在仓库中不存在**；所有 `--srs_emb_path` 实际指向 `handled/itm_emb_pomrec.pkl`。CLAUDE.md 该处为过时描述。
- toys 的 `itm_emb_pomrec.pkl` 对应 checkpoint 为 `model/PoMRec/toys__1__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt`。仓库中另有 `itm_emb_pomrec1.pkl`（字节数与主文件相同）以及多个 `PoMRec__toys__*` checkpoint，**均与导出文件不匹配**（maxdiff = 2.50）。这与 beauty/ml-1m 的精确匹配（maxdiff = 0.0）不同，说明 toys 的导出流程曾有过一次未记录的重复运行。新建 `cgscd_basis` 时必须用 sha1 锁定来源。
- `llm_table.pkl`（原始 4096 维）在 **ML-1M 上存在 NaN/Inf**（方差计算返回 nan）。`pca1536` 版本干净（已通过 `load_llm_table` 的 NaN/Inf 校验）。任何直接使用原始 4096 维表的新工具都必须先加校验。

---

## 7. 风险登记册

| # | 风险 | 影响章节 | 证据 | 缓解 |
|---|---|---|---|---|
| R1 | 跨视图共享信号本身很弱 | Ch3 | CF 视图对 LLM 的线性 R² 仅 0.041 (beauty) / 0.017 (ml-1m)；非线性 MLP 探针 0.091 (beauty) | 重新表述：论证"LLM 语义绝大部分是私有的"本身就是本章发现；shared 分支不必承载主要信息量；提供非线性/可学习 basis 消融 |
| R2 | 协同视图选择是数据集相关的 | Ch3 | beauty PoMRec 4.1% > SVD128 1.6%；ml-1m PoMRec 1.7% < SVD128 24.8% | 把"协同视图来源"作为显式变量纳入分析实验，而非默认假设 |
| R3 | `M` 的秩 ≤ d_cf = 64，r 的上界被协同视图维度锁死 | Ch3 | 数学事实 | 明确记录；若需更大 r，必须先提升协同视图维度（换视图来源） |
| R4 | coverage loss 存在平凡解（K 个 interest 预测同一 profile） | Ch4 | 构造性分析：`P_k ≡ q_seq ⇒ q_hat = q_seq` 恒成立，与 w 无关 | 强制配套 focus / 稀疏先验；去重作为消融 |
| R5 | confidence 校准可能整体关闭 loss | Ch5 | 若 `p_teacher` 接近均匀则 `c → 0`；代理实测在 prototype 空间不饱和，但真实 4-way teacher 未测 | 实现前先做 runtime 熵探针；提供 margin-based τ-independent 置信度作为 fallback |
| R6 | target-interest 知识从未进入打分 | Ch5 | 现有 TASID 只影响训练期 KL，`prediction` 路径不变 | `L_disc` 用 `z_t = Σ π_k V_k` 直接进入打分，这是 Chapter 5 的核心增量 |
| R7 | 现有工具大量硬编码架构参数 | 全局 | `analyze_llmmirec_caisd.py` 硬编码 `history_max=20, semantic_rank=512, aspcf_gate_mode=basic, semantic_distill_mode=uniform`；`load_state_dict(strict=False)` 静默接受缺键 | 新章节的诊断工具不得复制该模式；至少加 key 存在性断言 |
| R8 | 模型文件间大量逐行复制 | 全局 | `_compute_relation_loss` 在 4 个文件中逐字节相同（~19 行 × 4）；CASIR ≈ 55-60% 逐字复制自 CAISD；CHIR ≈ 35% 复制自 ASPCF | 新模型（Ch3/4/5）应抽公共 mixin；但**本阶段不做重构**，仅在实现新文件时不再复制 |
| R9 | 实验记录表缺列导致消融不可区分 | 全局 | HSDIR 的 `summary_seed42.tsv` 无 loss_mode/margin/confidence/route_source 列，13 行中多行无法区分 | 新的 summary 表必须为每个被扫的超参数建列 |
| R10 | 脚本无 conda 激活，失败被记为 `status=OK` | 全局 | ASPCF/CAISD summary.tsv 中各有 10 行 `ModuleNotFoundError: No module named 'torch'` 被记录为 OK（total_seconds 0-2） | 新脚本加 `python -c "import torch"` 前置检查 + `set -o pipefail` |

---

## 8. 本阶段禁止事项（自我约束）

Phase 0 不做以下任何事：

- ❌ 修改 evaluation protocol、`helpers/BaseRunner.py` 的评测逻辑
- ❌ 修改数据 split、`data/*/train.csv|dev.csv|test.csv`
- ❌ 运行正式训练
- ❌ 删除任何旧代码、旧实验、旧模型、旧日志
- ❌ 重构项目结构
- ❌ 实现 Chapter 3 / 4 / 5 的任何模块
- ❌ 自动进入下一阶段

Phase 0 允许且仅做了：阅读代码、数据/文件 shape 与 provenance 检查、秒级 Python 探针、文档生成。

---

## 9. 待决问题（进入实现阶段前需要确认）

1. Chapter 3 是否接受把"共享信号弱"重新表述为正面发现（R1）？这直接决定章节的叙事方式。
2. Chapter 3 是否愿意同时维护 PoMRec-64 与 SVD-128 两个协同视图（R2）？这会增加消融量但显著增强说服力。
3. Chapter 4 的 `L_focus` 采用熵先验还是稀疏（L2/L1）先验？
4. Chapter 5 的 confidence 是否默认采用 margin-based 而非 entropy-based（R5）？
5. 三章的最终主实验数据集是 Beauty + ML-1M 两个，还是加入 Toys？（Toys 的溯源问题 R3 需要先解决）

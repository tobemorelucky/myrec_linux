# CHAPTER3_FINAL.md

> Chapter 3 最终资产冻结文档
> 冻结日期：2026-10-06
> 状态：**FROZEN — ASPCF 结构不再修改**
>
> 本文件固定 Chapter 3 的最终方法、贡献、实验资产与缺口。
> CGSCD / RASRF 两条探索路线的记录见 `THESIS_CH3_ROUND1.md` 与本文件 §6。
> 最终比较目标见 `FINAL_EXPERIMENT_TARGETS.md`。

---

## 1. ASPCF 最终结构（冻结）

模型：`models/sequential/LLMMIRecASPCF.py`，编码器：`ItemEncoder(mode="aspcf")`（`models/sequential/llmmi_components.py`）

```
llm_table[item_id]  (PCA 1536 维，frozen buffer)
  │
  ├─ z_high = z[:, :512]                高方差主子空间
  │    └─ semantic_branch: Linear(512→128) → GELU → Linear(128→32) → s   [32]
  │
  └─ z_low  = z[:, 512:]                尾部分量（1024 维）
       ├─ complement_tail: Linear(1024→64) → GELU            [64]
       └─ complement_id_emb[item_id]  ← 可训练 ID 嵌入        [64]
            └─ concat → complement_mlp: Linear(128→64) → GELU → Linear(64→32) → c  [32]

  gate: [s; c] → Linear(64→64) → GELU → Linear(64→2) → softmax → [α_s, α_c]
  e_item = concat[ sqrt(α_s + ε)·s , sqrt(α_c + ε)·c ]        [64]

  L_total = L_BPR + λ_relation · L_relation
```

**关系保持损失** `L_relation`（`LLMMIRecASPCF.py::_compute_relation_loss`）：
从 batch 中采样 ≤128 个 unique item，teacher = frozen `z_high` 的 item-item 余弦关系，
student = `semantic_branch(z_high)` 的对应关系，去对角线后行 softmax，KL(teacher ‖ student)。
温度 teacher/student 均为 0.1。

**固定超参数**：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| `emb_size` | 64 | `semantic_rank` | 512 |
| `attn_size` | 64 | `semantic_dim` | 32 |
| `K` | 4 | `semantic_hidden` | 128 |
| `history_max` | 20 | `complement_dim` | 32 |
| `dropout` | 0.1 | `tail_hidden` / `complement_hidden` / `gate_hidden` | 64 |
| `lambda_relation` | 0.01 | `aspcf_gate_mode` | basic |
| `l2` | 1e-6 | `optimizer` | Adam |
| `batch_size` | 1024 | `eval_batch_size` | 256 |
| `epoch` / `early_stop` | 200 / 10 | `num_neg` | 1 |
| **Beauty `lr`** | **0.004** | **ML-1M `lr`** | **0.001** |

参数量：**943,174**（Beauty）；ML-1M 为 405,894。

---

## 2. 三个核心贡献

### 贡献 1 — 按子空间拆分 LLM 语义，而非整体替换或整体残差

**问题**：LLM item embedding 是 4096 维、经 PCA 压到 1536 维的稠密语义向量。此前两种用法都过于粗糙——
`llm_replace`（整个换掉 CF 嵌入）在 ML-1M 上大幅失效（NDCG@5 0.2130 → 0.1795）；
`residual`（全局 γ 加权相加）需要人为设定 γ 且无法区分语义的不同成分。

**方法**：按 PCA 方差把 LLM 语义**显式拆成两个子空间**——高方差主子空间 `z[:, :512]` 作为"语义主体"，
尾部分量 `z[:, 512:]` 作为"待补充的语义残余"，两路分别编码。

**关键设计点**：互补分支**不是纯语义尾部**，而是 `[尾部分量, 可训练 ID 嵌入]` 的融合。
ID 嵌入提供纯协同锚点，使互补分支编码的是"LLM 尾部 + 协同信号"的联合信息，而不是孤立的语义噪声。
这是与简单子空间切分（如 `z_high`/`z_low` 各自独立过 MLP）的实质差别。

### 贡献 2 — 自适应软门控融合，取代固定权重

`e_item = concat[√α_s·s, √α_c·c]`，`[α_s, α_c] = softmax(gate([s; c]))`。

门控是 **item-specific** 的（每个 item 有自己的 α），且用 `√α` 而非 `α` 加权——
`√α` 使融合后的向量范数近似守恒，避免 gate 通过整体缩放而非"选择"来影响表示。

**注（诚实记录）**：诊断显示训练后 `α_sem ≈ 0.011`，即门控把语义分支压到了很小权重
（`semantic_branch` 输出范数为 31.9，`complement` 为 2.0；门控用 α 抑制前者）。
门控并非无效，但实测行为是"语义分支以极小门控权重、极大自身范数参与"，而非"平衡的两路融合"。
**这一点必须在论文中如实描述，不能声称门控在语义/互补之间做了均衡选择。**

### 贡献 3 — 语义关系保持损失

不止对齐 item 的**表示**，而是对齐 item 之间的**关系结构**：
frozen `z_high` 中两个 item 的余弦相似度关系，被蒸馏到 `semantic_branch` 的输出空间。

**为什么这条有效（实测）**：

| 配置 | Beauty NDCG@5 |
|---|---|
| ASPCF, λ_relation = 0（PURE_BPR） | 0.1034 |
| ASPCF, λ_relation = 0.01 | 0.1088（seed 42） |
| 提升 | **+5.2%** |

Phase 1 配置（lr=0.001, bs=256）下同样成立：0.1003 → 0.1100（+9.7%）。

**机制**：该 loss 显著改变了 item 表示的几何——训练后历史 item 表示的平均两两余弦
从 **0.144**（λ=0）升到 **0.531**（λ=0.01），**3.7 倍**。这与排序指标的提升同步，
说明关系蒸馏的作用是"给表示空间注入可泛化的相似度结构"，而非单纯正则化。

**这是三个贡献中证据最强、机制最清楚的一条。**

---

## 3. 与 LLMEmb 的继承关系和明确区别

### 3.1 继承关系

| 层级 | 模型 | Beauty 5-seed | ML-1M 5-seed |
|---|---|---|---|
| ① 纯 CF 主干 | `PoMRec` | NDCG@5 0.0986 (seed42) | 0.2074 (seed42) |
| ② PoMRec + LLM 融合 | `PoMRecLLMEmb` (`llm_fuse_mode=replace`) | 0.1076 (seed42) | 0.2046 (seed42) |
| ③ 独立干净主干 + LLM 替换 | `LLMMIRec` (`item_encoder=llm_replace`) | 0.1016 (seed42) | 0.1795 (seed42) |
| ④ **Chapter 3 最终** | **`LLMMIRecASPCF`** | **0.1075 ± 0.0014** | **0.2140 ± 0.0023** |

Chapter 3 从 ② 继承了三个思想：**用 LLM 语义增强 item 表示**、**独立于 PoMRec 的干净多兴趣主干**、
**门控/权重化的融合而非固定系数**。

而从 ② 到 ④ 的关键变化是：不再把 LLM 当作一个整体去"替换"或"加权叠加"，而是**按语义子空间拆分后分别使用**。

### 3.2 明确区别（必须在论文中写清）

| 维度 | `PoMRecLLMEmb`（②，继承对象） | `LLMMIRecASPCF`（④，本章方法） |
|---|---|---|
| **主干** | PoMRec（`interest_extractor`，`prompt_num=3`，`lamb=4.0`） | 独立实现的 `QueryMultiInterestExtractor` + `InterestAggregator`，无 PoMRec 依赖 |
| **LLM 用法** | 整体替换 / 整体残差（`llm_fuse_mode ∈ {replace, residual, none}`） | **按 PCA 方差子空间拆分**为 semantic(前512) + complement(尾1024) |
| **融合权重** | 全局 `gamma`（标量，可训练或固定） | **item-specific 软门控** `[α_s, α_c]`，`√α` 加权拼接 |
| **协同信号的位置** | 仅在主干（PoMRec 自身的 ID 嵌入） | **显式注入互补分支**（`complement_id_emb` 与语义尾部融合） |
| **辅助监督** | InfoNCE 对齐（`--use_llmemb`, `--alpha`, `--tau`）——**对齐表示** | **关系保持 KL**——对齐 item-item **关系结构**，不对齐表示本身 |
| **是否有语义教师** | 无（LLM 嵌入直接作为输入） | 有（frozen `z_high` 作为 relation teacher） |
| **参数量** | 855 KB checkpoint（含 PoMRec 全部） | 943,174（Beauty） |
| **可复现性** | 依赖 PoMRec 预训练与 warm-start | 无 warm-start、无 pretrained checkpoint、单文件可复现 |

**一句话区别**：② 把 LLM 嵌入当作**一个整体输入**去替代/叠加 CF 嵌入；
④ 把 LLM 嵌入当作**一个可以被拆解的结构**，按子空间分工，并用关系蒸馏保持语义几何。

---

## 4. 已有实验资产

### 4.1 主实验（5 seeds：0 / 1 / 2 / 41 / 42）

**Beauty**（`new_log/llmmirec_aspcf_phase2/beauty/`，lr=0.004, bs=1024）

| seed | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | best iter | 秒 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1589 | 0.2243 | 0.3063 | 0.1089 | 0.1301 | 0.1507 | 83 | 1069 |
| 1 | 0.1571 | 0.2296 | 0.3143 | 0.1059 | 0.1294 | 0.1508 | 79 | 1053 |
| **42** | **0.1592** | **0.2292** | **0.3171** | **0.1088** | **0.1313** | **0.1535** | 61 | 869 |
| 2 | 0.1590 | 0.2276 | 0.3132 | 0.1079 | 0.1300 | 0.1515 | 102 | 1291 |
| 41 | 0.1566 | 0.2250 | 0.3133 | 0.1058 | 0.1278 | 0.1501 | 47 | 718 |
| **mean** | **0.1582** | **0.2271** | **0.3128** | **0.1075** | **0.1297** | **0.1513** | | |
| **std** | **0.0011** | **0.0022** | **0.0036** | **0.0014** | **0.0011** | **0.0012** | | |

**ML-1M**（`new_log/llmmirec_aspcf_phase2/ml-1m/`，lr=0.001, bs=1024）

| seed | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | best iter | 秒 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3066 | 0.4323 | 0.5662 | 0.2133 | 0.2539 | 0.2878 | 115 | 4644 |
| 1 | 0.3108 | 0.4341 | 0.5651 | 0.2185 | 0.2584 | 0.2915 | 125 | 5517 |
| **42** | **0.3068** | **0.4321** | **0.5601** | **0.2134** | **0.2538** | **0.2861** | 92 | 4279 |
| 2 | 0.3101 | 0.4354 | 0.5654 | 0.2133 | 0.2538 | 0.2868 | 113 | 5045 |
| 41 | 0.3040 | 0.4305 | 0.5601 | 0.2116 | 0.2523 | 0.2852 | 120 | 5317 |
| **mean** | **0.3077** | **0.4329** | **0.5634** | **0.2140** | **0.2544** | **0.2875** | | |
| **std** | **0.0025** | **0.0017** | **0.0027** | **0.0023** | **0.0021** | **0.0022** | | |

> **注**：实际已有 **5 个种子**（0/1/2/41/42），不是 3 个。全部为已有日志，未重新训练。
> 论文中建议报告 5-seed mean±std，而非单 seed。

### 4.2 消融实验（已有）

**(a) item encoder 对比**（Phase 0，`new_log/llmmirec_phase0/`，lr=0.001, bs=256, seed 42）

| dataset | id | llm_replace | residual |
|---|---|---|---|
| Beauty NDCG@5 | 0.0832 | **0.1016** | 0.0946 |
| ML-1M NDCG@5 | 0.2130 | 0.1795 | **0.2131** |
| Toys NDCG@5 | 0.1187 | **0.1433** | — |

**(b) relation loss 消融**（Phase 1，`new_log/llmmirec_phase1_aspcf/`，lr=0.001, bs=256, seed 42）

| dataset | λ_relation = 0 | λ_relation = 0.01 | Δ |
|---|---|---|---|
| Beauty NDCG@5 | 0.1003 | **0.1100** | **+9.7%** |
| ML-1M NDCG@5 | —（缺） | 0.2129 | — |

**在最终配置（lr=0.004, bs=1024）下重做的 PURE_BPR 对照**（seed 42，本轮新增）：

| dataset | λ=0 | λ=0.01 | Δ |
|---|---|---|---|
| Beauty | 0.1534 / 0.1034 | **0.1592 / 0.1088** | **+3.8% / +5.2%** |
| ML-1M | 0.3078 / 0.2123 | 0.3068 / 0.2134 | −0.3% / +0.5% |

> ⚠️ **relation loss 的收益高度数据集相关**：Beauty +5.2%，ML-1M ≈ 0。这是必须在论文中说明的。

**(c) gate 结构消融**（Phase 1，`new_log/llmmirec_phase1_gate/`，bs=2048）

| dataset | basic（最终采用） | conflict |
|---|---|---|
| Beauty NDCG@5 | 0.1053 | **0.1101** |
| ML-1M NDCG@5 | **0.2144** | 0.2106 |

> 两个数据集方向相反。最终固定为 `basic`（更简单、参数量更少、ML-1M 更好）。

**(d) llm_replace + relation 对照**（排除"收益来自 backbone 而非 ASPCF"）

| dataset | LLMMIRec llm_replace + relation | LLMMIRecASPCF |
|---|---|---|
| Beauty NDCG@5 | 0.1046 | **0.1100** |
| ML-1M NDCG@5 | 0.1831 | **0.2129** |

**(e) batch size / lr 扫描**（`new_log/llmmirec_bs_sweep/`，三数据集 × 7–9 组配置）

### 4.3 诊断实验（已有）

| 指标 | 值 | 来源 |
|---|---|---|
| `α_sem`（历史 item 平均） | **0.0110** | `tools/analyze_cgscd_itemrep.py` |
| `α_sem` > 0.95 的 item 占比 | 0.0% | 同上 |
| `‖s‖` / `‖c‖` | 31.92 / 2.00 | 同上 |
| `cos(s, c)` | 0.049 | 同上 |
| **历史 item 表示两两余弦（λ=0.01）** | **0.5314** | 同上 |
| 历史 item 表示两两余弦（λ=0） | 0.1439 | 同上 |
| 有效秩（item 表示） | 26.03 | 同上 |
| 兴趣间余弦 / 有效秩 / route 熵 | 0.9457 / 1.714 / 0.913 | `new_log/llmmirec_hsdir_phase1/beauty/diagnostics/baseline/stats.tsv` |

---

## 5. 仅用于最终论文展示、目前缺失的图表与分析

按重要性排序：

| # | 缺口 | 说明 | 成本 |
|---|---|---|---|
| **1** | **Toys 上的 ASPCF 主实验完全缺失** | Beauty / ML-1M 有 5 seeds，**Toys 一个 ASPCF run 都没有**。若论文要主张跨三数据集有效，必须先跑 Toys（5 seeds） | 高（Toys 未跑过，运行时间未知，预计显著长于 ML-1M） |
| **2** | **relation loss 在 ML-1M 的 λ=0 消融缺失** | Beauty 有 0 / 0.01 两点，ML-1M 在 Phase 1 配置下只有 0.01。**最终配置下的 λ=0 对照已补**（0.3078/0.2123），可直接用 | 低（已完成） |
| **3** | **seed 方差可视化** | 5-seed 数据齐全，但无箱线图/误差棒图 | 低（纯绘图） |
| **4** | **gate α 分布图** | 数据已由诊断工具产出（α_sem ≈ 0.011），但无直方图 | 低 |
| **5** | **λ_relation 敏感性曲线** | 目前只有 0 与 0.01 两点，不足以画曲线。建议补 0.001 / 0.005 / 0.05 | 中（Beauty 单 seed 可先做） |
| **6** | **`semantic_rank` 敏感性** | 固定 512，未扫。建议 {128, 256, 512, 768, 1024} | 中 |
| **7** | **t-SNE / 表示几何可视化** | 需对比 id / llm_replace / ASPCF(λ=0) / ASPCF(λ=0.01) 四种表示的 item 分布 | 低-中 |
| **8** | **效率对比表** | 参数量、训练时间已知（本文件 §1、§4.1），但无与基线并列的表 | 低 |
| **9** | **与外部基线的对比表** | 见 `FINAL_EXPERIMENT_TARGETS.md`：目前仅 Beauty 超过 PoMRec，两数据集均未超 SATCRec | 取决于 Ch4/Ch5 |
| **10** | **图：ASPCF 结构示意图** | 论文方法图，尚无 | 低（绘图） |

---

## 6. Negative Exploration 记录（不再修改、不再重跑）

### 6.1 CGSCD — Collaborative-Guided Shared/Complementary Decomposition

**文档**：`THESIS_CH3_ROUND1.md`（完整记录）
**代码**：`models/sequential/LLMMIRecCGSCD.py`、`tools/build_cgscd_basis.py`、`tools/analyze_cgscd_itemrep.py`、`tests`/单测
**状态**：❌ 停止，不再修改、不再重跑

**做了什么**：用 LLM 视图 `Z` 与协同视图 `C` 的跨视图关联（cross-covariance SVD 或 regularized CCA）
求解共享子空间基 `U_r`，把 LLM 表示拆成 shared / private 两路，与协同嵌入融合后过软门控。

**关键结果**：

| 配置（Beauty / ML-1M，seed 42） | HR@5 Beauty | HR@5 ML-1M |
|---|---|---|
| ASPCF PURE_BPR | 0.1534 | 0.3078 |
| CGSCD-crosscov PURE_BPR | 0.1575 (+2.7%) | 0.2954 (**−4.0%**) |
| CGSCD-crosscov + relation | 0.1629 (+2.3%) | 0.2838 (**−7.5%**) |
| CGSCD-CCA PURE_BPR | 0.1588 | 0.2773 (**−9.9%**) |

**失败原因**：架构净增益的**符号跨数据集相反**。剥离 relation loss 后 ML-1M 仍下降，
故不是"新表示与旧监督不匹配"，而是**分解本身在 ML-1M 上不成立**。

**设计教训**：
1. **必须在两个数据集上同时验证，且先看 PURE_BPR。** 若只看 Beauty、或只看叠加了 relation loss 的结果，会得出完全相反的结论。
2. **CCA 不可用于此场景**：其规范相关（0.81）来自低方差方向，共享子空间只占 1.5% 方差、仅捕获 11.3% 的协同可解释方差，实质是空操作。
3. **离线诊断指标好 ≠ 推荐指标好**：crosscov 的 `pred_share_in_shared` 达 94.5%（三数据集一致），但推荐指标并不迁移。
4. **"共享子空间的内容确实在起作用"**（对照实验：crosscov 比 random 基高 8.9% HR@5）——这条正面结论保留，作为 sub-space 选择重要性的证据。

**保留价值**：`tools/build_cgscd_basis.py` 的 null/random 对照设计（控制变量法）可复用于任何表示学习实验。

### 6.2 RASRF — Reliability-Aware Semantic Residual Fusion

**文档**：本轮结果尚未成文（见下）
**代码**：`models/sequential/LLMMIRecRASRF.py`、`tools/build_semantic_neighborhood_agreement.py`、`tools/test_llmmirec_rasrf.py`、`new_bash/run_llmmirec_rasrf_phase1_beauty.sh`
**状态**：❌ 停止，不再修改、不再重跑

**做了什么**：`e_final = e_cf + gate(consistency) * e_sem`。
`e_cf` 为可训练 ID 嵌入（CF 主路径），`e_sem = adapter(llm_table)` 为语义修正，
`gate` 为 item-specific，输入为跨视图一致性信号：`cos(e_cf, e_sem)`、`|e_cf − e_sem|`、
`e_cf * e_sem`，可选离线邻域一致度先验。`LOSS = 原 BPR`，无任何辅助 loss。

**关键结果**（Beauty，seed 42，全部 8 指标一致）：

| config | HR@5 | NDCG@5 | vs ASPCF |
|---|---|---|---|
| ASPCF（rel=.01） | 0.1592 | 0.1088 | — |
| rasrf_scalar | 0.1407 | 0.0943 | −11.6% |
| rasrf_vector | 0.1391 | 0.0952 | −12.6% |
| rasrf_vector_neigh | 0.1395 | 0.0956 | −12.4% |
| residual_gamma（全局 gamma 对照） | 0.1378 | 0.0934 | −13.4% |

**失败原因（已定位到具体实现缺陷）**：

并非"可靠度门控"这一想法被否证，而是**门控输入的参数化有缺陷**：

| gate MLP 输入分量 | 范数占比 |
|---|---|
| `cos(e_cf, e_sem)` ← 唯一尺度无关的可靠度信号 | **0.31%** |
| `\|e_cf − e_sem\|` | **96.1%** |
| `e_cf * e_sem` | 27.0% |

训练后 `‖e_sem‖ = 59.4` 而 `‖e_cf‖ = 2.25`（26 倍失衡），因此 `|e_cf − e_sem|` 退化为
"语义分支有多大"的幅度代理，把真正的可靠度信号淹没。门控最终收敛到 `sigmoid(−4) ≈ 0.017`
（初始 0.5），语义路径被压至近关闭。

**支持性证据**：尽管被淹没，学到的 gate 仍与离线邻域一致度先验**正相关**
（scalar +0.21、vector_neigh +0.25）——说明信号存在，只是未被有效利用。

**设计教训**：
1. **门控输入必须尺度归一化**。逐元素特征（`|a−b|`、`a*b`）在两侧分支范数失衡时会退化为幅度特征。修法：先 L2 normalize 两侧分支，或对 gate 输入加 LayerNorm，或只保留尺度无关特征。
2. **item-specific 门控相对全局 gamma 只高 2.1%**（0.1407 vs 0.1378）。在当前实现下，"item-specific" 这一主张没有获得支持——但这可能是因为门控输入被淹没，而非机制本身无效。
3. **加性残差 vs 拼接**：RASRF 用 `e_cf + g·e_sem`（加性），而 CGSCD/ASPCF 用 `concat`。RASRF 的表示两两余弦降到 **0.038**（CGSCD-rel0 为 0.166，ASPCF 为 0.508），过度分散，可能是排序下降的直接原因（**假设，未验证**）。
4. **再次验证了 §3 的教训**：Beauty 单数据集结果不足以判定方向。

### 6.3 两条路线共同暴露的**结构性风险**

> **LLM 相关的改动在本项目上高度数据集相关。** 已出现第四次：
> ① Phase 0 `llm_replace`：Beauty +22%、ML-1M −16%
> ② CGSCD：Beauty +2.7%、ML-1M −4.0%
> ③ relation loss：Beauty +5.2%、ML-1M ≈ 0
> ④ RASRF：Beauty −12%
>
> **这应在选定 Chapter 4/5 方向前作为一个诊断问题正面回答**，而不是继续假设"在 Beauty 上调好即可迁移"。

---

## 7. 封板结论

### 7.1 ASPCF 与 PoMRec 的逐指标对比

**Beauty —— ASPCF 已全面超过 PoMRec（6/6 指标）**

| 指标 | 本仓库 PoMRec | 论文 PoMRec | **ASPCF (5-seed)** | vs 论文 PoMRec |
|---|---|---|---|---|
| HR@5 | 0.1436 | 0.1448 | **0.1582** | **+9.25%** |
| HR@10 | 0.2003 | 0.2028 | **0.2271** | **+11.98%** |
| HR@20 | 0.2679 | 0.2683 | **0.3128** | **+16.59%** |
| NDCG@5 | 0.0986 | 0.1007 | **0.1075** | **+6.75%** |
| NDCG@10 | 0.1169 | 0.1195 | **0.1297** | **+8.54%** |
| NDCG@20 | 0.1339 | 0.1360 | **0.1513** | **+11.25%** |

**ML-1M —— ASPCF 相对论文报告值低约 2%（0/6 指标）**

| 指标 | 本仓库 PoMRec | **论文 PoMRec** | **ASPCF (5-seed)** | vs 论文 PoMRec | vs 本仓库 PoMRec |
|---|---|---|---|---|---|
| HR@5 | 0.3046 | **0.3151** | 0.3077 | **−2.35%** | **+1.02%** |
| HR@10 | 0.4308 | **0.4422** | 0.4329 | **−2.10%** | +0.49% |
| HR@20 | 0.5647 | **0.5752** | 0.5634 | **−2.05%** | −0.23% |
| NDCG@5 | 0.2074 | **0.2188** | 0.2140 | **−2.19%** | **+3.18%** |
| NDCG@10 | 0.2481 | **0.2598** | 0.2544 | **−2.08%** | +2.54% |
| NDCG@20 | 0.2820 | **0.2933** | 0.2875 | **−1.98%** | +1.95% |

### 7.2 ⚠️ 必须精确表述的一点

**ML-1M 的 2% 差距是相对「论文报告的 PoMRec 数值」，不是相对「本仓库复现的 PoMRec」。**

- 本仓库自跑的 PoMRec（`new_log/pomrec_standard/ml-1m/PoMRec_seed42.log`）为 **0.3046 / 0.2074**；
- 论文报告值为 **0.3151 / 0.2188**，比本仓库复现值高 **3.4%**（HR@5）；
- ASPCF（0.3077 / 0.2140）**在本仓库协议下超过本仓库复现的 PoMRec 5/6 指标**。

因此 ML-1M 的"剩余差距"包含**两个来源**，在后续分析中必须分开：

| 来源 | 量级 | 性质 |
|---|---|---|
| (i) 本仓库 PoMRec 复现与论文的差异 | HR@5 约 **3.4%** | 复现/协议差异，**不是建模差距** |
| (ii) ASPCF 与本仓库 PoMRec 的差异 | HR@5 约 **−1.0% ~ +1.0%**（指标间不一致） | 真实建模差距，量级很小 |

**这条直接改变了 Chapter 4 的诊断问题**：不能假设"ML-1M 上还差 2% 需要靠新结构补回来"，
需先确认这 2% 中有多少属于复现差异。见 `CHAPTER4_PHASE0_DIAGNOSIS` 的诊断设计。

### 7.3 两个数据集的重要差异（Chapter 3 结论）

**Beauty —— LLM semantic 带来的收益明显**

| 对比 | Beauty NDCG@5 | Δ |
|---|---|---|
| `LLMMIRec` id（纯 CF） | 0.0832 | — |
| `LLMMIRec` llm_replace | **0.1016** | **+22.1%** |
| `LLMMIRecASPCF` | **0.1075** | **+29.2%** |

LLM 语义表示在 Beauty 上是**主要增益来源**。

**ML-1M —— ASPCF 相比 LLMMIRec-ID 提升很小**

| 对比 | ML-1M NDCG@5 | Δ |
|---|---|---|
| `LLMMIRec` id（纯 CF） | 0.2130 | — |
| `LLMMIRec` llm_replace | 0.1795 | −15.7% |
| `LLMMIRec` residual | 0.2131 | +0.05% |
| `LLMMIRecASPCF` | **0.2140** | **+0.47%** |

**结论：ML-1M 上，把 LLM 语义做得再精细，相对纯 CF 也几乎没有增益（+0.47%）。
该数据集的主要剩余瓶颈可能位于 multi-interest backbone，
而不是 item semantic representation。**

### 7.4 对后续章节的硬性要求

1. **Chapter 3 正式冻结为 ASPCF。** 不再通过修改 item encoder 来解决任何剩余差距。
2. **最终三章融合模型必须在 Beauty 与 ML-1M 上都超过 PoMRec 的六项指标。**
   （Beauty 已达成；ML-1M 待定，且需先厘清 §7.2 的复现差异）
3. **当前不跑 Toys。**
4. Chapter 4 的改进重点应放在 **multi-interest backbone**（`history → K interests` 与 aggregation/scoring），
   与 `CHAPTER3_FINAL.md` §7.3 的结论一致。

---

## 8. 冻结声明

自 2026-10-06 起：

- ✅ `LLMMIRecASPCF` 与 `ItemEncoder(mode="aspcf")` **结构冻结**，不再修改任何超参数或损失
- ✅ Chapter 3 的实验资产以本文件 §4 为准
- ✅ CGSCD / RASRF 代码与结果**完整保留**，但不再修改、不再重跑
- ⏸ Toys 上的 ASPCF 主实验为**已知缺口**，是否补跑由后续决定

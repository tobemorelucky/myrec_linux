# THESIS_WORKLOAD_REVIEW.md

> THESIS PHASE 0 — 研究工作量评估
> 建立日期：2026-10-03
>
> 本文档从**硕士学位论文工程量**角度评估三个 Chapter，并回答五个指定问题。
> 评估基准：现有仓库的实测运行时间、现有代码的复用率、以及 Phase 0 的实测探针结果。

---

## 0. 运行成本基准（评估的基础）

从 `summary.tsv` 的 `total_seconds` 字段实测（5 seed 均值）：

| 配置 | 单次运行时长 | 说明 |
|---|---|---|
| Beauty ASPCF（lr=0.004, batch=1024, 5 seeds） | **~1050 s ≈ 17.5 min** | 范围 741–1313 s |
| Beauty CAISD（同上） | **~770 s ≈ 13 min** | 范围 570–1022 s |
| ML-1M ASPCF（lr=0.001, batch=1024, 5 seeds） | **~4970 s ≈ 83 min** | 范围 4291–5529 s |
| ML-1M CAISD（同上） | **~4840 s ≈ 81 min** | 范围 3966–6810 s |
| Toys | **未在 Phase 2 跑过** | 79K items，`dev.csv` 1.4 GB，预计显著更慢 |

**一个"5 seeds × 2 数据集"的完整配置**（beauty 在 GPU0、ml-1m 在 GPU1 并行）：

```
wall-clock = max(5 × 17.5 min, 5 × 83 min) ≈ 6.9 小时
```

**这是本论文所有实验估算的基本单位。记作 1 CU（configuration unit）≈ 7 小时。**

若加入 Toys（假设单次 ~3–5× ML-1M），1 CU 会膨胀到 ~20 小时，且需要第三块 GPU 才能并行。**建议主实验限定 Beauty + ML-1M，Toys 仅用于最终补充验证。**

---

## Chapter 3 — CGSCD 工作量

### 统计

| 项 | 数量 | 明细 |
|---|---|---|
| **新方法模块数** | **6** | ①跨视图分解（离线）②shared 分支 ③complement 分支 ④自适应融合（复用 gate 结构）⑤双教师关系约束 ⑥跨侧冗余惩罚 |
| **新增 loss 数** | **3** | `L_rel_shared`（保持 Z_shared 的 item-item 关系）、`L_rel_private`（保持 Z_private 的关系）、`L_redundancy`（s 与 c 的去相关）。另有 `L_relation`（现有）被重写而非新增 |
| **新增预处理/离线工具** | **2** | `tools/build_cgscd_basis.py`、`tools/build_cf_svd_view.py`（对照视图） |
| **需要的主实验** | **2×5 seeds = 2 CU** | Beauty + ML-1M，r=32，frozen basis，与 ASPCF 5-seed 对比 |
| **需要的消融** | **7 组** | ① `r ∈ {8,16,32,64}` ② `basis_mode ∈ {frozen, learnable}` ③ `cgscd_use_cf_input ∈ {0,1}` ④ `cf_view ∈ {pomrec, svd128}` ⑤ `L_rel_shared on/off` ⑥ `L_rel_private on/off` ⑦ `L_redundancy on/off` |
| **参数实验** | **5** | `λ_rel_shared`、`λ_rel_private`、`λ_redundancy`、`shared_dim`、`compl_dim`（gate 维度由前两者决定） |
| **诊断实验** | **3** | 共享/私有方差占比、gate α 分布（item 级）、奇异值谱能量累积 |
| **可视化** | **4 类** | 奇异值谱 + 累积能量曲线；shared/private 方差占比条形图（可加 PCA-512 人工切片作对照）；gate α 分布；item 表示 t-SNE（ASPCF vs CGSCD） |

### 工作量估算

| 阶段 | 估算 |
|---|---|
| 离线工具（basis + cf_svd） | 1–2 天（含验证） |
| ItemEncoder 新 mode + 新 loss + 新模型文件 | 3–5 天 |
| CPU 单测 + 冒烟测试 | 1–2 天 |
| 主实验（2 CU，可并行） | **1 天** |
| 消融 7 组（14 CU，可并行） | **4–5 天** |
| 参数实验 5 组（10 CU） | **3 天** |
| 诊断 + 可视化 | 2–3 天 |
| **合计** | **约 2.5–3 周**（其中 GPU 时间约 1.5 周） |

### 判定：✅ **足以独立形成一个完整研究章节**

且额外具备一个**分析型贡献**：Phase 0 已实测"PCA 高方差方向与协同信号弱关联"（前 512 维占 78% 方差但仅 5% 可由 CF 解释；`z[:, 512:1024]` 与 `z[:, 1024:]` 降至 0.5% / 0.4%）。这一发现独立于新方法是否成功，本身即可构成章节的 motivation 与一节分析实验。

---

## Chapter 4 — ISPC 工作量

### 统计

| 项 | 数量 | 明细 |
|---|---|---|
| **新方法模块数** | **4** | ①prototype 重建（基于 Ch3 语义空间）②`q_seq` 构造 ③`q_hat_seq` 覆盖约束 ④`L_focus` 专业化约束 |
| **新增 loss 数** | **2** | `L_coverage`、`L_focus`。（`L_profile` 从 CAISD 原样复用，`L_sem_relation` 可选复用） |
| **新增预处理/离线工具** | **1**（+2 处修改） | 新建 `analyze_ispc_coverage.py`；修改 `build_llmmi_semantic_prototypes.py`（+30 行）与 `build_llmmi_hierarchical_teacher.py`（+10 行） |
| **需要的主实验** | **2×5 = 2 CU** | Beauty + ML-1M，CGSCD prototype + coverage + focus |
| **需要的消融** | **6 组** | ① **`coverage × focus` 2×2**（最关键）② `coverage_metric ∈ {js, kl, hellinger}` ③ `λ_coverage` ④ `λ_focus` ⑤ `focus_type ∈ {entropy, l2}` ⑥ `prototype_source ∈ {pca512, cgscd_r32, cgscd_r64}` |
| **参数实验** | **2** | `λ_coverage`、`λ_focus`（与消融 ③④ 合并） |
| **诊断实验** | **4** | 覆盖率（`1 − JS(q_seq, q_hat)` 的分布）、`P_k` 稀疏度（熵）、`w_k` 分布、逐用户覆盖率与收益的相关性 |
| **可视化** | **4 类** | `P_k` 热图（K×P，teacher vs student）；`q_seq` vs `q_hat` 对比（堆叠条/单形）；覆盖率-收益散点；`w_k` 分布 |

### 工作量估算

| 阶段 | 估算 |
|---|---|
| prototype 重建工具（含 Ch3 依赖） | 1 天（+ 依赖 Ch3 的 basis） |
| coverage + focus 实现 + 新模型文件 | 3–4 天 |
| CPU 单测（含平凡解防护测试） | 2 天 |
| 主实验（2 CU） | **1 天** |
| 消融 6 组（12 CU） | **4 天** |
| 诊断 + 可视化 | 3 天 |
| **合计** | **约 2–2.5 周** |

### 判定：⚠️ **边界性足够——但只在 `L_focus` 存在的前提下**

如果只有 `L_coverage`，Chapter 4 会塌缩成"一个 KL loss"（见 Q2）。加入 `L_focus` + 平凡解防护 + prototype 重建后，它才有 4 个模块、2 个 loss、6 组消融，达到章节体量。

**结构性风险**：Ch4 的主实验**硬依赖 Ch3 的 basis**（`cgscd_basis_r32.pkl` → `cgscd_proto32_r32.pkl`）。Ch3 未完成前，Ch4 只能跑 `prototype_source=pca512` 的版本作为早期验证，而那恰是 Ch4 想超越的旧方案。

---

## Chapter 5 — TDIR 工作量

### 统计

| 项 | 数量 | 明细 |
|---|---|---|
| **新方法模块数** | **3** | ①confidence 校准（entropy + margin 两种）②semantic hard-negative 采样与过滤 ③target-discriminative ranking |
| **新增 loss 数** | **1** | `L_disc`。`L_tasid` 从 TASID 复用并改造为 confidence 加权形式 |
| **新增预处理/离线工具** | **2**（+1 扩展） | 新建 `build_user_full_items.py`、`analyze_tdir_decision.py`；扩展 `scripts/build_semantic_hardneg.py`（+25 行，保存 cosine 值） |
| **需要的主实验** | **2×5 = 2 CU** | Beauty + ML-1M，TASID + confidence + `L_disc` |
| **需要的消融** | **5 组** | ① `confidence_mode ∈ {none, entropy, margin}` ② `λ_disc` ③ `disc_margin` ④ `hn_rank_lo/hi ∈ {[1,20], [2,20], [2,50], [10,100]}` ⑤ `disc_mode ∈ {hn_only, hn_plus_rand}` |
| **参数实验** | **2** | `λ_disc`、`disc_margin`（与消融 ②③ 合并） |
| **诊断实验** | **4** | `π` 分布熵、confidence 直方图、hard-negative 命中率与难度分布、`s_pos`/`s_hn` 边际分布 |
| **可视化** | **3 类** | `π` 分布（按目标语义分组的堆叠条）；confidence 直方图（entropy vs margin）；hard-negative 难度 vs 收益曲线 |

### 工作量估算

| 阶段 | 估算 |
|---|---|
| hard-negative bank 扩展 + user_full_items | 1 天（bank 已存在，工作量很小） |
| confidence + `L_disc` 实现 + 新模型文件 | 2–3 天 |
| CPU 单测（含硬负例过滤正确性） | 2 天 |
| 主实验（2 CU） | **1 天** |
| 消融 5 组（10 CU） | **3 天** |
| 诊断 + 可视化 | 3 天 |
| **合计** | **约 2 周** |

### 判定：✅ **足够，但增量最小，且独立性完全押在 `L_disc` 上**

- **可复用红利最大**：TASID 全部机制已实现（`LLMMIRecCAISD.py:316-364`），`semantic_hardneg_top100.pkl` 三数据集已就绪且质量已验证。
- **但增量也最小**：若不看 `L_disc`，Ch5 几乎是"给 TASID 加一个 confidence 权重 + 换一个负样本来源"。
- **独立性来自 `L_disc`**：这是唯一让 target-interest 知识**进入打分路径**的机制。现有 TASID 的 `prediction` 完全不受 target 影响。

---

## 五个关键问题的回答

---

### 问题 1：每章是否足以独立形成一个完整研究章节？

| 章节 | 独立成章 | 依据 | 缺口 |
|---|---|---|---|
| **Ch3** | ✅ **是** | 6 模块 / 3 loss / 2 离线工具 / 7 组消融 / 4 类可视化；另有独立的分析型发现（PCA 高方差 ≠ 协同相关） | 无 |
| **Ch4** | ⚠️ **条件性成立** | 4 模块 / 2 loss / 6 组消融 / 4 类可视化 | 必须有 `L_focus`；否则塌缩为单 loss 章节。且硬依赖 Ch3 |
| **Ch5** | ⚠️ **条件性成立** | 3 模块 / 1 loss / 5 组消融 / 3 类可视化 | 必须有 `L_disc`；否则是 TASID 增量章节 |

**结论**：三章中 Ch3 无条件是完整章节；Ch4 和 Ch5 的完整性各依赖一个关键机制能否成立。**这两个依赖必须在实现早期（而非晚期）验证。**

---

### 问题 2：Chapter 4 是否过于依赖一个 KL loss？

**答：如果只有 `L_coverage`，是的——而且是致命的依赖。但规划中的 Ch4 不止这一个 loss。**

**为什么单靠 KL 会失败（构造性论证）**：

`q_hat_seq = Σ_k w_k P_k`，其中 `Σ_k w_k = 1`（`InterestAggregator` 的 softmax 保证）。若所有 interest 学出同一个 profile `P_k ≡ P`，则：

```
q_hat_seq = Σ_k w_k P = P · Σ_k w_k = P
```

**与 `w_k` 完全无关。** 于是只要 `P ≈ q_seq`，`L_coverage → 0`。coverage 被完美满足，但 K 个 interest 没有任何语义分工——恰是 Chapter 4 要解决的问题的反面。

这是一个**平凡解**，KL 本身无法排除它。JS、Hellinger、cosine 同样无法排除。

**因此 Ch4 的实际结构必须是**：

| # | 技术声明 | 对应的 loss / 机制 | 是否是"KL loss" |
|---|---|---|---|
| 1 | 每个 interest 有明确的语义身份 | `L_profile`（从 CAISD 复用） | 是（KL） |
| 2 | K 个 interest **合起来**覆盖历史语义 | `L_coverage` | 是（JS，推荐） |
| 3 | 上述覆盖**不能**由 K 个相同 interest 平凡达成 | `L_focus`（低熵/稀疏先验） | 不是（熵/范数正则） |
| 4 | 语义坐标来自跨视图分解而非 PCA 切片 | prototype 重建（Ch3 依赖） | 不是（表示层） |
| 5 | 可选：interest 之间的语义关系被保留 | `L_sem_relation`（JS，从 CAISD 复用） | 是 |

**回答**：
- 若只保留声明 1+2 → Ch4 = "两个 KL loss" → **过于单薄**。
- 加入声明 3（focus）与 4（prototype 重建）后，Ch4 有 4 个模块、2 个必需 loss（一个非 KL）、6 组消融、2×2 最小完整消融矩阵 → **达到章节体量**。
- **且 2×2 消融（`coverage × focus`）是这个章节最有价值的实验**：它同时证明"coverage 有效"和"coverage 单独无效、必须配 focus"，这本身就是一个不平凡的研究结论。

**额外风险**：`L_focus` 与 HSDIR 的教训存在张力。HSDIR 强行让兴趣"分化"（去相关），结构指标大幅改善（cosine 0.9457→0.8691）却未带来收益。`L_focus` 本质上是让 `P_k` 变得尖锐，方向类似。**必须在 Ch4 中显式回答"为什么 focus 不会重复 HSDIR 的失败"**——初步论证是：HSDIR 约束的是**行为分组**（哪些 item 归为同一 interest），而 focus 约束的是**语义 profile 的形状**（每个 interest 对应少数 prototype），后者直接服务于 `q_hat ≈ q_seq` 这一可验证的目标，而前者没有可验证的全局目标。**这个论证需要在实验中得到支持，不能只停留在文字。**

---

### 问题 3：Chapter 5 是否只是 TASID 小修？

**答：只要 `L_disc` 成立，就不是小修；若 `L_disc` 不成立，Ch5 确实会塌缩成 TASID 小修。**

**量化对比**：

| 维度 | 现有 TASID | 规划中的 Ch5 | 增量性质 |
|---|---|---|---|
| teacher 的语义坐标 | `semantic_branch(z_target[:, :512])`，PCA-512 切片 | Ch3 的共享语义空间 | **弱增量**（同机制，换表示） |
| teacher 构造 | `softmax(cos(q_target, T_k)/τ)` | 同 | 无增量 |
| student 分布 | `llm_only` / `asymmetric` | 同 | 无增量 |
| KL 加权 | `reduction="batchmean"`，全样本等权 | `c = 1 − H/logK` 加权 | **中等增量**（见下） |
| **是否影响打分** | **否** —— `prediction = <Σ w_k V_k, e_cand>`，`w_k` 是 history-only | **是** —— `z_t = Σ_k π_k V_k` 进入 `L_disc` | **结构性增量** |

**关键判断依据**：

现有 TASID 的 target 知识**从未接触过打分路径**。它的全部影响是通过 KL 反向传播改变 `V_k` 的方向，然后 `V_k` 再以 history-only 的权重参与打分。这意味着：
- target 信息在打分时**已经被"用完并丢弃"**；
- 模型无法学出"这个目标需要哪个 interest"这种**条件化**的选择能力；
- 这解释了 Phase 0 核对出的现象——TASID 在 5-seed fair 对比下未超越 ASPCF（`THESIS_PROGRESS.md` §2.4）。

`L_disc` 直接改变了这一点：`z_t` 由 target-conditioned 的 `π` 加权而成，`L_disc` 的梯度同时推动 `V_k` 更有判别性、`π` 更准确。**这是一个机制层面的差异，不是超参或损失权重的调整。**

**因此 Ch5 的 go/no-go 判据是单一的**：`L_disc` 是否在 5-seed 下超过 ASPCF 与 CAISD。
- 若成立 → Ch5 是完整章节。
- 若不成立 → 剩下的只有 confidence 校准 + teacher 换表示，**判定为 TASID 小修，应合并进 Ch4 或降级为 Ch4 的一节**。

**建议**：把 `L_disc` 的最小验证（单 seed、仅 beauty、只有 `L_disc` 而无 confidence、无 coverage）提前到 Ch5 开工的第一天，用不到 1 小时的 GPU 时间换取"这一章是否成立"的答案。

---

### 问题 4：三章是否真正递进，而不是三个拼接模型？

**答：结构上是递进的（有真实的产物依赖链），但存在两个可能断开递进关系的具体工程风险，必须显式处理。**

**递进的真实证据（产物依赖链）**：

| 从 | 到 | 传递的产物 | 是否可绕过 |
|---|---|---|---|
| Ch3 | Ch4 | `cgscd_basis_r32.pkl` 定义的共享语义子空间 → prototype 的聚类空间 | 可绕过（用 PCA-512 prototype），但那样 Ch4 就与 Ch3 无关 |
| Ch3 | Ch5 | 共享语义空间 → `q_target` 的语义坐标 | 可绕过（用 PCA-512），但那样 Ch5 与 Ch3 无关 |
| Ch4 | Ch5 | `P_k` / `T_k`（interest 语义 profile） → target-conditioned teacher 分布 | **不可绕过**（Ch5 的 teacher 必须建立在此之上，否则无 target-interest 语义） |
| Ch4 | Ch5 | `w_k`（history-only 兴趣权重） → 与 `π` 对比，证明 target-conditioning 的必要性 | 可绕过但会削弱论证 |
| Ch5 | Ch4 | `L_disc` 的梯度回传到 `V_k` 与 `π`，间接要求 Ch4 的 `P_k` 具有目标可判别性 | 反馈关系，非硬依赖 |

**两个断点风险**：

**断点 1（Ch3 → Ch4）**：若 Ch3 的共享子空间太弱，prototype 质量会下降。
- 实测：`r=32` 的共享空间在 Beauty 上只捕获 **21.5%** 的 LLM 方差（`r=64` 为 26.6%）；在 ML-1M 上 `r=32` 为 33.3%。
- 也就是说 `cgscd_proto32_r32.pkl` 是在一个 32 维、仅含 1/5 LLM 方差的空间上做 KMeans。而现有的 `llmmi_proto32_sr512.pkl` 是在 512 维、含 78% 方差的空间上做的。
- **风险**：prototype 质量下降 → Ch4 的 teacher 变差 → Ch4 的收益归因困难（是 coverage 的问题还是 prototype 的问题？）。
- **缓解**：`THESIS_MASTER_PLAN.md` §5.2 的三档方案（A: PCA-512 / B: CGSCD-shared / C: student 表示）**同时作为消融**，而不是静默替换。特别地，必须提供 `r=64` 的档 B 变体，否则"维度减少"与"表示更好"混杂。

**断点 2（Ch3 → Ch5）**：现有 TASID 直接用 `semantic_branch(z_target[:, :512])`——PCA-512 切片。若 Ch5 沿用这个实现，Ch5 与 Ch3 完全脱钩，三章变成"Ch3 一章 + Ch4/Ch5 一章"。
- **缓解**：把"`q_target` 的语义坐标来源"作为 Ch5 的显式消融（`target_semantic_source ∈ {pca512, cgscd_shared}`），并把它写进方法描述。

**如何证明递进而非拼接（论文写作要求）**：

1. §5.1 的监督信号流图必须能在代码中逐条对应到具体的张量。
2. 每一章必须包含一个"**用上一章输出 vs 不用上一章输出**"的对照实验（即断点 1、2 的消融）。
3. `L_disc` 从 Ch5 回传到 `V_k` 与 `π` 的路径，是"Ch5 反过来约束 Ch4"的证据，可作为一个专门的分析小节。

**判定**：**当前规划下，递进关系是"设计上成立、工程上有两个断点需要显式处理"。** 若两个断点都用消融覆盖，则是真递进；若任一断点被静默绕过（直接复用 PCA-512 而不做对照），论文会被读成"三个模型拼接"。

---

### 问题 5：哪个 Chapter 当前创新风险最高？

**答：Chapter 4 风险最高，且高出另外两章一个量级。**

**逐项论证**：

| 风险维度 | Ch3 | Ch4 | Ch5 |
|---|---|---|---|
| 有 fair 正收益证据作为前提？ | ✅ ASPCF 5-seed 稳定超越 id/llm_replace | ❌ **无**。HSDIR 有结构证据无收益证据；CAISD 5-seed **未超** ASPCF | ⚠️ TASID 有单 seed 正收益，5-seed 无 |
| 核心机制有平凡解？ | 否 | ✅ **是**（见 Q2 的构造性论证） | 否（`L_disc` 有明确梯度路径） |
| 需要新的度量/验证方式？ | 部分（方差占比） | ✅ **是**（"覆盖"需要新度量，且新度量本身需要被论证有意义） | 否（直接看 HR@K/NDCG@K） |
| 依赖上一章产物？ | 否 | ✅ **是**（硬依赖 Ch3 basis） | 是（软依赖，可绕过） |
| 继承上一章的风险？ | 否 | ✅ **是**（继承 R1/R2/R3） | 部分 |
| 技术不确定性 | 中（共享信号弱，有退路） | **高** | 低（机制清晰、资产就绪） |
| 与已有失败教训的距离 | 远 | ✅ **近**——HSDIR 已经证明"大幅改善兴趣结构仍可不带来收益" | 中 |

**Ch4 风险最高的根因**：

1. **它是唯一一章在开工前就没有正收益前提的**。Ch3 可以拿 ASPCF 的 5-seed 数字当靶子；Ch5 可以拿 TASID 当起点。Ch4 的起点 CAISD 在 fair 对比下**输给**了 ASPCF——这意味着 Ch4 不是"在有效方法上继续改进"，而是"先要证明这个方向本身有效"。

2. **它的核心 loss 有平凡解**，需要额外设计 `L_focus` 来排除，而 `L_focus` 的方向与 HSDIR 的失败方向高度相似（都是让兴趣更"分化/尖锐"）。因此 Ch4 必须同时论证两件事：coverage 有效，且 focus 不会重演 HSDIR 的失败。

3. **它的"完整覆盖"主张缺少自然的评价指标**。HR@K/NDCG@K 是排名的度量，不直接反映"覆盖"。若 coverage 提升了但排名不动，Ch4 面临与 HSDIR 完全相同的困境：结构改善无法转化为收益。**这是 Ch4 最本质的风险。**

**缓解策略**：

| 策略 | 说明 |
|---|---|
| 前置验证 | 在实现完整模型前，先做一个"离线覆盖实验"：用现有 CAISD checkpoint 的 `T_k` 与 `w_k` 计算 `q_hat`，度量其与 `q_seq` 的覆盖质量，并检查**覆盖率与用户收益是否相关**。这是一个几十分钟的 batch 级实验，不需要训练。若覆盖率与收益无相关，Ch4 的核心假设就不成立 |
| 退路 1 | 把 Ch4 重新定位为"诊断章节"：系统回答"为什么兴趣结构改变不转化为收益"，并给出 coverage 作为解释工具。这弱化了方法贡献但保住了章节 |
| 退路 2 | 若 coverage 单独有效但 focus 有害，考虑用 prototype 维度的重加权替代 focus（不改变 `P_k` 形状，只调整 `q_hat` 的构造方式） |
| 顺序建议 | **先做 Ch3，用 Ch3 的共享空间重建 prototype 后，Ch4 的诊断基础更强** |

**次高风险：Chapter 3**（R1：共享信号弱——CF 视图对 LLM 的线性 R² 仅 4.1%/1.7%，非线性探针 9.1%）。但 Ch3 有明确退路：把"LLM 语义绝大部分是私有的"重新表述为本章的正面发现，方法退化为"学会分解"而非"利用强共享信号"。此外 Ch3 有一个与主方法成功与否无关的分析型贡献（PCA 高方差 ≠ 协同相关）。

**风险最低：Chapter 5**。机制清晰（`L_disc` 的梯度路径明确）、资产就绪（hard-negative bank 已存在且已验证）、起点明确（TASID 已实现）、判据单一（5-seed 下是否超越 ASPCF/CAISD）。

---

## 2. 工作量总表

| | Ch3 CGSCD | Ch4 ISPC | Ch5 TDIR | 合计 |
|---|---|---|---|---|
| 新方法模块数 | 6 | 4 | 3 | 13 |
| 新增 loss 数 | 3 | 2 | 1 | 6 |
| 新增离线工具 | 2 | 1 (+2 修改) | 2 (+1 扩展) | 5 |
| 主实验（CU） | 2 | 2 | 2 | 6 CU ≈ 42 h |
| 消融组数（CU） | 7 (14) | 6 (12) | 5 (10) | 18 组 / 36 CU ≈ 252 h |
| 参数实验组数 | 5 (10) | 2 (4) | 2 (4) | 9 组 / 18 CU ≈ 126 h |
| 诊断实验 | 3 | 4 | 4 | 11 |
| 可视化类别 | 4 | 4 | 3 | 11 |
| 编码工作量 | 2.5–3 周 | 2–2.5 周 | 2 周 | **6.5–7.5 周** |
| GPU 时间（可并行） | ~1.5 周 | ~1.5 周 | ~1 周 | **~4 周** |
| **章节总计** | **~3 周** | **~2.5 周** | **~2 周** | **~7.5 周** |

**注**：GPU 时间按"两块 GPU 并行、Beauty 与 ML-1M 各占一块"估算。若加入 Toys，主实验与消融的 GPU 时间至少翻倍，且需要第三块 GPU 才能保持并行度。

**对硕士论文工程的判断**：合计约 7.5 周编码 + 4 周 GPU（部分重叠），即 **约 2–2.5 个月的净工作量**，不含论文写作。这是一个合理但偏紧的硕士论文工程量。**若要压缩，压缩点应该是消融的总数（18 组 → 12 组，优先砍掉参数实验与"关掉单个 loss"的细粒度消融），而不是主实验的 seed 数。**

---

## 3. 建议的起步路径

**结论：首先实现 Chapter 3。** 依据如下（按重要性排序）：

### 3.1 三条硬性理由

**① 它是唯一的硬依赖源。**
`THESIS_ENGINEERING_MAP.md` 的复用率汇总明确指出：Ch4 的 prototype 需要 `cgscd_basis_r32.pkl`，而 Ch5 的语义坐标（按规划）也应来自 Ch3 的共享空间。**Ch4 的主实验在 Ch3 完成前无法启动。** 先做 Ch3 是唯一能解锁后续两章的路径。

**② 它的风险最需要前置发现。**
R1（共享信号弱：线性 R² 4.1%/1.7%，非线性 9.1%）是一个**全局性风险**——如果共享语义信号本质上很弱，Ch3 需要重新表述、Ch4 的 prototype 重建方案需要重新评估、Ch5 的 `q_target` 坐标来源也要重新考虑。**越早发现越好。** 而且这个风险的检验成本极低：

```
build_cgscd_basis.py  →  秒级/分钟级
```

跑完这个离线工具就能立刻得到：
- `M` 的奇异值谱与能量累积（Beauty 3 秒、ML-1M 1 秒即可得到，Phase 0 已实测）
- shared/private 的方差占比
- 与 PCA-512 人工切片的直接对比

**这是一个不需要训练、不需要 GPU、几分钟内就能给出"这个方向是否可行"信号的检验。** 应该在写任何模型代码之前先做。

**③ 它有最清晰的成功判据。**
ASPCF 的 5-seed 基线是完整且可复现的（Beauty NDCG@5 0.10746 ± 0.00136；ML-1M 0.21402 ± 0.00234）。Ch3 的成功判据是"在 5-seed 下超过这两个数字"，边界清晰、无歧义。相比之下 Ch4 的判据（"覆盖是否改善了什么"）本身就模糊。

### 3.2 两条附加理由

**④ 复用率与投入产出比合适。**
Ch3 的 `L_relation` 重写、`ItemEncoder` 新 mode、gate 复用——都建立在已有的、稳定运行的代码上。而 Ch4 从 CAISD 复制骨架（一个已被证明 5-seed 未超基线的模型）作为起点，心理与技术上都更被动。

**⑤ 附带的分析型贡献可独立成立。**
即使 CGSCD 主方法未能超越 ASPCF，Ch3 仍有一个与成败无关的贡献：**"按 PCA 方差人工定义 semantic/complement 是不合理的"**——Phase 0 已用实测数据证明（前 512 维占 78% 方差但仅 5% 可由 CF 解释；后 1024 维更低至 0.5% / 0.4%）。这是论文中一节可以独立成立的分析。

### 3.3 建议的最小起步路径（Phase 1 Day 1–3）

| 步骤 | 内容 | 成本 | 产出 |
|---|---|---|---|
| 1 | 写 `tools/build_cgscd_basis.py` 并跑三个数据集 | ~1 天，无 GPU | 奇异值谱、shared/private 方差占比、sh a1 溯源。**这一步就能判定 R1 是否需要重新表述** |
| 2 | 修 `tools/build_llmmi_semantic_prototypes.py` 支持 `--input_rep cgscd_shared`，生成 `cgscd_proto32_r32.pkl` | ~0.5 天 | 检查在 32 维共享空间上 KMeans 的簇质量（与 PCA-512 版对比：簇大小分布、assignment 熵） |
| 3 | 写离线覆盖诊断：用现有 CAISD checkpoint 计算 `q_seq` / `q_hat` 的覆盖率，并检查其与用户收益的相关性 | ~1 天，1 次 batch | **判定 Ch4 的核心假设是否成立**（不需要训练） |
| 4 | 基于步骤 1–3 的结论，修订 `THESIS_MASTER_PLAN.md` 的 R1/R4，然后开始 Ch3 模型实现 | — | 有数据支撑的设计决策 |

**步骤 1–3 合计约 2.5 天，几乎不需要 GPU，却能在写任何模型代码之前回答三个最关键的不确定性问题**（Ch3 的共享信号强度、Ch4 的 prototype 质量、Ch4 的覆盖假设）。这是本评估中最强烈的一条建议。

### 3.4 不推荐的做法

- ❌ **先做 Ch5**：虽然它的工程量最小、资产最就绪，但它依赖 Ch4 的 `P_k`。先做 Ch5 会导致 teacher 只能退回 PCA-512（断点 2），三章的递进关系在第一周就断掉。
- ❌ **三章并行**：Ch4 与 Ch5 都依赖 Ch3 的产物，并行会造成接口反复变更。
- ❌ **先补 HSDIR/CASIR 的多 seed 实验**：它们已被冻结为探索实验。补实验不会改变结论的方向（HSDIR 的结构效应已足够强），只会消耗 GPU 预算。

---

## 4. 本阶段结束语

Phase 0 的四份文档完成了研究冻结与工程分解。最重要的三条发现是：

1. **现有 Ch4 方法（CAISD/TASID）在 5-seed fair 对比下未超越 ASPCF**（`THESIS_PROGRESS.md` §2.4）。这不是坏消息——它正是重新定义 Chapter 4 问题（从"分化"到"完整语义组织"）与 Chapter 5 问题（让 target 知识进入打分）的直接依据。

2. **ASPCF 的 PCA 切片假设已被实测否证**（`THESIS_MASTER_PLAN.md` §2.1）：前 512 维占 78% 方差但仅 5% 可由 CF 解释。这为 Chapter 3 提供了坚实且成本极低的 motivation。

3. **Chapter 4 是风险最高的章节**——它是唯一一章开工前没有正收益前提、核心 loss 有平凡解、且缺少自然评价指标的章节。建议在实现前先用 ~2.5 天的离线诊断验证其核心假设。

**Phase 0 到此停止。不自动进入下一阶段。**

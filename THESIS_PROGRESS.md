# THESIS_PROGRESS.md

> THESIS PHASE 0 — Research Freeze & Engineering Decomposition
> 建立日期：2026-10-03
>
> **本文件记录研究进度的客观状态。**
> 不改写 `PHASE1_SUMMARY.md` / `PHASE2_SUMMARY.md` 的历史结论——它们记录的是当时的判断。
> 本文件补充的是 Phase 0 期间**从磁盘上实际存在的文件与日志中重新核对**出的事实。
> 两者若有冲突，以本文件的实测记录为准，并在 §5 中明确标注差异。

---

## 1. 研究状态保护（Phase 0 第一步，已完成）

| 项 | 值 |
|---|---|
| 分支 | `main` |
| HEAD commit | `bf008e2de7de3ecd771fddf52f7ae7fb7d85b3ca` — "第一轮测试" |
| 上游 | `origin/main`，**无领先/落后**（working tree clean） |
| 未提交文件 | **无**（`git status` 报告"无文件要提交，干净的工作区"） |
| stash | **空** |
| HEAD commit 内容 | `PHASE2_SUMMARY.md` (+232) 与 `README.md` 重写 (+443/−523) |

**结论：无需保护性操作。** 没有未提交代码需要报告，没有被 reset/清理的风险，没有 stash 需要保留。

Phase 0 期间**未执行**任何写操作到既有实验资产：未删除、未覆盖、未移动任何 pkl / checkpoint / log / summary / 代码文件。新增的只有本阶段的 4 个 markdown 文档。

### 1.1 受版本控制与不受版本控制的资产

`.gitignore` 排除了：`data/`、`model/`、`new_model/`、`log/`、`logs/`、`new_log/`、`*.pt`、`*.pkl`、`*.log`、`*.out`、`__pycache__/`、`.claude/`。

因此：

- **受版本控制**：全部 `.py` 源码、`new_bash/*.sh`、`bash脚本/*.sh`、`README.md`、`PHASE1_SUMMARY.md`、`PHASE2_SUMMARY.md`、`AUDIT_REPORT.md`、`baseline.md`、`CLAUDE.md`、`analysis_figures/`（figures 被 gitignore 中的 `analysis_figures/` 条目排除，但 scripts 的追踪状态需按实际情况）
- **不受版本控制（仅存于磁盘）**：**所有实验结果、所有 checkpoint、所有预计算 pkl、所有日志**

**这意味着：实验资产没有 git 层面的保护。** 任何清理动作都会造成不可恢复的损失。Phase 0 的"不删除"约束因此是硬约束。**建议在未来阶段为 `data/*/handled/` 与 `new_log/` 建立外部备份或至少一份 manifest（文件名 + sha1 + 生成脚本）。**

---

## 2. 已完成且可复现的实验

以下结论基于磁盘上实际存在的 `summary.tsv` 与日志，逐行核对。

### 2.1 Phase 0 — 干净基线 LLMMIRec（三数据集，seed=42）

| 数据集 | id | llm_replace | residual |
|---|---|---|---|
| Beauty | 0.0832 | **0.1016** | 0.0946 |
| ML-1M | 0.2130 | 0.1795 | **0.2131** |
| Toys | 0.1187 | **0.1433** | 未完成 |

（NDCG@5）

**结论保留**：LLM 嵌入替换 CF 嵌入在 Beauty / Toys 上显著有效；在 ML-1M 上无效甚至有害；residual 融合在 ML-1M 上稳定。这三条结论方向不一致，本身是一个未充分解释的观察。

### 2.2 Chapter 3 — ASPCF（fair 5-seed，**COMPLETE**）

`new_log/llmmirec_aspcf_phase2/<ds>/summary.tsv`
注：每个文件含 10 行幻影记录（`ModuleNotFoundError: No module named 'torch'`，`total_seconds` 0–2，却记为 `status=OK`），**不计入**。

| 数据集 | 配置 | HR@5 mean±std | NDCG@5 mean±std |
|---|---|---|---|
| Beauty | lr=0.004, batch=1024, λ_rel=0.01 | 0.15816 ± 0.00109 | 0.10746 ± 0.00136 |
| ML-1M | lr=0.001, batch=1024, λ_rel=0.01 | 0.30766 ± 0.00249 | 0.21402 ± 0.00234 |

逐 seed NDCG@5：
- Beauty: s0 0.1089 / s1 0.1059 / s2 0.1079 / s41 0.1058 / s42 0.1088
- ML-1M: s0 0.2133 / s1 0.2185 / s2 0.2133 / s41 0.2116 / s42 0.2134

### 2.3 Chapter 4 — CAISD（fair 5-seed，**COMPLETE**）

`new_log/llmmirec_caisd_phase2/<ds>/summary.tsv`

| 数据集 | 配置 | HR@5 mean±std | NDCG@5 mean±std |
|---|---|---|---|
| Beauty | lr=0.004, λ_sem=0.01, teacher=responsibility, distill=uniform | 0.15816 ± 0.00219 | 0.10738 ± 0.00202 |
| ML-1M | lr=0.001, λ_sem=0.005, teacher=responsibility, distill=uniform | 0.30268 ± 0.00435 | 0.21042 ± 0.00204 |

### 2.4 ⚠️ Phase 0 核对出的关键事实：CAISD 未超越 ASPCF

`new_log/llmmirec_caisd_phase2/<ds>/multi_seed_comparison.json` 的 `relative_improvement_percent`：

**Beauty**（CAISD vs ASPCF）：
```
HR@5   : +0.0000 %      HR@10 : +0.0264 %      HR@20 : -0.4219 %
NDCG@5 : -0.0744 %      NDCG@10: -0.0617 %     NDCG@20: -0.3040 %
```

**ML-1M**（CAISD vs ASPCF）：
```
HR@5   : -1.6187 %      HR@10 : -0.6052 %      HR@20 : +0.3515 %
NDCG@5 : -1.6821 %      NDCG@10: -1.1319 %     NDCG@20: -0.6053 %
```

**结论：在 per-seed 对齐（5 seeds: 0/1/2/41/42）的公平对比下，CAISD 在 Beauty 上打平、在 ML-1M 上落后 ASPCF（6 个指标中 5 个下降）。**

这与 `PHASE2_SUMMARY.md` 中"TASID 稳定提升"的表述存在张力。差异的来源已在 Phase 0 定位清楚，见 §5。

**这不是失败，而是 Chapter 4 重新定义问题定义（从"分化"转向"完整语义组织"）的直接依据。** 它也解释了为什么 `THESIS_MASTER_PLAN.md` 中 Chapter 5 必须让 target-interest 知识进入打分路径。

### 2.5 探索性实验（单 seed=42，**PARTIAL**）

**HSDIR**（`new_log/llmmirec_hsdir_phase1/<ds>/summary_seed42.tsv`）：
- Beauty 13 行，其中 2 行全零（HR@5/NDCG@5 = 0.0000，训练早期即崩溃）、1 行 23 秒崩溃、1 行与另一行重复（并发竞态）。
- 有效行最佳：`hierarchical λ=0.01 → HR@5 0.1626 / NDCG@5 0.1098`；`fine λ=0.01 → 0.1601 / 0.1090`；`coarse λ=0.01 → 0.1567 / 0.1065`；`λ=0 → 0.1560 / 0.1053`。
- 与 ASPCF seed-42（HR@5 0.1592 / NDCG@5 0.1088）相比，`hierarchical` 略有提升，**但均为单 seed，且 `λ=0` 的对照本身也波动到 0.1560–0.1560，无法排除噪声**。
- ML-1M 13 行，含 1 行崩溃。
- **诊断结论（5 项结构性指标，可保留）**：兴趣间余弦 Beauty 0.9457→0.8691、ML-1M 0.7620→0.7377；有效秩 Beauty 1.714→2.280、ML-1M 2.762→3.028；route membership entropy 0.913→0.543；effective active K 3.833→2.632；route 与 fine/coarse 语义相关性 ≈0 → 0.36/0.38。
- **`summary_seed42.tsv` 无 `hsr_loss_mode` / `hsr_margin` / `hsr_pair_margin` / `hsr_confidence_mode` / `hsr_route_source` 列**，13 行中多行不可区分。这是一条需要修复的记录缺陷。

**CASIR**（`new_log/llmmirec_casir_phase1/<ds>/summary_seed42.tsv`）：
- 两个数据集各只有 2 行：`semantic_add` 与 `complement_coherence`。文档化了 4 个 mode，`none` 与 `complement` **从未运行**。
- Beauty：`semantic_add → 0.1537 / 0.1057`，`complement_coherence → 0.1486 / 0.1017`（均低于 ASPCF 0.1592 / 0.1088）。
- ML-1M：`semantic_add → 0.3013 / 0.2095`，`complement_coherence → 0.3088 / 0.2141`（vs ASPCF 0.3068 / 0.2134）。

**TASID sensitivity sweep**（`new_bash/run_llmmirec_caisd_tasid_sensitivity.sh`）：
- **从未运行**。目标目录 `new_log/llmmirec_caisd_tasid_sweep/` 与 `new_model/llmmirec_caisd_tasid_sweep/` **不存在**。
- 脚本实现与配套汇总工具 `tools/summarize_llmmirec_caisd_tasid_ablation.py` 均已就绪，随时可跑。
- TASID 现有的全部证据来自 ad-hoc `nohup` 运行（beauty seed 42，λ=0.01 τ=0.1）。

### 2.6 结果完整性总表

| 实验 | 状态 | 说明 |
|---|---|---|
| Phase 0 三数据集 × 3 encoder | ✅ 完成（Toys residual 除外） | seed 42 |
| ASPCF Phase 2 × 2 数据集 × 5 seeds | ✅ **完成** | fair 对比基线 |
| CAISD Phase 2 × 2 数据集 × 5 seeds | ✅ **完成** | fair 对比，**未超 ASPCF** |
| TASID λ/τ 敏感性扫描 | ❌ **未运行** | 脚本就绪 |
| TASID ad-hoc（beauty seed42） | ⚠️ 仅有 nohup 输出 | 无独立 summary 列，无法系统汇总 |
| HSDIR phase1 × 2 数据集 | ⚠️ **部分**（单 seed，含崩溃行） | 结构诊断完整 |
| CASIR phase1 × 2 数据集 | ⚠️ **部分**（4 mode 中只跑了 2 个） | 单 seed |
| CHIR phase2a–2e | ⚠️ 单 seed 探索 | 保留为探索记录 |

---

## 3. 当前研究结论（分层）

### 3.1 成立的结论（有 fair 多 seed 或强结构性证据）

1. **ASPCF 是当前最强的可复现基线**（Beauty / ML-1M，5 seeds，std ≈ 0.002）。
2. **LLM 语义 teacher 能系统性地改变兴趣结构**——HSDIR 的 5 项结构性指标在 Beauty 与 ML-1M 上方向一致，效应量远大于 seed 噪声。
3. **LLM 语义知识在 item 表示层有效**——Phase 0 的 llm_replace 在 Beauty/Toys 上大幅超越 id。
4. **ASPCF 的 relation preservation loss 有效**——Phase 1 Beauty 单 seed：λ=0.01 使 NDCG@5 从 0.1003 提升到 0.1100。

### 3.2 不成立的结论（Phase 0 修正）

1. ❌ **"CAISD/TASID 稳定提升推荐指标"** —— 5-seed fair 对比显示未超越 ASPCF（§2.4）。
2. ❌ **"兴趣分化可以转化为推荐收益"** —— HSDIR 显著改变了结构，但单 seed 收益在噪声内。
3. ❌ **"用户级因素可以预测 teacher 收益"** —— `teacher_js_shift` 跨数据集方向不一致（beauty ρ=0.0105 p=0.2336；ml-1m ρ=0.0281 p=0.0292），不可作为 gate。
4. ❌ **"CASIR 的语义残差注入有效"** —— Beauty 上两个 mode 均低于 ASPCF 基线。

### 3.3 未解释的观察（可作为论文的分析点）

1. **LLM 嵌入的效果高度数据集相关**：Beauty/Toys 上 llm_replace 大幅提升，ML-1M 上大幅下降（0.2130 → 0.1795）。
2. **跨视图共享信号很弱且数据集相关**（Phase 0 新增实测）：用 PoMRec CF 视图线性预测 LLM 的 R² 仅 4.1%（beauty）/ 1.7%（ml-1m）；非线性 MLP 探针 9.1%（beauty）。而换用交互矩阵 SVD128 视图后，beauty 降到 1.6%、ml-1m 升到 24.8%——**方向相反**。
3. **PCA 高方差方向与协同信号弱关联**：前 512 维占 78% 方差但仅 5% 可由 CF 解释（beauty）；`z[:, 512:1024]` 与 `z[:, 1024:]` 更低（0.5% / 0.4%）。

### 3.4 Phase 0 核心教训（延续并强化）

1. **结构改变 ≠ 收益**（PHASE1/2 已提出，Phase 0 用 fair 5-seed 证据强化）。
2. **target 相关性是关键**（PHASE2 提出）——但 Phase 0 进一步指出：现有 TASID 的 target 知识**从未进入打分路径**（`prediction` 计算只依赖 `Σ w_k V_k`，`w_k` 是 history-only），这解释了为什么收益不稳定。
3. **单 seed 结论不可靠**——本仓库的多个"有效"结论在 5-seed 下消失或反转。后续章节的最低证据标准是 5 seeds。

---

## 4. 冻结声明

自 2026-10-03 起，以下内容**冻结**，不作为后续优化的对象：

| 冻结项 | 状态 | 说明 |
|---|---|---|
| ASPCF（`LLMMIRecASPCF`） | ✅ 冻结为 Ch3 baseline | 5-seed 数据完整。不再调参、不再改结构 |
| CAISD（`LLMMIRecCAISD`） | ✅ 冻结为 Ch4/Ch5 baseline | 5-seed 数据完整。保留为对照，不再作为主方法 |
| HSDIR / CHIR / CASIR | ✅ 冻结为探索实验 | 保留代码与单 seed 结果，作为论文的 motivation / 失败分析素材 |
| Evaluation protocol | ✅ 冻结 | Phase 0 明确不改 |
| 数据 split | ✅ 冻结 | `data/*/{train,dev,test}.csv` 不改 |
| `helpers/BaseRunner.py` | ✅ 冻结 | 训练/评测循环不改 |

**冻结不等于删除。** 所有被冻结的代码、模型、日志、summary 均原样保留。

---

## 5. 与既有阶段总结的差异记录

`PHASE2_SUMMARY.md` 记载：

> ML-1M: CAISD baseline 0.3040/0.2104 → TASID asymmetric **0.3093/0.2161** (+1.7% / +2.7%)

Phase 0 核对出的情况：

| 项 | 说明 |
|---|---|
| 数据来源 | 该数字来自 **seed=42 单次 ad-hoc 运行**，记录在 `new_log/llmmirec_caisd_phase2/beauty|ml-1m/summary.tsv` 的额外行（2026-08-20 追加）与 `tasid_*.nohup.out` |
| 对照数字 | 用作对照的 `0.3040` 是 **ASPCF** 的 seed-41 值（`new_log/llmmirec_aspcf_phase2/ml-1m/summary.tsv`），不是同 seed 的 CAISD |
| 同 seed 的真实对照 | CAISD seed-42 = 0.3094/0.2135；ASPCF seed-42 = 0.3068/0.2134。TASID asymmetric seed-42 = 0.3093/0.2161（NDCG@5 +1.3% over CAISD，但只有 1 个 seed） |
| 5-seed fair 对比 | CAISD 均值 0.30268/0.21042 **低于** ASPCF 0.30766/0.21402 |
| **判定** | `PHASE2_SUMMARY.md` 的 "+1.7%/+2.7%" 是**跨 seed 混合对照 + 单 seed** 得到的，不能支撑"稳定提升"的表述 |

**Phase 0 明确不修改 `PHASE2_SUMMARY.md`。** 本节仅记录差异，供论文写作时决定如何表述。推荐做法：在论文中把 CAISD/TASID 作为 Chapter 4/5 的 **baseline 与 motivation**（"静态/非打分参与的 target 监督不足"），而不是作为已验证的 contribution。

其他次要差异：
- `CLAUDE.md` 提到 `--srs_emb_path ./data/beauty/srs_emb.pkl` 与 `--llm_emb_path ./data/beauty/llm_emb.pkl`，**这两个文件不存在**。所有历史脚本的 `--srs_emb_path` 实际指向 `data/<ds>/handled/itm_emb_pomrec.pkl`。CLAUDE.md 该处为过时描述。
- `PHASE1_SUMMARY.md` 的 "ASPCF 0.1053/0.1269" 与 "hierarchical 0.1091/0.1317" 用的是 batch_size=2048 / lr=0.008 的配置，与 Phase 2 fair 配置（batch=1024 / lr=0.004 or 0.001）不同，两组数字不可直接比较。

---

## 6. Phase 追踪表

| Phase | 名称 | 状态 | 产物 |
|---|---|---|---|
| Phase 0 | 干净基线 LLMMIRec | ✅ 完成 | `LLMMIRec.py`, `llmmi_components.py`, `llmmi_utils.py` |
| Phase 1 | ASPCF（Ch3 旧版） | ✅ 完成 | `LLMMIRecASPCF.py`, `build_llmmi_semantic_prototypes.py` |
| Phase 2 | HSDIR / 用户 gate / pair selective / CAISD / TASID | ✅ 完成（探索） | `LLMMIRecHSDIR.py`, `LLMMIRecCAISD.py`, 分析工具 |
| Phase 2 补充 | CHIR / CASIR | ⚠️ 部分 | `LLMMIRecCHIR.py`, `LLMMIRecCASIR.py` |
| **THESIS PHASE 0** | **Research Freeze & Engineering Decomposition** | ✅ **本阶段** | `THESIS_MASTER_PLAN.md`, `THESIS_PROGRESS.md`, `THESIS_ENGINEERING_MAP.md`, `THESIS_WORKLOAD_REVIEW.md` |
| THESIS PHASE 1 | Chapter 3 实现（CGSCD） | ⏸ 未开始 | — |
| THESIS PHASE 2 | Chapter 4 实现（ISPC） | ⏸ 未开始 | — |
| THESIS PHASE 3 | Chapter 5 实现（TDIR） | ⏸ 未开始 | — |

**本阶段到此停止。不自动进入 THESIS PHASE 1。**

---

## 7. 进入下一阶段前必须完成的准备工作

（这些不是 Phase 0 的任务，而是 Phase 1 开工前的 checklist）

1. **confidence 熵探针**：用 1 个 batch 测真实 `p_teacher` 在 K=4 上的熵分布，判定 entropy-based confidence 是否可用（见 `THESIS_MASTER_PLAN.md` R5）。
2. **CGSCD basis 来源锁定**：`build_cgscd_basis.py` 必须记录输入 pkl 的 sha1（`itm_emb_pomrec.pkl` 在 toys 上存在多个不匹配的候选来源）。
3. **修复评测脚本的基础设施缺陷**：新脚本加 `python -c "import torch"` 前置检查 + `set -o pipefail`，避免失败被记为 `status=OK`。
4. **新 summary 表为每个被扫超参建列**，避免 HSDIR 那样的不可区分行。
5. **建立预计算资产的 manifest**（文件名 + sha1 + 生成命令），因为 `data/` 与 `new_log/` 不受版本控制。
6. **决定 Chapter 4 的 prototype 重建方案**：是否接受"Ch4 主实验硬依赖 Ch3 完成"这一顺序约束，或先用 PCA-512 prototype 做 Ch4 的早期验证。

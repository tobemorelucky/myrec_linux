# THESIS_ENGINEERING_MAP.md

> THESIS PHASE 0 — 代码资产审计
> 建立日期：2026-10-03 · 基线 commit `bf008e2`
>
> 本文档回答三个 Chapter 各自：**A 可原样复用 / B 轻微修改复用 / C 应淘汰（仅保留为探索实验）/ D 必须新实现 / E 依赖的预计算数据 / F 新增代码文件 / G 预计实验脚本**。
>
> 审计基于完整阅读：`LLMMIRec.py`(456) `LLMMIRecASPCF.py`(268) `LLMMIRecHSDIR.py`(593) `LLMMIRecCAISD.py`(509) `LLMMIRecCHIR.py`(436) `LLMMIRecCASIR.py`(345) `llmmi_components.py`(516) `llmmi_utils.py`(164)，以及 `tools/build_llmmi*.py`、`tools/analyze_llmmi*.py`、`tools/summarize_*.py`、`new_bash/run_llmmirec*.sh`、`scripts/build_semantic_hardneg.py`。
> 代码行号引用以本 commit 为准。

---

## 0. 全局资产总览（三章共享）

### 0.1 可原样复用（不区分章节）

| 模块 | 位置 | 说明 |
|---|---|---|
| `BaseModel` / `GeneralModel` / `SequentialModel` | `models/BaseModel.py` | BPR loss、负采样、`item_num`/`user_num`、`parse_model_args` 链 |
| `SeqReader` | `helpers/SeqReader.py` | 历史序列构造（按时间排序 + position） |
| `BaseReader` | `helpers/BaseReader.py` | `n_items = item_id.max() + 1`（含 padding row 0） |
| `BaseRunner` | `helpers/BaseRunner.py` | 训练循环、HR@K / NDCG@K、NDCG@5 early stop。**Phase 0 明确不改** |
| `utils/utils.py` | — | seed 初始化、GPU 搬运、metric 格式化 |
| `main.py` | — | `--model_name` 动态解析模型类；新增模型只需加一行 import |
| `load_llm_table` | `llmmi_utils.py:22-106` | 严格 shape 校验（只接受 `n_items` 或 `n_items-1` 行）+ NaN/Inf 检查 + 自动补 padding row 0 |
| `check_nan_inf` | `llmmi_utils.py:113-137` | 首 batch 诊断 |
| `get_activation` | `llmmi_utils.py:144-164` | — |
| `QueryMultiInterestExtractor` | `llmmi_components.py:250-345` | K 个 query + scaled dot-product attention；已支持 `external_query` / `attention_prior` / `return_route_scores` 三个可选钩子 |
| `InterestAggregator` | `llmmi_components.py:352-409` | **history-only** 兴趣权重（mean + last → LayerNorm → MLP → softmax）。Ch4 的 `w_k` 来源 |
| ASPCF gate 结构 | `llmmi_components.py:139-151, 216-229` | `basic` / `conflict` 两种 mode，softmax 双路权重 + `√α ·` 加权拼接 |
| 位置编码 + dropout + 聚合 + 打分路径 | 各 model 的 `forward` step 2–7 | 三章完全一致：`history_emb_pos → extractor → aggregator → user_vector → dot(candidate)` |
| NaN/Inf 首 batch 检查 | 各 model step 9 | 直接照搬 |

### 0.2 已知系统性技术债（Phase 0 记录，不在本阶段修复）

| 债 | 证据 | 对新章节的影响 |
|---|---|---|
| 模型文件间逐行复制 | `_compute_relation_loss` 在 `LLMMIRec.py:413`、`LLMMIRecASPCF.py:250`、`LLMMIRecCAISD.py:491`、`LLMMIRecHSDIR.py:422` **逐字节相同**（19 行 × 4 处）。`LLMMIRecCASIR` ≈ 55-60% 逐字复制自 CAISD；`LLMMIRecCHIR` ≈ 35% 复制自 ASPCF | 新章节**必须**抽公共 mixin，否则 5 个模型 × 19 行会变成 9 个模型 |
| ASPCF 参数块在 4 个文件重复 | `parse_model_args` 中 ASPCF 段（`semantic_rank`…`aspcf_gate_mode`）4 处相同 | 同上 |
| 未校验的 mode 耦合 | `CHIR`: `prototype → 必须 aspcf`（`LLMMIRecCHIR.py:276,279`）、`dual → 必须 prototype`；`CASIR`: `refine != none → 必须 aspcf`（`LLMMIRecCASIR.py:232`）。均以 `TypeError`/`UnboundLocalError` 在 forward 内崩溃，而非构造期 `ValueError` | 新模型必须做构造期校验 |
| `residual` 模式被排除在 relation loss 外 | `LLMMIRec.py:355` 的 guard 只含 `("aspcf","llm_replace")`，`residual` 有 `llm_table` 也有 `adapter` 却被排除 | 公共实现应统一 |
| `return_intermediate` 训练期语义 | CAISD 把 `_sem_*` / `_tasid_*` 键在 `self.training` 时才写入，且在 `return_intermediate` 输出时过滤掉 `_` 前缀键（`LLMMIRecCAISD.py:425-431, 452-455`）。TASID 块本身 gate 在 `self.training`（`:316`） | 诊断工具无法从 checkpoint 在 eval 下复现 TASID 分布 |
| 诊断工具硬编码架构 | `tools/analyze_llmmirec_caisd.py:86-89` 从 state_dict 推断 `emb_size/K/attn_size`，其余（`history_max=20, semantic_rank=512, semantic_dim=32, aspcf_gate_mode=basic, semantic_distill_mode=uniform`）全部硬编码；用 `load_state_dict(strict=False)` 静默接受缺键 | 换架构后诊断数值静默失真 |
| 实验记录表缺列 | HSDIR `summary_seed42.tsv` 无 `hsr_loss_mode`/`hsr_margin`/`hsr_pair_margin`/`hsr_confidence_mode`/`hsr_route_source` 列 → 13 行中多行不可区分 | 新 summary 表必须为每个被扫超参建列 |
| 脚本失败被记为成功 | ASPCF/CAISD `summary.tsv` 各有 10 行 `ModuleNotFoundError: No module named 'torch'`，`total_seconds` 0–2，`status=OK`。根因：脚本无 conda 激活，且 `python … \| tee` 的退出码是 `tee` 的 | 新脚本加前置 `python -c "import torch"` + `set -o pipefail` |
| 重复行无去重 | `summary.tsv` 以 `>>` 追加，无幂等保护；已有重复行 | — |

---

## Chapter 3 — Item Representation (CGSCD)

方法：Collaborative-Guided Shared–Complementary Semantic Decomposition。
`Z`(frozen LLM) + `C`(frozen CF) → 跨视图分解 → shared/complement 双分支 + 自适应融合 + 语义关系/冗余约束。

### A. 可以原样复用的模块

| 模块 | 位置 | 复用理由 |
|---|---|---|
| `load_llm_table` | `llmmi_utils.py:22` | `Z` 的加载与校验完全适用 |
| `get_activation` | `llmmi_utils.py:144` | — |
| `check_nan_inf` | `llmmi_utils.py:113` | 新分支同样需要 |
| gate 结构（`basic` mode） | `llmmi_components.py:139-151` | `gate_in_dim = shared_dim + compl_dim`，结构与 ASPCF 完全同构 |
| `√α` 加权拼接 | `llmmi_components.py:224-229` | 逐行照搬 |
| padding 归零逻辑 | `llmmi_components.py:194-196, 231-233` | item 0 必须为零向量 |
| `return_components` 契约 | `llmmi_components.py:235-242` | 保持相同的 dict 键名，下游诊断工具可直接复用 |
| `QueryMultiInterestExtractor` | `llmmi_components.py:250` | 与 item encoder 解耦，无需改动 |
| `InterestAggregator` | `llmmi_components.py:352` | 同上 |
| `position_emb` + 位置编码公式 | 各 model step 2 | — |
| 打分路径 step 5–7 | 各 model | `user_vector = Σ w_k V_k`；`prediction = dot(user_vector, candidate_emb)` |
| `main.py` 模型注册机制 | `main.py` | 加一行 import 即可 |

### B. 可以轻微修改后复用的模块

| 模块 | 位置 | 改动内容 | 预计改动量 |
|---|---|---|---|
| `ItemEncoder` 新增 `cgscd` mode | `llmmi_components.py:23-243` | 增加 mode 校验项；`__init__` 增加 `shared_basis`/`z_mean`/`c_mean`/`cf_table`/`shared_hidden`/`shared_dim`/`compl_hidden`/`compl_dim`/`cgscd_use_cf_input` 参数；新增 `_forward_cgscd`（与 `_forward_aspcf` 同构，仅 `z_sh`/`z_pv` 构造不同）；`forward` 增加 dispatch 分支 | +90 ~ 120 行 |
| relation preservation loss 推广 | `LLMMIRec.py:413-456` (`_compute_relation_loss`) | teacher 从 `z[:, :semantic_rank]` 改为 `Z_shared`（shared 侧）与 `Z_private`（private 侧）两组 item-item 余弦关系 KL；增加跨侧冗余项 | 重写，+50 ~ 70 行 |
| `LLMMIRecASPCF.py` 作为骨架 | `LLMMIRecASPCF.py` 全体 | 复制为新 `LLMMIRecCGSCD.py` 后改 encoder 参数与 loss。**注意**：这是复制而非继承，会加剧技术债。**建议**：新建文件时顺手抽出 `_ASP CFLikeModelMixin`（ASPCF 参数块 + relation stash + NaN 检查 + `return_intermediate` 输出），Ch3/4/5 三个新模型共同使用 | 新文件 ~300 行，其中 mixin 可省 ~110 行 × 3 |
| `build_llmmi_semantic_prototypes.py` | `tools/` | 见 Chapter 4 §B（同一改动同时服务两章） | ~30 行 |
| `new_bash/run_llmmirec_aspcf_phase2_*.sh` | `new_bash/` | 复制为 CGSCD 版脚本，替换 `--model_name` / `--llm_emb_path` / 新增 `--cgscd_*` 参数 / 输出目录 | 每数据集 ~100 行 |

### C. 应淘汰 / 仅保留为探索实验

| 模块 | 处置 | 理由 |
|---|---|---|
| ASPCF 的 `z_high = z[:, :512]` / `z_low = z[:, 512:]` 人工切片 | **保留为 baseline，不再作为主方法** | Phase 0 实测否证了其合理性：前 512 维占 78% 方差但仅 5% 可由 CF 解释；`z[:, 512:1024]`、`z[:, 1024:]` 的 CF 可解释性更低（0.5% / 0.4%），说明"尾部 = 互补"的假设也不成立 |
| `LLMMIRecCHIR` | 保留，不进入 Ch3 | 属于 Ch4 的探索（prototype query + dual-view routing），且无额外 loss、完全靠 ranking loss 训练 |
| `LLMMIRecCASIR` | 保留，不进入 Ch3 | 它修改了打分路径（`V_refined` 进入 prediction），与 Ch3 的"只改 item 表示"层级不同 |
| `tools/analyze_llmmirec_aspcf.py`（53 KB） | 保留，可有限复用 | 它诊断 ASPCF/CHIR 的 alpha 分布与 sem/comp 余弦，CGSCD 的 gate 结构相同，部分指标可直接用 |

**不淘汰任何文件**——Phase 0 禁止删除。

### D. 必须新实现的模块

| 模块 | 说明 |
|---|---|
| `build_cgscd_basis.py` | 离线计算 `M = ZᵀC/N` → SVD → 保存 `U_r`、`Σ`、`μ_z`、`μ_c`、诊断量（`Σσ²` 累积能量、shared 方差占比、CF 可预测性 R²）。带 sha1 溯源 |
| `_forward_cgscd` | 模型内的分解：`z_sh = z_c @ U_r`，`z_pv = z_c - z_sh @ U_rᵀ` |
| `f_shared` / `f_compl` | 两个分支的 MLP（结构可复用 ASPCF 的 `semantic_branch` / `complement_mlp` 形状） |
| **跨侧冗余惩罚** | `s` 与 `c` 不应重复编码同一信息。候选形式：对 batch 内 item 计算 `s` 与 `c` 的 item-item 关系矩阵的相关性上界；或对 `[s;c]` 做去相关（如 Barlow-Twins 式的 off-diagonal 惩罚） |
| **双教师关系约束** | shared teacher = `Z_shared` 的关系图；private teacher = `Z_private` 的关系图；分别对 `s`、`c` 做 KL —— 这是对现有单一 `L_relation` 的推广 |
| `cgscd_basis_mode=learnable` 消融 | `U_r` 作为 `nn.Parameter`（SVD 初始化），验证"SVD 分解是否必要" |
| 协同视图对照（可选但建议） | `C_svd = SVD128(train.csv 交互矩阵)`，用于回答"共享信号弱是 PoMRec 的问题还是本质如此"。Phase 0 已实测该对照在 beautiful/ml-1m 上给出**相反**结论（1.6% vs 24.8%），因此这不是可选项，而是必要分析 |

### E. 依赖的预计算数据

| 数据 | 路径 | 状态 | 生成方式 |
|---|---|---|---|
| LLM 语义表 | `data/<ds>/handled/llm_table_pca1536.pkl` (n_items, 1536) | ✅ 已有 | 外部生成（4096→1536 PCA） |
| 协同 item 表示 `C` | `data/<ds>/handled/itm_emb_pomrec.pkl` (n_items−1, 64) | ✅ 已有 | `tools/export_pomrec_item_emb.py` 从 PoMRec ckpt 的 `interest_extractor.i_embeddings.weight[1:]` 导出 |
| 分解基 `U_r` | `data/<ds>/handled/cgscd_basis_r32.pkl` | ❌ **新建** | 新工具 `build_cgscd_basis.py` |
| 分解基 r=64 版 | `data/<ds>/handled/cgscd_basis_r64.pkl` | ❌ **新建** | 同上 `--rank 64` |
| SVD 协同视图（对照） | `data/<ds>/handled/cf_svd128.pkl` | ❌ **新建（可选）** | 新工具，对 `train.csv` 交互矩阵做截断 SVD |

**`C` 的溯源（Phase 0 逐元素验证完成）**：

| 数据集 | 对应 checkpoint | 验证 |
|---|---|---|
| beauty | `model/PoMRec/PoMRec__beauty__42__lr=0.002__l2=1e-06.pt` | `W[1:]` 与 pkl **逐元素相等**，maxdiff = 0.0 |
| ml-1m | `model/PoMRec/PoMRec__ml-1m__1__lr=0.001__l2=1e-06.pt` | maxdiff = 0.0 |
| toys | `model/PoMRec/toys__1__lr=0.001__l2=1e-06__lamb=3.8__history_max=20.pt` | maxdiff = 0.0（仓库另有 `itm_emb_pomrec1.pkl` 及多个 `PoMRec__toys__*` ckpt **不匹配**，maxdiff = 2.50） |

`C` 的统计（Phase 0 实测）：

| 数据集 | 形状 | 行范数 mean | 有效秩 | 重复行 |
|---|---|---|---|---|
| beauty | (12101, 64) | 2.283 | 60.83 | 0 |
| ml-1m | (3706, 64) | 2.943 | 51.14 | 0 |
| toys | (78771, 64) | 1.362 | 40.07 | 0 |

### F. 新增预计代码文件

| 文件 | 类型 | 预计行数 |
|---|---|---|
| `tools/build_cgscd_basis.py` | 离线工具 | ~180 |
| `tools/build_cf_svd_view.py` | 离线工具（对照视图） | ~120 |
| `models/sequential/LLMMIRecCGSCD.py` | 模型 | ~320（其中 ~110 来自 mixin） |
| `models/sequential/llmmi_components.py` | 修改 | +90 ~ 120 |
| `models/sequential/llmmi_relation.py`（建议新建） | 公共 loss mixin | ~150（可被 Ch3/4/5 共用，替代 4 处复制） |
| `tools/analyze_cgscd.py` | 诊断 | ~250 |
| `tools/test_llmmirec_cgscd.py` | CPU synthetic 单测 | ~200 |

### G. 预计实验脚本

| 脚本 | 内容 |
|---|---|
| `new_bash/run_llmmirec_cgscd_phase1_beauty.sh` | CGSCD 主实验（r=32, frozen basis） |
| `new_bash/run_llmmirec_cgscd_phase1_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_cgscd_ablation_beauty.sh` | 消融：`r ∈ {8,16,32,64}`；`basis_mode ∈ {frozen,learnable}`；`cgscd_use_cf_input ∈ {0,1}`；`cf_view ∈ {pomrec,svd128}`；`lambda_relation/shared/private/redundancy` |
| `new_bash/run_llmmirec_cgscd_ablation_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_cgscd_phase2_all_seeds.sh` | 5 种子队列（沿用 `run_llmmirec_caisd_phase2_all_seeds.sh` 的结构） |
| `new_bash/run_llmmirec_cgscd_diagnostics.sh` | shared/private 分支方差占比、gate α 分布、共享信号强度 |

---

## Chapter 4 — Interest Representation

方法：Prototype-Grounded Interest Semantic Profile + Multi-Interest Semantic Coverage。
`HSDIR` 为 preliminary，不进入最终方法。

### A. 可以原样复用的模块

| 模块 | 位置 | 复用理由 |
|---|---|---|
| **Interest semantic profile 蒸馏全部机制** | `LLMMIRecCAISD.py:240-300` | `Q = t_semantic_assign[history]`（`:248`）→ `T = bmm(A_teacher, Q)`（`:272`，或 responsibility 变体 `:253-270`）→ 归一化 → `P_logits = semantic_predictor(V)`（`:276`）→ `KL(T ‖ P)`（`:279`）。这是 Chapter 4 模块 1 的**完整实现** |
| 三种 teacher 变体 | `LLMMIRecCAISD.py:253-273` | `attention` / `responsibility` / `responsibility_power`（含 `--semantic_responsibility_alpha`） |
| confidence 加权 | `LLMMIRecCAISD.py:281-285` | `c = (1 - H(T)/log(32)).clamp(0,1).detach()`，`L = Σ c·kl / Σ c` |
| interest relational JS 蒸馏 | `LLMMIRecCAISD.py:371-405` | strict upper-triangle 的 pairwise JS + smooth_l1 —— **Ch4 可选复用**（作为 focus 的补充，非默认） |
| `semantic_predictor` 结构 | `LLMMIRecCAISD.py:213` | `nn.Linear(emb_size=64, 32)` |
| prototype 加载逻辑 | `LLMMIRec.py:171-187`（`interest_query_mode=prototype`） | 加载 `centers`/`soft_assignments`/`prototype_num` 并注册为 non-persistent buffer |
| teacher 的层级结构 | `LLMMIRecHSDIR.py:165-173` | fine 32 + coarse 8 的加载（仅 HSDIR 用，Ch4 主方法不需要） |
| `InterestAggregator` | `llmmi_components.py:352` | **`w_k` 的唯一来源**（history-only） |
| NaN/Inf、位置编码、打分路径 | 同 Ch3 | — |

### B. 可以轻微修改后复用的模块

| 模块 | 位置 | 改动内容 | 预计改动量 |
|---|---|---|---|
| `build_llmmi_semantic_prototypes.py` | `tools/` (105 行) | 增加 `--input_rep {pca512, cgscd_shared}` + `--basis_path` + `--input_dim`；输出文件名按表示来源编码（当前 `llmmi_proto{num}_sr{rank}.pkl` 无法区分来源） | ~30 行 |
| `build_llmmi_hierarchical_teacher.py` | `tools/` (85 行) | 只需路径参数化 + 输出命名加入 `input_rep`（其内部逻辑是"KMeans on fine centers + 求和聚合"，与输入维度无关） | ~10 行 |
| `LLMMIRecCAISD.py` 作为骨架 | `LLMMIRecCAISD.py` | 复制为新 `LLMMIRecISPC.py`；在 `:273` 之后插入 coverage 与 focus 计算；在 `:470-481` 的 loss 聚合中加入两个新项 | +70 ~ 110 行 |
| `run_llmmirec_caisd_phase2_*.sh` | `new_bash/` | 替换 `--model_name` / `--semantic_teacher_path` / 新增 `--coverage_*` / `--focus_*` | 每数据集 ~100 行 |

### C. 应淘汰 / 仅保留为探索实验

| 模块 | 处置 | 理由 |
|---|---|---|
| `LLMMIRecHSDIR` | **降级为 preliminary experiment** | 完整结论保留：能改变兴趣结构（cosine 0.9457→0.8691，有效秩 1.714→2.280），但不稳定转化为推荐收益。作为动机引用，不进入最终方法 |
| HSDIR 的 `pair_selective` / `relative` 两种 loss mode | 不进入主方法 | 5-seed 未超基线 |
| HSDIR 的 `support_confidence` 聚合校准 | 不进入主方法 | `--aggregation_mode`/`--support_beta` 定义了但**从未被任何脚本使用**，无实验证据 |
| 用户级自适应蒸馏 | 已废弃 | `tools/analyze_hsdir_benefit_factors.py` 的结论：`teacher_js_shift` 与收益的相关性跨数据集方向不一致（beauty ρ=0.0105 p=0.2336 不显著；ml-1m ρ=0.0281 弱正相关）。保留分析工具作证据 |
| `LLMMIRecCHIR` | 保留为探索 | prototype query + dual-view routing，无额外 loss，纯靠 ranking loss。与 Ch4 的"profile 蒸馏"路线不同 |
| `LLMMIRecCASIR` | 保留为探索 | 语义残差直接进入打分，属于"注入"而非"蒸馏"，与 Ch4 路线不同 |

### D. 必须新实现的模块

| 模块 | 说明 | 关键设计点 |
|---|---|---|
| **`q_seq` 构造** | `q_seq = normalize(Σ_{l valid} Q_{b,l,:})` | 必须 mask padding（`history > 0`）；归一化到单形 |
| **`q_hat_seq` 构造** | `q_hat = Σ_k w_k · P_k`，`P_k = softmax(g(V_k))`，`w_k = InterestAggregator(history)` | 是 K 个分布的凸组合，Σw=1 保证其自动是合法分布 |
| **`L_coverage`** | `D(q_seq ‖ q_hat)` | 见下方"度量选择" |
| **`L_focus`（必需配套）** | 对 `P_k` 的低熵/稀疏先验，防止 coverage 的平凡解 | **这是 Ch4 区别于 CAISD 的关键新增约束** |
| **coverage 平凡解防护** | 构造性分析：`P_k ≡ q_seq ∀k ⇒ q_hat = q_seq`，与 `w` 无关 | 必须在实现前设计好；仅靠 coverage 一项无法避免 |
| prototype 重建链 | 基于 `Z_shared` 而非 `z[:, :512]` | 见 `THESIS_MASTER_PLAN.md` §5.2 的三档方案 |

**`L_coverage` 度量选择（KL / JS / cosine）**：

| 度量 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| `KL(q_seq ‖ q_hat)` | 语义正确（zero-avoiding，强制 `q_hat` 在 `q_seq` 非零处也有质量，符合"覆盖"目标） | `q_hat → 0` 处梯度爆炸；需要 `eps` 保护 | **作为消融** |
| `KL(q_hat ‖ q_seq)` | — | mode-seeking：`q_hat` 会集中到 `q_seq` 的峰并忽略尾部，**直接违背"覆盖"目标** | **不用** |
| **`JS(q_seq, q_hat)`** | 对称、有界 `[0, ln2]`、无零除风险、可解释为度量的平方根；两侧都有梯度 | 接近最优点时梯度趋于 0 | **推荐主选** |
| `cosine(q_seq, q_hat)` | 数值最稳 | 不是散度，对单形几何不敏感——只要主峰对上就满足，无法支撑"覆盖"这一主张 | **不用作主 loss**（可作监控指标） |
| `Hellinger²` | 行为接近 JS，但在 0 附近梯度非退化 | 稍不常见 | **建议作为敏感性对照** |

**推荐**：主 loss 用 **JS**（稳定性 + 可解释性 + 有界），消融用 `KL(q_seq‖q_hat)` 与 `Hellinger²`，`cosine` 只作为日志监控量。

### E. 依赖的预计算数据

| 数据 | 路径 | 状态 |
|---|---|---|
| LLM 语义表 | `data/<ds>/handled/llm_table_pca1536.pkl` | ✅ |
| fine prototype（PCA-512 版） | `data/<ds>/handled/llmmi_proto32_sr512.pkl` (centers (32,512), soft_assignments (n_items,32)) | ✅ |
| hier teacher（PCA-512 版） | `data/<ds>/handled/llmmi_hier_proto32_8_sr512.pkl`（fine 32 + coarse 8 + `fine_to_coarse`） | ✅ |
| CGSCD basis | `data/<ds>/handled/cgscd_basis_r32.pkl` | ❌ 依赖 Ch3 |
| **CGSCD prototype（新）** | `data/<ds>/handled/cgscd_proto32_r32.pkl` | ❌ 新建 |
| CGSCD hier teacher（消融） | `data/<ds>/handled/cgscd_hier_proto32_8_r32.pkl` | ❌ 新建 |

**prototype 现有实现的细节**（`build_llmmi_semantic_prototypes.py`）：
- 输入 `z_high = table[1:, :512]`（**排除 row 0**）
- `MiniBatchKMeans(n_clusters=32, batch_size=1024, n_init=3, max_iter=100, random_state=42)`
- soft assignment：对 `z_high` 与 `centers` 分别 L2 归一化 → 余弦 → `softmax(cos/0.1)`
- 输出时**补回 row 0 = zeros**，故 `soft_assignments` 行数 = `n_items`（含 padding），`centers` 只有 32×512
- 验证：`coarse_assignments[1:].sum(axis=1)` 与 1 的最大偏差 < 0.01（hier 工具中的 assert）

### F. 新增预计代码文件

| 文件 | 类型 | 预计行数 |
|---|---|---|
| `tools/build_llmmi_semantic_prototypes.py` | 修改 | +30 |
| `tools/build_llmmi_hierarchical_teacher.py` | 修改 | +10 |
| `models/sequential/LLMMIRecISPC.py` | 模型（ISPC = Interest Semantic Profile + Coverage） | ~560（含 ~110 mixin 节省） |
| `tools/analyze_ispc_coverage.py` | 诊断：`q_seq` vs `q_hat` 的覆盖质量、`P_k` 稀疏度、`w_k` 分布、逐用户覆盖率 | ~300 |
| `tools/test_llmmirec_ispc.py` | CPU synthetic 单测 | ~220 |

### G. 预计实验脚本

| 脚本 | 内容 |
|---|---|
| `new_bash/run_llmmirec_ispc_phase1_beauty.sh` | 主实验（CGSCD prototype + coverage + focus） |
| `new_bash/run_llmmirec_ispc_phase1_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_ispc_ablation_beauty.sh` | 消融：`coverage_metric ∈ {js, kl, hellinger}`；`lambda_coverage`；`lambda_focus`；`focus_type ∈ {entropy, l2}`；`prototype_source ∈ {pca512, cgscd_r32, cgscd_r64}`；`coverage on/off` × `focus on/off`（2×2 最小完整消融） |
| `new_bash/run_llmmirec_ispc_ablation_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_ispc_phase2_all_seeds.sh` | 5 种子队列 |
| `new_bash/run_llmmirec_ispc_diagnostics.sh` | 调用 `analyze_ispc_coverage.py` |

---

## Chapter 5 — Interest Decision

方法：Target-Conditioned Semantic Interest Teacher + Confidence Calibration + Target-Discriminative Semantic Ranking。

### A. 可以原样复用的模块

| 模块 | 位置 | 复用理由 |
|---|---|---|
| **TASID 全部现有实现** | `LLMMIRecCAISD.py:316-364` | target 语义 query（`:320-323`）、teacher 分布（`:326-329`）、student 分布两种模式（`:332-353`）、KL（`:355-357`）——Chapter 5 的起点 |
| **semantic hard-negative bank** | `data/<ds>/handled/semantic_hardneg_top100.pkl` | ✅ **已存在**，已覆盖三数据集，质量已验证（见 §E） |
| **hard-negative 构建脚本** | `scripts/build_semantic_hardneg.py` | 已实现 chunked 余弦 top-K + 排除自身 + padding 处理 |
| TASID loss 注入点 | `LLMMIRecCAISD.py:484-487` | `if "_tasid_loss" in out_dict and self.lambda_tasid > 0:` |
| `distill_info` 诊断字典 | `LLMMIRecCAISD.py:290-300, 359-364` | 已携带 `_tasid_teacher_dist` / `target_interest_distribution` 等 |
| relation loss、NaN 检查、位置编码、打分路径 | 同 Ch3/Ch4 | — |

**TASID 现有实现精确定位**（`LLMMIRecCAISD.py`）：

| 组件 | 行号 | 实现 |
|---|---|---|
| target semantic query | `:320-323` | `z_target = item_encoder.llm_table[target_ids]`；`q_target = semantic_branch(z_target[:, :semantic_rank])`；`.detach()` |
| teacher distribution | `:326-329` | `cos_teacher = cosine_similarity(q_target, T)`；`p_tasid_teacher = softmax(cos_teacher / tasid_temp)`；`.detach()` |
| student（llm_only） | `:348-351` | `V_sem = V[..., :semantic_dim]`；`cos_student = cosine_similarity(q_target, V_sem)` |
| student（asymmetric） | `:332-347` | `c_target` = ASPCF complement 分支（detach）→ `student_query = cat([q_target, c_target])`；`student_interest = cat([V[...,32:], V[...,:32]])` |
| KL | `:355-357` | `F.kl_div(log_softmax(cos_student/τ), p_tasid_teacher, reduction="batchmean")` |
| loss 注入 | `:484-487` | `total += lambda_tasid * L_tasid` |

### B. 可以轻微修改后复用的模块

| 模块 | 位置 | 改动内容 | 预计改动量 |
|---|---|---|---|
| **confidence calibration 插入** | `LLMMIRecCAISD.py:326-357` 之间 | 见下方"最小改动方案" | ~6 行 |
| hard-negative bank 加载 | 无现成加载器 | 在 `__init__` 中 `register_buffer("hn_bank", torch.tensor(pickle.load(...)), persistent=False)`；形状 `(n_items, M)` int64 | ~8 行 |
| history 过滤 | 无现成实现 | 复用 `feed_dict["history_items"]`，做 `isin` 广播比较 | ~10 行 |
| `LLMMIRecCAISD.py` 作为骨架 | 全体 | 复制为 `LLMMIRecTDIR.py`，插入 `L_disc` | +80 ~ 120 行 |
| `scripts/build_semantic_hardneg.py` | 105 行 | 扩展：输出额外保存 cosine 值（当前只存 index）→ 支持按难度分档采样 | ~25 行 |

**confidence calibration 的最小改动位置与形式**：

现有 `:355-357` 使用 `reduction="batchmean"`，无法做 per-sample 加权。最小改动：

```python
# 在 :329 之后（p_tasid_teacher 已算出、已 detach）
H_teacher = -(p_tasid_teacher * torch.log(p_tasid_teacher + 1e-8)).sum(dim=-1)   # [B]
c_tasid   = (1.0 - H_teacher / math.log(self.K)).clamp(0.0, 1.0).detach()        # [B]

# 在 :355 替换 batchmean 形式
log_q    = F.log_softmax(cos_student / self.tasid_temp, dim=-1)                  # [B, K]
kl_pp    = F.kl_div(log_q, p_tasid_teacher, reduction="none").sum(dim=-1)        # [B]
L_tasid  = (c_tasid * kl_pp).sum() / c_tasid.sum().clamp(min=1e-8)
```

净增约 6 行。**但见下方风险**——公式位置不是主要难点，熵的量级才是。

**⚠️ confidence 校准的实测风险（Phase 0）**：

用 prototype 空间（P=32）作为代理测得 `softmax(cos/τ)` 的熵：

| τ | beauty 平均 H | `1 - H/ln32` | ml-1m 平均 H | `1 - H/ln32` |
|---|---|---|---|---|
| 0.05 | 0.483 | 0.861 | 0.534 | 0.846 |
| **0.10** | **1.617** | **0.533** | **1.458** | **0.580** |
| 0.20 | 2.977 | 0.141 | 2.760 | 0.204 |
| 0.50 | 3.413 | 0.015 | 3.369 | 0.028 |

- 在 32-way prototype 空间、τ=0.1 时，confidence 有良好区分度（≈0.53，方差可观）→ 校准**有效**。
- **但真实 teacher 分布在 K=4 个 interest 上**，且 `T_k = bmm(A, Q)` 是 prototype 的**加权平均**。若某 interest 覆盖语义多样的 item，`T_k` 趋近均匀 → `cos(q_target, T_k)` 各 k 接近 → `p_teacher` 趋近均匀 → `H → ln 4` → `c → 0` → **校准会把整个 loss 关闭**。这与代理测量的方向相反。
- **结论**：实现前必须先用一个 batch 测真实 `p_teacher` 的熵分布。若接近 `ln K`，改用 **τ-independent 的 margin-based 置信度**：`c = sigmoid(β · (cos_top1 − cos_top2))`。两种都实现，作为 `confidence_mode ∈ {none, entropy, margin}` 消融。

**semantic hard-negative bank：全局预计算 vs 训练时过滤**

**推荐：全局预计算 index + 训练时过滤。**

理由：
1. **全局预计算的规模是可控的**：bank 大小 `n_items × M`，M=100 时 beauty 12K×100×8B ≈ 9.7 MB、toys 79K×100×8B ≈ 63 MB。**已经存在**。
2. **per-(user, target) 预计算不可行**：过滤条件是"该用户的 history"，history 随样本变化（每个正样本对应一条不同的截断历史），无法离线枚举。
3. **训练时过滤成本极低**：`bank_batch = bank[target_ids]` `[B, M]`；`hist_mask = (bank_batch[:, :, None] == history[:, None, :]).any(-1)` `[B, M]`；取第一个幸存者。B=1024、M=100、L=20 时约 2M 次比较，可忽略。
4. **必须同时排除的集合**：
   - target 自身（**bank 构建时已排除**，verified：`self-in-bank count = 0`）
   - 用户 history 中的全部 item（**训练时过滤**）
   - **用户全量交互过的 item**（不只是截断窗口内的 history）——否则 hard negative 可能是该用户未来的正样本，构成假负例。需要额外加载一份 user→全量 item 集合（可从 `SeqReader.pkl` 或 `train.csv` 构造），或直接在 `bank[target]` 上做一次全局 user-item 查表。
5. **top-M 大小建议**：**M = 100（现状）足够**。实测过滤后幸存数：beauty min=83/mean=99.3，ml-1m min=34/mean=88.0。即便 ml-1m 最坏情况也有 34 个候选。
6. **难度分档建议**：`cos(target, rank-1) = 0.602 (beauty) / 0.587 (ml-1m)`，rank-2 0.528/0.546，rank-10 0.391/0.455，rank-50 0.281/0.349。**rank-1 与 rank-2 的余弦差只有 0.07/0.04，接近近重复**。建议从 **rank ∈ [2, 20]** 采样（跳过 rank-1 以避免退化成"同义 item"），并把难度档位 `hn_rank_lo/hi` 做成超参。

**target-discriminative ranking loss 的构造**：

```
# target-conditioned interest representation（让知识进入打分）
π        = student target-conditioned distribution  [B, K]      # 由 TASID student 侧得到
z_t      = Σ_k π_k · V_k                                        # [B, D]

e_pos    = item_encoder(pos_ids)                                # [B, D]
e_hn     = item_encoder(hn_ids)                                 # [B, D]（hn_ids 来自过滤后的 bank）

s_pos    = (z_t * e_pos).sum(-1)                                # [B]
s_hn     = (z_t * e_hn).sum(-1)                                 # [B]

L_disc   = F.relu(self.disc_margin - s_pos + s_hn).mean()
```

设计要点：
- **`π` 用 student 侧，不用 teacher 侧**——teacher 已 detach，无法传梯度；且 Chapter 5 的目标是训练模型自己的 target-conditioned 选择能力。
- **`z_t` 与 BPR 的 `user_vector = Σ w_k V_k` 是不同表示**：`w` 是 history-only 的，`π` 是 target-conditioned 的。这正是"target-interest 知识真正进入 ranking"的落点，也是 Chapter 5 区别于"TASID 小修"的核心证据。
- **margin 与 BPR 的关系**：BPR 已包含 `(pos, 随机负样本)` 的对比。`L_disc` 只处理**语义 hard negative**，二者互补而非重复。可提供 `disc_mode ∈ {hn_only, hn_plus_rand}` 消融。
- **可选扩展**：把 `π·V` 参与 `user_vector` 的构造（如 `user_vector_disc = Σ (w_k + β·π_k) V_k`），使 target-conditioned 选择直接影响打分。**但这会引入 train/test 不一致**（测试时无 target），需要谨慎——建议仅作为探索项，默认只用于训练期 loss。

### C. 应淘汰 / 仅保留为探索实验

| 模块 | 处置 | 理由 |
|---|---|---|
| 现有 TASID 实现 | **保留为 baseline 与出发点**，不作为最终方法 | fair 5-seed 显示未稳定超越 ASPCF（见 `THESIS_PROGRESS.md` §3）。核心缺陷：target-interest 知识从未进入打分 |
| `semantic_hardneg_top100.pkl` 的原始用法（`MyModelSHNC`） | 参考不复用 | 那是 MyModel 时代的用法，与 LLMMIRec 主线不同 |
| `tools/analyze_caisd_teacher_benefit.py` | 保留，可扩展 | paired 分析框架（两个 checkpoint 同 batch 对比）有价值；但需修掉 `user_rows[0]` 空列表崩溃与"两次独立训练混淆方差"的问题 |
| `tools/analyze_llmmirec_caisd.py` | 保留，需修硬编码 | 强制 `semantic_distill_mode="uniform"` 且 `load_state_dict(strict=False)` 会静默接受缺失的 `semantic_predictor` → 测量到的是随机初始化 head 的 KL |

### D. 必须新实现的模块

| 模块 | 说明 |
|---|---|
| **`confidence_mode = margin`** | τ-independent 置信度 `sigmoid(β·(cos_top1 − cos_top2))`，作为 entropy 版的 fallback（见上） |
| **`L_disc`** | target-discriminative semantic ranking loss（见上） |
| **hard-negative 采样器** | `bank[target_ids]` → 排除 history 与全量交互 → 按难度档 `[rank_lo, rank_hi]` 采样 → `hn_ids` |
| **user→全量交互 item 集合** | 用于硬负例的真负例校验；可从 `SeqReader.pkl` 或 `train.csv` 构造，需缓存 |
| **`build_semantic_hard_negative_bank.py`** | 复用 `scripts/build_semantic_hardneg.py` 的逻辑，但：① 额外保存 cosine 值以支持难度分档；② 参数化 `--input_rep`（支持 CGSCD 共享空间）；③ 用 sha1 锁定输入来源。**注意**：现有 bank 已可用，新工具的唯一实质增量是"保存余弦"与"支持 CGSCD 表示" |
| **`π` 的独立性校验** | 确保 `L_disc` 的 `π` 与 `L_tasid` 的 student 分布是同一个（避免出现两个不同的 target-conditioned 分布） |

### E. 依赖的预计算数据

| 数据 | 路径 | 状态 |
|---|---|---|
| LLM 语义表 | `data/<ds>/handled/llm_table_pca1536.pkl` | ✅ |
| CGSCD prototype（来自 Ch4） | `data/<ds>/handled/cgscd_proto32_r32.pkl` | ❌ 依赖 Ch3/4 |
| **semantic hard-negative bank** | `data/<ds>/handled/semantic_hardneg_top100.pkl` | ✅ **已存在** |
| hard-negative bank + cosine（新） | `data/<ds>/handled/semantic_hardneg_cos_top100.pkl` | ❌ 可选新建 |
| user→全量 item 集合 | 待定（建议 `data/<ds>/handled/user_full_items.pkl`） | ❌ 新建 |

**`semantic_hardneg_top100.pkl` 实测质量（Phase 0）**：

| 数据集 | 形状 | row 0 | 自引用 | 每行唯一数 | 排除 history 后幸存（of 100） |
|---|---|---|---|---|---|
| beauty | (12102, 100) int64 | 全零 ✅ | 0 ✅ | 100 | min 83 / mean 99.3 |
| ml-1m | (3707, 100) int64 | 全零 ✅ | 0 ✅ | 100 | min 34 / mean 88.0 |
| toys | (78772, 100) int64 | 全零 ✅ | 0 ✅ | 100 | — |

难度梯度（平均余弦）：beauty rank-1 0.602 / rank-2 0.528 / rank-10 0.391 / rank-50 0.281；ml-1m 0.587 / 0.546 / 0.455 / 0.349。

### F. 新增预计代码文件

| 文件 | 类型 | 预计行数 |
|---|---|---|
| `models/sequential/LLMMIRecTDIR.py` | 模型（TDIR = Target-Discriminative Interest Ranking） | ~600 |
| `tools/build_semantic_hard_negative_bank.py` | 离线工具（扩展版） | ~180 |
| `tools/build_user_full_items.py` | 离线工具 | ~70 |
| `tools/analyze_tdir_decision.py` | 诊断：`π` 分布、confidence 熵分布、hard-negative 命中率、`s_pos`/`s_hn` 边际分布 | ~350 |
| `tools/test_llmmirec_tdir.py` | CPU synthetic 单测（含硬负例过滤正确性） | ~250 |

### G. 预计实验脚本

| 脚本 | 内容 |
|---|---|
| `new_bash/run_llmmirec_tdir_phase1_beauty.sh` | 主实验（TASID + confidence + L_disc） |
| `new_bash/run_llmmirec_tdir_phase1_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_tdir_ablation_beauty.sh` | 消融：`confidence_mode ∈ {none, entropy, margin}`；`lambda_disc`；`disc_margin`；`hn_rank_lo/hi ∈ {[1,20],[2,20],[2,50],[10,100]}`；`disc_mode ∈ {hn_only, hn_plus_rand}`；`coverage/focus` 开关的 cross-ablation |
| `new_bash/run_llmmirec_tdir_ablation_ml1m.sh` | 同上 |
| `new_bash/run_llmmirec_tdir_phase2_all_seeds.sh` | 5 种子队列 |
| `new_bash/run_llmmirec_tdir_diagnostics.sh` | 调用 `analyze_tdir_decision.py` |

---

## 附：三章复用率汇总

| | Ch3 CGSCD | Ch4 ISPC | Ch5 TDIR |
|---|---|---|---|
| A 原样复用模块数 | 12 | 10 | 8 |
| B 轻微修改模块数 | 5 | 4 | 6 |
| C 淘汰/降级模块数 | 4 | 6 | 4 |
| D 必须新实现模块数 | 7 | 5 | 6 |
| 新增/修改代码文件数 | 7 | 5 | 5 |
| 预计新代码行数 | ~1200 | ~1120 | ~1450 |
| 预计新实验脚本数 | 6 | 6 | 6 |
| 现有可复用预计算资产 | 2/4 | 3/6 | 2/5 |

**关键复用红利**：
- `L_profile` 蒸馏机制（CAISD `:240-300`）在 Ch4 可**原样复用**，节省约 60 行核心逻辑与对应的调试时间。
- `semantic_hardneg_top100.pkl` 在 Ch5 **已存在且已验证**，省掉一个完整的离线构建环节与一次全库最近邻搜索。
- `InterestAggregator` 直接提供 Ch4 的 `w_k`，无需新实现。

**关键缺失**：
- Ch3 的 `U_r` 与 Ch4 的 CGSCD prototype 都要**新建**，且 Ch4 的 prototype 依赖 Ch3 的 basis —— 这构成一个**硬依赖**：Chapter 4 的主实验无法在 Chapter 3 完成前启动。

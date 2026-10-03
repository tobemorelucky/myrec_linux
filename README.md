# LLMMIRec — 大语言模型增强多兴趣序列推荐

本仓库是一个面向**多兴趣序列推荐（Multi-Interest Sequential Recommendation）**的 PyTorch 研究代码库。

当前主线工作是 **LLMMIRec**：一个独立于 PoMRec 的干净基线，逐步扩展为两章方法：

- **第三章 ASPCF** — Adaptive Semantic-Preserving Subspace Complementary Fusion（语义保持的子空间互补融合）
- **第四章 TASID** — Target-Aware Semantic Interest Distillation（目标感知多兴趣语义蒸馏）

> 仓库同时保留了早期的 PoMRec / MyModel 系列代码（见文末"历史代码"），但当前实验主线为 LLMMIRec。

---

## 一、整体研究进程与结论

### 1.1 完整路径

```
Phase 0  干净基线 LLMMIRec
   ↓
第三章   ASPCF（item 表示层的语义融合）
   ↓
第四章   HSDIR → 用户级 gate → 关系级蒸馏 → CAISD → TASID ✅
```

### 1.2 成功结论 ✅

| 阶段 | 结论 | 关键证据 |
|------|------|---------|
| **Phase 0** | LLM 嵌入替换 CF 嵌入有效但需 GELU 非线性；residual 融合稳定 | Beauty: llm_replace 0.1016 > residual 0.0946 > id 0.0832 |
| **第三章 ASPCF** | 子空间互补融合 + relation loss 稳定超越纯 CF | Beauty NDCG@5: ASPCF 0.1100 vs PoMRec 0.0986 |
| **第四章 HSDIR** | LLM 语义约束**能有效分化兴趣结构** | Beauty 兴趣 cosine 0.9457→0.8691，effective rank 1.714→2.280；ML-1M 0.7620→0.7377，rank 2.762→3.028 |
| **第四章 CAISD** | LLM prototype teacher 引入兴趣语义分布学习有效 | ML-1M: uniform HR@5=0.3091, responsibility HR@5=0.3109 |
| **第四章 TASID** | 目标感知非对称蒸馏**稳定提升** | Beauty: 0.1551→**0.1608** HR@5；ML-1M asymmetric: 0.3040→**0.3093** HR@5 (+1.7%), NDCG@5 0.2104→**0.2161** (+2.7%) |

### 1.3 失败结论 ❌

| 探索方向 | 失败原因 | 关键证据 |
|---------|---------|---------|
| **纯兴趣路由约束** | 兴趣分化 ≠ 推荐收益 | HSDIR 改变了兴趣结构但 Beauty HR@5 仅 0.1604，提升有限；ML-1M 不稳定 |
| **用户级自适应 gate** | 用户级因素无法稳定预测 teacher 收益 | Beauty teacher_js_shift ρ=0.0105, p=0.2336（不显著）；ML-1M ρ=0.0281 弱相关但跨数据集方向不一致 |
| **关系级选择性蒸馏** | 简单关系选择缺少目标相关性 | ML-1M pair selective HR@5=0.2987、attention contribution ≈0.3075，均未超基线 |
| **静态 teacher** | teacher 需与预测目标动态匹配 | CAISD 多种 seed 收益不稳定 |

### 1.4 核心教训

1. **兴趣结构约束有效但不充分** — 结构分化不直接转化为推荐收益
2. **用户级选择不可行** — 用户因素跨数据集不一致
3. **目标相关性是关键** — LLM 语义监督必须与推荐目标直接对齐

**最终创新链**：LLM 语义知识构建（第三章 prototype 空间）→ 兴趣级语义蒸馏（CAISD）→ 目标感知兴趣分配优化（TASID）

---

## 二、模型

### 2.1 现有模型总览

| 模型 | 说明 | 状态 |
|------|------|------|
| `LLMMIRec` | Phase 0 干净基线：4 种 item encoder（id / llm_replace / residual / aspcf） | 保留复现 |
| `LLMMIRecASPCF` | **第三章**：ASPCF + relation loss + learnable query | ✅ 冻结 |
| `LLMMIRecCAISD` | **第四章最终**：CAISD + TASID（asymmetric student） | ✅ 最终方法 |
| `LLMMIRecHSDIR` | 第四章探索：层次语义路由蒸馏 | 保留（探索记录） |
| `LLMMIRecCHIR` | 第四章探索：prototype query + dual-view routing | 保留（探索记录） |
| `LLMMIRecCASIR` | 第四章探索：协同锚定语义兴趣精炼 | 保留（探索记录） |

统一入口：

```bash
python main.py --model_name <MODEL_NAME> --dataset <DATASET_NAME>
```

### 2.2 第三章 LLMMIRecASPCF

ASPCF 子空间互补融合：

```
llm_table[item_id] (1536-dim PCA)
  ├─ z_high = [:512]  →  Linear(512→128) → GELU → Linear(128→32) → s (semantic, 32-dim)
  └─ z_low  = [512:]  →  Linear(1024→64) → GELU → low_feat (64-dim)
         + complement_id_emb[item]                  (64-dim)
         → concat → Linear(128→64) → GELU → Linear(64→32) → c (complement, 32-dim)

  Gate: [s; c] → Linear(64,64) → GELU → Linear(64,2) → softmax → [α_s, α_c]
  e = concat[√(α_s+ε)·s, √(α_c+ε)·c]   (64-dim)

Total loss = BPR + λ_relation · L_relation
  L_relation: frozen z_high 与 semantic_branch 输出的 item-item 余弦关系 KL 蒸馏
```

### 2.3 第四章 LLMMIRecCAISD + TASID

```
兴趣表示: V  (QueryMultiInterestExtractor 输出, [B,K,64])
兴趣权重: InterestAggregator (仅用历史)
推荐:     user_vector = Σ_k w_k·V_k → dot(candidate_emb)   ← 不变

训练期附加 (semantic distillation):
  T = normalize(A_detach @ Q)          # 兴趣的动态语义 profile (Q = LLM prototype 分配)
  L_profile = KL(T || softmax(predictor(V)))

TASID (目标感知):
  q_target = semantic_branch(llm_table[target][:,:512]) → detach
  Teacher:  p[k] = softmax_k(cos(q_target, T_k)/τ)
  Student:  llm_only:   q[k] = softmax_k(cos(q_target, V_k[:32])/τ)
            asymmetric: q[k] = softmax_k(cos(concat(CF_target, sem_target), V_k)/τ)
  L_tasid = KL(p_teacher || q_student)   # 非对称, target 侧 detach

Total = BPR + λ_rel·L_rel + λ_sem·L_sem + λ_tasid·L_tasid
```

**关键设计**：训练期 LLM 语义监督；测试期无 target 泄露；`tasid_mode=none` 完全退化为 CAISD。

### 2.4 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--item_encoder` | `aspcf` | item encoder 模式 |
| `--semantic_rank` | `512` | LLM PCA 高方差子空间维度 |
| `--aspcf_gate_mode` | `basic` | ASPCF gate 结构 |
| `--lambda_relation` | `0.01` | relation preservation 权重 |
| `--semantic_teacher_path` | — | LLM prototype teacher pkl |
| `--semantic_distill_mode` | `none` | profile 蒸馏模式（uniform / confidence） |
| `--semantic_teacher_mode` | `attention` | teacher 构造（attention / responsibility） |
| `--lambda_interest_semantic` | `0.01` | profile 蒸馏权重 |
| `--tasid_mode` | `none` | TASID 开关（none / target） |
| `--tasid_student_mode` | `llm_only` | student 构造（llm_only / asymmetric） |
| `--lambda_tasid` | `0.01` | TASID 权重 |
| `--tasid_temp` | `0.1` | TASID 温度 |

---

## 三、预计算文件

| 文件 | 路径 | 说明 |
|------|------|------|
| LLM 语义表 (PCA) | `data/<dataset>/handled/llm_table_pca1536.pkl` | 4096→1536 PCA 降维，按方差降序 |
| LLM 语义表 (原始) | `data/<dataset>/handled/llm_table.pkl` | 原始 4096 维 |
| Fine prototype | `data/<dataset>/handled/llmmi_proto32_sr512.pkl` | 32 个 prototype + soft assignment |
| Hier teacher | `data/<dataset>/handled/llmmi_hier_proto32_8_sr512.pkl` | fine32 → coarse8（HSDIR 用） |

构建命令：

```bash
python tools/build_llmmi_semantic_prototypes.py --dataset beauty
python tools/build_llmmi_hierarchical_teacher.py --dataset beauty
```

---

## 四、运行实验

### 4.1 最终公平实验（ASPCF baseline vs CAISD+TASID）

```bash
# Beauty (GPU 0, seed 42)
bash new_bash/run_llmmirec_aspcf_phase2_beauty.sh 0 42
TASID_MODE=target TASID_STUDENT_MODE=asymmetric LAMBDA_TASID=0.01 \
  bash new_bash/run_llmmirec_caisd_phase2_beauty.sh 0 42

# ML-1M (GPU 1, seed 42)
bash new_bash/run_llmmirec_aspcf_phase2_ml1m.sh 1 42
TASID_MODE=target TASID_STUDENT_MODE=asymmetric LAMBDA_TASID=0.01 \
  bash new_bash/run_llmmirec_caisd_phase2_ml1m.sh 1 42
```

多 seed（0/1/42）：

```bash
for S in 0 1 42; do
  bash new_bash/run_llmmirec_aspcf_phase2_beauty.sh 0 $S
  TASID_MODE=target TASID_STUDENT_MODE=asymmetric \
    bash new_bash/run_llmmirec_caisd_phase2_beauty.sh 0 $S
done
```

### 4.2 TASID 敏感性分析

```bash
# lambda_tasid 扫描
DATASET=beauty SWEEP_TYPE=lambda VALUES="0 0.001 0.005 0.01 0.05 0.1" \
  bash new_bash/run_llmmirec_caisd_tasid_sensitivity.sh 0 42

# temperature 扫描
DATASET=beauty SWEEP_TYPE=temp VALUES="0.05 0.1 0.2 0.5" \
  bash new_bash/run_llmmirec_caisd_tasid_sensitivity.sh 0 42

# 汇总
python tools/summarize_llmmirec_caisd_tasid_ablation.py --dataset beauty --sweep_type lambda
```

### 4.3 多 seed 汇总

```bash
python tools/summarize_llmmirec_caisd_phase2.py --dataset beauty --seeds 0 1 42
python tools/summarize_llmmirec_caisd_phase2.py --dataset ml-1m --seeds 0 1 42
```

### 4.4 诊断工具

| 工具 | 用途 |
|------|------|
| `tools/analyze_llmmirec_interests.py` | 兴趣结构诊断（10 项指标） |
| `tools/analyze_llmmirec_aspcf.py` | ASPCF/CHIR 诊断（alpha 分布、sem/comp 余弦） |
| `tools/analyze_hsdir_benefit_factors.py` | HSDIR 用户级收益因素分析 |
| `tools/analyze_caisd_teacher_benefit.py` | teacher 收益 paired 分析 |
| `tools/analyze_llmmirec_caisd.py` | CAISD routing 冗余诊断 |
| `tools/test_llmmirec_*.py` | 各模块 CPU synthetic 单测 |

---

## 五、代码结构

```text
.
├── main.py                      # 统一入口 (--model_name 动态加载)
├── helpers/
│   ├── BaseReader.py            # 读取 train/dev/test.csv
│   ├── SeqReader.py             # 构建用户历史序列 + position
│   └── BaseRunner.py            # 训练/验证/测试/Early Stop
├── models/
│   ├── BaseModel.py             # BaseModel / GeneralModel / SequentialModel
│   └── sequential/
│       ├── LLMMIRec.py          # Phase 0 基线
│       ├── LLMMIRecASPCF.py     # 第三章
│       ├── LLMMIRecCAISD.py     # 第四章最终 (CAISD + TASID)
│       ├── LLMMIRecHSDIR.py     # 第四章探索
│       ├── LLMMIRecCHIR.py      # 第四章探索
│       ├── LLMMIRecCASIR.py     # 第四章探索
│       ├── llmmi_components.py  # ItemEncoder / Extractor / Aggregator / DualView
│       ├── llmmi_utils.py       # embedding loader / NaN 检查 / activation
│       └── PoMRec.py, MyModel*.py   # 历史代码
├── tools/                       # 诊断、单测、prototype 构建、汇总
├── new_bash/                    # 当前实验脚本
├── bash脚本/                     # 历史实验脚本 (PoMRec/MyModel)
├── analysis_figures/            # 论文图表脚本
├── AUDIT_REPORT.md              # 仓库审计报告
├── PHASE1_SUMMARY.md            # 第三章完成总结
└── PHASE2_SUMMARY.md            # 第四章完成总结
```

**继承链**：

```
BaseModel → GeneralModel (BPR + 负采样) → SequentialModel (历史序列)
                                              ├─ LLMMIRec
                                              ├─ LLMMIRecASPCF
                                              ├─ LLMMIRecCAISD
                                              └─ ...
```

---

## 六、数据格式

数据集位于 `data/<dataset_name>/`，至少包含：

```text
train.csv
dev.csv
test.csv
```

每行至少包含 `user_id`、`item_id`、`time`（tab 分隔，可用 `--sep` 覆盖）；dev/test 可含 `neg_items`。

`SeqReader` 合并三个 split、按时间排序构建 `user_his`，并为每条交互补充 `position`。

---

## 七、评估

支持 `HR@K` / `NDCG@K`，默认 `--topk 5,10,20,50 --metric NDCG,HR`。
Early stop 基于 `NDCG@5`。

日志与模型默认保存到：
```text
./log/<model_name>/      或  --log_file 指定
./model/<model_name>/    或  --model_path 指定
```

Phase 2 实验使用独立目录：
```text
new_log/llmmirec_{aspcf,caisd}_phase2/<dataset>/seed<seed>/
new_model/llmmirec_{aspcf,caisd}_phase2/<dataset>/seed<seed>/
```

---

## 八、历史代码（PoMRec / MyModel 系列）

以下为早期工作，保留用于复现，非当前主线：

| 模型 | 说明 |
|------|------|
| `PoMRec` | 原始多兴趣主干 + 可选 LLM 语义融合（`--use_llmemb`） |
| `MyModel` | PoMRec + LLM + IPD（目标兴趣一致性）+ LGD（意图引导软去噪） |
| `MyModelV2/V4/V5`, `SIERec`, `MyModel{TIRL,CTIRL,ITIRL,CIRF,SHNC,HMIF}` | 各类探索变体 |

历史实验脚本在 `bash脚本/`，详细参数见该目录下脚本。历史模型参数说明可参考 git 历史中的旧版 README。

---

## 九、注意事项

1. `--use_llmemb 1` 时必须提供 `--llm_emb_path`；LLM embedding 应为二维 `.pkl`，代码自动处理 padding 行。
2. ASPCF/CAISD 需要 `--llm_emb_path`；CAISD 的 semantic distillation 还需要 `--semantic_teacher_path`。
3. `tasid_mode=none` / `semantic_distill_mode=none` 时新模块完全跳过，退化为对应基线。
4. 修改数据文件后建议 `--regenerate 1` 重建 corpus 缓存。
5. GPU 控制：所有 `new_bash/` 脚本第一个参数为 GPU id，第二个为 seed。

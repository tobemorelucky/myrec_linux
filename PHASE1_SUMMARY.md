# LLMMIRec Phase 0–1 完成总结

> 2026-08-06 ~ 2026-08-08

## 一、架构

```
LLMMIRec (SequentialModel)
  ├─ ItemEncoder   ← 共享 (history + candidates, 4 modes)
  ├─ QueryMultiInterestExtractor  ← K learnable queries + scaled dot-product attention
  ├─ InterestAggregator           ← history-only: mean + last → MLP → weights
  └─ Dropout
```

无 PoMRec 依赖、无 monkey patch、无 pretrained checkpoint。

## 二、文件清单

### 新增模型文件

| 文件 | 说明 |
|------|------|
| `models/sequential/LLMMIRec.py` | 主模型: 4种encoder模式, relation loss, gate mode |
| `models/sequential/llmmi_components.py` | ItemEncoder(id/llm_replace/residual/aspcf), Extractor, Aggregator |
| `models/sequential/llmmi_utils.py` | Embedding loader, NaN/Inf checker, activation helper |

### 修改的文件

| 文件 | 变更 |
|------|------|
| `main.py` | +`import models.sequential.LLMMIRec as LLMMIRec` |

### 实验脚本

| 脚本 | 数据集 | 模式 |
|------|--------|------|
| `new_bash/run_llmmirec_phase0_beauty_seed42.sh` | Beauty | id / llm_replace / residual |
| `new_bash/run_llmmirec_phase0_ml1m_seed42.sh` | ML-1M | id / llm_replace / residual |
| `new_bash/run_llmmirec_phase0_toys_seed42.sh` | Toys | id / llm_replace / residual |
| `new_bash/run_llmmirec_phase1_aspcf_beauty_seed42.sh` | Beauty | ASPCF (λ=0 / λ=0.01) |
| `new_bash/run_llmmirec_phase1_aspcf_ml1m_seed42.sh` | ML-1M | ASPCF (λ=0 / λ=0.01) |
| `new_bash/run_llmmirec_phase1_relation_control_beauty_seed42.sh` | Beauty | llm_replace + relation |
| `new_bash/run_llmmirec_phase1_relation_control_ml1m_seed42.sh` | ML-1M | llm_replace + relation |
| `new_bash/run_llmmirec_phase1_gate_beauty_seed42.sh` | Beauty | ASPCF gate ablation (basic/conflict) |
| `new_bash/run_llmmirec_phase1_gate_ml1m_seed42.sh` | ML-1M | ASPCF gate ablation (basic/conflict) |

### 诊断工具

| 文件 | 说明 |
|------|------|
| `tools/analyze_llmmirec_interests.py` | Phase 0 兴趣诊断 (10 stats) |
| `tools/analyze_llmmirec_aspcf.py` | ASPCF 诊断 (alpha dist, sem/comp cos) |
| `tools/test_llmmirec_aspcf.py` | ASPCF 单 batch 验证 |

### 报告

| 文件 | 说明 |
|------|------|
| `AUDIT_REPORT.md` | 仓库审计报告 |
| `new_log/llmmirec_phase0/PCA_AUDIT.md` | PCA 嵌入审计 |

## 三、ItemEncoder 四种模式

| 模式 | 公式 | 参数量 (Beauty) |
|------|------|----------------|
| id | `e = id_emb[item]` | 793K |
| llm_replace | `e = adapter(llm_table[item])` | 1.20M |
| residual | `e = id_emb[item] + γ · adapter(llm_table[item])` | 1.20M |
| aspcf | `e = [√α_s·s ; √α_c·c]` (见下方) | ~943K |

## 四、ASPCF 数据流

```
llm_table[item_id] (1536-dim PCA)
  ├─ z_high = [:512]  (高方差主子空间)
  │    └─ Linear(512→128) → GELU → Linear(128→32) → s (semantic, 32-dim)
  │
  └─ z_low = [512:]   (尾部分量, 1024-dim)
       ├─ Linear(1024→64) → GELU → low_feat (64-dim)
       └─ complement_id_emb[item_id] → id_feat (64-dim)
            └─ concat → Linear(128→64) → GELU → Linear(64→32) → c (complement, 32-dim)

  ┌─ Gate: [s; c; (opt: |s-c|; s*c)] → MLP → softmax → [α_s, α_c]
  └─ e = concat[√(α_s+ε)·s, √(α_c+ε)·c]  (64-dim)
```

### Gate 模式

| 模式 | 输入维度 | 结构 |
|------|---------|------|
| `basic` | 64 `[s;c]` | Linear(64,64)→GELU→Linear(64,2)→softmax |
| `conflict` | 128 `[s;c;|s-c|;s*c]` | Linear(128,64)→GELU→Linear(64,2)→softmax |

## 五、Semantic Relation Preservation Loss

```
训练 batch 中的 unique items (≤128个)
  Teacher: frozen z[:, :semantic_rank] → L2 → M×M cosine sim / τ_t
  Student: 
    aspcf:   semantic_branch(z_high) → L2 → M×M cosine sim / τ_s
    llm_replace: adapter(z) → L2 → M×M cosine sim / τ_s

  去对角线 → 行 softmax → KL(stopgrad(teacher) || log_student)
  Total = BPR + λ_relation × relation_loss
```

仅 `self.training` 时计算；λ=0 时完全等价于无 relation loss。

## 六、实验参数

### Phase 0 统一参数

```
emb_size=64, attn_size=64, K=4, history_max=20
adapter_hidden=256, adapter_activation=gelu, adapter_use_ln=0
dropout=0.1, lr=0.001, l2=1e-6
batch_size=256, eval_batch_size=256, num_neg=1
epoch=200, early_stop=10
```

### Phase 1 参数

```
semantic_rank=512, semantic_dim=32, semantic_hidden=128
complement_dim=32
tail_hidden=64, complement_hidden=64, gate_hidden=64
aspcf_gate_mode=basic|conflict
lambda_relation=0.0|0.01
relation_sample_size=128, relation_teacher_temp=0.1, relation_student_temp=0.1
```

### Gate 消融参数

```
Beauty:  batch_size=2048, lr=0.008
ML-1M:   batch_size=2048, lr=0.002
```

## 七、已完成实验结果

### Phase 0 (三数据集)

| 数据集 | id NDCG@5 | llm_replace NDCG@5 | residual NDCG@5 |
|--------|-----------|-------------------|----------------|
| Beauty | 0.0832 | **0.1016** | 0.0946 |
| ML-1M | 0.2130 | 0.1795 | **0.2131** |
| Toys | 0.1187 | **0.1433** | ⚠️ 未完成 |

### Phase 1 Beauty (seed=42)

| 实验 | NDCG@5 |
|------|--------|
| Phase 0 llm_replace | 0.1016 |
| ASPCF λ=0 | 0.1003 |
| ASPCF λ=0.01 | **0.1100** |

### 对照基线 (旧模型, Beauty seed=42)

| 模型 | Test NDCG@5 |
|------|------------|
| PoMRec standard | 0.0986 |
| PoMRecLLMEmb replace | 0.1076 |
| LLMMIRec llm_replace | 0.1016 |
| LLMMIRec ASPCF λ=0.01 | **0.1100** |

### Beauty 兴趣诊断 (Phase 0)

| 指标 | id | llm_replace | residual |
|------|-----|-------------|----------|
| 有效秩 (K=4) | 2.20 | 1.80 | 2.16 |
| 注意力熵 | 0.359 | 0.843 | 0.644 |
| 权重熵 | 0.855 | 0.994 | 0.928 |

## 八、PCA 嵌入结论

- 原始: **4096** 维 → sklearn.PCA → **1536** 维
- 按方差降序排列，前 r 维 = 高方差主子空间
- `explained_variance_ratio_` 未保存，可从 `llm_table.pkl` 重新拟合
- 原始 4096 维表存在: `data/<dataset>/handled/llm_table.pkl`

## 九、Git 提交

```
1666c21 fix LLMMIRec --dropout conflict with parent GeneralModel
26073e5 implement clean LLMMIRec phase0 baseline
91c8c5a update audit report with corrected replace vs residual ablation
```

待提交: Phase 1 所有新增/修改文件

## 十、GPU 控制

所有脚本使用 `$1` 参数控制 GPU:

```bash
bash <script>.sh 0   # GPU 0
bash <script>.sh 1   # GPU 1
```

SMOKE=1 支持 2-epoch 冒烟测试。




当前进度可以概括为：第三章 ASPCF 已经稳定，第四章 HSDIR 的核心“历史侧语义结构监督”也已经被验证有效。 在 Beauty、相同训练配置下，ASPCF baseline 的 NDCG@5/10 为 0.1053/0.1269；加入 HSDIR 后，hierarchical 为 0.1091/0.1317，fine-only 为 0.1090/0.1311，coarse-only 只有 0.1065/0.1284。同时 hierarchical 的诊断显示，兴趣余弦从 0.9457→0.8691、有效秩从 1.714→2.280、路由 membership entropy 从 0.913→0.543、effective active K 从 3.833→2.632，并且 route 与 fine/coarse 语义结构的相关性从接近 0 提升到约 0.36/0.38。因此，“用 LLM 语义作为训练期结构教师，而不是直接控制推荐 attention”这个方向已经成立。但目前还有一个重要事实：hierarchical 和 fine-only 排序几乎一样，说明“层次结构”本身的额外价值还没有被证明，主要收益很可能来自 fine-level semantic relation distillation。
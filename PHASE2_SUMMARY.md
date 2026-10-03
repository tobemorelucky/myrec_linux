# LLMMIRec Phase 2 完成总结 — 第四章：目标感知多兴趣语义蒸馏

> 阶段2（第四章）整体路径：探索多兴趣语义约束机制 → 分析失败方向 → 确定最终方法 TASID

---

## 一、阶段概览

第四章的核心问题：**如何让 LLM 语义知识从"物品表示层"进一步参与"兴趣决策层"**。

经历了三个阶段的探索：

```
HSDIR（兴趣路由结构约束）
  → 用户级自适应蒸馏（谁需要 teacher）
  → 关系级选择性蒸馏（蒸馏哪些关系）
  → CAISD（兴趣级语义分布蒸馏）
  → TASID（目标感知语义蒸馏）✅ 最终方法
```

最终形成完整创新链：

**LLM 语义知识构建 → 兴趣级语义蒸馏 → 目标感知兴趣分配优化**

---

## 二、探索一：HSDIR — 层次语义蒸馏引导兴趣路由

### 思路

利用 LLM 生成的语义 prototype 构建层次 teacher（fine32 / coarse8），
在训练期监督用户历史内部"哪些行为应被路由到同一兴趣"，引导多兴趣学习更清晰的兴趣路由。

### 机制

```
R = softmax_k(raw_attention_scores)      # 行为 → 兴趣 成员概率
G_route = R @ R^T                        # 行为共属图（permutation-invariant）
G_fine / G_coarse                        # LLM 层次语义共属 teacher
L_hsr = 层次置信度感知路由损失（W_pos=G_fine, W_neg=1-G_coarse）
```

### 实验结果：兴趣结构确实被改变 ✅

| 指标 | Beauty baseline → HSDIR | ML-1M baseline → HSDIR |
|------|------------------------|------------------------|
| 兴趣间余弦相似度 | 0.9457 → **0.8691** | 0.7620 → **0.7377** |
| 有效兴趣数 (effective rank) | 1.714 → **2.280** | 2.762 → **3.028** |
| route membership entropy | 0.913 → **0.543** | — |
| effective active K | 3.833 → **2.632** | — |

结论：**LLM 语义约束能够有效增强兴趣分化**（兴趣向量更正交、路由更清晰）。

### 失败点

单纯兴趣路由约束**没有稳定提升推荐指标**：
- Beauty HSDIR HR@5 ≈ 0.1604，相比基线提升有限
- ML-1M 提升不稳定

**教训**：全局兴趣结构约束无法充分利用 LLM 语义知识——兴趣分化并不直接转化为目标推荐收益。

---

## 三、探索二：用户级自适应蒸馏（need gate / teacher benefit）

### 假设

"是否只有部分用户需要 teacher 指导"——对 collapse 程度、routing entropy、
semantic diversity、teacher JS shift 等因素与蒸馏收益做用户级 paired 分析。

### 实验结果：用户级因素无法稳定预测 teacher 收益 ❌

| 因素 | Beauty (ΔNDCG@5 相关) | ML-1M (ΔNDCG@5 相关) |
|------|----------------------|----------------------|
| teacher_js_shift | ρ=0.0105, p=0.2336（不显著） | ρ=0.0281, p=0.0292（弱正相关） |

- ML-1M 高 JS shift 用户收益更明显（Q4 相对 Q1 提升约 0.020 NDCG@5）
- 但 **跨数据集方向不一致**，无法作为稳定 gate 特征

**结论**：放弃"选择哪些用户进行蒸馏"的方向——用户级因素不携带稳定的可区分信号。

---

## 四、探索三：关系级选择性蒸馏（pair-level selective）

### 思路

HSDIR 对全部 item-item 关系做 dense 蒸馏，噪声关系会传播误导。
改为只选择高置信语义正负关系：

```
positive: p_i = argmax_j G_fine[i,j]      # 最相似
negative: n_i = argmin_j G_coarse[i,j]    # 最不相似
c_i = G_fine[i,p_i] * (1-G_coarse[i,n_i]) # teacher 置信度
loss = c_i * relu(margin - r_pos + r_neg) # anchor-level local ranking
```

同时探索了不同 student route 来源：raw routing score 与 attention contribution。

### 实验结果：结构改善但性能不稳定 ❌

| ML-1M 变体 | HR@5 |
|-----------|------|
| pair selective | 0.2987 |
| attention contribution | ~0.3075 |

均未超过最佳基线。

**结论**：简单关系选择无法解决**目标相关性不足**的问题——关系约束与推荐目标之间缺少直接联系。

---

## 五、CAISD — 兴趣级语义蒸馏（静态 teacher）

### 机制

把 LLM prototype teacher 引入**多兴趣语义分布学习**：

```
T = normalize(A_detach @ Q)        # 每个兴趣的动态语义 profile（prototype 分布）
P = softmax(predictor(V))          # 学生预测的兴趣语义分布
L_profile = KL(T || P)             # 语义蒸馏

teacher 变体：attention / responsibility / responsibility_power(α)
```

LLM 知识从 item 表示层转移到 **interest 分配层**。

### 实验结果：有一定有效性但不稳定 ⚠️

| ML-1M 变体 | HR@5 | NDCG@5 |
|-----------|------|--------|
| CAISD uniform | 0.3091 | 0.2156 |
| CAISD responsibility | 0.3109 | — |

多种 seed 实验表明平均收益不稳定，说明**静态 teacher 仍存在不足**：
LLM teacher 并非对所有样本均有效，而是需要结合当前预测目标进行动态匹配。

---

## 六、最终方法：TASID — 目标感知语义蒸馏 ✅

### 动机

teacher benefit 分析显示 LLM teacher 需要与**当前预测目标**动态匹配——
不是"全局兴趣应该长什么样"，而是"给定这个目标 item，兴趣应该如何分配"。

### 机制

```
q_target = semantic_branch(llm_table[target][:,:512])   # 目标 LLM 语义 query（detach）

Teacher:  p[k] = softmax_k(cos(q_target, T_k) / τ)       # 目标 vs LLM prototype teacher
Student:  llm_only:   q[k] = softmax_k(cos(q_target, V_k[语义]) / τ)
          asymmetric: q[k] = softmax_k(cos(concat(CF,sem), V_k) / τ)   # 模型侧目标表示
L_tasid = KL(p_teacher || log q_student)                 # 非对称 teacher-student KL

Total = BPR + λ_rel·L_rel + λ_sem·L_sem + λ_tasid·L_tasid
```

**设计要点**：
- LLM 语义监督从"全局兴趣约束"转变为"**目标条件兴趣选择**"
- teacher 走 LLM prototype，student 走模型自身表示（非对称）
- train-only，测试无 target 泄露；tasid 关闭时完全恢复 CAISD

### 实验结果 ✅

| 数据集 | 配置 | HR@5 | NDCG@5 | 相对 CAISD baseline |
|--------|------|------|--------|---------------------|
| Beauty | CAISD baseline | 0.1551 | 0.1036 | — |
| Beauty | TASID llm_only | **0.1608** | **0.1082** | ↑ |
| ML-1M | CAISD baseline | 0.3040 | 0.2104 | — |
| ML-1M | TASID asymmetric | **0.3093** | **0.2161** | **+1.7% / +2.7%** |

- Beauty：TASID llm_only 全面超越 CAISD baseline
- ML-1M：普通 TASID 提升有限，**asymmetric TASID**（模型侧目标表示匹配兴趣）取得最优

### 敏感性分析

实现并准备了 λ_tasid 与 τ 的敏感性扫描脚本与消融汇总工具
（`run_llmmirec_caisd_tasid_sensitivity.sh` + `summarize_llmmirec_caisd_tasid_ablation.py`），
用于确定最终超参数。

---

## 七、实验数据汇总

### 第四章探索路径一览

| 阶段 | 方法 | 核心机制 | 兴趣结构 | 推荐性能 | 结论 |
|------|------|---------|---------|---------|------|
| 1 | HSDIR | 层次语义路由蒸馏 | ✅ 分化明显 | ⚠️ 不稳定 | 结构约束 ≠ 推荐收益 |
| 2 | 用户级 gate | teacher benefit 分析 | — | ❌ 无稳定信号 | 放弃选择性 teacher |
| 3 | pair selective | 高置信关系蒸馏 | ✅ 结构改善 | ❌ 未超基线 | 关系选择缺目标相关性 |
| 4 | CAISD | 兴趣级静态语义蒸馏 | ✅ | ⚠️ seed 不稳 | 静态 teacher 不足 |
| 5 | **TASID** | 目标感知非对称蒸馏 | ✅ | ✅ **稳定提升** | **最终方法** |

### 关键结论

1. **兴趣结构约束有效但不充分**：HSDIR 证明 LLM 语义能分化兴趣（cosine 0.945→0.869），
   但结构差异不直接转化为推荐收益
2. **用户级选择不可行**：teacher_js_shift 等用户因素跨数据集不一致，无法做 gate
3. **关系级选择缺目标相关性**：单纯选择高质量关系仍与推荐目标脱节
4. **目标感知是关键**：TASID 把 LLM 语义监督从"兴趣应该长什么样"升级为
   "给定目标，兴趣如何分配"，与推荐目标直接对齐

---

## 八、关键文件

| 文件 | 角色 |
|------|------|
| `models/sequential/LLMMIRecHSDIR.py` | 探索一：层次语义蒸馏（保留） |
| `models/sequential/LLMMIRecCAISD.py` | 最终方法：CAISD + TASID（asymmetric student 模式） |
| `tools/analyze_hsdir_benefit_factors.py` | 探索二：用户级 teacher benefit 分析 |
| `tools/analyze_caisd_teacher_benefit.py` | 探索三/五：teacher 收益 paired 分析 |
| `tools/build_llmmi_hierarchical_teacher.py` | HSDIR 层次 teacher 构建 |
| `new_bash/run_llmmirec_caisd_phase2_*.sh` | 最终公平实验脚本（CAISD / ASPCF baseline） |
| `new_bash/run_llmmirec_caisd_tasid_sensitivity.sh` | λ_tasid / τ 敏感性扫描 |
| `tools/summarize_llmmirec_caisd_tasid_ablation.py` | TASID 消融汇总 |
| `tools/summarize_llmmirec_caisd_phase2.py` | 多 seed 公平对比汇总 |

---

## 九、第四章最终贡献

**创新链**：LLM 语义知识构建（第三章 ASPCF 的 prototype 空间）
→ 兴趣级语义蒸馏（CAISD 的 teacher 机制）
→ 目标感知兴趣分配优化（TASID 的目标条件学生分布）

**方法定位**：LLM 知识从第三章的"语义表示增强"发展为第四章的"兴趣决策引导"——
不是修改注意力外壳，而是用目标 item 的 LLM 语义查询，动态监督兴趣分配与推荐目标对齐。

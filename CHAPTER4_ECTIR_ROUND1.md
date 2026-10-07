# CHAPTER4_ECTIR_ROUND1.md

> Chapter 4 Round 1 — ECTIR 失败报告
> 日期：2026-10-07
> 状态：**ECTIR 主方向停止**。不跑 ML-1M、不调 transport 超参、不新增 loss 抢救。
>
> 代码位置：commit `e80e156`（用户提交）
> `models/sequential/LLMMIRecECTIR.py`、`llmmi_components.py` 的
> `TransportInterestRouter` / `TransportEvidenceAggregator`、
> `tools/test_llmmirec_ectir.py`

---

## 1. 结果（Beauty, seed 42）

| config | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | params |
|---|---|---|---|---|---|---|---|
| **ASPCF**（基线） | **0.1592** | **0.1088** | 0.2292 | 0.1313 | 0.3171 | 0.1535 | 943,174 |
| **ECTIR-1**（M1 transport） | 0.1578 | 0.1063 | 0.2273 | 0.1288 | 0.3150 | 0.1509 | **943,174** |
| **ECTIR-2**（M1+M2 refine） | 0.1589 | 0.1069 | 0.2311 | 0.1301 | 0.3199 | 0.1525 | 947,334 |
| **ECTIR-Full**（M1+M2+M3） | **0.1449** | **0.0976** | 0.2124 | 0.1194 | 0.2949 | 0.1401 | 951,427 |

**Δ vs ASPCF**（ASPCF 5-seed std：HR@5 ±0.0011、NDCG@5 ±0.0014）

| config | HR@5 | NDCG@5 | HR@10 | HR@20 | 以 σ 计（HR@5 / NDCG@5） |
|---|---|---|---|---|---|
| ECTIR-1 | −0.88% | −2.30% | −0.83% | −0.66% | −1.27σ / −1.79σ |
| ECTIR-2 | −0.19% | −1.75% | +0.83% | +0.88% | −0.27σ / −1.36σ |
| ECTIR-Full | −8.98% | −10.29% | −7.33% | −7.00% | **−13.00σ / −8.00σ** |

**训练成本**：ECTIR-1 912 s、ECTIR-2 818 s、Full 961 s（ASPCF 同配置 869 s）。
即 transport 开销约 **+5% ~ +11%**，与设计预估（+15~25%）一致或更低，**不是瓶颈**。

---

## 2. 结构诊断（核心证据）

| model | interest 两两余弦 | 有效秩 effR | 权重熵 wEnt | active K |
|---|---|---|---|---|
| ASPCF | 0.8823 | 1.103 | 1.3852 | 3.991 |
| **ECTIR-1** | **0.7182** | **1.437** | 1.3834 | 3.977 |
| **ECTIR-2** | 0.7365 | 1.297 | 1.3835 | 3.978 |
| **ECTIR-Full** | **1.0000** | **1.000** | 1.3863 | 4.000 |

（effR 上界 = K−1 = 3；wEnt 上界 = ln4 = 1.3863）

---

## 3. 三条结论

### 3.1 `transport routing` 明显改善了兴趣结构，但排序指标下降

- interest cosine **0.8823 → 0.7182**（−19%）
- effective rank **1.103 → 1.437**（+30%）

设计假设"transport 的全局联合分配能缓解兴趣冗余"**在机制层面被证实**。

但排序**同时下降**：HR@5 −0.88%，NDCG@5 −2.30%。

> ### 因此：**"降低 interest redundancy" 不是当前 ranking performance 的充分条件。**
> ### 不能再把 interest diversity 本身当成核心优化目标。

这条结论的说服力来自对照的干净程度：**ECTIR-1 与 ASPCF 参数量完全相同（943,174）**，
是"同参数、仅替换前向算子"的对照；唯一变化就是路由算子，而它带来的结构改善没有转化为收益。

### 3.2 iterative refinement 无明确增量

ECTIR-2 相对 ECTIR-1：HR@10/20/50 为正（+0.83% / +0.88% / +2.28%），
HR@5/NDCG@5 为负（−0.19% / −0.56%），**全部在 1.4σ 以内**。
且 ECTIR-2 的 effR（1.297）**低于** ECTIR-1（1.437），
即 refinement 反而略微**增加**了冗余。**无证据支持 iteration 有实质价值。**

### 3.3 evidence aggregation 当前实现发生严重退化

ECTIR-Full：`interest cosine = 1.0000`、`effective rank = 1.000`、
`wEnt = 1.3863`（恰为 ln4）、`active K = 4.000`。

即 **4 个 interest 向量完全相同**，聚合权重退化为均匀分布。
结构探测时还出现 `invalid value encountered in divide`（部分样本兴趣向量范数为 0）。

**后果**：−8.98% HR@5、−10.29% NDCG@5（−13σ / −8σ）。

**该方向停止。**

> ⚠️ **表述限制**：本次失败应限定为
> **"当前 `TransportEvidenceAggregator` 实现发生退化"**，
> **不可**扩大为"所有 evidence-aware aggregation 必然无效"。
> 理由见 §4.2（该实现存在一处 batch-global 归一化缺陷，尚未修复验证）。

---

## 4. 两处实现偏差（更正记录，**不重跑**）

用户复查代码后发现两处与设计文档不符的事实。经实测确认如下。

### 4.1 `W_r` 不是零初始化 —— "严格初始恒等"的说法不成立

**设计文档原话**（`CHAPTER4_DESIGN.md` §4.2）：
> `W_r` 近零初始化 ⇒ 训练初始时 `W_r(V) ≈ 0` ⇒ 第 1 次 refinement 与 Module 1 等价。

**实际情况**：`LLMMIRecECTIR.__init__` 在 `_define_params()` 之后调用
`self.apply(self.init_weights)`，而 `init_weights` 对所有 `nn.Linear` 执行
`nn.init.normal_(m.weight, std=0.01)`。**因此 `W_r` 被重新初始化为 N(0, 0.01²)，不是零。**

实测（构造真实模型后立即测量）：

```
W_r.weight 范数 = 0.642350   绝对均值 = 0.00790990
是否零初始化        = False
||W_r(V)|| (per-interest) = 0.639
||q||      (per-interest) = 0.079
=> refinement 相对幅度  = 8.05x
```

**refinement 项在初始化时比基础 query 大 8 倍**，即 query 实际由 `W_r(V)` 主导，
而非"q 主导、refinement 微调"。

**处置**：
- ❌ **不重跑**。理由：ECTIR-1 **不使用 `W_r`**，本身已无排序收益；
  ECTIR-2 也无明确增量。修正初始化不会改变"transport 无收益"这一主结论。
- ✅ 在报告中如实记录该实现偏差，**不再声称"严格初始恒等"**。
- 备注：单测 `test_refine_starts_as_identity` 验证的是**数学性质**
  （强制 `W_r = 0` 时 `n_refine=1` 等价于 `n_refine=0`，max_err = 0），
  而非实际模型的初始化状态。该测试本身没错，但不能用来支撑"实际初始恒等"。

### 4.2 `pos_max` 是 batch-global 归一化，不是 per-user

**代码**（`llmmi_components.py:956`）：
```python
pos_max = position.max().clamp(min=1.0)      # ← 标量，跨整个 batch
rec = (transport * position[:, :, None]).sum(dim=1) / mass.clamp(min=1e-8)
rec = (rec / pos_max).clamp(0.0, 1.0)
```

`position.max()` 无 `dim` 参数 ⇒ 返回**整个 batch 的标量最大值**，
而不是每个用户自己的 `lengths - 1`。

**实测影响**（混合长度 batch，L=20）：

| user length | per-user max | 正确归一 | 现状归一 | 低估倍数 |
|---|---|---|---|---|
| 2 | 2 | 1.00 | 0.10 | **10.0×** |
| 3 | 3 | 1.00 | 0.15 | **6.7×** |
| 5 | 5 | 1.00 | 0.25 | **4.0×** |
| 10 | 10 | 1.00 | 0.50 | 2.0× |
| 20 | 20 | 1.00 | 1.00 | 1.0× |

后果有两条：
1. **短历史用户的 recency 证据被系统性压低**（最多 10×）；
2. **归一化尺度依赖 batch 组成**（由 batch 内最大 length 决定），特征不稳定。

**处置**：
- ❌ **不重跑**。理由：Full 跌幅过大（−13σ），且主方向 ECTIR-1 本身未改善 ranking；
  修复该归一化不足以翻盘。
- ✅ 结论严格限定为"当前实现退化"，**不推广**为方法层面的否定。

---

## 5. 对照既定停止判据

`CHAPTER4_DESIGN.md` §8：

| 判据 | 命中情况 |
|---|---|
| "若 ① ≈ ⓪ → 路由算子变更本身无效 → **停止**" | ✅ **命中**。ECTIR-1 不仅 ≈ ASPCF，还略差（−1.27σ / −1.79σ） |
| "若 ② ≈ ⓪ → 瓶颈不在此" | ✅ 命中（Module 2 无增量） |
| "若 ③ ≈ ⓪ 但诊断显示 aggregation 退化 → 先确认不是实现问题" | ⚠️ 反例：③ **显著有害**（−13σ），原因已定位为结构塌缩 + §4.2 的实现缺陷，非单测可发现的 bug（56 项单测全过；`baseline` 模式逐位复现 ASPCF） |

**判定：ECTIR 主候选不成立，停止。** 不进入 ML-1M、不调 transport 超参、不以新增 loss 抢救。

---

## 6. 从本轮得到的可复用资产

| 资产 | 价值 |
|---|---|
| `TransportInterestRouter` | **可作为通用"兴趣分化"工具**保留。它是目前仓库中**唯一被证实能有效降低兴趣冗余**的模块（cosine −19%、effR +30%），未来若确需分化兴趣可直接复用 |
| `tools/test_llmmirec_ectir.py` | 56 项单测，含 padding/mass/NaN/梯度/边界，可作为后续路由模块的测试模板 |
| `ectir_mode=baseline` 的严格退化路径 | 证明"新模型文件内可逐位复现旧模型"，是消除配置漂移的工程范式 |
| 本轮结果 | HSDIR / CAISD / ECTIR 三者的**共同负证据**（见 §7） |

---

## 7. 与本章其他失败的合并观察

三轮 Chapter 4 尝试机制完全不同，却得到同一结果：

| 方法 | 机制 | 兴趣结构变化 | 排序变化 |
|---|---|---|---|
| HSDIR | 共属图蒸馏（训练期 loss） | cosine 0.9457 → 0.8691、effR 1.714 → 2.280 | 不稳定，无稳定增益 |
| CAISD | 语义 profile 蒸馏（训练期 loss） | — | 5-seed 未超 ASPCF |
| **ECTIR-1** | **全局最优传输（前向算子）** | **cosine 0.8823 → 0.7182、effR 1.103 → 1.437** | **−0.88%** |
| **ECTIR-Full** | + 证据聚合 | cosine → **1.0000**（完全塌缩） | **−8.98%** |

**"降低兴趣冗余"与"提升排序"之间没有单调关系**，在 Full 上甚至是反向的。

这强烈提示：**Beauty 上的排序瓶颈可能根本不在 interest 的多样性上。**
`CHAPTER4_DIAGNOSIS.md` 中的 F1/F2 把"可观测的退化"当成了"性能瓶颈"，
这一步推断很可能是错的——这是本轮最重要的方法论教训。

**并促使 Chapter 4 的核心问题被重新定义，见 `CHAPTER4_DIAGNOSIS.md` §6.4。**

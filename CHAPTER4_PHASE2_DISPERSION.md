# CHAPTER4_PHASE2_DISPERSION.md

> Chapter 4 Phase 2 — Interest Dispersion Probe
> 日期：2026-10-07
> 性质：**inference-only 诊断，不是新模型，不训练**
> 脚本：`tools/probe_aspcf_interest_dispersion.py`
> 数据：`diagnostics_aspcf_dispersion/beauty_dispersion_probe.json`
>
> ⚠️ 本文件回答一个问题：**当前 ASPCF 的 attention-weighted dispersion
> 是否包含额外 ranking signal？**
> 它**不是** Chapter 4 的最终方法。`mean + variance` **不是**本工作创新——
> PoMRec 已使用 dispersion（§6 给出代码级对照）。

---

## 1. 代码事实核对（以仓库实现为准，不依据论文描述）

在写任何诊断代码之前，逐行核对 §0 要求的三件事：

### 1.1 `QueryMultiInterestExtractor` 只输出 attention-weighted mean ✅

`models/sequential/llmmi_components.py:550-576`：

```python
K_mat = self.Wk(history_emb)      # [B, L, attn_size]
V_mat = self.Wv(history_emb)      # [B, L, D]          ← 线性投影
scores = torch.bmm(Q, K_mat.transpose(1, 2)) / math.sqrt(self.attn_size)
scores = scores.masked_fill(attn_mask, float("-inf"))
attn = F.softmax(scores, dim=-1)                      # [B, K, L]
attn = attn.masked_fill(torch.isnan(attn), 0.0)
interest_vectors = torch.bmm(attn, V_mat)             # [B, K, D]  ← 只有一阶矩
```

**没有二阶矩，没有任何 dispersion 项。** ✅

### 1.2 当前 value space 确实是 `X = extractor.Wv(history_emb_pos)` ✅

- `LLMMIRecASPCF.py:189`：`history_emb_pos = history_emb_raw + self.position_emb(position)`
- `LLMMIRecASPCF.py:192`：`history_emb_pos = self.dropout(history_emb_pos)`（eval 下恒等）
- `llmmi_components.py:552`：`V_mat = self.Wv(history_emb)` ← 输入即 `history_emb_pos`

⇒ `X = Wv(history_emb_pos)`，`[B, L, D]`。✅

### 1.3 PoMRec 中确实存在 variance 并加到 interest_vectors ✅

`models/sequential/PoMRec.py:245-263`：

```python
interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)
var = []
for kk in range(self.K):
    x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, kk:kk+1, :]) ** 2
    var_k = torch.matmul(attn_score[:, kk:kk+1, :], x_mean_2)
    var_k = torch.sqrt(var_k)
    var.append(var_k)
variance = torch.cat(var, 1)
interest_vectors = interest_vectors + self.lamb * variance
```

即 `std_k = sqrt(Σ_l A[k,l] (x_l − mu_k)²)`，逐维 RMS 偏差。
`self.lamb` 是构造期传入的 **float 标量**（`--lamb`，非可学习参数）。✅

### 1.4 稳定的 PoMRec 脚本 / 日志记录的实际超参

来源：`new_bash/run_pomrec_standard_3datasets_seed42.sh` +
`new_log/pomrec_standard/summary_seed42.tsv`。

| dataset | **K** | **prompt_num** | **lamb** | attn_size | n_layers | lr | test HR@5 / NDCG@5 |
|---|---|---|---|---|---|---|---|
| beauty | **4** | **3** | **4.0** | 8 | 2 | 0.002 | 0.1436 / 0.0986 |
| ml-1m | **2** | **3** | **1.0** | 8 | 2 | 0.001 | 0.3046 / 0.2074 |
| toys | 4 | 3 | 4.0 | 8 | 2 | 0.002 | 0.1613 / 0.1134 |

补充事实（静态核对）：`max_prompt = 5` 为**硬编码**；
`pad_len = max_prompt − prompt_num`，故 beauty 的 `prompt_pad` 宽 2、`prompt1/prompt2` 各 3 行。
该脚本**未传** `--use_llmemb`（默认 0）⇒ PoMRec baseline 用的是**纯 CF embedding**。

**与 ASPCF 的关键配置差异**：ASPCF 的 `attn_size = 64`，PoMRec 为 **8**。
（记录，不修改现有 ASPCF 参数。）

### 1.5 评估协议（已核实）

`beauty/dev.csv` 与 `beauty/test.csv` 均为 22363 行、`neg_items` 长度 **1000**，
`--test_all` 默认 0 ⇒ 每个用户 **1 正 + 1000 负 = C=1001**。
λ 选择在 dev 上做，test 只评一次。

---

## 2. Probe 实现与自检

按 `LLMMIRecASPCF.forward` 的真实路径复算（eval、无 dropout、不修改模型文件）：

```
A  = out["attention_maps"]                  # [B,K,L]
X  = model.extractor.Wv(history_emb_pos)    # [B,L,D]
mu = bmm(A, X)                              # [B,K,D]
var_k = Σ_l A[k,l]·(X_l − mu_k)²            # [B,K,D]
std_k = sqrt(var_k + eps)
p  = model.aggregator(history_emb_raw, lengths)     # [B,K]
u_mu  = Σ_k p_k·mu_k ;  u_std = Σ_k p_k·std_k
score_λ = <u_mu + λ·u_std, e_j> = <u_mu,e_j> + λ·<u_std,e_j>
```

**关键代数**：`score_λ` 对 λ **仿射**，且两项都与 λ 无关
⇒ **一次前向即可精确得到所有 λ**，无需重跑。

### 2.1 自检结果（全部逐位通过）

| 检查 | 要求 | 实测 |
|---|---|---|
| `max‖mu − model_interest_vectors‖` | ≤ 1e-6 | **0.000e+00** ✅ |
| `max‖score(λ=0) − model_prediction‖` | ≤ 1e-6 | **0.000e+00** ✅ |
| λ=0 test HR@5 / NDCG@5 | == ASPCF 0.1592 / 0.1088 | **0.1592 / 0.1088** ✅ |
| popularity 分桶复现 `CHAPTER4_DIAGNOSIS.md` §2 | 逐位 | 0.0528/0.0373 … 0.2725/0.1865 ✅ |
| history 分桶复现 §3 | 逐位 | [20,∞) 0.2394/0.1651 ✅ |

⇒ padding 通过 `A` 自然为 0（softmax 后为精确 0），数值稳定，无 NaN。

---

## 3. λ 扫描（**只在 DEV 上选**）

`λ ∈ {0, 0.25, 0.5, 1.0, 2.0, 4.0}`：

| λ | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| **0.0** | **0.1891** | **0.2680** | **0.3538** | **0.1300** | **0.1555** | **0.1772** |
| 0.25 | 0.1857 | 0.2625 | 0.3480 | 0.1282 | 0.1530 | 0.1746 |
| 0.5 | 0.1762 | 0.2489 | 0.3328 | 0.1206 | 0.1440 | 0.1652 |
| 1.0 | 0.1435 | 0.2087 | 0.2890 | 0.0965 | 0.1175 | 0.1377 |
| 2.0 | 0.0767 | 0.1204 | 0.1820 | 0.0500 | 0.0640 | 0.0796 |
| 4.0 | 0.0263 | 0.0440 | 0.0771 | 0.0166 | 0.0222 | 0.0305 |

**λ=0 在全部 6 个指标上最优，且每一个 λ>0 都单调更差。**

| λ | Δ HR@5 | Δ NDCG@5 |
|---|---|---|
| 0.25 | −1.80% | −1.38% |
| 0.5 | −6.82% | −7.23% |
| 1.0 | −24.1% | −25.8% |
| 2.0 | −59.4% | −61.5% |
| 4.0 | −86.1% | −87.2% |

**dev 选出的 λ\* = 0.0。**

---

## 4. TEST 结果（λ\*=0.0，其余为退化轮廓，**未在 test 上做选择**）

| λ | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| **0.0** | **0.1592** | **0.2292** | **0.3171** | **0.1088** | **0.1313** | **0.1535** |
| 0.25 | 0.1514 | 0.2218 | 0.3072 | 0.1041 | 0.1268 | 0.1483 |
| 0.5 | 0.1436 | 0.2081 | 0.2894 | 0.0973 | 0.1180 | 0.1384 |
| 1.0 | 0.1121 | 0.1672 | 0.2396 | 0.0746 | 0.0923 | 0.1106 |
| 2.0 | 0.0536 | 0.0899 | 0.1432 | 0.0344 | 0.0460 | 0.0593 |
| 4.0 | 0.0155 | 0.0291 | 0.0546 | 0.0094 | 0.0137 | 0.0201 |

**Δ vs λ=0**：λ=0.25 即 −4.90% HR@5 / −4.32% NDCG@5；λ=4.0 时几乎完全崩塌。

λ=0 行 = **frozen ASPCF 的精确复现**（0.1592/0.1088），可作为整条链路的自检。

---

## 5. 机制诊断

### A. interest dispersion 统计量

| 量 | 值 |
|---|---|
| `‖std_k‖` mean / std | 4.8478 / 1.0163 |
| `‖std_k‖` p05 / p50 / p95 | 3.0818 / 4.8795 / 6.4637 |
| `‖std_k‖` 在**同一用户 K 个 interest 之间**的相对标准差 | mean **0.1232**，p50 0.1018 |
| `‖std_k‖ / ‖mu_k‖` | **0.9751** |

**读数**：dispersion 的模长与 mean 的模长**几乎同量级**（比值 0.975），
所以它**不是小扰动**——这解释了为什么 λ=0.25 就已经造成 −4.3% 的下降。
同一用户 K 个兴趣之间的 dispersion 差异很小（相对 std 仅 0.12）。

### B. centrality vs dispersion（candidate 层面）

| | 值 |
|---|---|
| n_pairs | 2,818,816 |
| **Pearson** | **+0.0720** |
| **Spearman** | **+0.0721** |
| `score_centrality` std | 3.2802 |
| `score_dispersion` std | 1.8359 |

**读数**：两者**几乎正交**（只有 ~7% 重合）。
⇒ **dispersion 不是 centrality 的冗余复制品**，它确实是一个**独立的信号**。
（这一点非常重要，见 §7。）

### C. pairwise ranking signal（positive vs 全部 1000 个 negative）

| score | P(s(pos) > s(neg)) | mean margin |
|---|---|---|
| **dispersion** | **0.4716** | **−0.1586** |
| centrality | 0.8036 | +4.1301 |

**读数（本轮最关键的发现）**：
dispersion 是一个**反向信号**——
`P = 0.4716 < 0.5`，且 **mean margin 为负**。
即 **正样本的 dispersion 分数系统性地低于负样本**。

这不只是"没有信号"，而是**方向与相关性相反**。
它直接解释了 §3/§4 的单调退化：把反向信号按正系数加进去，越加越差。

> 注：raw margin 只在同一个 score 内解释。此处两个 score 的 std 不同
> （3.28 vs 1.84），**不可跨 score 比较绝对尺度**；结论只依赖
> `P(pos>neg)` 这个尺度无关量。

### D. popularity 分桶（NDCG@5，完整 λ 网格）

| bin | n | λ=0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| [1,4) | 3141 | 0.0373 | 0.0363 | **0.0380** | 0.0332 | 0.0214 | 0.0072 |
| [4,6) | 3056 | 0.0371 | 0.0358 | 0.0366 | 0.0359 | 0.0258 | 0.0143 |
| [6,8) | 2078 | 0.0444 | 0.0428 | 0.0413 | 0.0366 | 0.0164 | 0.0066 |
| [8,15) | 4317 | 0.0666 | 0.0649 | 0.0610 | 0.0509 | 0.0290 | 0.0095 |
| [15,369) | 9771 | 0.1865 | 0.1776 | 0.1634 | 0.1186 | 0.0474 | 0.0091 |

**读数**：退化**基本均匀**。唯一的正向例外是 λ=0.5 时 [1,4) 桶 +1.9%
（0.0373 → 0.0380）——但该桶 NDCG@5 绝对值接近 0、只有 3141 个样本，
且同时全表 NDCG@5 掉了 10.6%。**这不构成可信的互补信号。**

### E. history 长度分桶（NDCG@5）

| bin | n | λ=0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| [0,5) | 7162 | 0.1024 | 0.0997 | 0.0949 | 0.0768 | 0.0420 | 0.0123 |
| [5,10) | 10959 | 0.1040 | 0.0979 | 0.0902 | 0.0687 | 0.0316 | 0.0092 |
| [10,15) | 2315 | 0.1126 | 0.1092 | 0.1027 | 0.0722 | 0.0258 | 0.0044 |
| [15,20) | 791 | 0.1406 | 0.1311 | 0.1240 | 0.0828 | 0.0231 | 0.0054 |
| [20,∞) | 1136 | 0.1651 | 0.1623 | 0.1518 | 0.1167 | 0.0385 | 0.0058 |

**读数**：**"长历史中 dispersion 更可靠"不成立。**
λ=0.25 时各桶相对降幅为 −2.6% / −5.9% / −3.0% / −6.8% / −1.7%，
**没有随历史变长而系统性变小**（−1.7% 的 [20,∞) 与 −5.9% 的 [5,10) 同量级）。
λ 增大后所有桶一起塌。

---

## 6. ⚠️ PoMRec 代码级对照（**不是复现**）

必须明确：**本 probe 与 PoMRec 的 variance 机制"结构类比"但"完全同构"不成立。**

| 维度 | PoMRec | 本 probe (ASPCF) | 是否相同 |
|---|---|---|---|
| value space | `his_vectors_prompt1` = `get_item_emb(his)+p_emb` **再 concat prompt** — 原始嵌入，**无投影** | `X = Wv(history_emb_pos)` — **学习到的线性投影** | ❌ **不同** |
| 被注意的序列 | `L + max_prompt(=5)` 个位置（prompt 恒为 valid） | 仅 `L` 个历史位置 | ❌ **不同** |
| attention 打分 | `W2(tanh(W1(x)))`，attn_size=**8**，直接出 K 个 logit | `Wq(q)·Wk(x)`，attn_size=**64**，K 个可学习 query | ❌ **不同** |
| 有效性 mask | `history > 0` | `arange(L) < lengths` | ⚠️ 通常等价，非恒等 |
| 聚合 | `proj(distri_vectors).softmax()`，`distri_vectors` 来自**另一条 prompt2 注意力分支** | `InterestAggregator` = `LN(mean_his + last_his)` → MLP → softmax | ❌ **不同** |
| variance 公式 | `sqrt(Σ_l A_kl (x_l − mu_k)²)` | 同式（在 Wv 空间） | ✅ 相同 |
| `eps` 处理 | `sqrt(...)` **无 eps** | `sqrt(... + 1e-8)` | ⚠️ 数值细节差异 |
| λ 的位置 | `V_k ← mu_k + lamb·std_k`（**逐兴趣、extractor 内**） | `u_λ = Σ_k p_k mu_k + λ·Σ_k p_k std_k`（**聚合后**） | ✅ **代数等价** |
| λ 的形式 | 标量 float，跨 k / 跨维共享 | 标量 float，跨 k / 跨维共享 | ✅ 相同 |

**关于"λ 的位置"为什么等价**：
两边对 `V_k` 都是线性的，λ 又是跨 k 共享的标量，故

```
Σ_k p_k·(mu_k + lamb·std_k)  =  Σ_k p_k·mu_k  +  lamb·Σ_k p_k·std_k
```

即 PoMRec 的"先加后聚合"与 probe 的"先聚合后加"**在数学上完全相同**。

**结论**：本 probe 是 PoMRec dispersion 的**类比实现**，
**不是复现**——value space、被注意序列、注意力函数、聚合器四处都不同，
且 `attn_size` 也不同（64 vs 8）。
因此**不能声称"我们复现了 PoMRec 的 variance"**，
只能说"我们在 ASPCF 自己的价值空间里构造了一个同公式的二阶统计量"。

---

## 7. 判定（按 §9 标准）

### 判定标准 A：

> 如果 Beauty dev 上所有 λ>0 都明显弱于 λ=0：停止 dispersion 方向。
> 不要训练。不要跑 ML-1M。

**✅ A 成立，且是强成立**：

| 条件 | 实测 |
|---|---|
| 所有 λ>0 弱于 λ=0？ | ✅ **6 个指标 × 5 个 λ>0，全部更差**（无例外） |
| dev 最优 λ | **0.0** |
| 最温和的 λ=0.25 的代价 | dev −1.38% NDCG@5；test **−4.32% NDCG@5 / −4.90% HR@5** |

⇒ **结论**：

> **停止 dispersion 方向。不训练。不跑 ML-1M。**

### 判定标准 B / C 均不成立

- **B 不成立**：dev 上不存在"稳定改善"的任何 λ。
- **C 不成立**：虽然 §5B 显示 dispersion 与 centrality **几乎正交（r=0.072）**——
  即它**确实是一个互补（非冗余）信号**，表面上符合 C 的前半段——
  但 **§5C 表明这个互补信号是反向的**（`P=0.4716 < 0.5`，margin 为负），
  且 **ranking 并非持平而是单调恶化**，C 的两个前提都不满足。

### 本轮真正的机制发现

dispersion 的失败**不是"没有信息"**，而是：

| 事实 | 证据 |
|---|---|
| dispersion 与 centrality **几乎正交** | Pearson/Spearman **+0.072** |
| 但它与相关性**反号** | `P(s_disp(pos)>s_disp(neg))` = **0.4716**；mean margin **−0.1586** |
| 且它与 mean **同量级**，不是小扰动 | `‖std_k‖/‖mu_k‖` = **0.9751** |
| ⇒ 加进去必然单调变差 | §3/§4 的严格单调退化 |

**为什么反号，本轮不做归因**（需要单独诊断）。
一个可检验的**假设**（**本轮未验证，也未据此调参**）：
ASPCF 的兴趣向量是"软平均"，正样本往往对应**行为集中**的用户-物品对，
其注意力分布更尖 → 二阶矩更小；负样本来自全库随机，注意力更发散 → 二阶矩更大。
这只是假设，**未经任何实验支持**。

> ⚠️ 明确记录：**本轮没有尝试负 λ**（即 `u_mu − λ·u_std`）。
> 看完 `P=0.4716` 之后再去扫负 λ 属于**在同一次数据上事后拟合**，
> 会立刻退化成"挑一个能涨的数字当方法"。若将来要验证该假设，
> 必须作为**独立的一轮**、用独立的选择准则重新走一遍流程。

---

## 8. 对 Chapter 4 全局诊断的补充

| 已确立事实 | 本轮补充 |
|---|---|
| F1 interests 冗余（effR 仅上界 37%/54%） | 未推翻 |
| F2 Beauty 上聚合权重恒均匀（wEnt=ln4） | 未推翻 |
| **F6 打分前压成单一 `u`** | **再次确认，且新增：换用二阶统计量也无效** |

**Phase 2 的净结论**：
在 ASPCF 的 value space 里，**一阶统计量（mean）已经是该空间里可用的最有用的统计量**；
二阶统计量（dispersion）虽然与一阶**正交**，但其方向与相关性**相反**，
因此无论以何种权重加入都会降低排序。

结合 PPCIM Round 1.5（改变**兴趣如何参与打分**无效且越彻底越差），
Chapter 4 目前已有**两条相互独立的负面证据**指向同一个判断：

> **瓶颈不在"兴趣的组织方式"，也不在"用兴趣的哪一阶统计量"。**

下一步（**本轮不执行**）：回到 `CHAPTER4_DIAGNOSIS.md` 的 **F4/F5**——
Beauty 上 ASPCF 的增益集中于长尾 item，而 ML-1M 无头部可优化——
提示应考察 **item 表示与 candidate 打分之间的几何**，而非 interest 侧。

---

## 9. 本轮禁止事项执行情况

未训练任何模型；未跑 ML-1M；未跑 Toys；未改 `LLMMIRecASPCF.py`；
未加新 loss；未加 Gaussian/Wasserstein 模型；未加 gate；未加 residual；
未做大规模参数扫描（λ 网格是用户指定的 6 个值，且在 dev 上选）；
未开始 Chapter 5；未 commit；未 push。

# CHAPTER4_DIAGNOSIS.md

> Chapter 4 Phase 0 诊断报告
> 日期：2026-10-07
> 性质：**不训练**，全部基于已有 checkpoint 的推理分析 + 一个 K 对照实验。
> 脚本：`tools/diagnose_ch4_phase0.py`、`new_bash/run_ch4_phase0_diagnosis.sh`

---

## 0. 方法学与有效性验证

### 0.1 诊断对象

| 模型 | checkpoint | 配置 |
|---|---|---|
| PoMRec | `new_model/pomrec_standard/<ds>/PoMRec_seed42.pt` | beauty: lr=0.002, bs=256, K=4, attn=8, n_layers=2, prompt=3, lamb=4.0, dropout=0；ml-1m: **K=2, lamb=1.0** |
| LLMMIRec-ID | `new_model/llmmirec_phase0/<ds>/id/...` | lr=0.001, bs=256, K=4, dropout=0.1 |
| LLMMIRecASPCF | `new_model/llmmirec_aspcf_phase2/<ds>/seed42/...` | beauty lr=0.004 / ml-1m lr=0.001, bs=1024, K=4 |
| LLMMIRecHSDIR | `new_model/llmmirec_hsdir_phase1/<ds>/...` | beauty lr=0.008 / ml-1m lr=0.002, bs=1024 — **不同配置，仅供参考** |

**这些 checkpoint 来自不同的训练配置，不是受控对比。** 表格中的配置列必须与数字一起引用。

### 0.2 有效性验证（重要）

诊断复算的指标与训练日志**逐位一致**：

| 模型 | 诊断 (Beauty) | 训练日志 | 诊断 (ML-1M) | 训练日志 |
|---|---|---|---|---|
| PoMRec | 0.1436 / 0.0986 | 0.1436 / 0.0986 ✅ | 0.3046 / 0.2074 | 0.3046 / 0.2074 ✅ |
| LLMMIRec-ID | 0.1210 / 0.0832 | 0.1210 / 0.0832 ✅ | 0.3065 / 0.2130 | 0.3065 / 0.2130 ✅ |
| ASPCF | 0.1592 / 0.1088 | 0.1592 / 0.1088 ✅ | 0.3068 / 0.2134 | 0.3068 / 0.2134 ✅ |

PoMRec 的 attention 复算带运行时自检（未修改 `PoMRec.py`，用带断言的复现实现）：
`max_err = 0.00e+00`。

### 0.3 两个必须注意的方法学提醒

1. **PoMRec 的 K 是数据集相关的**（beauty K=4 / ml-1m K=2），LLMMIRec 系列统一 K=4。
   ML-1M 上"兴趣数"并不对齐。
2. **`effR` 的上界是 K−1**，不能跨不同 K 直接比较。
   PoMRec-ML1M 的 `effR = 1.000` 是 K=2 的**最大值**（表示该模型内两个兴趣最大程度分散），
   与 Beauty 上 PoMRec 的塌缩**语义相反**。
   跨模型比较时应使用 `effR / (K−1)`。

---

## 1. 总体指标（test split）

**Beauty**

| model | HR@5 | NDCG@5 | NDCG@10 | NDCG@20 | icos | effR | attnH | wEnt | actK | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| PoMRec | 0.1436 | 0.0986 | 0.1169 | 0.1339 | 0.6208 | 1.000 | 1.3585 | **0.0000** | **1.000** | −4.8516 |
| LLMMIRec-ID | 0.1210 | 0.0832 | 0.1000 | 0.1160 | 0.4991 | 1.065 | 0.6672 | 1.1842 | 3.134 | −6.6674 |
| **ASPCF** | **0.1592** | **0.1088** | **0.1313** | **0.1535** | 0.8699 | 1.110 | 1.5051 | 1.3854 | 3.993 | **−4.2572** |
| HSDIR † | 0.1557 | 0.1064 | 0.1294 | 0.1507 | 0.9383 | 1.262 | 1.5956 | 1.3852 | 3.992 | −4.5921 |

**ML-1M**

| model | HR@5 | NDCG@5 | NDCG@10 | NDCG@20 | icos | effR | attnH | wEnt | actK | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| PoMRec (K=2) | 0.3046 | 0.2074 | 0.2481 | 0.2820 | −0.1831 | 1.000 ‡ | 1.2741 | 0.5474 | 1.603 | −2.6594 |
| LLMMIRec-ID | 0.3065 | 0.2130 | 0.2546 | 0.2876 | 0.7900 | 1.443 | 2.5528 | 1.2942 | 3.468 | −2.6914 |
| **ASPCF** | 0.3068 | 0.2134 | 0.2538 | 0.2861 | 0.7876 | 1.615 | 2.4691 | 1.2865 | 3.424 | **−2.4520** |
| HSDIR † | 0.3078 | 0.2144 | 0.2537 | 0.2854 | 0.7598 | 1.707 | 2.4859 | 1.2839 | 3.418 | −2.4538 |

† HSDIR 配置不同，不可直接比较。‡ K=2，effR=1.000 为最大值。

---

## 2. 按 target popularity 分桶（train 频次五分位）

**Beauty**（HR@5 / NDCG@5）

| popularity | n | PoMRec | LLMMIRec-ID | **ASPCF** | HSDIR |
|---|---|---|---|---|---|
| [1,4) | 3141 | 0.0169/0.0086 | 0.0067/0.0034 | **0.0528/0.0373** | 0.0360/0.0232 |
| [4,6) | 3056 | 0.0245/0.0152 | 0.0164/0.0094 | **0.0524/0.0371** | 0.0448/0.0295 |
| [6,8) | 2078 | 0.0428/0.0250 | 0.0332/0.0207 | **0.0679/0.0444** | 0.0645/0.0425 |
| [8,15) | 4317 | 0.0938/0.0604 | 0.0676/0.0437 | 0.0996/0.0666 | **0.1056/0.0687** |
| [15,369) | 9771 | 0.2650/0.1860 | 0.2328/0.1627 | **0.2725/0.1865** | 0.2704/0.1874 |

**ASPCF 相对 PoMRec 的增益倍率**：

| bucket | PoMRec | ASPCF | 倍率 |
|---|---|---|---|
| [1,4) | 0.0169 | 0.0528 | **3.13×** |
| [4,6) | 0.0245 | 0.0524 | **2.14×** |
| [6,8) | 0.0428 | 0.0679 | 1.59× |
| [8,15) | 0.0938 | 0.0996 | 1.06× |
| [15,369) | 0.2650 | 0.2725 | 1.03× |

→ **Beauty 上 ASPCF 的优势集中在长尾/sparse item**，头部几乎持平。

**ML-1M**（HR@5 / NDCG@5）

| popularity | n | PoMRec | LLMMIRec-ID | ASPCF | HSDIR |
|---|---|---|---|---|---|
| [1,23) | 75 | 0.0133/0.0067 | 0.0000/0.0000 | 0.0400/0.0182 | **0.0400/0.0333** |
| [23,74) | 265 | 0.0340/0.0193 | **0.0943/0.0555** | 0.0868/0.0581 | 0.0830/0.0608 |
| [74,186) | 711 | 0.1167/0.0732 | **0.1660/0.1076** | 0.1350/0.0911 | 0.1449/0.0958 |
| [186,423) | 1305 | **0.2100/0.1304** | 0.2192/0.1475 | 0.1954/0.1275 | 0.2023/0.1335 |
| [423,3388) | 3684 | 0.3998/0.2781 | 0.3860/0.2722 | **0.4007/0.2825** | 0.3982/0.2807 |

→ **ML-1M 上 PoMRec 并不占优头部**（[423,3388) 反被 ASPCF 小幅超过）。
PoMRec 唯一明显领先的是**中流行度段 [186,423)**（0.2100 vs 0.1954，+7.5%）。
LLMMIRec-ID 在中低流行度段反而最强。

---

## 3. 按 history 长度分桶（HR@5 / NDCG@5）

**Beauty**

| 长度 | n | PoMRec | LLMMIRec-ID | ASPCF | HSDIR |
|---|---|---|---|---|---|
| [0,5) | 7162 | 0.1254/0.0874 | 0.0996/0.0695 | 0.1509/0.1024 | **0.1516/0.1044** |
| [5,10) | 10959 | 0.1371/0.0936 | 0.1107/0.0762 | 0.1509/0.1040 | **0.1513/0.1045** |
| [10,15) | 2315 | 0.1598/0.1085 | 0.1460/0.0973 | **0.1711/0.1126** | 0.1542/0.1009 |
| [15,20) | 791 | 0.1884/0.1265 | **0.2023/0.1320** | 0.1985/0.1406 | 0.1719/0.1088 |
| ≥20 | 1136 | **0.2562/0.1768** | 0.2491/0.1745 | 0.2394/0.1651 | 0.2157/0.1472 |

→ ASPCF 在**短历史**上收益最大（[0,5) 相对 PoMRec +20.3%）；在**超长历史**上反而落后 PoMRec。

**ML-1M**

| 长度 | n | PoMRec | LLMMIRec-ID | ASPCF | HSDIR |
|---|---|---|---|---|---|
| [15,20) | 86 | 0.4070/0.2619 | 0.3721/0.2602 | 0.4070/0.2782 | 0.4070/**0.2883** |
| ≥20 | 5954 | 0.3032/0.2066 | 0.3055/0.2123 | 0.3053/0.2124 | **0.3063/0.2133** |

→ **ML-1M 几乎全部用户历史 ≥20**（5954/6040），分桶退化。该区间内四者基本持平，
**不存在"PoMRec 优势集中在长历史用户"这一现象**。

---

## 4. 结构指标分位数（判断多兴趣头是否塌缩）

**Beauty**

| model | icos p05/p50/p95 | effR p05/p50/p95 | wEnt p05/p50/p95 | actK p50/p95 |
|---|---|---|---|---|
| PoMRec (K=4) | 0.46/0.63/0.75 | 1.00/1.00/1.00 | **0.00/0.00/0.00** | **1.00/1.00** |
| LLMMIRec-ID | 0.23/0.49/0.78 | 1.00/1.02/1.32 | 0.75/1.27/1.38 | 3.26/3.97 |
| ASPCF | 0.69/0.89/0.98 | 1.01/1.09/1.28 | **1.38/1.39/1.39** | 4.00/4.00 |
| HSDIR | 0.81/0.96/1.00 | 1.03/1.19/1.78 | 1.39/1.39/1.39 | 3.99/3.99 |

**ML-1M**

| model | icos p05/p50/p95 | effR p05/p50/p95 | wEnt p05/p50/p95 | actK p50/p95 |
|---|---|---|---|---|
| PoMRec (K=2) | −0.39/−0.18/0.00 | 1.00/1.00/1.00 ‡ | 0.29/0.60/0.62 | 1.70/1.74 |
| LLMMIRec-ID | 0.65/0.80/0.90 | 1.21/1.42/1.76 | 1.11/1.32/1.37 | 3.58/3.91 |
| ASPCF | 0.63/0.80/0.91 | 1.19/1.59/2.14 | 1.10/1.32/1.38 | 3.52/3.94 |
| HSDIR | 0.59/0.77/0.90 | 1.23/1.69/2.26 | 1.08/1.32/1.38 | 3.52/3.93 |

‡ K=2，1.000 为最大值。

**两个已核实为真实（非 bug）的现象**：

1. **Beauty 上 PoMRec 的多兴趣头完全塌缩**：
   聚合权重在所有样本上**恰好 one-hot**（wEnt = 0.000 在 p05/p50/p95 全为 0），
   K=4 个 interest 向量中 **3 个完全相同**（两两余弦 = 1.0，范数 22.72/22.72/4.85/22.72），
   奇异值 `[19.44, 1e-4, 0, 0]`。即 PoMRec 在 Beauty 上**实际是单兴趣模型**。

2. **Beauty 上 ASPCF 的聚合权重恰好均匀**：
   `wEnt = 1.3854 ≈ ln 4 = 1.3863`，且 **p05 = p50 = p95**。
   即 `InterestAggregator` 在所有样本上输出几乎完全相同的均匀分布——
   **它没有在做选择，退化为对 4 个兴趣取平均**。

3. **ASPCF 的 interest 向量高度冗余**：
   Beauty `effR = 1.110`，上界 `K−1 = 3` → 仅 **37%**；
   ML-1M `effR = 1.615` → 仅 **54%**。两两余弦高达 0.87 / 0.79。

---

## 5. K=2 对照实验（fair-tuning baseline）

**动机**：PoMRec 的标准配置按数据集调 K（beauty K=4，ml-1m K=2）。
ASPCF 应被允许同样的 K 选择。**这是公平调参，不是方法创新。**
（`CHAPTER4_DESIGN.md` §0.2）

**设置**：除 `--K 4 → 2` 外，与 frozen ASPCF 稳定配置**逐字相同**。
静态核对 32 个 flag、运行时日志核对 41 项参数，**唯一差异均为 K**。

### 5.1 Beauty

| metric | K=4 (seed42) | K=2 (seed42) | Δ |
|---|---|---|---|
| HR@5 | 0.1592 | **0.1605** | +0.82% |
| HR@10 | 0.2292 | 0.2294 | +0.09% |
| HR@20 | 0.3171 | 0.3153 | −0.57% |
| NDCG@5 | 0.1088 | **0.1090** | +0.18% |
| NDCG@10 | 0.1313 | 0.1312 | −0.08% |
| NDCG@20 | 0.1535 | 0.1528 | −0.46% |

**→ Beauty：K=2 与 K=4 实质持平。** 全部差异在 ±0.8% 内，HR@5 的 +0.0013 约 1.2σ
（K=4 的 5-seed std = 0.0011）。K=2 虽略高，但只有 1 个种子，不能据此认为更好。

### 5.2 ML-1M

| metric | K=4 (seed42) | K=2 (seed42) | Δ | K=4 5seed mean±std |
|---|---|---|---|---|
| HR@5 | 0.3068 | **0.2972** | **−3.13%** | 0.3077 ± 0.0025 |
| HR@10 | 0.4321 | 0.4276 | −1.04% | 0.4329 ± 0.0017 |
| HR@20 | 0.5601 | 0.5593 | −0.14% | 0.5634 ± 0.0027 |
| NDCG@5 | 0.2134 | **0.2080** | **−2.53%** | 0.2140 ± 0.0023 |
| NDCG@10 | 0.2538 | 0.2501 | −1.46% | 0.2544 ± 0.0021 |
| NDCG@20 | 0.2861 | 0.2834 | −0.94% | 0.2875 ± 0.0022 |

**→ ML-1M：K=2 明显弱于 K=4。**
偏离 K=4 的 5-seed 分布：HR@5 **z = −4.20σ**，NDCG@5 **z = −2.57σ**，远超种子噪声。

附带观察：K=2 需要 172 epochs（110 分钟）才 early stop，K=4 seed42 只需 92 epochs（71 分钟）；
且 K=2 的 best dev 更高（0.3290 vs 0.3262）而 test 更低，dev/test 分歧更大。

### 5.3 结论

1. **K 是普通超参数，不是论文创新。** 不写成 Adaptive/Dynamic Interest Cardinality。
2. **两个数据集后续统一使用 K=4 的 ASPCF baseline**（Beauty 持平、ML-1M 更优）。
3. **ML-1M 的剩余性能差距不能通过简单匹配 PoMRec 的 K=2 解决**——
   匹配 K 反而让 ASPCF 变差（−3.13%）。
4. 附带发现（第五次数据集依赖现象）：**同一个 K 改动在两个数据集上方向相反**
   （Beauty 持平 / ML-1M −3.1%）。

---

## 6. 对 Chapter 4 的直接输入

### 6.1 已确立的事实（作为 Chapter 4 的问题基础）

| # | 事实 | 证据 |
|---|---|---|
| F1 | 现有 interests 之间**没有显式竞争**，产生冗余表示 | ASPCF effR = 1.110 / 1.615，仅为上界 (K−1=3) 的 37% / 54% |
| F2 | Beauty 上 ASPCF 的聚合权重**恒为均匀分布**，未做选择 | wEnt = 1.3854 ≈ ln4，p05 = p50 = p95 |
| F3 | Beauty 上 PoMRec 的多兴趣头**完全塌缩为单兴趣** | wEnt = 0，actK = 1.00，4 个 interest 中 3 个相同 |
| F4 | ML-1M 的差距**不在头部、不在长历史**，而在中流行度段 | §2、§3 |
| F5 | Beauty 上 ASPCF 的增益**集中在长尾/sparse item** | [1,4) 3.13×，头部 1.03× |

### 6.2 由事实导出的结构方向

- F1 + F2 → **routing 与 aggregation 两个前向环节都有问题**，需要同时改（对应 §6.3 的模块划分）
- F3 → "多兴趣"在这一任务上天然倾向于塌缩，**必须靠结构强制分配**，而不是靠 loss 鼓励
- F4 → ML-1M 没有一个"头部"可以重点优化；改进必须是全局性的

### 6.3 判定：属于情形 C（两者都有问题）

按 `CHAPTER4_DESIGN.md` 既定的 A/B/C 分支：

- **A**（routing 是瓶颈）、**B**（aggregation 是瓶颈）均**部分成立**；
- F1 指向 routing 缺竞争，F2 指向 aggregation 失效；
- 因此取 **C：再规划两阶段结构** —— 即 Chapter 4 的核心由
  **routing + aggregation 两个前向模块**共同构成。

---

### 6.4 ⚠️ 修订（2026-10-07）：比 interest redundancy 更重要的代码级瓶颈

> 本节由 ECTIR Round 1 的失败直接导出（`CHAPTER4_ECTIR_ROUND1.md`），
> **推翻了 §6.2 原先的方向判断**。

#### 6.4.1 事实 F6：所有 candidate 共用同一个 user vector

代码审计（全部 7 个模型逐一核对 scoring path）：

| 模型 | 文件:行 | 打分路径 |
|---|---|---|
| LLMMIRecASPCF | `:200-201` | `user_vector = Σ_k w_k V_k` → `pred_j = <user_vector, e_j>` |
| LLMMIRecHSDIR | `:293-294` | 同上 |
| LLMMIRecCAISD | `:409-410` | 同上（`V`） |
| LLMMIRecCASIR | `:275-276` | 同上（`V_refined`） |
| LLMMIRecCGSCD | `:234-235` | 同上 |
| LLMMIRecCHIR | `:359-360` | 同上 |
| LLMMIRecECTIR | `:279-280` | 同上 |

**七个模型逐字相同**：

```
interest_weights = history_only_aggregator(H)        # 只看 history，与 candidate 无关
u                = Σ_k w_k · V_k                     # ★ 打分前先压成单一向量
score_j          = uᵀ e_j                            # 所有 candidate 共用同一个 u
```

**K 个 interest 在打分之前就被压缩成一个 candidate-independent 的向量 `u`。**

#### 6.4.2 这解释了为什么改变 `V_k` 的结构不起作用

HSDIR / CAISD / ECTIR 都显著改变了 `V_k` 的结构（cosine、effR、routing 熵），
但这些改动**在被压缩成同一个 `u` 之后，候选侧无法区分**。
结构改变确实发生了，只是它对 `score_j` 的影响路径被 `u` 这个瓶颈截断了。

**ECTIR Round 1 的实测是该判断的直接证据**：

| | interest cosine | ranking |
|---|---|---|
| ASPCF | 0.8823 | 0.1592 / 0.1088 |
| ECTIR-1 | **0.7182**（−19%） | 0.1578 / 0.1063（**下降**） |
| ECTIR-Full | **1.0000**（完全塌缩） | 0.1449 / 0.0976（**−8.98%**） |

**冗余度与排序指标之间没有单调关系**，完全塌缩时才崩，而改善冗余时也不涨。

#### 6.4.3 Chapter 4 核心问题的修订

> **原问题（已废弃）**：
> ~~"如何让多个 interests 更分散"~~
>
> **修订后的问题**：
> ### **"如何让多个 interests 真正参与 candidate-specific ranking，
> ### 而不是在打分前重新压缩成单一 user vector"**

**判定依据的变化**：

| | 旧判断（§6.2） | 新判断（§6.4） |
|---|---|---|
| 瓶颈位置 | interest 冗余（routing 层） | **打分前的向量压缩（`Σ_k w_k V_k`）** |
| 证据 | F1 effR 仅为上界 37% | **F6 全部 7 模型共用 `u`** + ECTIR 的结构-性能解耦 |
| 方向 | 让兴趣竞争/分化 | **让 candidate 直接与 K 个 interest 交互** |

**ECTIR / HSDIR 保留为该判断的 negative evidence**：
它们证明了"只改 `V_k` 结构、不改打分路径"是不够的。

---

## 7. ⚠️ Unresolved Issue（必须在最终正式基线阶段单独追溯）

> **本仓库的 PoMRec ML-1M checkpoint 低于论文/最终冻结目标，
> 当前不能把这直接归因为协议差异。**

事实：

| | HR@5 | NDCG@5 |
|---|---|---|
| 本仓库复现 PoMRec (ML-1M, seed42) | 0.3046 | 0.2074 |
| `FINAL_EXPERIMENT_TARGETS.md` 中 PoMRec (ML-1M) | **0.3151** | **0.2188** |
| 差距 | **−3.33%** | **−5.21%** |

含义：

- ASPCF K=4 相对**本仓库复现的** PoMRec 是正的（+0.72% / +2.89%）；
- 相对**论文目标值**是负的（−2.63% / −2.47%）；
- 而本仓库的 PoMRec 复现本身就差 3.33%/5.21%。

**这 3.33%/5.21% 的缺口目前来源不明。** 可能来自：预处理/划分差异、
评测细节、超参未调到位、或论文报告值的口径不同。**当前不做归因。**

**处置**：
1. 记为 unresolved，**不得在论文中直接写成"协议差异"**；
2. **最终正式基线阶段必须单独追溯**（用本仓库协议重跑 PoMRec 并做消融式排查）；
3. `FINAL_EXPERIMENT_TARGETS.md` 中的高目标（PoMRec / SATCRec 的论文数值）**保持不变**，
   不因本仓库复现偏低而下调。

---

## 8. 产出文件

```
tools/diagnose_ch4_phase0.py
new_bash/run_ch4_phase0_diagnosis.sh
new_bash/run_llmmirec_aspcf_K.sh
new_bash/run_aspcf_k2_baseline.sh
diagnostics_ch4_phase0/{beauty,ml-1m}.json                 (未提交，gitignored)
new_log/ch4_phase0_diagnosis/{beauty,ml-1m}.nohup.out       (未提交，gitignored)
new_log/llmmirec_aspcf_K2/{beauty,ml-1m}/                   (未提交，gitignored)
```

---

## 9. 事实 F7 与三条已封存的负面轴（2026-10-07）

### 9.1 新结构事实 F7：进入 extractor 前不存在 history-history 交互

代码级核对（`LLMMIRecASPCF.py` + `llmmi_components.py`，逐行）：

```
history_emb_raw   = item_encoder(history)        # ← 逐 item 独立，无跨位置操作
                  + position_emb(position)       # ← 逐位置独立
                  -> dropout
                  -> QueryMultiInterestExtractor

QueryMultiInterestExtractor 内部：
    Q = Wq(learned interest queries)             # 与 history 无关的 K 个可学习向量
    K = Wk(history_position)                     # 逐位置独立投影
    V = Wv(history_position)                     # 逐位置独立投影
    attention:  [B,K,L] over L 个位置
```

**关键点**：`Q/K/V` 三者都是**逐位置独立**计算的；
唯一的跨位置操作是 extractor 内部那次 attention，而它是
**K 个固定的可学习 query 对 L 个位置**的读取，
**不是 history 位置之间的相互 contextualize**。

⇒ **F7：每个 history position 在进入 extractor 之前，
其表示完全没有被其他 history position 修改过。**
不存在 history-history self-attention / graph propagation / message passing。

`ItemEncoder` 亦为纯逐 item 模块（padding 项恒输出零向量），无跨 item 操作。

### 9.2 三条已封存的负面轴

围绕**已经生成的 `V_k`** 做的三类干预，全部已证否并封存：

| 轴 | 干预内容 | 结果 | 归档 |
|---|---|---|---|
| **ECTIR** | 改变 interest **diversity**（熵最优传输 routing） | ECTIR-1 cosine 0.7182（−19%）但 ranking 0.1578/0.1063 **下降**；ECTIR-Full 塌缩到 1.0000，−8.98%。**冗余度与排序无单调关系** | `CHAPTER4_ECTIR_ROUND1.md` |
| **PPCIM** | **candidate-specific selection**（K 个 interest 直接面对 candidate） | τ=1 未激活；τ≤0.2 真正激活后**机制越强 ranking 越差**；hardmax 最差（HR@5 −2.45% / NDCG@5 −2.39%）；6 setting × 6 指标**无一超过 ASPCF** | `PPCIM_ROUND1_5_ANALYSIS.md` |
| **Dispersion** | **second-order additive correction**（`u_mu + λ·u_std`） | 二阶统计与 centrality **几乎正交**（Pearson **+0.072**），**但本身是反向 ranking signal**：`P(s_disp(pos)>s_disp(neg)) = 0.4716`，mean margin −0.1586。所有 λ>0 在 dev **单调变差**，λ\*=0 | `CHAPTER4_PHASE2_DISPERSION.md` |

**共同点**：三者都只动 `V_k` 的**组织方式或统计量**，`V_k` 本身来自
**逐 item 独立编码 + 一次固定-query 读取**。三条轴独立地失败。

**因此停止**围绕已生成的 `V_k` 做：
- diversity manipulation
- candidate-specific selection
- second-order additive correction

### 9.3 新的待验证轴

> **History item contextualization**：
> 在进入 multi-interest extractor 之前，
> 让 history 位置之间互相 contextualize（self-attention），
> 是否优于当前"逐 item 独立表示"？

**这是一个结构性 control experiment，不是 Chapter 4 的最终方法。**
设计见 `CHAPTER4_PHASE3_CONTEXT.md`，实现见
`models/sequential/LLMMIRecContextControl.py`（`LLMMIRecASPCF.py` 未被修改）。

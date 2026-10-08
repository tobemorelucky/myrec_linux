# CHAPTER4_PHASE4_GEOMETRY.md

> Chapter 4 Phase 4 — Dual-Space Ranking Geometry Audit
> 日期：2026-10-07
> 性质：**inference-only 诊断，不训练、不实现新模型**
> 脚本：`tools/audit_dual_space_geometry.py`
> 数据：`diagnostics_ch4_phase4/{beauty,ml-1m}_geometry.json`
>
> ⚠️ **本文件严格区分「已证实」与「待验证」。**
> 完整 dev/test 审计**未完成**（原因见 §11 Limitation），
> 因此凡涉及 ranking 的结论**一律标注为待验证**，不作断言。

---

## 0. F8 —— 待验证问题（**不写成瓶颈**）

**代码级核对**（读取本地工作区，非 git HEAD）：

`models/sequential/llmmi_components.py:344-377`：

```python
z_high = z[..., :self.semantic_rank]          # 512
z_low  = z[..., self.semantic_rank:]
s = self.semantic_branch(z_high)              # [*, 32]   semantic_dim
id_emb  = self.complement_id_emb(item_ids)    # [*, 64]
low_feat = self.complement_tail(z_low)        # [*, 64]
c = self.complement_mlp(cat([id_emb, low_feat]))   # [*, 32]  complement_dim
gate_weights = F.softmax(self.gate(cat([s, c], -1)), dim=-1)
alpha_sem, alpha_comp = gate_weights[..., 0], gate_weights[..., 1]
e = cat([ sqrt(alpha_sem + 1e-8) * s,
          sqrt(alpha_comp + 1e-8) * c ], dim=-1)   # [*, 64]
```

✅ **确认**：`semantic_dim = 32`、`complement_dim = 32`、`emb_size = 64`，
`e = concat[√(α_sem+ε)·s, √(α_comp+ε)·c]`。

`models/sequential/llmmi_components.py:508-510`：

```python
self.Wq = nn.Linear(emb_size, attn_size)   # 64 -> 64
self.Wk = nn.Linear(emb_size, attn_size)   # 64 -> 64
self.Wv = nn.Linear(emb_size, emb_size)    # 64 -> 64
```

✅ **确认**：均为 **full dense projection**，
**没有保持 32+32 block independence**。

> ### **F8（待验证问题，非已确立瓶颈）**
> ASPCF 在 item level 明确构造 semantic / complement 两个子空间，
> 但 downstream multi-interest projection 与 final dot-product
> 是否**正确利用**了这两个子空间，目前**没有证据**。

---

## 1. 术语约定（写进结论，避免误读）

| 量 | 允许的称法 | 禁止的称法 |
|---|---|---|
| `u[:32]`, `u[32:]` | **semantic-coordinate / complement-coordinate user contribution** | ❌ "pure semantic interest" / "pure collaborative interest" |
| `e[:32]`, `e[32:]` | ASPCF ItemEncoder 的两个**最终输出块** | — |

**理由**：`Wv` 是 full 64×64 dense，user 侧的两个输出半块**可能已经跨输入子空间混合**；
只有 candidate 侧的 `e[:32]/e[32:]` 严格对应 ItemEncoder 的两个最终输出块。
§6 的实测（`r_cross = 0.14~0.19`）**证实了这种混合确实存在**。

---

## 2. 精确分解 —— ✅ **已证实**

```
score_sem  = <u[:32], e[:32]>
score_comp = <u[32:], e[32:]>
score_full = score_sem + score_comp
```

**自检门限说明**：该恒等式在 **float32** 下重组误差为 `3.8e-06`——
这是**累加结合律噪声**（把一个 64 项点积拆成两个 32 项部分和），
**不是逻辑错误**。已用 float64 复核证明：

| 数据集 | split | float64 重组误差 | float32 重组误差 |
|---|---|---|---|
| Beauty | dev | **3.553e-15** | 3.815e-06 |
| Beauty | test | **3.553e-15** | 2.861e-06 |
| ML-1M | dev | **3.553e-15** | 1.907e-06 |
| ML-1M | test | **3.553e-15** | 1.907e-06 |

⇒ **分解在数学上精确**（误差处于 float64 机器精度量级）。**已证实。**

---

## 3.–5., 7.–9. ⚠️ **待验证**（仅 256 users/split 的单 batch 验证）

> **样本量警告**：以下全部来自 **1 个 batch（256 users）**，
> 不是全量 dev/test。仅用于**确认分析代码与数学分解正确**，
> **不足以支撑任何关于 ranking 的结论**。所有数值标注为待验证。

### 3. coordinate-block ranking（TEST，n=256）

| score | HR@5 | NDCG@5 | P(pos>neg) | margin |
|---|---|---|---|---|
| Beauty full | 0.2031 | 0.1375 | 0.8337 | 4.7606 |
| Beauty sem | 0.1289 | 0.0804 | 0.8137 | 3.2747 |
| Beauty comp | 0.0391 | 0.0296 | 0.6863 | 1.4860 |
| ML-1M full | 0.3203 | 0.2216 | 0.9407 | 6.1175 |
| ML-1M sem | 0.1484 | 0.0986 | 0.8696 | 2.7076 |
| ML-1M comp | 0.1094 | 0.0681 | 0.8819 | 3.4099 |

**方向性观察（待验证）**：两个数据集上 `sem` 单独都明显强于 `comp` 单独。
Beauty 的 gap 尤其大（NDCG@5 0.0804 vs 0.0296）。
**但**：`full` 在 ML-1M 上比 `comp` 单独**高得多**，
说明两块并非简单相加可得——`u[:32]`/`u[32:]` 与 `e` 的对应关系在两个数据集上不同。

### 4. branch correlation / conflict（TEST，n=256）

| 量 | Beauty | ML-1M |
|---|---|---|
| Pearson(sem, comp) | +0.0573 | +0.1353 |
| Spearman(sem, comp) | +0.0644 | +0.1388 |
| Q1 (Δsem>0 & Δcomp>0) | 0.5642 | 0.7774 |
| **Q2** (Δsem>0 & Δcomp≤0) | **0.2495** | **0.0922** |
| **Q3** (Δsem≤0 & Δcomp>0) | **0.1221** | **0.1045** |
| Q4 (both≤0) | 0.0642 | 0.0259 |
| **Q2 − Q3** | **+0.1274** | **−0.0123** |

**方向性观察（待验证）**：两个 branch 的 candidate-level 相关性**很低**（0.06 / 0.14），
即它们**不是冗余信号**。Beauty 上 `Q2 − Q3 = +0.127`（semantic 更常是对的），
而 ML-1M 上 `−0.012`（几乎对称）。
**这个跨数据集反转的方向性提示值得全量复核，但 n=256 下不可下结论。**

### 5. ItemEncoder 实际能量 —— ✅ 定性已证实 / ⚠️ 定量待验证

> **最重要的读数修正**：`alpha_sem ≈ 0.01` **不能**代表语义分支的贡献。

| 量 | Beauty | ML-1M |
|---|---|---|
| `alpha_sem` mean (p50) | 0.0103 (0.0091) | 0.0172 (0.0120) |
| `alpha_comp` mean | 0.9897 | 0.9828 |
| `‖s‖`（candidate，mean） | **32.94** | **29.00** |
| `‖c‖`（candidate，mean） | **2.01** | **1.98** |
| **`rho_sem = E_sem/(E_sem+E_comp)`** pos | **0.6898** | **0.6899** |
| `rho_sem` all candidates | 0.7020 | 0.6654 |
| `rho_sem` history items | 0.6915 | 0.6905 |

其中 `E_sem = α_sem·‖s‖²`，`E_comp = α_comp·‖c‖²`（用**原始** branch 输出，非 `√α·s`）。

**结论（定性，已证实）**：
尽管 `alpha_sem ≈ 0.01`（看起来把语义"关掉"了），
但由于 `‖s‖ ≈ 30` ≫ `‖c‖ ≈ 2`，**semantic 分支实际占约 69% 的能量**。
⇒ **后续任何文档都不得再用 `alpha_sem` 单独代表语义贡献。**

**定量数值待验证**（256 users）。

### 7. dev-only beta calibration（`score_beta = score_comp + β·score_sem`）

| β | Beauty dev NDCG@5 | ML-1M dev NDCG@5 |
|---|---|---|
| 0.0 | 0.0144 | 0.0746 |
| 0.25 | 0.0587 | 0.1289 |
| 0.5 | 0.1317 | 0.1699 |
| **1.0** | **0.1431** | **0.2127** |
| 2.0 | 0.1421 | 0.1811 |
| 4.0 | 0.1111 | 0.1173 |

**两个数据集上 dev-best β 都恰好 = 1.0**（即 ASPCF 本身），
test 上曲线同样在 β=1 取峰。

**方向性观察（待验证）**：这**不支持**"equal-coordinate scoring 存在 calibration 问题"
（§10 的 A 方向）——若存在校准问题，β 最优值应明显偏离 1。
**但 n=256 下不足以断言，需全量复核。**

### 8. fine-grained history→candidate evidence（TEST，n=256）

| score | Beauty NDCG@5 | P(pos>neg) | r vs ASPCF | ML-1M NDCG@5 | P(pos>neg) | r vs ASPCF |
|---|---|---|---|---|---|---|
| `g_sem` | 0.1580 | 0.8267 | 0.4771 | 0.0523 | 0.7925 | 0.3653 |
| `g_comp` | 0.0091 | 0.6856 | 0.2940 | 0.0564 | 0.7975 | 0.5083 |
| `g_full` | 0.1380 | 0.8304 | 0.5293 | 0.0775 | 0.8596 | 0.6278 |
| `last_full` | 0.0895 | 0.7325 | 0.4248 | 0.1142 | 0.8281 | 0.2916 |

**方向性观察（待验证）**：这些 evidence score 确实有正向 ranking signal，
但**与 ASPCF score 的相关性并不低**（r = 0.29~0.63），
即它们**很大程度上是 ASPCF 已有的信息**，而非"压缩丢失的新证据"。
**这与 §10 的 B 方向所需的"低相关 + 补弱 bucket"前提不符**，但 n 太小，不作结论。

### 9. popularity / history-length buckets —— ❌ **未验证**

单 batch 下每个 bucket 的 n 低至 **1~12**（Beauty `[15,20)` n=1、ML-1M `[1,23)` n=2），
**数值无统计意义，不予报告、不作任何解读。**

---

## 6. Wv / Wk 跨子空间混合 —— ✅ **已证实**（权重级，全数据集精确）

> 本节**不依赖样本**：直接读取 checkpoint 权重，对**整个数据集**精确成立。

| 量 | Beauty | ML-1M |
|---|---|---|
| `‖Wv‖_F` | 14.2115 | 15.0321 |
| `‖W_ss‖_F` / `‖W_sc‖_F` / `‖W_cs‖_F` / `‖W_cc‖_F` | 11.3934 / 3.7561 / 4.8317 / 5.8910 | 12.1923 / 2.9080 / 4.7647 / 6.7937 |
| block 占 `‖Wv‖²_F` 比例 ss/sc/cs/cc | 0.6427 / 0.0699 / 0.1156 / 0.1718 | 0.6579 / 0.0374 / 0.1005 / 0.2043 |
| **`r_cross = (‖W_sc‖²+‖W_cs‖²)/‖Wv‖²_F`** | **0.1854** | **0.1379** |
| `‖Wv[:, :32]‖²_F` 占输入列能量比 | **0.758314** | **0.758325** |
| `‖Wv[:, 32:]‖²_F` 占输入列能量比 | 0.241686 | 0.241675 |
| `‖Wk[:, :32]‖²_F` 占输入列能量比 | 0.266075 | 0.328315 |

**结论（已证实）**：

1. **`r_cross = 13.8%~18.5%`** —— `Wv` **确实做了跨子空间混合**，
   而不是保持 32+32 的块对角结构。
   ⇒ **§1 的术语约束是必要的**：user 侧的 `u[:32]` / `u[32:]`
   **不能**解释为"纯语义/纯协同"。
2. **`Wv` 对输入两半的依赖极不对称**：
   输入列能量有 **75.8%** 落在 semantic 坐标上（两个数据集几乎相同）。
   这与 §5 的 `rho_sem ≈ 0.69`（semantic 占输出能量约 69%）方向一致。
3. **`Wk` 的输入依赖在两个数据集上不同**（0.266 vs 0.328），
   但注意 `Wk` 的 **64 维输出没有 semantic/complement 含义**，
   因此这里**只统计输入列，不拆输出**（按要求）。

> ⚠️ **一个值得注意但本轮不解释的观察**：
> `‖Wv[:, :32]‖²` 的能量占比在两个**独立训练**的模型上几乎相同
> （0.758314 vs 0.758325，差 1.1e-5）。
> 这可能是由输入坐标本身的尺度差异（`‖s‖≈30` vs `‖c‖≈2`）
> 造成的系统性效应，**也可能只是巧合**。
> **本轮不做归因，仅记录。** 若要判断，需要更多种子。

---

## 10. 判定框架应用

| 方向 | 条件 | 本轮证据状态 |
|---|---|---|
| **A. Dual-space calibration** | 两数据集 semantic/complement 贡献明显不同，且 dev-best β 明显偏离 1 或方向相反 | **未获支持（待验证）**。两数据集 dev-best β **都恰为 1.0**。 |
| **B. Fine-grained evidence** | `g_sem`/`g_comp` 至少一个有正向 signal，**且与 ASPCF score 相关性不高**，能补弱 bucket | **部分：有正向 signal，但相关性不低（0.29~0.63）；弱 bucket 因样本太小未验证。** |
| **C. 两者都没有** | β 校准无改善、fine-grained 也无增量 | **不能判定**——A/B 都因样本量不足而无法定论。 |

### ⚠️ **本轮判定：悬置（inconclusive）**

**不宣布 A、B 或 C 中的任何一个成立。**
本轮建立的是**方法**与一条**全数据集精确的权重级事实（§6）**，
而不是关于 ranking 的结论。

**唯一可以带走的、已证实的结论**：

> **F8 的部分回答（已证实）**：
> `Wv` 中 `r_cross = 13.8%~18.5%` 的跨子空间混合**实际存在**，
> 且其对输入两半的能量依赖高度不对称（semantic 坐标占 75.8%）。
> ⇒ **user 侧的 `u[:32]/u[32:]` 不是"纯语义/纯协同"**，
> 后续任何 dual-space 设计必须基于这一事实，不能假设 user 侧子空间是干净的。

---

## 11. ⚠️ Limitation（明确记录，不再无限排查）

### 11.1 未完成的部分

**Beauty / ML-1M 的完整 dev+test 审计均未跑完。**
已完成的是 **1 个 batch（256 users）的端到端正确性验证**（两数据集 EXIT=0）。

### 11.2 原因（已定位，未修复）

1. **内存爆炸（已修复）**：原实现在**每个 batch** 都做一次 float64 重组校验，
   把 `[B, C, 64]` 上采样到 float64（C=1001 时约 130 MB/batch），
   实测 **RSS 以 ~0.58 GB/batch 线性增长**，
   25 个 batch 即达 **50 GB**，全量 run 必被 OOM kill。
   **修复**：该恒等式校验**每个 pass 只需跑一次**，已改为 `n == 0` 时执行。
2. **剩余开销（未优化，按要求停止）**：修复后实测约 **20 s/batch**，
   Beauty 全量（dev 88 + test 88 = 176 batch）约需 **~1 小时**，
   远超本轮 3 分钟的验证预算。
3. 另有一次 SIGTERM（exit 143），来源未确定，非 OOM（当时系统 91 GB 空闲）。

### 11.3 处置

- **不再继续优化审计脚本**（按指示）。
- §3/§4/§5(定量)/§7/§8/§9 的结论**全部标注"待验证"**。
- 若后续需要全量结论，**必须重新安排一次专门的运行**
  （建议：拆成 dev-only / test-only 两次前台运行，或降低 `num_neg` 采样），
  **本轮不做**。

### 11.4 本轮损坏/污染检查

- `models/sequential/LLMMIRecASPCF.py`：**未修改**（`git diff` 为空）
- `models/sequential/llmmi_components.py`：**未修改**
- `models/sequential/PoMRec.py`：**未修改**
- 调试用 RSS instrumentation 与未使用的 `MARG` 累加：**已全部移除**
- 未 commit、未 push

---

## 12. 本轮修改清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `tools/audit_dual_space_geometry.py` | 新增 | §2–§9 审计脚本；含 4 处 bug 修复（见 §11.2） |
| `diagnostics_ch4_phase4/beauty_geometry.json` | 新增 | Beauty 1-batch 结果 |
| `diagnostics_ch4_phase4/ml-1m_geometry.json` | 新增 | ML-1M 1-batch 结果 |
| `CHAPTER4_PHASE4_GEOMETRY.md` | 新增 | 本文件 |

**本轮修复的 4 个 bug**（均在审计脚本内）：

1. `Namespace` 缺 `batch_size` → `AttributeError`（已修）
2. `MARG` 字典缺 `d_last_*` 键 → `KeyError`（该累加器最终**整体移除**，因其从未被报告）
3. `E = α·‖branch‖²` 误用已缩放的 `√α·s`（等于算了 `α²‖s‖²`）→ **实质错误**，
   导致 `rho_sem` 从 ~0.69 被低估到 ~0.029。已改用 `return_components` 的**原始** `s`/`c`。
4. `CORR` 缺 `last_sem/last_comp/last_full` 键 → §8 打印时 `KeyError`（已修）

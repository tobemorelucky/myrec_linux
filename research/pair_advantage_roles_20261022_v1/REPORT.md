# 正负样本对优势职责：有条件Go到受控完整对照，尚无效果证据

2026-10-22。不是对现有模型的逐式复制，但属于已有软混合/多兴趣训练的任务适配，不能把“职责”“软最小”或新增Loss本身当原创。**随机初始化直接启用不建议；先ASPCF-Softmax热身、再端到端职责学习，有条件值得做一次完整三臂对照，须用户另行批准。** 本轮未启动完整训练、未执行Test。

## 查重结论

|参照|核实的核心|本轮实际区别|
|---|---|---|
|BaM|官方源码用正例相关性softmax概率抽选头，并另加跨兴趣LSE的多兴趣损失|每个正负样本对都有独立的头职责，来自其区分损失；不改推理LSE/评分|
|REMI|按选中兴趣的负例分数进行难负例重加权，另有历史路由正则|不改变采样/主Softmax；在同一合法负例集合上分配头间训练责任|
|CIGR|前向分数保持不变，通过detach surrogate改变候选条件梯度；候选/聚合器不接受该额外梯度|使用明确可微标量目标，不伪装原Softmax梯度；职责同时依赖正负差值，物品编码器也可端到端更新|
|CAISD/TASID|历史语义profile/正例语义相关性的KL蒸馏|无语义teacher、无目标输入提取器；目标仅作为Train损失标签，职责来自推荐区分能力|
|旧MultiPositive|两个Train未来正目标的跨头排他分配＋BPR|仅当前正目标，对多个合法负例分别分工，无多目标匹配；仍共享成对排序思想|

BaM实际为**IEEE BigData 2024**，不是KDD2024。[作者机构记录](https://snu.elsevierpure.com/en/publications/towards-true-multi-interest-recommendation-enhanced-scheme-for-ba/)、[官方选择代码](https://raw.githubusercontent.com/snudatalab/BaM/main/src/BasicModel.py)、[官方多兴趣损失代码](https://raw.githubusercontent.com/snudatalab/BaM/main/src/SSLoss.py)已核查；作者PDF链接访问失败，不能声称已核全文。[REMI原文](https://arxiv.org/html/2302.14532)已核查。以上只是列举对象间的区别，非全领域新颖性证明。

## 最小目标：负例条件职责＋真实梯度

对Train正例i+和合法负例j，定义：
```text
z_ukj = V_uk · (e_i+ - e_j)
ell_ukj = softplus(-z_ukj)
L_pair = E_(u,j)[(1-alpha) mean_k ell_ukj
                  - alpha*T*log(mean_k exp(-ell_ukj/T))]
r_ukj = dL_pair/dell_ukj
      = (1-alpha)/K + alpha*softmax_k(-ell_ukj/T)
L = L_filtered_Softmax + eta*L_pair + .01*L_relation
```

固定短筛eta=.1、alpha=.5、T=1，无搜索。r按负例而变化，分配给较能区分该正负对的兴趣；下限为.5/K，避免零责任饥饿，但不保证各头总使用量均衡或语义可解释。使用logsumexp/softplus，无额外代理梯度。等式里的E按每用户合法负例先平均，再对有效行平均。

**必要控制**：相同eta和相同负例的alpha=0均匀成对损失；否则无法区分职责贡献与仅增加BPR监督。K=1时职责版与均匀版等价，均仍有辅助成对损失，不能冒充严格原SS退化。eta=0则预测、总loss、query梯度逐元素等价原SS。

保留ASPCF物品编码器及原语义资产、Learnable Query与可配置K；全部原推荐参数可更新，冻结LLM语义表。无新增训练参数（Beauty943174、ML A1 405894）。训练阶段才读正负标签；推理仍为历史提取/原聚合/点积，分别报告不加先验与加固定λpop=.30及原Train-RMS两种评分。

## 三分钟内正确性与16步Dev筛查

K=1/2/4/7数值gradcheck、职责解析导数、责任和/下限、全无效负例零loss/零梯度、eta0预测/loss/query梯度精确等价均通过。完全相同的头得到均匀职责，本目标**不会凭空打破完全对称**。

|真实eval模式诊断，无Dropout干扰|Beauty|ML-1M A1|
|---|---:|---:|
|冷启动query辅助梯度范数|7.30e-12|1.21e-11|
|冷启动职责偏离均匀|0|0|
|加载原ASPCF后query辅助梯度范数|.02060|.06441|
|加载后职责平均绝对偏离均匀|.00573|.01138|
|职责版V梯度相对均匀控制的变化|11.12%|19.86%|
|头间V辅助梯度平均余弦|.9400|.8920|

因此必须热身或使用预训练ASPCF阶段作为起点；不采用放大初始化噪声、强行随机路由来宣称分工已形成。热身后也仍有较高梯度相似度，独立功能与推荐收益待完整实验。

三个分支同一既有best-Dev权重初始化、同16批Train前缀、同Dropout/relation RNG，原filtered Batch Softmax＋relation；仅比较辅助目标。B1024、Adam lr=1e-4、L2=1e-6，16步；不进行checkpoint选择或保存短试训练权重。以下为固定频次先验下的Dev NDCG@5：

|数据集|原checkpoint|继续原SS 16步|均匀成对控制16步|职责版16步|
|---|---:|---:|---:|---:|
|beauty|0.184062|0.184775|0.184685|0.184690|
|ml-1m|0.294696|0.292522|0.292250|0.292275|

职责对均匀控制仅+0.0000048/+0.0000250，接近零；对继续SS为负。ML三分支短时Dev均降。这里没有声称性能Go，也不以16步未超过强基线直接否定训练机制；当前只证明数值正常、真实query/Wv均更新，且职责梯度不同。全部原分数/固定先验六项保存在每域CSV。

两域主体合计50.44秒；数学准备约9.17秒，含进程启动仍低于3分钟计算预算。三臂共同16步拟合Beauty2.21秒/ML2.24秒，峰值allocated约793/533MiB。Beauty计算完成后CSV导出有缺少函数别名错误；原JSON已写成，仅修复导出入口并复用结果，没有重复Beauty训练/Dev。失败日志、修复说明及完成回执保留。

## 待批准的完整方案与成本

1. 使用共同的原ASPCF-Softmax热身权重作为阶段0；K4可复用本轮固定资产。相同阶段0、初态、数据顺序、优化/早停预算比较：继续原SS、SS＋均匀成对控制、SS＋成对职责。每组同时输出原评分及固定先验评分，不额外训练一个“加先验模型”。
2. 初步建议最多50个继续训练epoch、Dev early-stop10，最少先检验K4/seed42，两域统一eta=.1/alpha=.5/T=1/lr=1e-4，不以本次16步选超参。职责对均匀控制无独立Dev贡献则停止。
3. 若完整Dev出现独立增量，再安排BaM式仅正例职责等针对性消融及K={1,2,4,8}，每个K必须共享同K阶段0和控制，不能只给新法挑最佳K。短筛不是K实验。
4. 最终按Dev固定各组checkpoint/配置，再一次Test六项，与原SS、已验收固定先验强基线及同预算控制比较。当前没有启动/执行这些阶段，已见Test后的适配如实标探索性。

额外训练主项为O(B²KD)，中间量[B,K,B]；新增参数0、推理开销0。三臂短试平均约46毫秒/步（含共同检查及原模型计算，不是职责相对开销的单独测量）；K4线性外推的纯训练单epoch量级约Beauty6秒、ML45秒，另加Dev及运行开销。为正式试验建议保守预留Beauty每臂10–20分钟、ML每臂1–2小时的上限预算，三臂两域约3.5–7 GPU小时，K/seed扩展另计；这是预算建议，不是已测完整训练耗时或承诺。

**结论：实现与机制有条件Go，推荐效果未验证。** 值得一次两阶段、同预算、含均匀监督控制的完整对照；不建议随机初始化直接开启职责。本轮已停止，后续仅用户批准后nohup后台启动，核实进程运行即停止操作，等待用户检查后再汇总。

## 证据

P920 HEAD 771b4c3826f07a8cc7fb9038fb8659161e9f8189；原有未跟踪目录pop_bucket_interest_screen_20261022_v1保留，本轮只新增research/pair_advantage_roles_20261022_v1/，tracked/staged diff空。旧源码/数据/checkpoint/embedding及先验锁SHA前后相同，无原资产覆盖。

- /home/yyx/hzgcode/pom2.0/research/pair_advantage_roles_20261022_v1/role_objective.py：独立PairRoleModel与目标；SHA 01c31dd7d0851fa1b2e18153d7960c04720e8aa88d54b5107247c7919e61b4f6。
- 同目录audit.py、resume.py、PREREG.json、REPAIR_report_only.json、COMPLETE.json、两域JSON/Dev六项CSV与原始日志。没有接入main.py或提供自动完整训练入口。
- 历史比较源码：models/sequential/LLMMIRecASPCFCIGR.py:7、LLMMIRecCAISD.py:240/316、LLMMIRecASPCFMultiPositive.py:34，全部以P920版本为准，SHA在PREREG/结果JSON。
- 本地轻量证据：E:\work\code\paper\myrec_linux\research\2026_baselines\pair_advantage_roles_20261022_v1\。


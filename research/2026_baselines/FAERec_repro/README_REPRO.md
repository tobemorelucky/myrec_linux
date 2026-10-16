# FAERec-SASRec统一协议复现

状态：Beauty与ML-1M seed42均已正常完成完整Dev-only训练及best权重重载校验，无Test。最终结果见文末。

官方：https://github.com/ZhifuWei/FAERec.git；许可证Apache-2.0。实际clone SHA将在完成后记录到adapter/provenance.json。upstream保持原样，不使用其中预训练权重作本项目结果。

所有写入限定本目录。Train/Dev来自项目现有CSV，只用Train统计与负例过滤；H20，原1001候选槽位及重复保留，使用原BaseRunner.evaluate_method，Dev NDCG@5严格改善选best，不调用Test。

adapter/run_faerec.py直接导入官方SASRecModel、SASRecTrainer和损失；仅隔离同名module导入冲突、适配数据/格式、替换训练控制和候选评价入口。训练保留官方序列位置BCE、物品InfoNCE、head/tail feature对齐及周期调度；不执行会自动Test的官方main。

预注册：D128、两层两头、hidden/attention dropout .4、mapper dropout .3、lr .001、Adam weight_decay=0；cl_weight .2、item_tou .05、w_min .9、w_max1、period30、warmup0。batch64从允许的16/32/64中选取，先GPU1小batch测试；max_epoch200、early_stop10、单组4小时。正式GPU资源/训练成本须实测。

数据训练单位保留原生每用户一条序列：Train末21个item产生最多20个因果输入/目标对，不重新留出；与ASPCF逐前缀的训练样本频次不同。训练负例沿官方均匀采样，仅排除Train已点击，并修正原dummy采样上界。Dev history为该用户Train末20条，仅Train/Dev文件被读取。语义表去已有padding后转换，官方loader恢复padding并加dummy，官方二次PCA拟合初始化不变；资产只写runs/assets。

训练排序候选使用ID embedding，服务使用LayerNorm(ID+mappedLLM)，按官方原公式保留。评价缓存该服务物品表以避免重复mapper计算，不改变点积值。不存在全目录用户打分或Test调用。

正式结果、显存、时间、SHA及任何最小修复将在本文件追加。本复现称统一数据/语义/H20/1001候选协议适配，不宣称复现论文原发表数字。

## 官方源码取得

已完成官方depth1 clone，commit 8e2d4be320a567d724535d426b72ded20e140dcb，Apache-2.0，upstream工作区干净。源码SHA见adapter/provenance.json。包含的作者权重不加载。本次clone使用临时SSH HTTPS转发解决服务器直连TLS问题，没有修改配置或关闭证书验证。

## Beauty专项与正式启动

batch64联合及三项独立损失反向通过，独立梯度L2为[4.908989429473877, 12.237777709960938, 266.8971252441406]；逐行语义对应、重复item对比mask、官方predict_full评分等价、单用户评价独立及严格checkpoint roundtrip均通过。专项峰值allocated/reserved MiB=282.947265625/372.0，耗时13.044074901379645秒。smoke中的64名Dev仅为接口验证，不是正式Dev报告。

正式：runs/beauty_seed42_20261015，PID3974845，GPU1；启动命令见同名.launch.json。前两轮约19.5秒/轮，200轮估算约65分钟，继续按原early stopping，无Test。配置已完整打印到.out并保存在config.json。

## 样本单位与资产绑定

{
  "beauty": {
    "train_rows": 153776,
    "eligible_train_users": 22363,
    "effective_positive_positions_per_epoch": 117194,
    "maximum_history": 20,
    "training_unit": "one_train_user_sequence",
    "asset_sha256": {
      "itm_emb_np.npy": "6773d4f5bb6e731978dc29bab20249d169d1d3c8847be30ac18c169718f667e9",
      "pca_itm_emb_np.npy": "4f70f68c10c6f6216e04015dfaaffa03f4a1cc5061ac2c0fe0dcfec0bf03ca3d"
    }
  },
  "ml-1m": {
    "train_rows": 988129,
    "eligible_train_users": 6040,
    "effective_positive_positions_per_epoch": 120283,
    "maximum_history": 20,
    "training_unit": "one_train_user_sequence",
    "asset_sha256": {
      "itm_emb_np.npy": "609519f48ffcaf16776e2474b090993917756417ff63a3fd73ea721dd96896c7",
      "pca_itm_emb_np.npy": "f109b9bc5a8806fb977e30670f0b7aaaac63cb91c1a3984deb58817888baaffd"
    }
  }
}

以上训练样本单位为原生user序列、监督位置数为每epoch实际有效正例数；与ASPCF逐前缀频次不同。语义输入1536维，因此官方动态Mapper隐藏宽度为768；不是作者原始语义源和权重配置的精确复制。原默认batch256改64，最多500轮改200，Dev/early stop从第1轮开始而非原epoch>60（0-based）后开始验证。公式未改，但这些训练设置可影响结果，不能把性能差异当单一结构因果效应。

## 最终结果（Dev-only，seed42，2026-10-15）

|数据集|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|best/end epoch|
|---|---|---|---|---|---|---|---|
|beauty|0.18302553|0.12988801|0.24714931|0.15053758|0.32629790|0.17051142|21/31|
|ml-1m|0.18675497|0.12539046|0.27847682|0.15472902|0.39552980|0.18422737|36/46|

|数据集|总参数|总秒数（含加载/训练/Dev/最终重载）|平均epoch秒|200轮推算秒|峰值allocated/reserved MiB|
|---|---|---|---|---|---|
|beauty|3326080|597.791|18.857|3771.347|284.356/520.000|
|ml-1m|2251520|262.579|5.552|1110.374|278.334/440.000|

两组均为early_stop，200轮和4小时上限均未触发。峰值为PyTorch每进程统计，不含完整驱动上下文开销；不是nvidia-smi总显存。200轮时间是当前普通epoch均值线性推算，非实际跑满200轮。

### beauty

日志：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/beauty_seed42_20261015.out

配置/逐轮记录/最终结果：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/beauty_seed42_20261015/config.json、epochs.jsonl、summary.json。

checkpoint：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/beauty_seed42_20261015/best.pt

SHA256：4b123e1981f5d1e802f10bcd4c520c469dde6228ba1e9d9895208d6e1842b4a0

### ml-1m

日志：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/ml-1m_seed42_20261015.out

配置/逐轮记录/最终结果：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/ml-1m_seed42_20261015/config.json、epochs.jsonl、summary.json。

checkpoint：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/ml-1m_seed42_20261015/best.pt

SHA256：1545d0b52d8ef740c6227f7540a06f8dd0fece555e178c79a2ff2974061375b8


## 数值情况与判断

Beauty专项独立梯度、原始预测公式、padding/ID对应、checkpoint和评价独立性通过。ML专项也通过，初始归一化对比项梯度较大：129个真实item的PCA初始化范数低于1e-5，最小约7.91e-7；原始PCA1536表没有零范数真实行，最小范数约1.01e-5。四步联合梯度范数16980→8316→6908→4087，损失下降且全部有限。没有修改初始化、归一化epsilon、loss权重或加入clip；不能把该观察视为已证明的低分原因。证据runs/short_gradient_ml-1m_seed42/gradient_check.json。

最终checkpoint tensor均有限，重载Dev六项与保存best完全一致，Dev复核前后checkpoint SHA不变，原Train/Dev/语义表与官方核心8文件哈希均未改变。upstream工作区仍干净。

Beauty结果接近历史ASPCF BPR seed42 Dev NDCG@5 .1300；ML-1M明显低于其.2271参考。两者目标、维度、batch与原生训练单位不同，这不是单变量算法归因。当前只能认定官方核心在统一协议下已跑通，不能认定强性能已复现或原方法无效。

## 行为变更与局限清单

- upstream没有修改任何文件；没有调用原main或作者预训练权重。新增代码只有adapter/run_faerec.py；provenance.json和unified_data_manifest.json为元数据。
- 纯数据/评价适配：CSV/ID映射、保留1001重复槽位、原Runner指标及并列处理、候选评分缓存、格式转换、禁止Test。
- 实际训练设置变化：H50→20，batch256→64，seed2024→42，最多500→200，延迟60轮验证/原NDCG@20选模改为第1轮起Dev NDCG@5/early_stop10。两组200轮上限都未触发，提前停止规则可能影响收敛。
- 训练负例排除集合只用Train，移除原额外dummy采样；既有真实目录ID和语义对应不变。
- 原生每用户末段seq-to-seq单位保留，每轮有效监督117194/120283，不等于ASPCF逐前缀训练频次；不能将整个性能差异归因于结构。
- 复用当前PCA1536而非作者原始LLM源，其来源链保留既有未闭合限制；本轮未生成LLM内容、未用随机向量补缺失。官方Mapper/PCA/融合/两种对齐/周期调度/训练服务评分差异均保留。
- 部分高阶官方训练/服务不对称及小batch对比负例数变化可能影响表现，当前没有因果归因证据，不根据Dev或Test追加调参。

本轮完成后停止。没有多种子、搜索、Test或第四章原创开发。此结果称统一协议适配结果，不称复现原论文发表数字。

## 必须保留的训练日程限制

**两组分别在第31/46轮停止，均未到官方默认epoch>60（0-based）才开始验证的阶段。本入口把验证与early stopping提前到第1轮，是实质性的训练控制适配；目前不能声称已充分复现官方训练日程或强性能。** 核心损失调度仍原样保留，但提前停止可能影响后期学习，当前没有实验可证明它是否造成ML-1M低分。不能依据本轮低分直接否定FAERec，也不在观察结果后悄悄延长、重跑或选择更好的配置。任何更接近官方日程的后续验证须重新明确协议并单独授权。

## 新轮预注册：官方延迟验证＋锁定版本一次Test

本轮只改变controller：沿官方zero-based epoch>60 gate；索引61、实际完成第62轮后首次Dev。之前dev=null、不存best、不计早停。其余与旧轮相同，包括batch64、seed42、lr.001、D128、H20及原全部loss/调度。新runs名为beauty_seed42_delayed60_20261015及ml-1m_seed42_delayed60_20261015；旧权重及旧入口保留。训练不Test；两组正常结束及best加载核验后，对各自锁定新版本仅一次统一Test，不以Test挑新旧版本。最大200轮、early_stop10、每组4小时；GPU1空闲才启动。


## 官方延迟验证seed42验收已完成（2026-10-15）

本节为最新结果，早期版本保留，旧版“无Test/等待授权”只描述当时阶段。两组新版本均按官方零基epoch>60首次在完成62轮后验证，正常早停并重载best后各一次探索性Test；没有测试旧版。

- beauty：best/end=78/88，Dev N5=0.138455、HR5=0.192729；Test N5=0.122175、HR5=0.169968；25.74分钟；SHA 9f23925cec58f4adf8c0e0ac42a7758e435510b004656251e331ff363b040dfb。
- ml-1m：best/end=62/72，Dev N5=0.130980、HR5=0.196854；Test N5=0.123408、HR5=0.183775；6.00分钟；SHA 11034426986ebd7ac7f0217c088ae6f45f972c22d983504ef451add4012cef20。

全部Dev/Test六项、旧日程比较、成本和基线核验见FAEREC_SEED42_TEST_COMPARISON.md；原始验收runs/delayed60_acceptance_20261015.json。依然是统一协议适配，不是原论文发表数字复现；FAE非显式多兴趣，batch64/当前划分/PCA1536语义源等差异仍保留。本轮停止，不追加训练。

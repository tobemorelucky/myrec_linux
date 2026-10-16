# FAERec-SASRec seed42：延迟验证验收与一次性探索性Test

## 裁定及模型身份

两组均正常完成、按Dev NDCG@5确定best、严格重载后各执行一次Test。旧版不测试、不覆盖。FAERec是LLM增强的单向量序列推荐模型，不是显式K兴趣模型；本结果为当前统一协议适配基线，不是复现原论文发表数字。Test为研发阶段可见的探索性结果，不是未参与全部研发的最终独立验证。

官方仓库：https://github.com/ZhifuWei/FAERec；commit 8e2d4be320a567d724535d426b72ded20e140dcb；Apache-2.0。upstream原样，8个核心源码SHA复核一致。

## 新旧训练日程与Dev

官方FAERec/src/main.py:73默认star_test=60，:150从0循环，:158条件epoch>star_test。因此首次验证索引61，即完成第62轮；此前61个完整epoch无Dev、无checkpoint选模及early-stop计数。没有把条件擅改为>=60或一基第61轮。

新增adapter/run_faerec_delayed.py仅替换训练控制日程，调用已有run_faerec.py的setup/objective/predict，官方BCE、item InfoNCE、feature alignment和周期调度未改。新旧配置差异仅日程/记录字段/独立日志路径；前10轮已观察训练loss逐值一致，支持初始化及早期训练随机流一致，不宣称全部轨迹必然相同。

|模型 / 日程|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|beauty 旧：第1轮验证|0.183026|0.129888|0.247149|0.150538|0.326298|0.170511|
|beauty 新：官方延迟验证|0.192729|0.138455|0.252739|0.157773|0.320664|0.174940|
|ml-1m 旧：第1轮验证|0.186755|0.125390|0.278477|0.154729|0.395530|0.184227|
|ml-1m 新：官方延迟验证|0.196854|0.130980|0.290894|0.161347|0.410762|0.191516|

|数据集/日程|best/end（完成轮数）|耗时秒/分钟|平均epoch秒|峰值allocated/reserved MiB|训练参数|
|---|---:|---:|---:|---:|---:|
|beauty 旧|21/31|597.791 / 9.96|18.857|284.356/520.000|3326080|
|beauty 新|78/88|1544.429 / 25.74|17.398|285.855/520.000|3326080|
|ml-1m 旧|36/46|262.579 / 4.38|5.552|278.334/440.000|2251520|
|ml-1m 新|62/72|359.819 / 6.00|4.895|278.334/440.000|2251520|

峰值为PyTorch进程allocated/reserved统计，不是nvidia-smi整卡总显存。新两组均在4小时内正常early stopping，未触发预算强停；训练耗时含启动及结束best-dev复核，平均epoch不含这些固定开销。旧版未测试，不能比较新旧Test挑选日程。

|数据集|新−旧Dev NDCG@5|相对变化|新−旧Dev HR@5|
|---|---:|---:|---:|
|beauty|+0.008567|+6.60%|+0.009704|
|ml-1m|+0.005589|+4.46%|+0.010099|

仅说明锁定的不同验证/早停日程对应不同best-dev及训练长度，不证明延迟验证是普遍更优算法，也不由Test反选旧版/新版。

## beauty：同协议Test六项

|模型 / 日程|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|PoMRec BPR K4|0.108100|0.074600|0.156200|0.090000|0.213600|0.104400|
|ASPCF Batch Softmax K4|0.181700|0.129400|0.247800|0.150800|0.316900|0.168200|
|ASPCF BPR K4|0.159200|0.108800|0.229200|0.131300|0.317100|0.153500|
|FAERec-SASRec（非K兴趣）|0.169968|0.122175|0.226848|0.140533|0.290480|0.156618|

|参考|FAE−参考Test NDCG@5|相对变化|FAE−参考Test HR@5|
|---|---:|---:|---:|
|PoMRec BPR|+0.047575|+63.77%|+0.061868|
|ASPCF Batch Softmax|-0.007225|-5.58%|-0.011732|
|ASPCF BPR|+0.013375|+12.29%|+0.010768|

## ml-1m：同协议Test六项

|模型 / 日程|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|PoMRec BPR K2|0.310400|0.214400|0.437900|0.255500|0.561900|0.286900|
|ASPCF Batch Softmax K4|0.333800|0.241800|0.441200|0.276700|0.548000|0.303700|
|ASPCF BPR K4|0.306800|0.213400|0.432100|0.253800|0.560100|0.286100|
|FAERec-SASRec（非K兴趣）|0.183775|0.123408|0.273841|0.152337|0.381457|0.179553|

|参考|FAE−参考Test NDCG@5|相对变化|FAE−参考Test HR@5|
|---|---:|---:|---:|
|PoMRec BPR|-0.090992|-42.44%|-0.126625|
|ASPCF Batch Softmax|-0.118392|-48.96%|-0.150025|
|ASPCF BPR|-0.089992|-42.17%|-0.123025|

## 比较口径与核验边界

- 所有表中模型均seed42、H20、原1正+1000负槽位、保留重复槽位，HR/NDCG@5/10/20；best由Dev选。FAE严格调用未修改的BaseRunner.evaluate_method及原argsort并列处理。Test历史使用按时间排序的Train+此前Dev，排除当前Test目标；训练与Dev不读取Test。Test前后checkpoint SHA与内存state_dict逐值不变。
- FAE固定D128、2层2头、dropout .4、mapper dropout .3、Adam lr .001/weight_decay0、batch64/eval128、item温度.05、cl_weight.2、w_min.9/w_max1/period30、max200、early_stop10；两数据集同一算法。没有据Test改任何设置。
- PoMRec BPR：Beauty K4/lr.002/lamb4；ML K2/lr.001/lamb1；D64/A8/prompt3+n_layers2对应原prompt常量槽、batch1024、use_llmemb0。属于合理历史完整模型配置参考，不是同K同loss的纯结构消融。
- ASPCF BPR和Softmax均K4/D64/A64/H20、batch1024、relation .01，Beauty lr .004、ML lr .001。Softmax额外使用批内合法候选，训练负例监督与FAE BCE、ASPCF BPR不同，不能把差距归因于单一编码器/LLM/兴趣模块。
- 历史Test仅从既有实际日志/summary读取，所有6个基线checkpoint实际SHA与归档一致。ASPCF BPR原Test日志与后续冻结C0测试记录相符。历史部分指标只有四位小数，表中补零不增加精度。没有重新运行任何旧基线Test。
- 当前CSV/cache哈希快照、原始命令、归档路径与checkpoint SHA保存在baseline_references_delayed60_20261015.json。并非所有历史运行都有当时CSV哈希，无法仅凭当前快照证明长期文件从未变化；既有协议及缓存可信度审查未发现不一致，本轮未重做全量数据审计。

## 与官方论文设置仍存在的差异

恢复epoch>60并不等于完整原论文复现。仍采用本项目Train/Dev/Test与H20、batch64（作者默认256）、seed42、最多200轮（原500）、Dev N5选模（原评价指标/全目录协议不同）、当前PCA1536来源语义；其完整生成来源链仍未闭合。本轮无新LLM生成。官方Mapper/PCA初始化/融合、BCE和双层对齐、周期调度、原训练与服务评分不对称均保留。

保留官方每用户Train末段seq-to-seq训练单位，而ASPCF是逐前缀样本，监督频次不同。训练负例过滤限定Train，排除额外dummy，属于合法数据适配。Test只有候选接口适配，不调用官方全物品评价。不能用该基线的高低分反证原论文有效性。

## 证据、权重与复现入口

### beauty

新run：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/beauty_seed42_delayed60_20261015

best SHA256：9f23925cec58f4adf8c0e0ac42a7758e435510b004656251e331ff363b040dfb

旧best保留：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/beauty_seed42_20261015/best.pt

旧SHA256：4b123e1981f5d1e802f10bcd4c520c469dde6228ba1e9d9895208d6e1842b4a0

新summary.json / config.json / data_protocol.json / epochs.jsonl：训练与Dev；test_once.started.json、test_once.json：仅一次Test执行与指标。外部.out为完整普通日志。

### ml-1m

新run：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/ml-1m_seed42_delayed60_20261015

best SHA256：11034426986ebd7ac7f0217c088ae6f45f972c22d983504ef451add4012cef20

旧best保留：/home/yyx/hzgcode/pom2.0/research/2026_baselines/FAERec_repro/runs/ml-1m_seed42_20261015/best.pt

旧SHA256：1545d0b52d8ef740c6227f7540a06f8dd0fece555e178c79a2ff2974061375b8

新summary.json / config.json / data_protocol.json / epochs.jsonl：训练与Dev；test_once.started.json、test_once.json：仅一次Test执行与指标。外部.out为完整普通日志。

基线明细：runs/baseline_references_delayed60_20261015.json；汇总验收：runs/delayed60_acceptance_20261015.json。

原始文件行为：新增adapter/run_faerec_delayed.py和adapter/evaluate_test_once.py；旧run_faerec.py、upstream、原ASPCF/PoMRec、主入口、数据/LLM/评价器未改。仅README和本比较报告更新说明。

## 多种子与停止决定

若用于正式论文定量结论，仍需要预先声明配对种子补齐训练稳定性；seed42及一次研发Test不足以声称统计优势。当前只完成本次授权两run和各一次Test，不自动增加种子或调参。FAE作为非显式多兴趣外部基线保留；第四章创新与第五章开发不在本轮范围。


## 结果解释

- Beauty：新日程Dev N5相对旧日程提高约6.60%。本次Test N5 .122175高于ASPCF BPR .1088（约+12.29%），低于ASPCF Softmax .1294（约−5.58%）；HR@5提高但HR@10/20没有全面超过ASPCF BPR，不能宣称全指标胜出。
- ML-1M：新日程Dev N5提高约4.46%，但Test N5 .123408仍明显低于PoMRec BPR .2144、ASPCF BPR .2134及Softmax .2418，六项均低于这三组参考。因此提前验证不是当前低分的充分解释；本轮没有证据把剩余差距归因于某一因素。
- 验收结论：执行、重载、固定候选与一次性Test的正确性验收通过；“达到原论文强性能”的结论未成立。现有差异（batch、语义来源、训练单位、统一选模/评测）限制外部效度，不以低分证明官方方法无效。
- 不使用已见Test推动任何补救调参。若正式论文需报告该适配基线，应先固定当前定义并另行授权配对种子验证；如先核查剩余实现/协议差异，也必须另立事前问题，不能以追逐Test为目标。本轮停止。

# 第四章 C1 正式冻结与恢复记录（2026-10-24）

真实仓库：/home/yyx/hzgcode/pom2.0；环境 hzg_py10；Git HEAD：771b4c3826f07a8cc7fb9038fb8659161e9f8189。未提交源码以索引中的 SHA 为准，HEAD 不能完整恢复未跟踪快照。

第三章：models/sequential/LLMMIRecASPCF.py::LLMMIRecASPCF；SS：LLMMIRecASPCFSampledSoftmax.py::LLMMIRecASPCFSampledSoftmax。通用入口 main.py --model_name LLMMIRecASPCF；实际第四章SS入口为各实验 train_task.py::ObjectiveSIL（objective_model.py），其SoftGuideSIL/C1继承 base_snapshot.py::LLMMIRecASPCFAdaptiveSIL。现行主源码 models/sequential/LLMMIRecASPCFAdaptiveSIL.py；正式权重须配对应快照、config与SOURCE_LOCK恢复，不推定全部快照与主源码相同。

## 两项核心结构（真实计算）

令 ASPCF 物品向量为 x_t，C1 按时间顺序每4步成片段，局部多头注意力仅允许相同片段且 s<=t 的有效位置。Q,K,V=W_qkv LN(x)，a_ts=masked_softmax(Q_t K_s^T/sqrt(d_head))，h_t=LN(x_t+W_o concat_head(sum_s a_ts V_s))。padding置零，不允许局部未来或跨片段读取。

片段记忆 m_j=mean_{t in j} h_t；细粒度记忆为最近最多3个真实行为 x_t。已在近期行为中的单例片段被屏蔽，避免重复注意力质量。近期上下文 r=mean(last3 x)+last(x)。q_k=W_q LN(v_k)+W_r LN(r)；将片段和近期行为拼为同一记忆 M，b_k=masked_softmax(q_k [W_key LN(M)]^T/sqrt(64))；v_new_k=v_k+W_out sum_j b_kj W_value M_j。只有一次联合softmax；W_out零初始化，长度<2的增量严格为0。兴趣聚合、Filtered Softmax与relation loss不变。

## 冻结结论与评测协议

Beauty K2 C1尚未稳定超过同容量C0；ML-1M A1 K3在seed0/42已有相对B0/C0的两种子正向证据。仅探索性两种子，不宣称显著性。未来六种子固定{0,1,2,3,41,42}。K是Dev选择的超参数，不能用Test重新选。

Genuine Train→Dev选择best→固定权重一次Test。六项顺序：HR5/N5/HR10/N10/HR20/N20。每用户1正+1000预存负例，重复候选槽位保留，NumPy逐行argsort排序；Test历史包含此前Dev。raw与locked_prior同遍输出。已有可信Test直接复用。

频次先验仅导出评分：s_corr=s_raw+lambda_raw*(log(1+c_i)-row_mean(log(1+c)))；lambda_raw=.30*locked_scale。尺度为Train用户折外(user%3)候选中心化raw评分RMS/频次RMS，用户整折排除；不得按新模型或Test拟合。每域资产精确值与SHA见index.train_only_assets；不混淆raw/修正分数或Dev/Test。

所有原始配置、seed、容量、checkpoint和日志路径/SHA、数据与embedding SHA、Dev/Test六项均在相邻机器索引。恢复时重建同config模型，严格load_state_dict对应best.pt；核验源文件/资产SHA，不覆盖旧输出。索引中的历史命令仅用于恢复定位，禁止直接复跑已验收任务。

## Beauty K2 / ML-1M A1 K3 与同K消融

每行两套六项；完整配置及权重指纹见机器索引。消融A1=去局部编码、A2=去行为记忆（保留近期query，恢复单例片段）。A1虽登记参数不变但16640参数不参与训练，不能当严格有效容量相等。

|数据|K|seed|模型|阶段|raw六项|locked-prior六项|
|---|---:|---:|---|---|---|---|
|beauty|2|0|SS-B0|Dev|0.200689 / 0.144969 / 0.263247 / 0.165224 / 0.332335 / 0.182727|0.252962 / 0.187805 / 0.323838 / 0.210680 / 0.400215 / 0.229917|
|beauty|2|0|SS-B0|Test|0.180432 / 0.129696 / 0.243527 / 0.150092 / 0.315342 / 0.168208|0.213030 / 0.155943 / 0.282386 / 0.178399 / 0.356929 / 0.197241|
|beauty|2|0|SS-C0|Dev|0.201672 / 0.145659 / 0.267182 / 0.166841 / 0.337075 / 0.184492|0.253723 / 0.186643 / 0.324867 / 0.209565 / 0.400304 / 0.228642|
|beauty|2|0|SS-C0|Test|0.182891 / 0.131520 / 0.247999 / 0.152555 / 0.319501 / 0.170636|0.215535 / 0.157281 / 0.285829 / 0.179987 / 0.362831 / 0.199417|
|beauty|2|0|SS-C1|Dev|0.203059 / 0.145899 / 0.268747 / 0.167169 / 0.341323 / 0.185488|0.253499 / 0.186531 / 0.326208 / 0.210064 / 0.405044 / 0.230014|
|beauty|2|0|SS-C1|Test|0.187184 / 0.132423 / 0.250861 / 0.152996 / 0.325404 / 0.171872|0.217860 / 0.157310 / 0.291151 / 0.180971 / 0.368600 / 0.200536|
|beauty|2|42|SS-A1|Dev|0.202433 / 0.145879 / 0.270894 / 0.167951 / 0.338595 / 0.184978|0.253812 / 0.187106 / 0.325538 / 0.210365 / 0.403390 / 0.230032|
|beauty|2|42|SS-A1|Test|0.186916 / 0.134462 / 0.250503 / 0.154968 / 0.321245 / 0.172832|0.216161 / 0.157855 / 0.288065 / 0.181074 / 0.366275 / 0.200833|
|beauty|2|42|SS-A2|Dev|0.201628 / 0.146362 / 0.266243 / 0.167280 / 0.334347 / 0.184486|0.252426 / 0.186142 / 0.325895 / 0.209859 / 0.399007 / 0.228376|
|beauty|2|42|SS-A2|Test|0.184099 / 0.131098 / 0.248401 / 0.151853 / 0.319859 / 0.169921|0.214774 / 0.156176 / 0.285963 / 0.179054 / 0.363413 / 0.198609|
|ml-1m|3|0|SS-B0|Dev|0.367550 / 0.267615 / 0.478974 / 0.303731 / 0.584272 / 0.330365|0.399669 / 0.292418 / 0.512086 / 0.328860 / 0.621689 / 0.356654|
|ml-1m|3|0|SS-B0|Test|0.337914 / 0.246398 / 0.436755 / 0.278402 / 0.544536 / 0.305695|0.368212 / 0.270443 / 0.476656 / 0.305458 / 0.589404 / 0.334010|
|ml-1m|3|0|SS-C0|Dev|0.374834 / 0.273988 / 0.486258 / 0.310135 / 0.593046 / 0.337160|0.404801 / 0.297426 / 0.520861 / 0.334923 / 0.637583 / 0.364443|
|ml-1m|3|0|SS-C0|Test|0.344040 / 0.252694 / 0.450331 / 0.287035 / 0.556291 / 0.313824|0.375331 / 0.276037 / 0.488411 / 0.312631 / 0.598179 / 0.340451|
|ml-1m|3|0|SS-C1|Dev|0.384106 / 0.281068 / 0.493377 / 0.316317 / 0.595861 / 0.342312|0.417384 / 0.309794 / 0.528477 / 0.345668 / 0.636424 / 0.373033|
|ml-1m|3|0|SS-C1|Test|0.350993 / 0.255708 / 0.452152 / 0.288508 / 0.559437 / 0.315646|0.383609 / 0.282484 / 0.489735 / 0.316760 / 0.601987 / 0.345231|
|ml-1m|3|42|SS-A1|Dev|0.373510 / 0.272492 / 0.482450 / 0.307799 / 0.588245 / 0.334543|0.410099 / 0.302672 / 0.517219 / 0.337223 / 0.629470 / 0.365544|
|ml-1m|3|42|SS-A1|Test|0.334106 / 0.243610 / 0.436424 / 0.276689 / 0.549007 / 0.305196|0.368046 / 0.271507 / 0.478974 / 0.307394 / 0.591887 / 0.336020|
|ml-1m|3|42|SS-A2|Dev|0.373344 / 0.272410 / 0.486424 / 0.308956 / 0.587583 / 0.334541|0.404967 / 0.296913 / 0.515397 / 0.332744 / 0.621854 / 0.359790|
|ml-1m|3|42|SS-A2|Test|0.337748 / 0.245518 / 0.443543 / 0.279709 / 0.552815 / 0.307424|0.367219 / 0.271691 / 0.477152 / 0.307304 / 0.592219 / 0.336341|
|ml-1m|3|42|SS-B0|Dev|0.372351 / 0.273183 / 0.479801 / 0.308014 / 0.586093 / 0.335010|0.404801 / 0.297856 / 0.510430 / 0.332011 / 0.625000 / 0.361123|
|ml-1m|3|42|SS-B0|Test|0.339238 / 0.246461 / 0.449172 / 0.281777 / 0.552483 / 0.307896|0.368377 / 0.270752 / 0.485099 / 0.308436 / 0.592550 / 0.335699|
|ml-1m|3|42|SS-C0|Dev|0.380960 / 0.278645 / 0.484106 / 0.312095 / 0.592219 / 0.339404|0.406788 / 0.300086 / 0.521689 / 0.337279 / 0.628477 / 0.364287|
|ml-1m|3|42|SS-C0|Test|0.343377 / 0.249399 / 0.449172 / 0.283584 / 0.549007 / 0.308894|0.376821 / 0.275138 / 0.486589 / 0.310490 / 0.594040 / 0.337715|
|beauty|2|42|SS-B0|Dev|0.198229 / 0.143102 / 0.265528 / 0.164843 / 0.335241 / 0.182450|0.250771 / 0.186432 / 0.324509 / 0.210292 / 0.403926 / 0.230292|
|beauty|2|42|SS-B0|Test|0.180700 / 0.129108 / 0.245092 / 0.149839 / 0.312123 / 0.166702|0.212360 / 0.154051 / 0.282207 / 0.176512 / 0.362518 / 0.196775|
|beauty|2|42|SS-C0|Dev|0.200376 / 0.144569 / 0.264947 / 0.165443 / 0.333944 / 0.182834|0.253857 / 0.187541 / 0.326700 / 0.211091 / 0.401824 / 0.230079|
|beauty|2|42|SS-C0|Test|0.188660 / 0.135052 / 0.251800 / 0.155479 / 0.321111 / 0.173056|0.219380 / 0.160210 / 0.289228 / 0.182801 / 0.366722 / 0.202374|
|beauty|2|42|SS-C1|Dev|0.201762 / 0.146115 / 0.264365 / 0.166334 / 0.333363 / 0.183740|0.254706 / 0.189103 / 0.322273 / 0.210968 / 0.396593 / 0.229737|
|beauty|2|42|SS-C1|Test|0.183294 / 0.131364 / 0.247775 / 0.152245 / 0.316594 / 0.169598|0.217100 / 0.157974 / 0.287350 / 0.180660 / 0.361624 / 0.199440|
|ml-1m|3|42|SS-C1|Dev|0.391556 / 0.284759 / 0.490397 / 0.316730 / 0.598841 / 0.344318|0.417715 / 0.310573 / 0.525662 / 0.345455 / 0.635927 / 0.373351|
|ml-1m|3|42|SS-C1|Test|0.350497 / 0.255690 / 0.458775 / 0.290505 / 0.560762 / 0.316227|0.378477 / 0.278324 / 0.493709 / 0.315610 / 0.599503 / 0.342329|

## 历史K、BPR、M1/M2

历史K与BPR/M2均完整保留在索引 records 和 historical_fixed_Test（含原报告路径、SHA与配置）。M1硬语义阈值分段没有稳定超过固定C1；ML K8单点收益不能概括为全面优越。M2为同双粒度读取上的语义偏置复用，额外K参数，不能称为独立新基础算子。BPR/SS横跨不同Loss、最佳K与容量时不冒充严格结构归因。

|数据|K|模型|raw Test N5|先验 Test N5|来源|
|---|---:|---|---:|---:|---|
|beauty|2|SS-B0|0.129696|0.155943|research/elector_sil_night12_20261023_v1/runs/beauty_k2_s0_SS-B0/result.json|
|beauty|2|SS-C0|0.131520|0.157281|research/elector_sil_night12_20261023_v1/runs/beauty_k2_s0_SS-C0/result.json|
|beauty|2|SS-C1|0.132423|0.157310|research/elector_sil_night12_20261023_v1/runs/beauty_k2_s0_SS-C1/result.json|
|beauty|2|SS-A1|0.134462|0.157855|research/elector_sil_night12_20261023_v1/runs/beauty_k2_s42_SS-A1/result.json|
|beauty|2|SS-A2|0.131098|0.156176|research/elector_sil_night12_20261023_v1/runs/beauty_k2_s42_SS-A2/result.json|
|ml-1m|3|SS-B0|0.246398|0.270443|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s0_SS-B0/result.json|
|ml-1m|3|SS-C0|0.252694|0.276037|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s0_SS-C0/result.json|
|ml-1m|3|SS-C1|0.255708|0.282484|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s0_SS-C1/result.json|
|ml-1m|3|SS-A1|0.243610|0.271507|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s42_SS-A1/result.json|
|ml-1m|3|SS-A2|0.245518|0.271691|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s42_SS-A2/result.json|
|ml-1m|3|SS-B0|0.246461|0.270752|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s42_SS-B0/result.json|
|ml-1m|3|SS-C0|0.249399|0.275138|research/elector_sil_night12_20261023_v1/runs/ml-1m_k3_s42_SS-C0/result.json|
|beauty|2|SS-B0|0.129108|0.154051|research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-B0/result.json|
|beauty|2|SS-C0|0.135052|0.160210|research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C0/result.json|
|beauty|2|SS-C1|0.131364|0.157974|research/elector_sil_formal_20261023_v1/runs/beauty_k2_s42_SS-C1/result.json|
|beauty|8|SS-B0|0.134741|0.157131|research/elector_sil_formal_20261023_v1/runs/beauty_k8_s42_SS-B0/result.json|
|beauty|8|SS-C0|0.129968|0.155231|research/elector_sil_formal_20261023_v1/runs/beauty_k8_s42_SS-C0/result.json|
|beauty|3|SS-C1|0.134331|0.156350|research/elector_sil_k3_20261023_v1/runs/beauty_k3_s42_SS-C1/result.json|
|ml-1m|3|SS-C1|0.255690|0.278324|research/elector_sil_k3_20261023_v1/runs/ml-1m_k3_s42_SS-C1/result.json|
|beauty|3|SS-B0|0.134394|0.156177|research/elector_sil_supp7_20261023_v1/runs/beauty_k3_s42_SS-B0/result.json|
|beauty|3|SS-C0|0.130894|0.154219|research/elector_sil_supp7_20261023_v1/runs/beauty_k3_s42_SS-C0/result.json|
|beauty|4|SS-A1|0.131642|0.158044|research/elector_sil_supp7_20261023_v1/runs/beauty_k4_s42_SS-A1/result.json|
|beauty|4|SS-A2|0.131848|0.155410|research/elector_sil_supp7_20261023_v1/runs/beauty_k4_s42_SS-A2/result.json|
|beauty|8|SS-C1|0.132383|0.155575|research/elector_sil_supp7_20261023_v1/runs/beauty_k8_s42_SS-C1/result.json|
|ml-1m|2|SS-A1|0.246539|0.274455|research/elector_sil_supp7_20261023_v1/runs/ml-1m_k2_s42_SS-A1/result.json|
|ml-1m|2|SS-A2|0.249443|0.270752|research/elector_sil_supp7_20261023_v1/runs/ml-1m_k2_s42_SS-A2/result.json|
|beauty|4|BPR-B0|0.111858|0.110366|research/elector_sil_objective_full_20261023_v2/beauty/arms/BPR-B0/result.json|
|beauty|4|BPR-C1|0.106191|0.109990|research/elector_sil_objective_full_20261023_v2/beauty/arms/BPR-C1/result.json|
|beauty|4|SS-C0|0.129981|0.155925|research/elector_sil_objective_full_20261023_v2/beauty/arms/SS-C0/result.json|
|beauty|4|SS-M2|0.131477|0.154663|research/elector_sil_objective_full_20261023_v2/beauty/arms/SS-M2/result.json|
|ml-1m|2|BPR-B0|0.203733|0.196489|research/elector_sil_objective_full_20261023_v2/ml-1m/arms/BPR-B0/result.json|
|ml-1m|2|BPR-C1|0.224393|0.218166|research/elector_sil_objective_full_20261023_v2/ml-1m/arms/BPR-C1/result.json|
|ml-1m|2|SS-C0|0.248720|0.274520|research/elector_sil_objective_full_20261023_v2/ml-1m/arms/SS-C0/result.json|
|ml-1m|2|SS-M2|0.251817|0.280092|research/elector_sil_objective_full_20261023_v2/ml-1m/arms/SS-M2/result.json|
|beauty|4|B0|0.127853|0.153401|historical_fixed_Test:beauty_k4_B0|
|beauty|4|C1|0.131428|0.155037|historical_fixed_Test:beauty_k4_C1|
|beauty|4|M1|0.131110|0.154911|historical_fixed_Test:beauty_k4_M1|
|ml-1m|2|B0|0.242137|0.265489|historical_fixed_Test:ml-1m_k2_B0|
|ml-1m|2|C1|0.254955|0.282760|historical_fixed_Test:ml-1m_k2_C1|
|ml-1m|2|M1|0.249014|0.277088|historical_fixed_Test:ml-1m_k2_M1|
|ml-1m|4|B0|0.244156|0.270136|historical_fixed_Test:ml-1m_k4_B0|
|ml-1m|4|C1|0.252526|0.279545|historical_fixed_Test:ml-1m_k4_C1|
|ml-1m|4|M1|0.252384|0.279319|historical_fixed_Test:ml-1m_k4_M1|
|ml-1m|8|B0|0.242113|0.267269|historical_fixed_Test:ml-1m_k8_B0|
|ml-1m|8|C1|0.251880|0.277896|historical_fixed_Test:ml-1m_k8_C1|
|ml-1m|8|M1|0.255515|0.280420|historical_fixed_Test:ml-1m_k8_M1|

## 尚缺工作

六种子确认性复现/区间与显著性尚缺，历史Test暴露后的结构选择须如实标明探索性。Beauty对同容量C0未成立稳定优势；不可按域关闭C1模块。Toys四组为独立既有任务，BPR仍在运行，SS评测修复/恢复另记；不得因归档移动或覆盖权重。第四章冻结，第五章新增分支继承C1，不改写本章源码。

启动与恢复环境：source /home/yyx/anaconda3/etc/profile.d/conda.sh; conda activate hzg_py10; cd /home/yyx/hzgcode/pom2.0; export PYTHONDONTWRITEBYTECODE=1。原任务实际CLI和参数见每个record.config及train_command，调度入口见原目录scheduler.py与LAUNCH；原命令存在独占输出保护，不可直接重跑。

同K消融现有记录为seed42；seed0消融尚缺。索引train_command已由各真实入口的argparse接口核对，旧目录有独占输出保护，只用于恢复定位。

补充BPR/M2归因结论（固定已验收权重，不重跑Test）：Beauty K4 BPR-B0→BPR-C1原始Test N5为0.111858→0.106191，ML A1 K2为0.203733→0.224393，故结构效果跨域不同，不能宣称无条件增益。SS-M2 Beauty K4原始/先验N5=0.131477/0.154663，ML K2=0.251817/0.280092；M2在两域的先验N5均低于相应C1（0.155037/0.282760），Beauty原始分数局部更好，属评分口径混合证据，不继续硬阈值/语义偏置搜索。横跨Loss和先验的数值不当作严格结构收益。

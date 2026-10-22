# ML-1M 生产准备：路线A / B（2026-10-20）
本次只修改既有独立V2目录、CPU测试及只读检查。V1-Original永久不改；没有7B前向、SCFT训练、批量编码、正式PCA或推荐评估。

## 本轮结论
- 旧命令1是Toys/max180；ML留存脚本max64。命令3是PoMRec物品协同向量导出模板，不是Prompt生成。命令片段1缺--out_jsonl、3缺--dataset，不能当完整执行证据。P920和租用机bash history均无匹配行。
- 转换器先保存完整raw，再把非有限行均值插补后PCA。因此会留下raw NaN，并可解释PCA均值行中心化近零；不会合法生成连续1568行零块。该零块和两个边界部分归零仍是未定位的文件损伤，不“修复”为均值。
- 仅租用机旧PCA原地候选恢复检查：3577正常raw行，重投影相对误差0.0114203、最大0.0425155、正交误差0.0165532，未通过预设1e-5/1e-4门槛；只在内存计算，未拟合/保存正式PCA。
- P920当前PCA版本不同，跨机复制只读参考表被自动审批拒绝，未绕过。故当前未能可靠恢复P920旧坐标，禁止把新129行PCA插入旧表。路线A采用独立V1-Rebuilt全目录重编码及新PCA。
- 原LoRA448张量全有限、SHA已知；tau只在labels存在的SCFT loss中使用。CPU微型模型改变tau后推理输出逐元素一致，无“必须重训”理由。

## 代码复用及生产修正
原upstream 64文件不改；旧适配代码快照在checks/production_source_before。
- common/model_patch：原masked FP32池化、原SCFT FP32代数；严格ID升序/唯一/完整、finite、非padding范数>1e-4、无未解释重复向量。无均值/零/随机插补。
- training_support.V2Trainer：只覆盖保存与优化器有限梯度保护，不重写训练循环。明确只存adapter+tau，避免通用Trainer将vendored PEFT当普通nn.Module而写13GB底座。
- 检查点含tau、tokenizer、training_args.json/bin、Trainer optimizer/scheduler/trainer_state/RNG；on_train_end另存lora_final和额外RNG。
- run_scft仍调用固定官方main/Trainer/PEFT；SCFT前完整输入/代码/权重SHA。
- run_encode复用同一模型补丁；显式验证全部LoRA参数确已加载；保存raw.jsonl、table_raw.npy、manifest及COMPLETE。
- quality_pca可单表或Base/Rich：Base的Train目录物品上fit一份full-SVD PCA1536，统一mean/components；存模型/版本/seed/输入输出SHA，近零即失败。
- preflight不构建7B：检查文件、tokenizer、safetensors header/index、LoRA/输入身份、环境、输出保护、磁盘/GPU。
- scripts/route_a.sh / route_b.sh：默认check；生产需显式选择阶段及批准GPU_INDEX。所有HF/缓存路径限定本独立目录，离线模式；失败set -e，不自动改batch/精度或续跑。

## 路线A：原LoRA + V1-Rebuilt
输入：原3706 Base Prompt、canonical item2id、已找回ML checkpoint-1000；不使用损坏raw或插补PCA作新数值输入。
配置：configs/encode_v1_rebuilt.json；原max64、batch4、FP16基座、FP32 masked mean、原tokenizer/LoRA、seed42。
输出：runs/v1_rebuilt_raw/{raw.jsonl,table_raw.npy,manifest,COMPLETE}；runs/v1_rebuilt_pca1536/{base_pca.npy,pca_model.npz,manifest}。
这不是“原坐标只修129行”，新PCA会改变正常行坐标；Original保持逐元素不变。
完整旧raw的3577正常向量用于来源核对；不与新129行未经验证混成新表。
若未来取得原PCA模型或充分验证坐标，才单独批准只改129行的修复：必须校验正常行不变及新向量空间与旧空间一致，当前没有可用certificate，未开放写回路径。

租用机命令（本轮未执行生产阶段）：
```bash
cd /root/autodl-tmp/pomrec/research/ml1m_embedding_v2_20261020
bash scripts/route_a.sh check
GPU_INDEX=<已批准编号> bash scripts/route_a.sh encode
bash scripts/route_a.sh pca
```

## 路线B：V2共同新SCFT + Base/Rich编码
原Base Prompt固定SHA e399ec0c…；3883原电影映射3706，原字段不改。
Rich结构支持plot/overview（二选一）、themes、director、cast，字段可缺，但添加字段必须有电影title/year核验、source URL/ID/日期/许可/text SHA；不允许评分/交互统计和自由生成剧情。Rich目前无真实资料，示例只是schema，禁止拿它跑生产。
Base和Rich都读同一新checkpoint-1000，不能一臂用旧、一臂用新。仅Base电影元数据SCFT一次；这能隔离Rich文本输入作用，不单独对Rich再训另一模型。
预注册：
- r8/alpha32/dropout.1，q/k/v/o/down/gate/up；seed42；初始tau3、可训练FP32。
- 常规微批16原电影/32增强视图，累积8，优化器batch最多128；尾微批10，尾优化器步122。对比候选是当前微批16或10，不是128。
- 1000 optimizer step（232微批/epoch、29次更新/epoch，约34.48轮）、lr2e-4、AdamW betas(.9,.999)/eps1e-8/decay0，linear/warmup0、clip1、固定gradient checkpointing、FP16 AMP；使用单GPUHF Trainer，不启用旧DeepSpeed ZeRO-2，不根据结果调参。
- 标题两视图必保留；其余真实属性shuffle/drop.4，总体至少留两属性（若不足则全部保留）。Base每视图标题+年份或genre，随机允许两视图相同；不是自动制造难负例。
- 数据集每电影一条，epoch内无放回，保留原尾批drop_last=False（232微批可整除累积8）；两个增强视图是同电影正对。没有把同一电影不同Prompt作为跨样本负例。不同电影可能语义相似，原soft-target loss对此赋权；不冒称无false-negative。
- 保留官方soft-target对称代数（targets不detach、对称项使用targets.T），不是改写成hard-label InfoNCE。它不是标准双向独立归一化CE；这种区别明确保留，不称新loss。
- CPU测试确认FP32相似度/池化避免合成FP16溢出；不能据此保证7B每层稳定。非有限loss/权重/实际更新梯度失败，不填补。
- max256，两臂同encoder协议；themes24、plot/overview128、director16、cast24 token字段预算，再统一末端截断+EOS，逐项报告截断。Base没有剧情字段，不被字段预算改写。
- 预处理前seed42，单进程，新缓存目录；两视图在训练前生成一次，不宣称每轮重采样。
- 需≥24GiB实际空闲显存作微批16准入。16GB推理可验，未经GPU测试不承诺16GB完整SCFT；OOM停止，不减batch冒充等价。

命令：
```bash
bash scripts/route_b.sh check
GPU_INDEX=<已批准编号> bash scripts/route_b.sh train
GPU_INDEX=<已批准编号> bash scripts/route_b.sh encode-base
# Rich真实内容核验、SHA固定到配置后才可执行
GPU_INDEX=<已批准编号> bash scripts/route_b.sh encode-rich
bash scripts/route_b.sh pca
```
route_b train/encode都不会推荐Test；输出各为独立新目录，checkpoint和数值资产不覆盖。新训练失败不自动回落旧adapter；不自动微调另一模型。
配置中Rich expected_prompt_sha256为null，属于明确阻断，不允许脚本静默选一个文件。

## 环境/资源
租用机Python3.10.19，torch2.7.1+cu126/CUDA编译12.6，transformers4.30.2，外部peft0.6.2；实际使用bundled llm.peft，datasets4.4.2/deepspeed.18.3/numpy2.1/sklearn1.7.1/tokenizers.13.3。框架导入通过。
底座仅选择safetensors两分片，共13,476,876,272字节；本轮32.4秒锁定两SHA（configs/base_weights_lock.json），不读重复bin格式，不安装/清理任何文件。
autodl剩约13.6GiB，超过代码8GiB最低reserve；root约7.2GiB，缓存全部放autodl本工作区。预估两路线新增数GB，需保留余量，不删除已有资产。
当前无GPU（torch.cuda.available=False/0 devices/nvidia-smi No devices），不能验显存、AMP及真实吞吐，也不能承诺生产可启动。旧ML SCFT44.5min只属旧配置/未知硬件记录。
原始Prompt、mapping、LoRA和权重metadata在本机可用；Rich缺失。租用机将只接收SHA锁定源码包，不接收P920私有Embedding/PCA。P920为代码和CPU验证源，7B计算只安排租用机。

## 已验收/仍阻断
CPU：既有13+生产8项通过；tiny随机Llama/PEFT/保存测试，不执行Trainer.train，无7B加载。Shell语法通过；官方64源文件保持SHA。
仍阻断：GPU未分配；真实7B短样本前向与AMP显存未验；Rich元数据/身份/许可/Prompt SHA未取得；原PCA坐标恢复未通过，不能129行拼接；完整生产只在新授权后执行。
用户批准后可先走路线A或路线B-Base的GPU短样本准入；Rich阶段须先补齐真实资料。不得把当前准备称为全流程数值生产已成功。

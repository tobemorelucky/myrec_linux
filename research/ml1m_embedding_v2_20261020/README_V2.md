# ML-1M LLM Embedding V2：独立恢复工作区（2026-10-20）

本轮只完成来源追溯、独立代码修复、CPU合成检查和dry-run。**没有LoRA训练、7B前向、全量编码、真实PCA拟合、推荐训练或Dev/Test**。V1原文件不改。

## 来源与布局

- `upstream/`：官方LLMEmb固定commit `3458a5e225062e94b4f1a01e41f3ec82089f0407`，64文件约416KB，MIT；原文件未修改。SHA在UPSTREAM_MANIFEST.json。仓库https://github.com/Applied-Machine-Learning-Lab/LLMEmb ，论文https://arxiv.org/abs/2409.19925 。Llama2权重另受自身许可约束。
- `adapter/common.py`：ID、向量质量、FP32 masked mean、属性增强和不可覆盖写出。
- `prepare_prompts.py`：原movies.dat+canonical item2id；Rich只接收人工核验title/year、逐字段来源的JSON。无评分/交互特征。
- `model_patch.py`：继承原LlamaRSEmb；传递真实attention_mask，FP32均值与原SCFT公式，label=None不分对；学习tau保持FP32，异常立即失败。
- `run_scft.py`：直接调用固定官方main/Trainer/PEFT；仅移除强制GPU编号、替换上述补丁与保存回调。保存所有非LoRA可训练状态（含tau），禁止旧回调删除文件；不重写训练框架。
- `run_encode.py`：显式base+LoRA加载；无BOS、截L-1再EOS，与历史processor规则一致；右padding、unk pad、FP32池化；批次异常停止，未完成目录不会标COMPLETE。
- `quality_pca.py`：校验全部3706 ID+padding0、逐行有限和非零；Base/Rich必须encoder_key一致。只在Base的Train目录物品拟合一份PCA(full SVD, float64, whiten=False, seed42)，两者共享mean/components；保存npz模型及npy，不独立拟合、不填异常值。
- `configs/`：拟议配置，@V2_ROOT@自动指向当前独立目录，可整体部署后复用；不是已执行参数。所有重任务默认dry-run，需用户新授权后显式--execute。
- `tests/test_offline.py`：仅CPU微型随机Llama（16维、1层）、合成向量和合成PCA；不读7B权重。

## 已找回的V1

Windows原始movies.dat：
`E:\work\dataset\meta_ml1m\ml-1m\movies.dat`，3883条。与`E:\work\dataset\ml-1m\movies.dat` SHA一致。
Prompt转换脚本：
`E:\work\code\python\paper\PoMRec-main\data\ml-1m\build_item_prompt.py`及副本。
最终原Prompt在Windows dataset/ml-1m、上述代码data/ml-1m、P920当前项目、租用机actual project，SHA相同：
`e399ec0c6e58a85f6b1525d878f4865bc700199a32c3ef939e09bcf43d37e40d`。
仅title/year/genres，target为空；同一份电影列表由processor在线造两增强视图用于SCFT，再原文本用于编码。不是额外有监督label文件。增强版average rating用了全ratings.dat，不用于V2。

租用机实际ML脚本：
`/root/autodl-tmp/pomrec/experiments/ml-1m/pretrain/avg.bash`。
训练部分现被注释，推理部分仍指定checkpoint-1000；结合训练状态/训练参数可证1000步、batch16、accum8、lr2e-4、seed42、bf16=True、AdamW/decay0、beta(.9,.999)、eps1e-8、warmup0。主源码先.half().cuda()，不能把它描述为纯BF16端到端。
原max_source64，3706条实际27–56 token，无截断/空分母。LoRA r8/alpha32/dropout.1、q/k/v/o/down/gate/up，448 tensors、19,988,480参数，全有限。
原SCFT约2668.8628秒，不能据此确定原GPU型号。
checkpoint与lora_final SHA相同：
`6b33f280d7856fa37903c5b3582d5defceab6891ab39df82372ae66317bbe186`。
底座config为Llama-2-7b-hf、4096维，原revision未锁定；tau训练但旧LoRA未保存。不能承诺精确恢复训练/旧输出字节。

## 异常定位与限制

1. actual SCFT输出`results/llm-emb/0722_avg_ml1m.json`有3706 ID/4096维，129行非有限、0个全零；129 ID与现用PCA近零ID完全相同。输出SHA `2a950f5cf1acea823b2781cfe023a357e582e985eaf1d8f115a1016ae87c0c27`。
2. 租用机及Windows旧converter先保存raw，再把非有限行换成正常行均值后PCA；均值插补经中心化落近原点，支持129近零的解释。未找回当次PCA模型/solver/seed/完整运行日志，不能宣称字节复原。
3. P920 raw的81全NaN是上述129的子集；其余48位于连续零块。1568全零ID恰为1085..2652，1084最后37维和2653前4058维也为0。2055有效行与SCFT逐字节一致，2个边界行不一致。强烈指向raw文件块损伤/写出不完整，具体写入者/时间未知，不假称缺文本或统一NaN清洗。
4. 租用机与P920同名PCA的文件和数组SHA均不同，均有129近零，不能视为同一版本。
5. `/root/PomRec/data/results/llm-emb/ml-1m/default.json`为3706全有限旧base编码；文本/ID顺序匹配，但不是当前SCFT输出，也不能冒充旧PCA的精确raw。
6. 原forward忽略attention_mask、FP16池化/相似度有风险；合成测试证明溢出可能性，但未定位旧129行究竟在哪一层首次出现NaN。本轮不通过7B前向重现。
7. 原属性dropout把末尾空字段算入属性；原随机增强在Trainer设置seed之前、且缓存可能复用，旧增强样本不能精确重建。V2先设置seed、只保留真实字段和至少一个属性。

## 待批准后执行的唯一主流程

优先**复用已找回且全有限的原LoRA**，不以权重丢失为由直接重训。

1. 审批目标GPU/独立目录部署和inputs staging；只复制canonical mapping和纯物品元数据。Windows原始数据与V2模板已在内存逐ID核对，3706 Prompt完全一致。
2. 经审批取得Rich真实元数据：每条电影ID/title/year人工可复核、来源URL/外部ID/时间/许可/text SHA齐全。无元数据不造剧情；统计缺失及fallback。禁止average rating、评论、榜单和Dev/Test交互。
3. 固定一个encoder+tokenizer+LoRA：两臂均max256、batch1、FP16底座+FP32池化、相同特殊token；旧adapter缺tau仅推理可接受（tau不影响编码）；不得因此精确resume SCFT。
4. 先另获批准做小样本GPU编码验收，再各生成一次V2-Base/V2-Rich；通过全目录QA后才拟合一次Base/Train-catalog PCA，统一变换两臂。两个COMPLETE/encoder_key、权重分片SHA、源码SHA、Prompt SHA、包版本都保留。
5. 若用户明确决定重新SCFT：独立new run，沿原公开SCFT，不加载旧输出作为监督；建议仅Base训练一次后两臂共用同一adapter，以隔离Rich文本影响。拟议microbatch16/accum8/1000步/lr2e-4/r8/alpha32/.1/drop.4/seed42；max256、FP16+FP32数值修复、显式gradient checkpointing，是与旧流程的差异。梯度累积不能弥补缩小contrastive microbatch造成的负样本变化；OOM停止，不暗改batch。
6. 再以独立授权运行下游推荐配对；V1永久保留。禁止按Test选资产或算法。

命令草案（**本轮不可执行--execute**，需先将本目录部署到有底座的租用机独立目录，填充经批准inputs）：
```bash
cd /root/autodl-tmp/pomrec/research/ml1m_embedding_v2_20261020
# 原始metadata转换：Rich参数另加 --rich inputs/rich_verified.json
python -B adapter/prepare_prompts.py --movies inputs/movies.dat --mapping inputs/item2id.json --out inputs/base_prompts.jsonl --execute
# 复用旧LoRA时直接进入Base/Rich同encoder编码；必须先GPU小样本准入
python -B adapter/run_encode.py --config configs/encode_base_proposed.json --execute
python -B adapter/run_encode.py --config configs/encode_rich_proposed.json --execute
python -B adapter/quality_pca.py --base-dir runs/encode_base --rich-dir runs/encode_rich --mapping inputs/item2id.json --train inputs/train.csv --out runs/common_pca1536 --execute
# 仅当另行明确批准从头SCFT时；编码配置同时指向这个唯一新adapter，禁止只换一臂
python -B adapter/run_scft.py --config configs/scft_proposed.json --execute
```

## 成本与未闭合事项

- 旧基座约13GB，raw每臂约58MiB，PCA1536每臂约21.7MiB；单份common float64 PCA components约48MiB。保存JSON约0.3–0.5GB/臂，含LoRA/优化器/日志建议另留10–20GB（不含基座），不覆盖旧文件。
- 原ML SCFT1000步44.5min仅是历史记录、硬件未知。新Base训练仍短文本，预计同档硬件约1–2小时仅作规划；16GB能否保持microbatch16未测，建议≥32/48GB GPU作稳妥准入，不承诺Quadro5000直接可跑。
- Base≤56token、Rich≤256token，总输入token上界约1.16M，两臂编码耗时需小样本测速；PCA/full SVD CPU规划分钟级，未在真实4096数据执行。
- 本轮租用机nvidia-smi为No devices；完整GPU训练/编码尚未验收。Rich资料未收集，旧base weight revision未完全恢复。**代码与CPU检查就绪，不等于现已具备可立即启动的完整V2生产条件。**

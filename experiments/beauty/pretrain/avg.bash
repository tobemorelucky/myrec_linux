export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true


dataset="beauty"
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/root/autodl-tmp/LLMemb/llama-7b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="data/${dataset}/"  # 你的数据集路径，未做任何修改
your_checkpopint_path="/root/autodl-tmp/pomrec/saved"  # 模型存储路径，未做任何修改
MAX_STEPS=1000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="0722_avg_beauty"
MAX_SOURCE_LENGTH=224

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

# Training Command
# deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm.py \
#     --deepspeed llm/ds.config \
#     --do_train \
#     --train_file $your_data_path/item_str.jsonline \
#     --cache_dir $your_data_path \
#     --prompt_column input \
#     --response_column target \
#     --model_name_or_path $model_name_or_path \
#     --output_dir $your_checkpopint_path/lora-$date \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 196 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_steps ${MAX_STEPS} \
#     --logging_steps 100 \
#     --save_steps ${MAX_STEPS} \
#     --learning_rate $LR \
#     --lora_rank ${lora_rank} \
#     --trainable ${lora_trainable} \
#     --modules_to_save ${modules_to_save} \
#     --lora_dropout ${lora_dropout} \
#     --pool_type avg \
#     --dropout_ratio 0.4 \
#     --bf16 \
#     --save_strategy steps \
#     --save_steps 500 \
#     --save_total_limit 3




# Training Command
# deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm.py \
#     --deepspeed llm/ds.config \
#     --do_train \
#     --train_file $your_data_path/item_str.jsonline \
#     --cache_dir $your_data_path \
#     --prompt_column input \
#     --response_column target \
#     --overwrite_cache \
#     --model_name_or_path $model_name_or_path \
#     --output_dir $your_checkpopint_path/lora-$date \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 196 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_steps ${MAX_STEPS} \
#     --logging_steps 100 \
#     --save_steps ${MAX_STEPS} \
#     --learning_rate $LR \
#     --lora_rank ${lora_rank} \
#     --trainable ${lora_trainable} \
#     --modules_to_save ${modules_to_save} \
#     --lora_dropout ${lora_dropout} \
#     --pool_type avg \
#     --dropout_ratio 0.4 \
#     --bf16

# Testing Command
deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm.py \
    --do_predict \
    --test_file $your_data_path/item_str.jsonline \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/lora-$date/checkpoint-$MAX_STEPS \
    --output_dir results/llm-emb \
    --output_file $date.json \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --pool_type avg
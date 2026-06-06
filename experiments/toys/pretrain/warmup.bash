export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true

dataset="beauty"
your_data_path="data/${dataset}"
model_name_or_path="/root/autodl-tmp/LLMemb/llama-7b"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm.py \
  --deepspeed llm/ds.config \
  --do_train \
  --train_file ${your_data_path}/item_str.jsonline \
  --cache_dir ${your_data_path} \
  --prompt_column input \
  --response_column target \
  --overwrite_cache \
  --model_name_or_path ${model_name_or_path} \
  --output_dir /root/autodl-tmp/Pomrec2.0 \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 196 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 5 \
  --logging_steps 1 \
  --save_steps 9999999 \
  --learning_rate 2e-4 \
  --lora_rank 8 \
  --trainable q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
  --modules_to_save null \
  --lora_dropout 0.1 \
  --pool_type avg \
  --dropout_ratio 0.4 \
  --fp16

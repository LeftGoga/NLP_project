#!/bin/bash

# ==============================
# Parameters settings
# ==============================
BS=64
USE_SWICHBACK=true
CUDA_VISIBLE_DEVICES=0
NUM_FTUNES=1
MODEL_NAME_OR_PATH="BASE MODEL PATH"
OUTPUT_MODEL_NAME="BASE MODEL NAME"
OUTPUT_DIR="../models_iter_ft_swichback_${USE_SWICHBACK}"
LOG_DIR="./logs_use_switchback_${USE_SWICHBACK}"
EVAL_TASKS="hellaswag,boolq,swag,winogrande,xwinograd_en"
EVAL_OUTPUT_DIR="./lmeval_res_use_switchback_${USE_SWICHBACK}"
FT_DATASET=oasst1

# ==============================
# Creating folders
# ==============================
mkdir -p "${LOG_DIR}/ft"
mkdir -p "${LOG_DIR}/eval"
mkdir -p "${EVAL_OUTPUT_DIR}"

# ==============================
# Main Loop
# ==============================
for ((i=1; i<=NUM_FTUNES; i++))
do
    echo "Starting fine-tuning iteration $i"
    
    python3 script.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --output_dir "${OUTPUT_DIR}/${OUTPUT_MODEL_NAME}_iter${i}" \
        --logging_steps 10 \
        --save_strategy epoch \
        --data_seed 42 \
        --save_total_limit 6 \
        --max_new_tokens 32 \
        --dataloader_num_workers 3 \
        --group_by_length=False \
        --logging_strategy steps \
        --remove_unused_columns False \
        --do_train \
        --warmup_ratio 0.05 \
        --lr_scheduler_type constant \
        --dataset "${FT_DATASET}" \
        --source_max_len 16 \
        --target_max_len 512 \
        --per_device_train_batch_size "${BS}" \
        --max_steps 0 \
        --num_train_epochs 5 \
        --learning_rate 1e-5 \
        --adam_beta2 0.999 \
        --max_grad_norm 1.0 \
        --weight_decay 0.0 \
        --seed 0 \
        --trust_remote_code \
        --swichback_mode "${USE_SWICHBACK}" \
        --fp16 \
        1>"${LOG_DIR}/ft/ft_${i}.log" 2>&1

    echo "Determining latest checkpoint for iteration $i"
    CHECKPOINT_SUFFIX=$(ls -d "${OUTPUT_DIR}/${OUTPUT_MODEL_NAME}_iter${i}/checkpoint-"* | sort -V | tail -n 1 | awk -F'/' '{print $NF}')

    echo "Using checkpoint: $CHECKPOINT_SUFFIX for evaluation"

    echo "Starting evaluation for fine-tuning iteration $i"
    
    lm_eval --model hf \
            --model_args "pretrained=${OUTPUT_DIR}/${OUTPUT_MODEL_NAME}_iter${i}/${CHECKPOINT_SUFFIX}" \
            --tasks "$EVAL_TASKS" \
            --device cuda:"$CUDA_VISIBLE_DEVICES" \
            --batch_size 32 \
            --output_path "${EVAL_OUTPUT_DIR}/ft_${i}_eval" \
            1>"${LOG_DIR}/eval/eval_${i}.log" 2>&1
done

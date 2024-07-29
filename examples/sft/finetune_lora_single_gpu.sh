#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="Qwen/Qwen2-1.5B-Instruct" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="gsm8k/gsm8k_train.jsonl"
# EVAL_DATA="../qwen2_gsm8k_test.jsonl"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=0

python ./examples/sft/finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --do_train True \
  --bf16 True \
  --output_dir gsm8k/output_qwen_1_3 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 150 \
  --save_total_limit 10 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora

# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json
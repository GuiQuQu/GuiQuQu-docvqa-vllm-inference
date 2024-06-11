#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
PROJECT_DIR="/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
cd $PROJECT_DIR

MODEL="/home/klwang/pretrain-model/Qwen-VL-Chat-Int4" # Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA_DIR="/home/klwang/data/spdocvqa-dataset"
DATA="$DATA_DIR/train_v1.0_withQT.json"
OCR_DIR="$DATA_DIR/ocr"
IMAGE_DIR="$DATA_DIR/images"
DS_CONFIG_PATH="${PROJECT_DIR}/ds_config/ds_config_zero2.json"
USE_LORA=True
Q_LORA=True

export CUDA_VISIBLE_DEVICES=0

# Remember to use --fp16 instead of --bf16 due to autogptq
# --fix_vit只有在全量微调时才会生效
python ${PROJECT_DIR}/src/finetune_qwen-vl.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --ocr_dir $OCR_DIR \
    --image_dir $IMAGE_DIR \
    --layout_type "none" \
    --add_image True \
    --add_layout False \
    --fp16 True \
    --fix_vit True \
    --output_dir $PROJECT_DIR/output_qwen-vl_sp_qlora_only_image \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --seed 2024 \
    --save_steps 100 \
    --save_total_limit 50 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 2048 \
    --max_doc_token_length 1024 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    # --deepspeed ${DS_CONFIG_PATH}
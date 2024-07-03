# Qwen-VL zero-shot推理,带图文

PROJECT_DIR="/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
DATA_DIR="/home/klwang/data/spdocvqa-dataset"
PRETRAIN_MODEL_DIR="/home/klwang/pretrain-model"
cd $PROJECT_DIR

RESULT_DIR="$PROJECT_DIR/result/qwen-vl_only-inference"
if [ ! -d "$RESULT_DIR" ]; then
    mkdir $RESULT_DIR
fi
    # --few-shot \
    # --few_shot_example_json_path "$PROJECT_DIR/few_shot_examples/sp_few_shot_example.json" \

python src/main_qwen-vl_inference2.py \
    --model_name_or_path "${PRETRAIN_MODEL_DIR}/Qwen-VL-Chat-Int4" \
    --eval_json_data_path "${DATA_DIR}/val_v1.0_withQT.json" \
    --data_image_dir "${DATA_DIR}/images" \
    --data_ocr_dir "${DATA_DIR}/ocr" \
    --batch_size 1 \
    --max_new_tokens 128 \
    --max_doc_token_cnt 1024 \
    --experiment_name "qwen-vl_no-sft_zero-shot-'vl_inference_template'" \
    --layout_type "all-star" \
    --log_path "$RESULT_DIR/qwen-vl_no-sft_zero-shot-'vl_inference_template'.jsonl" \
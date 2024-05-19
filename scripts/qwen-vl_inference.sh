PROJECT_DIR="/root/GuiQuQu-docvqa-vllm-inference"
cd $PROJECT_DIR

RESULT_DIR="$PROJECT_DIR/result"
if [ ! -d "$RESULT_DIR" ]; then
    mkdir $RESULT_DIR
fi

python src/main_qwen-vl_inference.py \
    --model_name_or_path "/root/pretrain-model/Qwen-VL-Chat-Int4" \
    --eval_json_data_path "/root/autodl-tmp/spdocvqa-dataset/val_v1.0_withQT.json" \
    --data_image_dir "/root/autodl-tmp/spdocvqa-dataset/images" \
    --data_ocr_dir "/root/autodl-tmp/spdocvqa-dataset/ocr" \
    --batch_size 1 \
    --few-shot \
    --few_shot_example_json_path "$PROJECT_DIR/few_shot_examples/sp_few_shot_example.json" \
    --max_new_tokens 128 \
    --max_doc_token_cnt 1024 \
    --add_image \
    --layout_type "all-star" \
    --log_path "$RESULT_DIR/qwen-vl_no-sft_with-ocr_all-star_template-with-img_with-image_few-shot.jsonl" \
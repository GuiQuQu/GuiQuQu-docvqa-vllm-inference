PROJECT_DIR="/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
ADAPTER_DIR="$PROJECT_DIR/output_qwen-vl_sp_qlora_only_image"
cd $PROJECT_DIR

start=100
step=100
end=4900

for (( i=$start; i<=$end; i+=$step ))
do
echo -e "Inference 'checkpoint-$i'"
python src/main_qwen-vl_inference.py \
    --adapter_name_or_path ${ADAPTER_DIR}/checkpoint-${i} \
    --add_image \
    --log_dir result/qwen-vl_only-image_qlora \
    --experiment_name qwen-vl-int4_sft_only_image \
    --log_path result/qwen-vl_only-image_qlora_old/qwen-vl-int4_sft_only-image_checkpoint-${i}.jsonl
done

echo -e "Inference 'final'"
state="final"
python src/main_qwen-vl_inference.py \
    --adapter_name_or_path "${ADAPTER_DIR}" \
    --add_image \
    --log_dir result/qwen-vl_only-image_qlora \
    --experiment_name qwen-vl-int4_sft_only_image \
    --log_path result/qwen-vl_only-image_qlora_old/qwen-vl-int4_sft_only-image_checkpoint-${state}.jsonl

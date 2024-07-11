PROJECT_DIR="/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
ADAPTER_DIR="$PROJECT_DIR/output_qwen-vl_qlora_vl_inference_template"
cd $PROJECT_DIR

start=100
step=100
end=4900

for (( i=$start; i<=$end; i+=$step ))
do
echo -e "Inference 'checkpoint-$i'"
python src/main_qwen-vl_inference2.py \
    --adapter_name_or_path ${ADAPTER_DIR}/checkpoint-${i} \
    --log_dir result/qwen-vl_qlora_vl_inference_template \
    --experiment_name qlora_vl_inference_template_checkpoint-${i} \
    --log_path result/qwen-vl_qlora_vl_inference_template_old/qlora_vl_inference_template_checkpoint-${i}.jsonl
done

echo -e "Inference 'final'"

state="final"
python src/main_qwen-vl_inference2.py \
    --adapter_name_or_path ${ADAPTER_DIR} \
    --log_dir result/qwen-vl_qlora_vl_inference_template \
    --experiment_name qlora_vl_inference_template_${state} \
    --log_path result/qwen-vl_qlora_vl_inference_template_old/qlora_vl_inference_template_${state}.jsonl

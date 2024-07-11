# 从huggingface上下载模型(利用huggingface-cli)工具
export HF_ENDPOINT="https://hf-mirror.com"
REPO_ID="Qwen/Qwen-VL-Chat-Int4"
LOCAL_NAME="Qwen-VL-Chat-Int4"
MODEL_DIR="$HOME/pretrain-model"
huggingface-cli download \
    $REPO_ID \
    --local-dir $MODEL_DIR/$LOCAL_NAME \

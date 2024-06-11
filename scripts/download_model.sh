# 从huggingface上下载模型(利用huggingface-cli)工具
export HF_ENDPOINT="https://hf-mirror.com"
REPO_ID="internlm/internlm-xcomposer2-vl-7b-4bit"
LOCAL_NAME="internlm-xcomposer2-vl-7b-4bit"
MODEL_DIR="$HOME/pretrain-model"
huggingface-cli download \
    $REPO_ID \
    --local-dir $MODEL_DIR/$LOCAL_NAME \

# 从huggingface上下载模型(利用huggingface-cli)工具
export HF_ENDPOINT="https://hf-mirror.com"
REPO_ID="internlm/internlm-xcomposer2-4khd-7b"
LOCAL_NAME="internlm-xcomposer2-4khd-7b"
MODEL_DIR="$HOME/pretrain-model"
huggingface-cli download \
    $REPO_ID \
    --local-dir $MODEL_DIR/$LOCAL_NAME \

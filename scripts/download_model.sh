# 从huggingface上下载模型(利用huggingface-cli)工具
export HF_ENDPOINT="https://hf-mirror.com"
REPO_ID="Qwen/Qwen1.5-7B-Chat"
LOCAL_NAME="Qwen1.5-7B-Chat"
huggingface-cli download \
    --resume-download \
    $REPO_ID \
    --local-dir $HOME/pretrain-model/$LOCAL_NAME \
    --local-dir-use-symlinks False

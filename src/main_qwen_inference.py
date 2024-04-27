from transformers import AutoModelForCausalLM,AutoTokenizer

device = "cuda:0"

model_path = "/home/klwang/pretrain-model/Qwen1.5-7B-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map= "auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Give me a short introduction to large language model."
message = [
    {"role":"system", "content": "You are a helpful assistant."},
    {"role":"user","content":prompt}
]
text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
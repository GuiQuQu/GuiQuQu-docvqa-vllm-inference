"""
    一张4090根本玩不起这个模型,
    把hd_num开高才能让模型知道图像中的细节内容,否则模型只能给图像的大致描述
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
def main():
    ckpt_path = "/home/klwang/pretrain-model/internlm-xcomposer2-4khd-7b"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model.requires_grad_(False)
    model = model.eval()
    ###############
    # First Round
    ###############
    query = '<ImageHere>\nwhat is the detialed text content of the seventh part.'
    image = '/home/klwang/code/GuiQuQu-docvqa-vllm-inference/tmp-example.webp'
    with torch.cuda.amp.autocast():
        response, his = model.chat(tokenizer, query=query, image=image, hd_num=20, history=[], do_sample=True)
        print(json.dumps({"response": response},ensure_ascii=False,indent=4))

    ###############
    # Second Round
    ###############
    # query = 'what is the detailed explanation of the third part.'
    # with torch.cuda.amp.autocast():
    #     response, _ = model.chat(tokenizer, query=query, image=image, hd_num=10, history=his, do_sample=False, num_beams=3)
    #     print(json.dumps({"response": response},ensure_ascii=False,indent=4))
if __name__ == "__main__":
    main()
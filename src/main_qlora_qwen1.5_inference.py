import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
from functools import partial
import json
import pathlib
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader

import handle_ocr
import template

question_template = template.star_question_templatev3

few_shot_template = question_template + "{answer}\n"

def open_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_prompt(prompt: str, tokenizer: PreTrainedTokenizer):
    """
    给定prompt
    为其添加适用与qwen模型的system_message,并修改其为qwen模型可以识别的输入形式
    """
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    return prompt

def truncate_layout(layout:str, 
                    tokenizer:PreTrainedTokenizer = None, 
                    max_token_length:int = 1024):
    if tokenizer == None:
        return layout
    lines = layout.split("\n")
    lines_input_ids = [tokenizer([l], return_tensors="pt").input_ids for l in lines]
    reserve_lines = []
    ids_cnt = 0
    for i, input_ids in enumerate(lines_input_ids):
        if ids_cnt + input_ids.size(-1) < max_token_length:
            ids_cnt += input_ids.size(-1)
            reserve_lines.append(lines[i])
        else: break
    return "\n".join(reserve_lines)

class EvalSPDocVQADataset(Dataset):
    def __init__(
        self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        few_shot_examples: list,
        layout_func,
        tokenizer: PreTrainedTokenizer = None,
        max_doc_token_cnt:int=2048,
    ) -> None:
        super().__init__()

        def load_data(json_path: str):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["data"]

        self.data = load_data(json_data_path)
        self.image_dir = pathlib.Path(image_dir)
        self.ocr_dir = pathlib.Path(ocr_dir)
        self.layout_func = layout_func
        self.few_shot_examples = few_shot_examples
        self.tokenizer = tokenizer
        self.max_doc_token_cnt = max_doc_token_cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        question = item["question"]
        doc_id = item["image"].split("/")[-1].split(".")[0]
        image_path = self.image_dir / f"{doc_id}.png"
        ocr_path = self.ocr_dir / f"{doc_id}.json"
        layout = self.layout_func(json_path = ocr_path)
        prompt = ""
        # add few shot exmaple
        for e in self.few_shot_examples:
            text = few_shot_template.format(
                layout=truncate_layout(e["layout"], self.tokenizer, self.max_doc_token_cnt), 
                question=e["question"], 
                answer=e["answer"]
            )
            prompt += text
        # add question
        prompt += question_template.format(
            layout=truncate_layout(layout,self.tokenizer, self.max_doc_token_cnt), 
            question=question)
        ret = dict(
            prompt=prompt,
            question=question,
            image_path=str(image_path),
            ocr_path=str(ocr_path),
            layout=layout,
        )
        if "answers" in item.keys():
            ret.update({"answers": item["answers"]})
        # with open("prompt.txt","w",encoding="utf-8") as f:
        #     f.write(prompt)  
        return ret

def lora_model_inference(
        lora_model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        args,
):
    prompts = [get_prompt(p, tokenizer) for p in prompts]
    device = next(lora_model.parameters()).device
    model_inputs = tokenizer(prompts, return_tensors="pt",padding=True, truncation=True, max_length=args.max_length).to(device)
    generated_ids =  lora_model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

def get_layout_func(type:str):
    if type == "all-star":
        return partial(handle_ocr.sp_get_layout_by_json_path, placeholder="*")
    elif type == "lines":
        return handle_ocr.sp_get_lines_layout_by_json_path
    elif type == "words":
        return handle_ocr.sp_get_baseline_layout_by_json_path
    else:
        raise ValueError("Not support layout pattern")

def load_peft_model(args):
    peft_config = PeftConfig.from_pretrained(args.adapter_name_or_path,inference_mode=True)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,torch_dtype="auto",device_map="auto")
    lora_model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
    lora_model.eval()
    print("Start Merge Adapter...")
    lora_model.base_model.merge_adapter()
    print(f"{args.adapter_name_or_path} loaded..., dtype is {next(lora_model.parameters()).dtype}")
    for n, p in model.named_parameters():
        p.requires_grad = False
    return lora_model

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_name_or_path, padding_side='left')
    few_shot_examples = []
    layout_func = get_layout_func(args.layout_type)

    if args.few_shot:
        few_shot_examples = open_json(args.few_shot_example_json_path)
        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = layout_func(
                json_path = os.path.join(args.data_ocr_dir, e["layout"])
            )
    
    # # debug few_shot example
    # few_shot_examples_info = open_json(args.few_shot_example_json_path)
    # for i in range(len(few_shot_examples_info)):
    #     e = few_shot_examples_info[i]
    #     few_shot_examples_info[i]["image"] = os.path.join(args.data_image_dir, e["layout"].split(".")[0] + ".png")
    #     few_shot_examples_info[i]["layout"] = os.path.join(args.data_ocr_dir, e["layout"])
    #     layout = handle_ocr.sp_get_layout_by_json_path(few_shot_examples_info[i]["layout"])
    #     few_shot_examples_info[i]["layout_length"] = len(layout)
    #     few_shot_examples_info[i]["input_id_length"] = tokenizer([layout], return_tensors="pt").input_ids.size()[-1]
    #     truncated_layout = truncate_layout(layout,tokenizer,2048)
    #     few_shot_examples_info[i]["truncated_layout_length"] = len(truncated_layout)
    #     few_shot_examples_info[i]["truncated_layout_input_id_length"] = tokenizer([truncated_layout],return_tensors="pt").input_ids.size()[-1]
    
    eval_dataset = EvalSPDocVQADataset(
        json_data_path=args.eval_json_data_path,
        image_dir=args.data_image_dir,
        ocr_dir=args.data_ocr_dir,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        layout_func=layout_func,
        max_doc_token_cnt=1024
    )

    def collate_fn(batch):
        ret = dict()
        for key in batch[0].keys():
            ret[key] = [d[key] for d in batch]
        return ret

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

    lora_model = load_peft_model(args)
    # print(f"lora_model.base_model.use_cache = {lora_model.base_model.config.use_cache}")
    lora_model.base_model.use_cache = True
    
    # with open(args.log_path,"a",encoding="utf-8") as f:
    #     for i, item in enumerate(eval_dataset):
    #         prompt = item["prompt"]
    #         answers = item["answers"]
    #         input_ids = tokenizer([prompt], return_tensors="pt").input_ids
    #         # tqdm.write(json.dumps(few_shot_examples_info,ensure_ascii=False,indent=2))
    #         response = lora_model_inference(lora_model, tokenizer, [prompt],args)[0]
    #         log = dict(p=f"[{i+1}|{len(eval_dataset)}]",
    #                     prompt_len=len(prompt),
    #                     input_ids_length=input_ids.size(-1),
    #                     image_path=item["image_path"],
    #                     ocr_path=item["ocr_path"],
    #                     question=item["question"],
    #                     response=response,
    #                     answers=answers)
    #         tqdm.write(json.dumps(log,ensure_ascii=False))
    #         f.write(json.dumps(log,ensure_ascii=False) + "\n")

    with open(args.log_path,"a",encoding="utf-8") as f:
        for i, batch in enumerate(tqdm(dataloader)):
            prompt:List[str] = batch["prompt"]
            answers:List[List[str]] = batch["answers"]
            input_ids = tokenizer(prompt, return_tensors="pt",padding=True,truncation=True,max_length=args.max_length).input_ids # [bs,seq_len]
            # tqdm.write(json.dumps(few_shot_examples_info,ensure_ascii=False,indent=2))
            responses = lora_model_inference(lora_model, tokenizer, prompt,args)
            for j, t in enumerate(zip(responses,answers)):
                resp,anss = t
                log = dict(p=f"[{i*args.batch_size+j+1}|{len(eval_dataset)}]",
                            prompt_len=len(prompt[j]),
                            input_ids_length=input_ids.size(-1),
                            image_path=batch["image_path"][0],
                            ocr_path=batch["ocr_path"][0],
                            question=batch["question"][0],
                            response=resp,
                            answers=anss)
                tqdm.write(json.dumps(log,ensure_ascii=False))
                f.write(json.dumps(log,ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default="/home/klwang/code/GuiQuQu-docvqa-vllm-inference/output_qwen1.5_sp_assistant_label_qlora/checkpoint-210",
    )
    parser.add_argument(
        "--eval_json_data_path",
        type=str,
        default="/home/klwang/data/SPDocVQA/val_v1.0_withQT.json",
    )
    
    parser.add_argument(
        "--data_image_dir", type=str, default="/home/klwang/data/SPDocVQA/images"
    )
    parser.add_argument(
        "--data_ocr_dir", type=str, default="/home/klwang/data/SPDocVQA/ocr"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--seed",type=int,default=2024
    )
    parser.add_argument(
        "--few-shot", action="store_true",default=False,
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128
    )
    parser.add_argument(
        "--max_length", type=int, default=1408
    )
    parser.add_argument(
        "--few_shot_example_json_path",
        type=str,
        default="/home/klwang/code/GuiQuQu-docvqa-vllm-inference/few_shot_examples/sp_few_shot_example.json",
    )
    parser.add_argument(
        "--layout_type", type=str, default="all-star" ,choices=["all-star","lines","words"]
    )
    parser.add_argument("--log_path",type=str,default="../result/qwen1.5-7b-qlora_checkpoint-210_with-ocr_no-few-shot_all-stars_template-v3.jsonl")
    args = parser.parse_args()
    main(args)

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List,Union
from functools import partial
import json
import pathlib
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader

import handle_ocr
import template

question_template = template.star_question_template_with_img

few_shot_template = question_template + "{answer}\n"

def open_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

#################################################### qwen1.5 模型推理 #########################################################################
# 和 qwen-vl格式不对应,仅仅适用于被加入huggingface的transformers库的模型代码
# 例如 qwen1.5
def get_prompt_for_std_llm(prompt: str, tokenizer: PreTrainedTokenizer):
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


def model_inference(
        model: Union[PeftModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        args,
):
    """
        仅仅适用于符合huggingface规范的模型推理(一般来说要求huggingface的transformers库加入了对应模型)
    """
    prompts = [get_prompt_for_std_llm(p, tokenizer) for p in prompts]
    device = next(model.parameters()).device
    model_inputs = tokenizer(prompts, return_tensors="pt",padding=True, truncation=True, max_length=args.max_length).to(device)
    generated_ids =  model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

#################################################### qwen-vl 模型推理 #########################################################################

def get_input_ids_for_qwen_vl(prompt: str,tokenizer):
    from Qwen_VL.qwen_generation_utils import make_context
    raw_text, content_tokens  = make_context(tokenizer,prompt,history=None, system="You are a helpful assistant.",chat_format="chatml")
    return raw_text, content_tokens

def qwen_vl_inference(model,tokenizer,prompts:List[str],args) -> List[str]:
    responses = []
    for prompt in prompts:
        resp = model.chat(tokenizer, 
                   prompt, 
                   history=None, 
                   append_history=False,
                   max_length=args.max_length,)
        responses.append(resp)
    
    return responses


######################################################### 数据集 #################################################################################

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
        few_shot_template: str = few_shot_template,
        question_template: str = question_template,
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
        self.few_shot_template = few_shot_template
        self.question_template = question_template
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
            text = self.few_shot_template.format(
                layout=truncate_layout(e["layout"], self.tokenizer, self.max_doc_token_cnt), 
                question=e["question"], 
                answer=e["answer"]
            )
            prompt += text
        # add question
        prompt += self.question_template.format(
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
        return ret

class EvalSPDocVQADatasetWithImg(Dataset):
    def __init__(
        self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        few_shot_examples: list,
        layout_func,
        few_shot_template: str = few_shot_template,
        question_template: str = question_template,
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
        self.few_shot_template = few_shot_template
        self.question_template = question_template

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
            text = self.few_shot_template.format(
                image_path=image_path,
                layout=truncate_layout(e["layout"], self.tokenizer, self.max_doc_token_cnt), 
                question=e["question"], 
                answer=e["answer"]
            )
            prompt += text
        # add question
        prompt += self.question_template.format(
            image_path=image_path,
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
        return ret

def get_layout_func(type:str):
    if type == "all-star":
        return partial(handle_ocr.sp_get_layout_by_json_path, placeholder="*")
    elif type == "lines":
        return handle_ocr.sp_get_lines_layout_by_json_path
    elif type == "words":
        return handle_ocr.sp_get_baseline_layout_by_json_path
    else:
        raise ValueError("Not support layout pattern")

############################################### 加载模型 ####################################################################

def load_peft_model(adapter_name_or_path,trust_remote_code=False):
    peft_config = PeftConfig.from_pretrained(adapter_name_or_path,inference_mode=True)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,torch_dtype="auto",device_map="auto",trust_remote_code=trust_remote_code)
    lora_model = PeftModel.from_pretrained(model, adapter_name_or_path,trust_remote_code=trust_remote_code)
    lora_model.eval()
    print("Start Merge Adapter...")
    lora_model.base_model.merge_adapter()
    print(f"{adapter_name_or_path} loaded..., dtype is {next(lora_model.parameters()).dtype}")
    for _, p in model.named_parameters():
        p.requires_grad = False
    lora_model.base_model.use_cache = True
    return lora_model

def load_model(model_name_or_path,trust_remote_code=False):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                 device_map="auto", 
                                                 trust_remote_code=trust_remote_code, 
                                                 bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model.use_cache = True
    print(f"{model_name_or_path} loaded ..., dtype is {next(model.parameters()).dtype}")
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

############################################### 主函数 ####################################################################
def main(args):
    # check args
    if not args.model_name_or_path and not args.adapter_name_or_path:
        raise ValueError("Please provide 'model_name_or_path' or 'adapter_name_or_path' for model load")
    if args.model_name_or_path and args.adapter_name_or_path:
        raise ValueError("Only one of model_name_or_path and adapter_name_or_path should be provided")
    
    model_name_or_path:str = args.model_name_or_path if args.model_name_or_path else args.adapter_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
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

    if not args.add_image:
        question_template = template.star_question_templatev3
        eval_dataset = EvalSPDocVQADataset(
            json_data_path=args.eval_json_data_path,
            image_dir=args.data_image_dir,
            ocr_dir=args.data_ocr_dir,
            few_shot_examples=few_shot_examples,
            tokenizer=tokenizer,
            layout_func=layout_func,
            question_template= question_template,
            few_shot_template=question_template+"{answer}\n",
            max_doc_token_cnt=args.max_doc_token_cnt
        )
    else:
        question_template = template.star_question_template_with_img
        eval_dataset = EvalSPDocVQADatasetWithImg(
            json_data_path=args.eval_json_data_path,
            image_dir=args.data_image_dir,
            ocr_dir=args.data_ocr_dir,
            few_shot_examples=few_shot_examples,
            tokenizer=tokenizer,
            layout_func=layout_func,
            few_shot_template=question_template+"{answer}\n",
            question_template=question_template,
            max_doc_token_cnt=args.max_doc_token_cnt
        )
    
    def collate_fn(batch):
        ret = dict()
        for key in batch[0].keys():
            ret[key] = [d[key] for d in batch]
        return ret

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

    # load model
    trust_remote_code = "qwen-vl" in model_name_or_path.lower()
    if args.model_name_or_path:
        model = load_model(model_name_or_path,trust_remote_code=trust_remote_code)
    else:
        model = load_peft_model(model_name_or_path,trust_remote_code=trust_remote_code)
    
    if "qwen1.5" in model_name_or_path:
        with open(args.log_path,"a",encoding="utf-8") as f:
            for i, batch in enumerate(tqdm(dataloader)):
                prompt:List[str] = batch["prompt"]
                answers:List[List[str]] = batch["answers"]
                input_ids = tokenizer(prompt, return_tensors="pt",padding=True,truncation=True,max_length=args.max_length).input_ids # [bs,seq_len]
                # tqdm.write(json.dumps(few_shot_examples_info,ensure_ascii=False,indent=2))
                responses = model_inference(model, tokenizer, prompt,args)
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
    
    elif "qwen-vl" in model_name_or_path:
        with open(args.log_path, "a", encoding="utf-8") as f:
            for i, batch in enumerate(tqdm(dataloader)):
                prompt:List[str] = batch["prompt"]
                answers:List[List[str]] = batch["answers"]
                _ ,input_ids = get_input_ids_for_qwen_vl(prompt,tokenizer)
                # 目前的问题,因为qwen-vl没有提供长度截断的方法,因此目前是一条一条推理且不进行长度截断的
                # 目前仅仅依靠truncate_layout()来进行过长截断
                responses = qwen_vl_inference(model, tokenizer, prompt,args)
                for j,t in enumerate(zip(responses,answers)):
                    resp, anss = t
                    _, content_tokens = get_input_ids_for_qwen_vl(prompt[j],tokenizer)
                    log = dict(p=f"[{i*args.batch_size+j+1}|{len(eval_dataset)}]",
                                prompt_len=len(prompt[j]),
                                input_ids_length=len(content_tokens),
                                image_path=batch["image_path"][0],
                                ocr_path=batch["ocr_path"][0],
                                question=batch["question"][0],
                                response=resp,
                                answers=anss)
                    log_str = json.dumps(log, ensure_ascii=False)
                    tqdm.write(log_str)
                    f.write(log_str + "\n")
    else:
        raise ValueError("Not support model")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
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
    parser.add_argument("--max_doc_token_cnt",type=int,default=1024)
    parser.add_argument(
        "--add_image", action="store_true",default=False,
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

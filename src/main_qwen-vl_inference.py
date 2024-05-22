import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List,Union
import json
import pathlib
import time
import argparse


from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig
import peft
from peft import PeftModel, PeftConfig
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset,DataLoader

import template
import utils

question_template = template.star_question_template_with_img

few_shot_template = question_template + "{answer}\n"

def get_input_ids_for_qwen_vl(prompt: str,tokenizer):
    from Qwen_VL.qwen_generation_utils import make_context
    raw_text, content_tokens  = make_context(tokenizer,prompt,history=None, system="You are a helpful assistant.",chat_format="chatml")
    return raw_text, content_tokens

def qwen_vl_inference(model,tokenizer, prompt:Union[List[str],str], args) -> List[str]:
    """
        仅支持调用模型chat方法一条一条推理
    """
    responses = []
    if isinstance(prompt, str):
        prompt = [prompt]
        return_unwarped_res = True
    for p in prompt:
        resp = model.chat(tokenizer, 
                   p,
                   history=None, 
                   append_history=False,
                   max_new_tokens=args.max_new_tokens)
        responses.append(resp)
    return responses[0] if return_unwarped_res else responses

class EvalSPDocVQADatasetWithImg(Dataset):
    def __init__(
        self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        few_shot_examples: list,
        layout_func,
        few_shot_template: str,
        question_template: str,
        add_image:bool=False,
        tokenizer: PreTrainedTokenizer = None,
        max_doc_token_cnt:int=2048,
    ) -> None:
        super().__init__()

        self.data = utils.load_data(json_data_path)
        self.image_dir = pathlib.Path(image_dir)
        self.ocr_dir = pathlib.Path(ocr_dir)
        self.layout_func = layout_func
        self.few_shot_examples = few_shot_examples
        self.tokenizer = tokenizer
        self.max_doc_token_cnt = max_doc_token_cnt
        self.few_shot_template = few_shot_template
        self.question_template = question_template
        self.add_image = add_image
        
        try:
            self.question_template.format(image_path="test",layout="test",question="test")
        except Exception as e:
            raise ValueError("question_template format error")
        
    def __len__(self):
        return len(self.data)
 
    def _format_few_shot_prompt(self, layout, question, answer,image_path:str =None) -> str:
            if self.add_image:
                assert image_path is not None
                text = self.few_shot_template.format(
                    image_path= f"<img>{image_path}</img>",
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question, 
                    answer=answer
                )
            else:
                text = self.few_shot_template.format(
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), question=question,answer=answer)
            return text

    def _format_prompt(self, layout, question,image_path:str=None) -> str:
            if self.add_image:
                assert image_path is not None
                text = self.question_template.format(
                    image_path= f"<img>{image_path}</img>",
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question)
            else:
                text = self.question_template.format(
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question)
            return text

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
            prompt += self._format_few_shot_prompt(e["layout"], e["question"], e["answer"],image_path)
        # add question
        prompt = self._format_prompt(layout,question,image_path)
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

def load_qwen_vl_lora(adapter_name_or_path):
    """
        加载微调之后的qwen-vl模型
    """
    peft_config = PeftConfig.from_pretrained(adapter_name_or_path,inference_mode=True)
    model:nn.Module = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,torch_dtype="auto",device_map="auto",trust_remote_code=True)
    lora_model:peft.peft_model.PeftModelForCausalLM = PeftModel.from_pretrained(model, adapter_name_or_path)
    lora_model.eval()
    print("Start  merge_and_unloadMerge Adapter...")
    # print(f"lora_model class type is {type(lora_model)}")
    # print(f"lora_model.base_model type is {type(lora_model.base_model)}") # base_model is LoraModel
    # gptq 量化的模型不能merge ...
    # lora_model.base_model.merge_and_unload()
    print(f"{adapter_name_or_path} loaded, dtype is {next(lora_model.parameters()).dtype}")
    for _, p in model.named_parameters():
        p.requires_grad = False
    lora_model.base_model.use_cache = True
    return lora_model

def load_qwen_vl_model(model_name_or_path):
    """
        可以加载qwen-vl bf16 和 qwen-vl-chat-int4
    """
    model:nn.Module = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",trust_remote_code=True)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.use_cache = True
    print(f"{model_name_or_path} loaded ..., dtype is {next(model.parameters()).dtype}")
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model


def main(args):
    # check args
    if not args.model_name_or_path and not args.adapter_name_or_path:
        raise ValueError("Please provide 'model_name_or_path' or 'adapter_name_or_path' for model load")
    if args.model_name_or_path and args.adapter_name_or_path:
        raise ValueError("Only one of model_name_or_path and adapter_name_or_path should be provided")
    
    model_name_or_path:str = args.model_name_or_path if args.model_name_or_path else args.adapter_name_or_path
    
    utils.seed_everything(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left',trust_remote_code=True)
    few_shot_examples = []
    layout_func = utils.get_layout_func(args.layout_type)

    if args.few_shot:
        few_shot_examples = utils.open_json(args.few_shot_example_json_path)
        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = layout_func(
                json_path = os.path.join(args.data_ocr_dir, e["layout"])
            )

    if args.add_image:
        question_template = template.vl_ocr_question_template
    else:
        question_template = template.star_question_templatev4
    eval_dataset = EvalSPDocVQADatasetWithImg(
        json_data_path=args.eval_json_data_path,
        image_dir=args.data_image_dir,
        ocr_dir=args.data_ocr_dir,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        layout_func=layout_func,
        few_shot_template=question_template+"{answer}\n",
        add_image=args.add_image,
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
    if args.model_name_or_path :
        model = load_qwen_vl_model(model_name_or_path)
    elif args.adapter_name_or_path: 
        model = load_qwen_vl_lora(model_name_or_path)
    else:
        raise ValueError("Please provide 'model_name_or_path' or 'adapter_name_or_path' for model load")
    
    with open(args.log_path, "a", encoding="utf-8") as f:
        for i, batch in enumerate(tqdm(dataloader)):
            prompt:List[str] = batch["prompt"]
            answers:List[List[str]] = batch["answers"]
            # 目前的问题,因为qwen-vl没有提供长度截断的方法,因此目前是一条一条推理且不进行长度截断的
            # 目前仅仅依靠truncate_layout()来进行过长截断
            for j,t in enumerate(zip(prompt,answers)):
                p, anss = t
                _, content_tokens = get_input_ids_for_qwen_vl(p,tokenizer)
                start_time = time.time()
                resp, _ = qwen_vl_inference(model, tokenizer, p, args)
                execution_time = time.time() - start_time
                log = dict(p=f"[{i*args.batch_size+j+1}|{len(eval_dataset)}]",
                            time=f"{execution_time:.2f}s",
                            prompt_len=len(p),
                            token_len=len(content_tokens),
                            image=batch["image_path"][j],
                            ocr=batch["ocr_path"][j],
                            question=batch["question"][j],
                            response=resp,
                            answers=anss)
                log_str = json.dumps(log, ensure_ascii=False)
                tqdm.write(log_str)
                f.write(log_str + "\n")
    
if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/spdocvqa-dataset"
    project_dir = "/root/GuiQuQu-docvqa-vllm-inference"
    pretrain_model_dir = "/root/pretrain-model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,default=None)
    parser.add_argument("--adapter_name_or_path",type=str,default="/root/autodl-tmp/output_qwen-vl_sp_qlora/checkpoint-100")
    parser.add_argument("--eval_json_data_path",type=str,default= os.path.join(data_dir,"val_v1.0_withQT.json"))
    parser.add_argument("--data_image_dir", type=str, default=os.path.join(data_dir,"images"))
    parser.add_argument("--data_ocr_dir", type=str, default=os.path.join(data_dir,"ocr"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--few-shot", action="store_true",default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    # parser.add_argument("--max_length", type=int, default=1408)
    parser.add_argument("--max_doc_token_cnt",type=int,default=1024)
    parser.add_argument("--add_image", action="store_true",default=True)
    parser.add_argument("--few_shot_example_json_path",type=str,default=os.path.join(project_dir,"few_shot_examples","sp_few_shot_example.json"))
    parser.add_argument("--layout_type", type=str, default="all-star" ,choices=["all-star","lines","words"])
    parser.add_argument("--log_path",type=str,default=os.path.join(project_dir,"result/qwen-vl-int4_sft-vl.jsonl"))
    args = parser.parse_args()
    main(args)

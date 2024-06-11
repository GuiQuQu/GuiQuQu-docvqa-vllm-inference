import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List,Union,Dict
import json
import pathlib
import time
import argparse

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer,PreTrainedModel
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig
import peft
from peft import PeftModel, PeftConfig
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset,DataLoader

from Qwen_VL.tokenization_qwen import QWenTokenizer,ENDOFTEXT

import template
import utils
import metrics

question_template = template.star_question_template_with_img

few_shot_template = question_template + "{answer}\n"

def get_prompt(tokenizer:QWenTokenizer, 
                        messages, 
                        default_system_message:str = "You are a helpful assistant.",
                        tokenize:bool = False):
    """
    messages: List
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response},
    ]
    <|im_start|>system\nsystem_message<|im_end|>\n
    <|im_start|>user\nuser_message<|im_end|>\n
    <|im_start|>assistant\nresp<|im_end|>\n
    <|im_start|>user\nuser_message<|im_end|>\n
    <|im_start|>assistant\n
    """
    im_start,  = "<|im_start|>", 
    im_end,  = "<|im_end|>",


    def _msg_prompt(role, content):
        return f"{im_start}{role}\n{content}{im_end}\n"
    # 编码prompt
    assert len(messages) >= 1
    prompt = ""
    
    # add system message
    if messages[0]["role"] == "system":
        prompt += _msg_prompt("system",msg["content"])
        messages = messages[1:]
    else:
        prompt += _msg_prompt("system",default_system_message)

    # role loop
    role_loop =["user", "assistant"]
    expect_idx = 0
    for _, msg in enumerate(messages):
        if msg["role"] != role_loop[expect_idx]:
            ValueError(f"expect role is {role_loop[expect_idx]}, but actually is {msg['role']}")
        prompt += _msg_prompt(msg["role"],msg["content"])
        expect_idx = (expect_idx + 1) % 2
    if role_loop[expect_idx] != "assistant":
        ValueError("message last item should is user message")
    # add assistant resp
    prompt += f"{im_start}assistant\n"
    if not tokenize:
        return prompt
    im_start_tokens = tokenizer.encode(im_start)
    im_end_tokens =  tokenizer.encode(im_end)
    nl_tokens = tokenizer.encode("\n")
    allowed_special = set(tokenizer.IMAGE_ST)
    raise NotImplementedError    

def decode_tokens(tokens: List[int],
           tokenizer:QWenTokenizer,
           end_token_ids: List[int],
           prompt_text_len:int,
           prompt_token_len:int,
           stop_words: List[str] = [],
           errors:str = "replace"):
    """
        解码genernate产生的输出
    """
    end_reason = f"Genernate length {len(tokens)}"
    eod_token_idx = prompt_token_len
    response = dict()
    
    for eod_token_idx in range(prompt_token_len, len(tokens)):
        if tokens[eod_token_idx] in end_token_ids:
            end_reason = f"Genernate {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break
    
    trim_decode_text = tokenizer.decode(tokens[:eod_token_idx], errors = errors)[prompt_text_len:]
    # response["raw_generate_w/o_EOD"] = tokenizer.decode(tokens,errors=errors)[prompt_text_len:]
    # response["raw_generate"] = trim_decode_text
    response["end_reason"] = end_reason
    # 删除停止词 
    # for stop_word in stop_words:
    #     trim_decode_text = trim_decode_text.replace(stop_word,"").strip()
    response["response"] = trim_decode_text
    return response

def get_input_ids_for_qwen_vl(prompt: str,tokenizer):
    from Qwen_VL.qwen_generation_utils import make_context
    raw_text, content_tokens  = make_context(tokenizer,prompt,history=None, system="You are a helpful assistant.",chat_format="chatml")
    return raw_text, content_tokens



def qwen_vl_inference(model:PreTrainedModel, 
                      tokenizer:QWenTokenizer, 
                      prompts:List[List[Dict]],args) -> utils.Response | List[utils.Response]:
    """
        模型的tokenizer不支持batch_infer,所以batch_infer目前无法实现
    """
    if not isinstance(prompts,list):
        prompts = [prompts]
        need_warpped = True
    prompts = [get_prompt(tokenizer,p) for p in prompts]
    device = next(model.parameters()).device
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,max_length=args.max_length).to(device)
    # 停止输出词
    stop_words_ids = [[tokenizer.im_end_id],[tokenizer.im_start_id]]
    generate_ids = model.generate(
        model_inputs.input_ids,
        stop_words_ids=stop_words_ids,
        attention_mask = model_inputs.attention_mask,
        generation_config=model.generation_config,
        max_new_tokens=args.max_new_tokens)

    responses = []
    for p,input_ids, output_ids in zip(prompts,model_inputs.input_ids, generate_ids):
        input_ids:List[int] = input_ids.cpu().numpy().tolist()
        output_ids:List[int] = output_ids.cpu().numpy().tolist()
        resp = decode_tokens(output_ids,
                             prompt_text_len=len(p),
                             prompt_token_len=len(input_ids),
                             end_token_ids=[tokenizer.im_start_id,tokenizer.im_end_id],
                             tokenizer=tokenizer)
        
        responses.append(utils.Response(
            text=resp["response"],
            prompt=p,
            end_reason=resp["end_reason"],
            input_ids=input_ids,
            output_ids=output_ids[len(input_ids):],
        ))
    if need_warpped:
        return responses[0]
    else:
        return responses

def qwen_vl_inference2(model,tokenizer, prompt:Union[List[str],str], args) -> List[str]:
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
        layout_dir:str,
        few_shot_examples: list,
        layout_func,
        few_shot_template: str,
        question_template: str,
        add_layout:bool=False,
        add_image:bool=False,
        tokenizer: PreTrainedTokenizer = None,
        max_doc_token_cnt:int=2048,
    ) -> None:
        super().__init__()

        self.data = utils.load_data(json_data_path)
        self.image_dir = pathlib.Path(image_dir)
        self.ocr_dir = pathlib.Path(ocr_dir)
        self.layout_dir = pathlib.Path(layout_dir)
        self.layout_func = layout_func
        self.few_shot_examples = few_shot_examples
        self.tokenizer = tokenizer
        self.max_doc_token_cnt = max_doc_token_cnt
        self.few_shot_template = few_shot_template
        self.question_template = question_template
        self.add_image = add_image
        self.add_layout = add_layout

        
    def __len__(self):
        return len(self.data)
 
    def _format_few_shot_prompt(self, question, answer,layout=None,image_path:str =None) -> str:
            if self.add_image and self.add_layout:
                assert image_path is not None and layout is not None
                text = self.few_shot_template.format(
                    image_path= f"<img>{image_path}</img>",
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question, 
                    answer=answer
                )    
            elif not self.add_image and self.add_layout:
                assert layout is not None
                text = self.few_shot_template.format(
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question, 
                    answer=answer
                )
            elif self.add_image and not self.add_layout:
                assert image_path is not None
                text = self.few_shot_template.format(
                    image_path= f"<img>{image_path}</img>",
                    question=question, 
                    answer=answer
                )
            else:
                raise ValueError("add_image and add_layout can't be False at the same time")
            return text
    
            # if self.add_image:
            #     assert image_path is not None
            #     text = self.few_shot_template.format(
            #         image_path= f"<img>{image_path}</img>",
            #         layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
            #         question=question, 
            #         answer=answer
            #     )
            # else:
            #     text = self.few_shot_template.format(
            #         layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), question=question,answer=answer)
            # return text

    def _format_prompt(self, question, layout=None, image_path:str=None) -> str:
            if self.add_image and self.add_layout:
                assert image_path is not None and layout is not None
                text = self.question_template.format(
                    image_path= f"<img>{image_path}</img>",
                    layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                    question=question)
            elif self.add_image and not self.add_layout:
                assert image_path is not None
                text = self.question_template.format(
                    image_path= f"<img>{image_path}</img>",
                    question=question)
            elif not self.add_image and self.add_layout:
                assert layout is not None
                text = self.question_template.format(
                layout=utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt), 
                question=question)
            else:
                raise ValueError("add_image and add_layout can't be False at the same time")

            return text

    def __getitem__(self, i) -> dict: 
        item = self.data[i]
        qid = item["questionId"]
        question = item["question"]
        doc_id = item["image"].split("/")[-1].split(".")[0]
        image_path = str(self.image_dir / f"{doc_id}.png")
        ocr_path = str(self.ocr_dir / f"{doc_id}.json")
        if self.layout_dir is not None: layout_path = str(self.layout_dir / f"{doc_id}.txt")
        else: layout_path = None
        
        prompt = ""
        # add few shot exmaple
        for e in self.few_shot_examples:
            prompt += self._format_few_shot_prompt(layout = e["layout"], 
                                                   question=e["question"], 
                                                   answer=e["answer"],
                                                   image_path=image_path)
        # add question
        layout = None
        if self.layout_func is not None:layout = self.layout_func(json_path = ocr_path)
        prompt = self._format_prompt(question=question,layout=layout,image_path=image_path)
        prompt = [{
            "role": "user",
            "content": prompt
        }]
        ret = dict(
            qid=qid,
            prompt=prompt,
            question=question,
            layout_path=layout_path,
            image_path=image_path,
            ocr_path=ocr_path,
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
    # print("Start  merge_and_unload Merge Adapter...")
    # print(f"lora_model class type is {type(lora_model)}")
    # print(f"lora_model.base_model type is {type(lora_model.base_model)}") # base_model is LoraModel
    # gptq 量化的模型不能merge ...
    # lora_model.base_model.merge_and_unload()
    print(f"'{adapter_name_or_path}' loaded, dtype is '{next(lora_model.parameters()).dtype}'")
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

def get_question_template(add_image:bool,add_layout:bool):
    if add_image and add_layout:
        return template.vl_ocr_question_template
    elif add_image and not add_layout:
        return template.visual_question_template
    elif not add_image and add_layout:
        return template.star_question_templatev4
    else:
        raise ValueError("add_image and add_layout can't be False at the same time")

def main(args):
    # check args
    if not args.model_name_or_path and not args.adapter_name_or_path:
        raise ValueError("Please provide 'model_name_or_path' or 'adapter_name_or_path' for model load")
    if args.model_name_or_path and args.adapter_name_or_path:
        raise ValueError("Only one of model_name_or_path and adapter_name_or_path should be provided")
    
    model_name_or_path:str = args.model_name_or_path if args.model_name_or_path else args.adapter_name_or_path
    
    utils.seed_everything(args.seed)
    
    tokenizer:QWenTokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left',trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id
    few_shot_examples = []
    # layout_func = utils.get_layout_func(args.layout_type)
    layout_func = None
    if args.add_layout:layout_func = utils.sp_get_layout_func2(args.layout_type)

    if args.few_shot:
        few_shot_examples = utils.open_json(args.few_shot_example_json_path)
        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = layout_func(
                json_path = os.path.join(args.data_ocr_dir, e["layout"])
            )

    question_template = get_question_template(args.add_image,args.add_layout)
    
    eval_dataset = EvalSPDocVQADatasetWithImg(
        json_data_path=args.eval_json_data_path,
        image_dir=args.data_image_dir,
        ocr_dir=args.data_ocr_dir,
        layout_dir=args.layout_dir,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        layout_func=layout_func,
        few_shot_template=question_template+"{answer}\n",
        add_image=args.add_image,
        add_layout=args.add_layout,
        question_template=question_template,
        max_doc_token_cnt=args.max_doc_token_cnt
    )
    
    def collate_fn(batch):
        "List[dict{key,value}] -> dict{key,List[value]}"
        ret = dict()
        for key in batch[0].keys():
            ret[key] = [d[key] for d in batch]
        return ret

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

    # load model
    if args.model_name_or_path:
        model = load_qwen_vl_model(model_name_or_path)
    elif args.adapter_name_or_path: 
        model = load_qwen_vl_lora(model_name_or_path)
    else:
        raise ValueError("Please provide 'model_name_or_path' or 'adapter_name_or_path' for model load")
    experiment_name = f"{args.experiment_name}_{args.adapter_name_or_path.split('/')[-1]}"
    anls = metrics.ANLS(
        result_dir=args.log_dir,
        experiment_name=experiment_name,
        dataset_name="spdocvqa"
    )
    qids,questions, predictions = [], [], []
    gts, image_paths, ocr_paths, layout_paths = [], [], [], []

    with open(args.log_path, "a", encoding="utf-8") as f:
        for i, batch in enumerate(tqdm(dataloader)):
            prompt:List[List[Dict]] = batch["prompt"]
            answers:List[List[str]] = batch["answers"]

            for j,t in enumerate(zip(prompt,answers)):
                p, anss = t
                p = p[-1]['content']
                _, content_tokens = get_input_ids_for_qwen_vl(p,tokenizer)
                start_time = time.time()
                resp = qwen_vl_inference2(model, tokenizer, p, args)
                execution_time = time.time() - start_time

                question = batch["question"][j]
                image_path = batch["image_path"][j]
                ocr_path = batch["ocr_path"][j]
                layout_path = batch["layout_path"][j]

                qids.append(batch["qid"][j])
                questions.append(question)
                predictions.append(resp)
                gts.append(anss)
                image_paths.append(image_path)
                ocr_paths.append(ocr_path)
                layout_paths.append(layout_path)

                # 旧的日志记录
                log = dict(p=f"[{i*args.batch_size+j+1}|{len(eval_dataset)}]",
                            time=f"{execution_time:.2f}s",
                            prompt_len=len(p),
                            token_len=len(content_tokens),
                            image_path=image_path,
                            ocr_path=ocr_path,
                            layout_path=layout_path,
                            question=question,
                            response=resp,
                            answers=anss)
                log_str = json.dumps(log, ensure_ascii=False)
                tqdm.write(log_str)
                f.write(log_str + "\n")

    score = anls.compute_and_save_docvqa(
        qids=qids,
        questions=questions,
        predictions=predictions,
        answers=gts,
        image_paths=image_paths,
        ocr_paths=ocr_paths,    
        layout_paths=layout_paths,
        split="val"
    )
    print(f"{experiment_name} spdocvqa val ANLS score is {score:.4f}")


if __name__ == "__main__":
    data_dir = "/home/klwang/data/spdocvqa-dataset"
    project_dir = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
    pretrain_model_dir = "/home/klwang/pretrain-model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,default=None)
    parser.add_argument("--adapter_name_or_path",type=str,
                        default="/home/klwang/code/GuiQuQu-docvqa-vllm-inference/output_qwen-vl_sp_qlora_only_image/checkpoint-100")
    parser.add_argument("--eval_json_data_path",type=str,default= os.path.join(data_dir,"val_v1.0_withQT.json"))
    parser.add_argument("--data_image_dir", type=str, default=os.path.join(data_dir,"images"))
    parser.add_argument("--add_image", action="store_true",default=True)
    parser.add_argument("--add_layout", action="store_true",default=False)
    parser.add_argument("--layout_type", type=str, default="all-star" ,choices=["all-star","lines","words"])
    parser.add_argument("--data_ocr_dir", type=str, default=os.path.join(data_dir,"ocr"))
    parser.add_argument("--layout_dir",type=str,default=os.path.join(data_dir,"layout"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--few-shot", action="store_true",default=False)
    parser.add_argument("--few_shot_example_json_path",type=str,default=os.path.join(project_dir,"few_shot_examples","sp_few_shot_example.json"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1408)
    parser.add_argument("--max_doc_token_cnt",type=int,default=1024)
    # new anls log
    parser.add_argument("--log_dir", type=str, default=os.path.join(project_dir,"result/qwen-vl_only-inference"))
    parser.add_argument("--experiment_name", type=str, default="qwen-vl-int4_no-sft-few-shot_vl_''")
    # old log
    parser.add_argument("--log_path",type=str,default=os.path.join(project_dir,"result/qwen-vl_only-image_old/qwen-vl-int4_sft_only-image_checkpoint-100.jsonl"))
    args = parser.parse_args()
    main(args)

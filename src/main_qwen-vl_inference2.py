from  typing import List
import argparse
import os

from torch import nn
from transformers import PreTrainedTokenizer,AutoModelForCausalLM,AutoTokenizer
from transformers.generation import GenerationConfig
import peft
from peft import PeftModel, PeftConfig

from Qwen_VL.tokenization_qwen import QWenTokenizer
from docvqa_evalator import DocVQAEvaluator,DocVQAEvaluateDatasetImageAndLayout
import utils
import template

question_template = template.vl_ocr_question_template

def qwen_vl_inference(model, tokenizer, prompt:List[str] | str, max_new_tokens:int, max_length:int) -> str | List[str]:
    responses = []
    if isinstance(prompt, str):
        prompt = [prompt]
        return_unwarped_res = True
    for p in prompt:
        resp, _ = model.chat(tokenizer, 
                   p,
                   system = "You are a helpful assistant.",
                   history=None, 
                   append_history=False,
                   max_new_tokens=max_new_tokens)
        responses.append(resp)
    return responses[0] if return_unwarped_res else responses

def qwen_vl_get_input_ids(prompt:str, tokenizer: PreTrainedTokenizer) -> List[int]:
    from Qwen_VL.qwen_generation_utils import make_context
    _, content_tokens  = make_context(tokenizer,prompt,history=None, system="You are a helpful assistant.",chat_format="chatml")
    return content_tokens

def load_qwenvl_model(model_name_or_path:str):
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

def load_qwenvl_lora(adapter_name_or_path):
    """
        加载微调之后的qwen-vl模型
    """
    peft_config = PeftConfig.from_pretrained(adapter_name_or_path,inference_mode=True)
    model:nn.Module = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,torch_dtype="auto",device_map="auto",trust_remote_code=True)
    lora_model:peft.peft_model.PeftModelForCausalLM = PeftModel.from_pretrained(model, adapter_name_or_path)
    lora_model.eval()
    # gptq 量化的模型不能merge ...
    # lora_model.base_model.merge_and_unload()
    print(f"'{adapter_name_or_path}' loaded, dtype is '{next(lora_model.parameters()).dtype}'")
    for _, p in model.named_parameters():
        p.requires_grad = False
    lora_model.base_model.use_cache = True
    return lora_model

def main(args):
    utils.seed_everything(args.seed)

    
    if args.model_name_or_path is not None and args.adapter_name_or_path is not None:
        raise ValueError("model_name_or_path and adapter_name_or_path can't be set at the same time")
    if args.model_name_or_path is None and args.adapter_name_or_path is None:
        raise ValueError("model_name_or_path and adapter_name_or_path can't be None at the same time")
    model_name_or_path:str = args.model_name_or_path if args.model_name_or_path  \
        else args.adapter_name_or_path
    
    tokenizer:QWenTokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left',trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id
    # create dataset
    few_shot_examples = []
    layout_func = utils.sp_get_layout_func2(args.layout_type)
    if args.few_shot:
        few_shot_examples = utils.open_json(args.few_shot_example_json_path)

    eval_dataset = DocVQAEvaluateDatasetImageAndLayout(
        json_data_path=args.eval_json_data_path,
        image_dir=args.data_image_dir,
        layout_dir=args.layout_dir,
        ocr_dir=args.data_ocr_dir,
        few_shot_examples=few_shot_examples,
        question_template=question_template,
        few_shot_template=question_template+"f{answer}\n",
        tokenizer=tokenizer,
        layout_func=layout_func,
        max_doc_token_cnt=args.max_doc_token_cnt,
    )
    # load model
    if args.adapter_name_or_path is not None:
        model = load_qwenvl_lora(model_name_or_path)
    else:
        model = load_qwenvl_model(model_name_or_path)

    evaluator = DocVQAEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        result_dir=args.log_dir,
        experiment_name=args.experiment_name,
        log_path=args.log_path,
        model_inference=qwen_vl_inference,
        get_input_ids=qwen_vl_get_input_ids,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    evaluator.evaluate()

if __name__ == "__main__":
    data_dir = "/home/klwang/data/spdocvqa-dataset"
    project_dir = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
    pretrain_model_dir = "/home/klwang/pretrain-model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,
                        default="/home/klwang/pretrain-model/Qwen-VL-Chat-Int4")
    parser.add_argument("--adapter_name_or_path",type=str,
                        default=None)
    parser.add_argument("--eval_json_data_path",type=str,default= os.path.join(data_dir,"val_v1.0_withQT.json"))
    parser.add_argument("--data_image_dir", type=str, default=os.path.join(data_dir,"images"))
    parser.add_argument("--layout_type", type=str, default="all-star" ,choices=["all-star","lines","words"])
    parser.add_argument("--data_ocr_dir", type=str, default=os.path.join(data_dir,"ocr"))
    parser.add_argument("--layout_dir",type=str,default=os.path.join(data_dir,"layout"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--few-shot", action="store_true",default=True)
    parser.add_argument("--few_shot_example_json_path",type=str,default=os.path.join(project_dir,"few_shot_examples","sp_few_shot_example.json"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1408)
    parser.add_argument("--max_doc_token_cnt",type=int,default=1024)
    # new anls log
    parser.add_argument("--log_dir", type=str, default=os.path.join(project_dir,"result/qwen-vl_only-inference"))
    parser.add_argument("--experiment_name", type=str, default="qwen-vl-int4_no-sft_vl_'vl_ocr_question_template'")
    # old log
    parser.add_argument("--log_path",type=str,default=os.path.join(project_dir,"result/qwen-vl_only-inference/qwen-vl-int4_no-sft_vl_'vl_ocr_question_template'.jsonl"))
    args = parser.parse_args()
    main(args)
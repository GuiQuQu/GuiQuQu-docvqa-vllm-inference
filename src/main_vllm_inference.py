import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
from functools import partial
import json
import pathlib
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm


from torch.utils.data import Dataset

import handle_ocr
import template
import utils
import metrics

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
        qid = item["questionId"]
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
            qid=qid,
            prompt=prompt,
            question=question,
            image_path=str(image_path),
            ocr_path=str(ocr_path),
            layout_path=None,
            layout=layout,
        )
        if "answers" in item.keys():
            ret.update({"answers": item["answers"]})
        return ret


def vllm_inference(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
) -> List[str]:
    prompts = [get_prompt(p, tokenizer) for p in prompts]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generateed_text = output.outputs[0].text
        ret.append(generateed_text)
        # print(f"Prompt: {prompt}\nGenerated text: {generateed_text!r}")
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

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(model=args.model_name_or_path, dtype="bfloat16", max_model_len=7248,seed=args.seed)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20)
    few_shot_examples = []
    # layout_func = get_layout_func(args.layout_type)
    layout_func = utils.sp_get_layout_func2(args.layout_type)
    if args.few_shot:
        few_shot_examples = open_json(args.few_shot_example_json_path)

        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = layout_func(
                json_path = os.path.join(args.data_ocr_dir, e["layout"])
            )
    
    eval_dataset = EvalSPDocVQADataset(
        json_data_path=args.eval_json_data_path,
        image_dir=args.data_image_dir,
        ocr_dir=args.data_ocr_dir,
        few_shot_examples=few_shot_examples,
        tokenizer=tokenizer,
        layout_func=layout_func,
        max_doc_token_cnt=2048
    )
    anls = metrics.ANLS(
        result_dir=args.result_dir,
        experiment_name=args.experiment_name,
        dataset_name="spdocvqa"
    )
    qids, questions, predictions, image_paths, ocr_paths, layout_paths, answers = [], [], [], [], [], [], []
    with open(args.log_path,"a",encoding="utf-8") as f:
        for i, item in enumerate(eval_dataset):
            prompt = item["prompt"]
            anss = item["answers"]
            qid = item["qid"]
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids
            response = vllm_inference(llm, sampling_params, tokenizer, [prompt])[0]

            qids.append(qid)
            questions.append(item["question"])
            predictions.append(response)
            image_paths.append(item["image_path"])
            ocr_paths.append(item["ocr_path"])
            layout_paths.append(item["layout_path"])
            answers.append(anss)

            log = dict(p=f"[{i+1}|{len(eval_dataset)}]",
                        prompt_len=len(prompt),
                        input_ids_length=input_ids.size(-1),
                        image_path=item["image_path"],
                        ocr_path=item["ocr_path"],
                        question=item["question"],
                        response=response,
                        answers=anss)
            log_str = json.dumps(log,ensure_ascii=False)
            tqdm.write(log_str)
            f.write(log_str + "\n")
    score = anls.compute_and_save_docvqa(
        qids=qids,
        questions=questions,
        predictions=predictions,
        image_paths=image_paths,
        ocr_paths=ocr_paths,
        layout_paths=layout_paths,
        answers=answers,
        split="val"
    )
    print(f"ANLS Score: {score}")

if __name__ == "__main__":
    project_dir = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
    data_dir = "/home/klwang/data/SPDocVQA"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,default="/home/klwang/pretrain-model/Qwen1.5-7B-Chat")
    parser.add_argument("--eval_json_data_path",type=str,default=os.path.join(data_dir,"val_v1.0_withQT.json"))
    parser.add_argument("--data_image_dir", type=str, default=os.path.join(data_dir,"images"))
    parser.add_argument("--data_ocr_dir", type=str, default=os.path.join(data_dir,"ocr"))
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--few-shot", action="store_true",default=True)
    parser.add_argument("--few_shot_example_json_path",type=str,
                        default=os.path.join(project_dir,"few_shot_examples/sp_few_shot_example.json"))
    parser.add_argument("--layout_type", type=str, default="all-space" ,choices=["all-star","lines","words","all-space"])
    parser.add_argument("--result_dir",type=str,default=os.path.join(project_dir,"result/qwen1.5_vllm"))
    parser.add_argument("--experiment_name",type=str,default="qwen1.5-7b_vllm_all-space_'star_question_templatev3'")
    parser.add_argument("--log_path",type=str,
                        default=os.path.join(project_dir,"result/qwen1.5_vllm/qwen1.5-7b_vllm_all-space_'star_question_templatev3'.jsonl"))
    args = parser.parse_args()
    main(args)

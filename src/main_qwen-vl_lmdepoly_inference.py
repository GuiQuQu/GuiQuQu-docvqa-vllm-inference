"""
    shit lmdeploy
"""
from typing import List, Dict, Union, Tuple
import copy
import os
import json
from tqdm import tqdm
import pathlib
from torch.utils.data import DataLoader,Dataset

from transformers import PreTrainedTokenizer,AutoTokenizer

from lmdeploy.model import MODELS, Qwen7BChat
from lmdeploy import pipeline,TurbomindEngineConfig,GenerationConfig,ChatTemplateConfig

from template import lmdeploy_vl_ocr_question_template
import utils

prompt_template =  lmdeploy_vl_ocr_question_template

few_shot_vl_ocr_question_template =prompt_template + "{answer}\n"

@MODELS.register_module(name='qwen-vl-chat-few-shot')
class QwenVLChatTemplate(Qwen7BChat):
    """Qwen vl chat template."""

    def __init__(self,
                 session_len=8192,
                 top_p=0.3,
                 top_k=None,
                 temperature=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
    
    def _concat_image_info(self, prompt):
        """Append image placeholder."""
        if isinstance(prompt, str):
            return prompt
        prompt, nimg = prompt
        res = ''
        for i in range(nimg):
            res += f'Picture {str(i)}:<img>placeholder</img>\n'
        prompt = res + prompt
        return prompt
    
    def init_few_shot_prompt(self,few_shot_json_path):
        few_shot_examples = utils.load_data(few_shot_json_path)
        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = self.layout_func(
                json_path = self.ocr_dir / e["layout"]
            )
        """

        """

    def get_prompt(self, prompt, sequence_start=True):
        """Apply chat template to prompt."""
        prompt = self._concat_image_info(prompt)
        return super().get_prompt(prompt, sequence_start)

    def messages2prompt(self, messages, sequence_start=True):
        """Apply chat template to history."""
        if isinstance(messages, str) or isinstance(messages[0], str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(user=self.user,
                       assistant=self.assistant,
                       system=self.system)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.separator,
                       system=self.eosys)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role'] 
            content = message['content']
            if role == 'user' and not isinstance(content, str):
                content = [content[0]['text'], len(content) - 1]
                content = self._concat_image_info(content)
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret


class EvalSPDocVQADatasetWithImg(Dataset):
    def __init__(
        self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        few_shot_example_json_path:str,
        layout_func,
        question_template: str = prompt_template,
        tokenizer: PreTrainedTokenizer = None,
        max_doc_token_cnt:int=1024,
    ) -> None:
        super().__init__()

        self.data = utils.load_data(json_data_path)
        self.image_dir = pathlib.Path(image_dir)
        self.ocr_dir = pathlib.Path(ocr_dir)
        self.layout_func = layout_func
        self.tokenizer = tokenizer
        self.max_doc_token_cnt = max_doc_token_cnt
        self.question_template = question_template
        # self.few_shot_prompt = []
        self.init_few_shot_examples(utils.open_json(few_shot_example_json_path))
    
    def init_few_shot_examples(self,few_shot_examples:List[Dict]):
        for i in range(len(few_shot_examples)):
            e = few_shot_examples[i]
            few_shot_examples[i]["layout"] = self.layout_func(
                json_path = self.ocr_dir / e["layout"]
            )
        # 处理形成few_shot prompt dict (符合openai url的格式)
        few_shot_prompt = []
        for e in few_shot_examples:
            image_path = str(self.image_dir / e["image"])
            text = self.question_template.format(
                layout= utils.truncate_layout(e["layout"],self.tokenizer, self.max_doc_token_cnt),
                question = e['question']
            )
            messages = [
                {
                    'role': 'user',
                    'content': [{'type':'text','text':text},{'type':'image_url','image_url':{'url':image_path}}]
                },
                {
                    'role': 'assistant',
                    'content': [{'type':'text','text':e['answer']}]
                }
            ]
            few_shot_prompt.extend(messages)

        self.few_shot_prompt = few_shot_prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i) ->dict:
        """
            return return_value["prompt"] 是符合openai格式的prompt dict
        """
        item = self.data[i]
        question = item["question"]
        doc_id = item["image"].split("/")[-1].split(".")[0]
        image_path = str(self.image_dir / f"{doc_id}.png")
        ocr_path = self.ocr_dir / f"{doc_id}.json"
        layout = self.layout_func(json_path = ocr_path)
        t_layout = utils.truncate_layout(layout, self.tokenizer, self.max_doc_token_cnt)

        # add few shot str
        prompt = copy.deepcopy(self.few_shot_prompt)
        text = self.question_template.format(layout = t_layout,question =question)
        message = {
            'role':'user',
            'content': [{'type':'text','text':text},{'type':'image_url','image_url':{'url':image_path}}]
        }
        prompt.append(message)
        ret = dict(
            prompt = prompt,
            question = question,
            image_path = str(image_path),
            ocr_path=str(ocr_path),
            layout=t_layout,
        )
        if "answers" in item.keys():
            ret.update({"answers": item["answers"]})
        return ret

def main():
    model_path = "/root/autodl-tmp/pretrain-model/Qwen-VL-Chat"
    pipe = pipeline(
        model_path = model_path,
        backend_config=TurbomindEngineConfig(
            session_len=8192,
        ),
        chat_template_config=ChatTemplateConfig(model_name="qwen-vl-chat-few-shot"),
        log_level="INFO",
    )
    gen_config = GenerationConfig(
        top_k=0,top_p=0.5,max_new_tokens=128
    )
    # tokenizer = pipe.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left',trust_remote_code=True)
    eval_dataset = EvalSPDocVQADatasetWithImg(
        json_data_path="/root/autodl-tmp/spdocvqa-dataset/val_v1.0_withQT.json",
        image_dir="/root/autodl-tmp/spdocvqa-dataset/images",
        ocr_dir="/root/autodl-tmp/spdocvqa-dataset/ocr",
        few_shot_example_json_path="/root/GuiQuQu-docvqa-vllm-inference/few_shot_examples/sp_few_shot_example.json",
        layout_func=utils.get_layout_func("all-star"),
        tokenizer=tokenizer
    )
    with open("/root/GuiQuQu-docvqa-vllm-inference/result/qwen-vl_lmdeploy_zero-shot-with-img.jsonl","a", encoding="utf-8") as f:
        for i, batch in (enumerate(tqdm(eval_dataset))):
            if i == 2:
                break
            resp = pipe.batch_infer(batch["prompt"], gen_config=gen_config)
            log = dict(p=f"[{i}|{len(eval_dataset)}]",
                        input_ids_length=resp.input_token_len,
                        image_path=batch["image_path"],
                        ocr_path=batch["ocr_path"],
                        question=batch["question"],
                        response=resp.text,
                        answers=batch["answers"])
            tqdm.write(json.dumps(log,ensure_ascii=False))
            f.write(json.dumps(log,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
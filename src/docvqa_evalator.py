from abc import abstractmethod
import json
import time
from typing import Callable,List,Dict
import pathlib
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from torch.utils.data import DataLoader,Dataset

import utils
import metrics

class BaseDocVQADataset(Dataset):
    def __init__(self,
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        layout_dir:str,
        few_shot_examples: List[Dict[str, str]],
        question_template: str,
        few_shot_template: str,
        tokenizer:PreTrainedTokenizer, 
    ) -> None:
        """
            if no layout_dir, set it to None
        """
        self.data = utils.load_data(json_data_path)
        self.image_dir = pathlib.Path(image_dir)
        self.ocr_dir = pathlib.Path(ocr_dir)
        self.layout_dir = pathlib.Path(layout_dir)
        self.few_shot_examples = few_shot_examples
        self._few_shot_prompt = None
        self.question_template = question_template
        self.few_shot_template = few_shot_template
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    @abstractmethod
    def get_few_shot_prompt(self):
        """
            get few-shot prompt function
        """
        return NotImplementedError

    def __getitem__(self, idx):
        """
            return
            {   qid: str,
                question: str,
                image_path: str,
                ocr_path: str,
                layout_path: str, (maybe None)
                answer: str(if exist)
            }
        """
        item = self.data[idx]
        qid = item["questionId"]
        question = item["question"]
        doc_id = item["image"].split("/")[-1].split(".")[0]
        image_path = str(self.image_dir / f"{doc_id}.png")
        ocr_path = str(self.ocr_dir / f"{doc_id}.json")
        layout_path = None
        if self.layout_dir is not None:
            layout_path = str(self.layout_dir / f"{doc_id}.txt")
        ret = dict(
            qid=qid,
            question=question,
            image_path=image_path,
            ocr_path=ocr_path,
            layout_path=layout_path,
        )
        if "answers" in item:
            ret["answers"] = item["answers"]
        return ret

class DocVQAEvaluateDatasetOnlyImage(BaseDocVQADataset):
    def __init__(self,
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        layout_dir:str,
        few_shot_examples: List[Dict[str, str]],
        question_template: str,
        few_shot_template: str,
        tokenizer:PreTrainedTokenizer,
        # 子类参数 
        ) -> None:
        super().__init__(json_data_path, image_dir, ocr_dir, layout_dir, few_shot_examples, question_template, few_shot_template, tokenizer)
        
        try:
            self.question_template.format(
                image_path="<img>image_path</img>",
                question="question",
            )
            self.few_shot_template.format(
                image_path="<img>image_path</img>",
                question="question",
                answer="answer",
            )
        except Exception as e:
            raise ValueError("Invalid question or few-shot template")

    def get_few_shot_prompt(self):
        if self._few_shot_prompt != None:
            return self._few_shot_prompt
        text = ""
        for e in self.few_shot_examples:
            image_path = str(self.image_dir / f"{e['image']}")
            text += self.few_shot_template.format(
                image_path=f"<img>{image_path}</img>",
                question=e["question"],
                answer=e["answer"],
            )
        self._few_shot_prompt = text
        return self._few_shot_prompt

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        query = self.get_few_shot_prompt()
        # add question to prompt
        query += self.question_template.format(
            image_path=f"<img>{item['image_path']}</img>",
            question=item["question"],
        )
        prompt = [
            {'role':'user', 'content':query}
            ]
        return item

class DocVQAEvaluateDatasetOnlyLayout(BaseDocVQADataset):
    def __init__(self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        layout_dir: str, 
        few_shot_examples: List[Dict[str, str]], 
        question_template: str, 
        few_shot_template: str, 
        tokenizer: PreTrainedTokenizer,
        # 子类参数
        layout_func: Callable,
        max_doc_token_cnt:int = 2048
        ) -> None:
        super().__init__(json_data_path, 
                         image_dir, 
                         ocr_dir, 
                         layout_dir, 
                         few_shot_examples, 
                         question_template, 
                         few_shot_template, 
                         tokenizer)
        self.layout_func = layout_func
        self.max_doc_token_cnt = max_doc_token_cnt

    def get_few_shot_prompt(self):
        if self._few_shot_prompt != None:
            return self._few_shot_prompt
        text = ""
        for e in self.few_shot_examples:
            # image_path = str(self.image_dir / f"{e['image']}")
            ocr_path = str(self.ocr_dir / f"{e['layout']}")
            layout = self.layout_func(ocr_path)
            layout, _ = utils.truncate_layout2(layout=layout,
                                           tokenizer=self.tokenizer,
                                           max_token_length=self.max_doc_token_cnt)
            text += self.few_shot_template.format(
                layout=layout,
                question=e["question"],
                answer=e["answer"],
            )
        self._few_shot_prompt = text
        return self._few_shot_prompt

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        query = self.get_few_shot_prompt()
        layout = self.layout_func(item["ocr_path"])
        layout, is_truncated = utils.truncate_layout2( \
            layout=layout,
            tokenizer=self.tokenizer,
            max_token_length=self.max_doc_token_cnt)
        
        query += self.question_template.format(
            layout=layout,
            question=item["question"],
        )
        prompt = [
            {'role':'user', 'content':query}
            ]
        item['is_truncated'] = is_truncated
        item['prompt'] = prompt
        return item

class DocVQAEvaluateDatasetImageAndLayout(BaseDocVQADataset):
    def __init__(self, 
        json_data_path: str, 
        image_dir: str, 
        ocr_dir: str, 
        layout_dir: str, 
        few_shot_examples: List[Dict[str, str]], 
        question_template: str, 
        few_shot_template: str, 
        tokenizer: PreTrainedTokenizer,
        # 子类参数
        layout_func: Callable,
        max_doc_token_cnt:int = 2048
        ) -> None:
        super().__init__(json_data_path, 
                         image_dir, 
                         ocr_dir, 
                         layout_dir, 
                         few_shot_examples, 
                         question_template, 
                         few_shot_template, 
                         tokenizer)
        self.layout_func = layout_func
        self.max_doc_token_cnt = max_doc_token_cnt
    
    def get_few_shot_prompt(self):
        if self._few_shot_prompt != None:
            return self._few_shot_prompt
        text = ""
        for e in self.few_shot_examples:
            image_path = str(self.image_dir / f"{e['image']}")
            ocr_path = str(self.ocr_dir / f"{e['layout']}")
            layout = self.layout_func(ocr_path)
            layout,_ = utils.truncate_layout2( \
                layout=layout,
                tokenizer=self.tokenizer,
                max_token_length=self.max_doc_token_cnt)
            text += self.few_shot_template.format(
                image_path=f"<img>{image_path}</img>",
                layout=layout,
                question=e["question"],
                answer=e["answer"],
            )
        self._few_shot_prompt = text
        return self._few_shot_prompt
    

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        query = self.get_few_shot_prompt() # if no few shot, it will be ""
        layout = self.layout_func(item["ocr_path"])
        layout, is_truncated = utils.truncate_layout2( \
            layout=layout,
            tokenizer=self.tokenizer,
            max_token_length=self.max_doc_token_cnt)
        query += self.question_template.format(
            image_path=f"<img>{item['image_path']}</img>",
            layout=layout,
            question=item["question"],
        )
        prompt = [
            {'role':'user', 'content':query}
            ]
        item['is_truncated'] = is_truncated
        item['prompt'] = prompt
        return item

def merge_dict_collate_fn(batch):
    "List[dict{key,value}] -> dict{key,List[value]}, value is not Tensor"
    ret = dict()
    for key in batch[0].keys():
        ret[key] = [d[key] for d in batch]
    return ret

def _model_inference_(model, tokenizer, prompt:List[str] | str, max_new_tokens:int, max_length:int) -> str | List[str]:
    return NotImplementedError

def _get_input_ids(prompt:str, tokenizer:PreTrainedTokenizer) -> List[int]:
    return NotImplementedError

class DocVQAEvaluator(object):
    def __init__(self,
                model,
                tokenizer:PreTrainedTokenizer,
                dataset:BaseDocVQADataset,
                result_dir:str,
                experiment_name:str,
                log_path:str, # 记录推理log
                model_inference:Callable, # 模型推理函数(input is str)
                get_input_ids:Callable, # 获取输入input_ids函数
                max_new_tokens:int = 128,
                max_length:int = 1024,
                batch_size:int = 1,
                ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self._create_dataloader()
        self.log_path = log_path
        self.anls = metrics.ANLS(
            result_dir=result_dir,
            experiment_name=experiment_name,
            dataset_name="spdocvqa",
        )
        self.get_input_ids = get_input_ids
        self.model_inference = model_inference
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
    
    def _create_dataloader(self):
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=merge_dict_collate_fn
        )

    def evaluate(self):
        qids, questions, answers, predictions = [], [], [], []
        gts, image_paths, ocr_paths, layout_paths = [], [], [], []
        with open(self.log_path, 'a', encoding='utf-8') as f:
            for i, batch in enumerate(tqdm(self.data_loader)):
                prompt:List[List[Dict]] = batch["prompt"]
                answers:List[List[str]] = batch["answers"]
                for j, tmp in enumerate(zip(prompt,answers)):
                    p, anss = tmp
                    p = p[-1]['content']
                    start_time = time.time()
                    resp = self.model_inference(model = self.model, 
                                                tokenizer = self.tokenizer, 
                                                prompt = p, 
                                                max_new_tokens=self.max_new_tokens, 
                                                max_length=self.max_length)
                    exec_time = time.time() - start_time

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
                    input_ids = self.get_input_ids(p,self.tokenizer)
                    log = dict(p=f"[{i*self.batch_size+j+1}|{len(self.dataset)}]",
                                time=f"{exec_time:.2f}s",
                                prompt_len=len(p),
                                token_len=len(input_ids),
                                is_truncated=batch['is_truncated'][j] if 'is_truncated' in batch else None,
                                image_path=image_path,
                                ocr_path=ocr_path,
                                layout_path=layout_path,
                                question=question,
                                response=resp,
                                answers=anss)
                    log_str = json.dumps(log, ensure_ascii=False)
                    tqdm.write(log_str)
                    f.write(log_str + "\n")

        score = self.anls.compute_and_save_docvqa(
            qids=qids,
            questions=questions,
            answers=gts,
            predictions=predictions,
            image_paths=image_paths,
            ocr_paths=ocr_paths,
            layout_paths=layout_paths,
        )
        print(f"{self.anls.experiment_name} score: {score:.4f}")
        

    def __call__(self):
        self.evaluate()
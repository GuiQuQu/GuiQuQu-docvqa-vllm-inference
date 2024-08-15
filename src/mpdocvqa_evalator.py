from typing import Callable, List, Dict, Any, Tuple
import json
import time
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

import utils
import metrics


class BaseMPDocVQADatasetForEval(Dataset):
    def __init__(
        self,
        json_data_path: str,
        image_dir: str,
        layout_dir: str,
        ocr_dir: str,
        question_template: str,
        tokenizer: PreTrainedTokenizer,
    ):
        self.data = utils.load_data(json_data_path)
        self.image_dir = Path(image_dir)
        self.layout_dir = Path(layout_dir) if isinstance(layout_dir, str) else layout_dir
        self.ocr_dir = Path(ocr_dir)
        self.question_template = question_template
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        item: dict = self.data[index]
        qid = item["questionId"]
        question = item["question"]
        page_ids = item["page_ids"]

        image_paths = [self.image_dir / f"{page_id}.jpg" for page_id in page_ids]
        ocr_paths = [self.ocr_dir / f"{page_id}.json" for page_id in page_ids]
        layout_paths = [None] * len(page_ids)
        if self.layout_dir is not None:
            layout_paths = [self.layout_dir / f"{page_id}.txt" for page_id in page_ids]
        ret = dict(
            qid=qid,
            question=question,
            page_ids=page_ids,
            image_paths=image_paths,
            ocr_paths=ocr_paths,
            layout_paths=layout_paths,
        )
        if "answers" in item.keys() and "answer_page_idx" in item.keys():
            true_answers = item["answers"]
            answer_page_idx = item["answer_page_idx"]
            ret.update(
                {"true_answers": true_answers, "answer_page_idx": answer_page_idx}
            )
        return ret


class MPDocVQADatasetForEvalWithLayoutAndImage(BaseMPDocVQADatasetForEval):
    def __init__(
        self,
        json_data_path: str,
        image_dir: str,
        layout_dir: str,
        ocr_dir: str,
        question_template: str,
        tokenizer: PreTrainedTokenizer,
        #
        layout_func: Callable[[str], List[str]],
        max_doc_token_cnt: int = 2048,
    ) -> None:
        super().__init__(
            json_data_path, image_dir, layout_dir, ocr_dir, question_template, tokenizer
        )
        self.layout_func = layout_func
        self.max_doc_token_cnt = max_doc_token_cnt

    def __getitem__(self, index) -> Any:
        item = super().__getitem__(index)
        image_cnt = len(item["page_ids"])
        layouts = [self.layout_func(ocr_path) for ocr_path in item["ocr_paths"]]
        truncated_layouts = []
        is_truncateds = []
        for layout in layouts:
            truncated_layout, is_truncated = utils.truncate_layout2(
                layout, self.tokenizer, self.max_doc_token_cnt
            )
            truncated_layouts.append(truncated_layout)
            is_truncateds.append(is_truncated)
        querys = [
            self.question_template.format(
                image_path=f"<img>{item['image_paths'][i]}</img>",
                layout=truncated_layouts[i],
                question=item["question"],
            )
            for i in range(image_cnt)
        ]
        prompts = [ [{'role':'user','content':query}] for query in querys]
        item['prompts'] =prompts
        item['is_truncateds'] = is_truncateds
        return item

def default_collate_fn(batch):
    ret = dict()
    all_keys = list(batch[0].keys())
    for key in all_keys:
        if isinstance(batch[0][key], torch.Tensor):
            ret[key] = torch.stack([item[key] for item in batch])
        else:
            ret[key] = [item[key] for item in batch]
    return ret

class MPDocVQAEvaluator:
    def __init__(
        self, 
        model, 
        tokenizer: PreTrainedTokenizer, 
        dataset: BaseMPDocVQADatasetForEval,
        result_dir:str,
        experiment_name:str,
        model_inference_fn:Callable,
        get_input_ids_fn:Callable,
        max_new_tokens:int = 128,
        max_length:int = 1024,
        # batch_size:int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = 1 # batch_size 必须是1
        self._create_dataloader()
        self.anls = metrics.ANLS(
            result_dir=result_dir,
            experiment_name=experiment_name,
            dataset_name="mpdocvqa"
        )
        self.get_input_ids_fn = get_input_ids_fn
        self.model_inference_fn = model_inference_fn
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.log_path = Path(result_dir) / f"{experiment_name}_eval.log"
    
    def _create_dataloader(self):
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_collate_fn
        )
    
    def evaluate(self):
        mpdocvqa_items = []
        with open(self.log_path, "w") as f:
            for i, batch in enumerate(tqdm(self.data_loader)): # batch_size=1
                prompts:List[List[List[Dict]]] = batch["prompts"]
                true_answers:List[List[str]] = batch['true_answers']
                answer_page_idxs:List[List[int]] = batch['answer_page_idx']
                for j, item in enumerate(zip(prompts,true_answers,answer_page_idxs)): # item,item有一个prompt_list
                    prompt_list, answers, answer_page_idx = item
                    pred_result = [] 
                    for prompt in prompt_list:
                        p:str = prompt[-1]['content']
                        st = time.time()
                        resp_dict = self.model_inference_fn(
                            model = self.model,
                            tokenizer = self.tokenizer,
                            prompt = p,
                            max_new_tokens = self.max_new_tokens,
                            max_length = self.max_length
                        )
                        pred_result.append((resp_dict['score'], resp_dict['response']))
                        
                        exec_time = time.time() - st
                    # 记录model_inference_fn 结果
                    question = batch['question'][j]
                    image_paths = batch['image_paths'][j]
                    ocr_paths = batch['ocr_paths'][j]
                    layout_paths = batch['layout_paths'][j]
                    qid = batch['qid'][j]
                    
                    mpdocvqa_item = metrics.MPDocVQAItem(
                        qid=qid,
                        question=question,
                        predictions=pred_result,
                        image_paths=image_paths,
                        ocr_paths=ocr_paths,
                        layout_paths=layout_paths,
                        answers = answers,
                        answer_page_idx=answer_page_idx,
                        exec_time=exec_time
                    )
                    mpdocvqa_items.append(mpdocvqa_item)
                    progress = f"[{i*self.batch_size+j}|{len(self.data_loader)}]"
                    progress_dict = {"progress":progress}
                    progress_dict.update(mpdocvqa_item.to_result_dict())
                    f.write(json.dumps(progress_dict, ensure_ascii=False) + "\n")
                    f.flush()
        anls_score = self.anls.compute_and_save_mpdocvqav2(
            mpdocvqa_items=mpdocvqa_items,
            split="val"
        )
        return anls_score

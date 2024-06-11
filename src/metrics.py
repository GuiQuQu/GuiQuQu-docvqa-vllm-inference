"""
    计算anls指标
"""
from typing import List
import Levenshtein
import os
import json

def anls(predict_answer: List[str], ground_truth: List[List[str]], threshold=0.5) -> float:
    """
    n = len(predict_answer),问题的数量
    predict_answer: List[str], 每个问题的预测答案
    ground_truth: List[List[str]], 每个问题的真实答案[一个问题可能存在多个]

    reference:
    https://github.com/shunk031/ANLS
    """
    res = 0.0
    n = len(predict_answer)
    for pa, gts in zip(predict_answer, ground_truth):
        y_pred = " ".join(pa.strip().lower().split())
        anls_scores: List[float] = []
        for gt in gts:
            y_true = " ".join(gt.strip().lower().split())
            anls_score = similarity(y_true, y_pred, threshold=threshold)
            anls_scores.append(anls_score)
        res += max(anls_scores)
    return res / n


# Normalized Levenshtein distance
def similarity(answer_ij: str, predict_i: str, threshold: float = 0.5) -> float:
    maxlen = max(len(answer_ij), len(predict_i))
    edit_dist = Levenshtein.distance(answer_ij, predict_i)
    nl_score = 0.0
    if maxlen != 0:
        nl_score = float(edit_dist) / float(maxlen)

    return 1-nl_score if nl_score < threshold else 0.0

# borrow from 'https://github.com/WenjinW/LATIN-Prompt/blob/main/metric/anls.py'
class ANLS(object):
    def __init__(self,
                 result_dir,
                 experiment_name,
                 dataset_name,
                 replace_star:bool =True,
                 replace_n:bool = True,) -> None:
        super().__init__()
        self.result_dir = result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.replace_star = replace_star
        self.replace_n = replace_n
    
    def _ls(self, s1:str,s2:str, threshold=0.5):
        # s1 = " ".join(s1.strip().lower().split())
        # s2 = " ".join(s2.strip().lower().split())
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        nls = Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
        return 1-nls if nls < threshold else 0.0

    def _ls_multiple(self, pred, answers:List[str],threshold=0.5):
        if self.replace_star:
            pred = pred.replace("*","")
        if self.replace_n:
            pred = pred.replace("\n","")
        return max([self._ls(pred, ans, threshold) for ans in answers])
    
    def compute_and_save_docvqa(self,qids:List[int],
                                questions:List[str], 
                                predictions:List[str],
                                image_paths:List[str] = None,
                                ocr_paths:List[str] = None,
                                layout_paths:List[str] = None,
                                answers:List[List[str]]=None, 
                                split="val"):
        """
            保存计算结果,如果answers不为None,则计算anls
        """
        if answers is not None:
            assert image_paths is not None, "for dev data, image_paths is None"
            assert ocr_paths is not None, "for dev data, ocr_paths is None"
            assert layout_paths is not None, "for dev data,layout_paths is None"
        all_anls = 0.0
        results = []
        for i in range(len(qids)):
            result = {
                "questionId": qids[i],
                "answer": predictions[i]
            }
            if answers is not None:
                anls = self._ls_multiple(predictions[i], answers[i])
                all_anls += anls
                result["question"] = questions[i]
                result['image_path'] = image_paths[i]
                result['ocr_path'] = ocr_paths[i]
                result['layout_path'] = layout_paths[i]
                result["ground_truth"] = answers[i]
                result["anls"] = anls
            results.append(result)
        save_path = os.path.join(self.result_dir, f"{self.experiment_name}_{self.dataset_name}_{split}.json")
        if answers is not None:
            score = all_anls / len(qids)
            results.insert(0,{"anls":score})
        with open(save_path,"w",encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        return all_anls / len(qids)


def old_anls2new_anls(
        result_jsonl:str, 
        data_json:str,
        new_result_dir,
        experiment_name):
    """
        把旧的结果文件转成新的结果文件格式
    """
    import utils
    line = ""
    results = []
    data = utils.load_data(data_json)
    with open(result_jsonl,"r",encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line: break
            result = json.loads(line)
            results.append(result)
    
    anls = ANLS(
        result_dir=new_result_dir,
        experiment_name=experiment_name,
        dataset_name="spdocvqa",
    )
    qids = []
    questions = []
    predictions = []
    image_paths = []
    ocr_paths = []
    layout_paths = []
    answers = []
    for result, item in zip(results, data):
        assert result["question"] == item["question"]
        qids.append(item["questionId"])
        questions.append(item["question"])
        predictions.append(result["response"])
        image_path = result["image_path"] if "image_path" in result else result['image']
        ocr_path:str = result["ocr_path"] if "ocr_path" in result else result['ocr']
        image_paths.append(image_path)
        ocr_paths.append(ocr_path)
        layout_path = ocr_path.replace("ocr","layout").replace(".json",".txt")
        layout_paths.append(layout_path)
        answers.append(item["answers"])
    
    anls.compute_and_save_docvqa(
        qids=qids,
        questions=questions,
        predictions=predictions,
        image_paths=image_paths,
        ocr_paths=ocr_paths,
        layout_paths=layout_paths,
        answers=answers,
        split="val"
    )
    """
    result = {
        "questionId": qids[i],
        "answer": predictions[i]
        "question": questions[i]
        'image_path': image_path[i]
        'ocr_path': ocr_path[i]
        'layout_path': layout_path[i]
        "ground_truth": answers[i]
        "anls": anls
    }
    """

if __name__ == "__main__":
        project_dir = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
        result_jsonl = os.path.join(project_dir,
                                    "result/qwen-vl_only-image_qlora_old/qwen-vl-int4_sft_only-image_checkpoint-final.jsonl")
        data_json = "/home/klwang/data/spdocvqa-dataset/val_v1.0_withQT.json"
        new_result_dir = os.path.join(project_dir, "result/qwen-vl_only-image_qlora/")
        old_anls2new_anls(
            result_jsonl=result_jsonl,
            data_json=data_json,
            new_result_dir=new_result_dir,
            experiment_name="qwen-vl-int4_sft_only-image_checkpoint-final"
        )

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
                 exp_name,
                 dataset_name) -> None:
        super().__init__()
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.dataset_name = dataset_name
    
    def _ls(self, s1,s2, threshold=0.5):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        nls = Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
        return 1-nls if nls < threshold else 0.0

    def _ls_multiple(self, pred, answers:List[str],threshold=0.5):
        return max([self._ls(pred, ans, threshold) for ans in answers])
    
    def compute_and_save_docvqa(self,qids:List[int],
                                questions:List[str], predictions:List[str],
                                  answers:List[List[str]]=None, split="val"):
        """
            保存计算结果,如果answers不为None,则计算anls
        """
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
                result["answer"] = answers[i]
                result["anls"] = anls
            results.append(result)
        save_path = os.path.join(self.result_dir, f"{self.exp_name}_{self.dataset_name}_{split}.json")
        with open(save_path,"w",encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        return all_anls / len(qids)
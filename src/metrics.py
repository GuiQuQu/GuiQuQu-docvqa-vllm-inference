"""
    计算anls指标
"""
from typing import List
import Levenshtein


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

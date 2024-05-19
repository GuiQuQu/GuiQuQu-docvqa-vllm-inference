
from functools import partial

from transformers import PreTrainedTokenizer
import json
import random
import numpy as np
import torch

import handle_ocr

def truncate_layout(layout:str, 
                    tokenizer:PreTrainedTokenizer = None, 
                    max_token_length:int = 1024):
    if tokenizer == None:
        return layout
    lines = layout.split("\n")
    lines_input_ids = [tokenizer([l], return_tensors="pt").input_ids for l in lines]
    reserved_lines = []
    ids_cnt = 0
    for i, input_ids in enumerate(lines_input_ids):
        if ids_cnt + input_ids.size(-1) < max_token_length:
            ids_cnt += input_ids.size(-1)
            reserved_lines.append(lines[i])
        else: break
    return "\n".join(reserved_lines)


def open_json(json_path: str):
    """
        打开一个json文件
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_data(json_path: str):
    """
        打开json文件,并返回其中的data字段value内容
    """
    data = open_json(json_path)
    return data["data"]

def seed_everything(seed):
    """
        seed_everything
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_layout_func(type:str):
    """
        根据给定的type返回对应的layout处理方式
    """
    if type == "all-star":
        return partial(handle_ocr.sp_get_layout_by_json_path, placeholder="*")
    elif type == "lines":
        return handle_ocr.sp_get_lines_layout_by_json_path
    elif type == "words":
        return handle_ocr.sp_get_baseline_layout_by_json_path
    else:
        raise ValueError("Not support layout pattern")
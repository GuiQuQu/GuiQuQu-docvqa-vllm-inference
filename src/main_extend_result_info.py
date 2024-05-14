"""
    扩展一下result结果中的相关信息
"""
import json
from typing import List

def open_result(result_path):
    result = []
    with open(result_path,"r",encoding="utf-8") as f:
        line = ""
        while True:
            line = f.readline()
            if not line: break
            res = json.loads(line)
            result.append(res)
    return result

def save_result(result,save_path):
    with open(save_path,"w",encoding="utf-8") as f:
        for res in result:
            f.write(json.dumps(res,ensure_ascii=False)+"\n")

def extand_info(data_path, result_path):
    # data_path = "/home/klwang/data/SPDocVQA/val_v1.0_withQT.json"
    # result_path = "../result/debug.jsonl"

    eval_dict = dict()
    with open(data_path,"r",encoding="utf-8") as f:
        eval_data = json.load(f)["data"]
    
    for item in eval_data:
        key = item["question"] + "_" + item["image"].split("/")[-1]
        eval_dict[key] = item
    result = open_result(result_path)
    for i,res in enumerate(result):
        key = res["question"] + "_" + res["image_path"].split("/")[-1]
        info = eval_dict[key]
        question = result[i]["question"]
        response = result[i]["response"]
        answers = result[i]["answers"]
        result[i].pop("question")
        result[i].pop("response")
        result[i].pop("answers")
        result[i].update({
            "question_types" : info["question_types"],
            "question":question,
            "response":response,
            "answers":answers
        })
    save_result(result=result,save_path=result_path.replace(".jsonl","_extend.jsonl"))

def dict_str(d, expect_keys):
    nd = dict()
    for key in expect_keys:
        nd[key] = d[key]
    return json.dumps(nd,ensure_ascii=False)

def main():
    # extand_info(data_path="/home/klwang/data/SPDocVQA/val_v1.0_withQT.json",result_path="../result/debug.jsonl")
    # 统计一下错误的问题类型信息
    question_type_dict = dict()
    result = open_result(result_path="../result/debug_extend.jsonl")
    for item in result:
        question_types:List[str] = item["question_types"]
        for qt in question_types:
            if qt not in question_type_dict:
                question_type_dict[qt] = 0
            question_type_dict[qt] += 1
    
    expect_keys = question_type_dict.keys()
    print("debug.jsonl=>\n",dict_str(question_type_dict,expect_keys))

    data_path = "/home/klwang/data/SPDocVQA/val_v1.0_withQT.json"
    question_type_dict = dict()
    with open(data_path,"r",encoding="utf-8") as f:
        eval_data = json.load(f)["data"]
    for item in eval_data:
        question_types:List[str] = item["question_types"]
        for qt in question_types:
            if qt not in question_type_dict:
                question_type_dict[qt] = 0
            question_type_dict[qt] += 1
    
    print("val_v1.0_withQT.json=>\n",dict_str(question_type_dict,expect_keys))

if __name__ == "__main__":
    main()

# debug.jsonl=>
#  {"figure/diagram": 137, "others": 56, "handwritten": 91, "form": 165, "Image/Photo": 49, "table/list": 423, "free_text": 130, "layout": 491, "Yes/No": 11}
# val_v1.0_withQT.json=>
#  {"figure/diagram": 265, "others": 236, "handwritten": 319, "form": 1021, "Image/Photo": 98, "table/list": 1780, "free_text": 765, "layout": 1981, "Yes/No": 28}

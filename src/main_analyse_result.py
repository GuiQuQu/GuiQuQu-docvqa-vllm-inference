
import json
from metrics import anls

def get_result(result_path):
    result = []
    with open(result_path,"r",encoding="utf-8") as f:
        line = ""
        while True:
            line = f.readline()
            if not line: break
            res = json.loads(line)
            result.append(res)
    return result

def check(resp)-> bool:
    score = anls([resp["response"]], [resp["answers"]])
    if (score == 0):
        return True
    return False

def main():
    jsonl_path = "../result/qwen1.5-7b-qlora_checkpoint-140_with-ocr_no-few-shot_all-stars_template-v3.jsonl"
    result = get_result(jsonl_path)
    save_path = "../result/debug.jsonl"
    # 过滤检查之后的内容不一致的数据
    wrong_result = []
    for resp in result:
        if (check(resp)):
            wrong_result.append(resp)
    with open(save_path,"w",encoding="utf-8") as f:
        for res in wrong_result:
            f.write(json.dumps(res,ensure_ascii=False)+"\n")
    print(f"Done,get wrong result [{len(wrong_result)}|{len(result)}]")

if __name__ == "__main__":
    main()
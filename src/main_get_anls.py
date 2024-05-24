
import json

from metrics import anls

def main():
    # 0.73.89
    result_path = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference/result/qwen-vl/qwen-vl-int4_sft-vl-checkpoint-100.jsonl"
    reponses = []
    gts = []
    with open(result_path,"r",encoding="utf-8") as f:
        line = ""
        while True:
            line = f.readline()
            if not line: break
            res = json.loads(line)
            resp = res["response"].replace("*","").replace("\n","")
            reponses.append(resp)
            gts.append(res["answers"])

    print(anls(reponses,gts))

if __name__ == "__main__":
    main()

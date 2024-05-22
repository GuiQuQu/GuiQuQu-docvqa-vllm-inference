
import json

from metrics import anls

def main():
    result_path = "../result/qwen-vl-int4_sft-vl.jsonl"
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

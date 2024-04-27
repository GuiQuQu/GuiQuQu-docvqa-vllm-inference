
import json

from metrics import anls

def main():
    result_path = "../result/qwen1.5-7b-vllm_with-ocr_words_template-v3.jsonl"
    reponses = []
    gts = []
    with open(result_path,"r",encoding="utf-8") as f:
        line = ""
        while True:
            line = f.readline()
            if not line: break
            res = json.loads(line)
            reponses.append(res["response"].replace("*",""))
            gts.append(res["answers"])
    # qwen-1.5-7b + simple ocr 使用*做placeholder
    # qwen1.5-7b-vllm-with_ocr_star_placeholderv1.jsonl
    # 0.5013919903487696

    # qwen-1.5-7b + simple ocr 使用space做placeholder
    # qwen1.5-7b-vllm-with_ocr-space_placeholder.jsonl
    # 0.4855360550527471

    # qwen-1.5-7b + simple ocr 使用*做placeholder+增加了细致的prompt(v2)
    # qwen1.5-7b-vllm-with_ocr-star_placeholderv2.jsonl
    # 0.6421302824522843

    # qwen-1.5-7b + simple ocr 使用*做placeholder+增加了细致的prompt(v2),将所有的*去掉(replace("*",""))
    # qwen1.5-7b-vllm-with_ocr-star_placeholderv2.jsonl
    # 0.6606421199650596

    # qwen-1.5-7b + simple ocr 使用*做placeholder+增加了细致的prompt,extract answer(v3),将所有的*去掉
    # qwen1.5-7b-vllm-with_ocr-star_placeholderv3.jsonl
    # 0.6756124286673922

    # qwen-1.5-7b + simple ocr+采用line布局+增加了细致的prompt(v3)
    # qwen1.5-7b-vllm_with-ocr_lines_template-v3.jsonl
    # 0.6638352409777092

    # qwen-1.5-7b + simple ocr+采用words布局+增加了细致的prompt(v3)
    # qwen1.5-7b-vllm_with-ocr_lines_template-v3.jsonl
    # 0.6383523218491968
    print(anls(reponses,gts))

if __name__ == "__main__":
    main()

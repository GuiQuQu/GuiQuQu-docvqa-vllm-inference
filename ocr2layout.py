import json

def main():
    jsonl_file = '/home/klwang/code/GuiQuQu-docvqa-vllm-inference/result/qwen-vl/qwen-vl-int4_sft-vl-checkpoint-400-cp.jsonl'
    new_res = []
    with open(jsonl_file,'r',encoding='utf-8') as f:
        line = ''
        while True:
            line = f.readline()
            if not line: break
            item = json.loads(line)
            ocr_path = item["ocr_path"] if "ocr_path" in item else item["ocr"]
            image_path = item["image_path"] if "image_path" in item else item["image"]
            # ocr_path 'xxx/ocr/xxx.json' => 'xxx/layout/xxx.txt'
            ocr_path = ocr_path.replace("ocr","layout").replace(".json",".txt")
            new_item={}
            for key in item.keys():
                if key in ["ocr_path","ocr"]:
                    new_item["ocr_path"] = ocr_path
                elif key in ["image_path","image"]:
                    new_item["image_path"] = image_path
                else:
                    new_item[key] = item[key]
            new_res.append(new_item)
    with open(jsonl_file,"w",encoding="utf-8") as f:
        for item in new_res:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
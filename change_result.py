"""
    修改结果文件中指向的路径
    包括 
    'ocr_path', or 'ocr'
    'image_path' or 'image'
    example "/root/autodl-tmp/spdocvqa-dataset/ocr/fqvw0217_34.json
"""
import json
def main():
    jsonl_file="/home/klwang/code/GuiQuQu-docvqa-vllm-inference/result/qwen-vl/qwen-vl-int4_no-ft_few-shot3-all-star_add-image.jsonl"
    old_dir = "/root/autodl-tmp/spdocvqa-dataset"
    new_dir = "/home/klwang/data/spdocvqa-dataset"
    new_res = []
    with open(jsonl_file,"r",encoding="utf-8") as f:
        line = ""
        while True:
            line = f.readline()
            if not line: break
            item = json.loads(line)
            ocr_path = item["ocr_path"] if "ocr_path" in item else item["ocr"]
            image_path = item["image_path"] if "image_path" in item else item["image"]
            ocr_path = ocr_path.replace(old_dir,new_dir)
            image_path = image_path.replace(old_dir,new_dir)
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
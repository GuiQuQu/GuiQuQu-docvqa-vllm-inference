"""
    将一个json文件修改为更容易阅读的方式
"""
import json  

json_path = "/home/klwang/data/SPDocVQA/ocr/mnbx0223_74.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

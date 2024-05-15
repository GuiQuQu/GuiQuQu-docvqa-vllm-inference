
"""
    目前不考虑使用docowl1.5的模型
"""
from DocOwl1_5.docowl_infer import DocOwlInfer


model_path = "/home/klwang/pretrain-model/DocOwl1.5-Omni"

docowl = DocOwlInfer(model_path, anchors="grid_9", add_global_img=True)
image = "./handle_ocr/sp/jmlh0227_8.png"
query = "Convert the picture to Markdown syntax"
answer = docowl.inference(image, query)
print(answer)
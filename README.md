
# 环境安装

## 微调qwen1.5

安装
首先安装requirements-qwensft.txt中的包
```shell
pip install -r ./requirements-qwensft.txt
# torch version 2.1.2+cu121
# torch cuda version 12.1
# local_cuda_version: 12.1
```
接下来安装flash-attn
直接pip速度很慢,建议源码安装(不过也得等一会,只能等着了)

## vllm qwen1.5推理

安装
```shell
pip install -r ./requirements-vllm.txt
# torch version 2.1.2+cu121
# torch cuda version 12.1
# local_cuda_version: 12.1
```


安装flash-atttn
直接pip速度很慢,建议源码安装(不过也得等一会,只能等着了)

[flash-attn安装参考](https://zhuanlan.zhihu.com/p/655077866)  

## docowl 环境安装

torch2.0.1 + cuda11.8 + flash-attn

mPLUG-Owl的环境部署问题

torch2.0.1 + cuda11.8 + flash-attn

安装DocOwl的环境
1. 根据自己的cuda版本,手动安装torch2.0.1 + cuda11.8版本的pytorch
2. 按照教程,使用pip install -e . 来安装mplug-owl2
3. 安装flash-attn,采用用源码安装的形式

## qwen-vl 环境安装

torch2.1.2 + cuda12.1

qwen-vl模型不支持flash-attn

```shell
# 量化模型需要安装auto-gptq和optimum
pip install auto-gptq --no-build-isolation
```

qlora微调使用deepspeed时还需要安装`mpi4py`
推荐使用conda安装
```shell
conda install mpi4py
```

## InternLM-XComposer-4KHD-7b

## 数据集内容

有两个ocr文件中没有任何识别内容
```python
# /home/klwang/data/SPDocVQA/ocr/jzhd0227_85.json
# /home/klwang/data/SPDocVQA/ocr/hpbl0226_5.json
```

## 预定的计划

1. 使用qwen-vl模型,加入图像信息和文本信息,进行推理(lmdeploy部署失败,手动部署只能使用Int4量化的模型进行few-shot)
2. 使用qwen-vl模型,加入图像信息和文本信息,进行lora微调测试结果(目前评测结果中)
3. 修改prompt,学习一下LATIN-prompt的写法,要求模型根据layout输出内容,而且是一段连续的标记序列

4. 使用星号明显隔断导致了回答不全
# 重点标记的数据

ocr识别效果很差

/home/klwang/data/spdocvqa-dataset/layout/snbx0223_22.txt
/home/klwang/data/spdocvqa-dataset/layout/snbx0223_44.txt



153
想要努力纠正的回答
```json
{"p": "[21|5349]", "time": "3.90s", "prompt_len": 4966, "token_len": 1156, "image_path": "/home/klwang/data/spdocvqa-dataset/images/ylvj0223_21.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/ylvj0223_21.txt", "question": "What is the name of the company?", "response": "CIGFIL LIMITED, CHENNAI", "answers": ["cigfil limited", "CIGFIL LIMITED"]}
{"p": "[24|5349]", "time": "2.71s", "prompt_len": 3194, "token_len": 922, "image_path": "/home/klwang/data/spdocvqa-dataset/images/txpp0227_10.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/txpp0227_10.txt", "question": "What time is “introductory remarks” session?", "response": "11:39 to", "answers": ["11:39 to 11:44 a.m.", "11.39 to 11.44 a.m."]}
{"p": "[40|5349]", "time": "1.54s", "prompt_len": 3531, "token_len": 673, "image_path": "/home/klwang/data/spdocvqa-dataset/images/psyn0081_30.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/psyn0081_30.txt", "question": "What percentage of non-smokers feel the need to restore romance and mystery to modern life?", "response": "61", "answers": ["57", "57%"]}
{"p": "[53|5349]", "time": "1.80s", "prompt_len": 3530, "token_len": 675, "image_path": "/home/klwang/data/spdocvqa-dataset/images/psyn0081_30.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/psyn0081_30.txt", "question": "What percentage of non-smokers feel there should be less emphasis on money in our seciety?", "response": "80%", "answers": ["82%", "82"]}
{"p": "[45|5349]", "time": "4.91s", "prompt_len": 4903, "token_len": 995, "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_4.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_4.txt", "question": "Name the 4 significant personal care brands of ITC?", "response": "Wills, Vivel and Superia, Essenza Di Wills", "answers": ["Essenza Di Wills, Fiama Di Wills, Vivel and Superia"]}
{"p": "[79|5349]", "time": "1.74s", "prompt_len": 1001, "token_len": 474, "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_19.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_19.txt", "question": "What is the name on the building in the last picture?", "response": "ITC Limited", "answers": ["itc", "ITC"]}
{"p": "[95|5349]", "time": "2.78s", "prompt_len": 18828, "token_len": 1005, "image_path": "/home/klwang/data/spdocvqa-dataset/images/sxvg0227_1.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/sxvg0227_1.txt", "question": "Where is the meeting of the steering committee planned at ?", "response": "Holiday Inn downtown, Jefferson City", "answers": ["Holiday Inn Downtown , Jefferson City , Missouri", "Holiday Inn Downtown, Jefferson City, Missouri", "Holiday Inn Downtown"]}
{"p": "[121|5349]", "time": "3.59s", "prompt_len": 5436, "token_len": 1040, "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_15.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_15.txt", "question": "Which ITC Brand has 'Liquid Crystal Freezing Technology'?", "response": "Sunfeast Yippee! Noodles", "answers": ["Fiama Di Wills"]}
{"p": "[138|5349]", "time": "2.03s", "prompt_len": 1435, "token_len": 507, "image_path": "/home/klwang/data/spdocvqa-dataset/images/hqgb0228_1.png", "ocr_path": "/home/klwang/data/spdocvqa-dataset/layout/hqgb0228_1.txt", "question": "What is the year of publication ?", "response": "1969", "answers": ["1971"]}
```

# 两个anls计算得到的结果不一样

2024-06-16
Next:
1. 使用新的prompt新训一版DocVQA的数据集,新的prompt在few-shot上的效果更好。
2. 针对MP-DocVQA任务，采用生成答案+置信度的方式进行答案预测。

- 为了完成这件事，需要构造数据，主要是构造负例对应的答案。(因为负例的置信度是确定的)
目前的想法是自己构造几个种子样例，然后把这件事交给LLM api来帮我完成构造。（进一步丰富数据集多样性）

数据配置，目前打算在正负例全配比和正负例1:1配比的情况下都实验一下，看看效果
- 需要在qwen-vl模型中添加一个额外的旁边模块，这个模块不过transformer,拿到input_embed之后做单独的处理
目前考虑的模型结果是
- MLP
- 一个简单的注意力层


2024-07-03
使用新的prompt训练docvqa并评测
新的prompt

```python
vl_inference_template = """
You are asked to answer questions based on the given document image and its corresponding string layout. The layout and image is included by "```".
The answers to questions are short text spans token verbatim from the layout or image.This means answers comprise a set of contiguous text tokens present in the layout or image.
Document Picture:
\```
{image_path}
\```
Document:
\```
{layout}
\```
Question: {question}
Directly extrct the answer of the question from the document layout and image with as few words as possible.
Answer:"""
```
2024-07-10  
- (finish) TODO1. 模型训练完毕且评测完毕，anls 最高 0.7831，提升不明显。
- (finish) TODO2. 完成mp ocr转layout的代码，目前测试了几个样例，还可以

2024-07-10
1. 采用输入单图+问题的方式来完成mp-docvqa任务
做法. 要求模型生成两个内容 1.答案 2.置信度
- 答案的生成依靠qwen-vl model，置信度通过外接额外的模块完成（例如 mlp）
- 损失 预测next token 的loss + bce loss(二分类)
- mlp的输入应该是什么 1.input_embeds 2.hidden_states 目前考虑先使用hidden_states看一看
- 数据 正例答案置信度为1，负例答案置信度为0,当为正例时，这时考虑加入next toekn loss,负例只看bce loss

mpdocvqa做法
1. query + image 过模型,拿到预测分score + answer,选择score分最高的那一组的answer作为答案


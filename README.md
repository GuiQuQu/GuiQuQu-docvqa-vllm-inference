
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

## 数据集内容

有两个ocr文件中没有任何识别内容
```python
# /home/klwang/data/SPDocVQA/ocr/jzhd0227_85.json
# /home/klwang/data/SPDocVQA/ocr/hpbl0226_5.json
```

## 预定的计划

1. 使用qwen-vl模型,加入图像信息和文本信息,进行推理



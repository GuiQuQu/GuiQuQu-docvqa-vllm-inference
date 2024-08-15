
如何自定义模型结构，内部包含大模型,然后外接一个随机初始化的模型，例如MLP,
大模型内部使用lora，冻结大部分参数,同时要求外接部分也进行训练
在实现完模型之后，采用Trainer的Api进行训练
使用from_pretrained加载模型
1. 采用auto-gptq量化模型

使用最后使用lora包装模型

## 要点
1. 最后在使用lora包装模型,确保模型保存时只保存lora权重(失败，如果要这么做，量化搞不了)
2. 可以使用Trainer进行训练，并且只保存adapter的参数
3. 使用gptq量化模型
4. 模型可以使用from_pretrained加载
5. 要求支持gradient_checkpointing


目前采用先用lora包装base_model的做法可以成功train,但是模型无法使用from_pretrained加载

**要点1**

要点1失败,如果要这么做,那么模型必须是使用from_pretrained加载的才可以使用lora进行包装
但是实际上我的模型是初始化加载的,内部使用from_pretrained加载qwenvl
因此还是实际上只用lora包装qwenvl模型,而不是整个模型,所以只保存lora权重的想法也就无法实现了

**要点2**
之前的做法是可以的,那么模型的权重应该如何加载
只能通过model.load_state_dict来加载权重


## 如何利用Trainer Api训练自己的自定义模型

1. 一般来说我们的自定义模型会使用一些预训练模型,例如bert
2. model的forward函数需要返回loss,这样我们就不需要修改Trainer的compute_loss函数

### 定义model类
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = GPT2Model.from_pretrained('gpt2')
        ...
    def forward(self, input_ids):
        ...
    
    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()
```

### 训练参数
训练参数使用TrainingArguments类,可以使用HfArgumentParser进行解析
这个类的Trainer附带的参数类,然后其他的参数可以自己进行定义
```python
import transformers
parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
)
(
    model_args,
    data_args,
    training_args,
    lora_args,
) = parser.parse_args_into_dataclasses()
```

### 训练数据集

继承Dataset类即可

### 开始训练
```
from transformers import Trainer, TrainingArguments


```

[bsz,1]

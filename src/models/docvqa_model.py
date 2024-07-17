from typing import List
from torch import nn
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import GPTQConfig
from transformers import PreTrainedModel

from Qwen_VL.modeling_qwen import QWenLMHeadModelForDocVQA, QWenConfig,QWenModel
from Qwen_VL.tokenization_qwen import QWenTokenizer


class MLP(nn.Module):
    def __init__(self, sizes: List[int], bias: bool = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x
    
class MPDocVQAConfig:
    def __init__(self, model_path: str, qwenvl_device_map, lora_config: dict = None, test_mode: bool = False, q_lora: bool = True, gradient_checkpointing: bool = False, freeze_modules: List[str] = ["transformer.visual"]):
        self.model_path = model_path
        self.qwenvl_device_map = qwenvl_device_map
        self.lora_config = lora_config
        self.test_mode = test_mode
        self.q_lora = q_lora
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_modules = freeze_modules

class PreMPDocVQAModel(PreTrainedModel):
    def __init__(self,config):
        super(PreMPDocVQAModel, self).__init__(config)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value

class MPDocVQAModel(PreMPDocVQAModel):
    def __init__(
        self,
        config,
    ):
        super(MPDocVQAModel, self).__init__()
        self.on_test_mode = config.test_mode
        self.config = QWenConfig.from_pretrained(config.model_path)
        if not config.test_mode:
            self.config.use_cache = False
        self.q_lora = config.q_lora
        self.gradient_checkpointing = config.gradient_checkpointing
        self.qwenvl: QWenLMHeadModelForDocVQA = (
            QWenLMHeadModelForDocVQA.from_pretrained(
                config.model_path,
                device_map = config.qwenvl_device_map, 
                config=self.config,
                quantization_config= GPTQConfig(bits=4, disable_exllama=True) if config.lora_config and config.q_lora else None
            )
        )
        self.tokenizer: QWenTokenizer = QWenTokenizer.from_pretrained(config.model_path)
        self.pad_id = self.tokenizer.eod_id
        self.freeze_params(config.freeze_modules)
        # if gradient_checkpointing and not lora_config:
        #     self.qwenvl.gradient_checkpointing_enable()

        if config.lora_config:
            self.lora_model(config.lora_config, config.q_lora, config.gradient_checkpointing)
        self.mlp = MLP([self.config.hidden_size, 512, 256, 1])

    def lora_model(self, lora_config, q_lora: bool, gradient_checkpointing):
        lora_config = LoraConfig(**lora_config)
        if q_lora:
            self.qwenvl = prepare_model_for_kbit_training(
                self.qwenvl, use_gradient_checkpointing=gradient_checkpointing
            )
        self.qwenvl = get_peft_model(self.qwenvl, lora_config)
        if gradient_checkpointing:
            self.qwenvl.enable_input_require_grads()

    def freeze_params(self, freeze_modules):
        for name, param in self.qwenvl.named_parameters():
            if any(freeze_module in name for freeze_module in freeze_modules):
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        cls_labels: torch.Tensor = None,
    ):
        outputs = self.qwenvl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        llm_loss = outputs.loss  # [bsz]
        hidden_states = outputs.hidden_states[-1]
        llm_embedding = self.get_last_one_hidden_states(hidden_states, input_ids)
        logits = self.mlp(llm_embedding)
        classification_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        cls_loss = classification_loss_fct(logits, cls_labels)  # [bsz]
        llm_loss_weight = (cls_labels == 1).long()
        loss = (llm_loss * llm_loss_weight + cls_loss).mean()

        loss = loss.mean()
        score = torch.sigmoid(logits)
        return loss, score

    def get_last_one_hidden_states(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ):
        bsz, _, _ = hidden_states.size()

        padding_on_left = False
        if self.on_test_mode:
            padding_on_left = True

        if padding_on_left:
            return hidden_states[:, -1, :].view(bsz, -1)
        else:
            device = input_ids.device
            temp_pad = torch.tensor([self.pad_id] * bsz).long().view(bsz, -1)  # [bsz,1]
            temp_pad = temp_pad.to(device)
            temp_input_ids = torch.cat([input_ids, temp_pad], dim=1)
            bsz_idx, last_idx = [], []
            for i in range(bsz):
                temp = temp_input_ids[i]
                bsz_idx.append(i)
                t = torch.nonzero(temp == self.pad_id, as_tuple=True)
                last_idx.append(t[0].min() - 1)

            bsz_idx = torch.tensor(bsz_idx).to(device)
            last_idx = torch.tensor(last_idx).to(device)

            return hidden_states[bsz_idx, last_idx, :]  # [bsz,hidden_size]

    def generate(self, *args, **kwargs):
        return self.qwenvl.generate(*args, **kwargs)

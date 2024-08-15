from typing import Any, Dict, List, Optional, Tuple
import sys

sys.path.append("/home/klwang/code/GuiQuQu-docvqa-vllm-inference/src")

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    GenerationConfig,
)
from transformers import GPTQConfig
from einops import rearrange, repeat

from Qwen_VL.modeling_qwen import (
    QWenLMHeadModelForDocVQA,
    QWenConfig,
    QWenModel,
    _SENTINEL,
    _ERROR_BAD_CHAT_FORMAT,
    _ERROR_STREAM_IN_CHAT,
)

from Qwen_VL.configuration_qwen import QWenConfig
from Qwen_VL.tokenization_qwen import QWenTokenizer
from Qwen_VL.qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)


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


class MPDocVQAConfig(PretrainedConfig):
    model_type = "mpdoc_vqa_model"

    def __init__(
        self,
        qwenvl_path: str = "Qwen-VL-Chat-Int4",
        qwenvl_device_map: str = "auto",
        freeze_modules: List[str] = ["transformer.visual"],
        use_lora: bool = False,
        use_q_lora: bool = False,
        **kwargs,
    ):
        super(MPDocVQAConfig, self).__init__(**kwargs)
        self.qwenvl_path = qwenvl_path
        self.qwenvl_device_map = qwenvl_device_map
        self.freeze_modules = freeze_modules
        self.qwenvl_config_path = f"{self.qwenvl_path}/config.json"
        self.use_lora = use_lora
        self.use_q_lora = use_q_lora


class PreMPDocVQAModel(PreTrainedModel):
    config_class = MPDocVQAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(PreMPDocVQAModel, self).__init__(config)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel) or isinstance(
            module, QWenLMHeadModelForDocVQA
        ):
            module.gradient_checkpointing = value


class MPDocVQAModel(PreMPDocVQAModel):
    def __init__(self, config: MPDocVQAConfig):
        super(MPDocVQAModel, self).__init__(config)
        # necessary part
        self.gradient_checkpointing = False
        # custom part
        qwenvl_config = QWenConfig.from_pretrained(config.qwenvl_path)
        self.qwenvl = QWenLMHeadModelForDocVQA.from_pretrained(
            config.qwenvl_path,
            config=qwenvl_config,
            device_map=config.qwenvl_device_map,
            quantization_config=(
                GPTQConfig(bits=4, disable_exllama=True)
                if config.use_lora and config.use_q_lora
                else None
            ),
        )

        self.tokenizer: QWenTokenizer = QWenTokenizer.from_pretrained(
            config.qwenvl_path
        )
        self.pad_id = self.tokenizer.eod_id
        self.freeze_params(config.freeze_modules)
        self.page_mlp = MLP([qwenvl_config.hidden_size, 512, 256, 1])

    def freeze_params(self, freeze_modules: List[str]):
        for name, param in self.qwenvl.named_parameters():
            for module in freeze_modules:
                if module in name:
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
            output_hidden_states=True,
            return_dict=True,
        )

        bsz, _ = input_ids.size()
        llm_loss = outputs.loss.view(bsz, -1)  # [bsz,seq_len]
        hidden_states = outputs.hidden_states[-1]
        llm_embedding = self.get_last_one_hidden_states(
            hidden_states, input_ids
        )  # [bsz,hidden_size]
        logits = self.page_mlp(llm_embedding)  # [bsz,1]
        score = torch.sigmoid(logits)
        loss = None
        if cls_labels is not None and labels is not None:
            classification_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            cls_labels = cls_labels.to(device=logits.device, dtype=logits.dtype)
            cls_loss = classification_loss_fct(
                logits, cls_labels.view(bsz, -1)
            )  # [bsz,1]
            llm_loss_weight = (cls_labels == 1).long().view(bsz, -1)  # [bsz,1]
            # llm_loss size: [bsz, seq_len]
            llm_loss_weight = repeat(
                llm_loss_weight, "b 1-> b (s 1)", s=llm_loss.size(1)
            )
            loss = (llm_loss * llm_loss_weight + cls_loss).mean()
            loss = loss.mean()
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

    def predict(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        generation_config: Optional[GenerationConfig],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Tuple[str, HistoryType]:
        generation_config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )

        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == "chatml", _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get("max_window_size", None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(
            get_stop_words_ids(generation_config.chat_format, tokenizer)
        )
        input_ids = torch.tensor([context_tokens]).to(self.device)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        # predict score
        _, score = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors="replace",
        )

        if append_history:
            history.append((query, response))

        return response, score, history


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_lora_model_for_train(
    model_config: MPDocVQAConfig, lora_config, use_q_lora, use_graditent_checkpointing
):
    base_model = MPDocVQAModel(model_config)
    lora_config = LoraConfig(**lora_config)
    if use_q_lora:
        # base_model.qwenvl = prepare_model_for_kbit_training(
        #     base_model.qwenvl, use_gradient_checkpointing=use_graditent_checkpointing
        # )
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=use_graditent_checkpointing
        )

    lora_model = get_peft_model(base_model, lora_config)
    if use_graditent_checkpointing:
        lora_model.enable_input_require_grads()

    return base_model


if __name__ == "__main__":
    config = MPDocVQAConfig(
        qwenvl_path="/home/klwang/pretrain-model/Qwen-VL-Chat-Int4",
        qwenvl_device_map="auto",
        freeze_modules=["transformer.visual"],
        use_lora=True,
        use_q_lora=True,
    )
    lora_config = {
        "r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["c_attn","attn.c_proj","w1","w2",],
        "task_type": "CAUSAL_LM",
        "modules_to_save": ["page_mlp"],
    }
    lora_model = get_lora_model_for_train(
        config, lora_config, use_q_lora=True, use_graditent_checkpointing=True
    )
    print(lora_model)

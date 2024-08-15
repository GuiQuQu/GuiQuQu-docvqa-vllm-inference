from typing import Any, Dict, List, Optional, Tuple
from torch import nn
import torch
from einops import repeat
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import GPTQConfig
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

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


# class MPDocVQAConfig(QWenConfig):
#     model_type = "qwen_for_mpdocvqa"

#     def __init__(
#         self,
#         model_path: str = "Qwen-VL-Chat-Int4",
#         qwenvl_device_map:str = "auto",
#         lora_config: dict = None,
#         test_mode: bool = False,
#         q_lora: bool = True,
#         gradient_checkpointing: bool = False,
#         freeze_modules: List[str] = ["transformer.visual"],
#         **kwargs,
#     ):
#         super(MPDocVQAConfig, self).__init__(**kwargs)
#         self.model_path = model_path
#         self.qwenvl_device_map = qwenvl_device_map
#         self.lora_config = lora_config
#         self.test_mode = test_mode
#         self.q_lora = q_lora
#         self.gradient_checkpointing = gradient_checkpointing
#         self.freeze_modules = freeze_modules


# class PreMPDocVQAModel(PreTrainedModel):
#     config_class = MPDocVQAConfig
#     supports_gradient_checkpointing = True

#     def __init__(self, config):
#         super(PreMPDocVQAModel, self).__init__(config)

#     def _set_gradient_checkpointing(self, module, value=False):
#         if isinstance(module, QWenModel):
#             module.gradient_checkpointing = value


def transform_precision(module: nn.Module, precision):
    if precision == "fp16":
        module.half()
    else:
        raise ValueError(f"Unsupported precision: {precision}")


class MPDocVQAModel(nn.Module):
    def __init__(
        self,
        qwenvl_model_path,
        qwenvl_device_map,
        on_test_mode,
        lora_config,
        q_lora,
        gradient_checkpointing,
        freeze_modules,
        precision="fp16",
    ):
        super(MPDocVQAModel, self).__init__()
        qwenvl_config = QWenConfig.from_pretrained(qwenvl_model_path)
        self.qwenvl = QWenLMHeadModelForDocVQA.from_pretrained(
            qwenvl_model_path,
            device_map=qwenvl_device_map,
            config=qwenvl_config,
            quantization_config=(
                GPTQConfig(bits=4, disable_exllama=True)
                if lora_config and q_lora
                else None
            ),
        )
        self.page_mlp = MLP([qwenvl_config.hidden_size, 512, 256, 1])
        self.on_test_mode = on_test_mode
        self.tokenizer: QWenTokenizer = QWenTokenizer.from_pretrained(qwenvl_model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.pad_token = self.tokenizer.eod_id
        # if gradient_checkpointing and not lora_config:
        #     self.qwenvl.gradient_checkpointing_enable()
        if lora_config:
            self.lora_model(lora_config, q_lora, gradient_checkpointing)

        self.freeze_params(freeze_modules)
        transform_precision(self, precision)

    def gradient_checkpointing_enable(self):
        self.qwenvl.gradient_checkpointing_enable()

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
            output_hidden_states=True,
            return_dict=True,
        )

        bsz, _ = input_ids.size()

        hidden_states = outputs.hidden_states[-1]
        llm_embedding = self.get_last_one_hidden_states(
            hidden_states, input_ids
        )  # [bsz,hidden_size]
        logits = self.page_mlp(llm_embedding)  # [bsz,1]
        score = torch.sigmoid(logits)
        loss = None
        if cls_labels is not None and labels is not None:
            llm_loss = outputs.loss.view(bsz, -1)  # [bsz,seq_len]
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

    # def generate(self, *args, **kwargs):
    #     return self.qwenvl.generate(*args, **kwargs)
    @torch.no_grad()
    def predict(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Tuple[str, float, HistoryType]:
        generation_config = (
            generation_config
            if generation_config is not None
            else self.qwenvl.generation_config
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
        input_ids = torch.tensor([context_tokens]).to(self.qwenvl.device)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        # predict score
        # score [bsz,1]
        _, score = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.qwenvl.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs,
        )
        # response:str
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
        score = score.view(-1).cpu().numpy().tolist()[0]
        return response, score, history


def mpdocvqa_model_inference(
    model: MPDocVQAModel,
    tokenizer,
    prompt: List[str] | str,
    max_new_tokens: int,
    max_length: int,
) -> Dict[str, List[Any] | Any]:
    resp_list, score_list = [], []
    is_str = False
    if isinstance(prompt, str):
        prompt = [prompt]
        is_str = True
    for p in prompt:
        resp, score, _ = model.predict(
            tokenizer,
            p,
            history=None,
            append_history=None,
            max_new_tokens=max_new_tokens,
        )
        # resp [bsz] score [bsz,1]
        resp_list.append(resp)
        score_list.append(score)

    return {
        "response": resp_list if not is_str else resp_list[0],
        "score": score_list if not is_str else score_list[0],
    }


def mpvqa_get_input_ids(prompt: str, tokenizer: PreTrainedTokenizer) -> List[int]:
    _, content_tokens = make_context(
        tokenizer,
        prompt,
        history=None,
        system="You are a helpful assistant.",
        chat_format="chatml",
    )
    return content_tokens

from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
import random
from typing import Dict, Optional, List
from functools import partial
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer

from transformers import Trainer, BitsAndBytesConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType


import handle_ocr
import template 

question_template = template.star_question_templatev3

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None


def truncate_layout(layout:str, 
                    tokenizer:PreTrainedTokenizer = None, 
                    max_token_length:int = 1024):
    if tokenizer == None:
        return layout
    lines = layout.split("\n")
    lines_input_ids = [tokenizer([l], return_tensors="pt").input_ids for l in lines]
    reserve_lines = []
    ids_cnt = 0
    for i, input_ids in enumerate(lines_input_ids):
        if ids_cnt + input_ids.size(-1) < max_token_length:
            ids_cnt += input_ids.size(-1)
            reserve_lines.append(lines[i])
        else: break
    return "\n".join(reserve_lines)

def get_layout_func(type:str):
    if type == "all-star":
        return partial(handle_ocr.sp_get_layout_by_json_path, placeholder="*")
    elif type == "lines":
        return handle_ocr.sp_get_lines_layout_by_json_path
    elif type == "words":
        return handle_ocr.sp_get_baseline_layout_by_json_path
    else:
        raise ValueError("Not support layout pattern")

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/klwang/pretrain-model/Qwen1.5-7B-Chat")


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/klwang/data/SPDocVQA/train_v1.0_withQT.json", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    ocr_dir: str = field(
        default="/home/klwang/data/SPDocVQA/ocr", metadata={"help": "Path to the OCR data."}
    )
    image_dir : str = field(
        default="/home/klwang/data/SPDocVQA/images", metadata={"help": "Path to the image data."}
    )
    layout_type: str = field(
        default="all-star",
        metadata={"help": "Layout type for the OCR data. Options: all-star, lines, words."},
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = True


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(messages: List[List[Dict[str,str]]], 
               tokenizer: transformers.PreTrainedTokenizer, 
               max_len: int) -> Dict:
    """
        Preprocesses the data for supervised fine-tuning.
        messages = [
            [{"role": "system", "content": "system message 1"}, {"role": "user", "content": "user message 1"}, {"role": "assistant", "content": "assistant message 1"}],
            [{"role": "system", "content": "system message 2"}, {"role": "user", "content": "user message 2"}, {"role": "assistant", "content": "assistant message 2"}],
            ]
    """
    roles = {"system": "<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start:List[int] = tokenizer('<|im_start|>').input_ids
    im_end:List[int] = tokenizer('<|im_end|>').input_ids
    nl_tokens:List[int] = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, target_ids = [], []
    for i, msg in enumerate(messages):
        input_id, target = [], []
        for j, sentence in enumerate(msg):
            if sentence["role"] == "system":
                system_input_id = im_start + _system + tokenizer(sentence["content"]).input_ids + im_end + nl_tokens
                input_id += system_input_id
                target += im_start + [IGNORE_TOKEN_ID] * (len(system_input_id) -3) + im_end + nl_tokens
                assert len(input_id) == len(target)
            elif sentence["role"] == "user":
                user_input_id = im_start + _user + tokenizer(sentence["content"]).input_ids + im_end + nl_tokens
                input_id += user_input_id
                target += im_start + [IGNORE_TOKEN_ID] * (len(user_input_id) -3) + im_end + nl_tokens
                assert len(input_id) == len(target)
            elif sentence["role"] == "assistant":
                assistant_input_id = im_start + _assistant + tokenizer(sentence["content"]).input_ids + im_end + nl_tokens
                input_id += assistant_input_id
                target += im_start + [IGNORE_TOKEN_ID] * len(_assistant) + assistant_input_id[len(_assistant)+len(im_start):-2] + im_end + nl_tokens
                assert len(input_id) == len(target)
            else:
                raise ValueError("role must be one of [system, user, assistant]")
        # 删除最后的nl_tokens
        # input_id = input_id[:-1]
        # target = target[:-1]
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        target_ids.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int) # [bs, max_len]
    target_ids = torch.tensor(target_ids, dtype=torch.int)
    return dict(
        input_ids=input_ids,
        target_ids=target_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

# def preprocess(
#     messages: list,
#     tokenizer: transformers.PreTrainedTokenizer,
#     max_len: int,
# ) -> Dict:
#     """Preprocesses the data for supervised fine-tuning."""

#     texts = []
#     for i, msg in enumerate(messages):
#         input_text = tokenizer.apply_chat_template( 
#                 msg,
#                 chat_template=TEMPLATE,
#                 tokenize=True,
#                 add_generation_prompt=False,
#                 padding=True,
#                 max_length=max_len,
#                 truncation=True,
#         )
#         texts.append(input_text)
#         print(msg)
#         print(texts[-1])
#     input_ids = torch.tensor(texts, dtype=torch.int)
#     target_ids = input_ids.clone()
#     target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
#     attention_mask = input_ids.ne(tokenizer.pad_token_id)

#     return dict(
#         input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
#     )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    system_message = "You are a helpful assistant."
    def __init__(
        self, raw_data, ocr_dir, image_dir, layout_func, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        """
            raw_data: list of dict,是直接获取的json数据
        """ 
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

        self.layout_func = layout_func
        self.ocr_dir = ocr_dir
        self.image_dir = image_dir

    def __len__(self):
        return len(self.raw_data)
    
    def prepare_stf_data(self, item):
        question = item["question"]
        answer = random.choice(item["answers"])
        ocr_path = os.path.join(self.ocr_dir, item["image"].split("/")[-1].split(".")[0] + ".json")
        layout = self.layout_func(json_path=ocr_path)
        layout = truncate_layout(layout, self.tokenizer, max_token_length=1024)
        prompt = question_template.format(layout = layout, question = question, answer = answer)
        messages = [
            {"role":"system", "content": self.system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
            ]
        return messages

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.prepare_stf_data(self.raw_data[i])], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
            question = self.raw_data[i]["question"],
            image = self.raw_data[i]["image"]
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f:
        train_data = json.load(f)["data"]

    train_dataset = dataset_cls(train_data,
                                ocr_dir=data_args.ocr_dir,
                                image_dir=data_args.image_dir,
                                layout_func= get_layout_func(data_args.layout_type),
                                tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 # WORD_SIZE > 1 means using ddp
    if lora_args.q_lora: # modify device_map for q-lora
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()
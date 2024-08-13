# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
from dataclasses import dataclass, field
import json
import math
import random
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
import logging
import utils
import template

from Qwen_VL.configuration_qwen import QWenConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

logger = logging.getLogger(__name__)


def logging_init(log_level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=log_level,
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/root/pretrain-model/Qwen-VL-Chat-Int4"
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/klwang/data/MPDocVQA/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    ocr_dir: str = field(
        default="/home/klwang/data/MPDocVQA/ocr",
        metadata={"help": "Path to the OCR data."},
    )
    image_dir: str = field(
        default="/home/klwang/data/MPDocVQA/images",
        metadata={"help": "Path to the image data."},
    )
    layout_type: str = field(
        default="all-star",
        metadata={
            "help": "Layout type for the OCR data. Options: all-star, lines, words, none."
        },
    )
    add_layout: bool = field(
        default=True,
        metadata={"help": "Whether to add layout information to the prompt."},
    )
    add_image: bool = field(
        default=True,
        metadata={"help": "Whether to add image information to the prompt."},
    )
    lazy_preprocess: bool = True
    max_doc_token_length: int = 1024


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = True
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "c_attn",
            "attn.c_proj",
            "w1",
            "w2",
        ]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# 修改了最后mlp的准入权限
def get_peft_state_maybe_zero_3(named_params, bias, to_save_names=[]):
    if bias == "none":
        to_return = {
            k: t
            for k, t in named_params
            if "lora_" in k or any([n in k for n in to_save_names])
        }
    elif bias == "all":
        to_return = {
            k: t
            for k, t in named_params
            if "lora_" in k or "bias" in k or any([n in k for n in to_save_names])
        }
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
            elif any([n in k for n in to_save_names]):
                to_return[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def rank0_info(*args):
    if local_rank == 0:
        logger.info(*args)


def rank0_warn(*args):
    if local_rank == 0:
        logger.warning(*args)


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


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    """
    messages = [
    [{"role": "system", "content": "system message 1"}, {"role": "user", "content": "user message 1"}, {"role": "assistant", "content": "assistant message 1"}],
    [{"role": "system", "content": "system message 2"}, {"role": "user", "content": "user message 2"}, {"role": "assistant", "content": "assistant message 2"}],
    ]
    """
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = [tokenizer.im_start_id]
    im_end = [tokenizer.im_end_id]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, target_ids = [], []
    for i, msg in enumerate(messages):
        input_id, target = [], []
        for j, sentence in enumerate(msg):

            def get_system_message(system_message):
                system_input_id = (
                    im_start
                    + _system
                    + tokenizer(system_message).input_ids
                    + im_end
                    + nl_tokens
                )
                system_target_id = (
                    im_start
                    + [IGNORE_TOKEN_ID] * (len(system_input_id) - 3)
                    + im_end
                    + nl_tokens
                )
                return system_input_id, system_target_id

            if j == 0 and sentence["role"] != "system":
                _input_id, _target = get_system_message(system_message)
                input_id += _input_id
                target += _target
                assert len(input_id) == len(target)

            if sentence["role"] == "system":
                _input_id, _target = get_system_message(sentence["content"])
                input_id += _input_id
                target += _target
                assert len(input_id) == len(target)
            elif sentence["role"] == "user":
                user_input_id = (
                    im_start
                    + _user
                    + tokenizer(sentence["content"]).input_ids
                    + im_end
                    + nl_tokens
                )
                input_id += user_input_id
                target += (
                    im_start
                    + [IGNORE_TOKEN_ID] * (len(user_input_id) - 3)
                    + im_end
                    + nl_tokens
                )
                assert len(input_id) == len(target)
            elif sentence["role"] == "assistant":
                assistant_input_id = (
                    im_start
                    + _assistant
                    + tokenizer(sentence["content"]).input_ids
                    + im_end
                    + nl_tokens
                )
                input_id += assistant_input_id
                target += (
                    im_start
                    + [IGNORE_TOKEN_ID] * len(_assistant)
                    + assistant_input_id[len(_assistant) + len(im_start) : -2]
                    + im_end
                    + nl_tokens
                )
                assert len(input_id) == len(target)
            else:
                raise ValueError("role must be one of [system, user, assistant]")

        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        target_ids.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    target_ids = torch.tensor(target_ids, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=target_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


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
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class MPDocVQALazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    system_message = "You are a helpful assistant."

    def __init__(
        self,
        raw_data,
        ocr_dir,
        image_dir,
        layout_func,
        question_template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        add_layout: bool = True,
        add_image: bool = True,
        max_doc_token_length: int = 1024,
    ):
        """
        raw_data: list of dict,是直接获取的json数据
        """
        super(MPDocVQALazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        # self.raw_data = raw_data
        self.init_data(raw_data)
        self.cached_data_dict = {}

        self.layout_func = layout_func
        self.ocr_dir = ocr_dir
        self.image_dir = image_dir
        self.max_doc_token_length = max_doc_token_length
        # config
        self.add_layout = add_layout
        self.add_image = add_image
        self.question_template = question_template

    def init_data(self, raw_data):
        data = []
        for item in raw_data:
            qid = item["questionId"]
            question = item["question"]
            answers = item["answers"]
            answer_page_idx = item["answer_page_idx"]
            for i, page_id in enumerate(item["page_ids"]):
                cls_label = int(answer_page_idx == i)
                new_item = dict(
                    qid=qid,
                    question=question,
                    answers=answers,
                    page_id=page_id,
                    cls_label=cls_label,
                )
            data.append(new_item)
        self.data = data

    def __len__(self):
        return len(self.data)

    def _prepare_prompt(self, question, image_path=None, layout=None):
        """
        根据是否添加layout和image来准备prompt
        """
        if self.add_image and self.add_layout:
            return self.question_template.format(
                image_path=f"<img>{image_path}</img>", layout=layout, question=question
            )
        elif self.add_image and not self.add_layout:
            return self.question_template.format(
                image_path=f"<img>{image_path}</img>", question=question
            )
        elif self.add_layout and not self.add_image:
            return self.question_template.format(layout=layout, question=question)
        else:
            return question

    def prepare_sft_data(self, item):
        question = item["question"]
        answer = random.choice(item["answers"])
        page_id = item["page_id"]
        ocr_path = os.path.join(self.ocr_dir, page_id + ".json")
        image_path = os.path.join(self.image_dir, page_id + ".jpg")
        layout = None
        if self.add_layout:
            layout = self.layout_func(json_path=ocr_path)
            layout = utils.truncate_layout(
                layout, self.tokenizer, self.max_doc_token_length
            )
        prompt = self._prepare_prompt(question, image_path, layout)
        message = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        return message

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        message = self.prepare_sft_data(self.data[i])
        ret = preprocess([message], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            question=self.data[i]["question"],
            page_id=self.data[i]["page_id"],
            qid=self.data[i]["qid"],
            cls_labels=self.data[i]["cls_label"],
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
        MPDocVQALazySupervisedDataset
        if data_args.lazy_preprocess
        else SupervisedDataset
    )
    rank0_print("Loading data...")

    def get_template(data_args):
        if data_args.add_layout and data_args.add_image:
            return template.vl_inference_template
        elif data_args.add_layout and not data_args.add_image:
            return template.star_question_templatev4
        elif not data_args.add_layout and data_args.add_image:
            return template.visual_question_template
        else:
            return ValueError("add_layout and add_image cannot be both False")

    train_data = utils.load_data(data_args.data_path)
    train_dataset = dataset_cls(
        train_data,
        ocr_dir=data_args.ocr_dir,
        image_dir=data_args.image_dir,
        layout_func=utils.mp_get_layout_func(data_args.layout_type),
        question_template=get_template(data_args),
        tokenizer=tokenizer,
        max_len=max_len,
        add_layout=data_args.add_layout,
        add_image=data_args.add_image,
        max_doc_token_length=data_args.max_doc_token_length,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
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
    logging_init(log_level=logging.INFO)
    if getattr(training_args, "deepspeed", None) and getattr(
        lora_args, "q_lora", False
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # # Set RoPE scaling factor
    # config = transformers.AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     trust_remote_code=True,
    # )
    # config.use_cache = False

    # # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     quantization_config=GPTQConfig(
    #         bits=4, disable_exllama=True
    #     )
    #     if training_args.use_lora and lora_args.q_lora
    #     else None,
    # )
    from models.docvqa_model import MPDocVQAModel, MPDocVQAConfig

    qwenvl_config: QWenConfig = QWenConfig.from_pretrained(model_args.model_name_or_path)
    if training_args.use_lora:
        config = MPDocVQAConfig(
            model_path=model_args.model_name_or_path,
            qwenvl_device_map=device_map,
            lora_config={
                "r": lora_args.lora_r,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "bias": lora_args.lora_bias,
                "target_modules": lora_args.lora_target_modules,
                "task_type": "CAUSAL_LM",
                "modules_to_save": None,
            },
            q_lora=lora_args.q_lora,
            gradient_checkpointing=training_args.gradient_checkpointing,
            freeze_modules=["transformer.visual"] if training_args.fix_vit else [],
            **qwenvl_config.to_dict(),
        )
        model = MPDocVQAModel(config=config)
    else:
        config = MPDocVQAConfig(
            model_path=model_args.model_name_or_path,
            device_map=device_map,
            lora_config=None,
            q_lora=False,
            gradient_checkpointing=training_args.gradient_checkpointing,
            freeze_modules=["transformer.visual"] if training_args.fix_vit else [],
            **qwenvl_config.to_dict(),
        )
        model = MPDocVQAModel(config=config)

    # if not training_args.use_lora:
    #     if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
    #         model.transformer.visual.requires_grad_(False)
    #         if hasattr(model.transformer.visual,'attn_pool'):
    #             model.transformer.visual.attn_pool.requires_grad_(True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()

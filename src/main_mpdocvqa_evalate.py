import logging

import torch
import transformers
from transformers import GenerationConfig
from transformers import GPTQConfig

from Qwen_VL.tokenization_qwen import QWenTokenizer
from models.docvqa_modelv3 import MPDocVQAModel
from models.docvqa_modelv3 import mpdocvqa_model_inference, mpvqa_get_input_ids
from mpdocvqa_evalator import (
    MPDocVQAEvaluator,
    MPDocVQADatasetForEvalWithLayoutAndImage,
)
from template import vl_inference_template
from utils import mp_get_layout_func,seed_everything

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

args = {
    "model": {
        "cpkt_path": "../output_mpdocvqav2_qwenvl_qlora_vl_inference_template/checkpoint-5/pytorch_model.bin",
        "qwenvl_path": "/home/klwang/pretrain-model/Qwen-VL-Chat-Int4",
        "lora_config": {
            "r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["c_attn", "attn.c_proj", "w1", "w2"],
            "task_type": "CAUSAL_LM",
            "modules_to_save": None,
        },
        "on_test_mode": True,
        "q_lora": True,
        "gradient_checkpointing": True,
        "freeze_modules": ["transformer.visual"],
        "device": "cuda:0",
        "precision": "fp16",
    },
    "data": {
        "json_data_path": "/home/klwang/data/MPDocVQA/val.json",
        "image_dir": "/home/klwang/data/MPDocVQA/images",
        "layout_dir": None,
        "ocr_dir": "/home/klwang/data/MPDocVQA/ocr",
        "question_template": vl_inference_template,
        "max_doc_token_cnt": 2048,
        "layout_type": "all-star",
        "tokenizer_path": "/home/klwang/pretrain-model/Qwen-VL-Chat-Int4",
    },
    "result_dir": "../result/qwenvl_mpdocvqa_ckpt5",
    "experiment_name": "ckpt5",
    "max_new_tokens": 128,
}


def load_model():
    model_args = args['model']
    model = MPDocVQAModel(
        qwenvl_model_path=model_args["qwenvl_path"],
        qwenvl_device_map="auto",
        lora_config=model_args["lora_config"],
        on_test_mode=model_args["on_test_mode"],
        q_lora=model_args["q_lora"],
        gradient_checkpointing=model_args["gradient_checkpointing"],
        freeze_modules=model_args["freeze_modules"],
    )

    model.qwenvl.use_cache = True
    model.qwenvl.generate_config = GenerationConfig.from_pretrained(model_args["qwenvl_path"])
    model.eval()
    model = model.to(device=model_args["device"])
    # load state dict
    state_dict = torch.load(model_args["cpkt_path"])
    for k,v in state_dict.items():
        if v.dtype == torch.float32:
            v = v.to(torch.float16)
        state_dict[k] = v
    model.load_state_dict(state_dict,strict=True)
    del state_dict
    model.to(device=model_args["device"])
    torch.cuda.empty_cache()
    logger.info(">>>>> Model loaded(%s)", model_args["cpkt_path"])

    return model


def load_tokenizer():
    return QWenTokenizer.from_pretrained(args['model']["qwenvl_path"])


def load_dataset():
    data_args = args["data"]
    tokenizer = QWenTokenizer.from_pretrained(data_args["tokenizer_path"])
    layout_func = mp_get_layout_func(data_args["layout_type"])
    data_args.pop("tokenizer_path")
    data_args.pop("layout_type")
    return MPDocVQADatasetForEvalWithLayoutAndImage(
        **data_args,
        tokenizer=tokenizer,
        layout_func=layout_func,
    )


if __name__ == "__main__":
    seed_everything(2024)
    transformers.utils.logging.set_verbosity_info()
    logger.info("Start to evaluate MPDocVQA model")
    logger.info(">>>>> Loading model and tokenizer")
    model = load_model()
    tokenizer = load_tokenizer()
    logger.info(">>>>> Loading dataset")
    dataset = load_dataset()
    evlator = MPDocVQAEvaluator(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        result_dir=args["result_dir"],
        experiment_name=args["experiment_name"],
        model_inference_fn=mpdocvqa_model_inference,
        get_input_ids_fn=mpvqa_get_input_ids,
        max_new_tokens=args["max_new_tokens"],
    )
    evlator.evaluate()

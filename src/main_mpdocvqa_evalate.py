import logging

from transformers import GenerationConfig

from models.docvqa_model import MPDocVQAModel
from models.docvqa_model import mpdocvqa_model_inference, mpvqa_get_input_ids
from mpdocvqa_evalator import (
    MPDocVQAEvaluator,
    MPDocVQADatasetForEvalWithLayoutAndImage,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

args = {
    "cpkt_path": "../output_mpdocvqa_qwenvl_qlora_vl_inference_template/checkpoint-1750",
    "qwenvl_path": "/home/klwang/pretrain-model/Qwen-VL-Chat-Int4",
}


def load_model():
    model = MPDocVQAModel.from_pretrained(args["cpkt_path"], device_map="auto")
    model.generate_config = GenerationConfig.from_pretrained(args["qwenvl_path"])
    model.use_cache = True
    model.eval()
    return model


if __name__ == "__main__":
    logger.info("Start to evaluate MPDocVQA model")

    logger.info(">>>>> Loading model")
    model = load_model()
    evlator = MPDocVQAEvaluator(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        result_dir="",
        experiment_name="",
        model_inference_fn=mpdocvqa_model_inference,
        get_input_ids_fn=mpvqa_get_input_ids,
        max_new_tokens=128,
    )
    evlator.evaluate()

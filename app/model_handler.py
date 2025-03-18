from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from app.config import config
from app.system_prompt import sys_prompt, prompt_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    try:
        logger.info("Loading: model and tokenizer...")
        model_name = config["model"]["name"]
        device = config["model"]["device"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        logger.info("Model and tokenizer loaded")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Model and tokenizer loading error: {e}")
        raise


def summarize_text(text: str, max_length: int = None) -> str:
    try:
        model, tokenizer = load_model()

        if max_length is None:
            max_length = config["generation"]["max_tokens"]

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_builder.format(messages_text=text)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        logger.error(f"LLM answer generation error: {e}")
        raise

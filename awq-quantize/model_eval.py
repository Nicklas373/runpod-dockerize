# Import required variables and libraries
import argparse
import torch
from pathlib import Path
from compressed_tensors.offload import load_offloaded_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)

# Check if the model is multimodal
def is_multimodal_model(model_id: str) -> bool:
    """
    Detect whether a model should be loaded with
    AutoModelForImageTextToText.

    This is more reliable than checking model names.
    """

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    architectures = config.architectures or []

    multimodal_keywords = (
        "ConditionalGeneration",
        "ImageTextToText",
        "Vision",
        "VL",
    )

    return any(
        any(keyword in arch for keyword in multimodal_keywords)
        for arch in architectures
    )

# Model evaluation function
def model_eval(
    model_id: str,
    trust_remote_code: bool,
): 
    is_mm_model = is_multimodal_model(model_id)

    ModelClass = (
        AutoModelForImageTextToText
        if is_mm_model
        else AutoModelForCausalLM
    )

    with load_offloaded_model():

        model = ModelClass.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            offload_folder="./offload_model",
        )

    # Prepare tokenizer
    if is_mm_model:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        processor = None

    # Ensure pad token exists (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable KV cache (saves VRAM during calibration)
    model.eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)

    prompt = "Explain what a neural network is in simple terms."

    if is_mm_model:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

    else:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, prompt_length:]

    if is_mm_model:
        response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]
    else:
        response = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

    return print(response)

# Main function to parse arguments and run the test
def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Evaluation Script"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to evaluate.")
    parser.add_argument("--trust_remote_code", type=bool, default=False, help=("Whether to trust and execute custom model code from the Hugging Face repository. Required for many community models."))
    args = parser.parse_args()

    model_eval(
        model_id=args.model_id,
        trust_remote_code=args.trust_remote_code,
     )

# Run the main function
if __name__ == "__main__":
        main()
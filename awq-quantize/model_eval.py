# Import required variables and libraries
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)

# Check if the model is multimodal
def is_multimodal_model(model_id: str) -> bool:
    keywords = keywords = [
        "apriel",
        "mistral",
        "ministral",
        "ministral3",
        "kimivl",
        "qwen3-vl",
        "qwen3.5"
    ] # Add more keywords as needed
    return any(k in model_id.lower() for k in keywords)

# Model evaluation function
def model_eval(
    model_id: str,
    trust_remote_code: bool,
): 
    is_mm_model = is_multimodal_model(model_id)
    
    if is_mm_model:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            device_map="cuda",
        )

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token exists (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable KV cache (saves VRAM during calibration)
    model.eval()
    model.config.use_cache = False

    if is_mm_model:
        inputs = processor(
            images=None,
            text="Hello, can you explain yourself ? ",
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = tokenizer(
            "Hello, can you explain yourself ? ",
            return_tensors="pt"
        ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=50,
    )

    return print(tokenizer.decode(out[0], skip_special_tokens=True))

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
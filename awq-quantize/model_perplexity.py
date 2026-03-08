import argparse
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from tqdm import tqdm

# --------------------------------------------------
# Multimodal Model Variables & Functions
# --------------------------------------------------
def is_multimodal_model(model_id: str) -> bool:
    keywords = [
        "apriel",
        "mistral",
        "ministral",
        "ministral3",
        "kimivl",
        "qwen3-vl",
        "qwen3.5"
    ] # Add more keywords as needed
    return any(k in model_id.lower() for k in keywords)

# ------------------------------------------------
# 1. Load AWQ model + tokenizer
# ------------------------------------------------
def run_perplexity(model_id: str):
    # Initialize multimodal model flag
    is_mm_model = is_multimodal_model(model_id)

    # Step 2: Manually load the model with the trust flag
    print(f"Loading model from {model_id}...")
    if is_mm_model:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="cuda",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="cuda",
        )
    
    # Load tokenizer and model with offloading
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable KV cache (saves VRAM during calibration)
    model.eval()
    model.config.use_cache = False

    # ------------------------------------------------
    # 2. Load WikiText-2 (benchmark standard)
    # ------------------------------------------------
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    # ------------------------------------------------
    # 3. Tokenize & concatenate
    # ------------------------------------------------
    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    all_input_ids = []
    for ids in tokenized["input_ids"]:
        all_input_ids.extend(ids)

    # ------------------------------------------------
    # 4. Chunk into fixed blocks
    # ------------------------------------------------
    block_size = 2048   # try 4096 if it fits

    blocks = []
    for i in range(0, len(all_input_ids) - block_size, block_size):
        blocks.append(all_input_ids[i : i + block_size])

    # Optional: quick sanity check
    # blocks = blocks[:20]

    # ------------------------------------------------
    # 5. Compute benchmark perplexity
    # ------------------------------------------------
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for block in tqdm(blocks):
            input_ids = torch.tensor(
                block,
                device=model.device,
            ).unsqueeze(0)

            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
            )

            loss = outputs.loss           # mean NLL per token
            n_tokens = input_ids.numel()

            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    ppl = math.exp(total_nll / total_tokens)
    return f"WikiText-2 Perplexity: {ppl:.3f}"

def main():
    parser = argparse.ArgumentParser(
        description="AWQ 4‑bit quantization by using LLM-Compressor"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to download and evaluate")
    args = parser.parse_args()
    
    result = run_perplexity(args.model_id)
    print(result)

# Run the main function
if __name__ == "__main__":
        main()
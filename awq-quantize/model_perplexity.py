import argparse
import math
import torch
from compressed_tensors.offload import load_offloaded_model
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from tqdm import tqdm


# --------------------------------------------------
# Detect multimodal model
# --------------------------------------------------
def is_multimodal_model(model_id: str) -> bool:
    keywords = [
        "apriel",
        "mistral",
        "ministral",
        "ministral3",
        "kimivl",
        "qwen3-vl",
        "qwen3.5",
        "vl"
    ]
    return any(k in model_id.lower() for k in keywords)


# --------------------------------------------------
# Load model + tokenizer
# --------------------------------------------------
def load_model(model_id: str):

    is_mm_model = is_multimodal_model(model_id)

    print(f"Loading model from {model_id}...")

    if is_mm_model:
        with load_offloaded_model():
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
                offload_folder="./offload_model",
            )
    else:
        with load_offloaded_model():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
                offload_folder="./offload_model",
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)

    return model, tokenizer


# --------------------------------------------------
# Compute perplexity
# --------------------------------------------------
def run_perplexity(model_id: str, block_size: int, max_blocks: int):

    model, tokenizer = load_model(model_id)

    # ------------------------------------------------
    # Load WikiText-2
    # ------------------------------------------------
    print("Loading WikiText-2 dataset...")

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    # ------------------------------------------------
    # Tokenize
    # ------------------------------------------------
    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # ------------------------------------------------
    # Concatenate tokens
    # ------------------------------------------------
    all_input_ids = []

    for ids in tokenized["input_ids"]:
        all_input_ids.extend(ids)

    # ------------------------------------------------
    # Create blocks
    # ------------------------------------------------
    blocks = []

    for i in range(0, len(all_input_ids) - block_size, block_size):
        blocks.append(all_input_ids[i : i + block_size])

    if max_blocks > 0:
        blocks = blocks[:max_blocks]

    print(f"Total blocks: {len(blocks)}")

    # ------------------------------------------------
    # Perplexity calculation
    # ------------------------------------------------
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():

        for block in tqdm(blocks):

            input_ids = torch.tensor(
                block,
                device=model.device
            ).unsqueeze(0)

            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            # correct token count (ignore first token)
            n_tokens = input_ids.size(1) - 1

            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    ppl = math.exp(total_nll / total_tokens)

    print(f"\nWikiText-2 Perplexity: {ppl:.3f}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Perplexity benchmark for LLM/VLM models"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Token block size",
    )

    parser.add_argument(
        "--max_blocks",
        type=int,
        default=0,
        help="Limit number of blocks (0 = full dataset)",
    )

    args = parser.parse_args()

    run_perplexity(
        args.model_id,
        args.block_size,
        args.max_blocks,
    )


if __name__ == "__main__":
    main()
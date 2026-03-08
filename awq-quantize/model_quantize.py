# AWQ Quantize Script using LLM-Compressor
# Reference: https://github.com/xhedit/quantkit/

import argparse
from compressed_tensors.offload import dist_init, offloaded_model
import os, os.path
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from io import BytesIO
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from pathlib import Path
from PIL import Image
import requests
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)

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

def detect_image_column(dataset):
    for col in dataset.column_names:
        if "image" in col.lower() or "img" in col.lower():
            return col
    return None

def load_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        if image.startswith("http"):
            return Image.open(BytesIO(requests.get(image).content)).convert("RGB")
        return Image.open(image).convert("RGB")
    raise ValueError("Unsupported image format")

def detect_text_column(dataset):
    for col in dataset.column_names:
        if col in ["text", "content", "prompt", "messages"]:
            return col
    return dataset.column_names[0]

# --------------------------------------------------
# 1. Download & Prepare Model Directory
# --------------------------------------------------
def get_model_path(
        branch: str,
        force_download: bool,
        hf_cache: bool,
        model_id: str,
):
    path = Path(model_id)

    if path.is_dir() and (path / "config.json").is_file():
        return path

    folder_name = model_id.split("/")[-1]
    local_path = Path(folder_name)

    print(f"Downloading {model_id} to {local_path}...")
    snapshot_download(
        repo_id=model_id,
        revision=branch,
        local_dir=local_path,
        local_dir_use_symlinks=hf_cache,
        force_download=force_download
    )
    return local_path

# --------------------------------------------------
# 2. Main Quantization Logic
# --------------------------------------------------
def run_awq_quantization(
        branch: str,
        dataset_id,
        dataset_config,
        dataset_mix,
        dataset_split: str,
        hf_cache: bool,
        max_seq_length: int,
        num_samples: int,
        model_id: str,
        text_column: str,
        trust_remote_code: bool,
        trust_remote_code_model: bool,
):
    # Step 1: Download/Verify local model
    model_path = get_model_path(branch, False, hf_cache, model_id)
    is_mm_model = is_multimodal_model(model_id)
   
    # Make sure offload cache directory exists
    if not os.path.exists("./offload_cache"):
        os.mkdir("./offload_cache")

    # Debug info
    print(f"Multimodal model detected: {is_mm_model}")

    # Initialize distributed process group
    dist_init()

    # Step 2: Manually load the model with the trust flag
    print(f"Loading model from {model_path}...")
    if is_mm_model:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            device_map="cuda",
            offload_folder="./offload_cache",
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            device_map="cuda",
            offload_folder="./offload_cache",
        )

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token exists (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable KV cache (saves VRAM during calibration)
    with offloaded_model(): # Enables model CT Offloading for quantization
        model.eval()
        model.config.use_cache = False

    # Step 2: Prepare Custom Dataset
    all_samples = []

    dataset_ids = dataset_id.split(",")

    # Parse dataset configs (support multiple)
    dataset_configs = (
        dataset_config.split(",") if dataset_config else [None] * len(dataset_ids)
    )

    assert len(dataset_configs) == len(dataset_ids), \
        "Number of dataset configs must match dataset_ids"

    if dataset_mix:
        ratios = [float(x) for x in dataset_mix.split(",")]
        assert len(ratios) == len(dataset_ids)
    else:
        ratios = [1 / len(dataset_ids)] * len(dataset_ids)

    for dataset_id, dataset_config, ratio in zip(dataset_ids, dataset_configs, ratios):
        sample_count = max(1, int(num_samples * ratio))
        print(f"Loading dataset: {dataset_id} (ratio={ratio}, samples={sample_count})")
        
        # Load only a slice to avoid downloading full datasets
        if dataset_id in ["allenai/c4"]:
            ds = load_dataset(
                dataset_id,
                dataset_config,
                split=dataset_split,
                streaming=True
            )
            ds = Dataset.from_list(list(ds.take(sample_count)))
        else:
            if dataset_config:
                ds = load_dataset(
                    dataset_id,
                    dataset_config,
                    split=f"{dataset_split}[:{sample_count}]"
                )
            else:
                ds = load_dataset(
                    dataset_id,
                    split=f"{dataset_split}[:{sample_count}]"
                )

        # Detect multimodal dataset
        image_column = detect_image_column(ds)
        is_mm_ds = image_column is not None

        text_column = detect_text_column(ds)
        print("Detected text column:", text_column)

        # Filter empty text rows (important for datasets like WikiText)
        ds = ds.filter(lambda x: x.get(text_column) and str(x[text_column]).strip() != "")
        
        # Shuffle and cap samples
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(min(sample_count, len(ds))))

        for row in ds:
            if is_mm_ds:
                image = load_image(row[image_column])
                text = str(row[text_column])

                inputs = processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                )

                all_samples.append({
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "pixel_values": inputs["pixel_values"][0],
                })
            else:
                content = row[text_column]
                if isinstance(content, list):
                    text = tokenizer.apply_chat_template(
                        content,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    text = str(content)

                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_seq_length,
                    return_tensors="pt",
                )

                all_samples.append({
                    "input_ids": tokens["input_ids"][0],
                    "attention_mask": tokens["attention_mask"][0],
                })

    # Final calibration dataset
    calibration_dataset = Dataset.from_list(all_samples).shuffle(seed=42)

    # Step 3: Define Recipe (Standard 4-bit AWQ)
    ignore_modules = (
            "re:.*model.embed_tokens",
            "re:.*model.norm",
            "re:.*lm_head",
        )
    
    if is_mm_model:
        ignore_modules += (
            "re:.*vision_tower.*",
            "re:.*vision_encoder.*",
            "re:.*multi_modal_projector.*",
            "re:model[.]visual.*",
        )

    recipe = [
        SmoothQuantModifier(
            smoothing_strength=0.6,
        ),
        AWQModifier(
            targets=["Linear"],
            ignore=ignore_modules,
            config_groups={
                "main": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 32,
                        "observer": "mse",
                        "dynamic": False,
                    },
                }
            },
        )
    ]

    # Step 4: Run Oneshot Quantization
    output_dir = f"{model_path.name}-AWQ"

    print(f"Starting AWQ quantization. Output: {output_dir}")
    oneshot(
        model=model, # Pass the local directory string
        dataset=calibration_dataset,
        recipe=recipe,
        num_calibration_samples=(len(calibration_dataset) if calibration_dataset else None),
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        trust_remote_code_model=trust_remote_code_model,
    )

    print(f"Success! Quantized model saved to {output_dir}")

# --------------------------------------------------
# 3. Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AWQ 4‑bit quantization by using LLM-Compressor"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to download and quantize.")
    parser.add_argument("--dataset_id", required=True, help="Comma-separated dataset IDs")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name (if applicable).")
    parser.add_argument("--dataset_mix", default=None, help="Comma-separated ratios, e.g. 0.7,0.3")
    parser.add_argument("--dataset_split", type=str, default="train", help=("Which split of the dataset to use for AWQ calibration (e.g. 'train', 'validation'). Ignored if --dataset_id is not provided."))
    parser.add_argument("--text_column", type=str, default=None, help=("Name of the column containing text data in the dataset. Required when --dataset_id is provided. If the column contains a list of messages, their 'content' fields ""will be concatenated"))
    parser.add_argument("--num_samples", type=int, default=512, help=("Number of samples used for AWQ calibration. More samples can improve accuracy but require more VRAM and time. Typical values: 128 to 512."))
    parser.add_argument("--max_seq_length", type=int, default=2048, help=("Maximum sequence length (in tokens) used during calibration. Longer sequences improve weight calibration for long-context models but increase VRAM usage."))
    parser.add_argument("--hf_cache", type=bool, default=False, help=( "Whether to use Hugging Face cache symlinks when downloading the model. Enable if you want to reuse cached files; disable for fully local copies."))
    parser.add_argument("--branch", type=str, default="main", help=( "Model repository branch or revision to download from Hugging Face (e.g. 'main', 'fp16', 'bf16')."))
    parser.add_argument("--trust_remote_code", type=bool, default=False, help=("Whether to trust and execute custom model code from the Hugging Face repository. Required for many community models."))
    parser.add_argument("--trust_remote_code_model", type=bool, default=False, help=("Whether to trust and execute custom model code when loading the model. Required for many community models."))
    args = parser.parse_args()

    run_awq_quantization(
        branch=args.branch,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_mix=args.dataset_mix,
        dataset_split=args.dataset_split,
        hf_cache=args.hf_cache,
        max_seq_length=args.max_seq_length,
        model_id=args.model_id,
        num_samples=args.num_samples,
        text_column=args.text_column,
        trust_remote_code=args.trust_remote_code,
        trust_remote_code_model=args.trust_remote_code_model,
     )

# Example Usage
if __name__ == "__main__":
        main()
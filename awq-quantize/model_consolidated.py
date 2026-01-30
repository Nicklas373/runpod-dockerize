import argparse
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

def consolidate(output_dir):
    output_dir = Path(output_dir)
    shards = sorted(output_dir.glob("model-*.safetensors"))
    tensors = {}

    for shard in shards:
        shard_tensors = load_file(shard)
        for k, v in shard_tensors.items():
            if k in tensors:
                raise ValueError(f"Duplicate tensor key: {k}")
            tensors[k] = v

    save_file(tensors, output_dir / "consolidated.safetensors")

def consolidated_conf(output_dir):
    output_dir = Path(output_dir)
    ckpt = output_dir / "consolidated.safetensors"
    tensors = load_file(ckpt)

    index = {
        "metadata": {"total_size": ckpt.stat().st_size},
        "weight_map": {k: "consolidated.safetensors" for k in tensors}
    }

    with open(output_dir / "consolidated.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Consolidation script for Mistral models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to already-quantized HF model directory"
    )
    args = parser.parse_args()

    consolidate(args.model_path)
    consolidated_conf(args.model_path)

if __name__ == "__main__":
    main()
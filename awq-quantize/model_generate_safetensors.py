
import argparse
import json
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import snapshot_download

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

def generate_safetensors_index(model_dir: Path):
    """
    Generate a safetensors index file for the model directory.
    """

    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} is not a directory.")

    if not any(model_dir.glob("*.safetensors")):
        raise ValueError(f"No .safetensors files found in {model_dir}.")

    weight_map = {}
    total_size = 0

    for shard in sorted(model_dir.glob("*.safetensors")):
        print(f"Scanning {shard.name}")

        total_size += shard.stat().st_size

        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name

    index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }

    output_file = model_dir / "model.safetensors.index.json"

    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved: {output_file}")
    print(f"Tensors: {len(weight_map)}")

def main():
    parser = argparse.ArgumentParser(description="Generate safetensors index for a model directory.")
    parser.add_argument("model_dir", type=str, help="Path to the model directory.")
    args = parser.parse_args()

    model_dir =  get_model_path("main", False, True, args.model_dir)
    generate_safetensors_index(model_dir)

if __name__ == "__main__":
    main()
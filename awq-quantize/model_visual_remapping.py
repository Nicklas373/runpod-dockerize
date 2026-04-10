import argparse
import json
import safetensors.torch as st
from pathlib import Path

def remapping(model_path: str):
    model_dir = Path(model_path)

    print("Begin visual structure re-mapping for: {model_path}" )

    for file in model_dir.glob("*.safetensors"):
        ckpt = st.load_file(file)
        new_ckpt = {}

        for k, v in ckpt.items():
            new_k = k.replace("language_model.visual", "visual")
            new_ckpt[new_k] = v

        st.save_file(new_ckpt, file)

    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        new_weight_map = {}
        for k, v in index["weight_map"].items():
            new_k = k.replace("language_model.visual", "visual")
            new_weight_map[new_k] = v

        index["weight_map"] = new_weight_map

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    print("Done visual structure re-mapping for: {model_path}" )

def main():
    parser = argparse.ArgumentParser(
        description="Model Visual Re-mapping"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path",
    )

    args = parser.parse_args()

    remapping(
        args.model_path
    )

if __name__ == "__main__":
    main()
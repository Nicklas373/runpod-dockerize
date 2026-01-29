import os
import argparse
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    parser = argparse.ArgumentParser(
        description="Huggingface Model Uploader"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, HF_TOKEN env var is used."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target repository (e.g. nicklas373/Qwen3-8B-AWQ)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Local directory containing model files"
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private"
    )

    args = parser.parse_args()

    # -------------------------
    # HF Token Validation
    # -------------------------
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF token not provided. Use --hf_token or set HF_TOKEN env var."
        )

    # -------------------------
    # HF Login
    # -------------------------
    login(token=token)
    api = HfApi()

    # -------------------------
    # Create HF Repo if not exists
    # -------------------------
    try:
        api.repo_info(args.repo_id, repo_type=args.repo_type)
        print(f"Repo exists: {args.repo_id}")
    except RepositoryNotFoundError:
        print(f"Creating repo: {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
        )

    # -------------------------
    # Upload to HF
    # -------------------------
    print(f"Uploading from {args.local_dir} → {args.repo_id}")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message="Upload model to HF using HuggingFace API",
    )

    print(f"{args.repo_id} uploaded to HF !")

if __name__ == "__main__":
    main()

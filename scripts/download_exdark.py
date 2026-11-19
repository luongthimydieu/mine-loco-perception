# -*- coding: utf-8 -*-
"""Download ExDark (YOLO format) from Hugging Face.

ExDark is a public low-light dataset released under CC BY 4.0. This script
only downloads the ``data/`` split (images + labels); dust degradation is
synthesized locally by the training scripts.

Usage:
    python scripts/download_exdark.py --dest data/exdark
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_exdark.py
"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/exdark",
                    help="destination directory (default: data/exdark)")
    ap.add_argument("--repo", default="dronefreak/ExDark")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Please install huggingface_hub: pip install huggingface_hub")

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading", args.repo, "->", dest)
    if os.environ.get("HF_ENDPOINT"):
        print("HF_ENDPOINT =", os.environ["HF_ENDPOINT"])
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["data/*"],
    )

    for split in ("train", "valid"):
        img = dest / "data" / split / "images"
        lab = dest / "data" / split / "labels"
        n_img = len(list(img.glob("*"))) if img.exists() else 0
        n_lab = len(list(lab.glob("*"))) if lab.exists() else 0
        print(f"  {split:5s}: {n_img:5d} images, {n_lab:5d} labels")
    print("Done. Expected layout: data/exdark/data/{train,valid}/{images,labels}")


if __name__ == "__main__":
    main()

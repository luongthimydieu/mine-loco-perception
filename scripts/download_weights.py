# -*- coding: utf-8 -*-
"""Download a COCO-pretrained YOLO checkpoint into weights/.

Usage:
  python scripts/download_weights.py                # yolov8n.pt
  python scripts/download_weights.py --model yolov8s.pt
"""
import argparse
import os

from mine_loco_system.paths import WEIGHTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.chdir(WEIGHTS_DIR)
    from ultralytics import YOLO
    YOLO(args.model)
    path = os.path.join(WEIGHTS_DIR, args.model)
    if os.path.exists(path):
        print("[ok]", path, os.path.getsize(path) // 1024, "KB")
    else:
        print("[warn] download finished but file not found at", path)


if __name__ == "__main__":
    main()

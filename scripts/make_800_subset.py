# -*- coding: utf-8 -*-
"""Create the 800-image ExDark subset used by the small-scale experiment.

Copies the first N images and labels of each split:
  data/exdark/data/{train,valid}  ->  data/exdark800/data/{train,valid}

Usage:
  python scripts/make_800_subset.py
  python scripts/make_800_subset.py --n-train 800 --n-valid 737
"""
import argparse
import shutil

from mine_loco_system.paths import EXDARK_DATA, EXDARK800_DATA


def copy_split(split, n):
    for kind in ("images", "labels"):
        src = EXDARK_DATA / split / kind
        dst = EXDARK800_DATA / split / kind
        if not src.is_dir():
            print("[missing]", src)
            continue
        files = sorted(src.glob("*"))[:n]
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f.name)
        print("  %-5s %-6s %4d files -> %s" % (split, kind, len(files), dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-valid", type=int, default=737)
    args = ap.parse_args()
    print("source:", EXDARK_DATA)
    copy_split("train", args.n_train)
    copy_split("valid", args.n_valid)
    print("done:", EXDARK800_DATA)


if __name__ == "__main__":
    main()

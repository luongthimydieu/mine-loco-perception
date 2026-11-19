# -*- coding: utf-8 -*-
"""Repository integrity check: code, reference JSONs, site assets, optional data/weights.

Usage:
  python scripts/verify_workspace.py            # code + references + site
  python scripts/verify_workspace.py --strict   # also require data/ and weights/
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
WEIGHTS = os.path.join(ROOT, "weights")
REFS = os.path.join(ROOT, "assets", "reference")
SITE = os.path.join(ROOT, "site")

checks = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print("%s  %-42s %s" % ("PASS" if ok else "FAIL", name, detail))


def count(p):
    return sum(len(f) for _, _, f in os.walk(p)) if os.path.isdir(p) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    ck("package src", os.path.isdir(os.path.join(ROOT, "src", "mine_loco_system")))
    ck("main.py", os.path.isfile(os.path.join(ROOT, "src", "mine_loco_system", "main.py")))
    ck("paths.py", os.path.isfile(os.path.join(ROOT, "src", "mine_loco_system", "paths.py")))
    ck("scripts", os.path.isdir(os.path.join(ROOT, "scripts")))
    ck("tests", os.path.isdir(os.path.join(ROOT, "tests")))
    ck("site index", os.path.isfile(os.path.join(SITE, "index.html")))
    ck("site demo video", os.path.isfile(os.path.join(SITE, "assets", "demo.mp4")))
    for f in ("simulation_metrics.json", "scaleup_metrics.json",
              "extension_eval.json", "benchmark_inference.json", "leak_diagnostic.json",
              "per_class.json", "threshold_sweep.json"):
        ck("reference " + f, os.path.isfile(os.path.join(REFS, f)))

    exdark = os.path.join(DATA, "exdark", "data")
    if os.path.isdir(exdark):
        n_tr = count(os.path.join(exdark, "train", "images"))
        n_va = count(os.path.join(exdark, "valid", "images"))
        ck("ExDark train", n_tr >= 800, str(n_tr))
        ck("ExDark valid", n_va >= 100, str(n_va))
    else:
        ck("ExDark data (optional)", not args.strict,
           "run scripts/download_exdark.py" if args.strict else "not downloaded")

    yolo = os.path.join(WEIGHTS, "yolov8n.pt")
    ck("yolov8n.pt (optional)", os.path.isfile(yolo) or not args.strict,
       "place in weights/" if not os.path.isfile(yolo) else "")

    failed = [n for n, ok, _ in checks if not ok]
    print("=" * 78)
    print("%d checks, %d failed" % (len(checks), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Per-class AP analysis for the scale-up models.

For each model (M_clean / M_phys / M_noise / M_hsv) and evaluation set
(clean / V=20 / V=10), run Ultralytics validation and export per-class AP@0.5.

Outputs:
  outputs/results/per_class.json
  outputs/results/per_class.csv
  outputs/figures/fig11_per_class_v20.png
"""
import argparse
import contextlib
import csv
import io
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from ultralytics import YOLO

from mine_loco_system.paths import CASES_DIR, RESULTS_DIR, FIGURES_DIR

EVALS = ["clean_eval", "dustV20_eval", "dustV10_eval"]
MODELS = ["M_clean", "M_phys", "M_noise", "M_hsv"]


def quiet_val(model, yaml, imgsz=640):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=yaml, imgsz=imgsz, conf=0.001, iou=0.6,
                      verbose=False, plots=False, workers=0, device=0)
    names = [r.names[i] for i in range(len(r.names))]
    ap50 = [round(float(x) * 100, 2) for x in getattr(r.box, "ap50", [])]
    ap = [round(float(x) * 100, 2) for x in getattr(r.box, "maps", [])]
    return {
        "mAP50": round(float(r.box.map50) * 100, 3),
        "mAP50_95": round(float(r.box.map) * 100, 3),
        "P": round(float(r.box.mp) * 100, 3),
        "R": round(float(r.box.mr) * 100, 3),
        "classes": names,
        "ap50": ap50,
        "ap50_95": ap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES_DIR / "_cases_scaleup"))
    ap.add_argument("--scale", default=str(RESULTS_DIR / "results_scaleup"))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--fig", default=str(FIGURES_DIR))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig, exist_ok=True)
    results = {}
    for m in MODELS:
        w = os.path.join(args.scale, m, "weights", "best.pt")
        if not os.path.exists(w):
            print("[missing]", w)
            continue
        model = YOLO(w)
        results[m] = {}
        for ev in EVALS:
            y = os.path.join(args.cases, ev, "data.yaml")
            if not os.path.exists(y):
                print("[missing]", y)
                continue
            results[m][ev] = quiet_val(model, y)
            print("%-8s %-12s mAP50=%6.2f" % (m, ev, results[m][ev]["mAP50"]))

    with open(os.path.join(args.out, "per_class.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "per_class.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval", "class", "AP50", "AP50_95"])
        for m, evs in results.items():
            for ev, d in evs.items():
                for i, c in enumerate(d["classes"]):
                    w.writerow([m, ev, c, d["ap50"][i] if i < len(d["ap50"]) else "",
                                d["ap50_95"][i] if i < len(d["ap50_95"]) else ""])
    print("[写出]", os.path.join(args.out, "per_class.json"))

    make_fig(results, args.fig)


def make_fig(results, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("Microsoft YaHei", "SimHei"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
        except Exception:
            pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 300

    if "M_clean" not in results or "dustV20_eval" not in results["M_clean"]:
        return
    d0 = results["M_clean"]["dustV20_eval"]
    d1 = results.get("M_phys", {}).get("dustV20_eval")
    if not d1:
        return
    names = d0["classes"]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, d0["ap50"], width, label="M_clean", color="tab:blue")
    ax.bar(x + width / 2, d1["ap50"], width, label="M_phys", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("AP@0.5 (%)")
    ax.set_title("Per-class AP@0.5 at V=20 m (368-image independent set)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig11_per_class_v20.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[图]", out)


if __name__ == "__main__":
    main()

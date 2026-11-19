# -*- coding: utf-8 -*-
"""Confidence-threshold sweep for deployment.

For M_clean and M_phys on clean / V=20 / V=10, evaluate precision, recall and
F1 at several confidence thresholds. mAP is threshold-free; this table is for
choosing an operating point.

Outputs:
  outputs/results/threshold_sweep.json
  outputs/results/threshold_sweep.csv
  outputs/figures/fig12_threshold_f1.png
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
MODELS = ["M_clean", "M_phys"]
THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


def quiet_val(model, yaml, conf, imgsz=640):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=yaml, imgsz=imgsz, conf=conf, iou=0.6,
                      verbose=False, plots=False, workers=0, device=0)
    p = float(r.box.mp) * 100
    rec = float(r.box.mr) * 100
    f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
    return {"P": round(p, 3), "R": round(rec, 3), "F1": round(f1, 3),
            "mAP50": round(float(r.box.map50) * 100, 3)}


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
                continue
            results[m][ev] = []
            for t in THRESHOLDS:
                r = quiet_val(model, y, t)
                r["conf"] = t
                results[m][ev].append(r)
            best = max(results[m][ev], key=lambda x: x["F1"])
            print("%-8s %-12s best F1=%.2f @ conf=%.2f (P=%.1f R=%.1f)" %
                  (m, ev, best["F1"], best["conf"], best["P"], best["R"]))

    with open(os.path.join(args.out, "threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "threshold_sweep.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval", "conf", "P", "R", "F1", "mAP50"])
        for m, evs in results.items():
            for ev, rows in evs.items():
                for r in rows:
                    w.writerow([m, ev, r["conf"], r["P"], r["R"], r["F1"], r["mAP50"]])
    print("[写出]", os.path.join(args.out, "threshold_sweep.json"))

    make_fig(results, args.fig)


def make_fig(results, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 300
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    labels = {"clean_eval": "Clean", "dustV20_eval": "V=20 m",
              "dustV10_eval": "V=10 m"}
    colors = {"M_clean": "tab:blue", "M_phys": "tab:red"}
    for ax, ev in zip(axes, EVALS):
        for m in MODELS:
            if m not in results or ev not in results[m]:
                continue
            xs = [r["conf"] for r in results[m][ev]]
            ys = [r["F1"] for r in results[m][ev]]
            ax.plot(xs, ys, marker="o", lw=2, color=colors[m], label=m)
        ax.set_title(labels[ev])
        ax.set_xlabel("confidence threshold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 (%)")
    axes[0].legend()
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig12_threshold_f1.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[图]", out)


if __name__ == "__main__":
    main()

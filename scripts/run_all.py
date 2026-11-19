# -*- coding: utf-8 -*-
"""One-click pipeline: tests -> pre-flight -> E2/E3/E4 -> figures -> site check.

Usage:
  python scripts/run_all.py                 # tests + simulation + figures + site
  python scripts/run_all.py --full          # + extended eval, benchmark, per-class, threshold
  python scripts/run_all.py --with-train    # + 800-image detection experiment
  python scripts/run_all.py --keep-going    # continue after a failed step

Logs: outputs/logs/run_all_<timestamp>.log
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(ROOT, "outputs", "logs")


def run(name, args, log, keep_going=False):
    print("\n" + "=" * 72)
    print("[%s] %s" % (name, " ".join(args)))
    print("=" * 72)
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n\n===== %s  %s =====\n" % (dt.datetime.now().isoformat(), name))
        p = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    print("  -> exit=%d  %.1fs  log=%s" % (p.returncode, time.time() - t0, log))
    if p.returncode != 0 and not keep_going:
        print("  [stop] step failed; use --keep-going to continue.")
        sys.exit(p.returncode)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run eval_extension, benchmark, per-class, threshold sweep")
    ap.add_argument("--with-train", action="store_true",
                    help="also run the 800-image detection experiment")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, "run_all_%s.log" % dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print("repo:", ROOT)
    print("log :", log)

    run("unit tests", ["-m", "pytest", "-q", "tests"], log, args.keep_going)
    run("lint", ["-m", "ruff", "check", "src", "scripts", "tests"], log, args.keep_going)
    run("site links", ["scripts/check_site_links.py"], log, args.keep_going)
    run("pre-flight", ["scripts/check_repro_requirements.py"], log, args.keep_going)
    run("E2 fusion", ["scripts/rerun_e2_offset.py"], log, args.keep_going)
    run("E3 LiDAR", ["scripts/rerun_e3_lidar.py"], log, args.keep_going)
    run("E4 health", ["scripts/rerun_e4_lifetime.py"], log, args.keep_going)
    run("seed sensitivity", ["scripts/sensitivity_check.py"], log, args.keep_going)
    run("redraw figures", ["scripts/redraw_figs.py"], log, args.keep_going)

    if args.full:
        run("extended eval", ["scripts/eval_extension.py"], log, args.keep_going)
        run("benchmark", ["scripts/benchmark_inference.py"], log, args.keep_going)
        run("per-class AP", ["scripts/per_class_analysis.py"], log, args.keep_going)
        run("threshold sweep", ["scripts/threshold_sweep.py"], log, args.keep_going)

    if args.with_train:
        run("800-image train/eval",
            ["scripts/exp_real_detect_fixed.py", "--epochs", "15", "--batch", "16"],
            log, args.keep_going)

    print("\nall done:", log)


if __name__ == "__main__":
    main()

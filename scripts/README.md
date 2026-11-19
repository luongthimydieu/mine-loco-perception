# Scripts

Run from the repository root after `pip install -e .`. All scripts resolve paths through
`mine_loco_system.paths` (`data/`, `weights/`, `outputs/`; override with `MINE_LOCO_DATA`,
`MINE_LOCO_WEIGHTS`, `MINE_LOCO_OUTPUTS`).

## Data and pre-flight

| Script | Purpose |
|---|---|
| `download_exdark.py` | Download ExDark (CC BY 4.0) to `data/exdark` |
| `check_repro_requirements.py` | PASS/FAIL checklist for data, weights, GPU, network |
| `download_weights.py` | Download `yolov8n.pt` into `weights/` |
| `make_800_subset.py` | Build the 800-image ExDark subset from the full download |

## Detection experiments

| Script | Purpose | Output |
|---|---|---|
| `exp_real_detect_fixed.py` | 800-image, corrected three-way protocol | `outputs/train/results_real_fixed*/` |
| `exp_scaleup_ablation.py` | 5000-image, four-model ablation | `outputs/results/results_scaleup/` |
| `eval_extension.py` | V=30/15 curve, TTA, model soup | `outputs/results/extension_eval.json`, `outputs/figures/` |
| `leak_diagnostic.py` | Half-leaky vs independent evaluation | console + temporary `outputs/repro/_leak_diag` |
| `benchmark_inference.py` | Batch-1 latency / FPS | `outputs/results/benchmark_inference.json` |
| `per_class_analysis.py` | Per-class AP for four models × three sets | `outputs/results/per_class.json`, `fig11` |
| `threshold_sweep.py` | Confidence/F1 operating point | `outputs/results/threshold_sweep.json`, `fig12` |

## Simulation experiments

| Script | Purpose | Output |
|---|---|---|
| `rerun_e2_offset.py` | Robust fusion (E2) | `outputs/repro/e2_offset.json` |
| `rerun_e3_lidar.py` | LiDAR double-pass extinction (E3) | `outputs/repro/e3_lidar.json` |
| `rerun_e4_lifetime.py` | Bearing envelope + RUL (E4) | `outputs/repro/e4_lifetime.json` |
| `sensitivity_check.py` | Seed sensitivity | `outputs/repro/sensitivity.json` |
| `redraw_figs.py` | Corrected figures (cm→m) | `outputs/figures/corrected/` |

## Utilities

| Script | Purpose |
|---|---|
| `make_fig_real_degrade.py` | Regenerate the degradation comparison figure |
| `smoke_test.py` | 1-epoch, 40-image pipeline smoke test (GPU) |
| `run_scaleup.bat` | Windows launcher for the scale-up run |
| `run_all.py` | One-click pipeline (tests → simulation → figures → site) |
| `verify_workspace.py` | Repository integrity check (code/reference/site; optional data/weights) |
| `check_site_links.py` | Verify every local link/asset in `site/index.html` |

## Typical sequence

```bash
python scripts/download_exdark.py
python scripts/check_repro_requirements.py
pytest -q
python scripts/rerun_e2_offset.py
python scripts/rerun_e3_lidar.py
python scripts/rerun_e4_lifetime.py
python scripts/sensitivity_check.py
python scripts/redraw_figs.py
python scripts/exp_real_detect_fixed.py --epochs 15 --batch 16
python scripts/exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640
python scripts/eval_extension.py
```

The reference numbers and protocol are in `docs/method.md` and `docs/results.md`.

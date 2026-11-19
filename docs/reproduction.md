# Reproduction

## 1. Environment

- Python 3.10+ (tested on 3.13.9)
- Windows 11 / Linux; CUDA GPU recommended (tested on RTX 5060 Laptop 8 GB)
- Install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[full]"
# or: pip install -r requirements.txt
```

If you see `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` (the scripts already do this).

## 2. Data

```bash
python scripts/download_exdark.py --dest data/exdark
```

Expected: 5000 train and 737 valid images (YOLO labels included). See `data/README.md`.

For the demo, the code also accepts a smaller `data/exdark800/data/{train,valid}` subset.

## 3. Weights

- `weights/yolov8n.pt`: COCO pretrained. Ultralytics can download it on first use; move it into
  `weights/` so the demo finds it.
- Trained checkpoints: produce them with the training scripts below, or place them at
  `outputs/results/results_scaleup/<model>/weights/best.pt`.

Check everything:

```bash
python scripts/check_repro_requirements.py
```

It prints a PASS/FAIL list for data, weights, GPU and network.

## 4. Unit tests

```bash
pytest -q
```

18 tests cover:
- `β=3/V`, contrast retention (30 m/10 m → 36.8%; 20 m/6 m → 40.7%), depth-map monotonicity
- LiDAR: V=20 m reliable range 6.55 m; β=0.40 detection rate zero
- E2 fusion: constrained version beats pure vision; ablation does not improve
- E4: −9 dB envelope detection; health weights sum to one

## 5. Simulation experiments (E2/E3/E4)

```bash
python scripts/rerun_e2_offset.py
python scripts/rerun_e3_lidar.py
python scripts/rerun_e4_lifetime.py
python scripts/sensitivity_check.py
python scripts/redraw_figs.py
```

Outputs: `outputs/repro/e2_offset.json`, `e3_lidar.json`, `e4_lifetime.json`,
`sensitivity.json`; figures in `outputs/figures/corrected/`.

The reference original metrics are in `assets/reference/simulation_metrics.json`. The rerun scripts
compare against it and print a PASS/FAIL summary.

## 6. Real detection experiments

Small-scale (800 images / 416 px / 15 epochs, ~2 min/model):

```bash
python scripts/exp_real_detect_fixed.py --epochs 15 --batch 16
```

Seed replication:

```bash
REPRO_SHIFT=101 REPRO_TAG=_s101 python scripts/exp_real_detect_fixed.py --epochs 15 --batch 16
```

Scale-up + ablation (5000 images / 640 px / 80 epochs × 4 models, ~8 h on one GPU):

```bash
python scripts/exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640
# Windows: scripts\run_scaleup.bat
```

Outputs: `outputs/train/results_real_fixed*/`, `outputs/results/results_scaleup/`.

## 7. Extended evaluation

```bash
python scripts/eval_extension.py
```

Builds V=30/V=15 evaluation sets (368 images each), evaluates the four scale-up models, 800-image
models, TTA and model soup. Outputs `outputs/results/extension_eval.json/.csv` and figures
`outputs/figures/fig9_visibility_curve.png`, `fig10_tta_soup.png`.

Latency:

```bash
python scripts/benchmark_inference.py
```

## 8. Demo system

```bash
python -m mine_loco_system.main
python -m mine_loco_system.main --screenshots shots
python -m mine_loco_system.main --demo --record demo.mp4 --duration 48
```

If no weights are found, the UI still runs but the detector reports “not ready”; place
`weights/yolov8n.pt` or train a checkpoint.

## 9. Expected numbers

After a correct run, the scale-up table should match `assets/reference/scaleup_metrics.json` and
`docs/results.md`:

- V=20 m: M_clean 22.09, M_phys **32.24** (best); 22.19 / **33.12** (last)
- V=10 m: 17.96 / **26.85** (best); 17.11 / **27.93** (last)
- Clean: 61.81 / 42.61 (best); 62.52 / 47.09 (last)

Small numerical differences can come from GPU/cuDNN nondeterminism; the protocol and trend should be
stable.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `OMP: Error #15` | `set KMP_DUPLICATE_LIB_OK=TRUE` |
| `ModuleNotFoundError: mine_loco_system` | `pip install -e .` from the repo root |
| `data.yaml path` not found | `python scripts/download_exdark.py`; check `data/exdark/data` |
| Ultralytics tries to write to a read-only dir | set `YOLO_CONFIG_DIR=outputs/logs/.yolo_cfg` |
| CUDA OOM at 640 px | reduce `--batch`; the scripts use batch 16 |
| Chinese labels garbled in terminal | `PYTHONIOENCODING=utf-8` |

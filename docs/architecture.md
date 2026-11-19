# Architecture

## Package

```
src/mine_loco_system/
├─ paths.py            repository-relative paths + env overrides
├─ main.py             CLI entry: normal / --screenshots / --demo --record
├─ core/
│  ├─ physics.py       β=3/V, depth map, degradation, contrast, radar range
│  ├─ detector.py      YOLOv8n wrapper, model choices, TTA flag
│  ├─ fusion.py        E2 Kalman filter with track prior and robust gating
│  ├─ lidar.py         E3 double-pass extinction simulation
│  ├─ health.py        E4a envelope spectrum + E4b RUL
│  └─ alarms.py        alarm bus / work-condition state
└─ ui/
   ├─ theme.py, widgets.py, app.py
   ├─ dashboard.py     overview: KPI cards, error curve, alarms
   ├─ perception.py    degradation slider, detector, TTA, CSV export
   ├─ correction.py    30 Hz fusion demo
   ├─ lidar_page.py    point cloud and detection
   ├─ health_page.py   envelope spectrum and RUL
   └─ replay.py        history CSV replay
```

## Data flow

```
image ──► physical degradation (V slider) ──► YOLOv8n ──► detections
                                                   │
track prior + IMU ──► robust Kalman ──► lateral offset ──► alarm / replay
LiDAR beams ──► double-pass extinction ──► detection rate / reliable range
bearing vibration ──► envelope spectrum ──► health index ──► RUL / advice
```

## Demo pages

1. **Dashboard** — three subsystem status lights, four KPI cards, real-time error curve, alarms.
2. **Perception** — original / degraded / detection triptych, visibility slider, model selector
   (M_clean / M_phys), TTA checkbox, contrast-retention curve, CSV export.
3. **Correction** — 30 Hz robust fusion, vision vs fused trajectory, gate/rejection markers.
4. **LiDAR** — beam point cloud, detection result, reliable-range curve.
5. **Health** — envelope spectrum at adjustable SNR, health index, RUL table, maintenance advice.
6. **Replay** — load a per-frame CSV (sample included) and scrub through the timeline.

## Experiments

| Script | Purpose |
|---|---|
| `scripts/exp_real_detect_fixed.py` | 800-image corrected-protocol detection |
| `scripts/exp_scaleup_ablation.py` | 5000-image, four-model ablation |
| `scripts/eval_extension.py` | V=30/15 curve, TTA, model soup |
| `scripts/leak_diagnostic.py` | half-leaky vs independent evaluation |
| `scripts/rerun_e2_offset.py` | E2 fusion rerun |
| `scripts/rerun_e3_lidar.py` | E3 LiDAR rerun |
| `scripts/rerun_e4_lifetime.py` | E4a/E4b rerun |
| `scripts/sensitivity_check.py` | seed sensitivity |
| `scripts/redraw_figs.py` | corrected figures (cm→m) |
| `scripts/benchmark_inference.py` | latency / FPS |
| `scripts/check_repro_requirements.py` | pre-flight checklist |
| `scripts/download_exdark.py` | ExDark downloader |

## Output layout

```
outputs/
├─ cases/       generated YOLO datasets (disposable, regenerated from data/seeds)
├─ train/       800-image training runs
├─ results/     scale-up results + extension_eval.json
├─ repro/       E2/E3/E4 + sensitivity JSON
├─ figures/     generated figures
└─ logs/        run logs and YOLO config
```

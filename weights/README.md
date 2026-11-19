# Weights

Checkpoints are **not** committed. The code searches for them in this order:

1. `weights/m_clean_800/best.pt` — 800-image / 416 px baseline used by the demo
2. `outputs/results/results_scaleup/M_clean/weights/best.pt` — scale-up baseline
3. `weights/yolov8n.pt` — COCO-pretrained fallback

## COCO pretrained

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')   # downloads to the current directory
PY
```

Move the downloaded `yolov8n.pt` to `weights/`.

## Train your own

```bash
python scripts/exp_real_detect_fixed.py --epochs 15 --batch 16
python scripts/exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640
```

The first writes to `outputs/train/results_real_fixed*/`, the second to
`outputs/results/results_scaleup/`. The demo's model selector will pick them up automatically.

## Model card (scale-up)

| Model | Training set | Input | Epochs | Purpose |
|---|---|---|---|---|
| M_clean | clean | 640 | 80 | baseline |
| M_phys | physical dust V=20 m | 640 | 80 | physics-guided |
| M_noise | uniform darkening + Gaussian noise | 640 | 80 | non-physical control |
| M_hsv | photometric jitter | 640 | 80 | traditional control |

See `docs/results.md` for the numbers and `docs/method.md` for the protocol.

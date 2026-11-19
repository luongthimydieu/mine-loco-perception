# Model Card

## Models

| Name | Training set | Input | Epochs | Role |
|---|---|---|---|---|
| `M_clean` | ExDark clean, 5000 images | 640 | 80 | baseline |
| `M_phys` | ExDark + physical dust V=20 m | 640 | 80 | physics-guided |
| `M_noise` | ExDark + uniform darkening + Gaussian noise | 640 | 80 | non-physical control |
| `M_hsv` | ExDark + photometric jitter | 640 | 80 | traditional control |
| `M_clean_800` | ExDark clean, 800 images | 416 | 15–80 | small-scale baseline |
| `M_phys_800` | ExDark + physical dust, 800 images | 416 | 15–80 | small-scale physics |

All are YOLOv8n fine-tuned from COCO-pretrained weights, seed 0, batch 16.
The only variable across the four scale-up models is the offline degradation of the training set.

## Intended use

- Research on physics-guided synthetic degradation and evaluation protocols.
- Demonstration of a mine-locomotive perception/health prototype.
- Teaching reproducible evaluation: train/calibration/evaluation separation, disjoint seeds,
  best/last reporting, ablation controls.

## Not intended for

- Safety-critical deployment in a real mine without field validation.
- Claiming mine-track obstacle accuracy: ExDark classes are generic low-light classes.
- Any use where a synthetic-dust result is presented as a real-dust result.

## Metrics

See [results.md](results.md). Summary at V=20 m (368-image independent set, mAP@0.5):

| Model | Clean | V=20 m | V=10 m |
|---|---:|---:|---:|
| M_clean | 61.81 | 22.09 | 17.96 |
| M_phys | 42.61 | **32.24** | **26.85** |
| M_noise | 42.70 | 18.76 | 18.20 |
| M_hsv | 55.46 | 21.65 | 16.35 |

## Training data

ExDark (CC BY 4.0), 5000 train / 737 valid, YOLO format, 12 classes. Dust is synthesized with
`β=3/V` and a per-pixel depth map. The calibration/evaluation split is the last 50% of the valid
split (368 images); seeds are disjoint.

## Limitations

- Synthetic dust; no real mine images.
- Single-seed scale-up; 4-seed small-scale.
- Clean-domain cost is real (−15 to −19 pp); model soup/TTA are compromises.
- Class imbalance and domain mismatch (ExDark vs mine track) limit absolute accuracy.
- No temporal video evaluation; frames are independent.

## Ethical / environmental

- No personal data, no mine operational data.
- Training four models at 640 px for 80 epochs on one consumer GPU is on the order of 8 GPU-hours;
  inference is ~9 ms/frame at 640 px.

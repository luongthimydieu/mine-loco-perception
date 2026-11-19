# Method

## 1. Problem

Underground mine locomotives operate in dust and low light. Two constraints dominate:

- **Physical**: contrast decays exponentially with distance; a single-line LiDAR has a hard geometric
  ceiling (≈6.5 m at V=20 m) because of double-pass extinction.
- **Data**: there is no large labeled dataset of mine-track obstacles. Collecting one is expensive and
  safety-critical.

The project treats the data gap as the research object: synthesize the deployment domain from a
physical model, then ask whether training on it helps — and whether the gain is specific to the
physics rather than to “more augmentation”.

## 2. Visibility-parameterized degradation

Atmospheric scattering (Koschmieder):

```
I(x) = J(x)·t(x) + A·(1 − t(x))
t(x) = exp(−β·d(x))
β = 3 / V
```

- `V` is visibility in metres, a quantity a mine can measure and write into operating rules.
- `d(x)` is per-pixel depth from a monocular ground-plane projection:
  `d = h·f / (y − y_h)`, with `h=0.80 m`, horizontal FOV 60°, focal length from image width.
- `A=60` (airlight, DN), read noise `σ=1.5` DN.
- The depth map is row-wise, so the transmission map has **depth structure** — the key difference from
  a global darkening augmentation.

The same generator produces:

| Split | Purpose | Seed |
|---|---|---|
| train | model fitting | `SEED_TRAIN` (clean or V=20) |
| calibration | model selection only | `CAL_SEED=70001` |
| evaluation clean | unbiased clean test | `EVAL_SEED_CLEAN=91001` |
| evaluation V=20 | unbiased dust test | `EVAL_SEED_V20=91002` |
| evaluation V=10 | unbiased dust test | `EVAL_SEED_V10=91003` |
| extended V=30 / V=15 | visibility curve | `91004 / 91005` |

Calibration uses the first 50% of the valid split (368 images); evaluation uses the last 50% (368).
**No seed is shared between training, calibration, and evaluation.** `best.pt` (with selection bias)
and `last.pt` (no selection bias) are both reported.

## 3. Why the original result vanished

The original implementation generated the model-selection set and the dust evaluation set with the
same valid split, visibility and seed. The YOLO `best.pt` was selected on the evaluation set, so
`+3.89 pp = biased − unbiased`. After the three-way split, 4 independent runs × 2 weights = 8
comparisons gave **0/8** in favor of the physics-trained model at V=20 m (mean −2.9 pp). A leakage
diagnostic (same model, half-leaky evaluation set) raised mAP@0.5 from 14.52 to 16.31 (+1.79 pp),
matching the observed inflation.

This negative result is kept in the repository because it is the reason the protocol exists.

## 4. Scale-up and ablation

Four models share everything except the offline degradation of the training set:

| Model | Training-set degradation | Role |
|---|---|---|
| `M_clean` | none | baseline |
| `M_phys` | physical scattering V=20 m, per-pixel depth | physics-guided |
| `M_noise` | uniform darkening (global mean transmission of M_phys) + Gaussian noise | non-physical control: same global gray statistics, no depth structure |
| `M_hsv` | brightness / gamma / channel gain only | traditional photometric control |

5000 images, 640 px, 80 epochs, YOLOv8n + COCO pretraining, seed 0, batch 16. The only variable is the
training-set degradation. At this scale the conclusion reverses: `M_phys` gains **+10.15 / +10.93 pp**
at V=20 m (best/last), while `M_noise` and `M_hsv` do not. The depth structure is the active ingredient.

## 5. Deployment-side evaluation

- **TTA** (`augment=True`): multi-scale/flip test-time augmentation. Helps `M_phys` on clean and dust;
  costs ~2× latency.
- **Model soup**: average the weights of `M_clean` and `M_phys`. A compromise: clean-domain cost drops
  from −19.2 to −10.8 pp while retaining +7.0 pp dust gain.
- **Latency**: batch-1, 640 px, RTX 5060 Laptop: `M_phys` 9.19 ms (109 FPS); TTA 20.75 ms (48.5 FPS).

## 6. Other subsystems

- **E2 robust fusion**: Kalman filter with track-geometry prior and robust gating. High dust
  (β=0.30) reduces lateral RMSE by 66.9–80.3% and raises frame success from 65.6% to 82.1%; removing
  the track constraint makes the fusion worse than pure vision, showing the gain comes from the
  constraint, not from averaging.
- **E3 LiDAR**: double-pass extinction, 481 beams × 0.25°, reliable range
  `R = ln(1/I_thr)/(2β)`; at V=20 m, R≈6.55 m; β≥0.3 gives zero detection in simulation.
- **E4 health**: bearing envelope spectrum detects BPFO at −9 dB; multi-source RUL fusion is reported
  honestly (the gain over the best single channel is limited).

## 7. Boundaries

- Dust is synthetic; ExDark classes (12 generic low-light classes) are not mine-track obstacles.
- The scale-up run is a single seed; small-scale runs have 4 seeds.
- No real mine images are included. The project validates the method and the protocol, not mine-field
  accuracy.

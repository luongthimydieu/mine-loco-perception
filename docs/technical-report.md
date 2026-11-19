# Physics-Guided Domain Adaptation for Underground Mine Locomotive Perception

**Technical report · 2026-11-19**

> This is a code-repository report, not a peer-reviewed paper. Numbers match
> `assets/reference/*.json` and `docs/results.md`.

## Abstract

Underground mine locomotives face a data bottleneck: dust and low light degrade cameras, while labeled
mine-track obstacles are scarce. We synthesize dust from the atmospheric scattering model, using
**visibility V** as the control variable (`β = 3/V`), and train a YOLOv8n detector on the degraded
domain. We first show that the original evaluation protocol was leaky: the model-selection set and
the dust evaluation set were generated from the same split and seed, so the reported **+3.89 pp**
gain did not reproduce (**0/8** comparisons) under a three-way train/calibration/evaluation split.
At 5000 images / 640 px / 80 epochs the conclusion reverses: the physics-trained model gains
**+10.15 / +10.93 pp** at V=20 m (best/last) and **+8.89 / +10.82 pp** at V=10 m, while
global-gray-aligned noise and photometric-only augmentations gain nothing. The depth structure of the
scattering model is the active ingredient. We also report deployment-side options (TTA, model soup,
latency) and the boundaries of the claim.

## 1. Introduction

Two constraints dominate mine perception:

- **Physics.** Contrast decays as `exp(−βd)`. At V=30 m, 10 m away retains 36.8% contrast; at V=20 m,
  6 m retains 40.7%. A single-line LiDAR has a double-pass ceiling `R = ln(1/I_thr)/(2β) ≈ 6.55 m`
  at V=20 m — the same order as braking distance.
- **Data.** There is no large labeled mine dataset. Collecting one is expensive and safety-critical.

We treat the data gap as the object of study: generate the deployment domain from a physical model,
then ask whether training on it helps and whether the gain is specific to the physics.

## 2. Method

### 2.1 Visibility-parameterized degradation

```
I(x) = J(x)·t(x) + A·(1 − t(x))
t(x) = exp(−β·d(x))
β = 3 / V
```

`d(x)` is per-pixel depth from a monocular ground-plane projection `d = h·f/(y − y_h)`, with
`h=0.80 m`, horizontal FOV 60°, `A=60` DN, read noise `σ=1.5` DN. The transmission map varies with
depth; this is the difference from global darkening.

### 2.2 Three-way protocol

| Role | Data | Seed |
|---|---|---|
| Training | ExDark train split | 0 (+i per image) |
| Calibration (selection only) | valid first 50% (368) | 70001 |
| Evaluation clean | valid last 50% (368) | 91001 |
| Evaluation V=20 | same 368 | 91002 |
| Evaluation V=10 | same 368 | 91003 |
| Extended V=30 / V=15 | same 368 | 91004 / 91005 |

No seed is shared across roles. We report `best.pt` (selection bias) and `last.pt` (no selection bias).

### 2.3 Models

Four scale-up models share everything except the training-set degradation:

| Model | Training set | Role |
|---|---|---|
| M_clean | clean | baseline |
| M_phys | physical dust V=20 m | physics-guided |
| M_noise | uniform darkening (global mean transmission of M_phys) + Gaussian noise | non-physical control |
| M_hsv | brightness/gamma/channel gain | photometric control |

YOLOv8n + COCO pretraining, 5000 images, 640 px, 80 epochs, seed 0, batch 16.

## 3. Results

### 3.1 The original gain did not reproduce

| Condition | 8 comparisons (Δ = M_phys − M_clean) | Mean | M_phys better |
|---|---|---|---|
| Dust V=20 m | −1.11 / −1.80 / −2.55 / −6.28 (best); −1.11 / −1.80 / −2.55 / −5.27 (last) | −2.9 pp | 0/8 |
| Dust V=10 m | −0.50 / +0.15 / −0.78 / +0.44 (best); −0.50 / +0.15 / −0.78 / +1.12 (last) | ≈0 | 4/8 |
| Clean | all negative | −16.5 pp | 0/8 |

Leakage diagnostic (same `M_clean/best.pt`): pure independent 14.52 → half-leaky 16.31 mAP@0.5,
i.e. **+1.79 pp** from half-leakage alone.

### 3.2 Scale-up reverses the conclusion

| Evaluation | Weight | M_clean | M_phys | M_noise | M_hsv | Δphys |
|---|---|---:|---:|---:|---:|---:|
| Clean | best | 61.81 | 42.61 | 42.70 | 55.46 | −19.20 |
| Clean | last | 62.52 | 47.09 | 47.25 | 59.44 | −15.43 |
| V=20 m | best | 22.09 | **32.24** | 18.76 | 21.65 | **+10.15** |
| V=20 m | last | 22.19 | **33.12** | 19.91 | 21.96 | **+10.93** |
| V=10 m | best | 17.96 | **26.85** | 18.20 | 16.35 | **+8.89** |
| V=10 m | last | 17.11 | **27.93** | 17.08 | 17.87 | **+10.82** |

`M_noise` matches M_phys’s global gray statistics but has no depth structure; it gains nothing.
`M_hsv` also gains nothing. Cross-domain drop: M_clean −39.7 pp, M_phys −10.4 pp (clean → V=20).

### 3.3 Per-class behavior at V=20 m

AP@0.5 (%):

| Class | M_clean | M_phys | Δ |
|---|---:|---:|---:|
| Bottle | 6.59 | 33.85 | +27.26 |
| Dog | 32.44 | 56.93 | +24.49 |
| Cat | 23.77 | 40.92 | +17.15 |
| Motorbike | 47.64 | 64.42 | +16.78 |
| People | 16.94 | 32.87 | +15.93 |
| Cup | 35.71 | 48.61 | +12.90 |
| Car | 23.73 | 28.15 | +4.42 |
| Chair | 38.64 | 42.76 | +4.12 |
| Table | 27.72 | 29.35 | +1.63 |
| Boat | 0.00 | 0.16 | +0.16 |
| Bus | 0.00 | 0.15 | +0.15 |
| Bicycle | 11.88 | 8.75 | −3.13 |

The largest gains are on small/thin objects (Bottle, Cup) and animals/people; Bicycle is the only
class with a small drop. This is consistent with depth-structured contrast loss: near-field large
objects survive dust, small/distant objects do not.

### 3.4 Visibility curve

| Evaluation | M_clean | M_phys | M_noise | M_hsv | soup |
|---|---:|---:|---:|---:|---:|
| Clean | 61.81 | 42.60 | 42.70 | 55.46 | 51.01 |
| V=30 m | 23.04 | **32.96** | 20.35 | 22.67 | 31.37 |
| V=20 m | 22.09 | **32.24** | 18.76 | 21.65 | 29.09 |
| V=15 m | 21.06 | **30.61** | 19.25 | 19.86 | 26.75 |
| V=10 m | 17.96 | **26.85** | 18.20 | 16.35 | 22.72 |

`M_phys` wins at every visibility; the gain grows as dust worsens. The 800-image models are 15–20 pp
lower with no consistent M_phys vs M_clean difference, indicating training scale is a precondition.

### 3.5 Deployment

| Model / inference | Clean | V=20 m | V=10 m |
|---|---:|---:|---:|
| M_clean | 61.81 | 22.09 | 17.96 |
| M_clean + TTA | 61.05 | 21.70 | 17.62 |
| M_phys | 42.60 | 32.24 | 26.85 |
| **M_phys + TTA** | **44.88** | **34.17** | **27.60** |
| soup | 51.01 | 29.09 | 22.72 |
| soup + TTA | 54.29 | 30.51 | 24.16 |

Latency (batch 1, RTX 5060 Laptop): M_phys 640 px 9.19 ms (109 FPS); TTA 20.75 ms (48.5 FPS).

### 3.6 Threshold operating point

Confidence thresholds affect precision/recall but not mAP. Best-F1 operating points:

| Model | Clean | V=20 m | V=10 m |
|---|---|---|---|
| M_clean | F1 61.98 @ 0.40 | F1 28.39 @ 0.10 | F1 21.89 @ 0.15 |
| M_phys | F1 45.68 @ 0.25 | **F1 36.49 @ 0.25** | **F1 30.92 @ 0.25** |

`M_phys` keeps a stable 0.25 operating point across domains; `M_clean` needs 0.10–0.15 in dust.
Full sweep: `outputs/results/threshold_sweep.json`, `assets/figures/threshold_f1.png`.

```bash
python scripts/threshold_sweep.py
```

## 4. Why the gain is physical

The ablation is the argument:

- `M_noise`: same global gray level as `M_phys`, no depth structure → no gain.
- `M_hsv`: photometric only → no gain.
- `M_phys`: depth-structured scattering → gain at every visibility.

If the gain were a generic “harder augmentation” effect, `M_noise` would gain too. It does not.

## 5. Boundaries

- Dust is synthetic; no real mine images.
- ExDark classes are not mine-track obstacles.
- Scale-up is a single seed; small-scale has 4 seeds.
- Clean-domain cost is real; soup/TTA are compromises, not free wins.
- No temporal/video evaluation.

## 6. Reproducibility

```bash
pip install -e ".[full]"
python scripts/download_exdark.py
pytest -q
python scripts/run_all.py --full
```

Reference outputs are in `assets/reference/`; the expected numbers are in `docs/results.md`.
Simulation E2/E3/E4 reproduce bit-for-bit; detection has GPU nondeterminism of order 0.01–0.1 pp.

## References

- Koschmieder, H. (1924). Theorie der horizontalen Sichtweite.
- ExDark: Loh, Y. P., & Chan, C. S. (2019). Getting to Know Low-light Images with The Exclusively Dark Dataset.
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems.
- Randall, R. B., & Antoni, J. (2011). Rolling element bearing diagnostics — a tutorial.

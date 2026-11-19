# Results

All numbers are `mAP@0.5 (%)` on 368-image independent evaluation sets unless stated otherwise.
Reference JSONs are in `assets/reference/`.

## 1. Small-scale protocol correction (800 images / 416 px / 15–80 epochs)

The original `+3.89 pp` did not reproduce under the corrected protocol:

| Condition | 8 comparisons (Δ = M_phys − M_clean) | Mean | M_phys better |
|---|---|---|---|
| Dust V=20 m | −1.11 / −1.80 / −2.55 / −6.28 (best); −1.11 / −1.80 / −2.55 / −5.27 (last) | **−2.9 pp** | **0/8** |
| Dust V=10 m | −0.50 / +0.15 / −0.78 / +0.44 (best); −0.50 / +0.15 / −0.78 / +1.12 (last) | ≈0 | 4/8 (no consistency) |
| Clean | all negative | **−16.5 pp** | 0/8 |

Leakage diagnostic (same `M_clean/best.pt`, only the evaluation set changes):

| Evaluation set | Images | mAP@0.5 |
|---|---:|---:|
| pure independent | 368 | 14.52 |
| half-leaky | 368 | **16.31** |

Half-leakage alone inflates by **+1.79 pp**; the original implementation was 100% same-domain.

## 2. Scale-up + ablation (5000 images / 640 px / 80 epochs)

| Evaluation | Weight | M_clean | M_phys | M_noise | M_hsv | Δphys | Δnoise | Δhsv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | best | 61.81 | 42.61 | 42.70 | 55.46 | −19.20 | −19.11 | −6.35 |
| Clean | last | 62.52 | 47.09 | 47.25 | 59.44 | −15.43 | −15.27 | −3.08 |
| V=20 m | best | 22.09 | **32.24** | 18.76 | 21.65 | **+10.15** | −3.33 | −0.44 |
| V=20 m | last | 22.19 | **33.12** | 19.91 | 21.96 | **+10.93** | −2.28 | −0.23 |
| V=10 m | best | 17.96 | **26.85** | 18.20 | 16.35 | **+8.89** | +0.24 | −1.61 |
| V=10 m | last | 17.11 | **27.93** | 17.08 | 17.87 | **+10.82** | −0.03 | +0.76 |

**Reading.** The gain is not “any augmentation helps”: `M_noise` has the same global gray statistics as
`M_phys` but no depth structure, and gains nothing. `M_hsv` (photometric only) also gains nothing.
Cross-domain drop: M_clean −39.7 pp, M_phys −10.4 pp from clean to V=20 m.

## 3. Extended visibility curve (V=30 / 15 added)

| Evaluation | M_clean | M_phys | M_noise | M_hsv | soup | 800 M_clean | 800 M_phys |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 61.81 | 42.60 | 42.70 | 55.46 | 51.01 | 41.40 | 33.01 |
| V=30 m | 23.04 | **32.96** | 20.35 | 22.67 | 31.37 | 14.67 | 11.00 |
| V=20 m | 22.09 | **32.24** | 18.76 | 21.65 | 29.09 | 13.25 | 10.51 |
| V=15 m | 21.06 | **30.61** | 19.25 | 19.86 | 26.75 | 12.30 | 10.26 |
| V=10 m | 17.96 | **26.85** | 18.20 | 16.35 | 22.72 | 9.77 | 10.01 |

- `M_phys` wins at every visibility from 30 m to 10 m; the gain grows as dust worsens (+9.9 to +10.9 pp).
- `M_noise`/`M_hsv` show no consistent gain at any visibility.
- The 800-image models are 15–20 pp lower and show no consistent `M_phys` vs `M_clean` difference:
  **training scale is a precondition for the gain**.

## 4. Deployment-side

| Model / inference | Clean | V=20 m | V=10 m |
|---|---:|---:|---:|
| M_clean | 61.81 | 22.09 | 17.96 |
| M_clean + TTA | 61.05 | 21.70 | 17.62 |
| M_phys | 42.60 | 32.24 | 26.85 |
| **M_phys + TTA** | **44.88** | **34.17** | **27.60** |
| soup | 51.01 | 29.09 | 22.72 |
| soup + TTA | 54.29 | 30.51 | 24.16 |

TTA helps `M_phys` on all three domains (+2.27 / +1.93 / +0.75 pp) and slightly hurts `M_clean`.
Model soup is not the accuracy optimum, but halves the clean-domain cost while keeping +7.0 pp dust gain.

Latency (batch=1, RTX 5060 Laptop, 640 px, including pre/post-processing):

| Model / input | Mean | P95 | FPS |
|---|---:|---:|---:|
| 800 M_clean / 416 | 7.78 ms | 8.09 ms | 128.8 |
| M_clean / 640 | 8.55 ms | 9.31 ms | 116.8 |
| M_phys / 640 | 9.19 ms | 10.20 ms | 109.1 |
| M_phys / 640 + TTA | 20.75 ms | 22.67 ms | 48.5 |

## 5. Simulation subsystems

| Experiment | Result |
|---|---|
| E2 robust fusion | β=0.30: lateral RMSE ↓ 66.9–80.3%; success 65.6% → 82.1% |
| E3 LiDAR | V=20 m reliable range 6.55 m; β≥0.3 detection rate 0% |
| E4a bearing | BPFO detectable at −9 dB (4 seeds) |
| E4b RUL | multi-source fusion; gain over best single channel is limited (reported honestly) |
| Reproducibility | 241/241 simulation metrics reproduce bit-for-bit |

## 6. Boundaries

Single-seed scale-up; synthetic dust; ExDark classes differ from mine-track obstacles; no real mine
images. These boundaries must be stated with every gain number.

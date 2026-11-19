# FAQ

## The paper says “no gain”, the website says “+10.9 pp”. Contradiction?

No — different training scales.

| Setting | Result |
|---|---|
| 800 images / 416 px / 15–80 epochs | no gain: 0/8 comparisons, V=20 m mean −2.9 pp |
| 5000 images / 640 px / 80 epochs | gain: +10.15 / +10.93 pp at V=20 m (best/last) |

Both use the **same corrected protocol** (train/calibration/evaluation split, disjoint seeds,
best/last). The full story is “the method works, but is constrained by training scale”.

## How do you know the gain is “physics-guided” and not just augmentation?

Three controls share the protocol:

| Control | V=20 m (best/last) | Why |
|---|---|---|
| `M_noise` (uniform darkening + Gaussian noise, same global gray level as M_phys) | −3.33 / −2.28 pp | no depth structure → no gain |
| `M_hsv` (brightness/gamma/channel gain) | −0.44 / −0.23 pp | photometric only → no gain |
| `M_phys` (β=3/V, per-pixel depth) | **+10.15 / +10.93 pp** | depth-structured scattering → gain |

`M_noise` matches M_phys’s global gray statistics; the only difference is whether transmission varies
with depth. It gains nothing. That is the ablation’s point.

## The clean-domain cost is 15–19 pp. Is the model usable?

The deployment domain is dust, and the “domain replacement” design intentionally aligns training with
deployment. If both domains matter:

- **model soup** (average M_clean and M_phys weights): clean 51.01 (−10.8 pp), V=20 29.09 (+7.0 pp);
- **switch by condition**: clean → M_clean, dust → M_phys (the demo has a model selector);
- **M_phys + TTA**: clean 44.88, V=20 34.17, V=10 27.60; all three domains improve, at ~2× latency.

A 1:1 clean+dust mixed training is a natural next experiment and is not done yet.

## Is TTA just tuning on the test set?

No. TTA is standard multi-scale/flip augmentation at inference, applied uniformly. It helps M_phys on
all three domains and slightly hurts M_clean. The demo exposes it as a checkbox so you can see the
latency/accuracy trade-off.

## Why are 800-image models so much worse?

They are 15–20 pp lower on every evaluation set, and their M_phys vs M_clean difference has no
consistency (V=10 even slightly favors M_phys). This is the evidence that **training scale is a
precondition**, not that the method is invalid.

## Can I reproduce the numbers?

Yes:

```bash
pytest -q
python scripts/check_repro_requirements.py
python scripts/rerun_e2_offset.py
python scripts/rerun_e3_lidar.py
python scripts/rerun_e4_lifetime.py
python scripts/eval_extension.py
```

The reference JSONs are in `assets/reference/`. Small GPU/cuDNN differences are expected; the protocol
and trend should be stable.

## What is not included?

- ExDark images (235 MB) — download with `scripts/download_exdark.py` (CC BY 4.0).
- Trained checkpoints — train with `scripts/exp_scaleup_ablation.py` or place them in `weights/`.
- Competition documents, personal data, patent/software-copyright drafts — not part of this repo.

## What are the biggest limitations?

1. Dust is synthetic; no real mine images.
2. ExDark classes are generic low-light classes, not mine-track obstacles.
3. The scale-up run is one seed; the small-scale runs have four.
4. The clean-domain cost is real; model soup/TTA are compromises, not free wins.

# Data Card

## ExDark

- **What**: a low-light image dataset with 12 object classes, YOLO-format labels.
- **Source**: `dronefreak/ExDark` on Hugging Face (mirror of the original ExDark release).
- **License**: **CC BY 4.0** — attribution required; check the original ExDark terms.
- **Splits used**: 5000 train images (first 5000 by filename), 737 valid images.
- **Classes**: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.
- **Not mine-specific**: these are generic low-light classes; they are not mine-track obstacles.

Download:

```bash
python scripts/download_exdark.py --dest data/exdark
```

## Synthetic dust

The deployment domain is generated from the atmospheric scattering model:

```
I = J·t + A(1−t),  t = exp(−βd),  β = 3/V
```

- `V` is visibility (m); `d` is per-pixel depth from a ground-plane projection.
- `A=60` DN; read noise σ=1.5 DN; images resized to 640 px (or 416 px for the small-scale runs).
- Seeds: train `0` (+i per image), calibration `70001`, evaluation clean/V20/V10 `91001/91002/91003`,
  extended V30/V15 `91004/91005`. **No seed is shared across roles.**

## 800-image subset

`python scripts/make_800_subset.py` copies the first 800 train / 737 valid images and labels to
`data/exdark800/data`. The demo uses this subset if present, otherwise the full valid images.

## Known biases

- ExDark scenes differ from underground tunnels (lighting, geometry, object mix).
- The depth map assumes a flat ground plane; walls and irregular surfaces are approximated.
- Dust is homogeneous; real dust has gusts, particle size distributions, and spatially varying
  visibility.
- Class frequencies are imbalanced; per-class AP varies widely (see `docs/results.md` and
  `outputs/results/per_class.json`).

## Privacy

No personal data. No mine operational data. The dataset is public; do not add private mine images
to this repository without permission.

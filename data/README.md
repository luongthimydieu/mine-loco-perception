# Data

## ExDark (CC BY 4.0)

This project uses **ExDark** as the real low-light image base. Dust is then synthesized with the
atmospheric scattering model; no real mine images are included.

- Dataset: `dronefreak/ExDark` on Hugging Face (YOLO format, 12 classes)
- License: **CC BY 4.0** — please cite the original ExDark paper and keep the attribution.

### Download

```bash
pip install huggingface_hub
python scripts/download_exdark.py --dest data/exdark
```

Expected layout:

```
data/
└─ exdark/
   └─ data/
      ├─ train/{images,labels}
      └─ valid/{images,labels}
```

The downloader only fetches `data/*` and prints image/label counts when finished. In mainland China
you can set `HF_ENDPOINT=https://hf-mirror.com` before running it.

### 800-image subset (optional)

The demo system looks for `data/exdark800/data/valid/images` first, then falls back to
`data/exdark/data/valid/images`. To create the 800-image subset used by the small-scale experiment:

```bash
python - <<'PY'
from pathlib import Path
import shutil
src = Path('data/exdark/data'); dst = Path('data/exdark800/data')
for split, n in [('train', 800), ('valid', 737)]:
    for kind in ('images', 'labels'):
        files = sorted((src / split / kind).glob('*'))[:n]
        (dst / split / kind).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / split / kind / f.name)
print('done')
PY
```

## Sample data

The demo replay file lives in `assets/sample/correction_frames.csv`, independent of the dataset root.

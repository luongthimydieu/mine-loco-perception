# Third-Party Notices

This repository's own code is MIT (see `LICENSE`). It depends on third-party software and data with
their own licenses. This file is a summary, not legal advice.

| Component | License | Notes |
|---|---|---|
| Ultralytics YOLOv8 | **AGPL-3.0** | Detector training/inference. Distributing a binary that includes it may require AGPL compliance or a commercial Ultralytics license. |
| PySide6 / Qt for Python | LGPL-3.0 (or commercial) | Desktop UI. Dynamic linking and license notices required. |
| PyTorch | BSD-3-Clause | Deep learning runtime. |
| torchvision | BSD-3-Clause | Image transforms / model utilities. |
| OpenCV (opencv-python) | Apache-2.0 | Image I/O in some paths. |
| NumPy | BSD-3-Clause | Numerical arrays. |
| Pillow | HPND | Image I/O. |
| Matplotlib | PSF-based (Matplotlib license) | Figures. |
| pandas | BSD-3-Clause | Tables/metrics. |
| PyYAML | MIT | YAML configs. |
| pyqtgraph | MIT | Plots in the UI. |
| python-docx | MIT | Document generation (optional). |
| python-pptx | MIT | Presentation generation (optional). |
| pytest | MIT | Tests (dev only). |
| Ruff | MIT | Lint (dev only). |
| ExDark dataset | **CC BY 4.0** | Low-light images. Attribution required; not included in this repo. |
| YOLOv8 COCO weights | AGPL-3.0 | Downloaded by Ultralytics; subject to the same AGPL terms. |

## If you need a permissive binary

- Replace Ultralytics YOLOv8 with an Apache-2.0 detector (e.g. YOLOX) or obtain a commercial
  Ultralytics license.
- Keep PySide6 dynamically linked and include its license text.
- Keep ExDark attribution and do not redistribute the dataset without checking its terms.

## What this repository does not include

- ExDark images or labels (download with `scripts/download_exdark.py`).
- Trained checkpoints.
- Any mine operational data or personal data.

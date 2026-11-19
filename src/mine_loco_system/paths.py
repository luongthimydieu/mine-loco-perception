# -*- coding: utf-8 -*-
"""Repository path configuration.

All paths are relative to the repository root and can be overridden with
environment variables:

    MINE_LOCO_DATA      dataset root      (default: <repo>/data)
    MINE_LOCO_WEIGHTS   pretrained weights (default: <repo>/weights)
    MINE_LOCO_OUTPUTS   outputs root       (default: <repo>/outputs)

Expected dataset layout (YOLO format, from the ExDark dataset):
    data/exdark/data/{train,valid}/{images,labels}
    data/exdark800/data/{train,valid}/{images,labels}   # optional 800-image subset
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MINE_LOCO_DATA", REPO_ROOT / "data"))
WEIGHTS_DIR = Path(os.environ.get("MINE_LOCO_WEIGHTS", REPO_ROOT / "weights"))
OUTPUTS_DIR = Path(os.environ.get("MINE_LOCO_OUTPUTS", REPO_ROOT / "outputs"))
ASSETS_DIR = REPO_ROOT / "assets"
FIGURES_DIR = OUTPUTS_DIR / "figures"

EXDARK_DATA = DATA_DIR / "exdark" / "data"
EXDARK800_DATA = DATA_DIR / "exdark800" / "data"
CASES_DIR = OUTPUTS_DIR / "cases"
TRAIN_DIR = OUTPUTS_DIR / "train"
RESULTS_DIR = OUTPUTS_DIR / "results"
REPRO_DIR = OUTPUTS_DIR / "repro"
LOGS_DIR = OUTPUTS_DIR / "logs"
REFERENCE_DIR = ASSETS_DIR / "reference"


def default_weight():
    """Return the first existing default detector weight, or None."""
    candidates = [
        WEIGHTS_DIR / "m_clean_800" / "best.pt",
        RESULTS_DIR / "results_scaleup" / "M_clean" / "weights" / "best.pt",
        WEIGHTS_DIR / "yolov8n.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def default_image_dir():
    """First existing valid-image directory for the demo."""
    for p in (EXDARK800_DATA / "valid" / "images",
              EXDARK_DATA / "valid" / "images"):
        if p.exists():
            return str(p)
    return str(EXDARK800_DATA / "valid" / "images")


def scaleup_weight(model):
    """Path to a scale-up ablation checkpoint (M_clean / M_phys / M_noise / M_hsv)."""
    return RESULTS_DIR / "results_scaleup" / model / "weights" / "best.pt"


def ensure_outputs():
    for p in (OUTPUTS_DIR, CASES_DIR, TRAIN_DIR, RESULTS_DIR, REPRO_DIR,
              LOGS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)

# -*- coding: utf-8 -*-
r"""Repository path configuration.

All paths are relative to the repository root and can be overridden with
environment variables:

    MINE_LOCO_DATA      dataset root      (default: <repo>/data)
    MINE_LOCO_WEIGHTS   pretrained weights (default: <repo>/weights)
    MINE_LOCO_OUTPUTS   outputs root       (default: <repo>/outputs)

Expected dataset layout (YOLO format, from the ExDark dataset):
    data/exdark/data/{train,valid}/{images,labels}
    data/exdark800/data/{train,valid}/{images,labels}   # optional 800-image subset

Frozen (PyInstaller) layout:
    MineLoco.exe
    weights/yolov8n.pt          # optional, next to the exe
    data/exdark/data/...        # optional
    outputs/                    # created under %LOCALAPPDATA%\MineLoco\outputs
    _internal/assets/...        # bundled sample data and icons
"""
import os
import sys
from pathlib import Path

# Source checkout: <repo>/src/mine_loco_system/paths.py
# Frozen bundle:  <dist>/MineLoco/MineLoco.exe  (+ assets in _MEIPASS)
_FROZEN = bool(getattr(sys, "frozen", False))
if _FROZEN:
    APP_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
    REPO_ROOT = APP_DIR
    _default_data = APP_DIR / "data"
    _default_weights = APP_DIR / "weights"
    _default_outputs = Path(os.environ.get("LOCALAPPDATA", APP_DIR)) / "MineLoco" / "outputs"
    ASSETS_DIR = BUNDLE_DIR / "assets"
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    _default_data = REPO_ROOT / "data"
    _default_weights = REPO_ROOT / "weights"
    _default_outputs = REPO_ROOT / "outputs"
    ASSETS_DIR = REPO_ROOT / "assets"

DATA_DIR = Path(os.environ.get("MINE_LOCO_DATA", _default_data))
WEIGHTS_DIR = Path(os.environ.get("MINE_LOCO_WEIGHTS", _default_weights))
OUTPUTS_DIR = Path(os.environ.get("MINE_LOCO_OUTPUTS", _default_outputs))
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

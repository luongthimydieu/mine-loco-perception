# Packaging the desktop app

The software is the PySide6 application in `src/mine_loco_system/` (six pages: dashboard,
perception, fusion, LiDAR, health, replay). You can run it from source or ship a Windows build via
**GitHub Releases**.

## 1. Run from source (recommended for development)

```bash
pip install -e ".[full]"
python -m mine_loco_system.main
# or, after install:
mine-loco
```

## 2. Build a Windows one-folder app

```bash
python -m pip install -e ".[full]" pyinstaller
pyinstaller packaging/mine_loco.spec --noconfirm --clean
```

Output:

```
dist/MineLoco/MineLoco.exe
dist/MineLoco/_internal/...
```

Zip it:

```bat
powershell -NoProfile -Command "Compress-Archive -Path 'dist\MineLoco\*' -DestinationPath 'dist\MineLoco-windows-x64.zip' -Force"
```

or simply run `packaging\build_windows.bat`.

## 3. Publish on GitHub Releases

Do **not** commit `dist/` (it is git-ignored). Attach the zip to a Release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The `Release` workflow (`.github/workflows/release.yml`) builds on `windows-latest` and uploads
`MineLoco-windows-x64.zip` automatically. You can also run it manually from the Actions tab.

## 4. What users do

1. Download and unzip `MineLoco-windows-x64.zip`.
2. Put a checkpoint next to `MineLoco.exe`:
   ```
   weights/yolov8n.pt                 # COCO pretrained fallback
   # or
   weights/m_clean_800/best.pt        # trained baseline
   # or
   outputs/results/results_scaleup/M_phys/weights/best.pt
   ```
3. Double-click `MineLoco.exe`.

Optional:

- `data/exdark/data/valid/images/` next to the exe gives the perception page sample images.
- `outputs/` is created under `%LOCALAPPDATA%\MineLoco\outputs` by default (writable), or set
  `MINE_LOCO_OUTPUTS`.
- `--screenshots DIR`, `--demo --record out.mp4` still work from a terminal.

Weights and ExDark are **not** bundled: the zip stays small and avoids redistributing third-party
assets. Put them beside the exe or use the environment variables.

## 5. Size and limits

A one-folder build with PyTorch + Ultralytics + PySide6 is large: the local 2026-11-20 build was
**4.7 GB** uncompressed (115 MB launcher exe + `_internal`); zipped it is typically **2–3 GB**, so
the release workflow splits it. GitHub Releases allows 2 GB per file. The release workflow splits the zip into `.001/.002/...`
parts with `packaging/split_archive.py`; users recombine:

```bat
copy /b MineLoco-windows-x64.zip.001 + MineLoco-windows-x64.zip.002 MineLoco-windows-x64.zip
```

or with 7-Zip: `7z x MineLoco-windows-x64.zip.001`. If you want a single small asset, ship an
ONNX Runtime build instead of PyTorch (smaller, requires an export step), or exclude training-only
dependencies (risky).

## 6. License caveat (important)

Our code is MIT, but the runtime dependencies are not all permissive:

- **Ultralytics YOLOv8** is **AGPL-3.0**. If you distribute a binary that includes it, the AGPL
  obligations apply to the combined distribution (source availability, notices), unless you have a
  commercial Ultralytics license, or you switch to a permissive detector (e.g. YOLOX, Apache-2.0).
- **PySide6 / Qt for Python** is LGPL-3.0 (or commercial); dynamic linking and notices are required.
- ExDark is **CC BY 4.0**; trained weights derived from it should keep the attribution.

See `THIRD_PARTY_NOTICES.md`. This is not legal advice; check with your institution before publishing
a binary.

## 7. Windows SmartScreen

Unsigned executables may show “Windows protected your PC”. Options:

- click “More info → Run anyway” (fine for a student release),
- sign with a code-signing certificate (paid; not required),
- publish the source install path as the primary distribution and the zip as convenience.

## 8. Reproducible build notes

- Build on the same Python minor version (3.13) and pin `requirements.txt`.
- On Anaconda/Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` before PyInstaller; otherwise the torch
  submodule collection can abort with `OMP: Error #15`.
- The spec collects `ultralytics` and `pyqtgraph` data; PyInstaller hooks handle PySide6/torch.
- If a hidden import is missing, add it to `hiddenimports` in `packaging/mine_loco.spec`.
- The spec excludes `PyQt5`/`PyQt6`/`PySide2`: PyInstaller refuses to bundle multiple Qt bindings.
  Keep only PySide6 installed in the build environment (or keep the excludes).
- Test on a clean Windows VM: unzip, add weights, launch, open all six pages.

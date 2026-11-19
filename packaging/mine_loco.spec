# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Mine-Loco desktop app (Windows one-folder build).

Build:
    pyinstaller packaging/mine_loco.spec --noconfirm --clean

Output:
    dist/MineLoco/MineLoco.exe
    dist/MineLoco-windows-x64.zip   (after build_windows.bat / CI)
"""
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(ROOT, "assets"), "assets")]
binaries = []
hiddenimports = []

# Ultralytics has data files (cfg yaml) and dynamic imports; pyqtgraph too.
for pkg in ("ultralytics", "pyqtgraph"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MineLoco",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(ROOT, "packaging", "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MineLoco",
)

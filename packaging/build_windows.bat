@echo off
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..

echo [1/3] install build tools
python -m pip install -e ".[full]" pyinstaller || goto :err

echo [2/3] build one-folder app
pyinstaller packaging\mine_loco.spec --noconfirm --clean || goto :err

echo [3/3] zip + split for GitHub 2 GB limit
powershell -NoProfile -Command "Compress-Archive -Path 'dist\MineLoco\*' -DestinationPath 'dist\MineLoco-windows-x64.zip' -Force" || goto :err
python packaging\split_archive.py dist\MineLoco-windows-x64.zip --part-mb 1900 || goto :err

echo.
echo Done: dist\MineLoco-windows-x64.zip.001 ... + dist\SHA256SUMS.txt
echo Put weights\yolov8n.pt (or weights\m_clean_800\best.pt) next to MineLoco.exe.
exit /b 0

:err
echo Build failed.
exit /b 1

@echo off
rem 扩大训练规模重跑 + 传统增广消融（后台运行，日志写 outputs\logs）
cd /d %~dp0
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
if not exist "%~dp0..\outputs\logs" mkdir "%~dp0..\outputs\logs"
set LOG=%~dp0..\outputs\logs\scaleup_run.log
D:\APP\Anaconda\python.exe -u exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640 > "%LOG%" 2>&1
echo DONE >> "%LOG%"

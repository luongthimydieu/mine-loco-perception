# -*- coding: utf-8 -*-
r"""复现可行性诊断：逐项检查重跑真实检测实验所需的全部前置条件。

用法：
    python check_repro_requirements.py

输出一份逐项 PASS/FAIL 清单，用于在报告中如实说明"哪些实验能重跑、哪些不能"。
"""
import importlib
import io
import os
import socket
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from mine_loco_system.paths import EXDARK_DATA, WEIGHTS_DIR, TRAIN_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = str(EXDARK_DATA.parent.parent)
_WEIGHTS = str(WEIGHTS_DIR / 'yolov8n.pt')
_TRAIN_ROOT = str(TRAIN_DIR / 'results_real_fixed')

CHECKS = []


def check(name, ok, detail):
    CHECKS.append((name, bool(ok), detail))


# ── 1. Python 包 ──
PKGS = ['numpy', 'pandas', 'matplotlib', 'cv2', 'torch', 'ultralytics', 'yaml',
        'docx', 'pptx']
print('=' * 78)
print('解释器:', sys.executable)
print('Python :', sys.version.split()[0])
print('=' * 78)
print('\n【1】Python 依赖')
for m in PKGS:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', 'ok')
        check(f'包 {m}', True, f'{ver}')
        print(f'  PASS  {m:12s} {ver}')
    except Exception as ex:
        check(f'包 {m}', False, type(ex).__name__)
        print(f'  FAIL  {m:12s} {type(ex).__name__}')

# ── 2. 数据集与权重 ──
print('\n【2】数据集与预训练权重')
paths = {
    'ExDark 数据根目录': os.path.join(_DATA_ROOT, 'exdark', 'data'),
    'ExDark train/images': os.path.join(_DATA_ROOT, 'exdark', 'data', 'train', 'images'),
    'ExDark valid/images': os.path.join(_DATA_ROOT, 'exdark', 'data', 'valid', 'images'),
    'YOLOv8n 预训练权重': _WEIGHTS,
    '原始训练产物 results_real': _TRAIN_ROOT,
}
for label, p in paths.items():
    ok = os.path.exists(p)
    if ok and os.path.isdir(p):
        n = sum(len(f) for _, _, f in os.walk(p))
        detail = f'{n} 个文件'
    elif ok:
        detail = f'{os.path.getsize(p)/1048576:.1f} MB'
    else:
        detail = '不存在'
    check(label, ok, detail)
    print(f'  {"PASS" if ok else "FAIL"}  {label:26s} {detail}')

# 若能找到训练产物，检查 best.pt / last.pt 是否可用于"免训练重评"
if os.path.exists(paths['原始训练产物 results_real']):
    for name in ('M_clean', 'M_phys'):
        for w in ('best.pt', 'last.pt'):
            wp = os.path.join(paths['原始训练产物 results_real'], name, 'weights', w)
            check(f'{name}/{w}', os.path.exists(wp), wp)
            print(f'  {"PASS" if os.path.exists(wp) else "FAIL"}  {name}/{w}')

# ── 3. 网络 ──
print('\n【3】网络（下载 ExDark / 安装 ultralytics 需要）')
try:
    socket.setdefaulttimeout(6)
    socket.create_connection(('huggingface.co', 443), timeout=6)
    net = True
    detail = 'huggingface.co:443 可连'
except Exception as ex:
    net = False
    detail = f'{type(ex).__name__}: {ex}'
check('外网连通', net, detail)
print(f'  {"PASS" if net else "FAIL"}  huggingface.co:443  {detail}')

# ── 4. 结论 ──
print('\n' + '=' * 78)
print('【结论】')
fails = [c for c in CHECKS if not c[1]]
if not fails:
    print('  全部前置条件满足 → 可以重跑真实检测实验。')
    print('  运行：python exp_real_detect_fixed.py --epochs 15 --limit 800')
else:
    print(f'  有 {len(fails)} 项前置条件不满足 → **真实检测实验无法在本机重跑**。')
    for n, _, d in fails:
        print(f'    - {n}: {d}')
    print('\n  可做的替代方案：')
    print('    (a) 拿到 ExDark 与 yolov8n.pt 后运行 exp_real_detect_fixed.py（已实现无偏评测协议）')
    print('    (b) 若已有原训练的 best.pt / last.pt，可跳过训练直接重评（--skip-build 不适用，需单独评测脚本）')

speed = '未知'
print('\n  仿真实验（E2/E3/E4）不依赖上述任何项 —— 只需 numpy，可独立重跑。')
print('=' * 78)

# 退出码：0=全部满足，1=有缺口（便于脚本化判断）
sys.exit(0 if not fails else 1)

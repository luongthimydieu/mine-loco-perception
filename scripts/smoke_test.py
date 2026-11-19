# -*- coding: utf-8 -*-
r"""Smoke test: 1 epoch, tiny subset, GPU — validates the whole pipeline before the real run."""
import io
import os
import sys
import time

from mine_loco_system.paths import LOGS_DIR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['YOLO_CONFIG_DIR'] = str(LOGS_DIR / '.yolo_cfg')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_real_detect_fixed as F  # noqa

print('DATA   =', F.DATA)
print('WEIGHTS=', F.WEIGHTS, os.path.exists(F.WEIGHTS))
print('CASES  =', F.CASES)

# 只构建一个很小的子集做冒烟
F.build_all(limit=40)
from ultralytics import YOLO
y = os.path.join(F.CASES, 'dustV20_train', 'data.yaml')
print('yaml:', y, os.path.exists(y))
print(open(y, encoding='utf-8').read())

t0 = time.time()
m = YOLO(F.WEIGHTS)
m.train(data=y, epochs=1, imgsz=F.IMGSZ, batch=8, project=F.OUT, name='_smoke',
        exist_ok=True, pretrained=True, verbose=True, seed=0, workers=0, device=0)
print(f'\n1 epoch 耗时 {time.time()-t0:.0f}s')

best = os.path.join(F.OUT, '_smoke', 'weights', 'best.pt')
last = os.path.join(F.OUT, '_smoke', 'weights', 'last.pt')
print('best.pt:', os.path.exists(best), ' last.pt:', os.path.exists(last))

# 用极小子集评测一次，确认 val 通路可用
r = YOLO(last).val(data=os.path.join(F.CASES, 'dustV20_eval', 'data.yaml'),
                   imgsz=F.IMGSZ, conf=0.001, iou=0.6, verbose=False,
                   plots=False, workers=0, device=0)
print('冒烟评测 mAP@0.5 =', round(float(r.box.map50) * 100, 2))
print('SMOKE OK')

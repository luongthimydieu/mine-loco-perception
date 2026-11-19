# -*- coding: utf-8 -*-
"""推理延迟基准：800/5000 模型、416/640、TTA 开关。

输出：
  outputs/results/benchmark_inference.json
  outputs/results/benchmark_inference.csv

注意：单张 batch=1、含前后处理；GPU 为 RTX 5060 Laptop 8 GB。
CPU/其他 GPU 的绝对值会不同，但相对关系可参考。
"""
import argparse
import csv
import json
import os
import statistics
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from ultralytics import YOLO

from mine_loco_system.paths import RESULTS_DIR, TRAIN_DIR, CASES_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(RESULTS_DIR)
IMG_DIR = str(CASES_DIR / '_cases_scaleup' / 'dustV20_eval' / 'val' / 'images')


def pick_image():
    for n in sorted(os.listdir(IMG_DIR)):
        if n.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(IMG_DIR, n)
    raise FileNotFoundError(IMG_DIR)


def bench(model, img, imgsz, augment=False, warmup=5, runs=30):
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, conf=0.25, verbose=False, device=0,
                      augment=augment)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, conf=0.25, verbose=False, device=0,
                      augment=augment)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        'mean_ms': round(statistics.mean(ts), 2),
        'median_ms': round(statistics.median(ts), 2),
        'p95_ms': round(ts[int(0.95 * (len(ts) - 1))], 2),
        'fps_median': round(1000.0 / statistics.median(ts), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Inference latency benchmark")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    if not os.path.isdir(IMG_DIR):
        raise SystemExit("Evaluation images not found: %s\n"
                         "Run scripts/eval_extension.py first." % IMG_DIR)
    img = pick_image()
    cases = [
        ('800_M_clean/416', str(TRAIN_DIR / 'results_real_fixed_long/M_clean/weights/best.pt'), 416, False),
        ('M_clean/640', str(RESULTS_DIR / 'results_scaleup/M_clean/weights/best.pt'), 640, False),
        ('M_phys/640', str(RESULTS_DIR / 'results_scaleup/M_phys/weights/best.pt'), 640, False),
        ('M_phys/640+TTA', str(RESULTS_DIR / 'results_scaleup/M_phys/weights/best.pt'), 640, True),
    ]
    rows = []
    for name, w, isz, aug in cases:
        if not os.path.exists(w):
            print('[missing]', name)
            continue
        m = YOLO(w)
        r = bench(m, img, isz, aug, warmup=args.warmup, runs=args.runs)
        r.update({'case': name, 'imgsz': isz, 'augment': aug})
        rows.append(r)
        print('%-18s mean=%7.2f ms median=%7.2f p95=%7.2f FPS=%6.1f' %
              (name, r['mean_ms'], r['median_ms'], r['p95_ms'], r['fps_median']))

    meta = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'image': os.path.basename(img),
        'batch': 1,
        'note': '单张推理，含前后处理；未做 TensorRT/ONNX 加速。',
        'results': rows,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'benchmark_inference.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'benchmark_inference.csv'), 'w', encoding='utf-8-sig',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'imgsz', 'augment', 'mean_ms', 'median_ms', 'p95_ms', 'fps_median'])
        for r in rows:
            w.writerow([r['case'], r['imgsz'], r['augment'], r['mean_ms'],
                        r['median_ms'], r['p95_ms'], r['fps_median']])
    print('[写出]', os.path.join(OUT, 'benchmark_inference.json'))


if __name__ == '__main__':
    main()

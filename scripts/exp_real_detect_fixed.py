# -*- coding: utf-8 -*-
r"""修正版真实检测实验（PIL 版，避免 cv2 编解码问题）—— 无偏跨域评测协议。

解决的问题：原 `原代码\exp_real_detect.py` 用**同一 valid split、同一能见度、
同一 seed0=90000** 同时生成训练的"内部验证集"与最终评测集 `dustV20_eval`，
因此 YOLO 的 `best.pt` 是在评测集上选出来的 → 19.63% / 46.87% 含模型选择偏差。

本脚本的修正（三刀切开）：
  1. 训练集  : ExDark train split（干净 或 V=20 m 退化），seed0 = 0
  2. 标定集  : valid split 前 50%，固定 seed（CAL_SEED）→ **只用于选点**
  3. 评测集  : valid split 后 50%，用互不相同的 seed 退化 → 两模型都没见过
  4. 报告口径: 同时给 best.pt（含选点偏差）与 last.pt（无选点偏差）

图像读写全部走 PIL，不依赖 cv2。
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
from PIL import Image

from mine_loco_system.paths import (EXDARK_DATA, WEIGHTS_DIR, CASES_DIR,
                                    TRAIN_DIR, LOGS_DIR)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_CFG = str(LOGS_DIR / '.yolo_cfg')
os.makedirs(_CFG, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', _CFG)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = str(EXDARK_DATA)
WEIGHTS = str(WEIGHTS_DIR / 'yolov8n.pt')

# 允许通过环境变量整体平移所有随机种子（多组复制实验：REPRO_SHIFT=101 → 各种子 +101）
_SHIFT = int(os.environ.get('REPRO_SHIFT', '0'))
TAG = os.environ.get('REPRO_TAG', '')

CASES = str(CASES_DIR / ('_cases_fixed' + TAG))
OUT = str(TRAIN_DIR / ('results_real_fixed' + TAG))

IMGSZ = 416
H_LENS = 0.80
H_FOV_DEG = 60.0
AIRLIGHT = 60.0
NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "People", "Table"]

SEED_TRAIN_V20 = 0
CAL_SEED = 70001
EVAL_SEED_CLEAN = 91001
EVAL_SEED_V20 = 91002
EVAL_SEED_V10 = 91003
CAL_FRAC = 0.5

if _SHIFT:
    SEED_TRAIN_V20 += _SHIFT
    CAL_SEED += _SHIFT
    EVAL_SEED_CLEAN += _SHIFT
    EVAL_SEED_V20 += _SHIFT
    EVAL_SEED_V10 += _SHIFT

try:                       # Pillow >= 9.1
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:     # 兼容旧版
    LANCZOS = Image.LANCZOS


def depth_map(h, w):
    """单目地面平面投影：逐像素纵向距离（米）。"""
    f = (w / 2.0) / np.tan(np.deg2rad(H_FOV_DEG / 2.0))
    y_h = h / 2.0
    yy = np.arange(h, dtype=np.float32)[:, None]
    return np.clip(H_LENS * f / np.maximum(yy - y_h, 1.0), 1.0, 200.0)


def degrade(arr, depth, V, seed=0):
    """大气散射退化：I = J·t + A·(1−t)，t = exp(−βd)，β = 3/V。"""
    rng = np.random.default_rng(seed)
    b = 3.0 / V
    t = np.exp(-b * depth)
    out = arr.astype(np.float32) * t + AIRLIGHT * (1.0 - t)
    out = out + rng.normal(0, 1.5, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def list_split(split):
    d = os.path.join(DATA, split, 'images')
    return sorted(n for n in os.listdir(d)
                  if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))


def build(split, dst_root, V, seed0, names=None, limit=None, sub=None):
    """把 split 按工况 V 改写（V=None 表示干净），输出到 dst_root/<sub>/images|labels。"""
    src_img = os.path.join(DATA, split, 'images')
    src_lab = os.path.join(DATA, split, 'labels')
    sub = sub or ('train' if split == 'train' else 'val')
    oi = os.path.join(dst_root, sub, 'images')
    ol = os.path.join(dst_root, sub, 'labels')
    os.makedirs(oi, exist_ok=True)
    os.makedirs(ol, exist_ok=True)
    imgs = list(names) if names is not None else list_split(split)
    if limit:
        imgs = imgs[:limit]
    n = 0
    for i, name in enumerate(imgs):
        p = os.path.join(src_img, name)
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                nh = max(2, int(IMGSZ * im.height / im.width))
                im = im.resize((IMGSZ, nh), LANCZOS)
                arr = np.asarray(im)
        except Exception:
            continue
        if V is not None:
            arr = degrade(arr, depth_map(arr.shape[0], IMGSZ), V, seed0 + i)
        Image.fromarray(arr).save(os.path.join(oi, name), quality=92)
        stem = os.path.splitext(name)[0]
        s_lab = os.path.join(src_lab, stem + '.txt')
        d_lab = os.path.join(ol, stem + '.txt')
        if os.path.exists(s_lab):
            shutil.copyfile(s_lab, d_lab)
        else:
            open(d_lab, 'w').close()
        n += 1
    return n


def make_yaml(root):
    os.makedirs(root, exist_ok=True)
    txt = (f"path: {root.replace(chr(92), '/')}\n"
           f"train: train/images\nval: val/images\n"
           f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    p = os.path.join(root, 'data.yaml')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(txt)
    return p


def build_all(limit=None):
    val = list_split('valid')
    n_cal = int(len(val) * CAL_FRAC)
    cal, ev = val[:n_cal], val[n_cal:]
    print(f'[数据] valid {len(val)} → 标定 {len(cal)} / 评测 {len(ev)}', flush=True)
    if limit:
        cal, ev = cal[:limit], ev[:limit]

    for tag, V, sd in (('clean_train', None, 0), ('dustV20_train', 20.0, SEED_TRAIN_V20)):
        root = os.path.join(CASES, tag)
        ntr = build('train', root, V, sd, limit=limit)
        nva = build('valid', root, V, CAL_SEED, names=cal, sub='val')
        make_yaml(root)
        print(f'      {tag:14s} 训练 {ntr} / 标定 {nva}', flush=True)

    for tag, V, sd in (('clean_eval', None, EVAL_SEED_CLEAN),
                       ('dustV20_eval', 20.0, EVAL_SEED_V20),
                       ('dustV10_eval', 10.0, EVAL_SEED_V10)):
        root = os.path.join(CASES, tag)
        n = build('valid', root, V, sd, names=ev, sub='train')
        shutil.copytree(os.path.join(root, 'train'), os.path.join(root, 'val'),
                        dirs_exist_ok=True)
        make_yaml(root)
        print(f'      {tag:14s} 评测 {n}', flush=True)


def _patch_threadpool():
    """避开沙箱对命名管道的限制。

    ultralytics 的 StarDataset.__init__ 用 multiprocessing.pool.ThreadPool 做并行
    标签校验，而 ThreadPool 内部通过命名管道通信；本机沙箱禁止打开命名管道
    （PermissionError: [WinError 5]）。这里把 ThreadPool 替换成单线程的等价实现，
    只影响"预扫描标签"这一环节，不改变任何数值结果。
    """
    class _SeqPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap(self, func=None, iterable=None, **k):
            return (func(x) for x in (iterable or []))

        def map(self, func, iterable, **k):
            return [func(x) for x in iterable]

    import ultralytics.utils as _u
    try:
        _u.NUM_THREADS = 1
    except Exception:
        pass
    import ultralytics.data.dataset as _ds
    _ds.ThreadPool = _SeqPool
    _ds.NUM_THREADS = 1
    print('[patch] ThreadPool → 单线程（规避沙箱命名管道限制）', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=IMGSZ)
    ap.add_argument('--device', default='0')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='仅 1 epoch / 少量数据')
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if not args.skip_build:
        print('[1/3] 构建数据集（修正协议：标定集与评测集分离）...', flush=True)
        build_all(args.limit)

    from ultralytics import YOLO
    _patch_threadpool()
    plans = [('M_clean', 'clean_train'), ('M_phys', 'dustV20_train')]
    table = {}
    print('[2/3] 训练 + 交叉评测 ...', flush=True)
    for name, trainset in plans:
        y = os.path.join(CASES, trainset, 'data.yaml')
        t1 = time.time()
        print(f'\n  >>> 训练 {name} ← {trainset}  epochs={args.epochs} '
              f'batch={args.batch} imgsz={args.imgsz} device={args.device}', flush=True)
        YOLO(WEIGHTS).train(data=y, epochs=args.epochs, imgsz=args.imgsz,
                            batch=args.batch, project=OUT, name=name, exist_ok=True,
                            pretrained=True, verbose=False, seed=0, workers=0,
                            device=args.device, plots=False, val=True)
        print(f'  <<< {name} 训练完成 {time.time()-t1:.0f}s', flush=True)
        table[name] = {}
        for wtag in ('best', 'last'):
            wp = os.path.join(OUT, name, 'weights', f'{wtag}.pt')
            if not os.path.exists(wp):
                continue
            table[name][wtag] = {}
            for vtag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
                r = YOLO(wp).val(data=os.path.join(CASES, vtag, 'data.yaml'),
                                 imgsz=args.imgsz, conf=0.001, iou=0.6, verbose=False,
                                 plots=False, workers=0, device=args.device)
                table[name][wtag][vtag] = {
                    'mAP@0.5': round(float(r.box.map50) * 100, 2),
                    'mAP@0.5:0.95': round(float(r.box.map) * 100, 2),
                    'Precision': round(float(r.box.mp) * 100, 2),
                    'Recall': round(float(r.box.mr) * 100, 2),
                }
                print(f'      {name}/{wtag}/{vtag}: mAP50='
                      f'{table[name][wtag][vtag]["mAP@0.5"]}', flush=True)

    rows = []
    for vtag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
        for wtag in ('best', 'last'):
            b = table.get('M_clean', {}).get(wtag, {}).get(vtag, {}).get('mAP@0.5')
            p = table.get('M_phys', {}).get(wtag, {}).get(vtag, {}).get('mAP@0.5')
            if b is not None and p is not None:
                rows.append({'权重': wtag, '评测集': vtag, 'M_clean': b,
                             'M_phys': p, '变化(百分点)': round(p - b, 2)})

    payload = {
        '协议说明': {
            '修正点': ['标定集（valid 前 50%）与评测集（valid 后 50%）分离',
                     '评测集退化种子与训练/标定互不相同',
                     '同时报告 best.pt（含选点偏差）与 last.pt（无选点偏差）'],
            '种子': {'训练集退化': SEED_TRAIN_V20, '标定集': CAL_SEED,
                    '评测集_clean': EVAL_SEED_CLEAN,
                    '评测集_dustV20': EVAL_SEED_V20,
                    '评测集_dustV10': EVAL_SEED_V10},
            '训练轮数': args.epochs, '输入尺寸': args.imgsz,
            'batch': args.batch, '训练规模上限': args.limit,
            '实现说明': '图像读写用 PIL，未使用 cv2（本机 cv2 5.0.0 JPEG 解码异常）',
        },
        '交叉评测': table,
        '口径对照': rows,
        '耗时秒': round(time.time() - t0, 1),
    }
    p = os.path.join(OUT, 'real_metrics_fixed.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n结果 -> {p}', flush=True)
    for r in rows:
        print(f"  {r['权重']:4s} {r['评测集']:14s} M_clean={r['M_clean']:6.2f} "
              f"M_phys={r['M_phys']:6.2f}  Δ={r['变化(百分点)']:+.2f}pp", flush=True)


if __name__ == '__main__':
    main()

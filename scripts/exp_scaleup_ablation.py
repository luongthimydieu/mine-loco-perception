# -*- coding: utf-8 -*-
r"""扩大训练规模重跑 + 传统增广消融对照（研究报告 6.3 节待办 4 的落实）。

对应《研究报告》第 4.4 节结论边界的两个未决问题：
  (1) 训练规模 800 张 / 输入 416 / 15~80 轮时"物理引导退化训练无增益"的结论，
      在 5000 张 / 640 px / 80 轮（充分收敛）下是否成立；
  (2) 该结论是"物理引导增广"特有的，还是一切合成退化增广共有的——
      补"高斯噪声（非物理）增广"与"仅光度（HSV 类）增广"两个对照。

四个模型（唯一变量 = 训练集的离线退化方式，其余完全一致）：
  M_clean  干净（YOLO 内置增广照常开启）          —— 常规做法基线
  M_phys   物理散射退化 V=20 m（β=3/V，逐像素深度）—— 物理引导
  M_noise  均匀变暗 + 高斯噪声（无深度结构），
           全局透射率与 M_phys 的全图平均值对齐    —— 非物理对照
  M_hsv    仅光度抖动（亮度/伽马/通道增益，无噪声、无散射）—— 传统光度对照

评测协议与 exp_real_detect_fixed.py 完全一致（三刀切开）：
  标定集 = valid 前 50%（368 张，CAL_SEED）
  评测集 = valid 后 50%（368 张，EVAL_SEED_*，与训练/标定互异）
  同时报告 best.pt（含选点偏差）与 last.pt（无选点偏差）。

运行（建议用本目录 run_scaleup.bat 后台启动）：
  python exp_scaleup_ablation.py --epochs 80 --batch 16 --imgsz 640
输出：
  results_scaleup/real_metrics_scaleup.json
  results_scaleup/对照表.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from mine_loco_system.paths import (EXDARK_DATA, WEIGHTS_DIR, CASES_DIR,
                                    RESULTS_DIR, LOGS_DIR)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_CFG = str(LOGS_DIR / '.yolo_cfg')
os.makedirs(_CFG, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', _CFG)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = str(EXDARK_DATA)
WEIGHTS = str(WEIGHTS_DIR / 'yolov8n.pt')

CASES = str(CASES_DIR / '_cases_scaleup')
OUT = str(RESULTS_DIR / 'results_scaleup')

IMGSZ = 640
N_TRAIN = 5000
H_LENS = 0.80
H_FOV_DEG = 60.0
AIRLIGHT = 60.0
NOISE = 1.5
NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "People", "Table"]

SEED_TRAIN = 0
CAL_SEED = 70001
EVAL_SEED_CLEAN = 91001
EVAL_SEED_V20 = 91002
EVAL_SEED_V10 = 91003
CAL_FRAC = 0.5
V_DEGRADE = 20.0

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


# ---------------------------------------------------------------- 物理模型
def depth_map(h, w):
    f = (w / 2.0) / np.tan(np.deg2rad(H_FOV_DEG / 2.0))
    y_h = h / 2.0
    yy = np.arange(h, dtype=np.float32)[:, None]
    return np.clip(H_LENS * f / np.maximum(yy - y_h, 1.0), 1.0, 200.0)


def degrade_phys(arr, depth, V, seed):
    """物理散射：I = J·t + A(1−t)，t = exp(−β·d)。"""
    rng = np.random.default_rng(seed)
    b = 3.0 / V
    t = np.exp(-b * depth)
    out = arr.astype(np.float32) * t[:, :, None] + AIRLIGHT * (1.0 - t[:, :, None])
    out = out + rng.normal(0, NOISE, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_noise(arr, depth, V, seed):
    """非物理对照：全图均匀变暗（透射率取物理退化全图均值）+ 高斯噪声。

    与 M_phys 的全局灰度统计一致，唯一差别 = 无深度相关的空间结构。
    """
    rng = np.random.default_rng(seed)
    b = 3.0 / V
    t_mean = float(np.mean(np.exp(-b * depth)))
    out = arr.astype(np.float32) * t_mean + AIRLIGHT * (1.0 - t_mean)
    out = out + rng.normal(0, NOISE, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_hsv(arr, depth, V, seed):
    """仅光度（HSV 类）对照：亮度-伽马-通道增益随机抖动，无噪声、无散射。"""
    rng = np.random.default_rng(seed)
    a = arr.astype(np.float32) / 255.0
    gain = rng.uniform(0.35, 0.75)
    gamma = rng.uniform(0.8, 1.4)
    a = np.power(np.clip(a * gain, 0, 1), gamma)
    ch = rng.uniform(0.9, 1.1, size=(1, 1, 3))
    return (np.clip(a * ch, 0.0, 1.0) * 255).astype(np.uint8)


DEGRADES = {
    'clean_train': None,
    'physV20_train': lambda a, d, s: degrade_phys(a, d, V_DEGRADE, s),
    'noise_train': lambda a, d, s: degrade_noise(a, d, V_DEGRADE, s),
    'hsv_train': lambda a, d, s: degrade_hsv(a, d, V_DEGRADE, s),
}


# ---------------------------------------------------------------- 数据构建
def list_split(split):
    d = os.path.join(DATA, split, 'images')
    return sorted(n for n in os.listdir(d)
                  if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))


def build(split, dst_root, degr_fn, seed0, names=None, sub=None):
    """把 split 按退化函数改写输出（degr_fn=None 表示干净）。"""
    src_img = os.path.join(DATA, split, 'images')
    src_lab = os.path.join(DATA, split, 'labels')
    sub = sub or ('train' if split == 'train' else 'val')
    oi = os.path.join(dst_root, sub, 'images')
    ol = os.path.join(dst_root, sub, 'labels')
    os.makedirs(oi, exist_ok=True)
    os.makedirs(ol, exist_ok=True)
    imgs = list(names) if names is not None else list_split(split)
    jobs = [(os.path.join(src_img, n), src_lab, oi, ol, degr_fn,
             seed0 + i, n) for i, n in enumerate(imgs)]

    def work(job):
        src_img_p, src_lab_p, oi_p, ol_p, fn, seed, name = job
        try:
            with Image.open(src_img_p) as im:
                im = im.convert('RGB')
                nh = max(2, int(IMGSZ * im.height / im.width))
                im = im.resize((IMGSZ, nh), LANCZOS)
                arr = np.asarray(im)
        except Exception:
            return False
        if fn is not None:
            arr = fn(arr, depth_map(arr.shape[0], IMGSZ), seed)
        Image.fromarray(arr).save(os.path.join(oi_p, name), quality=92)
        stem = os.path.splitext(name)[0]
        s_lab = os.path.join(src_lab_p, stem + '.txt')
        d_lab = os.path.join(ol_p, stem + '.txt')
        if os.path.exists(s_lab):
            with open(s_lab, 'rb') as f, open(d_lab, 'wb') as g:
                g.write(f.read())
        else:
            open(d_lab, 'w').close()
        return True

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, jobs))
    return sum(1 for r in results if r)


def make_yaml(root):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"path: {root.replace(chr(92), '/')}\n"
                f"train: train/images\nval: val/images\n"
                f"nc: {len(NAMES)}\nnames: {NAMES}\n")


def build_all():
    val = list_split('valid')
    n_cal = int(len(val) * CAL_FRAC)
    cal, ev = val[:n_cal], val[n_cal:]
    print(f'[数据] valid {len(val)} → 标定 {len(cal)} / 评测 {len(ev)}', flush=True)
    train_imgs = list_split('train')[:N_TRAIN]
    print(f'[数据] train 取前 {len(train_imgs)} 张', flush=True)

    # 四个训练集（各含同一标定集）
    for tag, fn in DEGRADES.items():
        root = os.path.join(CASES, tag)
        done_mark = os.path.join(root, '.done')
        if os.path.exists(done_mark):
            print(f'  {tag}: 已构建，跳过', flush=True)
            continue
        t0 = time.time()
        ntr = build('train', root, fn, SEED_TRAIN, names=train_imgs)
        nva = build('valid', root, fn, CAL_SEED, names=cal, sub='val')
        make_yaml(root)
        open(done_mark, 'w').close()
        print(f'  {tag}: 训练 {ntr} / 标定 {nva}  ({time.time()-t0:.0f}s)',
              flush=True)

    # 三个评测集（干净 / V=20 / V=10，种子互异且与训练/标定互异）
    evals = [('clean_eval', None, EVAL_SEED_CLEAN),
             ('dustV20_eval', 'phys20', EVAL_SEED_V20),
             ('dustV10_eval', 'phys10', EVAL_SEED_V10)]
    for tag, kind, sd in evals:
        root = os.path.join(CASES, tag)
        if os.path.exists(os.path.join(root, '.done')):
            print(f'  {tag}: 已构建，跳过', flush=True)
            continue
        if kind is None:
            fn = None
        else:
            V = 20.0 if kind == 'phys20' else 10.0
            fn = (lambda a, d, s, V=V: degrade_phys(a, d, V, s))
        n = build('valid', root, fn, sd, names=ev, sub='train')
        # val 目录与 train 相同（评测集自评）
        import shutil
        shutil.copytree(os.path.join(root, 'train'), os.path.join(root, 'val'),
                        dirs_exist_ok=True)
        make_yaml(root)
        open(os.path.join(root, '.done'), 'w').close()
        print(f'  {tag}: 评测 {n}', flush=True)


# ---------------------------------------------------------------- 训练评测
def _patch_threadpool():
    """规避沙箱/Windows 命名管道限制（同 exp_real_detect_fixed.py）。"""
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


PLANS = [('M_clean', 'clean_train'), ('M_phys', 'physV20_train'),
         ('M_noise', 'noise_train'), ('M_hsv', 'hsv_train')]


def train_and_eval(epochs, batch, imgsz, device, only=None):
    from ultralytics import YOLO
    _patch_threadpool()
    table = {}
    os.makedirs(OUT, exist_ok=True)
    for name, trainset in PLANS:
        if only and name not in only:
            continue
        y = os.path.join(CASES, trainset, 'data.yaml')
        t1 = time.time()
        print(f'\n>>> 训练 {name} ← {trainset}  epochs={epochs} '
              f'batch={batch} imgsz={imgsz}', flush=True)
        YOLO(WEIGHTS).train(data=y, epochs=epochs, imgsz=imgsz, batch=batch,
                            project=OUT, name=name, exist_ok=True,
                            pretrained=True, verbose=False, seed=0,
                            workers=0, device=device, plots=False, val=True)
        print(f'<<< {name} 完成 {time.time()-t1:.0f}s', flush=True)
        table[name] = {}
        for wtag in ('best', 'last'):
            wp = os.path.join(OUT, name, 'weights', f'{wtag}.pt')
            if not os.path.exists(wp):
                continue
            table[name][wtag] = {}
            for vtag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
                r = YOLO(wp).val(data=os.path.join(CASES, vtag, 'data.yaml'),
                                 imgsz=imgsz, conf=0.001, iou=0.6,
                                 verbose=False, plots=False, workers=0,
                                 device=device)
                table[name][wtag][vtag] = {
                    'mAP@0.5': round(float(r.box.map50) * 100, 2),
                    'mAP@0.5:0.95': round(float(r.box.map) * 100, 2),
                    'Precision': round(float(r.box.mp) * 100, 2),
                    'Recall': round(float(r.box.mr) * 100, 2),
                }
                print(f'    {name}/{wtag}/{vtag}: '
                      f'mAP50={table[name][wtag][vtag]["mAP@0.5"]}', flush=True)
        # 每个模型训完立刻落盘，防中断丢失
        dump_table(table)


def dump_table(table):
    p = os.path.join(OUT, 'real_metrics_scaleup.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f'[落盘] {p}', flush=True)


def write_report(table, epochs, imgsz, elapsed):
    rows = []
    for vtag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
        for wtag in ('best', 'last'):
            vals = {n: table.get(n, {}).get(wtag, {}).get(vtag, {}).get('mAP@0.5')
                    for n, _ in PLANS}
            if vals['M_clean'] is None:
                continue
            rows.append({'评测集': vtag, '权重': wtag,
                         'M_clean': vals['M_clean'], 'M_phys': vals['M_phys'],
                         'M_noise': vals['M_noise'], 'M_hsv': vals['M_hsv'],
                         'Δphys−clean': (round(vals['M_phys'] - vals['M_clean'], 2)
                                          if vals['M_phys'] is not None else None),
                         'Δnoise−clean': (round(vals['M_noise'] - vals['M_clean'], 2)
                                           if vals['M_noise'] is not None else None),
                         'Δhsv−clean': (round(vals['M_hsv'] - vals['M_clean'], 2)
                                        if vals['M_hsv'] is not None else None)})
    csv_p = os.path.join(OUT, '对照表.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['评测集', '权重', 'M_clean', 'M_phys', 'M_noise',
                            'M_hsv'])
        w.writeheader()
        w.writerows(rows)

    payload = {
        '实验': '扩大训练规模重跑 + 传统增广消融（研究报告 6.3 待办 4/6 的落实）',
        '协议': {
            '训练规模': N_TRAIN, '输入尺寸': imgsz, '轮数': epochs,
            '标定集': 'valid 前 50%%（368 张，种子 %d）' % CAL_SEED,
            '评测集': 'valid 后 50%%（368 张，种子 %d/%d/%d）' % (
                EVAL_SEED_CLEAN, EVAL_SEED_V20, EVAL_SEED_V10),
            '说明': '四个模型唯一变量为训练集离线退化方式；'
                    'M_noise 的均匀透射率与 M_phys 全图均值对齐；'
                    'YOLO 内置增广（HSV/翻转/马赛克）对所有模型一致开启。',
        },
        '交叉评测': table,
        '对照表': rows,
        '耗时秒': round(elapsed, 1),
    }
    with open(os.path.join(OUT, 'real_metrics_scaleup.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('\n===== 对照表（mAP@0.5，%）=====', flush=True)
    for r in rows:
        print('  {评测集:12s} {权重:4s} clean={M_clean:6.2f} phys={M_phys:6.2f} '
              'noise={M_noise} hsv={M_hsv}'.format(**r), flush=True)
    print(f'\n结果 -> {os.path.join(OUT, "real_metrics_scaleup.json")}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=IMGSZ)
    ap.add_argument('--device', default='0')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--only', default=None,
                    help='仅训练指定模型，逗号分隔：M_clean,M_phys')
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_build:
        print('[1/3] 构建数据集（5000 张 × 4 种退化 + 3 个评测集）...', flush=True)
        build_all()
    only = set(args.only.split(',')) if args.only else None
    print('[2/3] 训练 + 交叉评测 ...', flush=True)
    train_and_eval(args.epochs, args.batch, args.imgsz, args.device, only)
    table_p = os.path.join(OUT, 'real_metrics_scaleup.json')
    if os.path.exists(table_p):
        with open(table_p, encoding='utf-8') as f:
            table = json.load(f)
        write_report(table if '交叉评测' not in table else table['交叉评测'],
                     args.epochs, args.imgsz, time.time() - t0)
    print('[3/3] 完成', flush=True)


if __name__ == '__main__':
    main()

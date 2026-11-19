# -*- coding: utf-8 -*-
r"""扩展评测：能见度曲线 + TTA + 模型汤（部署侧优化）。

在既有 clean / V=20 / V=10 评测集之外，新增 V=30 / V=15 两个评测集，
并对扩训四模型、800 张两模型、模型汤（M_clean+M_phys 权重平均）与
TTA（多尺度/翻转增强推理）做统一评测。

所有评测集均为 valid 后 50%（368 张），退化种子互异且与训练/标定不同。
输出：
  outputs/results/extension_eval.json
  outputs/results/extension_eval.csv
  outputs/figures/fig9_visibility_curve.png
  outputs/figures/fig10_tta_soup.png
"""
import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from ultralytics import YOLO

from mine_loco_system.paths import (CASES_DIR, RESULTS_DIR, TRAIN_DIR,
                                    FIGURES_DIR, LOGS_DIR)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(RESULTS_DIR)
EXT = str(CASES_DIR / '_cases_ext')
FIG = str(FIGURES_DIR)
SCALE = str(RESULTS_DIR / 'results_scaleup')
TRAIN800 = str(TRAIN_DIR)

# 复用扩训脚本的数据构建函数（导入不会训练）
sys.path.insert(0, HERE)
import exp_scaleup_ablation as S  # noqa: E402

EVAL_TAGS = ['clean_eval', 'visV30_eval', 'dustV20_eval', 'visV15_eval',
             'dustV10_eval']
NEW = {'visV30_eval': (30.0, 91004), 'visV15_eval': (15.0, 91005)}


def build_new_evals():
    val = S.list_split('valid')
    n_cal = int(len(val) * S.CAL_FRAC)
    ev = val[n_cal:]
    os.makedirs(EXT, exist_ok=True)
    for tag, (V, seed) in NEW.items():
        root = os.path.join(EXT, tag)
        done = os.path.join(root, '.done')
        if os.path.exists(done):
            print('[skip]', tag)
            continue
        t0 = time.time()
        n = S.build('valid', root, lambda a, d, s, V=V: S.degrade_phys(a, d, V, s),
                    seed, names=ev, sub='train')
        shutil.copytree(os.path.join(root, 'train'),
                        os.path.join(root, 'val'), dirs_exist_ok=True)
        S.make_yaml(root)
        open(done, 'w').write('ok')
        print('[build] %s  %d 张  %.1fs' % (tag, n, time.time() - t0))


def yaml_of(tag):
    base = EXT if tag in NEW else str(CASES_DIR / '_cases_scaleup')
    return os.path.join(base, tag, 'data.yaml')


def quiet_val(model, yaml, imgsz, augment=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=yaml, imgsz=imgsz, conf=0.001, iou=0.6,
                      verbose=False, plots=False, workers=0, device=0,
                      augment=augment,
                      project=str(LOGS_DIR / 'ultralytics_val'),
                      name='val', exist_ok=True)
    return {
        'mAP50': round(float(r.box.map50) * 100, 3),
        'mAP50_95': round(float(r.box.map) * 100, 3),
        'P': round(float(r.box.mp) * 100, 3),
        'R': round(float(r.box.mr) * 100, 3),
    }


def soup_state(w1, w2):
    c1 = torch.load(w1, map_location='cpu', weights_only=False)
    c2 = torch.load(w2, map_location='cpu', weights_only=False)
    s1 = (c1.get('ema') or c1['model']).float().state_dict()
    s2 = (c2.get('ema') or c2['model']).float().state_dict()
    if set(s1) != set(s2):
        raise RuntimeError('state_dict 键不一致，无法做模型汤')
    return {k: ((s1[k] + s2[k]) / 2.0 if s1[k].dtype.is_floating_point else s1[k])
            for k in s1}


def load_soup(w1, state):
    m = YOLO(w1)
    m.model.load_state_dict(state)
    m.model.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--skip-tta', action='store_true')
    args = ap.parse_args()

    if not args.skip_build:
        build_new_evals()

    models = [
        ('800_M_clean', os.path.join(TRAIN800, 'results_real_fixed_long', 'M_clean', 'weights', 'best.pt'), 416),
        ('800_M_phys', os.path.join(TRAIN800, 'results_real_fixed_long', 'M_phys', 'weights', 'best.pt'), 416),
        ('M_clean', os.path.join(SCALE, 'M_clean', 'weights', 'best.pt'), 640),
        ('M_phys', os.path.join(SCALE, 'M_phys', 'weights', 'best.pt'), 640),
        ('M_noise', os.path.join(SCALE, 'M_noise', 'weights', 'best.pt'), 640),
        ('M_hsv', os.path.join(SCALE, 'M_hsv', 'weights', 'best.pt'), 640),
    ]
    results = {}
    for name, w, isz in models:
        if not os.path.exists(w):
            print('[missing]', name, w)
            continue
        model = YOLO(w)
        results[name] = {}
        for tag in EVAL_TAGS:
            t0 = time.time()
            results[name][tag] = quiet_val(model, yaml_of(tag), isz)
            print('%-12s %-12s mAP50=%6.2f  (%.1fs)' %
                  (name, tag, results[name][tag]['mAP50'], time.time() - t0))

    # TTA：对干净基线与物理引导模型
    if not args.skip_tta:
        for name in ('M_clean', 'M_phys'):
            w = os.path.join(SCALE, name, 'weights', 'best.pt')
            model = YOLO(w)
            results[name + '+TTA'] = {}
            for tag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
                t0 = time.time()
                results[name + '+TTA'][tag] = quiet_val(model, yaml_of(tag), 640,
                                                       augment=True)
                print('%-12s %-12s mAP50=%6.2f  (%.1fs)' %
                      (name + '+TTA', tag, results[name + '+TTA'][tag]['mAP50'],
                       time.time() - t0))

    # 模型汤：M_clean + M_phys 权重平均
    w1 = os.path.join(SCALE, 'M_clean', 'weights', 'best.pt')
    w2 = os.path.join(SCALE, 'M_phys', 'weights', 'best.pt')
    if os.path.exists(w1) and os.path.exists(w2):
        st = soup_state(w1, w2)
        model = load_soup(w1, st)
        results['soup'] = {}
        for tag in EVAL_TAGS:
            t0 = time.time()
            results['soup'][tag] = quiet_val(model, yaml_of(tag), 640)
            print('%-12s %-12s mAP50=%6.2f  (%.1fs)' %
                  ('soup', tag, results['soup'][tag]['mAP50'], time.time() - t0))
        if not args.skip_tta:
            model = load_soup(w1, st)
            results['soup+TTA'] = {}
            for tag in ('clean_eval', 'dustV20_eval', 'dustV10_eval'):
                t0 = time.time()
                results['soup+TTA'][tag] = quiet_val(model, yaml_of(tag), 640,
                                                     augment=True)
                print('%-12s %-12s mAP50=%6.2f  (%.1fs)' %
                      ('soup+TTA', tag, results['soup+TTA'][tag]['mAP50'],
                       time.time() - t0))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'extension_eval.json'), 'w', encoding='utf-8') as f:
        json.dump({'eval_sets': EVAL_TAGS, 'results': results}, f,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'extension_eval.csv'), 'w', encoding='utf-8-sig',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'eval_set', 'mAP50', 'mAP50_95', 'P', 'R'])
        for name, d in results.items():
            for tag, m in d.items():
                w.writerow([name, tag, m['mAP50'], m['mAP50_95'], m['P'], m['R']])
    print('[写出]', os.path.join(OUT, 'extension_eval.json'))

    make_figs(results)
    return results


def make_figs(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for f in ('Microsoft YaHei', 'SimHei'):
        try:
            matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']
            break
        except Exception:
            pass
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 300
    os.makedirs(FIG, exist_ok=True)

    xs = [100, 30, 20, 15, 10]  # 100 代表干净（无粉尘）
    tags = ['clean_eval', 'visV30_eval', 'dustV20_eval', 'visV15_eval',
            'dustV10_eval']
    colors = {'M_clean': 'tab:blue', 'M_phys': 'tab:red', 'M_noise': 'tab:green',
              'M_hsv': 'tab:orange', '800_M_clean': 'tab:blue',
              '800_M_phys': 'tab:red', 'soup': 'tab:purple'}
    styles = {'M_clean': '-', 'M_phys': '-', 'M_noise': '-', 'M_hsv': '-',
              '800_M_clean': '--', '800_M_phys': '--', 'soup': '-.'}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in ('M_clean', 'M_phys', 'M_noise', 'M_hsv', 'soup'):
        if name not in results:
            continue
        ys = [results[name][t]['mAP50'] for t in tags]
        ax.plot(xs, ys, styles.get(name, '-'), color=colors.get(name), marker='o',
                lw=2, label=name)
    for name in ('800_M_clean', '800_M_phys'):
        if name not in results:
            continue
        ys = [results[name][t]['mAP50'] for t in tags]
        ax.plot(xs, ys, '--', color=colors.get(name), marker='s', lw=1.6,
                alpha=0.75, label=name + '（800张/416）')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(['干净', '30', '20', '15', '10'])
    ax.set_xlabel('能见度 V (m) ｜ 干净 = 无粉尘')
    ax.set_ylabel('mAP@0.5 (%)')
    ax.set_title('不同能见度下的检测性能（368 张独立评测集）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out1 = os.path.join(FIG, 'fig9_visibility_curve.png')
    fig.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print('[图]', out1)

    # TTA / soup 对比
    groups = ['clean_eval', 'dustV20_eval', 'dustV10_eval']
    gnames = ['干净', 'V=20 m', 'V=10 m']
    series = [('M_clean', 'M_clean'), ('M_clean+TTA', 'M_clean+TTA'),
              ('M_phys', 'M_phys'), ('M_phys+TTA', 'M_phys+TTA'),
              ('soup', 'soup'), ('soup+TTA', 'soup+TTA')]
    if all(k in results for k, _ in series):
        x = np.arange(len(groups))
        width = 0.12
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for i, (key, lab) in enumerate(series):
            ys = [results[key][g]['mAP50'] for g in groups]
            ax.bar(x + (i - 2.5) * width, ys, width, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(gnames)
        ax.set_ylabel('mAP@0.5 (%)')
        ax.set_title('部署侧优化：TTA 与模型汤（5000 张 / 640 px / 80 轮）')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        out2 = os.path.join(FIG, 'fig10_tta_soup.png')
        fig.savefig(out2, bbox_inches='tight')
        plt.close(fig)
        print('[图]', out2)


if __name__ == '__main__':
    main()

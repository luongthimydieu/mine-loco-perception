# -*- coding: utf-8 -*-
r"""泄漏诊断实验：检验"评测集与训练期验证集同源"能带来多大虚高。

思路（不需重新训练，只做评测）：
  原实现让 M_phys 的**训练内部验证集**与 dustV20_eval 用同一批图（同 seed0=90000）。
  本诊断构造一个"半泄漏"评测集：
    - 前一半图像用**训练期标定种子**（CAL_SEED，即模型选点见过的域）
    - 后一半图像用**独立评测种子**（EVAL_SEED_V20）
  再分别用 M_clean 评测"纯独立"与"半泄漏"，比较 mAP 差异。
  若泄漏能显著抬高 mAP，说明原报告的 +3.89 至少部分来自口径问题。
"""
import io
import json
import os
import shutil
import sys

from mine_loco_system.paths import LOGS_DIR, REPRO_DIR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
_CFG = str(LOGS_DIR / '.yolo_cfg')
os.makedirs(_CFG, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', _CFG)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_real_detect_fixed as F  # noqa

D = str(REPRO_DIR / '_leak_diag')
shutil.rmtree(D, ignore_errors=True)

val = F.list_split('valid')
n_cal = int(len(val) * F.CAL_FRAC)
ev = val[n_cal:]                      # 独立评测用的图像（后 50%）
half = len(ev) // 2
seen_part = ev[:half]                 # 用标定种子退化 → "曾经见过"
fresh_part = ev[half:]                # 用独立种子退化 → 全新

print(f'valid {len(val)} → 评测池 {len(ev)}：半泄漏 {len(seen_part)} + 独立 {len(fresh_part)}', flush=True)

V20 = 20.0
# A) 纯独立评测集
rootA = os.path.join(D, 'pure')
F.build('valid', rootA, V20, F.EVAL_SEED_V20, names=ev, sub='train')
shutil.copytree(os.path.join(rootA, 'train'), os.path.join(rootA, 'val'), dirs_exist_ok=True)
F.make_yaml(rootA)

# B) 半泄漏评测集：一半用标定种子（=选点见过的域），一半用独立种子
rootB = os.path.join(D, 'leaky')
nb1 = F.build('valid', rootB, V20, F.CAL_SEED, names=seen_part, sub='train')
nb2 = F.build('valid', rootB, V20, F.EVAL_SEED_V20, names=fresh_part, sub='train')
shutil.copytree(os.path.join(rootB, 'train'), os.path.join(rootB, 'val'), dirs_exist_ok=True)
F.make_yaml(rootB)
print(f'  半泄漏集：标定种子 {nb1} 张 + 独立种子 {nb2} 张', flush=True)

from ultralytics import YOLO
F._patch_threadpool()

res = {}
for tag, root in (('pure_independent', rootA), ('half_leaky', rootB)):
    w = os.path.join(F.OUT, 'M_clean', 'weights', 'best.pt')
    r = YOLO(w).val(data=os.path.join(root, 'data.yaml'), imgsz=F.IMGSZ,
                    conf=0.001, iou=0.6, verbose=False, plots=False,
                    workers=0, device='0')
    res[tag] = {'n': nb1 + nb2 if tag == 'half_leaky' else len(ev),
                'mAP@0.5': round(float(r.box.map50) * 100, 2),
                'mAP@0.5:0.95': round(float(r.box.map) * 100, 2)}
    print(f'  {tag:18s} n={res[tag]["n"]:4d}  mAP@0.5={res[tag]["mAP@0.5"]:.2f}', flush=True)

delta = res['half_leaky']['mAP@0.5'] - res['pure_independent']['mAP@0.5']
print(f'\n半泄漏 − 纯独立 = {delta:+.2f} 个百分点', flush=True)
print('解读：把一半评测图换成"选点见过的退化域"，mAP 抬高约 '
      f'{delta:.2f} 个百分点；原实现是**全部**评测图都与选点同域。', flush=True)

out = os.path.join(ROOT, '泄漏诊断结果.json')
json.dump({'说明': '同一个 M_clean/best.pt，在两个不同评测集上评测；差异只来自评测集构成',
           '结果': res, '半泄漏减纯独立(百分点)': round(delta, 2),
           '备注': '半泄漏集 = 前一半图用标定种子(选点见过的域)、后一半用独立种子'},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('->', out, flush=True)

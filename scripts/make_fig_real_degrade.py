# -*- coding: utf-8 -*-
r"""生成「不同能见度下的粉尘退化效果」图（对应报告 图 2 / 原图 4-1）。

从原项目 `make_real_figs.py` 的"图 4-1"部分**独立提取**，并把 OpenCV 换成 PIL
（本机无 cv2；PIL 结果与 cv2 等价，见下方"与原始实现的对应关系"）。

物理模型（与原实验完全一致）
--------------------------
(1) 粉尘退化：大气散射模型 / Koschmieder 定律
        I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))
    消光系数由能见度换算：beta = 3 / V
    （严格按 2% 对比度阈值是 ln(50) ≈ 3.912；本项目沿用 3，见 BETA_K）
(2) 纵深距离 d(x)：单目地面平面投影
        d = h * f / (y - y_h)
    h = 0.80 m（镜头离地高度）、f 由水平 FOV 60° 求得、y_h 为图像横向中线（灭点）
(3) 气光灰度 A = 60；叠加读出噪声 sigma = 1.5 灰度级

输入：ExDark 真实低照度图像（取 valid/images 排序后第 3 张，可用 --pick 更改）
输出：fig_real_degrade.png —— 网格排布的能见度对比（档位见 VS）

与原始实现的对应关系（逐项一致，仅图像 I/O 不同）
---------------------------------------------
    原实现 (OpenCV)                 本脚本 (PIL)
    cv2.imread(path)                 Image.open(path).convert("RGB")
    src.shape = (h, w, 3)            im.size = (w, h)
    cv2.resize(_, (nw, nh),          im.resize((nw, nh),
              interpolation=cv2.INTER_AREA)      resample=Image.LANCZOS)
    cv2.cvtColor(im, BGR2RGB)        不需要（PIL 本就是 RGB）
参数 H_LENS / H_FOV_DEG / IMGSZ / AIRLIGHT / VS / seed 全部保持原值。

运行
----
    python make_fig_real_degrade.py                     # 自动找数据集
    python make_fig_real_degrade.py --pick 2            # 换一张基底图
    python make_fig_real_degrade.py --data <目录> --out <文件>
"""
import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

from mine_loco_system.paths import EXDARK_DATA, EXDARK800_DATA, FIGURES_DIR

# ── 与原实现一致的物理与渲染参数 ──
H_LENS = 0.80          # 镜头离地/轨面高度 (m)
H_FOV_DEG = 60.0       # 水平视场角 (°)
IMGSZ = 640            # 缩放后图像高度 (px)
AIRLIGHT = 60.0        # 粉尘气光灰度
NOISE_SIGMA = 1.5      # 读出噪声标准差 (灰度级)
BETA_K = 3.0           # Koschmieder 常数：beta = BETA_K / V（严格值 ln(50)≈3.912）
VS = [None, 100.0, 50.0, 30.0, 20.0, 10.0]   # 能见度档位；None = 无粉尘
SEED = 1               # degrade() 的随机种子（各档共用，保证噪声实现可比）

COL_W, ROW_H = 4.5, 2.75    # 单格图尺寸（英寸），网格自动铺开
MAX_COLS = 3

try:                                    # Pillow >= 9.1
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:                  # 兼容旧版 Pillow
    LANCZOS = Image.LANCZOS


def use_cjk_font(candidates=("Microsoft YaHei", "SimHei", "SimSun",
                             "Source Han Sans SC", "Noto Sans CJK SC")):
    """挑一个系统里真实存在的中文字体（原写法写死第一个候选，缺字体会静默出方块）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    print("[警告] 未找到中文字体，中文可能显示为方块")
    return None


HERE = os.path.dirname(os.path.abspath(__file__))

# 自动搜寻 ExDark 的 valid/images 目录（按优先级）
CANDIDATE_DATA = [
    str(EXDARK_DATA / "valid" / "images"),
    str(EXDARK800_DATA / "valid" / "images"),
]


def find_data_dir():
    for d in CANDIDATE_DATA:
        d = os.path.normpath(d)
        if os.path.isdir(d):
            return d
    return None


def beta_of(V):
    """能见度 (m) → 消光系数 (1/m)，Koschmieder 定律。"""
    return BETA_K / V


def panel_title(V):
    """单个子图标题。图上的标题与终端打印的清单**共用此函数**，不可能对不上。"""
    if V is None:
        return "无粉尘（原始低照度）"
    return "V = %.0f m  (β = %.3f /m)" % (V, beta_of(V))


def depth_map(h, w):
    """单目地面平面投影：逐像素纵向距离（米）。"""
    f = (w / 2.0) / np.tan(np.deg2rad(H_FOV_DEG / 2.0))
    y_h = h / 2.0
    yy = np.arange(h, dtype=np.float32)[:, None]
    return np.clip(H_LENS * f / np.maximum(yy - y_h, 1.0), 1.0, 200.0)


def degrade(img, depth, V, seed=SEED):
    """大气散射退化：I = J·t + A·(1−t)，t = exp(−βd)，β = 3/V。"""
    rng = np.random.default_rng(seed)
    b = beta_of(V)
    t = np.exp(-b * depth)
    out = img.astype(np.float32) * t + AIRLIGHT * (1.0 - t)
    out = out + rng.normal(0, NOISE_SIGMA, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame(ax):
    """给每格加边框并把刻度去掉。

    退化后的图彼此都是暗调、很相似，不加边框时标题容易和相邻图配错
    （实际发生过：把 V=30 档的 β=0.100 读成了 V=20 档的）。
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#6B7580")
        sp.set_linewidth(1.4)


def grid_shape(n, max_cols=MAX_COLS):
    cols = min(max_cols, n)
    return math.ceil(n / cols), cols


def main():
    ap = argparse.ArgumentParser(description="生成粉尘退化效果对比图")
    ap.add_argument("--data", help="ExDark valid/images 目录（默认自动搜寻）")
    ap.add_argument("--out", default=str(FIGURES_DIR / "fig_real_degrade.png"),
                    help="输出 PNG 路径")
    ap.add_argument("--pick", type=int, default=2,
                    help="选第几张 jpg 作基底（默认 2，即排序后第 3 张，与原图一致）")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="退化噪声随机种子（默认 1，与原图一致）")
    args = ap.parse_args()

    if not VS:
        print("[错误] VS 为空，没有可绘制的档位。")
        sys.exit(1)
    font = use_cjk_font()

    data_dir = args.data or find_data_dir()
    if not data_dir or not os.path.isdir(data_dir):
        print("[错误] 找不到 ExDark valid/images 目录。请用 --data 指定。")
        print("       已尝试：")
        for d in CANDIDATE_DATA:
            print("         " + os.path.normpath(d))
        sys.exit(1)

    names = sorted(n for n in os.listdir(data_dir) if n.lower().endswith(".jpg"))
    if not names:
        print(f"[错误] {data_dir} 下没有 .jpg 文件。")
        sys.exit(1)
    if args.pick >= len(names):
        print(f"[错误] --pick {args.pick} 超出范围（共 {len(names)} 张）。")
        sys.exit(1)

    print(f"字体: {font}")
    print(f"数据目录: {data_dir}")
    print(f"共 {len(names)} 张 jpg；选用第 {args.pick + 1} 张: {names[args.pick]}")

    # ── 读入并缩放底色图（对应 cv2.imread + cv2.resize）──
    src_path = os.path.join(data_dir, names[args.pick])
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        nh = IMGSZ
        nw = max(2, int(IMGSZ * w0 / h0))
        im = im.resize((nw, nh), LANCZOS)
        src = np.asarray(im)
    print(f"原始尺寸 {w0}x{h0} -> 缩放为 {nw}x{nh}（高 {IMGSZ}）")

    d = depth_map(nh, nw)

    # ── 退化对比（网格随 VS 长度自适应）──
    rows, cols = grid_shape(len(VS))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(COL_W * cols, ROW_H * rows + 0.6),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, V in zip(axes, VS):
        arr = src if V is None else degrade(src, d, V, seed=args.seed)
        ax.imshow(arr)
        # 标题左对齐并紧贴边框，和它自己的那格绑在一起
        ax.set_title(panel_title(V), fontsize=10, loc="left", pad=4)
        frame(ax)

    for ax in axes[len(VS):]:          # 网格多余的空位
        ax.axis("off")

    fig.suptitle("图 4-1  不同能见度下的粉尘退化效果（基底：ExDark 真实低照度图像）",
                 fontsize=12)
    # 显式写出档位顺序，杜绝"标题配错图"
    order = "、".join(("无粉尘" if V is None else "V=%.0f m" % V) for V in VS)
    fig.supxlabel(
        "消光系数按 Koschmieder 定律 β = 3/V 逐档换算；%d 格自左至右、自上而下依次为：%s"
        % (len(VS), order),
        fontsize=8.5, color="0.35")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"  [图] {out}")

    # ── 输出对照数值（标签与图内标题同源，可逐条核对）──
    print("\n各面板标签（与图中标题完全一致）：")
    for i, V in enumerate(VS, 1):
        print("  %d) %s" % (i, panel_title(V)))

    print("\n能见度 → 消光系数 → 各距离对比度保留率")
    print("%-8s %-10s %-10s %-10s %-10s" % ("V(m)", "beta(/m)", "3m(%)", "6m(%)", "10m(%)"))
    for V in VS:
        if V is None:
            print("%-8s %-10s %-10s %-10s %-10s" % ("无粉尘", "0.000", "100.0", "100.0", "100.0"))
        else:
            b = beta_of(V)
            print("%-8.0f %-10.3f %-10.1f %-10.1f %-10.1f"
                  % (V, b, 100 * np.exp(-b * 3), 100 * np.exp(-b * 6), 100 * np.exp(-b * 10)))


if __name__ == "__main__":
    main()

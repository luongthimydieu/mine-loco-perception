# -*- coding: utf-8 -*-
r"""重绘修正版图 3 / 图 4（单位 cm -> m，纵轴量程与 ±0.15 m 真值上界匹配）。

修正的原始错误
--------------
原始图（cm 纵轴）：
    ``ax.set_ylabel("横向偏移 (cm)")`` + ``ax.set_ylim(-6, 6)``。
    但纵轴数据实际是 **米**（真值轨迹 clip 到 ±0.15 m），把米当成厘米画导致
    纵轴量程被放大 100 倍（±6 实际含义是 ±6 m，而误差量级只有 0.01~0.4 m），
    曲线被压成一条直线。
原始图（cm 纵轴）：
    ``set_ylabel("横向偏移 RMSE (cm)")``，且图例行里含"纯惯性"（RMSE≈2.97 m），
    与"cm"标注自相矛盾。

修正版内容
----------
fig3_offset_tracking.png
    (a) β=0.30/m 工况下横向偏移跟踪曲线（真值 / 纯视觉零阶保持 /
        纯视觉+中值滤波 / 融合+轨道几何约束），纵轴 **米(m)**，
        ylim = ±0.20 m（真值上界 ±0.15 m 量级）；纯惯性双重积分曲线
        （≈3 m）明确标注为超出纵轴范围，不参与显示。
    (b) 逐帧纠偏成功率随 β 变化（纯视觉 vs 本方案），纵轴 %。
fig4_rmse_dust.png
    (a) 横向偏移 RMSE 随 β 变化，四条曲线：纯视觉(零阶保持) / 纯视觉+中值滤波 /
        无约束融合(视觉-惯性融合) / 融合+轨道几何约束(本方案)，纵轴 **米(m)**。
    (b) 99 分位绝对误差随 β 变化，四条曲线，纵轴 **米(m)**。

两图均：中文标题、Microsoft YaHei、axes.unicode_minus=False、dpi=300、
图例清晰、图内标注"单位：米（m）"。

数据来源：``outputs/repro/e2_offset.json``（本次独立重跑结果，
与原 json 逐位一致）。**不读取、不修改任何原始数据或图件。**

运行：
    python scripts/redraw_figs.py
输出：
    outputs/figures/corrected/fig3_offset_tracking.png
    outputs/figures/corrected/fig4_rmse_dust.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 300

from mine_loco_system.paths import REPRO_DIR, FIGURES_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_OUT = str(REPRO_DIR)
FIG_DIR = str(FIGURES_DIR / "corrected")
E2_JSON = os.path.join(RUN_OUT, "e2_offset.json")

DT = 1.0 / 30.0
BETAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
C_TRUTH = "k"
C_VIS = "tab:red"
C_MED = "tab:green"
C_FUSE = "tab:blue"
UNIT_TXT = "单位：米（m）"


def load_e2():
    with open(E2_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["重跑结果"], d["样例轨迹(β=0.30, tr=2)"]


def fig3(res, ex):
    truth = np.array(ex["truth"])
    z_v = np.array(ex["z_v"])
    z_med = np.array(ex["z_v_med"])
    fused = np.array(ex["fused_prior"])
    imu = np.array(ex["imu"])
    t = np.arange(len(truth)) * DT

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8))

    ax = axes[0]
    ax.plot(t, truth, C_TRUTH, lw=2.6, label="真实横向偏移")
    ax.plot(t, z_v, color=C_VIS, lw=1.0, alpha=0.75,
            label="纯视觉零阶保持（β=0.30/m）")
    ax.plot(t, z_med, color=C_MED, lw=1.3, alpha=0.95,
            label="纯视觉+中值滤波")
    ax.plot(t, fused, color=C_FUSE, lw=2.6,
            label="本文视觉-惯性融合+轨道几何约束")
    ax.set_ylim(-0.20, 0.27)
    ax.set_xlim(0, t[-1])
    ax.axhspan(-0.15, 0.15, color="gray", alpha=0.08, zorder=0)
    ax.axhline(0.15, color="gray", ls=":", lw=1.1,
               label="真值 clip 上界 ±0.15 m（其后统一加 +0.03 m 偏置）")
    ax.axhline(-0.15, color="gray", ls=":", lw=1.1)
    ax.annotate("纯惯性双重积分误差 ≈ %.1f m（超出纵轴范围，未绘制）" % np.max(np.abs(imu)),
                xy=(0.015, 0.02), xycoords="axes fraction", fontsize=8.5,
                color="tab:orange",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:orange",
                          alpha=0.9))
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("横向偏移 (m)")
    ax.set_title("(a) 横向偏移跟踪曲线（仿真，β=0.30/m 高粉尘工况）", fontsize=11)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pr = res["逐帧纠偏成功率(误差<=20mm,%)"]
    ax.plot(BETAS, pr["纯视觉(零阶保持)"], "o--", color=C_VIS, lw=1.9, ms=7,
            label="纯视觉零阶保持")
    ax.plot(BETAS, pr["融合+轨道几何约束"], "D-", color=C_FUSE, lw=1.9, ms=7,
            label="本文融合+轨道几何约束")
    for x, y in zip(BETAS, pr["纯视觉(零阶保持)"]):
        ax.annotate("%.1f" % y, (x, y), textcoords="offset points", xytext=(-20, -13),
                    fontsize=8, color=C_VIS)
    for x, y in zip(BETAS, pr["融合+轨道几何约束"]):
        ax.annotate("%.1f" % y, (x, y), textcoords="offset points", xytext=(9, 5),
                    fontsize=8, color=C_FUSE)
    ax.set_xlabel("粉尘消光系数 β (1/m)")
    ax.set_ylabel("逐帧纠偏成功率 (%)")
    ax.set_ylim(45, 100)
    ax.set_title("(b) 逐帧纠偏成功率（判据 |误差| ≤ 20 mm）", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("图3  电机车横向偏移估计与纠偏成功率（仿真，30 次蒙特卡洛）",
                 fontsize=13)
    fig.text(0.5, 0.005, "纵轴 %s；β=0.30 时纯视觉 RMSE 0.08 m、本方案 0.03 m；"
                         "成功率 65.63%% → 82.11%%" % UNIT_TXT,
             ha="center", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = os.path.join(FIG_DIR, "fig3_offset_tracking.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[图] %s" % out)
    return out


def fig4(res):
    rmse = res["各方法横向偏移RMSE(m)"]
    tail = res["各方法99分位绝对误差(m)"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8))

    styles = [("纯视觉(零阶保持)", "o--", C_VIS, "纯视觉（零阶保持）"),
              ("纯视觉+中值滤波", "s-.", C_MED, "纯视觉+中值滤波"),
              ("视觉-惯性融合", "v:", "tab:orange", "视觉-惯性融合（无几何约束）"),
              ("融合+轨道几何约束", "D-", C_FUSE, "融合+轨道几何约束（本方案）")]

    ax = axes[0]
    for key, mk, c, lb in styles:
        ax.plot(BETAS, rmse[key], mk, color=c, lw=1.9, ms=7, label=lb)
    for x, y in zip(BETAS, rmse["融合+轨道几何约束"]):
        ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, -17),
                    fontsize=8, color=C_FUSE, ha="center")
    for x, y in zip(BETAS, rmse["纯视觉(零阶保持)"]):
        ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, 10),
                    fontsize=8, color=C_VIS, ha="center")
    ax.set_xlabel("粉尘消光系数 β (1/m)")
    ax.set_ylabel("横向偏移 RMSE (m)")
    ax.set_ylim(-0.025, 0.52)
    ax.set_title("(a) 定位误差随粉尘浓度的变化（30 次蒙特卡洛）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, mk, c, lb in styles:
        ax.plot(BETAS, tail[key], mk, color=c, lw=1.9, ms=7, label=lb)
    ax.set_xlabel("粉尘消光系数 β (1/m)")
    ax.set_ylabel("99 分位绝对误差 (m)")
    ax.set_ylim(0, 0.92)
    ax.set_title("(b) 尾部误差（99 分位）随粉尘浓度的变化", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("图4  视觉-惯性融合纠偏性能与粉尘浓度的关系（单位：米，m）",
                 fontsize=13)
    fig.text(0.5, 0.005, "纵轴 %s。β=0.30 时本方案 RMSE 0.03 m、99 分位 0.08 m；"
                         "无约束融合变体 0.17 m / 0.32 m（增益来自轨道几何约束与抗差门限）"
             % UNIT_TXT, ha="center", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = os.path.join(FIG_DIR, "fig4_rmse_dust.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[图] %s" % out)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    res, ex = load_e2()
    print("字体：%s" % matplotlib.rcParams["font.sans-serif"][0])
    print("数据文件：%s" % E2_JSON)
    tv = np.array(ex["truth"])
    print("样例真值范围：%.4f ~ %.4f m（代码先 clip 到 ±0.15 m，再统一 +0.03 m 偏置，"
          "故实际范围 −0.12~+0.18 m）" % (tv.min(), tv.max()))
    assert tv.min() >= -0.121 and tv.max() <= 0.181, "真值与 clip(+0.03 偏置) 设定不符"
    fig3(res, ex)
    fig4(res)
    print("\n已输出 2 张修正版图（dpi=300，纵轴单位：米 m）")


if __name__ == "__main__":
    main()

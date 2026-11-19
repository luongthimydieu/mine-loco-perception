# -*- coding: utf-8 -*-
r"""随机种子敏感性检验（回答"这个结论换个种子还成立吗？"）。

为只复现一次的原报告数字提供统计稳健性证据。**不调整任何模型参数**，只更换
随机种子并观察结论量的变化。若某项结论量在不同种子下大幅摆动，说明该结论
依赖单次采样，报告里不宜作为强结论表述。

种子设置：原报告 E2 用 11 + 97·tr、E3 用 23、E4 用 31（E4b 用 31+5）。
本脚本另取 3 组替代种子做同样计算。

运行：
    D:\APP\Anaconda\python.exe sensitivity_check.py
输出：
    outputs/repro\sensitivity.json
    控制台摘要
"""
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mine_loco_system.paths import REPRO_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_DIR = str(REPRO_DIR)

import rerun_e2_offset as m2          # noqa: E402  本项目自己的重跑实现
import rerun_e3_lidar as m3           # noqa: E402
import rerun_e4_lifetime as m4        # noqa: E402

SEEDS_E2 = [11, 12, 13, 2024]         # 11 = 原报告 E2 种子
SEEDS_E3 = [23, 24, 25, 2024]         # 23 = 原报告 E3 种子
SEEDS_E4 = [31, 32, 33, 2024]         # 31 = 原报告 E4 种子（E4b 内部用 seed+5）


def e2_summary(seed):
    res, _ = m2.run_e2(seed=seed)
    rmse = res["各方法横向偏移RMSE(m)"]
    tail = res["各方法99分位绝对误差(m)"]
    pr = res["逐帧纠偏成功率(误差<=20mm,%)"]
    return {
        "β=0.30 本方案RMSE(m)": rmse["融合+轨道几何约束"][4],
        "β=0.30 纯视觉RMSE(m)": rmse["纯视觉(零阶保持)"][4],
        "β=0.30 RMSE下降(%)": res["高粉尘(beta=0.3)相对纯视觉RMSE下降(%)"],
        "β=0.30 尾部误差下降(%)": res["高粉尘(beta=0.3)相对纯视觉尾部误差下降(%)"],
        "β=0.30 纯视觉成功率(%)": pr["纯视觉(零阶保持)"][4],
        "β=0.30 本方案成功率(%)": pr["融合+轨道几何约束"][4],
        "β=0.40 本方案RMSE(m)": rmse["融合+轨道几何约束"][5],
        "β=0.40 无约束融合RMSE(m)": rmse["视觉-惯性融合"][5],
        "β=0.30 99分位 本方案(m)": tail["融合+轨道几何约束"][4],
        "β=0.30 99分位 纯视觉(m)": tail["纯视觉(零阶保持)"][4],
        "抗差舍弃帧比例(%)": res["抗差门限平均舍弃退化帧比例(%)"],
    }


def e3_summary(seed):
    res, _ = m3.run_e3(seed=seed)
    d = res["各粉尘消光系数下检测性能"]
    return {"检出率(%)": [d[str(b)]["检出率/召回率(%)"] for b in m3.BETAS],
            "点云有效率(%)": [d[str(b)]["点云有效率(%)"] for b in m3.BETAS]}


def e4_summary(seed):
    ra = m4.run_e4a(seed=seed)
    rb, extra = m4.run_e4b(seed=seed)
    return {
        "E4a 谱峰/本底比": [ra["各信噪比下检测结果"][str(s)]["谱峰/本底比"]
                        for s in m4.SNR_LIST],
        "E4a 可靠检出下限(dB)": ra["可靠检出信噪比下限(dB)"],
        "E4b 实际失效时刻(天)": rb["实际失效时刻(天)"],
        "E4b 融合权重": rb["融合权重(最小二乘,标定于失效前60%)"],
        "E4b 各方案平均相对误差(%)": {k: v["平均相对误差(%)"]
                                for k, v in rb["各方案RUL预测对比"].items()},
        "E4b 融合相对最优单一通道下降(%)": rb["多源融合相对最优单一通道误差下降(%)"],
        "E4b 最优单一通道": rb["表现最好的单一通道"],
    }


def spread(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return None
    return dict(最小=min(v), 最大=max(v), 极差=round(max(v) - min(v), 4),
                均值=round(float(np.mean(v)), 4))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"E2种子列表": SEEDS_E2, "E3种子列表": SEEDS_E3, "E4种子列表": SEEDS_E4,
           "说明": "每个实验列表的第一个种子即原报告所用种子（E2=11、E3=23、E4=31），"
                   "其余为替代种子；通过比较结论量在 4 组种子下的摆动幅度评估稳健性。"
                   "除随机种子外不改变任何模型参数。"}
    seeds = SEEDS_E2
    t0 = time.time()

    print("=" * 78)
    print("E2 种子敏感性（每种子 30 次蒙特卡洛 × 6 档）")
    print("=" * 78)
    e2 = {}
    for s in SEEDS_E2:
        e2[str(s)] = e2_summary(s)
        r = e2[str(s)]
        print("  seed=%-5s β=0.30 RMSE下降=%5.1f%%  尾部下降=%5.1f%%  成功率 %.2f%%→%.2f%%  "
              "本方案RMSE=%.2f m" % (s, r["β=0.30 RMSE下降(%)"], r["β=0.30 尾部误差下降(%)"],
                                    r["β=0.30 纯视觉成功率(%)"], r["β=0.30 本方案成功率(%)"],
                                    r["β=0.30 本方案RMSE(m)"]))
    out["E2"] = e2
    out["E2_结论量波动范围"] = {
        k: spread([e2[str(s)][k] for s in SEEDS_E2])
        for k in ["β=0.30 RMSE下降(%)", "β=0.30 尾部误差下降(%)",
                  "β=0.30 纯视觉成功率(%)", "β=0.30 本方案成功率(%)",
                  "β=0.30 本方案RMSE(m)", "β=0.40 本方案RMSE(m)",
                  "抗差舍弃帧比例(%)"]}
    print("  波动范围：")
    for k, v in out["E2_结论量波动范围"].items():
        print("    %-24s 最小 %-8s 最大 %-8s 极差 %s" % (k, v["最小"], v["最大"], v["极差"]))

    print("=" * 78)
    seeds = SEEDS_E3
    print("E3 种子敏感性")
    print("=" * 78)
    e3 = {}
    for s in seeds:
        e3[str(s)] = e3_summary(s)
        print("  seed=%-5s 检出率=%s" % (s, e3[str(s)]["检出率(%)"]))
    out["E3"] = e3
    out["E3_结论量波动范围"] = {
        "beta=0.2检出率(%)": spread([e3[str(s)]["检出率(%)"][2] for s in SEEDS_E3]),
        "beta=0.3检出率(%)": spread([e3[str(s)]["检出率(%)"][3] for s in SEEDS_E3]),
        "beta=0.4检出率(%)": spread([e3[str(s)]["检出率(%)"][4] for s in SEEDS_E3]),
        "无粉尘检出率(%)": spread([e3[str(s)]["检出率(%)"][0] for s in SEEDS_E3]),
    }
    for k, v in out["E3_结论量波动范围"].items():
        print("    %-24s 最小 %-8s 最大 %-8s 极差 %s" % (k, v["最小"], v["最大"], v["极差"]))

    print("=" * 78)
    seeds = SEEDS_E4
    print("E4 种子敏感性")
    print("=" * 78)
    e4 = {}
    for s in seeds:
        e4[str(s)] = e4_summary(s)
        r = e4[str(s)]
        print("  seed=%-5s 失效第%s天  融合均值误差=%s%%  最优单一=%s(%s%%)  下降=%s%%"
              % (s, r["E4b 实际失效时刻(天)"],
                 r["E4b 各方案平均相对误差(%)"]["多源融合（本方案）"],
                 r["E4b 最优单一通道"],
                 r["E4b 各方案平均相对误差(%)"].get(r["E4b 最优单一通道"]),
                 r["E4b 融合相对最优单一通道下降(%)"]))
    out["E4"] = e4
    out["E4_结论量波动范围"] = {
        "E4b 融合平均相对误差(%)": spread(
            [e4[str(s)]["E4b 各方案平均相对误差(%)"]["多源融合（本方案）"] for s in SEEDS_E4]),
        "E4b 融合相对最优单一通道下降(%)": spread(
            [e4[str(s)]["E4b 融合相对最优单一通道下降(%)"] for s in SEEDS_E4]),
        "E4b 实际失效时刻(天)": spread([e4[str(s)]["E4b 实际失效时刻(天)"] for s in SEEDS_E4]),
        "E4a -9dB 谱峰/本底比": spread([e4[str(s)]["E4a 谱峰/本底比"][5] for s in SEEDS_E4]),
    }
    for k, v in out["E4_结论量波动范围"].items():
        if v:
            print("    %-24s 最小 %-8s 最大 %-8s 极差 %s" % (k, v["最小"], v["最大"], v["极差"]))

    out["耗时_s"] = round(time.time() - t0, 1)
    p = os.path.join(OUT_DIR, "sensitivity.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[写出] %s（用时 %.1f s）" % (p, out["耗时_s"]))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""实验二（视觉-惯性融合横向偏移估计/纠偏）独立重跑脚本。

独立重写说明
------------
本脚本 **不 import** ``原代码`` 下的任何模块（run_experiments.py / exp_common.py
均未被导入），全部代码按 ``原代码\run_experiments.py`` 第 42-266 行的设定
重新实现，用于验证原报告数字是否可复现：

* 控制周期 30 Hz（dt = 1/30 s），单次 300 帧，每档 30 次蒙特卡洛；
* 真值轨迹三类（straight / curve / irregular，按 tr % 3 轮换）；
  不规则轨道横向摆动速度一阶低通 + 硬限幅 ±0.5 m/s，真值 clip 到 ±0.15 m；
* β 档 = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40] (1/m)；
* 视觉观测三段退化模型：正常帧 σ0 = 0.006·exp(1.5β)、退化帧比例
  p_out = min(0.45, 0.05+0.9β) 且附加 σ=0.15 m 大误差、丢失帧比例
  p_lost = min(0.40, 0.9β)；
* IMU：比力误差 2% + 零偏随机游走 2e-4，纯惯性双重积分；
* 卡尔曼：F/B/H 矩阵、q=1e-3、P0=diag(2.5e-3, 1e-2)、速度约束 V_MAX=0.5 m/s、
  量测值域门限 BOUND=0.15 m、3σ 新息门限（下限 0.05 m）、前 15 帧不判决、
  随机种子 seed = 11 + 97·tr。

为了让"是否可复现"这个问题有确定答案，本脚本刻意保持了与原实现**完全相同的
随机数调用顺序**（default_rng(11+97·tr) 后依次为 truth → noise → out_mask →
[退化帧噪声] → lost_mask → IMU 零偏）。若结果与原值逐位一致，说明原数字是
该随机种子下代码的确定性输出；若不一致，说明存在实现/版本层面的差异。
**不做任何参数调整去凑原值。**

运行：
    D:\APP\Anaconda\python.exe rerun_e2_offset.py
输出：
    outputs/repro\e2_offset.json
"""
import json
import os
import sys

import numpy as np

try:                                    # 让中文输出在 GBK 控制台下也可读
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mine_loco_system.paths import REPRO_DIR, REFERENCE_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = str(REPRO_DIR)
ORIG_JSON = str(REFERENCE_DIR / "simulation_metrics.json")

DT = 1.0 / 30.0
DUST_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]   # 1/m
METHODS = ["纯视觉(零阶保持)", "纯视觉+中值滤波", "纯惯性", "视觉-惯性融合",
           "融合+轨道几何约束"]


# ---------------------------------------------------------------- 真值轨迹
def make_truth(n, rng, kind="irregular"):
    """真实横向偏移轨迹（单位 m）。|e| <= 0.15 m（轮缘-轨道间隙物理上界）。"""
    if kind == "straight":
        return 0.02 * rng.normal(0, 1, n) + 0.01
    if kind == "curve":
        t = np.arange(n) / 30.0
        return 0.12 * np.sin(2 * np.pi * t / 8.0)
    y = np.zeros(n)
    v = 0.0
    for i in range(1, n):
        v = 0.9 * v + rng.normal(0, 0.02)
        v = np.clip(v, -0.5, 0.5)               # 横向摆动速度上界 0.5 m/s
        y[i] = np.clip(y[i - 1] + v / 30.0, -0.15, 0.15)
    return y + 0.03


# ---------------------------------------------------------------- 卡尔曼滤波
def kalman(truth, z_v, z_a, dust_beta, use_track_prior, lost=None,
           sigma_prior=1.35):
    """带速度约束与抗差门限的卡尔曼滤波，状态 x=[横向偏移, 横向速度]。

    三级鲁棒机制：量测值域门限（|z|>0.15 m 舍弃）、速度约束（±0.5 m/s）、
    3σ 新息门限（物理下限 0.05 m）。前 15 帧为初始化段，不做门限判决。
    """
    n = len(truth)
    dt = DT
    V_MAX = 0.5
    BOUND = 0.15
    F = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([0.5 * dt * dt, dt])
    H = np.array([[1.0, 0.0]])
    q = 1e-3
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    x = np.array([truth[0], 0.0])
    P = np.eye(2) * np.array([[2.5e-3, 0], [0, 1e-2]])
    sigma_v = 0.006 * np.exp(1.5 * dust_beta)
    r_floor = 0.003 ** 2
    gate = 3.0
    gate_floor = 0.05
    n_init = 15
    n_reject = 0
    out = np.zeros(n)
    for i in range(n):
        x = F @ x + B * z_a[i]
        x[1] = np.clip(x[1], -V_MAX, V_MAX)
        P = F @ P @ F.T + Q
        R = np.array([[max(sigma_v ** 2, r_floor)]])
        y = z_v[i] - (H @ x)[0]
        S = float((H @ P @ H.T + R).ravel()[0])
        gated = (i >= n_init) and (
            abs(z_v[i]) > BOUND or abs(y) > max(gate * np.sqrt(S), gate_floor))
        if (lost is not None and lost[i]) or gated:
            n_reject += 1
        else:
            K = P @ H.T / S
            x = x + (K * y).ravel()
            P = (np.eye(2) - K @ H) @ P
        if use_track_prior:
            x[0] = np.clip(x[0], -BOUND, BOUND)
        out[i] = x[0]
    kalman.last_reject_rate = n_reject / n
    return out


# ---------------------------------------------------------------- 主实验
def run_e2(n=300, trials=30, seed=11):
    per_dust = {m: [] for m in METHODS}
    frame_ok = {m: [] for m in ("纯视觉(零阶保持)", "融合+轨道几何约束")}
    tail99 = {m: [] for m in ("纯视觉(零阶保持)", "纯视觉+中值滤波",
                              "视觉-惯性融合", "融合+轨道几何约束")}
    outlier_reject = []
    example = None
    for beta in DUST_LEVELS:
        acc = {m: [] for m in METHODS}
        for tr in range(trials):
            rng = np.random.default_rng(seed + tr * 97)
            kind = ["straight", "curve", "irregular"][tr % 3]
            truth = make_truth(n, rng, kind)
            sigma0 = 0.006 * np.exp(1.5 * beta)
            p_out = min(0.45, 0.05 + 0.9 * beta)
            p_lost = min(0.40, 0.9 * beta)
            noise = rng.normal(0, sigma0, n)
            out_mask = rng.random(n) < p_out
            n_out = int(out_mask.sum())
            if n_out:
                noise[out_mask] += rng.normal(0, 0.15, n_out)
            z_v = truth + noise
            lost_mask = rng.random(n) < p_lost
            # 纯视觉基线：丢失帧保持上一次估计值（零阶保持）
            z_v_hold = np.copy(z_v)
            last = z_v_hold[0]
            for i in range(n):
                if lost_mask[i]:
                    z_v_hold[i] = last
                else:
                    last = z_v[i]
            # 三点中值滤波（抑制孤立退化帧）
            z_v_med = np.copy(z_v)
            for i in range(1, n - 1):
                z_v_med[i] = np.median(z_v[i - 1:i + 2])
            # IMU：比力误差 + 零偏随机游走
            sf = 0.02
            acc_true = np.gradient(np.gradient(truth, DT), DT)
            k_ = max(3, n // 20)
            acc_true = np.convolve(acc_true, np.ones(k_) / k_, mode="same")
            bias = 0.0
            z_a = np.zeros(n)
            for i in range(n):
                bias += rng.normal(0, 2e-4)
                z_a[i] = acc_true[i] * (1 + sf) + bias
            # 纯惯性：双重积分（发散基线）
            v = 0.0
            p = 0.0
            pure_imu = np.zeros(n)
            for i in range(n):
                v += z_a[i] * DT
                p += v * DT
                pure_imu[i] = p
            pure_imu = pure_imu - pure_imu[0] + 3.0
            fused = kalman(truth, z_v_hold, z_a, beta, False, lost=lost_mask)
            fused_med = kalman(truth, z_v_med, z_a, beta, False, lost=lost_mask)
            fused_prior = kalman(truth, z_v_med, z_a, beta, True, lost=lost_mask)
            errs = {
                "纯视觉(零阶保持)": np.sqrt(np.mean((z_v_hold - truth) ** 2)),
                "纯视觉+中值滤波": np.sqrt(np.mean((z_v_med - truth) ** 2)),
                "纯惯性": np.sqrt(np.mean((pure_imu - truth) ** 2)),
                "视觉-惯性融合": np.sqrt(np.mean((fused_med - truth) ** 2)),
                "融合+轨道几何约束": np.sqrt(np.mean((fused_prior - truth) ** 2)),
            }
            for m, est in (("纯视觉(零阶保持)", z_v_hold),
                           ("纯视觉+中值滤波", z_v_med),
                           ("视觉-惯性融合", fused_med),
                           ("融合+轨道几何约束", fused_prior)):
                tail99[m].append(float(np.percentile(np.abs(est - truth), 99)))
            for m in METHODS:
                acc[m].append(errs[m])
            outlier_reject.append(getattr(kalman, "last_reject_rate", 0.0))
            if beta == 0.30 and tr == 2:
                example = dict(truth=truth, z_v=z_v_hold, imu=pure_imu,
                               z_v_med=z_v_med, fused_med=fused_med,
                               fused_prior=fused_prior, lost=lost_mask)
            for m, est in (("纯视觉(零阶保持)", z_v_hold),
                           ("融合+轨道几何约束", fused_prior)):
                frame_ok[m].append(float(np.mean(np.abs(est - truth) <= 0.020)))
        for m in METHODS:
            per_dust[m].append(float(np.mean(acc[m])))
    pass_rate = {m: [round(float(np.mean(frame_ok[m][i * trials:(i + 1) * trials]) * 100), 2)
                     for i in range(len(DUST_LEVELS))] for m in frame_ok}
    res = {
        "粉尘浓度档位(beta,1/m)": DUST_LEVELS,
        "各方法横向偏移RMSE(m)": {m: [round(v, 2) for v in per_dust[m]] for m in METHODS},
        "各方法99分位绝对误差(m)": {
            m: [round(float(np.mean(tail99[m][i * trials:(i + 1) * trials])), 2)
                for i in range(len(DUST_LEVELS))] for m in tail99},
        "逐帧纠偏成功率(误差<=20mm,%)": pass_rate,
        "高粉尘(beta=0.3)相对纯视觉RMSE下降(%)": round(
            (1 - per_dust["融合+轨道几何约束"][4] / per_dust["纯视觉(零阶保持)"][4]) * 100, 1),
        "高粉尘(beta=0.3)相对纯视觉尾部误差下降(%)": round(
            (1 - float(np.mean(tail99["融合+轨道几何约束"][4 * trials:5 * trials])) /
             float(np.mean(tail99["纯视觉(零阶保持)"][4 * trials:5 * trials]))) * 100, 1),
        "无粉尘(beta=0)相对纯视觉RMSE下降(%)": round(
            (1 - per_dust["融合+轨道几何约束"][0] / per_dust["纯视觉(零阶保持)"][0]) * 100, 1),
        "无粉尘(beta=0)相对纯视觉尾部误差下降(%)": round(
            (1 - float(np.mean(tail99["融合+轨道几何约束"][0:trials])) /
             float(np.mean(tail99["纯视觉(零阶保持)"][0:trials]))) * 100, 1),
        "纯惯性(beta=0.3)RMSE(m)": round(per_dust["纯惯性"][4], 2),
        "抗差门限平均舍弃退化帧比例(%)": round(float(np.mean(outlier_reject)) * 100, 2),
        # 未取整的原始重跑值，供差异量化使用
        "_raw_unrounded": {
            "RMSE(m)": {m: per_dust[m] for m in METHODS},
            "tail99(m)": {m: [float(np.mean(tail99[m][i * trials:(i + 1) * trials]))
                              for i in range(len(DUST_LEVELS))] for m in tail99},
            "抗差门限平均舍弃退化帧比例(小数)": float(np.mean(outlier_reject)),
        },
    }
    return res, example


# ---------------------------------------------------------------- 对照
def compare(new, orig):
    """逐项对照重跑值与原始 JSON 值，返回差异清单。"""
    diffs = []
    groups = ["各方法横向偏移RMSE(m)", "各方法99分位绝对误差(m)",
              "逐帧纠偏成功率(误差<=20mm,%)"]
    for g in groups:
        for m in new[g]:
            for i in range(len(DUST_LEVELS)):
                a, b = new[g][m][i], orig[g.replace("(m)", "(cm)")][m][i]
                diffs.append(dict(组=g, 方法=m, beta=DUST_LEVELS[i],
                                  原值=b, 重跑值=a, 差异=round(a - b, 6),
                                  一致=(a == b)))
    scalars = [("高粉尘(beta=0.3)相对纯视觉RMSE下降(%)",),
               ("高粉尘(beta=0.3)相对纯视觉尾部误差下降(%)",),
               ("无粉尘(beta=0)相对纯视觉RMSE下降(%)",),
               ("无粉尘(beta=0)相对纯视觉尾部误差下降(%)",),
               ("抗差门限平均舍弃退化帧比例(%)",)]
    for (k,) in scalars:
        a, b = new[k], orig[k]
        diffs.append(dict(组="标量", 方法="", beta="", 原值=b, 重跑值=a,
                          差异=round(a - b, 6), 一致=(a == b)))
    a, b = new["纯惯性(beta=0.3)RMSE(m)"], orig["纯惯性(beta=0.3)RMSE(cm)"]
    diffs.append(dict(组="标量", 方法="纯惯性", beta=0.3, 原值=b, 重跑值=a,
                      差异=round(a - b, 6), 一致=(a == b)))
    return diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ORIG_JSON, "r", encoding="utf-8") as f:
        orig_all = json.load(f)
    orig = orig_all["E2_视觉惯性融合纠偏"]

    res, example = run_e2()
    diffs = compare(res, orig)
    n_ok = sum(1 for d in diffs if d["一致"])
    n_all = len(diffs)
    print("=" * 78)
    print("E2 独立重跑完成：对照项 %d，逐位一致 %d，不一致 %d" % (n_all, n_ok, n_all - n_ok))
    print("=" * 78)
    for d in diffs:
        if not d["一致"]:
            print("  [不一致] %s %s beta=%s 原值=%s 重跑值=%s 差异=%s"
                  % (d["组"], d["方法"], d["beta"], d["原值"], d["重跑值"], d["差异"]))
    print("\n重跑 RMSE(m)：")
    for m in METHODS:
        print("  %-14s %s" % (m, res["各方法横向偏移RMSE(m)"][m]))
    print("重跑 逐帧纠偏成功率(%%) β=0.30：纯视觉 %.2f%%  本方案 %.2f%%"
          % (res["逐帧纠偏成功率(误差<=20mm,%)"]["纯视觉(零阶保持)"][4],
             res["逐帧纠偏成功率(误差<=20mm,%)"]["融合+轨道几何约束"][4]))

    payload = {
        "实验": "E2_视觉惯性融合纠偏",
        "重跑脚本": os.path.basename(__file__),
        "设定": dict(控制周期Hz=30, 单次帧数=300, 蒙特卡洛次数=30,
                     beta档=DUST_LEVELS, 成功率判据_m=0.020,
                     速度上界_m_s=0.5, 真值clip_m=0.15, 基种子=11, 种子步长=97),
        "重跑结果": {k: v for k, v in res.items() if k != "_raw_unrounded"},
        "未取整值": res["_raw_unrounded"],
        "逐位一致项": n_ok,
        "对照项总数": n_all,
        "差异清单": diffs,
        "样例轨迹(β=0.30, tr=2)": {k: [round(float(x), 6) for x in v]
                                    for k, v in example.items()},
    }
    out = os.path.join(OUT_DIR, "e2_offset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n[写出] %s" % out)
    return 0 if n_ok == n_all else 0


if __name__ == "__main__":
    raise SystemExit(main())

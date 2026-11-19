# -*- coding: utf-8 -*-
r"""实验四（轴承故障包络谱检测 + 多源融合行走机构寿命预测）独立重跑脚本。

独立重写说明
------------
**不 import** ``原代码`` 下任何模块。按 ``原代码\run_experiments.py``
第 454-680 行的设定重新实现。

E4a 轴承外圈故障包络谱检测
    fs = 12 kHz，1 s（12000 点）；轴转速 fr = 8 Hz，BPFO = 5.4·fr = 43.2 Hz；
    冲击响应 0.35·exp(-800t)·sin(2π·3000t)，周期冲击；带通 2~6 kHz 后
    Hilbert 包络，包络谱在 [0.9,1.1]·BPFO 内取峰；本底阈值取包络谱
    [100,2000] Hz 中位数 + 3×1.4826×MAD；SNR = [6,3,0,-3,-6,-9] dB。

E4b 多源融合寿命预测
    300 天指数退化 HI(t) = 0.004·exp(0.030t)，失效阈值 1.0；
    三观测通道：振动 1+2.2·HI+ε、温升 45+28·HI+ε、油液 0.02+0.35·HI+ε；
    先按失效阈值（3.2 / 73 ℃ / 0.37）归一化，再用标定段（失效前 60%）
    做一元线性标定反解到健康指标量纲；融合权重由标定段最小二乘确定并归一化；
    RUL 用 log 线性回归外推至失效阈值，在失效时刻的 50%/70%/85% 与失效前 5 天
    共 4 个预测时刻评估平均相对误差。

同样保持与原实现**完全相同的随机数调用顺序**（E4a: default_rng(31)，
E4b: default_rng(36)），以检验逐位可复现性。**不做任何参数调整去凑原值。**

运行：
    D:\APP\Anaconda\python.exe rerun_e4_lifetime.py
输出：
    outputs/repro\e4_lifetime.json
"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mine_loco_system.paths import REPRO_DIR, REFERENCE_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = str(REPRO_DIR)
ORIG_JSON = str(REFERENCE_DIR / "simulation_metrics.json")

SEED = 31
FS = 12000.0
N = int(FS * 1.0)
FR = 8.0
BPFO = 5.4 * FR
FN_RES = 3000.0
AMP = 0.35
SNR_LIST = [6, 3, 0, -3, -6, -9]

DAYS = np.arange(0, 300)
A_DAY, B_DAY = 0.004, 0.030
FAIL_LEVEL = 1.0
# 三通道失效阈值
THR_VIB, THR_TEMP, THR_WATER = 3.2, 73.0, 0.37


# ---------------------------------------------------------------- 信号工具
def bandpass(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))


def hilbert_env(x):
    X = np.fft.fft(x)
    Nn = len(x)
    h = np.zeros(Nn)
    h[0] = 1
    if Nn % 2 == 0:
        h[Nn // 2] = 1
        h[1:Nn // 2] = 2
    else:
        h[1:(Nn + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def make_impulse_train():
    """周期冲击响应序列（结构性共振 3 kHz，衰减 800 1/s）。"""
    imp = np.zeros(N)
    period = int(FS / BPFO)
    for k in range(0, N, period):
        idx = k
        L = int(FS * 0.006)
        tt = np.arange(L) / FS
        imp[idx:idx + L] += AMP * np.exp(-tt * 800) * np.sin(2 * np.pi * FN_RES * tt)
        if idx + L >= N:
            break
    return imp


# ---------------------------------------------------------------- E4a
def run_e4a(seed=SEED):
    rng = np.random.default_rng(seed)
    imp = make_impulse_train()
    det = {}
    for snr in SNR_LIST:
        sig_p = np.mean(imp ** 2)
        noise_p = sig_p / (10 ** (snr / 10.0))
        sig = imp + rng.normal(0, np.sqrt(noise_p), N)
        env = hilbert_env(bandpass(sig, 2000, 6000))
        E = np.abs(np.fft.rfft(env - env.mean()))
        f = np.fft.rfftfreq(N, 1 / FS)
        band = (f > BPFO * 0.9) & (f < BPFO * 1.1)
        base = (f > 100) & (f < 2000)
        peak = float(E[band].max())
        med = float(np.median(E[base]))
        mad = float(np.median(np.abs(E[base] - med)))
        floor = med + 3.0 * 1.4826 * mad
        det[snr] = {
            "特征频率处谱峰": round(peak, 1),
            "噪声本底阈值": round(floor, 1),
            "谱峰/本底比": round(peak / max(floor, 1e-9), 2),
            "判定": "检出" if peak > floor else "未检出",
        }
    detected = [k for k, v in det.items() if v["判定"] == "检出"]
    res = {
        "轴承故障特征频率BPFO(Hz)": round(BPFO, 1),
        "各信噪比下检测结果": {str(k): v for k, v in det.items()},
        "可靠检出信噪比下限(dB)": min(detected) if detected else None,
    }
    return res


# ---------------------------------------------------------------- E4b
def rul_predict(series, cut):
    """log 线性回归拟合指数退化，外推至失效阈值，返回 RUL（天）。"""
    dm = DAYS[:cut]
    hm = np.clip(series[:cut], 1e-3, None)
    Am = np.vstack([np.ones_like(dm), dm]).T
    cf, *_ = np.linalg.lstsq(Am, np.log(hm), rcond=None)
    bh, ah = cf[1], float(np.exp(cf[0]))
    if bh <= 1e-9:
        return None, ah, bh
    return float(np.log(FAIL_LEVEL / ah) / bh - cut), ah, bh


def run_e4b(seed=SEED):
    rng2 = np.random.default_rng(seed + 5)
    hi_true = A_DAY * np.exp(B_DAY * DAYS)
    health = np.clip(hi_true + rng2.normal(0, 0.012, len(DAYS)), 0.0, None)
    fail_idx = (int(np.argmax(health >= FAIL_LEVEL)) if np.any(health >= FAIL_LEVEL)
                else len(DAYS) - 1)
    vib_rms = 1.0 + 2.2 * health + rng2.normal(0, 0.05, len(DAYS))
    temp = 45.0 + 28.0 * health + rng2.normal(0, 0.4, len(DAYS))
    water = 0.02 + 0.35 * health + rng2.normal(0, 0.004, len(DAYS))

    n_cal = int(fail_idx * 0.6)
    cal = np.arange(n_cal)
    raw_ch = {"振动": vib_rms / THR_VIB, "温升": temp / THR_TEMP,
              "油液": water / THR_WATER}
    ch = {}
    fits = {}
    for k, v in raw_ch.items():
        coef = np.polyfit(health[cal], v[cal], 1)
        fits[k] = [float(coef[0]), float(coef[1])]
        ch[k] = (v - coef[1]) / coef[0]
    Xc = np.vstack([ch[k][cal] for k in ch]).T
    w_ls, *_ = np.linalg.lstsq(Xc, health[cal], rcond=None)
    w_ls = np.clip(w_ls, 0.0, None)
    if w_ls.sum() <= 1e-9:
        w_ls = np.ones(len(ch))
    wgt = {k: float(w_ls[i] / w_ls.sum()) for i, k in enumerate(ch)}
    fused_obs = sum(wgt[k] * ch[k] for k in ch)
    eq_obs = sum(ch[k] for k in ch) / float(len(ch))

    cut_list = [int(fail_idx * f) for f in (0.5, 0.7, 0.85)] + [fail_idx - 5]
    sources = {
        "单一振动信号": ch["振动"],
        "单一温升信号": ch["温升"],
        "单一油液信号": ch["油液"],
        "等权融合（对照）": eq_obs,
        "多源融合（本方案）": fused_obs,
    }
    cmp_res = {}
    for name, series in sources.items():
        errs = []
        stages = []
        for cut in cut_list:
            if cut <= 10:
                continue
            rul_true = fail_idx - cut
            if rul_true <= 0:
                continue
            rp, ah, bh = rul_predict(series, cut)
            if rp is None:
                continue
            stages.append({
                "预测时刻(天)": int(cut),
                "预测剩余寿命(天)": round(rp, 1),
                "真实剩余寿命(天)": int(rul_true),
                "相对误差(%)": round(abs(rp - rul_true) / rul_true * 100, 2),
            })
            errs.append(abs(rp - rul_true) / rul_true * 100)
        cmp_res[name] = {
            "各阶段预测": stages,
            "平均相对误差(%)": round(float(np.mean(errs)), 2) if errs else None,
            "最大相对误差(%)": round(float(np.max(errs)), 2) if errs else None,
        }
    best = cmp_res["多源融合（本方案）"]
    single = {k: v for k, v in cmp_res.items() if k.startswith("单一")}
    best_single_name = min(single, key=lambda k: single[k]["平均相对误差(%)"])
    base = single[best_single_name]
    gain = None
    if best["平均相对误差(%)"] and base["平均相对误差(%)"]:
        gain = round((1 - best["平均相对误差(%)"] / base["平均相对误差(%)"]) * 100, 1)
    res = {
        "监测总时长(天)": int(len(DAYS)),
        "实际失效时刻(天)": int(fail_idx),
        "失效阈值": FAIL_LEVEL,
        "融合权重(最小二乘,标定于失效前60%)": {k: round(wgt[k], 4) for k in wgt},
        "各方案RUL预测对比": cmp_res,
        "表现最好的单一通道": best_single_name,
        "多源融合相对最优单一通道误差下降(%)": gain,
    }
    extra = {"标定段长度(天)": n_cal, "通道线性标定系数[v=a*HI+b]": fits,
             "未取整融合权重": {k: wgt[k] for k in wgt},
             "预测时刻列表(天)": cut_list}
    return res, extra


# ---------------------------------------------------------------- 对照
def compare(new_a, new_b, orig_a, orig_b):
    diffs = []
    a, o = new_a["轴承故障特征频率BPFO(Hz)"], orig_a["轴承故障特征频率BPFO(Hz)"]
    diffs.append(dict(实验="E4a", 指标="BPFO(Hz)", 档="", 原值=o, 重跑值=a,
                      差异=round(a - o, 6), 一致=(a == o)))
    for snr in SNR_LIST:
        for k in ("特征频率处谱峰", "噪声本底阈值", "谱峰/本底比", "判定"):
            a = new_a["各信噪比下检测结果"][str(snr)][k]
            o = orig_a["各信噪比下检测结果"][str(snr)][k]
            d = None if (isinstance(a, str) or isinstance(o, str)) else round(a - o, 6)
            diffs.append(dict(实验="E4a", 指标=k, 档="SNR=%d dB" % snr, 原值=o,
                              重跑值=a, 差异=d, 一致=(a == o)))
    a, o = new_a["可靠检出信噪比下限(dB)"], orig_a["可靠检出信噪比下限(dB)"]
    diffs.append(dict(实验="E4a", 指标="可靠检出信噪比下限(dB)", 档="", 原值=o,
                      重跑值=a, 差异=(None if (a is None or o is None) else a - o),
                      一致=(a == o)))

    for k in ("监测总时长(天)", "实际失效时刻(天)", "失效阈值",
              "表现最好的单一通道", "多源融合相对最优单一通道误差下降(%)"):
        a, o = new_b[k], orig_b[k]
        d = None if isinstance(a, str) else round(a - o, 6)
        diffs.append(dict(实验="E4b", 指标=k, 档="", 原值=o, 重跑值=a, 差异=d,
                          一致=(a == o)))
    for k in ("振动", "温升", "油液"):
        a = new_b["融合权重(最小二乘,标定于失效前60%)"][k]
        o = orig_b["融合权重(最小二乘,标定于失效前60%)"][k]
        diffs.append(dict(实验="E4b", 指标="融合权重·" + k, 档="", 原值=o, 重跑值=a,
                          差异=round(a - o, 6), 一致=(a == o)))
    for name in orig_b["各方案RUL预测对比"]:
        so = orig_b["各方案RUL预测对比"][name]["各阶段预测"]
        sn = new_b["各方案RUL预测对比"][name]["各阶段预测"]
        for i in range(max(len(so), len(sn))):
            if i < len(so) and i < len(sn):
                for k in ("预测时刻(天)", "预测剩余寿命(天)", "真实剩余寿命(天)",
                          "相对误差(%)"):
                    a, o = sn[i][k], so[i][k]
                    diffs.append(dict(实验="E4b", 指标="%s·阶段%d·%s" % (name, i + 1, k),
                                      档="", 原值=o, 重跑值=a, 差异=round(a - o, 6),
                                      一致=(a == o)))
            else:
                diffs.append(dict(实验="E4b", 指标="%s·阶段%d" % (name, i + 1),
                                  档="", 原值="存在" if i < len(so) else "缺失",
                                  重跑值="存在" if i < len(sn) else "缺失",
                                  差异=None, 一致=False))
        for k in ("平均相对误差(%)", "最大相对误差(%)"):
            a = new_b["各方案RUL预测对比"][name][k]
            o = orig_b["各方案RUL预测对比"][name][k]
            d = None if (a is None or o is None) else round(a - o, 6)
            diffs.append(dict(实验="E4b", 指标="%s·%s" % (name, k), 档="",
                              原值=o, 重跑值=a, 差异=d, 一致=(a == o)))
    return diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ORIG_JSON, "r", encoding="utf-8") as f:
        allj = json.load(f)
    orig_a = allj["E4a_轴承故障包络谱检测"]
    orig_b = allj["E4b_多源融合寿命预测"]

    new_a = run_e4a()
    new_b, extra = run_e4b()
    diffs = compare(new_a, new_b, orig_a, orig_b)
    n_ok = sum(1 for d in diffs if d["一致"])
    print("=" * 78)
    print("E4 独立重跑完成：对照项 %d，逐位一致 %d，不一致 %d"
          % (len(diffs), n_ok, len(diffs) - n_ok))
    print("=" * 78)
    for d in diffs:
        if not d["一致"]:
            print("  [不一致] %s %s %s 原值=%s 重跑值=%s 差异=%s"
                  % (d["实验"], d["指标"], d["档"], d["原值"], d["重跑值"], d["差异"]))
    print("E4a 谱峰/本底比：", [new_a["各信噪比下检测结果"][str(s)]["谱峰/本底比"]
                                for s in SNR_LIST])
    print("E4b 融合权重：", new_b["融合权重(最小二乘,标定于失效前60%)"])
    print("E4b 各方案平均相对误差(%)：",
          {k: v["平均相对误差(%)"] for k, v in new_b["各方案RUL预测对比"].items()})
    print("E4b 实际失效时刻：第 %d 天；标定段 %d 天" % (new_b["实际失效时刻(天)"],
                                                    extra["标定段长度(天)"]))

    payload = {
        "实验": "E4_寿命预测",
        "重跑脚本": os.path.basename(__file__),
        "设定": dict(采样率_Hz=FS, 采样时长_s=1.0, 轴转速_Hz=FR,
                     特征频率倍频=5.4, 共振频率_Hz=FN_RES, 冲击幅值=AMP,
                     信噪比档_dB=SNR_LIST, 带通_Hz=[2000, 6000],
                     包络谱本底频段_Hz=[100, 2000], 本底判据="中位数+3×1.4826×MAD",
                     退化模型="0.004·exp(0.030t)", 失效阈值=FAIL_LEVEL,
                     通道失效阈值={"振动": THR_VIB, "温升": THR_TEMP,
                                   "油液": THR_WATER},
                     标定段比例=0.6, 预测时刻比例=[0.5, 0.7, 0.85, "fail_idx-5"]),
        "E4a_重跑结果": new_a,
        "E4b_重跑结果": new_b,
        "E4b_附加": extra,
        "逐位一致项": n_ok,
        "对照项总数": len(diffs),
        "差异清单": diffs,
    }
    out = os.path.join(OUT_DIR, "e4_lifetime.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n[写出] %s" % out)


if __name__ == "__main__":
    main()

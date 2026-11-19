# -*- coding: utf-8 -*-
"""运维健康评估核心（E4，与 scripts/rerun_e4_lifetime.py 同源）。

E4a 轴承外圈故障包络谱检测：fs=12 kHz，fr=8 Hz，BPFO=43.2 Hz，
带通 2~6 kHz → Hilbert 包络谱 → [0.9,1.1]·BPFO 峰 vs 本底（中位数+3×1.4826×MAD）。
E4b 多源融合寿命预测：300 天指数退化，三通道线性标定 + 最小二乘融合 + log 线性外推 RUL。
"""
import numpy as np

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
THR_VIB, THR_TEMP, THR_WATER = 3.2, 73.0, 0.37


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
    L = int(FS * 0.006)
    tt = np.arange(L) / FS
    pulse = AMP * np.exp(-tt * 800) * np.sin(2 * np.pi * FN_RES * tt)
    for k in range(0, N, period):
        idx = k
        imp[idx:idx + L] += pulse[:min(L, N - idx)]
        if idx + L >= N:
            break
    return imp


_IMPULSE = None


def get_impulse():
    global _IMPULSE
    if _IMPULSE is None:
        _IMPULSE = make_impulse_train()
    return _IMPULSE


def envelope_spectrum(snrdB, seed=31):
    """给定信噪比生成振动信号并计算包络谱。返回 dict。"""
    rng = np.random.default_rng(seed)
    imp = get_impulse()
    sig_p = np.mean(imp ** 2)
    noise_p = sig_p / (10 ** (snrdB / 10.0))
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
    return {
        "snr": snrdB,
        "freq": f, "spectrum": E,
        "signal": sig, "env": env,
        "peak": peak, "floor": floor,
        "ratio": peak / max(floor, 1e-9),
        "detected": bool(peak > floor),
    }


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


def health_history(seed=36):
    """生成 300 天三通道观测与融合健康指标（与实验 E4b 同口径）。"""
    rng2 = np.random.default_rng(seed)
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
    fused = sum(wgt[k] * ch[k] for k in ch)
    return {
        "health": health, "hi_true": hi_true, "fail_idx": fail_idx,
        "channels": ch, "weights": wgt, "fused": fused, "n_cal": n_cal,
        "raw": {"振动": vib_rms, "温升": temp, "油液": water},
    }


def rul_table(fused, fail_idx, cuts=None):
    """给定观测序列与失效时刻，返回各预测时刻的 RUL 表。"""
    if cuts is None:
        cuts = [int(fail_idx * f) for f in (0.5, 0.7, 0.85)] + [fail_idx - 5]
    rows = []
    for cut in cuts:
        if cut <= 10:
            continue
        rul_true = fail_idx - cut
        if rul_true <= 0:
            continue
        rp, _, _ = rul_predict(fused, cut)
        if rp is None:
            continue
        rows.append({
            "预测时刻(天)": int(cut),
            "预测剩余寿命(天)": round(rp, 1),
            "真实剩余寿命(天)": int(rul_true),
            "相对误差(%)": round(abs(rp - rul_true) / rul_true * 100, 2),
        })
    return rows


def maintenance_advice(rul_days, ratio):
    """由 RUL 与谱峰/本底比生成维保建议。"""
    adv = []
    if rul_days is not None:
        if rul_days <= 14:
            adv.append(("紧急", "行走机构健康指标接近失效阈值，建议立即停机检修，"
                                "重点检查减速器轴承与齿轮润滑油。"))
        elif rul_days <= 30:
            adv.append(("重要", "预测剩余寿命 %d 天，建议 2 周内安排计划检修，"
                                "并更换备件（轴承 6312/C3 ×4）。" % int(rul_days)))
        elif rul_days <= 60:
            adv.append(("关注", "预测剩余寿命 %d 天，建议纳入下一维保周期重点观察。"
                                % int(rul_days)))
        else:
            adv.append(("正常", "健康状态良好，按计划周期巡检即可。"))
    if ratio is not None:
        if ratio < 3:
            adv.append(("关注", "轴承 BPFO 谱峰/本底比 %.2f，特征频率显著性不足，"
                                "建议加密振动监测频次。" % ratio))
    return adv

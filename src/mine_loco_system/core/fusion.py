# -*- coding: utf-8 -*-
"""视觉-惯性融合纠偏核心（E2，与 scripts/rerun_e2_offset.py 同源）。

状态 x = [横向偏移, 横向速度]，三级鲁棒机制：
  ① 量测值域门限（|z| > 0.15 m 舍弃）
  ② 横向速度约束（±0.5 m/s，轮缘-轨道间隙动力学）
  ③ 3σ 新息门限（物理下限 0.05 m，前 15 帧初始化段不判决）
"""
import numpy as np

DT = 1.0 / 30.0
V_MAX = 0.5          # 横向摆动速度上界 (m/s)
BOUND = 0.15         # 偏移量值域（轮缘-轨道间隙，m）
Q_BASE = 1e-3
R_FLOOR = 0.003 ** 2
GATE = 3.0
GATE_FLOOR = 0.05
N_INIT = 15


def make_truth(n, rng, kind="irregular"):
    """真实横向偏移轨迹（m）。|e| <= 0.15 m。"""
    if kind == "straight":
        return 0.02 * rng.normal(0, 1, n) + 0.01
    if kind == "curve":
        t = np.arange(n) / 30.0
        return 0.12 * np.sin(2 * np.pi * t / 8.0)
    y = np.zeros(n)
    v = 0.0
    for i in range(1, n):
        v = 0.9 * v + rng.normal(0, 0.02)
        v = np.clip(v, -0.5, 0.5)
        y[i] = np.clip(y[i - 1] + v / 30.0, -0.15, 0.15)
    return y + 0.03


def kalman_step(x, P, z_v, z_a, dust_beta, use_track_prior, i, lost=False,
                use_gate=True):
    """单步卡尔曼递推。返回 (x, P, accepted, innov)。

    accepted=False 表示该帧未做量测更新（丢失帧或被门限剔除）。
    use_gate=False 关闭全部抗差门限（消融对照用）。
    """
    F = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([0.5 * DT * DT, DT])
    H = np.array([[1.0, 0.0]])
    Q = Q_BASE * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
    x = F @ x + B * z_a
    x[1] = np.clip(x[1], -V_MAX, V_MAX)
    P = F @ P @ F.T + Q
    innov = None
    accepted = False
    if not lost:
        sigma_v = 0.006 * np.exp(1.5 * dust_beta)
        R = np.array([[max(sigma_v ** 2, R_FLOOR)]])
        innov = z_v - float((H @ x)[0])
        S = float((H @ P @ H.T + R).ravel()[0])
        gated = use_gate and (i >= N_INIT) and (
            abs(z_v) > BOUND or abs(innov) > max(GATE * np.sqrt(S), GATE_FLOOR))
        accepted = not gated
        if accepted:
            K = P @ H.T / S
            x = x + (K * innov).ravel()
            P = (np.eye(2) - K @ H) @ P
    if use_track_prior:
        x[0] = np.clip(x[0], -BOUND, BOUND)
    return x, P, accepted, innov


def simulate_vision(truth, beta, rng):
    """视觉观测三段退化模型：正常帧 + 退化帧（大误差） + 丢失帧。

    返回 (z_v, lost_mask, out_mask)。
    """
    n = len(truth)
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
    return z_v, lost_mask, out_mask


def simulate_imu(truth, rng):
    """IMU：比力比例误差 2% + 零偏随机游走 2e-4。"""
    n = len(truth)
    sf = 0.02
    acc_true = np.gradient(np.gradient(truth, DT), DT)
    k_ = max(3, n // 20)
    acc_true = np.convolve(acc_true, np.ones(k_) / k_, mode="same")
    bias = 0.0
    z_a = np.zeros(n)
    for i in range(n):
        bias += rng.normal(0, 2e-4)
        z_a[i] = acc_true[i] * (1 + sf) + bias
    return z_a


class FusionSession:
    """一次纠偏仿真会话：可逐帧步进（供实时演示），也可一次跑完。"""

    def __init__(self, beta=0.30, n=300, seed=11, use_track_prior=True,
                 use_fusion=True, use_gate=True, kind="irregular"):
        self.beta = beta
        self.n = n
        self.use_track_prior = use_track_prior
        self.use_fusion = use_fusion
        self.use_gate = use_gate
        rng = np.random.default_rng(seed)
        self.truth = make_truth(n, rng, kind)
        self.z_v, self.lost_mask, self.out_mask = simulate_vision(
            self.truth, beta, rng)
        self.z_a = simulate_imu(self.truth, rng)
        # 纯视觉零阶保持基线
        z_hold = np.copy(self.z_v)
        last = z_hold[0]
        for i in range(n):
            if self.lost_mask[i]:
                z_hold[i] = last
            else:
                last = self.z_v[i]
        self.z_hold = z_hold
        # 三点中值滤波观测（与实验口径一致：融合使用中值滤波后的视觉观测）
        z_med = np.copy(self.z_v)
        for i in range(1, n - 1):
            z_med[i] = np.median(self.z_v[i - 1:i + 2])
        self.z_med = z_med
        # 滤波初值
        self.x = np.array([self.truth[0], 0.0])
        self.P = np.eye(2) * np.array([[2.5e-3, 0], [0, 1e-2]])
        self.i = 0
        self.est = np.zeros(n)
        self.accepted = np.zeros(n, dtype=bool)
        self.n_reject = 0

    def step(self):
        """推进一帧。返回 dict（当前帧信息）；结束后返回 None。"""
        if self.i >= self.n:
            return None
        i = self.i
        if self.use_fusion:
            x, P, acc, innov = kalman_step(
                self.x.copy(), self.P.copy(), float(self.z_med[i]),
                self.z_a[i], self.beta, self.use_track_prior, i,
                lost=bool(self.lost_mask[i]), use_gate=self.use_gate)
            self.x, self.P = x, P
            self.est[i] = x[0]
            self.accepted[i] = acc
            if not acc:
                self.n_reject += 1
        else:
            self.est[i] = self.z_hold[i]
            self.accepted[i] = bool(not self.lost_mask[i])
        info = {
            "i": i, "t": i * DT,
            "truth": float(self.truth[i]),
            "vision": float(self.z_hold[i]),
            "vision_raw": float(self.z_v[i]),
            "est": float(self.est[i]),
            "lost": bool(self.lost_mask[i]),
            "outlier": bool(self.out_mask[i]),
            "accepted": bool(self.accepted[i]),
            "err_vision": float(abs(self.z_hold[i] - self.truth[i])),
            "err_est": float(abs(self.est[i] - self.truth[i])),
        }
        self.i += 1
        return info

    def summary(self):
        """跑完后的统计（与 E2 指标口径一致）。"""
        done = self.i
        if done == 0:
            return {}
        est = self.est[:done]
        truth = self.truth[:done]
        z_hold = self.z_hold[:done]
        rmse_est = float(np.sqrt(np.mean((est - truth) ** 2)))
        rmse_vis = float(np.sqrt(np.mean((z_hold - truth) ** 2)))
        p99_est = float(np.percentile(np.abs(est - truth), 99))
        p99_vis = float(np.percentile(np.abs(z_hold - truth), 99))
        ok_est = float(np.mean(np.abs(est - truth) <= 0.020) * 100)
        ok_vis = float(np.mean(np.abs(z_hold - truth) <= 0.020) * 100)
        return {
            "RMSE_纯视觉_m": rmse_vis,
            "RMSE_本方案_m": rmse_est,
            "P99_纯视觉_m": p99_vis,
            "P99_本方案_m": p99_est,
            "成功率_纯视觉_%": ok_vis,
            "成功率_本方案_%": ok_est,
            "RMSE下降_%": (1 - rmse_est / rmse_vis) * 100 if rmse_vis > 0 else 0.0,
            "门限剔除帧比例_%": self.n_reject / done * 100,
        }


def run_e2_full(beta=0.30, trials=30, n=300, seed=11):
    """完整蒙特卡洛（与 rerun_e2_offset.run_e2 同口径），返回统计表。"""
    methods = ["纯视觉(零阶保持)", "纯视觉+中值滤波", "纯惯性",
               "视觉-惯性融合", "融合+轨道几何约束"]
    per = {m: [] for m in methods}
    ok_rate = {m: [] for m in ("纯视觉(零阶保持)", "融合+轨道几何约束")}
    for tr in range(trials):
        rng = np.random.default_rng(seed + tr * 97)
        kind = ["straight", "curve", "irregular"][tr % 3]
        truth = make_truth(n, rng, kind)
        z_v, lost_mask, out_mask = simulate_vision(truth, beta, rng)
        z_a = simulate_imu(truth, rng)
        z_hold = np.copy(z_v)
        last = z_hold[0]
        for i in range(n):
            if lost_mask[i]:
                z_hold[i] = last
            else:
                last = z_v[i]
        z_med = np.copy(z_v)
        for i in range(1, n - 1):
            z_med[i] = np.median(z_v[i - 1:i + 2])
        # 纯惯性
        v = 0.0
        p = 0.0
        pure_imu = np.zeros(n)
        for i in range(n):
            v += z_a[i] * DT
            p += v * DT
            pure_imu[i] = p
        pure_imu = pure_imu - pure_imu[0] + 3.0
        # 融合（与实验一致：融合用中值滤波观测）
        x = np.array([truth[0], 0.0])
        P = np.eye(2) * np.array([[2.5e-3, 0], [0, 1e-2]])
        est_med = np.zeros(n)
        for i in range(n):
            x, P, _, _ = kalman_step(x.copy(), P.copy(), z_med[i], z_a[i],
                                     beta, False, i, lost=bool(lost_mask[i]))
            est_med[i] = x[0]
        x = np.array([truth[0], 0.0])
        P = np.eye(2) * np.array([[2.5e-3, 0], [0, 1e-2]])
        est_prior = np.zeros(n)
        for i in range(n):
            x, P, _, _ = kalman_step(x.copy(), P.copy(), z_med[i], z_a[i],
                                     beta, True, i, lost=bool(lost_mask[i]))
            est_prior[i] = x[0]
        errs = {
            "纯视觉(零阶保持)": np.sqrt(np.mean((z_hold - truth) ** 2)),
            "纯视觉+中值滤波": np.sqrt(np.mean((z_med - truth) ** 2)),
            "纯惯性": np.sqrt(np.mean((pure_imu - truth) ** 2)),
            "视觉-惯性融合": np.sqrt(np.mean((est_med - truth) ** 2)),
            "融合+轨道几何约束": np.sqrt(np.mean((est_prior - truth) ** 2)),
        }
        for m, e in errs.items():
            per[m].append(float(e))
        for m, est in (("纯视觉(零阶保持)", z_hold),
                       ("融合+轨道几何约束", est_prior)):
            ok_rate[m].append(float(np.mean(np.abs(est - truth) <= 0.020)))
    return {
        "beta": beta,
        "RMSE_m": {m: float(np.mean(per[m])) for m in methods},
        "成功率_%": {m: float(np.mean(ok_rate[m])) * 100 for m in ok_rate},
    }

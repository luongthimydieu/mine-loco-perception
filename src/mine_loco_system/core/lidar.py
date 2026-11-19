# -*- coding: utf-8 -*-
"""二维激光雷达轨道障碍物检测核心（E3，与 scripts/rerun_e3_lidar.py 同源）。

雷达装于车头、轨面以上 0.15 m，水平扫描：481 束 × 0.25°，±60°，量程 40 m。
粉尘双程消光：回波强度 ×exp(-2βd)，丢包概率 1-exp(-2βd)，
后向散射杂散回波概率 (1-exp(-2β·15))·0.25。
判据：|方位角|≤8°、|y|≤0.90 m、d∈[4,25) m、强度≥0.12，簇内回波数≥2。
"""
import numpy as np

N_BEAMS = 481
Y_WALL = 2.30
CORRIDOR = 0.90
OBS_HALF = 0.90
OBS_MAX = 6.5
DET_RANGE = 25.0
D_MIN = 4.0
I_THR = 0.12
A_MAX = np.deg2rad(8.0)

ANGLES = np.linspace(-np.pi / 3, np.pi / 3, N_BEAMS)
SIN_A = np.sin(ANGLES)
ANG_OK = np.abs(ANGLES) <= A_MAX
with np.errstate(divide="ignore", invalid="ignore"):
    D_WALL = np.where(np.abs(SIN_A) > 1e-9,
                      Y_WALL / np.abs(np.where(np.abs(SIN_A) > 1e-9, SIN_A, 1.0)),
                      np.inf)
D_WALL = np.where(D_WALL <= 40.0, D_WALL, np.nan)


def simulate_frame(rng, beta, obstacle=True, obs_d=None):
    """生成一帧点云 (d_meas, rho_meas)。与实验 simulate() 单帧逻辑一致。"""
    if obstacle and obs_d is None:
        obs_d = float(rng.uniform(4.0, OBS_MAX))
    d_true = D_WALL.copy()
    rho = rng.uniform(0.02, 0.05, N_BEAMS)
    if obstacle:
        y_hit = np.abs(obs_d * SIN_A) <= OBS_HALF
        hit = np.isfinite(d_true) & y_hit & (obs_d <= DET_RANGE)
        d_true = np.where(hit, obs_d, d_true)
        rho = np.where(hit, rng.uniform(0.85, 1.00), rho)
    valid = np.isfinite(d_true)
    t2 = np.exp(-2.0 * beta * np.where(valid, d_true, 0.0))
    rho_meas = rho * t2
    p_drop = 1.0 - t2
    keep = valid & (rng.random(N_BEAMS) > p_drop)
    sigma = 0.02 + 0.01 * np.where(valid, d_true, 0.0) + \
        0.6 * beta * np.where(valid, d_true, 0.0)
    d_meas = np.where(valid, d_true, np.nan) + rng.normal(0.0, 1.0, N_BEAMS) * sigma
    d_meas = np.where(keep, d_meas, np.nan)
    p_sc = (1.0 - np.exp(-2.0 * beta * 15.0)) * 0.25
    sc = rng.random(N_BEAMS) < p_sc
    d_meas = np.where(sc, rng.uniform(3.0, 20.0, N_BEAMS), d_meas)
    rho_meas = np.where(sc, rng.uniform(0.01, 0.05, N_BEAMS), rho_meas)
    rho_meas = rho_meas + np.abs(rng.normal(0.0, 0.02, N_BEAMS))
    return d_meas, rho_meas


def detect(d_meas, rho_meas):
    """候选回波簇判决。返回 (distance or None, n_candidates)。"""
    y_lat = d_meas * SIN_A
    cand = (ANG_OK & (np.abs(y_lat) <= CORRIDOR) & (d_meas >= D_MIN) &
            (d_meas < DET_RANGE) & (rho_meas >= I_THR))
    k = int(cand.sum())
    est = float(np.nanmedian(d_meas[cand])) if k >= 2 else None
    return est, k


class LidarSession:
    """雷达演示会话：维护统计与场景状态。"""

    def __init__(self, beta=0.0, seed=23, obstacle_prob=0.6):
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.obstacle_prob = obstacle_prob
        self.tp = self.fp = self.fn = self.tn = 0
        self.errs = []
        self.n_echo_sum = 0.0
        self.frames = 0
        self.obstacle_distance = None
        self.last_frame = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.tp = self.fp = self.fn = self.tn = 0
        self.errs = []
        self.n_echo_sum = 0.0
        self.frames = 0

    def step(self, force_obstacle=None, obs_d=None):
        """推进一帧：仿真 → 判决 → 更新统计。返回帧信息 dict。"""
        if force_obstacle is None:
            obstacle = self.rng.random() < self.obstacle_prob
        else:
            obstacle = force_obstacle
        if obstacle and obs_d is None:
            obs_d = float(self.rng.uniform(4.0, OBS_MAX))
        d_meas, rho_meas = simulate_frame(self.rng, self.beta, obstacle, obs_d)
        est, k = detect(d_meas, rho_meas)
        has_obs = obstacle
        verdict = "none"
        if est is not None:
            if has_obs and abs(est - obs_d) < 0.5:
                self.tp += 1
                self.errs.append(abs(est - obs_d))
                verdict = "hit"
            elif has_obs:
                self.fn += 1
                verdict = "miss"
            else:
                self.fp += 1
                verdict = "false"
        else:
            if has_obs:
                self.fn += 1
                verdict = "miss"
            else:
                self.tn += 1
                verdict = "clear"
        n_echo = int(np.sum(~np.isnan(d_meas)))
        self.n_echo_sum += n_echo
        self.frames += 1
        self.obstacle_distance = obs_d if obstacle else None
        self.last_frame = (d_meas, rho_meas)
        return {
            "d_meas": d_meas, "rho_meas": rho_meas,
            "estimate": est, "n_cand": k, "verdict": verdict,
            "has_obstacle": has_obs, "obs_d": obs_d if obstacle else None,
            "n_echo": n_echo,
        }

    def metrics(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        n = max(self.frames, 1)
        return {
            "检出率_%": round(tp / max(tp + fn, 1) * 100, 2),
            "精确率_%": round(tp / max(tp + fp, 1) * 100, 2),
            "虚警率_%": round(fp / max(tp + fp, 1) * 100, 2),
            "测距RMSE_m": (round(float(np.sqrt(np.mean(np.square(self.errs)))), 3)
                           if self.errs else None),
            "点云有效率_%": round(self.n_echo_sum / n / N_BEAMS * 100, 2),
            "帧数": self.frames,
        }


def reliable_range_curve(vs, rho0=1.0, i_thr=0.14):
    """理论可靠检出距离随能见度的曲线（双程消光）。"""
    return [reliable_range(V, rho0, i_thr) for V in vs]


def reliable_range(V, rho0=1.0, i_thr=0.14):
    beta = 3.0 / max(V, 0.1)
    return float(np.log(rho0 / i_thr) / (2.0 * beta))

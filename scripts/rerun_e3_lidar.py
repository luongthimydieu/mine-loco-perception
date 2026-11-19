# -*- coding: utf-8 -*-
r"""实验三（二维激光雷达轨道障碍物检测）独立重跑脚本。

独立重写说明
------------
**不 import** ``原代码`` 下任何模块。按 ``原代码\run_experiments.py``
第 272-451 行的设定重新实现：

* 雷达装于车头、轨面以上 0.15 m，水平扫描，角分辨率 0.25°，共 481 条射线，
  方位范围 ±60°，量程 40 m；侧壁横向 ±2.30 m，反射率 0.02~0.05；
* 障碍物：横跨整个通行区（半宽 0.90 m），高 0.7 m，距离 U(4.0, 6.5) m，
  出现概率 0.6；反射率 U(0.85, 1.00)；
* 粉尘：双程消光 exp(-2βd)，回波补齐概率 1-exp(-2βd) 丢失，测距噪声
  σ = 0.02 + 0.01d + 0.6βd；后向散射杂散回波概率 (1-exp(-2β·15))·0.25，
  强度 0.01~0.05，另叠加 |N(0,0.02)| 强度噪声；
* 判据：|方位角| ≤ 8°、|y|=d·sinθ ≤ 0.90 m、d ∈ [4, 25) m、强度 ≥ 0.12；
  候选回波簇中值测距，簇内回波数 ≥ 2 才上报；
* β = [0.0, 0.1, 0.2, 0.3, 0.4]，每档 400 帧，同一条 default_rng(23) 流。

同样保持与原实现**完全相同的随机数调用顺序**，以检验逐位可复现性。
**不做任何参数调整去凑原值。**

运行：
    D:\APP\Anaconda\python.exe rerun_e3_lidar.py
输出：
    outputs/repro\e3_lidar.json
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

SEED = 23
N_FRAMES = 400
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4]
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


def simulate(rng, beta, n_frames=N_FRAMES):
    """按物理模型生成 n_frames 帧点云，返回 [(d_meas, rho_meas, obs_d), ...]。"""
    frames = []
    for _ in range(n_frames):
        has_obs = rng.random() < 0.6
        obs_d = float(rng.uniform(4.0, OBS_MAX)) if has_obs else -1.0
        d_true = D_WALL.copy()
        rho = rng.uniform(0.02, 0.05, N_BEAMS)
        if has_obs:
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
        frames.append((d_meas, rho_meas, obs_d))
    return frames


def evaluate(frames):
    tp = fp = fn = tn = 0
    errs = []
    n_echo = []
    n_cand = []
    for d_meas, rho_meas, obs_d in frames:
        n_echo.append(int(np.sum(~np.isnan(d_meas))))
        y_lat = d_meas * SIN_A
        cand = (ANG_OK & (np.abs(y_lat) <= CORRIDOR) & (d_meas >= D_MIN) &
                (d_meas < DET_RANGE) & (rho_meas >= I_THR))
        k = int(cand.sum())
        n_cand.append(k)
        est = float(np.nanmedian(d_meas[cand])) if k >= 2 else None
        has_obs = obs_d > 0
        if est is not None:
            if has_obs and abs(est - obs_d) < 0.5:
                tp += 1
                errs.append(abs(est - obs_d))
            elif has_obs:
                fn += 1
            else:
                fp += 1
        else:
            if has_obs:
                fn += 1
            else:
                tn += 1
    n = len(frames)
    return {
        "检出率/召回率(%)": round(tp / max(tp + fn, 1) * 100, 2),
        "准确率(%)": round((tp + tn) / n * 100, 2),
        "精确率(%)": round(tp / max(tp + fp, 1) * 100, 2),
        "虚警率(%)": round(fp / max(tp + fp, 1) * 100, 2),
        "测距平均误差(m)": round(float(np.mean(errs)), 3) if errs else None,
        "测距RMSE(m)": round(float(np.sqrt(np.mean(np.square(errs)))), 3) if errs else None,
        "点云有效率(%)": round(float(np.mean(n_echo)) / N_BEAMS * 100, 2),
        "候选回波簇平均点数": round(float(np.mean(n_cand)), 2),
    }


def run_e3(seed=SEED):
    rng = np.random.default_rng(seed)
    results_by_beta = {}
    raw = {}
    for beta in BETAS:
        frames = simulate(rng, beta)
        results_by_beta[beta] = evaluate(frames)
        raw[str(beta)] = dict(results_by_beta[beta])
    res = {
        "各粉尘消光系数下检测性能": {str(k): v for k, v in results_by_beta.items()},
        "无粉尘检出率(%)": results_by_beta[0.0]["检出率/召回率(%)"],
        "高粉尘beta=0.4检出率(%)": results_by_beta[0.4]["检出率/召回率(%)"],
        "检出率下降(百分点)": round(results_by_beta[0.0]["检出率/召回率(%)"] -
                                    results_by_beta[0.4]["检出率/召回率(%)"], 2),
        "无粉尘测距RMSE(m)": results_by_beta[0.0]["测距RMSE(m)"],
        "高粉尘测距RMSE(m)": results_by_beta[0.4]["测距RMSE(m)"],
    }
    return res, raw


def compare(new, orig):
    diffs = []
    for b in BETAS:
        a_all = new["各粉尘消光系数下检测性能"][str(b)]
        o_all = orig["各粉尘消光系数下检测性能"][str(b)]
        for k in o_all:
            a, o = a_all[k], o_all[k]
            diffs.append(dict(组="各粉尘消光系数下检测性能", 指标=k, beta=b,
                              原值=o, 重跑值=a, 差异=(None if (a is None or o is None)
                                                        else round(a - o, 6)),
                              一致=(a == o)))
    for k in ("无粉尘检出率(%)", "高粉尘beta=0.4检出率(%)", "检出率下降(百分点)",
              "无粉尘测距RMSE(m)", "高粉尘测距RMSE(m)"):
        a, o = new[k], orig[k]
        diffs.append(dict(组="标量", 指标=k, beta="", 原值=o, 重跑值=a,
                          差异=(None if (a is None or o is None) else round(a - o, 6)),
                          一致=(a == o)))
    return diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ORIG_JSON, "r", encoding="utf-8") as f:
        orig = json.load(f)["E3_激光雷达障碍物检测"]
    res, raw = run_e3()
    diffs = compare(res, orig)
    n_ok = sum(1 for d in diffs if d["一致"])
    print("=" * 78)
    print("E3 独立重跑完成：对照项 %d，逐位一致 %d，不一致 %d"
          % (len(diffs), n_ok, len(diffs) - n_ok))
    print("=" * 78)
    for d in diffs:
        if not d["一致"]:
            print("  [不一致] %s %s beta=%s 原值=%s 重跑值=%s 差异=%s"
                  % (d["组"], d["指标"], d["beta"], d["原值"], d["重跑值"], d["差异"]))
    for b in BETAS:
        r = res["各粉尘消光系数下检测性能"][str(b)]
        print("  β=%.2f 检出率=%s%% 准确率=%s%% 点云有效=%s%% 测距RMSE=%s m"
              % (b, r["检出率/召回率(%)"], r["准确率(%)"], r["点云有效率(%)"],
                 r["测距RMSE(m)"]))
    payload = {
        "实验": "E3_激光雷达障碍物检测",
        "重跑脚本": os.path.basename(__file__),
        "设定": dict(随机种子=SEED, 每档帧数=N_FRAMES, beta档=BETAS,
                     射线数=N_BEAMS, 角分辨率_deg=0.25, 量程_m=40.0,
                     侧壁横向_m=Y_WALL, 通行区半宽_m=CORRIDOR,
                     障碍物距离范围_m=[4.0, OBS_MAX], 出现概率=0.6,
                     方位角门限_deg=8.0, 强度门限=I_THR, 制动盲区_m=D_MIN),
        "重跑结果": res,
        "逐位一致项": n_ok,
        "对照项总数": len(diffs),
        "差异清单": diffs,
        "_raw": raw,
    }
    out = os.path.join(OUT_DIR, "e3_lidar.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n[写出] %s" % out)


if __name__ == "__main__":
    main()

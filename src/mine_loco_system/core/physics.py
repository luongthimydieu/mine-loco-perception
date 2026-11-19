# -*- coding: utf-8 -*-
"""物理核心：大气散射退化建模与激光雷达双程消光边界。

与 docs/method.md、scripts/exp_real_detect_fixed.py 同源：
  I = J·t + A·(1-t)，t = exp(-β·d)，β = 3 / V（Koschmieder 定律）
  单目地面平面投影：d(y) = h·f / (y - y_h)
  雷达双程消光：ρ_meas = ρ·exp(-2β·d)，可靠检出距离 R = ln(ρ0/I_thr)/(2β)
"""
import numpy as np

H_LENS = 0.80        # 镜头离地高度 (m)
H_FOV_DEG = 60.0     # 水平视场角 (°)
AIRLIGHT = 60.0      # 气光灰度 A（DN）
NOISE_SIGMA = 1.5    # 传感器噪声等效灰度 (DN)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:                                  # pragma: no cover
    _HAS_PIL = False


def beta_from_visibility(V):
    """能见度 V (m) → 消光系数 β = 3/V (1/m)。"""
    return 3.0 / max(V, 0.1)


def visibility_from_beta(beta):
    return 3.0 / max(beta, 1e-6)


def focal_px(width):
    """由水平视场角计算像素焦距。"""
    return (width / 2.0) / np.tan(np.deg2rad(H_FOV_DEG / 2.0))


def depth_map(h, w):
    """单目地面平面投影：逐像素纵向距离（米）。"""
    f = focal_px(w)
    y_h = h / 2.0
    yy = np.arange(h, dtype=np.float32)[:, None]
    return np.clip(H_LENS * f / np.maximum(yy - y_h, 1.0), 1.0, 200.0)


def contrast_retention(beta, d):
    """目标-背景对比度保留率 exp(-β·d)。"""
    return float(np.exp(-beta * d))


def radar_reliable_range(V, rho0=1.0, i_thr=0.14):
    """激光雷达双程消光下的理论可靠检出距离 R = ln(ρ0/I_thr)/(2β)。"""
    beta = beta_from_visibility(V)
    return float(np.log(rho0 / i_thr) / (2.0 * beta))


def degrade_array(arr, depth, V, seed=0, noise=NOISE_SIGMA):
    """按大气散射模型退化 uint8 图像数组（与实验 degrade() 一致）。"""
    rng = np.random.default_rng(seed)
    b = beta_from_visibility(V)
    t = np.exp(-b * depth)                       # (H,1) 逐行透射率
    out = arr.astype(np.float32) * t[:, :, None] + AIRLIGHT * (1.0 - t[:, :, None])
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_uniform_noise(arr, t_mean, seed=0, noise=NOISE_SIGMA):
    """非物理对照：全图均匀变暗 + 加性高斯噪声（无深度结构）。

    t_mean 取物理退化在全图深度上的平均透射率，保证两种增广的
    全局灰度统计量级一致，唯一差别是是否具有深度相关的空间结构。
    """
    rng = np.random.default_rng(seed)
    out = arr.astype(np.float32) * t_mean + AIRLIGHT * (1.0 - t_mean)
    out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def mean_transmission(depth, V):
    """全图平均透射率（用于非物理对照的强度匹配）。"""
    b = beta_from_visibility(V)
    return float(np.mean(np.exp(-b * depth)))


if _HAS_PIL:

    def degrade_pil_image(im, V, seed=0, target_w=416):
        """读入 PIL Image → 缩放 → 物理退化。返回 (退化图, 深度图)。"""
        try:
            lanczos = Image.Resampling.LANCZOS
        except AttributeError:                       # pragma: no cover
            lanczos = Image.LANCZOS
        im = im.convert('RGB')
        nh = max(2, int(target_w * im.height / im.width))
        im = im.resize((target_w, nh), lanczos)
        arr = np.asarray(im)
        depth = depth_map(arr.shape[0], target_w)
        return degrade_array(arr, depth, V, seed), depth


def photometric_jitter(arr, seed=0):
    """仅 HSV/光度类传统增广：亮度-伽马-对比度随机抖动（无散射结构、无噪声）。"""
    rng = np.random.default_rng(seed)
    a = arr.astype(np.float32) / 255.0
    gain = rng.uniform(0.35, 0.75)               # 亮度压暗
    gamma = rng.uniform(0.8, 1.4)                # 伽马
    a = np.power(a * gain, gamma)
    # 轻度饱和度扰动（RGB 通道增益）
    ch_gain = rng.uniform(0.9, 1.1, size=(1, 1, 3))
    a = np.clip(a * ch_gain, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)

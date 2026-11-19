# -*- coding: utf-8 -*-
"""E3 激光雷达：双程消光边界与高粉尘检出率归零。"""
import numpy as np

from mine_loco_system.core import lidar as L


def test_reliable_range_curve_increases_with_visibility():
    vals = L.reliable_range_curve([10, 20, 30, 50])
    assert vals == sorted(vals)
    assert abs(L.reliable_range(20.0) - 6.55) < 0.02


def _detection_rate(beta, frames=80, seed=7):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(frames):
        d, rho = L.simulate_frame(rng, beta, obstacle=True, obs_d=6.0)
        est, k = L.detect(d, rho)
        hits += int(est is not None)
    return hits / frames


def test_clean_detection_is_high():
    assert _detection_rate(0.0) > 0.8


def test_heavy_dust_detection_is_zero():
    # β=0.40 时 6 m 处回波强度 exp(-2*0.4*6)=0.008，远低于 I_THR=0.12
    assert _detection_rate(0.40) == 0.0


def test_detection_degrades_with_dust():
    rates = [_detection_rate(b) for b in (0.0, 0.1, 0.3, 0.4)]
    assert rates == sorted(rates, reverse=True)

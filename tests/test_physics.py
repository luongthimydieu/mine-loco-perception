# -*- coding: utf-8 -*-
"""物理核心的不变量测试（大气散射 / 能见度 / 雷达双程消光）。"""
import numpy as np
import pytest

from mine_loco_system.core import physics as P


def test_beta_visibility_roundtrip():
    assert P.beta_from_visibility(20.0) == pytest.approx(0.15)
    assert P.visibility_from_beta(0.15) == pytest.approx(20.0)
    assert P.beta_from_visibility(30.0) == pytest.approx(0.10)


def test_contrast_retention_matches_report():
    # 报告 §3 的两个关键数字：30 m / 10 m -> 36.8%；20 m / 6 m -> 40.7%
    assert P.contrast_retention(P.beta_from_visibility(30.0), 10.0) == pytest.approx(0.3679, abs=1e-4)
    assert P.contrast_retention(P.beta_from_visibility(20.0), 6.0) == pytest.approx(0.4066, abs=1e-4)


def test_radar_reliable_range_matches_report():
    # R = ln(1/0.14)/(2β)，V=20 m -> 6.51 m
    assert P.radar_reliable_range(20.0) == pytest.approx(6.55, abs=0.02)
    # 能见度越高，可靠距离越远
    vals = [P.radar_reliable_range(v) for v in (10, 20, 30, 50)]
    assert vals == sorted(vals)


def test_depth_map_monotonic_and_bounded():
    d = P.depth_map(240, 416)
    assert d.shape == (240, 1)
    assert np.all(d >= 1.0) and np.all(d <= 200.0)
    # 图像下方（近处）距离不大于上方（远处）
    assert np.all(np.diff(d[:, 0]) <= 1e-6)


def test_degrade_is_darker_at_lower_visibility():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 416, 3), dtype=np.uint8)
    depth = P.depth_map(240, 416)
    a = P.degrade_array(img, depth, 50.0, seed=1)
    b = P.degrade_array(img, depth, 10.0, seed=1)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert b.mean() < a.mean()
    # 同参数同种子必须可复现
    assert np.array_equal(a, P.degrade_array(img, depth, 50.0, seed=1))


def test_mean_transmission_decreases_with_dust():
    depth = P.depth_map(240, 416)
    assert P.mean_transmission(depth, 50.0) > P.mean_transmission(depth, 20.0) > \
        P.mean_transmission(depth, 10.0)


def test_photometric_jitter_shape():
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    out = P.photometric_jitter(img, seed=0)
    assert out.shape == img.shape and out.dtype == np.uint8

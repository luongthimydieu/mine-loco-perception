# -*- coding: utf-8 -*-
"""E2 抗差融合：约束版本应优于纯视觉零阶保持。"""
import numpy as np
import pytest

from mine_loco_system.core.fusion import FusionSession


def _run(**kw):
    s = FusionSession(**kw)
    while s.step() is not None:
        pass
    return s.summary()


def test_fusion_beats_pure_vision():
    m = _run(beta=0.30, n=300, seed=11)
    assert m['RMSE_本方案_m'] < m['RMSE_纯视觉_m']
    assert m['成功率_本方案_%'] > m['成功率_纯视觉_%']
    assert m['RMSE下降_%'] > 50.0


def test_track_prior_and_gate_ablation():
    base = _run(beta=0.30, n=300, seed=11)
    ablated = _run(beta=0.30, n=300, seed=11, use_track_prior=False,
                   use_gate=False)
    # 去掉轨道几何约束与抗差门限后，RMSE 不应优于完整方案
    assert ablated['RMSE_本方案_m'] >= base['RMSE_本方案_m']


def test_no_dust_is_easier_than_high_dust():
    clean = _run(beta=0.0, n=300, seed=11)
    dusty = _run(beta=0.30, n=300, seed=11)
    assert clean['RMSE_纯视觉_m'] <= dusty['RMSE_纯视觉_m']

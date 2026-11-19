# -*- coding: utf-8 -*-
"""E4 运维健康：−9 dB 包络谱检出、寿命预测与融合权重。"""
import numpy as np

from mine_loco_system.core import health as H


def test_envelope_detected_at_minus9db():
    r = H.envelope_spectrum(-9.0, seed=31)
    assert r['detected'] is True
    assert r['ratio'] > 1.0
    assert r['peak'] > r['floor']


def test_envelope_ratio_improves_with_snr():
    ratios = [H.envelope_spectrum(s, seed=31)['ratio'] for s in (-12, -9, -6, 0)]
    assert ratios == sorted(ratios)


def test_health_history_and_weights():
    h = H.health_history(seed=36)
    assert h['fail_idx'] == 184
    assert abs(sum(h['weights'].values()) - 1.0) < 1e-6
    assert len(h['fused']) == len(h['health'])


def test_rul_table_reasonable():
    h = H.health_history(seed=36)
    rows = H.rul_table(h['fused'], h['fail_idx'])
    assert rows, 'RUL 表不应为空'
    for r in rows:
        assert r['预测剩余寿命(天)'] > 0
        assert np.isfinite(r['相对误差(%)'])

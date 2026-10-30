"""Permanent regression checks for final data preparation and metric definitions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))


def test_next_step_and_no_future_fill():
    from prepare_data import make_windows
    raw = np.arange(40, dtype=float).reshape(20, 2)
    raw[6, 0] = np.nan
    x, y, idx = make_windows(raw, target=0, depth=3)
    assert not any(i in idx for i in [6, 7, 8, 9])
    assert np.array_equal(x[0], raw[:3])
    assert y[0] == raw[3, 0]
    assert np.array_equal(y, raw[idx, 0])
    assert np.isfinite(x).all()


def test_windows_do_not_cross_segments():
    from prepare_data import make_windows
    raw = np.arange(24, dtype=float).reshape(12, 2)
    _, _, idx = make_windows(raw, 0, 3, np.repeat([0, 1], 6))
    assert idx.tolist() == [3, 4, 5, 9, 10, 11]


def test_train_only_scaling_constant_feature_and_disjoint_targets():
    from prepare_data import scale_splits
    x = np.arange(36, dtype=float).reshape(6, 3, 2)
    x[:, :, 1] = 5
    y = np.arange(6, dtype=float)
    x[-1, :, 0] = 1000
    d = scale_splits(x, y, np.arange(3, 9), (6, 7))
    assert d['train_idx'].tolist() == [3, 4, 5]
    assert d['val_idx'].tolist() == [6]
    assert d['test_idx'].tolist() == [7, 8]
    assert d['test_x'][-1, :, 0].min() > 1
    assert np.isfinite(d['train_x']).all()
    assert np.array_equal(d['train_y'] * d['y_scale'] + d['y_min'], y[:3])


def test_metric_definition():
    from repro.engine import metrics
    d = metrics(np.array([1., 3.]), np.array([2., 1.]))
    assert d['rmse'] == pytest.approx(np.sqrt(2.5))
    assert d['mae'] == 1.5

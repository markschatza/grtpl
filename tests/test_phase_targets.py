import numpy as np

from grtpl.targets import circular_error, next_crossing_for_samples


def test_circular_error_wraps_across_boundary():
    pred = np.deg2rad(np.array([359.0]))
    target = np.deg2rad(np.array([1.0]))
    err_deg = np.rad2deg(circular_error(pred, target))
    assert np.allclose(err_deg, [-2.0])


def test_next_crossing_for_samples_uses_future_crossing():
    samples = np.array([0.1, 0.5, 1.2])
    crossings = np.array([0.4, 1.0, 1.6])
    targets = next_crossing_for_samples(samples, crossings)
    assert np.allclose(targets, [0.4, 1.0, 1.6])


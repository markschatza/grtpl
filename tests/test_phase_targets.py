import numpy as np

from grtpl.hc3 import load_eeg_channels
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


def test_load_eeg_channels_reads_interleaved_int16(tmp_path):
    samples = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype="<i2",
    )
    eeg_path = tmp_path / "toy.eeg"
    samples.tofile(eeg_path)

    selected = load_eeg_channels(eeg_path, n_channels=3, channels=[0, 2])

    assert selected.dtype == np.float32
    assert np.array_equal(selected, np.array([[1, 3], [4, 6], [7, 9]], dtype=np.float32))

from __future__ import annotations

import numpy as np
from scipy import signal


def design_theta_sos(sample_rate_hz: float, band_hz: tuple[float, float] = (6.0, 10.0), order: int = 4):
    nyquist = sample_rate_hz / 2.0
    low, high = band_hz
    if not 0 < low < high < nyquist:
        raise ValueError(f"Invalid band {band_hz} for sample rate {sample_rate_hz}")
    return signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")


def causal_bandpass(x: np.ndarray, sample_rate_hz: float, band_hz=(6.0, 10.0), order: int = 4):
    sos = design_theta_sos(sample_rate_hz, band_hz, order)
    zi = signal.sosfilt_zi(sos) * float(x[0])
    y, zf = signal.sosfilt(sos, x, zi=zi)
    return y, sos, zf


def acausal_bandpass(x: np.ndarray, sample_rate_hz: float, band_hz=(6.0, 10.0), order: int = 4):
    sos = design_theta_sos(sample_rate_hz, band_hz, order)
    return signal.sosfiltfilt(sos, x)


def hilbert_phase(x: np.ndarray) -> np.ndarray:
    analytic = signal.hilbert(x)
    return np.angle(analytic)


def wrap_phase_radians(phase: np.ndarray | float):
    return (phase + np.pi) % (2 * np.pi) - np.pi


def decimate_by_timestamp(
    timestamps: np.ndarray,
    values: np.ndarray,
    output_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have the same length")
    step = 1.0 / output_rate_hz
    sample_times = np.arange(timestamps[0], timestamps[-1], step)
    sample_indices = np.searchsorted(timestamps, sample_times, side="left")
    sample_indices = sample_indices[sample_indices < len(values)]
    return timestamps[sample_indices], values[sample_indices]


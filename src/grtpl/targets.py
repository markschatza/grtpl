from __future__ import annotations

import numpy as np


def circular_error(pred_phase: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred_phase - target_phase), np.cos(pred_phase - target_phase))


def find_next_phase_crossings(
    timestamps: np.ndarray,
    reference_phase: np.ndarray,
    target_phase_rad: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = np.unwrap(reference_phase - target_phase_rad)
    cycle = 2 * np.pi
    crossing_numbers = np.floor((shifted - shifted[0]) / cycle)
    crossing_mask = np.diff(crossing_numbers) > 0
    crossing_indices = np.flatnonzero(crossing_mask) + 1
    return timestamps[crossing_indices], crossing_indices


def next_crossing_for_samples(
    sample_timestamps: np.ndarray,
    crossing_timestamps: np.ndarray,
    max_horizon_s: float | None = None,
) -> np.ndarray:
    indices = np.searchsorted(crossing_timestamps, sample_timestamps, side="right")
    targets = np.full(sample_timestamps.shape, np.nan, dtype=float)
    valid = indices < len(crossing_timestamps)
    targets[valid] = crossing_timestamps[indices[valid]]
    if max_horizon_s is not None:
        too_far = (targets - sample_timestamps) > max_horizon_s
        targets[too_far] = np.nan
    return targets


def nominal_phase_targets(
    current_phase: np.ndarray,
    sample_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    nominal_frequency_hz: float,
) -> np.ndarray:
    delta_t = target_timestamps - sample_timestamps
    phase = current_phase + 2 * np.pi * nominal_frequency_hz * delta_t
    return np.arctan2(np.sin(phase), np.cos(phase))


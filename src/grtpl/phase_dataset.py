from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grtpl.hc3 import download_buzsaki_hc3_session_files, load_eeg_channels, read_hc3_xml_metadata
from grtpl.signal import acausal_bandpass, causal_bandpass, decimate_by_timestamp, hilbert_phase
from grtpl.targets import find_next_phase_crossings, next_crossing_for_samples, nominal_phase_targets


@dataclass(frozen=True)
class HC3PhaseSession:
    topdir: str
    session: str
    selected_channels: tuple[int, int] = (38, 39)
    behavior: str | None = None


DEFAULT_EC013_MWHEEL_SESSIONS = (
    HC3PhaseSession("ec013.33", "ec013.541", behavior="Mwheel"),
    HC3PhaseSession("ec013.33", "ec013.543", behavior="Mwheel"),
    HC3PhaseSession("ec013.33", "ec013.544", behavior="Mwheel"),
)


def build_session_phase_target_table(
    raw_dir: Path,
    session_cfg: HC3PhaseSession,
    theta_band_hz: tuple[float, float] = (6.0, 10.0),
    nominal_frequency_hz: float = 8.0,
    transformer_input_rate_hz: float = 25.0,
    max_target_horizon_s: float = 0.5,
) -> pd.DataFrame:
    download_buzsaki_hc3_session_files(
        session_cfg.topdir,
        session_cfg.session,
        raw_dir,
        extensions=("xml", "eeg"),
    )

    session_dir = raw_dir / "hc3" / session_cfg.topdir / session_cfg.session
    xml_path = session_dir / f"{session_cfg.session}.xml"
    eeg_path = session_dir / f"{session_cfg.session}.eeg"

    metadata = read_hc3_xml_metadata(xml_path)
    sample_rate_hz = metadata["lfp_sampling_rate_hz"]
    n_channels = metadata["n_channels"]

    channel_lfp = load_eeg_channels(
        eeg_path,
        n_channels=n_channels,
        channels=list(session_cfg.selected_channels),
    )
    differential_lfp = channel_lfp[:, 1] - channel_lfp[:, 0]
    timestamps = np.arange(len(differential_lfp)) / sample_rate_hz

    causal_theta, _, _ = causal_bandpass(
        differential_lfp,
        sample_rate_hz,
        band_hz=theta_band_hz,
        order=4,
    )
    reference_theta = acausal_bandpass(
        differential_lfp,
        sample_rate_hz,
        band_hz=theta_band_hz,
        order=4,
    )

    causal_phase = hilbert_phase(causal_theta)
    reference_phase = hilbert_phase(reference_theta)
    input_timestamps, input_phase = decimate_by_timestamp(
        timestamps,
        causal_phase,
        transformer_input_rate_hz,
    )

    crossing_timestamps, _ = find_next_phase_crossings(
        timestamps,
        reference_phase,
        target_phase_rad=np.pi,
    )
    target_timestamps = next_crossing_for_samples(
        input_timestamps,
        crossing_timestamps,
        max_horizon_s=max_target_horizon_s,
    )
    target_phase = nominal_phase_targets(
        input_phase,
        input_timestamps,
        target_timestamps,
        nominal_frequency_hz=nominal_frequency_hz,
    )

    causal_phase_deg = np.rad2deg(input_phase) % 360
    table = pd.DataFrame(
        {
            "topdir": session_cfg.topdir,
            "session": session_cfg.session,
            "behavior": session_cfg.behavior,
            "input_time_s": input_timestamps,
            "phase_token": np.floor(causal_phase_deg).astype(np.int64),
            "causal_phase_deg": causal_phase_deg,
            "target_time_s": target_timestamps,
            "target_nominal_phase_rad": target_phase,
            "target_nominal_phase_deg": np.rad2deg(target_phase) % 360,
            "lead_time_ms": (target_timestamps - input_timestamps) * 1000,
        }
    )
    return table.dropna().reset_index(drop=True)


def build_multi_session_phase_target_table(
    raw_dir: Path,
    sessions: tuple[HC3PhaseSession, ...] = DEFAULT_EC013_MWHEEL_SESSIONS,
    theta_band_hz: tuple[float, float] = (6.0, 10.0),
    nominal_frequency_hz: float = 8.0,
    transformer_input_rate_hz: float = 25.0,
    max_target_horizon_s: float = 0.5,
) -> pd.DataFrame:
    tables = []
    for session_index, session_cfg in enumerate(sessions):
        table = build_session_phase_target_table(
            raw_dir=raw_dir,
            session_cfg=session_cfg,
            theta_band_hz=theta_band_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            transformer_input_rate_hz=transformer_input_rate_hz,
            max_target_horizon_s=max_target_horizon_s,
        )
        table.insert(0, "session_index", session_index)
        tables.append(table)

    return pd.concat(tables, ignore_index=True)

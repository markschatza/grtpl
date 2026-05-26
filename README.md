# GRTPL

Generative Real-Time Phase Lock.

This project explores transformer-based real-time phase-locked stimulation timing from hippocampal LFP. The first target is CRCNS HC-3, using adjacent hippocampal channels subtracted into differential LFP traces, causal 6-10 Hz theta phase estimation, aggressive final decimation, and a transformer trained to predict the nominal future phase corresponding to the next acausal 180 degree theta target.

## Start Here

The project is notebook-first. Begin with:

```text
notebooks/00_hc3_download_and_phase_targets.ipynb
```

That notebook downloads public HC-3 documentation/metadata and prepares a selected session download path. Full HC-3 LFP archives are large, so raw session data is intentionally excluded from git.

The first real-data example uses `ec013.33/ec013.544`, a small HC-3 Mwheel session. It downloads `ec013.544.xml` and `ec013.544.eeg` from the Buzsaki Lab mirror, loads adjacent channels 38 and 39 from shank/group 5, and builds a differential LFP trace as `channel_39 - channel_38`.

## Dataset

- CRCNS HC-3: https://crcns.org/data-sets/hc/hc-3/about-hc-3
- Public HC-3 file area: https://crcns.org/files/data/hc3/
- NERSC mirror pattern: https://portal.nersc.gov/project/crcns/download/hc-3/

CRCNS credentials may be required for session data. Do not commit credentials.

## Environment

Recommended local workflow:

```powershell
uv venv
uv pip install -e ".[notebook]"
```

On AMD Windows, install the AMD PyTorch build before running training. Do not let a plain dependency sync replace it with a CPU-only wheel. This repo intentionally keeps generic `torch` out of the base dependencies until the local AMD package/wheel path is confirmed.

## Current Scope

- Notebook-first exploration.
- Theta band: 6-10 Hz.
- Model input: causal phase only.
- Pre-transformer processing: 50-100 Hz where practical.
- Final transformer input: configurable aggressive decimation, initially testing 20, 25, and 50 Hz.
- Training target: nominal phase corresponding to the next acausal 180 degree target phase.

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the full design.

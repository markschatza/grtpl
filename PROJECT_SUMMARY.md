# Real-Time Theta Phase-Locked Stimulation Prediction

## Goal

Build a transformer-based model that predicts the timing of phase-locked stimulation from hippocampal LFP data in the theta band. The practical output of the model is a future stimulation timestamp targeting the next 180 degree phase crossing of the theta-band signal.

The intended use case is real-time prediction, so every online feature must be computed causally from past and present samples only. Acausal processing is allowed only for offline target generation.

## Initial Dataset Plan

Use CRCNS HC-3 hippocampal LFP recordings with multiple adjacent channels available per recording.

For each selected recording:

1. Download LFP data for a few adjacent hippocampal channels.
2. Select adjacent channel pairs.
3. Subtract one channel from another to form a local differential LFP time series.
4. Treat each differential trace as one training/evaluation series.

Resolved starting dataset details:

- Dataset: HC-3, hippocampus recordings from behaving rats.
- Source: https://crcns.org/data-sets/hc/hc-3/about-hc-3
- Public documentation files are available at `https://crcns.org/files/data/hc3/`.
- Full LFP downloads are large and should be pulled session-by-session.
- CRCNS credentials may be required for bulk/session data from the NERSC mirror.

Open dataset details to resolve after metadata inspection:

- Exact first sessions and rats.
- File format details for selected LFP archives.
- Sampling rate for selected LFP recordings.
- Anatomical channel ordering and adjacency definition from HC-3 channel order metadata.

Initial real-data subset:

- Top directory: `ec013.33`
- Session: `ec013.544`
- Behavior: `Mwheel`
- Metadata duration: 29.1 seconds
- LFP file: `ec013.544.eeg`
- XML metadata file: `ec013.544.xml`
- LFP sample rate: 1250 Hz
- Channel count: 65
- First adjacent pair: channels 38 and 39 from anatomical group/shank 5
- Differential trace: `channel_39 - channel_38`

Expanded transformer subset:

- Top directory: `ec013.33`
- Sessions: `ec013.541`, `ec013.543`, `ec013.544`
- Behavior: `Mwheel`
- Rationale: same animal/topdir and same `ec013.540_561` channel-order range, so channels 38 and 39 remain a consistent local pair.
- Current 25 Hz target-table size after filtering invalid horizons:
  - `ec013.541`: 3,431 rows
  - `ec013.543`: 2,988 rows
  - `ec013.544`: 728 rows
  - Total: 7,147 rows

## Signal Processing Pipeline

### Differential LFP

For each recording and channel pair:

```text
differential_lfp[t] = channel_a[t] - channel_b[t]
```

The subtraction should reduce common-mode signal and produce a single local LFP trace per adjacent pair.

### Causal Theta Bandpass

Apply a casual IIR bandpass filter in the theta range to the differential LFP.

Initial theta band:

```text
6-10 Hz
```

This should remain configurable because theta definitions can vary by species, brain region, behavioral state, and dataset.

Requirements:

- Use an IIR filter suitable for streaming.
- Preserve and expose filter state for real-time continuation.
- Avoid any forward-backward or future-looking filtering in the causal path.
- Record filter order, band edges, sampling rate, and implementation details in metadata.

### Causal Phase Estimation

Estimate instantaneous phase causally over the full time series.

Candidate approaches:

- Causal quadrature filter / Hilbert approximation.
- Causal analytic signal approximation from streaming filter outputs.
- Explicit oscillator/state-space phase estimator if needed later.

The phase estimate at time `t` must depend only on samples `<= t`.

Output:

```text
causal_phase[t] in degrees or radians
```

Use one canonical internal representation, probably radians, and convert to degrees only for reporting and loss interpretation.

### Decimation

Decimate the causal phase estimates to a rate appropriate for the theta-band signal and model.

Everything before the final transformer input should run at 50-100 Hz where practical. The final transformer input should be decimated as aggressively as possible while preserving enough timing information to predict the next 180 degree target.

For a 6-10 Hz theta band, the strict Nyquist rate is 20 Hz. The initial implementation should make the final transformer input rate configurable, starting with candidates such as 20, 25, and 50 Hz.

Output:

```text
phase_input[k] = decimated causal phase estimate
timestamp_input[k] = source timestamp for phase_input[k]
```

## Offline Target Generation

Generate training targets using an acausal reference signal.

For each differential LFP trace:

1. Apply an acausal theta-band filter to the full signal.
2. Estimate the reference instantaneous phase.
3. Identify future 180 degree target phase crossings.
4. For every decimated causal input sample `k`, find the next future 180 degree target in the acausal reference.

Output per input sample:

```text
target_timestamp[k] = timestamp of next future 180 degree reference phase crossing
```

## Nominal Signal Phase Target

For each decimated causal phase sample, play out a nominal theta signal from that point and find the nominal phase corresponding to the next offline 180 degree target timestamp.

Conceptually:

```text
current_time = timestamp_input[k]
target_time = target_timestamp[k]
delta_t = target_time - current_time
nominal_frequency = estimated or configured theta frequency
target_nominal_phase[k] = causal_phase[k] + 2*pi*nominal_frequency*delta_t
```

Wrap the target phase to a circular range:

```text
[-pi, pi) or [0, 2*pi)
```

This nominal future phase is the transformer's supervised target.

Open target-design details:

- Whether nominal frequency is fixed, session-level, window-level, or estimated online.
- Whether the target should be absolute nominal phase, phase delta, or time-to-stimulation.
- Whether samples with no future 180 degree crossing inside a maximum horizon should be dropped or masked.

## Transformer Model

The transformer consumes a sequence of decimated causal phase estimates and predicts the nominal signal phase associated with the next desired 180 degree stimulation target.

Input example:

```text
[phase_input[k - context + 1], ..., phase_input[k]]
```

The first baseline should use causal phase only. Treat phase as already tokenized over 0-360 degrees.

For neural-network stability, the implementation can still encode the phase token as `sin(phase)` and `cos(phase)` internally while preserving the conceptual 0-360 phase-token interface.

Model output options:

1. Predict target nominal phase as sin/cos.
2. Predict phase delta as sin/cos.
3. Predict time-to-target directly.

The user-specified primary target is nominal signal phase.

## Loss

The loss should reflect circular phase error in degrees.

For predicted phase `pred` and target phase `target`:

```text
phase_error = atan2(sin(pred - target), cos(pred - target))
loss = mean(abs(phase_error_degrees))
```

Training may use a smoother equivalent:

```text
loss = mean(1 - cos(pred - target))
```

Report evaluation metrics in degrees:

- Mean absolute circular error.
- Median absolute circular error.
- 90th percentile absolute circular error.
- Error by recording/session/channel pair.
- Error by theta amplitude and estimated frequency.

## Stimulation Timestamp Reconstruction

At inference time:

1. Compute causal theta phase from streaming LFP.
2. Decimate/update model input sequence.
3. Predict nominal future phase target.
4. Convert predicted nominal phase into a target delay or timestamp using the nominal oscillator.
5. Schedule stimulation at the reconstructed timestamp.

The timestamp conversion must account for:

- Sampling rate.
- Decimation interval.
- Filter/group delay or estimator latency.
- Hardware stimulation latency.
- Any buffering or model inference latency.

## Hardware And Runtime

Development target:

- Windows.
- AMD GPU.
- PyTorch via the `pytorch-amd` package/path requested by the user.

Implementation should isolate device setup so CPU fallback remains available for preprocessing and tests. Avoid normal dependency sync paths that replace the AMD Windows PyTorch build with a CPU wheel.

## Proposed Repository Structure

```text
grtpl/
  PROJECT_SUMMARY.md
  README.md
  pyproject.toml
  data/
    raw/
    interim/
    processed/
  notebooks/
    00_hc3_download_and_phase_targets.ipynb
    01_transformer_phase_model.ipynb
    01_transformer_phase_model_colab.ipynb
  src/
    grtpl/
      data/
      signal/
      targets/
      models/
      training/
      realtime/
      evaluation/
  tests/
```

## First Implementation Milestones

1. Confirm dataset source, licensing, file format, and channel metadata.
2. Add Python project skeleton with AMD-aware PyTorch dependency notes.
3. Implement data downloader for selected hippocampal LFP sessions.
4. Load a small number of adjacent channels and create differential traces.
5. Implement causal IIR theta filtering with saved filter state.
6. Implement offline acausal reference filtering and 180 degree target extraction.
7. Build a dataset object that emits phase-context windows and nominal phase targets.
8. Train a small baseline transformer.
9. Compare against simple oscillator baselines before scaling model size.

## Initial Transformer Baseline

The first transformer notebook uses only causal phase tokens as input:

- Input token vocabulary: 360 phase bins, representing 0-359 degrees.
- Window shape: fixed historical context of decimated causal phase tokens.
- Initial context length: 128 samples at 25 Hz, or about 5.1 seconds of history.
- Model: learned token embedding, learned positional embedding, causal Transformer encoder, final-state prediction head.
- Output: 2D unit vector `(cos(target_phase), sin(target_phase))`.
- Training loss: `mean(1 - cos(predicted_phase - target_phase))`.
- Evaluation metric: absolute circular phase error in degrees.

The notebook keeps PyTorch installation separate from the project dependencies so the AMD Windows PyTorch build is not replaced by a generic CPU wheel.

The transformer dataset is built per session and concatenated after target generation. The windowed dataset prevents historical contexts from crossing recording/session boundaries.

The Colab transformer notebook is a copy adapted for quick cloud smoke tests. It clones the public repo into `/content/grtpl`, installs the project without installing PyTorch, uses Colab's existing CUDA PyTorch runtime, and defaults to a smaller 64-token context with 200 training steps.

## Key Open Questions

1. Which HC-3 sessions should be the first small working subset?
2. Which channel pairs should define "adjacent" after we inspect the channel order files?
3. Should the nominal oscillator frequency be fixed, estimated per session, or estimated causally over time?
4. What maximum prediction horizon should be allowed for the next 180 degree target?
5. Should stimulation target the acausal filtered signal's 180 degree phase exactly, or should known system latency be included from the beginning?
6. What is the minimum acceptable phase error for the first useful model?

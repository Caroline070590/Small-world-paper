# Small-world QTN/GAF/MTF analysis code

This repository contains the essential code used to preprocess electrophysiological/physiological time series, construct QTN/QG, GAF, and MTF representations, and extract small-world network metrics.

The repository intentionally excludes Jupyter notebooks, plotting scripts, temporary outputs, raw data, and generated figures.

## Structure

```text
src/smallworld_qtn/
  representations.py          # QTN/QG, GAF, MTF transformations
  network_metrics.py          # graph thresholding + small-world metrics
  pipeline.py                 # cleaned signals -> representation metrics
  preprocessing/
    common.py                 # shared signal-cleaning utilities
    emg_plantar.py            # PhysioNet plantar EMG preprocessing
    ecg_physionet.py          # NSRDB + Fantasia ECG preprocessing
    cap_sleep.py              # CAP sleep EEG/EMG/respiration preprocessing
    respiratory_aeration.py   # respiratory aeration dataset preprocessing
    fieldtrip_meg.py          # FieldTrip Subject01 CTF-MEG preprocessing
scripts/
  run_emg_plantar.py
  run_ecg_physionet.py
  run_cap_sleep.py
  run_resp_aeration.py
  run_fieldtrip_meg.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Running examples

```bash
python scripts/run_emg_plantar.py
python scripts/run_ecg_physionet.py
python scripts/run_cap_sleep.py
python scripts/run_resp_aeration.py
python scripts/run_fieldtrip_meg.py
```

Each script writes CSV outputs with one row per file/subject and separate outputs for QTN, GAF, and MTF.

## Notes for publication release

- The code is organized into reusable modules rather than notebooks.
- Dataset-specific preprocessing is separated from representation construction and graph metrics.
- No raw data, intermediate files, or figures are included.
- The modules download public datasets when required by the original pipeline.


## Code audit

See `docs/CODE_AUDIT.md` for the cleanup notes and issues fixed from the working scripts.

#!/usr/bin/env python3
"""Run the PhysioNet plantar surface-EMG preprocessing + small-world pipeline.

This launches the verbatim pipeline in
``smallworld_qtn.preprocessing.emg_plantar``. Edit the CONFIG block at the top
of that module (BASE_URL, OUT_DIR, filtering and QC parameters) before running.

Usage:
    python scripts/run_emg_plantar.py

The pipeline downloads files from the PhysioNet plantar/1.0.0 dataset, applies
EMG band-pass (20-450 Hz) + 50 Hz notch + epoch QC, and writes per-file and
per-subject QTN/GAF/MTF metric CSVs into the configured output directory.
"""

import smallworld_qtn.preprocessing.emg_plantar as emg

if __name__ == "__main__":
    # The module defines run_emg(...); "run" executes processing.
    # fs is the EMG sampling rate (Hz) used by the published run.
    paths = emg.run_emg("run", fs=1000.0)
    print(paths)

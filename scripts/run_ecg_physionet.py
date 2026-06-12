#!/usr/bin/env python3
"""Run the PhysioNet ECG preprocessing + small-world pipeline (NSRDB + Fantasia).

Launches the verbatim pipeline in
``smallworld_qtn.preprocessing.ecg_physionet``. Edit the DATASETS / CONFIG block
at the top of that module (PhysioNet directories, output dirs, filtering and QC
parameters) before running.

Usage:
    python scripts/run_ecg_physionet.py

ECG signals are band-passed 0.5-40 Hz with a 50 Hz notch, resampled, epoch-QC'd,
and transformed into QTN/GAF/MTF networks; size-robust metrics are written to CSV.
"""

import runpy

if __name__ == "__main__":
    # The module guards its driver with `if __name__ == "__main__"`, so we run
    # it as a script to trigger processing of all configured datasets.
    runpy.run_module(
        "smallworld_qtn.preprocessing.ecg_physionet",
        run_name="__main__",
    )

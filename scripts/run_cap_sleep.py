#!/usr/bin/env python3
"""Run the CAP Sleep Database preprocessing pipeline (EEG / EMG / respiration).

Launches the verbatim pipeline in
``smallworld_qtn.preprocessing.cap_sleep``. Edit the CONFIG block at the top of
that module (subject list, modalities, sleep-stage selection, per-modality
filtering) before running.

Usage:
    python scripts/run_cap_sleep.py

Per-modality band-pass (EEG 0.5-40 Hz, EMG 10-45 Hz, RESP 0.01-1 Hz) and 50 Hz
notch are applied; each modality is transformed into QTN/GAF/MTF networks and
size-robust metrics are written to CSV. This is the matched-subject multimodal
dataset used for the controlled cross-modal comparison.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module(
        "smallworld_qtn.preprocessing.cap_sleep",
        run_name="__main__",
    )

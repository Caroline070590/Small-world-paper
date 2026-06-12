#!/usr/bin/env python3
"""Run the ds000117 (FieldTrip/OpenNeuro) MEG preprocessing pipeline.

Launches the verbatim pipeline in
``smallworld_qtn.preprocessing.fieldtrip_meg``. This pipeline relies on DataLad
/ git-annex to fetch the raw CTF-MEG recordings; see the conda-environment setup
cell (commented out) inside the module. Edit the CONFIG/paths block before
running.

Usage:
    python scripts/run_fieldtrip_meg.py

The first 60 s of each recording are band-passed 0.5-40 Hz (4th-order
Butterworth), notch-filtered at 50/100 Hz, resampled to 250 Hz, quality-checked,
and transformed into QTN/GAF/MTF networks; size-robust metrics are written to CSV.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module(
        "smallworld_qtn.preprocessing.fieldtrip_meg",
        run_name="__main__",
    )

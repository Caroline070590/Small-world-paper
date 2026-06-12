#!/usr/bin/env python3
"""Run the respiratory aeration dataset preprocessing pipeline.

Launches the verbatim pipeline in
``smallworld_qtn.preprocessing.respiratory_aeration``. Edit the CONFIG block at
the top of that module (dataset URL/paths, QC thresholds, filtering) before
running.

Usage:
    python scripts/run_resp_aeration.py

A conservative QC strategy for processed respiratory signals is applied (no
envelope extraction or extra smoothing), preserving fine-grained temporal
fluctuations; QTN/GAF/MTF size-robust metrics are written to CSV.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module(
        "smallworld_qtn.preprocessing.respiratory_aeration",
        run_name="__main__",
    )

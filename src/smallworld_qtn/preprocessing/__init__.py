"""Per-dataset preprocessing pipelines.

Each module preserves, verbatim, the download / cleaning / quality-control and
metric-export logic of the corresponding notebook used to produce the published
results:

* :mod:`smallworld_qtn.preprocessing.common`               -- shared utilities.
* :mod:`smallworld_qtn.preprocessing.emg_plantar`          -- PhysioNet plantar EMG.
* :mod:`smallworld_qtn.preprocessing.ecg_physionet`        -- NSRDB + Fantasia ECG.
* :mod:`smallworld_qtn.preprocessing.cap_sleep`            -- CAP sleep EEG/EMG/resp.
* :mod:`smallworld_qtn.preprocessing.respiratory_aeration` -- respiratory aeration.
* :mod:`smallworld_qtn.preprocessing.fieldtrip_meg`        -- ds000117 MEG.

These modules each carry their own CONFIG block and are designed to be run as
scripts (see the top-level ``scripts/`` directory). ``common`` is the only one
intended to be imported as a normal library module.
"""

from . import common

__all__ = ["common"]

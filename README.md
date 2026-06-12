# Small-World Organization from Time-Series Representations

Code accompanying the paper:

> **Small-World Organization as a Representation-Dependent Signature of Nonlinear
> Dynamics in Biological Time Series.**
> Preprint: [DOI 10.21203/rs.3.rs-9534233/v1](https://doi.org/10.21203/rs.3.rs-9534233/v1)
> (Research Square, 2026 — under review, Nature Portfolio)

The project studies how three time-series-to-matrix mappings — **Quantile Graphs
(QG / QTN)**, **Gramian Angular Fields (GAF)**, and **Markov Transition Fields
(MTF)** — shape the small-world topology inferred directly from individual time
series, across synthetic signals and a multimodal collection of biological
datasets (fMRI, MEG, calcium imaging, simulated MEA, ECG, EEG, EMG, respiration).

> **Status:** the associated manuscript is under review. This repository is
> organised so the published results can be reproduced; the analysis code is
> preserved verbatim from the original notebooks (see *Provenance* below).

---

## Repository layout

```
smallworld-qtn/
├── src/smallworld_qtn/
│   ├── representations.py          # QG/QTN, GAF, MTF transforms (+ Q≈2·T^(1/3))
│   ├── network_metrics.py          # thresholding + size-robust small-world metrics
│   ├── pipeline.py                 # cleaned signal → representation → metrics
│   └── preprocessing/
│       ├── common.py               # shared signal-cleaning utilities
│       ├── emg_plantar.py          # PhysioNet plantar EMG
│       ├── ecg_physionet.py        # NSRDB + Fantasia ECG
│       ├── cap_sleep.py            # CAP sleep EEG/EMG/respiration
│       ├── respiratory_aeration.py # respiratory aeration dataset
│       └── fieldtrip_meg.py        # ds000117 (OpenNeuro) MEG
├── scripts/                        # thin launchers for each preprocessing pipeline
│   ├── run_emg_plantar.py
│   ├── run_ecg_physionet.py
│   ├── run_cap_sleep.py
│   ├── run_resp_aeration.py
│   └── run_fieldtrip_meg.py
├── analysis/                       # figure- and table-generating scripts
│   ├── synthetic_generators_and_robustness.py   # Fig. 2, Fig. 3
│   ├── logistic_map_regimes.py                   # Fig. 4, Fig. S1
│   ├── fmri_basc122_pipeline.py                  # canonical fMRI processor
│   ├── fmri_download_basc122_export.py           # COBRE/ADHD download + BASC export
│   ├── abide_surrogate_computation.py            # real vs surrogate metrics (ABIDE/TD)
│   ├── abide_surrogate_validation.py             # Fig. 8
│   ├── statistical_tests.py                      # Fig. 7, Figs. S4–S5
│   ├── biological_heatmaps_and_boxplots.py       # Fig. 5, Fig. 6
│   ├── basc122_atlas_surface.py                  # Fig. S3 (atlas surface)
│   ├── representation_similarity.py              # supporting analysis
│   ├── circular_agreement.py                     # supporting figure
│   ├── merged_barplots.py                        # supporting figure
│   └── summary_all_plots.py                      # combined summaries
├── code_py/                        # full verbatim .py version of every original notebook
└── tests/                          # minimal sanity tests for the core library
```

The `code_py/`  folder holds a faithful, plain-.py version of every original
analysis notebook (code reproduced verbatim, markdown kept as comments; all files
were checked to compile and to match their source line-by-line). See
code_py/README.md for the script index and notes.

`src/` is the installable library (clean, documented, tested). `analysis/` and
the per-dataset preprocessing modules are the **exact** code used to produce the
results, each carrying its own configuration block at the top.

---

## Installation

```bash
git clone <your-repo-url>
cd smallworld-qtn
python -m pip install -e .            # core library only
# or, for the figures/statistics scripts as well:
python -m pip install -e ".[analysis]"
```

Dataset-specific extras can be installed à la carte, e.g. `pip install -e ".[ecg]"`
for the ECG pipeline (`wfdb`), `".[meg]"` for MEG/sleep (`mne`), `".[fmri]"` for
the atlas/surface figures and S3 download. See `requirements.txt` for the full
list and which pipeline each dependency belongs to.

Tested with Python ≥ 3.9.

---

## Quick start (core API)

```python
import numpy as np
from smallworld_qtn import representations as rep, pipeline

# a 1-D signal
x = your_signal  # np.ndarray

# number of quantile bins from the signal length (Eq. 1): Q ≈ 2·T^(1/3)
Q = rep.compute_Q_from_T(x.size)

# full size-robust metric set for one representation
metrics = pipeline.metrics_for_signal_lengthQ(x, "QTN")   # or "GAF", "MTF"
print(metrics["sigma_small_world"], metrics["transitivity"])
```

For a `time × channels` matrix (one feature vector per recording, as in the
paper):

```python
metrics = pipeline.metrics_for_signal_matrix(X, method="MTF", flavour="fmri")
```

---

## Network metrics

`network_metrics.compute_metrics_size_robust` returns the full set used in the
paper, including:

- `sigma_small_world` — small-world index σ = (C/C_rand)/(L/L_rand)  (Eq. 8)
- `gamma_C_over_Crand`, `lambda_L_over_Lrand` — normalized γ, λ  (Eq. 9)
- `transitivity` — clustering term C  (Eq. 10)
- `char_path_len_gcc` — characteristic path length on the giant component (Eq. 11)
- `global_efficiency` — global efficiency E  (Eq. 12)
- `omega`, `phi` — alternative small-world variants
- plus density, average degree, assortativity, weighted clustering, z-scores,
  normalized-Laplacian spectral quantities, and von Neumann entropy.

Networks are thresholded to a fixed density and compared against
degree-preserving (double-edge-swap) random null models; ω/φ additionally use a
ring-lattice reference.

---

## Datasets

Only **control / healthy** subjects are used throughout. Public sources
(see the paper's *Data Availability* for full details):

| Dataset                 | Source |
|-------------------------|--------|
| CAP Sleep (EEG/EMG/Resp)| PhysioNet `capslpdb/1.0.0` |
| Plantar EMG (gait)      | PhysioNet `plantar/1.0.0` |
| ECG — Fantasia          | PhysioNet `fantasia/1.0.0` |
| ECG — NSRDB             | PhysioNet `nsrdb/1.0.0` |
| Respiratory aeration    | PhysioNet `respiratory-heartrate-dataset/1.0.0` |
| MEG                     | OpenNeuro `ds000117` (multisubject, multimodal face processing) |
| fMRI                    | ABIDE I, COBRE, ADHD-200 (parcellated with the BASC-122 atlas) |
| Simulated MEA           | MEArec / SpikeSense generator (see below) |
| Calcium imaging         | available from the corresponding authors on reasonable request |

The simulated MEA data-generation code is the SpikeSense pipeline:
`https://github.com/tivenide/SpikeSense/tree/master/data_generation_MEArec`.

The pipelines download the PhysioNet / OpenNeuro data themselves (configure the
paths/URLs in each module's CONFIG block before running). Raw data and generated
CSVs/figures are intentionally git-ignored.

### Note on MEA and calcium-imaging data

This repository does **not** contain raw data for the **simulated MEA** or
**calcium-imaging** datasets, by design:

* **Simulated MEA** is *generated by code*, not stored. Raw recordings are
  produced with the external SpikeSense / MEArec generator (https://github.com/tivenide/SpikeSense/tree/master/data_generation_MEArec);
  re-running it reproduces the data. No raw data needs to be hosted here.
* **Calcium imaging is private data** and is therefore intentionally excluded.
  It cannot be redistributed; it is available from the corresponding authors on
  reasonable request, subject to institutional and ethical guidelines.

In the provided code, the network metrics for these two datasets enter the
downstream summary/statistics steps from **pre-computed metric CSVs**. To
regenerate those CSVs, the same `representations` + `network_metrics` core
applies (the calcium preprocessing is the grid-based AIP scheme described in
Supplementary S2). Raw data and generated CSVs are git-ignored and must not be
committed — this keeps the private calcium data out of version control.

---

## Reproducing the figures

After producing the per-dataset metric CSVs with the preprocessing pipelines
(`scripts/`) and the fMRI processor (`analysis/fmri_basc122_pipeline.py`):

- **Fig. 2 / Fig. 3** — `analysis/synthetic_generators_and_robustness.py`
- **Fig. 4 / Fig. S1** — `analysis/logistic_map_regimes.py`
- **Fig. 5 / Fig. 6** — `analysis/biological_heatmaps_and_boxplots.py`
- **Fig. 7 / S4 / S5** — `analysis/statistical_tests.py`
- **Fig. 8** — `analysis/abide_surrogate_computation.py` → `abide_surrogate_validation.py`
- **Fig. S3** — `analysis/basc122_atlas_surface.py`

Each script has a paths/config block near the top; set it to point at your CSV
locations before running.

---

## Provenance

The modules under `analysis/` and `src/smallworld_qtn/preprocessing/` were
extracted **verbatim** from the original Jupyter notebooks (kept in
`notebooks/`, with cell outputs stripped). Notebook cell boundaries are marked
with `# %% ---- next notebook cell ----`. The clean core library
(`representations.py`, `network_metrics.py`, `pipeline.py`, `preprocessing/common.py`)
was factored out of the canonical `SMALL-world-FINAL` notebook and exercised
with an end-to-end smoke test, without changing any numerical definitions.

---

## License

This repository (source code, documentation, and figures) is licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)** — see
[`LICENSE`](LICENSE). This matches the license of the associated preprint.

You may share and adapt the material for any purpose, including commercially,
**provided you give appropriate credit** by citing the work (see *Citation*
below and [`CITATION.cff`](CITATION.cff)). Attribution is the only condition.

Full license text: https://creativecommons.org/licenses/by/4.0/

---

## Citation

If you use this code, please cite the associated preprint:

> Caroline Alves, Camilla Bellone, Jan Hoelter, Christiane Thielemann,
> Loriz Francisco Sallum, Raphael Silva do Rosário, Thaise G. L. de O. Toutain,
> AmirAli Kalbasi, and Andriana S. L. O. Campanharo.
> *Small-World Organization as a Representation-Dependent Signature of Nonlinear
> Dynamics in Biological Time Series.* Research Square (preprint), 2026.
> DOI: [10.21203/rs.3.rs-9534233/v1](https://doi.org/10.21203/rs.3.rs-9534233/v1)

(Preprint — under review, Nature Portfolio. Update the citation with the journal,
year, volume, and DOI once the peer-reviewed version is published.)

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

---

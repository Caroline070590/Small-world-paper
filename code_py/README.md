# `notebooks_py/` — verbatim Python conversions of every notebook

This folder contains a **one-to-one, faithful `.py` conversion of every Jupyter
notebook** in `notebooks/`. Code cells are reproduced verbatim (no logic
changed); markdown cells are kept as comments; Jupyter shell-magics are
commented out only where they would otherwise be invalid Python (preserved for
reference with a `# [magic]` prefix). Every file compiles as valid Python and was
line-checked against its source notebook.

These are the literal scripts behind the analysis. For a clean, importable,
de-duplicated library, use `src/smallworld_qtn/` instead; for curated
single-purpose scripts, see `analysis/` and `scripts/`. This folder exists so
reviewers can map each result directly back to the exact original code.

## Mapping (original notebook → converted file)

| Original notebook | Converted file | Purpose |
|---|---|---|
| `all-funtions.ipynb` | `01_synthetic_generators_and_robustness.py` | Synthetic generators; σ-vs-Q robustness (Fig. 2, 3) |
| `logistic-maps.ipynb` | `02_logistic_map_regimes.py` | Logistic map across regimes (Fig. 4, S1) |
| `SMALL-world-FINAL.ipynb` | `03_fmri_basc122_pipeline.py` | Canonical size-robust fMRI processor |
| `Preprocessing-ABIDE-final.ipynb` | `04_fmri_download_basc122_export.py` | COBRE/ADHD download + BASC-122 export |
| `map.ipynb` | `05_basc122_atlas_surface.py` | BASC-122 atlas surface (Fig. S3) |
| `EMG.ipynb` | `10_emg_plantar_pipeline.py` | PhysioNet plantar EMG pipeline |
| `ecg-2-data-FINAL.ipynb` | `11_ecg_physionet_pipeline.py` | NSRDB + Fantasia ECG pipeline |
| `ECG-3-data.ipynb` | `11b_ecg_three_datasets.py` | ECG processing (variant) |
| `ECG-sinus.ipynb` | `11c_ecg_nsrdb_sinus.py` | ECG NSRDB (variant) |
| `sleep-data-CAPS.ipynb` | `12_cap_sleep_pipeline.py` | CAP sleep EEG/EMG/resp pipeline |
| `Respiratory-aeration-dataset.ipynb` | `13_respiratory_aeration_pipeline.py` | Respiratory aeration pipeline |
| `ds000117-final.ipynb` | `14_fieldtrip_meg_pipeline.py` | ds000117 MEG pipeline (final) |
| `ds000117.ipynb` | `14b_ds000117_meg.py` | ds000117 MEG (earlier) |
| `MEG-DS-FINAL.ipynb` | `14c_meg_ds_final.py` | MEG processing (variant) |
| `MEG.ipynb` | `14d_meg.py` | MEG processing (variant) |
| `Untitled1.ipynb` | `20_abide_surrogate_computation.py` | ABIDE/TD real-vs-surrogate metrics |
| `heatmap-surrogate analysis.ipynb` | `21_abide_surrogate_validation.py` | Surrogate validation plots (Fig. 8) |
| `Statistical-test-FINAL.ipynb` | `30_statistical_tests.py` | Wilcoxon / effect sizes / Friedman (Fig. 7, S4–S5) |
| `Statitical-test.ipynb` | `30b_statistical_tests_alt.py` | Statistical tests (variant) |
| `Statitical-test-final-heatmaps.ipynb` | `30c_statistical_heatmaps.py` | Significance heatmaps (variant) |
| `BOXPLOTS-ALL-FINAL.ipynb` | `31_biological_heatmaps_and_boxplots.py` | Heatmaps + boxplots (Fig. 5, 6) |
| `Boxplot-final.ipynb` | `31b_boxplot_final.py` | Boxplots (variant) |
| `Boxplot-ecg.ipynb` | `31c_boxplot_ecg.py` | ECG boxplots |
| `Boxplot-emg.ipynb` | `31d_boxplot_emg.py` | EMG boxplots |
| `boxplot-sleep.ipynb` | `31e_boxplot_sleep.py` | Sleep boxplots |
| `boxplot-aeration.ipynb` | `31f_boxplot_aeration.py` | Aeration boxplots |
| `merged-barplots.ipynb` | `32_merged_barplots.py` | Merged barplots (supporting) |
| `circular-agreament.ipynb` | `33_circular_agreement.py` | Circular agreement figure (supporting) |
| `similarity .ipynb` | `34_representation_similarity.py` | Representation similarity (supporting) |
| `Summary-all-plots.ipynb` | `35_summary_all_plots.py` | Combined summary plots |
| `Untitled.ipynb` | `36_paired_violin_sigma.py` | Paired violin σ (QTN/GAF/MTF) |

## Notes for reviewers

- Several pipelines exist in multiple variants (e.g. ECG `11/11b/11c`, MEG
  `14/14b/14c/14d`). The `…-FINAL` / `14_` files are the versions used for the
  reported results; the others are kept for completeness and transparency.
- The MEG environment-setup cell (`14_fieldtrip_meg_pipeline.py`) contains
  commented-out conda/datalad shell commands; these set up `datalad` + `git-annex`
  to fetch the raw recordings and are not part of the numerical analysis.

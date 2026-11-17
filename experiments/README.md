# Experiment Records

This directory stores small, tracked reproducibility manifests for the major
VOC-ALS experiments.

Generated run artifacts, participant-level predictions, and other large or
sensitive outputs are stored locally under ignored directories such as:

- `experiments/runs/`
- `results/generated/`

Tracked manifests record the methodology, fixed settings, and important
aggregate results needed to reproduce each experiment without committing
participant-level data.

Every record here is rewritten by `python src/run_all.py`.

## Records

| Record | Written by | Contents |
|---|---|---|
| `core_baseline.json` | `baseline.py` | Majority baseline and the full eight-task baseline with configurations frozen per outer fold |
| `exhaustive_task_search.json` | `discovery.py` | All 255 task subsets, tolerance counts, Pareto frontier (discovery only) |
| `bootstrap_task_comparisons.json` | `discovery.py` | Paired bootstrap of TA, E+TA+KA, and the full set |
| `nested_task_selection.json` | `nested_selection.py` | Leakage-safe nested task selection, the primary handcrafted result |
| `delta_sensitivity.json` | `nested_selection.py` | Nested selection at delta 0.01, 0.02, 0.05 |
| `protocol_stability.json` | `protocol_stability.py` | 50,000 participant-resampled inner selections |
| `protocol_robustness.json`, `protocol_robustness_key_candidates.csv` | `protocol_stability.py` | Fixed candidate protocols scored against the same resamples |
| `raw_audio_audit.json` | `audit_raw_audio.py` | WAV inventory, signal checks, exact-duplicate groups |
| `duplicate_exclusion_sensitivity.json` | `duplicate_sensitivity.py` | Nested selection without the two duplicate-connected participants |
| `duplicate_policy_sensitivity.json` | `duplicate_sensitivity.py` | Three alternative policies for the duplicated recording |
| `raw_audio_wavlm_embeddings.json` | `wavlm_embeddings.py` | WavLM model, revision, pooling, and extraction device |
| `wavlm_task_selection.json` | `wavlm_selection.py` | Primary WavLM nested selection and noninferiority bootstrap |
| `wavlm_pca_sensitivity.json` | `wavlm_selection.py` | 8, 16, and 32 PCA components per task |
| `wavlm_stability_aware_selection.json` | `wavlm_selection.py` | Exploratory rule ranking by worst inner-fold score (reported in the paper) |
| `final_stability_aware_wavlm.json` | `wavlm_selection.py` | Second exploratory rule ranking by worst inner-fold gap to the full set (not in the paper) |
| `recording_duration_burden.json` | `recording_burden.py` | Recording count and duration of the WavLM-selected protocols |
| `wavlm_vs_handcrafted_comparison.csv` | `paper_numbers.py` | Duplicate-filtered same-cohort comparison |

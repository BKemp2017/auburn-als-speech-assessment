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

## Current Experiments

- `core_baseline.json`
  - Majority-class sanity baseline
  - Full eight-task nested-CV baseline
  - Frozen model configurations selected within each outer training split

- `exhaustive_task_search.json`
  - Exhaustive evaluation of all 255 nonempty speech-task subsets
  - Macro F1 performance-versus-burden analysis
  - Prespecified tolerance analysis
  - Pareto frontier and task-frequency discovery results

- `bootstrap_task_comparisons.json`
  - 10,000 fold-stratified paired participant bootstrap resamples
  - 95% percentile confidence intervals
  - Reduced-versus-full and three-task-versus-one-task comparisons

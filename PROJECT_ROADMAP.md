# Research Project Roadmap and Checklist

## Scope Lock

- [x] Preserve the authoritative proposal unchanged.
- [x] Preserve the full project title and human-centered EyeSpeak-Assist motivation.
- [x] Define the project as assistive remote monitoring rather than ALS diagnosis.
- [x] Make the one-semester exhaustive task-subset study the required core.
- [x] Obtain explicit approval before changing the proposal's title, motivation, research questions, methodology, clinical boundaries, or scope.
- [x] Obtain explicit approval before downloading VOC-ALS, incurring paid API costs, or beginning implementation.

## Required Core Deliverables

- [x] Cleaned participant-level feature dataset using the 40 published VOC-ALS acoustic predictors.
- [x] Reproducible participant-level feature-table construction and fold-safe preprocessing pipeline.
- [x] Participant-disjoint nested cross-validation with all recordings/features from a participant kept in the same fold.
- [x] Majority-class sanity baseline and complete eight-task reference assessment.
- [x] Fixed-model exhaustive evaluation of all 255 nonempty task subsets.
- [x] Primary binary analysis: documented speech impairment versus normal speech among ALS participants.
- [x] Performance–burden Pareto frontier using task count as the primary burden proxy.
- [x] Prespecified minimal-protocol analysis using Macro F1 tolerance delta = 0.02, with 0.01 and 0.05 sensitivity analyses.
- [x] Task-frequency discovery analysis among high-performing protocols.
- [x] Participant-level bootstrap confidence intervals and paired reduced-versus-full comparisons.
- [ ] Protocol-stability analysis.
- [X] Leakage-safe nested task-selection validation of the reduced-protocol selection procedure.
- [ ] Final research paper.
- [ ] Auburn class presentation.
- [ ] Final reproducibility freeze and clean-environment rerun.

## Strong Extensions

- [ ] Rich, reproducible feature extraction from raw VOC-ALS audio.
- [ ] Re-run the minimal-task question with richer raw-audio features to test whether the reduced-protocol conclusion is representation-robust.
- [ ] Secondary ordinal ALSFRS-R speech-score prediction with appropriately limited claims.
- [ ] Calibration analysis.
- [ ] Cautious subgroup and error analysis.
- [ ] Robustness analyses across folds, metrics, feature sets, tolerances, and resamples.
- [ ] Recording-duration burden analysis if raw-audio durations are reliable.

## Optional Stretch Work

- [ ] NSGA-II joint optimization of speech tasks, acoustic features or feature groups, model family, and hyperparameters.
- [ ] Compare NSGA-II with equal-budget random search, conventional tuning, exhaustive task-only search, and ablations.
- [ ] Keep evolutionary selection entirely inside inner training folds.
- [ ] Optional on-device inference prototype.
- [ ] Future integration concepts for an assistive application.

NSGA-II is not required or justified for the eight-task subset search alone because exhaustive enumeration covers all 255 nonempty subsets.

## Semester Roadmap

### Gate 0 — Project approval and scope freeze

- [x] Read and preserve the proposal.
- [x] Create the project structure and roadmap.
- [ ] Confirm the proposal and roadmap with the research advisor or course professor.
- [x] Record the Macro F1 protocol correction in `proposal/` without rewriting the historical proposal.

### Weeks 1–2 — Literature and study-design audit

- [x] Review the VOC-ALS dataset paper, documentation, acquisition protocol, and released task structure.
- [ ] Build a complete literature evidence table covering cohort, target, split unit, features, models, metrics, limitations, and relevance.
- [x] Verify class counts, metadata fields, task names, participant identifiers, released feature structure, and audio availability.
- [x] Finalize the core analysis plan, leakage controls, primary metric, tolerance rule, and participant-disjoint validation design.
- [ ] Resolve any remaining dataset-license/access documentation details needed before raw-audio redistribution or publication.
- [ ] Decide whether Auburn requires an institutional determination for analysis of existing pseudonymized public human-participant data.

### Weeks 3–4 — Data audit and feature pipeline

- [x] Acquire and audit the VOC-ALS metadata/public feature table locally without committing participant-level data.
- [x] Freeze participant-disjoint outer and inner fold assignments.
- [x] Implement deterministic fold-safe standardization inside scikit-learn pipelines.
- [x] Build the 102-participant ALS-only feature table with 40 published acoustic predictors.
- [x] Add an explicit finite-value assertion to modeling-input validation and regression-check the frozen results.
- [ ] Download and validate raw audio when beginning the raw-audio extension.
- [ ] Create the raw-audio participant/task manifest and validate eight expected recordings per participant.
- [ ] Implement and validate the richer raw-audio feature-extraction pipeline.

### Week 5 — Full-protocol baselines

- [x] Implement the majority-class sanity baseline.
- [x] Compare the four prespecified candidate model families and 70 configurations under nested validation.
- [x] Select a model configuration within each outer training split using only inner-fold Macro F1.
- [x] Freeze each outer fold's selected configuration for the subsequent 255-subset comparison.
- [x] Verify participant-disjoint train/validation/test partitions and training-only preprocessing/model selection.

### Weeks 6–7 — Exhaustive minimal-protocol experiment

- [x] Enumerate each of the 255 nonempty task subsets exactly once.
- [x] Evaluate every subset using the same frozen participant folds and the corresponding frozen outer-fold configuration.
- [x] Save 1,275 fold-level subset metric records and 255 aggregate subset summaries locally.
- [x] Use task count as the primary assessment-burden measure.
- [x] Generate paired participant-level out-of-fold predictions for the selected reduced protocols and the complete reference for bootstrap analysis.
- [ ] Add recording duration only as a secondary burden proxy if raw-audio duration proves reliable.

### Week 8 — Statistical and Pareto analysis

- [x] Construct the task-level performance–burden Pareto frontier.
- [x] Apply the delta = 0.02 primary minimality rule.
- [x] Run delta = 0.01 and 0.05 sensitivity analyses.
- [x] Compare all single-task protocols.
- [x] Calculate task frequency among the top-performing discovery subsets.
- [x] Generate the initial single-task, task-frequency, and Pareto figures.
- [x] Calculate participant-level bootstrap confidence intervals.
- [x] Run paired reduced-versus-full bootstrap comparisons.
- [ ] Quantify protocol stability across participant resamples.
- [X] Perform nested task-selection validation to estimate the generalization of the selection procedure without outer-test-driven subset choice.

### Week 9 — Secondary and robustness analyses

- [ ] Run ordinal severity analysis as secondary work, with uncertainty and explicit small-class limitations.
- [ ] Extract richer raw-audio features and re-run the same minimal-task research question.
- [ ] Assess calibration and errors.
- [ ] Run cautious exploratory demographic and onset-type stratification only if counts support interpretation.
- [ ] Test sensitivity to folds, metrics, feature sets, and duration burden.

### Week 10 — Optional evolutionary extension

- [ ] Begin only if the required core is statistically validated and on schedule.
- [ ] Define genotype, genotype-to-phenotype mapping, feasibility rules, repair, and objectives for a genuinely larger joint search space.
- [ ] Run NSGA-II only within inner training folds.
- [ ] Compare its frontier and evaluation efficiency against prespecified baselines.

### Weeks 11–12 — Reporting and reproducibility

- [ ] Complete ablations and error analysis.
- [ ] Freeze final tables, figures, environment, seeds, configurations, and experiment manifests.
- [ ] Write the paper with clinical limitations adjacent to claims.
- [ ] Build the presentation around the research question, participant burden, leakage-safe design, Pareto tradeoffs, uncertainty, and limitations.
- [ ] Re-run the required experiment from a clean environment and verify reproducibility.

## Decision Gates

- [x] **Dataset gate:** The available labels, participant IDs, task structure, and published feature table support the planned ALS-only participant-level core study.
- [x] **Evaluation gate:** Five-outer/four-inner participant-disjoint folds are feasible and have been frozen.
- [x] **Feature-validation gate:** Add explicit finite-value validation and confirm the frozen baseline/subset results are unchanged.
- [ ] **Uncertainty gate:** Paired participant bootstrap and protocol-stability analyses support the reduced-protocol discovery before stronger claims are made.
- [X] **Selection-validation gate:** Nested task selection confirms that reduced-protocol selection generalizes without using outer-test performance to choose the subset.
- [ ] **Core-complete gate:** Core modeling, uncertainty, nested selection validation, paper, presentation, and reproducibility freeze are complete before optional NSGA-II work.
- [ ] **Claim gate:** Final paper/presentation claims remain limited to speech-impairment assessment in the studied ALS cohort, not ALS diagnosis, clinical replacement, or proven real-world fatigue reduction.

## Current Discovery Checkpoint

The exhaustive task-only discovery analysis is complete. The complete eight-task reference achieved mean Macro F1 of approximately 0.563. The descriptive Pareto frontier currently contains:

- `TA` alone: approximately 0.685 mean Macro F1
- `E+TA+KA`: approximately 0.694 mean Macro F1

`TA` appeared in 23 of the top 25 discovery subsets. These values are promising discovery results, but selected-subset performance is not yet an unbiased final estimate because the same outer-fold results were used to identify the strongest subsets.

Paired participant-level bootstrap analysis with 10,000 fold-stratified
resamples produced a 95% interval of 0.028 to 0.223 for the Macro F1
difference between `TA` and the complete eight-task assessment, and
0.052 to 0.215 for `E+TA+KA` versus the complete assessment. The
difference between `E+TA+KA` and `TA` was only 0.009, with a 95%
bootstrap interval of -0.082 to 0.102, providing no clear bootstrap
evidence that the additional two tasks improve performance over `TA`.
These remain post-selection discovery estimates pending nested
task-selection validation.

Leakage-safe nested task selection is now complete. With the locked
delta = 0.02 rule, the selected protocols contained one to three tasks
and achieved mean outer-test Macro F1 of approximately 0.572, compared
with approximately 0.563 for the complete eight-task assessment. The
delta = 0.01 sensitivity analysis reproduced the same selections, while
delta = 0.05 selected an average of 1.2 tasks and achieved mean
outer-test Macro F1 of approximately 0.618. Delta = 0.02 remains the
locked primary analysis.


## Immediate Next Action

Quantify protocol stability across participant resamples using the locked
delta = 0.02 nested-selection rule. Preserve participant-disjoint folds,
reuse paired inner out-of-fold predictions, and report exact-subset,
individual-task, and task-count selection frequencies.

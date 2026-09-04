# Research Project Roadmap and Checklist

## Scope Lock

- [x] Preserve the authoritative proposal unchanged.
- [x] Preserve the full project title and human-centered EyeSpeak-Assist motivation.
- [x] Define the project as assistive remote monitoring rather than ALS diagnosis.
- [x] Make the one-semester exhaustive task-subset study the required core.
- [ ] Obtain explicit approval before changing the proposal's title, motivation, research questions, methodology, clinical boundaries, or scope.
- [ ] Obtain explicit approval before downloading VOC-ALS, incurring paid API costs, or beginning implementation.

## Required Core Deliverables

- [ ] Cleaned participant-level feature dataset.
- [ ] Reproducible audio feature-extraction pipeline.
- [ ] Participant-disjoint nested cross-validation with all recordings from a participant kept in the same fold.
- [ ] Full eight-task baseline and fixed-model exhaustive evaluation of all 255 nonempty task subsets.
- [ ] Primary binary analysis: documented speech impairment versus normal speech among ALS participants.
- [ ] Performance–burden Pareto frontier.
- [ ] Prespecified minimal-protocol analysis using macro F1 tolerance delta = 0.02, with 0.01 and 0.05 sensitivity analyses.
- [ ] Participant-level bootstrap confidence intervals, paired comparisons, and protocol-stability analysis.
- [ ] Final research paper.
- [ ] Auburn class presentation.
- [ ] Reproducible source repository and experiment records.

## Strong Extensions

- [ ] Secondary ordinal ALSFRS-R speech-score prediction with appropriately limited claims.
- [ ] Rich, reproducible feature extraction from raw audio.
- [ ] Calibration analysis.
- [ ] Cautious subgroup and error analysis.
- [ ] Robustness analyses across folds, metrics, features, tolerances, and resamples.

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
- [ ] Record any explicitly approved amendment in `proposal/` without silently rewriting the authoritative proposal.

### Weeks 1–2 — Literature and study-design audit

- [ ] Review the VOC-ALS dataset paper, documentation, license, acquisition protocol, and related dysarthria studies.
- [ ] Build a literature evidence table covering cohort, target, split unit, features, models, metrics, limitations, and relevance.
- [ ] Verify class counts, metadata fields, task names, participant identifiers, missingness, and audio availability from documentation before download.
- [ ] Finalize the analysis plan, leakage controls, primary estimand, metrics, tolerance rule, and contingency for unstable five-fold splits.
- [ ] Decide whether institutional review or formal exemption confirmation is required for the public de-identified dataset.

### Weeks 3–4 — Data audit and feature pipeline

- [ ] After explicit approval, download and checksum the dataset without modifying raw files.
- [ ] Create a participant/task manifest and validate eight expected recordings per participant.
- [ ] Freeze participant-disjoint outer and inner fold assignments.
- [ ] Implement deterministic preprocessing and fold-safe standardization.
- [ ] Extract defensible acoustic features and validate missing or malformed recordings.
- [ ] Produce participant-level feature tables with provenance.

### Week 5 — Full-protocol baselines

- [ ] Implement majority-class and simple linear baselines.
- [ ] Compare the prespecified baseline model families under nested validation.
- [ ] Select and freeze the fixed primary pipeline for exhaustive task analysis using inner training data only.
- [ ] Verify that no participant, preprocessing statistic, feature decision, or hyperparameter leaks across folds.

### Weeks 6–7 — Exhaustive minimal-protocol experiment

- [ ] Enumerate each of the 255 nonempty task subsets exactly once.
- [ ] Evaluate every subset using identical frozen folds and the fixed pipeline.
- [ ] Save fold-level predictions and metrics for paired participant-level analysis.
- [ ] Calculate task-count burden and duration burden only if recording duration is reliable.

### Week 8 — Statistical and Pareto analysis

- [ ] Construct the task-level performance–burden Pareto frontier.
- [ ] Apply the delta = 0.02 primary minimality rule.
- [ ] Run delta = 0.01 and 0.05 sensitivity analyses.
- [ ] Calculate participant-level bootstrap uncertainty and paired reduced-versus-full comparisons.
- [ ] Report task-selection stability rather than relying only on the best point estimate.

### Week 9 — Secondary and robustness analyses

- [ ] Run ordinal severity analysis as secondary work, with uncertainty and explicit small-class limitations.
- [ ] Assess calibration and errors.
- [ ] Run cautious exploratory demographic and onset-type stratification only if counts support interpretation.
- [ ] Test sensitivity to folds, metrics, feature sets, and duration burden.

### Week 10 — Optional evolutionary extension

- [ ] Begin only if the required core is on schedule and stable.
- [ ] Define genotype, genotype-to-phenotype mapping, feasibility rules, repair, and three objectives.
- [ ] Run NSGA-II only within inner training folds.
- [ ] Compare its frontier and evaluation efficiency against all prespecified baselines.

### Weeks 11–12 — Reporting and reproducibility

- [ ] Complete ablations and error analysis.
- [ ] Freeze tables, figures, environment, seeds, configurations, and result manifests.
- [ ] Write the paper with clinical limitations adjacent to claims.
- [ ] Build the presentation around the research question, participant burden, leakage-safe design, Pareto tradeoffs, uncertainty, and limitations.
- [ ] Re-run the required experiment from a clean environment and verify reproducibility.

## Decision Gates

- [ ] **Dataset gate:** License, files, labels, participant IDs, and missingness support the planned participant-level study.
- [ ] **Evaluation gate:** Nested participant-disjoint folds are feasible and stable; otherwise use the proposal's repeated stratified-group or leave-one-participant-out sensitivity plan.
- [ ] **Feature gate:** The deterministic feature pipeline passes validation before model comparison.
- [ ] **Core-complete gate:** All 255 subsets, Pareto analysis, uncertainty, paper, and presentation are deliverable before starting optional NSGA-II work.
- [ ] **Claim gate:** Every result is described as assessment of speech impairment in the studied ALS cohort, not diagnosis, clinical replacement, or proven real-world fatigue reduction.

## Immediate Next Action

Conduct a focused literature-and-dataset documentation review and turn it into a frozen analysis-plan draft. This should verify labels, participant identifiers, task metadata, licensing, prior evaluation protocols, and the exact leakage controls before any dataset download or implementation begins.

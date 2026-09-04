# Analysis Plan Draft

## Authority and change control

This draft operationalizes the authoritative research proposal. It does not replace or amend the proposal. Any material change to the title, motivation, research questions, methodology, primary target, clinical boundaries, or scope requires Blake's explicit approval.

## Primary estimand

Estimate how well a fixed machine-learning pipeline using a selected nonempty subset of the eight public VOC-ALS speech tasks discriminates documented speech impairment (`ALSFRS-R speech < 4`) from normal speech (`ALSFRS-R speech = 4`) among participants already diagnosed with ALS, when generalizing to unseen participants.

## Study population

- Primary analysis: 102 ALS participants.
- Expected binary counts from published documentation: 49 impaired and 53 normal-speech participants.
- Healthy controls are excluded from the primary outcome.
- Optional healthy-control-versus-ALS classification remains a separate exploratory dataset-classification analysis and is not ALS screening validation.

## Candidate task set

`T = {a, e, i, o, u, pa, ta, ka}`

Evaluate all `2^8 - 1 = 255` nonempty subsets for the required task-only search.

## Primary performance and burden quantities

- Primary selection metric: macro F1.
- Primary burden: number of required tasks, `|S|`.
- Primary tolerance: `delta = 0.02` macro F1 below the full eight-task protocol.
- Sensitivity tolerances: `delta = 0.01` and `delta = 0.05`.
- Secondary burden: summed or median recording duration, only if duration proves reliable.
- Supporting binary metrics: balanced accuracy, sensitivity, specificity, ROC AUC, PR AUC, Brier score, calibration curve, and confusion matrix.

The minimal protocol is the smallest task subset meeting the prespecified tolerance relative to the full protocol. Statistical uncertainty and selection stability must be considered before practical equivalence is claimed.

## Leakage controls

- Precompute or freeze participant identifiers before modeling.
- Keep all recordings and task-derived features for one participant in one fold.
- Use outer participant-disjoint folds only for final generalization estimates.
- Perform imputation, scaling, feature selection, model selection, and hyperparameter selection inside the applicable inner training folds.
- Never use pooled outer-test predictions to retune the pipeline or choose a new hypothesis.
- Use the same outer folds for all task subsets to enable paired participant-level comparisons.
- Store outer-fold predictions with participant ID, fold ID, true label, predicted class, and predicted probability.
- Do not include the target, target-derived severity label, participant ID, or clinical variables that trivially encode impairment as predictors.

## Cross-validation draft

Preferred starting design:

- Five stratified outer folds at the participant level.
- Inner stratified participant-level folds for tuning.
- Fixed random seeds and persisted fold assignments.
- Identical outer folds across all 255 task subsets.

Before freezing, verify that each fold contains adequate impaired and normal-speech participants. If five-fold evaluation is unstable, use the proposal-approved repeated stratified group design and retain leave-one-participant-out evaluation only as a sensitivity analysis.

## Fixed-pipeline task search

The exhaustive task comparison must use one prespecified pipeline selected without consulting outer test performance. A simple regularized, class-weighted logistic regression is the leading interpretable fixed-pipeline candidate. Linear/RBF SVM, random forest, and gradient-boosted trees may be compared during baseline development, but the exhaustive 255-subset analysis must not opportunistically select a different model for each subset using outer results.

## Feature plan

Start with two auditable feature tiers:

1. **Published-feature baseline:** mean F0, F0 standard deviation, HNR, local jitter, and local shimmer for each selected task.
2. **Prespecified raw-audio feature pipeline:** duration/voiced duration, energy, spectral summaries valid below 4 kHz, MFCC summaries, pitch statistics, voice-quality measures, and `/pa/`, `/ta/`, `/ka/` rate/regularity where robustly extractable.

Because audio is sampled at 8 kHz, every raw-audio feature must be justified and validated under a 4 kHz Nyquist limit. Rich raw-audio work is a strong extension; it must not delay the required published-feature baseline and exhaustive task search.

## Uncertainty and comparison plan

- Participant-level bootstrap confidence intervals.
- Paired bootstrap of reduced versus full-protocol predictions.
- Task-selection frequency across resamples.
- Pareto frontier over performance and task count.
- Sensitivity to tolerance, metric, fold construction, and feature tier.
- Emphasize uncertainty and stability rather than ranking tiny point-estimate differences.

## Secondary ordinal analysis

Predict ordered ALSFRS-R speech scores 1–4 only after the primary analysis is complete. Report MAE, quadratic weighted kappa, Spearman correlation, per-class recall, and ordinal confusion matrix. Make no reliable severe-class claim from five severe participants.

## Deferred work

- NSGA-II and evolutionary search are deferred unless required deliverables are complete and time remains.
- Smartphone/on-device prototyping is deferred until the core research is complete.
- Deep audio models are not required and are not part of the current core plan.

## Freeze checklist

- [ ] Confirm dataset access conditions and citation requirements.
- [ ] Confirm metadata schema, missingness, participant/task completeness, and severity counts.
- [ ] Confirm Auburn institutional requirements for this secondary-data study.
- [ ] Freeze participant inclusion/exclusion rules.
- [ ] Freeze outer and inner fold-generation procedure.
- [ ] Freeze primary metric, burden measure, tolerance, and equivalence language.
- [ ] Freeze published-feature baseline definitions.
- [ ] Freeze the fixed model pipeline and its inner-fold hyperparameter grid.
- [ ] Define the exact bootstrap unit and number of resamples.
- [ ] Define result filenames, prediction schema, random seeds, and experiment manifest.


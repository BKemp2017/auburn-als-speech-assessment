# Locked Modeling Protocol

## Project

**How Much Speech Is Enough? Multi-Objective Selection of Smartphone Voice Tasks for ALS Dysarthria Assessment**

This protocol defines the analysis decisions that must be fixed before model evaluation. Any later change must be documented with its rationale and treated as exploratory when appropriate.

## Clinical Scope

This study evaluates speech impairment among participants already diagnosed with ALS. It is an assistive remote-monitoring study and is not an ALS diagnostic system.

The models must not be described as diagnosing ALS, replacing clinical evaluation, or establishing clinical efficacy.

## Study Population

The primary analysis includes only the 102 VOC-ALS participants diagnosed with ALS.

Healthy-control participants are excluded from the primary analysis.

## Primary Target

Binary speech status derived from `ALSFRS-R_SpeechSubscore`:

- `0`: normal speech, score equal to 4
- `1`: documented speech impairment, score below 4

Class distribution:

- Normal speech: 53 participants
- Documented speech impairment: 49 participants

## Secondary Target

Ordinal dysarthria-severity prediction may be performed as a secondary exploratory analysis.

The small moderate and severe groups limit statistical power. Results must therefore be interpreted cautiously and must not replace the primary binary analysis.

## Predictor Variables

The required core uses 40 public acoustic features:

- Five features from each of eight speech tasks
- Sustained vowels: A, E, I, O, U
- Syllable-repetition tasks: PA, TA, KA

Features per task:

- Mean fundamental frequency
- Standard deviation of fundamental frequency
- Harmonics-to-noise ratio
- Local jitter
- Local shimmer

Clinical variables, demographic variables, nonpublic recordings, and questionnaire data are excluded from the required core predictors.

## Task-Subset Search

All nonempty subsets of the eight public speech tasks will be evaluated exhaustively:

\[
2^8 - 1 = 255
\]

Evolutionary search is not required for this task-only search because exhaustive enumeration is computationally feasible and guarantees evaluation of every possible task subset.

NSGA-II is reserved as an optional extension involving a larger joint search space.

## Nested Cross-Validation

The fixed participant-disjoint design uses:

- Five outer folds
- Four inner folds
- Stratification by the primary binary target
- Random seed `2026`

The existing fold-assignment files must be reused for every model and task subset.

### Outer Loop

The outer test fold estimates generalization performance.

No information from an outer test fold may influence preprocessing, feature selection, model selection, hyperparameter selection, threshold selection, or any other training decision.

### Inner Loop

The inner folds select the model and hyperparameters using only the corresponding outer-training participants.

All preprocessing must be fitted exclusively on each inner-training partition and then applied to its validation partition.

After selection, the chosen pipeline is refitted on the complete outer-training partition and evaluated once on the untouched outer-test partition.

## Preprocessing

The following operations must occur inside each cross-validation training fold:

1. Verify finite feature values.
2. Standardize each selected acoustic feature using training-fold means and standard deviations.
3. Apply the fitted transformation to validation or test data without refitting.

Preprocessing and classification must be combined in a scikit-learn `Pipeline` to prevent leakage.

## Candidate Models

### Regularized Logistic Regression

Logistic regression is the primary interpretable baseline.

Hyperparameter grid:

- `C`: `0.01`, `0.1`, `1`, `10`, `100`
- `class_weight`: `None`, `"balanced"`
- Maximum iterations: `5000`
- Solver: `"liblinear"`
- Random seed: `2026`

### RBF Support Vector Machine

The RBF SVM provides a nonlinear comparison.

Hyperparameter grid:

- `C`: `0.1`, `1`, `10`, `100`
- `gamma`: `"scale"`, `0.001`, `0.01`, `0.1`, `1`
- `class_weight`: `None`, `"balanced"`
- Probability estimation: disabled

## Primary Selection Metric

Macro F1 is the primary model-selection and reporting metric, matching the approved project proposal and proposal presentation:

\[
F1_{macro}
=
\frac{F1_{normal}+F1_{impaired}}{2}
\]

This gives equal weight to the F1 score of each class rather than reporting only the positive-class F1 score.

## Secondary Metrics

Report the following for outer-test predictions:

- Balanced accuracy
- ROC-AUC
- Sensitivity
- Specificity
- Positive-class F1 score
- Standard accuracy
- Task count

Sensitivity treats documented speech impairment as the positive class.

## Decision Threshold

The required core uses the classifier’s default decision boundary:

- Logistic regression probability threshold: `0.5`
- SVM decision-function threshold: `0`

Threshold optimization is excluded from the required core. If examined later, it must occur exclusively within the inner cross-validation loop and be labeled exploratory.

## Inner-Loop Selection Rule

For each task subset and outer fold:

1. Select the candidate configuration with the greatest mean inner-fold Macro F1.
2. If configurations tie, select the one with the smaller model-complexity setting.
3. If still tied, prefer logistic regression.
4. If still tied, select the first configuration in deterministic grid order.

No outer-test result may be used for model or hyperparameter selection.

## Task-Subset Comparison

For each of the 255 task subsets, aggregate the five outer-test results.

Report:

- Mean and standard deviation of each metric
- Number of tasks
- Included task names
- Performance difference from the complete eight-task assessment

The complete eight-task subset is the reference assessment.

## Minimal-Subset Selection

A subset will be considered competitive when its mean outer-fold Macro F1 is within a prespecified tolerance of the complete eight-task assessment.

Initial tolerance:

\[
\Delta = 0.02
\]

This means a subset is competitive when:

\[
F1_{macro,subset}
\ge
F1_{macro,full} - 0.02
\]

Among competitive subsets, prefer:

1. Fewer tasks
2. Greater mean balanced accuracy
3. Lower between-fold standard deviation
4. Lexicographic task order for deterministic final tie-breaking

The tolerance analysis must also report sensitivity to alternative tolerances of `0.01` and `0.05`, matching the proposal presentation.

## Required Comparisons

The core results must include:

- Complete eight-task assessment
- Best-performing subset for each task count
- Smallest subset within the primary tolerance
- All single-task models
- All vowel-only combinations
- All syllable-repetition-only combinations

## Reproducibility

The analysis must preserve:

- Fixed fold assignments
- Random seed `2026`
- Software environment specification
- Source-data checksum
- Complete task-subset definitions
- Selected hyperparameters for every outer fold
- Outer-test predictions
- Aggregate results
- Execution logs or configuration records

Participant-level fold assignments and raw data must remain excluded from Git.

## Optional Extensions

Only after completing the required core:

- Confidence intervals
- Ordinal severity analysis
- Alternative models
- Additional acoustic features extracted from public audio
- Smartphone recording comparison using the Samsung Galaxy S25 Ultra
- NSGA-II for joint optimization of tasks, acoustic features, model choice, and hyperparameters

NSGA-II must not be presented as necessary for selecting among eight tasks.

## Interpretation Boundaries

Results apply to the VOC-ALS cohort and its recording conditions.

Limitations include:

- Small sample size
- Single-center cohort
- Italian speech tasks
- Existing ALS diagnosis
- Limited numbers in more severe speech-score groups
- Potential device and recording-condition dependence
- Lack of external clinical validation

Results must be framed as evidence about reducing speech-assessment burden while retaining predictive information, not as proof of diagnosis, clinical replacement, or universal deployment readiness.
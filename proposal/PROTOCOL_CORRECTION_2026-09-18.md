# Protocol Correction — 2026-09-18

## Reason

The approved research proposal and proposal presentation specify **Macro F1** as the primary model-selection metric and define the minimal-protocol tolerance as 0.02 Macro F1, with sensitivity analyses at 0.01 and 0.05.

A later modeling-protocol draft unintentionally changed the primary metric to balanced accuracy and changed the upper sensitivity tolerance to 0.03. The first full eight-task baseline implementation followed that later draft.

## Correction

Before any 255-task-subset experiment was run, the project was realigned with the approved proposal:

- Primary model-selection metric: **Macro F1**
- Primary minimal-protocol tolerance: **0.02 Macro F1**
- Sensitivity tolerances: **0.01 and 0.05**
- Balanced accuracy remains a secondary reporting metric.
- Positive-class F1 may be reported separately, but it is not the primary F1 statistic.

## Status of the earlier balanced-accuracy run

The earlier eight-task balanced-accuracy-selected run is treated as a development/diagnostic run only. It is not the official reference result for the project.

The official eight-task reference baseline must be rerun with Macro F1 driving inner-loop model and hyperparameter selection before the exhaustive 255-subset analysis begins.

## Unchanged safeguards

The following remain unchanged:

- 102 ALS participants in the primary analysis
- binary target: ALSFRS-R speech score 4 versus below 4
- five outer folds and four inner folds
- participant-disjoint fixed fold assignments
- random seed 2026
- preprocessing fitted only within training partitions
- no outer-test information used for inner model selection
- all 255 nonempty speech-task subsets evaluated using the same frozen folds

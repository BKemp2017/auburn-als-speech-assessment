# Auburn ML Research — Minimal ALS Speech Assessment

## Authoritative Project Title

**How Much Speech Is Enough? Multi-Objective Selection of Smartphone Voice Tasks for ALS Dysarthria Assessment**

## Project Summary

This Auburn University graduate machine-learning research project asks whether a smaller subset of the eight smartphone speech exercises in VOC-ALS can preserve most of the predictive information of the complete assessment while reducing participant burden. The primary analysis concerns clinically documented speech impairment versus normal speech among participants already diagnosed with ALS. It is an assistive remote-monitoring study, not an ALS diagnostic or clinical-replacement system.

The required core uses the 102 VOC-ALS participants with ALS and the 40 published acoustic predictors: mean F0, F0 standard deviation, HNR, local jitter, and local shimmer from each of five sustained vowels and three repeated syllables. Participants remain disjoint across cross-validation folds. Macro F1 is the primary metric.

The full eight-task baseline and all 255 nonempty task subsets have now been evaluated using the frozen participant-disjoint outer folds and the model configuration selected independently within each outer training split. The current discovery analysis asks whether task count can be reduced while retaining performance relative to the complete eight-task assessment. Paired bootstrap uncertainty is complete; task-selection stability and leakage-safe nested task-selection validation remain before stronger generalization claims are made.

The work continues the human-centered motivation behind EyeSpeak-Assist. The researcher's aunt lived with ALS and ultimately succumbed to the disease; her experience motivated a Raspberry Pi–based blink-to-speech communication system and now this lower-burden approach to monitoring speech earlier in the progression of impairment.

## Scope Authority

The proposal in `proposal/ALS_Dysarthria_Research_Proposal.md` is the authoritative specification. Its title, motivation, research questions, methodology, clinical boundaries, and scope must not change without Blake's explicit approval.

The locked implementation details used for the current core experiment are documented in `proposal/MODELING_PROTOCOL.md`.

## Clinical and Ethical Guardrails

- Describe the work as assistive remote monitoring of speech impairment among people already diagnosed with ALS, not ALS diagnosis or screening.
- Keep all recordings and derived participant-level features from one participant in the same cross-validation fold.
- Perform preprocessing, model selection, hyperparameter selection, task selection, and any future evolutionary search only within training data at the appropriate validation level.
- Do not claim replacement of neurologists, speech-language pathologists, or ALSFRS-R assessment.
- Treat ordinal severity analysis as secondary and limit claims about moderate and severe groups, especially the five severe participants.
- Treat demographic subgroup findings as exploratory, not strong fairness conclusions.
- Follow VOC-ALS licensing, attribution, audio-handling, and derived-feature requirements.
- Treat task count as the primary burden proxy. Recording duration may be added as a secondary proxy after raw audio is validated; neither directly measures fatigue or perceived effort.
- Treat the current 255-subset results as discovery results until leakage-safe nested task-selection validation is completed.

## Directory Map

| Folder | Intended contents |
|---|---|
| `proposal/` | Authoritative proposal, locked modeling protocol, and documented amendments/corrections |
| `literature/` | Dataset documentation audit, papers, reading notes, evidence table, and bibliography |
| `data/` | Local VOC-ALS metadata, processed feature tables, and later raw-audio artifacts; participant-level data remain excluded from Git |
| `notebooks/` | Exploratory analyses and reproducible demonstrations |
| `src/` | Reusable auditing, feature-table, modeling, subset-evaluation, statistical-analysis, and plotting code |
| `experiments/` | Small tracked reproducibility manifests; generated runs remain ignored |
| `results/` | Tracked non-sensitive audits plus local generated metrics, predictions, confidence intervals, statistical comparisons, and tables |
| `figures/` | Tracked final plots, including single-task, task-frequency, and performance–burden Pareto figures |
| `presentation/` | Auburn class-presentation materials |
| `paper/` | Course proposal, manuscript, appendices, and future submission-ready artifacts |

## Current Core Status

Completed:

- VOC-ALS metadata/documentation audit
- 102-participant ALS-only binary target construction: 53 normal speech and 49 documented speech impairment
- Frozen five-outer/four-inner participant-disjoint folds with seed `2026`
- 40-feature participant-level modeling table from the published acoustic predictors
- Majority-class sanity baseline
- Nested comparison of 70 prespecified candidate model configurations
- Complete eight-task reference assessment
- Exhaustive evaluation of all 255 nonempty speech-task subsets
- Prespecified Macro F1 tolerance analysis at `0.01`, `0.02`, and `0.05`
- Task-frequency discovery analysis
- Performance-versus-task-count Pareto analysis and figures
- Paired participant-level out-of-fold predictions for `TA`, `E+TA+KA`, and the full eight-task reference
- 10,000-resample paired participant bootstrap confidence intervals and protocol comparisons

Current descriptive discovery results include a full eight-task mean Macro F1 of approximately `0.563`, a one-task `TA` result of approximately `0.685`, and a three-task `E+TA+KA` result of approximately `0.694`. These selected-subset values are discovery estimates and must not yet be interpreted as unbiased final generalization estimates.

Next core statistical work:

- Regression-check the frozen baseline and exhaustive subset results with explicit finite-value input validation in place
- Quantify protocol stability
- Perform nested task-selection validation so outer test participants do not influence which reduced protocol is selected

Raw-audio feature extraction, ordinal severity prediction, alternative representations/models, and NSGA-II remain secondary or optional extensions after the required core analysis is statistically validated.

Raw voice, participant-level metadata, fold assignments, feature rows, predictions, and generated participant-level research artifacts remain excluded from Git.

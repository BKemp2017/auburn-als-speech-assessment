# Research Proposal

## Proposed Title

**How Much Speech Is Enough? Multi-Objective Selection of Smartphone Voice Tasks for ALS Dysarthria Assessment**

### Short Title

**Minimal Smartphone Speech Assessment for ALS Dysarthria**

## Abstract

Amyotrophic lateral sclerosis (ALS) can progressively impair the motor control required for speech, producing dysarthria and increasing the burden of clinical and remote functional assessment. Smartphone voice recordings offer a scalable way to measure speech characteristics, but an assessment containing many vocal exercises may be unnecessarily tiring or difficult for people experiencing bulbar impairment. This study investigates how many and which speech tasks are required to retain useful performance when estimating ALS-related dysarthria from smartphone recordings. Experiments will use the public VOC-ALS dataset, which contains 1,224 recordings from 153 participants—102 people with ALS and 51 age- and sex-matched healthy controls—performing five sustained-vowel tasks (/a/, /e/, /i/, /o/, /u/) and three syllable-repetition tasks (/pa/, /ta/, /ka/). Acoustic features will be extracted from the raw recordings and evaluated using participant-disjoint nested cross-validation. The primary analysis will compare all 255 nonempty task subsets using fixed machine-learning pipelines to determine the Pareto frontier between predictive performance and assessment burden. The primary outcome will be detection of clinically documented speech impairment among participants with ALS; ordinal prediction of the ALSFRS-R speech subscore will be a secondary analysis because the moderate and severe classes are small. Logistic or ordinal regression, support-vector machines, random forests, and gradient-boosted trees will be compared using macro F1, balanced accuracy, mean absolute error, quadratic weighted kappa, calibration, and bootstrap confidence intervals. As an extension, NSGA-II will jointly optimize speech-task selection, acoustic-feature selection, and model configuration for predictive performance, assessment burden, and model complexity. The expected contribution is a reproducible estimate of the shortest smartphone speech protocol that preserves most of the information available from the complete eight-task assessment. This work is intended as an assistive remote-monitoring study, not as a standalone diagnostic system for ALS.

---

## 1. Background and Motivation

ALS is a progressive neurodegenerative disease that can impair the motor control required for speech. ALS-related dysarthria can affect articulation, phonation, respiratory support, intelligibility, and ultimately a person's ability to communicate. Smartphone voice recordings offer a practical way to supplement periodic clinical assessment with remote measurements, but the assessment process should not impose unnecessary effort on people already experiencing fatigue or bulbar impairment.

My motivation for studying this problem is personal. My aunt lived with ALS and ultimately succumbed to the disease. Watching ALS progressively affect her independence and ability to communicate showed me that assistive technology must be evaluated not only by its technical accuracy, but also by the physical and cognitive effort it requires from the person using it. That experience motivated me to develop **EyeSpeak-Assist**, a low-cost, Raspberry Pi–based system that converts intentional eye blinks into synthesized speech for people who can no longer rely on conventional motor input.

Developing EyeSpeak-Assist strengthened my interest in designing technology that adapts to the changing abilities of people with ALS. EyeSpeak addresses communication after substantial motor or speech impairment; the present study considers an earlier point in the communication-decline pathway, when speech remains available but may be weakening or becoming more difficult to sustain. A remote speech-monitoring system may be useful, but requiring numerous vocal exercises could create avoidable burden. This project therefore asks whether useful assessment performance can be preserved while requiring substantially less speech from the participant.

The VOC-ALS dataset was created using smartphone recordings and includes eight structured speech tasks from every participant:

- sustained `/a/`;
- sustained `/e/`;
- sustained `/i/`;
- sustained `/o/`;
- sustained `/u/`;
- repeated `/pa/`;
- repeated `/ta/`;
- repeated `/ka/`.

Existing studies establish that acoustic information in these recordings can distinguish speech impairment. The proposed study asks a different, human-centered question:

> **How much speech must a person provide before additional recording tasks offer little or no meaningful improvement?**

The project treats assessment design as a tradeoff between predictive information and participant burden. A shorter protocol could be easier to repeat, easier to integrate into a mobile application, and more practical for individuals experiencing fatigue or bulbar impairment. The guiding principle connecting this study with EyeSpeak-Assist is that technology intended for people with ALS should adapt to the user's abilities rather than require the user to adapt to the technology.

## 2. Problem Statement

Let the complete task set be

\[
T=\{a,e,i,o,u,pa,ta,ka\}.
\]

For a nonempty subset \(S\subseteq T\), a model is trained using only acoustic information from the tasks in \(S\). The study seeks task subsets that maximize predictive performance while minimizing assessment burden:

\[
\max_S \; Performance(S),
\]

\[
\min_S \; Burden(S).
\]

The simplest burden measure is the number of required tasks:

\[
Burden(S)=|S|.
\]

If recording duration is reliable, a secondary measure will use total duration:

\[
Burden_{time}(S)=\sum_{t\in S} Duration(t).
\]

The objective is not necessarily to identify a single universally optimal assessment. The analysis will identify a Pareto frontier of non-dominated protocols that offer different performance–burden compromises.

## 3. Purpose of the Study

The purpose of this study is to determine whether a reduced subset of smartphone voice tasks retains most of the predictive information contained in the complete VOC-ALS protocol for assessing ALS-related dysarthria.

The project will:

1. establish participant-independent performance using all eight tasks;
2. measure the individual and combined value of each vowel and syllable task;
3. enumerate the performance of every nonempty task subset;
4. identify non-dominated assessment protocols;
5. measure uncertainty around differences between protocols;
6. optionally use evolutionary multi-objective optimization for the larger joint task–feature–model search.

## 4. Research Questions

### Primary Research Question

**RQ1:** What is the smallest subset of smartphone speech tasks that retains performance within a prespecified tolerance of the full eight-task protocol when identifying clinically documented speech impairment among participants with ALS?

### Secondary Research Questions

**RQ2:** Do sustained vowels, syllable-repetition tasks, or combinations of both provide the best performance–burden tradeoff?

**RQ3:** Which acoustic features contribute consistently across participant-disjoint folds and model families?

**RQ4:** How accurately can reduced task subsets estimate the ordered ALSFRS-R speech impairment level?

**RQ5:** Does joint evolutionary optimization of tasks, acoustic features, and model configuration improve the Pareto frontier over task-only exhaustive search and conventional tuning?

**RQ6:** How stable are the selected minimal protocols across resamples, demographic strata, and alternative performance tolerances?

## 5. Hypotheses

**H1:** At least one subset containing three or fewer speech tasks will achieve performance within two percentage points of the complete eight-task protocol on the primary outcome.

**H2:** A mixed subset containing at least one vowel and one syllable-repetition task will outperform vowel-only and syllable-only subsets of equal size.

**H3:** Participant-disjoint evaluation will produce lower and more realistic performance than recording-level random splits.

**H4:** Joint task–feature–model optimization will identify configurations with lower assessment burden or model complexity without a statistically meaningful loss in performance.

These are falsifiable hypotheses. The project remains successful if the evidence shows that all eight tasks are required or that no stable minimal subset exists.

## 6. Dataset

### VOC-ALS

The project will use the publicly available **Voice Signals Database of ALS Patients with Different Dysarthria Severity and Healthy Controls (VOC-ALS)**.

Dataset composition:

- 153 participants;
- 102 participants with ALS;
- 51 age- and sex-matched healthy controls;
- 1,224 public `.wav` recordings;
- eight recordings per participant;
- demographic, clinical, and acoustic metadata;
- ALSFRS-R total and item-level clinical scores for participants with ALS.

Among the participants with ALS, the published speech-severity distribution is:

| ALSFRS-R speech score | Interpretation | Participants |
|---:|---|---:|
| 4 | Normal speech / non-dysarthric | 53 |
| 3 | Detectable speech disturbance / mild | 31 |
| 2 | Intelligible with repetition / moderate | 13 |
| 1 | Speech combined with non-vocal communication / severe | 5 |

The small moderate and severe groups make unrestricted four-class classification statistically fragile. The study will therefore distinguish between a primary, adequately supported outcome and a more exploratory ordinal outcome.

### Primary Target

Among participants with ALS:

\[
y=
\begin{cases}
0,& \text{ALSFRS-R speech score}=4\\
1,& \text{ALSFRS-R speech score}<4.
\end{cases}
\]

This target represents **clinically documented speech impairment versus no documented speech impairment**. It is not a diagnosis of ALS; every participant in this primary analysis already belongs to the ALS cohort.

### Secondary Target

Ordinal prediction of the speech subscore:

\[
y\in\{1,2,3,4\}.
\]

The secondary analysis will report uncertainty and will not claim reliable severe-class performance from five participants.

### Optional Contextual Analysis

A separate exploratory analysis may distinguish healthy controls from participants with ALS. It will be labeled clearly as dataset classification rather than a clinically validated screening or diagnostic system.

## 7. Acoustic Features

### Provided Features

The dataset includes basic features for each vocal task, including:

- mean fundamental frequency;
- standard deviation of fundamental frequency;
- harmonics-to-noise ratio;
- local jitter;
- local shimmer.

### Features Extracted from Raw Audio

The project will evaluate a reproducible feature-extraction pipeline containing a defensible subset of:

- duration and voiced duration;
- energy and energy variability;
- zero-crossing rate;
- spectral centroid, spread, rolloff, and flux;
- MFCC means and variability;
- pitch statistics;
- jitter and shimmer;
- harmonics-to-noise ratio;
- formant-related measurements where robustly obtainable;
- syllable-repetition rate and temporal regularity for `/pa/`, `/ta/`, and `/ka/`.

Feature extraction will be fixed before final evaluation. Features will be standardized inside each training fold to prevent leakage.

## 8. Machine-Learning Methods

### Baseline Models

1. Majority-class predictor
2. Logistic regression
3. Class-weighted logistic regression
4. Linear or radial-basis SVM

### Nonlinear Models

1. Random forest
2. Gradient-boosted trees, such as XGBoost or LightGBM
3. Optional small multilayer perceptron if the sample size and validation results justify it

### Ordinal Models

1. Ordinal logistic regression
2. Regression with bounded/rounded speech-score prediction
3. Optional ordinal gradient-boosting formulation

Deep audio models are not required for the core study. With 153 participants, a large neural architecture could overfit participant identity and distract from the assessment-design question.

## 9. Experimental Design

### Participant-Level Data Organization

The analytical unit is the participant, not the individual recording. Each participant contributes features from up to eight tasks. All recordings from a participant must remain in the same fold.

### Nested Cross-Validation

The planned evaluation uses participant-disjoint nested cross-validation:

- outer folds estimate generalization to unseen participants;
- inner folds select hyperparameters and any data-dependent features;
- preprocessing and feature selection occur inside the inner training data;
- predictions from outer held-out folds form the final evaluation set.

If class counts make five outer folds unstable, repeated stratified group cross-validation or leave-one-participant-out sensitivity analysis will be used.

### Task-Subset Search

Eight binary task decisions produce:

\[
2^8-1=255
\]

nonempty task subsets. Because this space is small, the primary task-selection experiment will evaluate all 255 subsets using a fixed, reproducible model pipeline. Exhaustive evaluation provides the true task-level Pareto frontier and avoids unjustified reliance on a heuristic search method.

### Prespecified Minimality Rule

The primary minimal protocol will be the smallest subset whose mean outer-fold performance is no more than a prespecified tolerance \(\delta\) below the full protocol:

\[
S^*=\arg\min_S |S|
\]

subject to

\[
Performance(S)\ge Performance(T)-\delta.
\]

The initial tolerance will be \(\delta=0.02\) in macro F1, with 0.01 and 0.05 evaluated in sensitivity analysis. Statistical uncertainty, not the point estimate alone, will be considered before declaring two protocols practically equivalent.

## 10. Evolutionary-Computing Extension

Evolutionary computation is unnecessary for only 255 task subsets, but becomes appropriate when the search jointly includes tasks, features, models, and hyperparameters.

### Genotype

An individual may be represented as:

\[
G=[T_1,\ldots,T_8,F_1,\ldots,F_p,M,H_1,\ldots,H_k],
\]

where:

- \(T_i\) is a binary gene specifying whether speech task \(i\) is included;
- \(F_j\) is a binary gene specifying whether feature or feature group \(j\) is included;
- \(M\) is a categorical gene encoding the model family;
- \(H_k\) encodes model-specific hyperparameters.

### Genotype-to-Phenotype Mapping

The genotype maps to a complete assessment and modeling pipeline. For example:

\[
[1,0,1,0,0,0,1,0,\ldots]
\]

may map to an assessment containing `/a/`, `/i/`, and `/ta/`, a selected acoustic-feature set, and a class-weighted SVM configuration.

### Feasibility

An individual is infeasible if it selects no speech tasks, no usable features, or an invalid hyperparameter combination. Repair operators will preferentially restore feasibility by enabling at least one task and the minimum required feature set.

### Fitness Objectives

NSGA-II will optimize:

\[
\max f_1(G)=\text{participant-disjoint validation performance},
\]

\[
\min f_2(G)=\text{assessment burden},
\]

\[
\min f_3(G)=\text{model or feature complexity}.
\]

The evolutionary search will be restricted to inner training folds. Outer test participants will never influence evolution, selection, or hyperparameter decisions.

### Evolutionary Baselines

- Random search with the same evaluation budget
- Conventional hyperparameter optimization
- Exhaustive task-only frontier
- NSGA-II ablations without feature selection or model-choice genes

The evolutionary extension will be considered successful only if it improves the obtained frontier or reaches comparable solutions using fewer pipeline evaluations.

## 11. Evaluation Metrics

### Primary Binary Outcome

- Macro F1
- Balanced accuracy
- Sensitivity and specificity
- Area under the ROC curve
- Area under the precision–recall curve
- Brier score and calibration curve
- Confusion matrix

Macro F1 will be the primary selection metric because both classes must receive meaningful weight.

### Secondary Ordinal Outcome

- Mean absolute error
- Quadratic weighted kappa
- Spearman correlation
- Per-class recall
- Ordinal confusion matrix

### Assessment-Burden Outcomes

- Number of required tasks
- Total or median recording duration
- Number of extracted features
- Optional estimated on-device inference time

### Statistical Analysis

- Participant-level bootstrap confidence intervals
- Paired bootstrap comparison of reduced and full protocols
- Permutation testing where appropriate
- Stability frequency of each selected task across resamples
- Sensitivity analysis across values of \(\delta\)
- Exploratory stratification by sex, age group, and spinal versus bulbar onset, with strong caution about small subgroups

## 12. Expected Results

The anticipated result is a small set of speech tasks—possibly a mixture of vowel phonation and syllable repetition—that retains most of the predictive information of the complete protocol. However, the proposal does not assume that three or fewer tasks must succeed.

Possible valid findings include:

1. a two- or three-task protocol performs comparably to all eight tasks;
2. syllable repetitions dominate sustained vowels;
3. a mixed protocol is necessary;
4. selected tasks differ across resamples, indicating no stable minimal protocol;
5. all eight tasks are necessary for reliable participant-independent performance.

Each outcome answers the research question and informs future assessment design.

## 13. Significance

This project shifts the emphasis from maximizing classification performance at any participant cost to balancing predictive information with assessment burden. Potential benefits include:

- shorter remote speech assessments;
- reduced repetition for users with speech fatigue or bulbar impairment;
- easier integration into smartphone or assistive-communication applications;
- clearer evidence about which vocal exercises provide distinct information;
- a reproducible framework for evaluating burden-aware digital biomarkers.

This project is a continuation of the motivation behind my prior EyeSpeak-Assist work. My aunt's experience with ALS first led me to investigate how affordable technology could preserve communication as motor abilities decline. EyeSpeak-Assist focuses on blink-driven communication after conventional input or speech becomes difficult, while the present study investigates lower-burden speech monitoring earlier in that progression. Together, the projects reflect the same human-centered objective: preserve useful functionality while minimizing the effort demanded from the person with ALS.

## 14. Ethical and Clinical Boundaries

- The study uses an existing de-identified public dataset and does not propose collecting new human-subject data.
- The system will not be described as diagnosing ALS.
- The primary analysis concerns speech impairment among participants already diagnosed with ALS.
- Results will not be presented as replacing neurologists, speech-language pathologists, or the ALSFRS-R clinical assessment.
- The severe category contains only five participants; conclusions about severe dysarthria will be explicitly limited.
- Demographic subgroup results will be exploratory because the dataset is too small for strong fairness claims.
- Audio and derived features will be handled according to the dataset license and attribution requirements.

## 15. Threats to Validity

### Internal Validity

- Leakage could occur if recordings from one participant appear in both training and testing.
- Feature and task selection outside nested cross-validation would inflate performance.
- Repeated model comparisons could encourage selecting a lucky configuration.

Controls include participant grouping, nested evaluation, prespecified primary metrics, and bootstrap uncertainty.

### Construct Validity

- ALSFRS-R speech scores are coarse clinical categories and do not capture every aspect of speech intelligibility or communication burden.
- Number of tasks is only a proxy for patient effort.
- Recording duration does not directly measure fatigue.

### External Validity

- Data come from one cohort and smartphone acquisition protocol.
- Models may not generalize across languages, accents, microphones, noise conditions, or clinical centers.
- The smallest severity groups limit reliable estimation of full ordinal performance.

### Conclusion Validity

- With 102 ALS participants, small performance differences may be statistically uncertain.
- Results will emphasize confidence intervals and protocol stability rather than only the best point estimate.

## 16. Work Plan

### Phase 1 — Literature and Reproduction

- Review VOC-ALS documentation and related ALS dysarthria studies.
- Download and validate audio and metadata.
- Reproduce published basic acoustic analyses.
- Freeze participant identifiers and evaluation folds.

### Phase 2 — Feature Pipeline

- Implement deterministic audio preprocessing.
- Extract and validate acoustic features.
- Produce participant-level feature tables.
- Add tests for missing and malformed recordings.

### Phase 3 — Full-Protocol Baselines

- Train simple baseline models using all eight tasks.
- Establish nested participant-disjoint evaluation.
- Select the primary model pipeline for exhaustive task analysis.

### Phase 4 — Minimal-Protocol Analysis

- Evaluate all 255 nonempty subsets.
- Construct performance–burden Pareto fronts.
- Apply the prespecified minimality rule.
- Estimate confidence intervals and selection stability.

### Phase 5 — Ordinal and Robustness Analyses

- Estimate ALSFRS-R speech scores.
- Run sensitivity analyses for tolerance, metrics, features, and folds.
- Evaluate subgroup and calibration results cautiously.

### Phase 6 — Evolutionary Extension

- Implement genotype, mapping, feasibility checks, and NSGA-II.
- Compare NSGA-II against random search and conventional tuning.
- Keep all evolutionary selection inside inner folds.

### Phase 7 — Reporting

- Prepare final report, reproducible notebook/scripts, figures, and presentation.
- Document limitations and prevent diagnostic overstatement.

## 17. Proposed Semester Timeline

| Weeks | Work |
|---|---|
| 1–2 | Literature review, data audit, reproducible environment |
| 3–4 | Audio preprocessing and feature extraction |
| 5 | Full-protocol baseline models |
| 6–7 | Exhaustive task-subset evaluation |
| 8 | Statistical analysis and Pareto frontier |
| 9 | Ordinal severity and robustness experiments |
| 10 | Evolutionary optimization extension |
| 11 | Ablations, error analysis, final figures |
| 12 | Paper and presentation |

The evolutionary extension may be reduced if data preparation or validation requires additional time. The exhaustive minimal-assessment study is the required core deliverable.

## 18. Resources and Tools

- Python
- NumPy and pandas
- scikit-learn
- librosa and/or openSMILE
- XGBoost or LightGBM
- DEAP, pymoo, or an equivalent evolutionary-computing framework
- Jupyter for exploratory analysis
- Reproducible scripts for final experiments

CPU execution should be sufficient for the core study. GPU use is optional and not required.

## 19. Expected Deliverables

1. Cleaned participant-level feature dataset
2. Reproducible audio feature-extraction pipeline
3. Participant-disjoint baseline evaluation
4. Exhaustive results for all 255 task subsets
5. Performance–burden Pareto frontier
6. Minimal-protocol recommendation with uncertainty estimates
7. Optional NSGA-II joint optimization experiment
8. Final research paper
9. Class presentation
10. Reproducible source repository

## 20. Scope Control

### Required Core

- Binary impairment outcome among ALS participants
- Participant-disjoint nested evaluation
- Fixed-model exhaustive search over 255 task subsets
- Pareto and statistical analysis
- Final written report and presentation

### Strong Extension

- Ordinal speech-score prediction
- Rich raw-audio feature extraction
- Calibration and subgroup error analysis

### Optional Stretch Extension

- NSGA-II joint task–feature–model optimization
- On-device inference prototype
- Integration concepts connecting the model to a future assistive application

This hierarchy ensures that the project remains complete and academically strong even if optional work is not finished.

## 21. Initial References

1. Dubbioso, R., Spisto, M., Verde, L., et al. (2024). *Voice signals database of ALS patients with different dysarthria severity and healthy controls*. Scientific Data, 11, 800. https://doi.org/10.1038/s41597-024-03597-2
2. Cedarbaum, J. M., Stambler, N., Malta, E., et al. (1999). *The ALSFRS-R: a revised ALS functional rating scale that incorporates assessments of respiratory function*. Journal of the Neurological Sciences, 169(1–2), 13–21.
3. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
4. VOC-ALS dataset description and public recordings: https://pmc.ncbi.nlm.nih.gov/articles/PMC11271596/
5. Recent VOC-ALS dysarthria-recognition study: https://eusipco2025.org/wp-content/uploads/pdfs/0001527.pdf

## 22. Concise Professor Pitch

> My motivation for this research is personal: my aunt lived with ALS and ultimately succumbed to the disease. Watching ALS affect her independence and communication motivated me to develop EyeSpeak-Assist, a low-cost blink-to-speech system for people who can no longer rely on conventional motor input. This study extends that assistive-technology work to an earlier point in speech decline. ALS-related dysarthria can make repeated assessment burdensome, yet smartphone-based systems may request several vocal exercises without establishing how many are actually necessary. I propose using the public VOC-ALS dataset to evaluate every combination of five vowel and three syllable-repetition tasks under participant-disjoint nested cross-validation. The primary goal is to identify the smallest protocol that retains performance close to the full eight-task assessment for detecting documented speech impairment among people with ALS. Because there are only 255 task subsets, the core study will use exhaustive evaluation. As an evolutionary-computing extension, I will use NSGA-II to jointly optimize tasks, acoustic features, and model configuration for predictive performance, assessment burden, and model complexity. The study is intended to inform lower-burden remote monitoring, not to diagnose ALS.

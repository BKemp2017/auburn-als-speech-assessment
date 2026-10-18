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
- [X] Protocol-stability analysis.
- [X] Leakage-safe nested task-selection validation of the reduced-protocol selection procedure.
- [ ] Final research paper.
- [ ] Auburn class presentation.
- [ ] Final reproducibility freeze and clean-environment rerun.

## Strong Extensions

- [x] Candidate-level robustness analysis for every fixed task protocol.
- [x] Determine whether any one-to-three-task protocol is consistently noninferior to the complete assessment.
- [x] Audit the raw VOC-ALS recordings and construct a validated participant/task audio manifest.
- [x] Extract frozen pretrained neural representations directly from the raw recordings.
- [x] Re-run the minimal-task question using neural audio representations to test whether the reduced-burden conclusion is representation-robust.
- [ ] Evaluate participant-specific adaptive recording and early stopping.
- [x] Quantify recording-count burden reduction.
- [ ] Quantify recording-duration and active-speech burden reduction.
- [ ] Secondary ordinal ALSFRS-R speech-score prediction with appropriately limited claims.
- [ ] Calibration analysis.
- [ ] Cautious subgroup and error analysis.
- [ ] Robustness analyses across folds, metrics, feature sets, tolerances, and resamples.

## Optional Future Work

- [ ] Optional on-device inference prototype.
- [ ] Prospective validation with directly measured fatigue, perceived effort, completion time, and participant preference.
- [ ] External validation on another ALS speech cohort, device, language, or recording environment when suitable data becomes available.
- [ ] Future integration concepts for an assistive application.

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
- [x] Download and validate the raw audio for the raw-audio extension.
- [x] Create the raw-audio participant/task manifest and validate eight expected recordings per participant.
- [x] Implement and validate the frozen WavLM embedding-extraction pipeline.

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
- [X] Quantify protocol stability across participant resamples.
- [X] Perform nested task-selection validation to estimate the generalization of the selection procedure without outer-test-driven subset choice.

### Week 9 — Protocol robustness and raw-audio audit

- [x] Calculate candidate-level paired performance differences against the complete eight-task assessment.
- [x] Estimate bootstrap confidence intervals and noninferiority probabilities for every task subset.
- [x] Rank candidate protocols by task count, lower confidence bound, fold stability, and bootstrap robustness.
- [x] Determine that the evidence supports a general adaptive low-task-burden policy rather than one universal fixed protocol.
- [x] Download and audit the raw VOC-ALS recordings without committing participant-level data.
- [x] Validate participant identifiers, task labels, recording coverage, sample rates, channels, clipping, silence, corruption, and durations.
- [x] Audit exact duplicate recordings and evaluate duplicate-exclusion and duplicate-policy sensitivity analyses.
- [x] Lock the neural encoder, metrics, folds, duplicate policy, and noninferiority margin before neural evaluation.

### Week 10 — Neural confirmation and adaptive stopping

- [x] Extract one frozen pretrained WavLM embedding for every retained participant/task recording.
- [x] Validate the embedding artifact for participant/task coverage, dimensional consistency, finite values, and a pinned encoder revision.
- [x] Evaluate all 255 task subsets inside the nested neural selection procedure.
- [x] Repeat leakage-safe nested task selection using neural embeddings.
- [x] Compare reduced and complete neural protocols using paired outer-test predictions.
- [x] Run a 10,000-replicate paired bootstrap with the locked delta = 0.02 noninferiority margin.
- [ ] Repeat the complete neural analysis with 8 and 32 PCA components per task as a representation sensitivity check.
- [ ] Learn task order and stopping thresholds using inner folds only.
- [ ] Evaluate adaptive early stopping on untouched outer-test participants.
- [ ] Quantify tasks, recording duration, and active speech avoided.

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
- [X] **Uncertainty gate:** Paired participant bootstrap and protocol-stability analyses support the reduced-protocol discovery before stronger claims are made.
- [X] **Selection-validation gate:** Nested task selection confirms that reduced-protocol selection generalizes without using outer-test performance to choose the subset.
- [x] **Raw-audio integrity gate:** Recording coverage, signal format, waveform quality, exact duplicates, and duplicate-policy sensitivity have been audited before neural modeling.
- [x] **Neural-evaluation gate:** Duplicate-filtered frozen WavLM embeddings have been evaluated with participant-disjoint nested selection and paired outer-test predictions.
- [ ] **Core-complete gate:** Core modeling, uncertainty analysis, nested selection validation, paper, presentation, and reproducibility freeze are complete before final submission.
- [ ] **Audio-extension gate:** Raw-audio and adaptive-stopping conclusions are included only after audio integrity, participant-disjoint evaluation, representation robustness, calibration, and burden calculations are validated.
- [ ] **Claim gate:** Final paper/presentation claims remain limited to speech-impairment assessment in the studied ALS cohort, not ALS diagnosis, clinical replacement, or proven real-world fatigue reduction.

## Current Discovery Checkpoint

The exhaustive handcrafted-feature discovery analysis is complete. The complete eight-task reference achieved mean Macro F1 of approximately 0.563. The descriptive Pareto frontier contained:

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
These remain post-selection discovery estimates. The completed nested
task-selection analysis provides the leakage-safe estimate of the
selection procedure's generalization performance.

Leakage-safe nested task selection is now complete. With the locked
delta = 0.02 rule, the selected protocols contained one to three tasks
and achieved mean outer-test Macro F1 of approximately 0.572, compared
with approximately 0.563 for the complete eight-task assessment. The
delta = 0.01 sensitivity analysis reproduced the same selections, while
delta = 0.05 selected an average of 1.2 tasks and achieved mean
outer-test Macro F1 of approximately 0.618. Delta = 0.02 remains the
locked primary analysis.

Participant-resampled protocol-stability analysis is complete using
10,000 paired resamples within each inner fold. Across 50,000 selection
decisions, 62.09% selected one task, 91.37% selected one or two tasks,
and 98.27% selected no more than three tasks. The complete eight-task
assessment was selected only 3 times. Exact protocol identity was less
stable: the original fold-specific protocol was reproduced approximately
20.7% of the time overall. KA was the most frequently selected individual
task, appearing in 50.57% of selections. These results support a stable
reduction-in-burden conclusion, but not a claim that one exact task
combination is universally optimal.

Fixed-protocol robustness analysis is complete for all 255 task subsets.
No reduced fixed protocol had point estimates within the delta = 0.02
margin in all five outer folds, and no reduced protocol had confidence-bound
support in every fold. `O+TA+KA` was the highest-ranked exploratory fixed
candidate, with a mean paired difference of approximately -0.002, but its
worst-fold difference was approximately -0.035 and its minimum fold-level
noninferiority probability was approximately 0.357. These results do not
support recommending one universal fixed subset.

The raw-audio audit is complete. VOC-ALS contains 1,224 recordings from
153 participants with all eight expected tasks represented. Every file is
8 kHz, mono, 16-bit PCM WAV. The waveform audit found no all-zero files,
one near-silent recording, one recording with at least 0.1% clipped samples,
and eight exact duplicate signal groups containing 17 files. One exact
duplicate relationship crossed ALS participant targets and outer-fold
boundaries. The conservative neural and duplicate-exclusion analyses therefore
removed both duplicate-connected ALS participants, leaving 100 participants
with a 52/48 target distribution.

Duplicate-excluded handcrafted nested selection remained favorable. The
selected protocols used one to three tasks and achieved mean outer-test
Macro F1 of approximately 0.597, compared with approximately 0.499 for the
duplicate-filtered complete assessment. Additional duplicate-policy analyses
that excluded task A or removed either member of the conflicting pair also
retained a positive selected-versus-reference mean difference. The primary
reduced-burden conclusion is therefore not explained by the identified exact
duplicate recordings.

Frozen WavLM evaluation is complete on 800 recordings from the 100 retained
participants. Embeddings were extracted with `microsoft/wavlm-base-plus` at
revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. Mean and standard-deviation
pooling produced 1,536 features per recording. Within every inner and outer
training split, each task was standardized and reduced to 16 PCA components
before balanced logistic-regression classification.

Leakage-safe WavLM nested selection chose `PA`, `A+PA`, `I`, `I`, and `PA`
across the five outer folds. The selected protocols used an average of 1.2
tasks and achieved mean outer-test Macro F1 of 0.650794, compared with
0.668563 for the complete eight-task WavLM assessment. The mean paired
difference was -0.017768, which is inside the locked -0.02 point-estimate
margin by only 0.002232.

The 10,000-replicate paired WavLM bootstrap did not establish formal
noninferiority. The two-sided 95% percentile interval for selected minus full
Macro F1 was [-0.099419, 0.062728], the one-sided 95% lower bound was
-0.086072, and the estimated probability of remaining within the margin was
0.520300. Only two of five outer folds were individually inside the margin.
The neural result independently supports low-task feasibility and stronger
absolute predictive performance, but uncertainty remains too large to claim
that one or two recordings are noninferior to all eight.

The combined evidence supports a cautious conclusion: large reductions in
recording count can preserve similar average speech-impairment classification
performance, but exact task identity is unstable and neither a universal fixed
protocol nor formal neural noninferiority has been demonstrated. The study did
not directly measure fatigue, perceived effort, completion time, or patient
preference. Any fatigue benefit remains a plausible motivation for prospective
validation rather than an observed outcome of this retrospective dataset.

## Immediate Next Action

Run the final WavLM representation sensitivity analysis with 8 and 32 PCA
components per task while preserving the duplicate-filtered cohort, frozen
participant folds, encoder revision, delta = 0.02 rule, regularization grid,
selection procedure, outer-test isolation, bootstrap seed, and all other
analysis choices. Treat the existing 16-component experiment as the locked
primary neural analysis.

Use the sensitivity analysis only to determine whether the qualitative
low-task and uncertainty conclusions depend materially on PCA dimensionality.
Do not tune the encoder, margin, task-selection rule, or classifier in response
to outer-test results. After this check, freeze all modeling outputs and move to
the final paper, presentation, and clean-environment reproducibility rerun.

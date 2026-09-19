# VOC-ALS Documentation Audit

## Status

Initial documentation review only. No dataset files have been downloaded or inspected.

## Verified dataset facts

- The published cohort contains 153 participants: 102 participants with ALS and 51 age- and sex-matched healthy controls.
- The public audio contains 1,224 WAV recordings: eight recordings per participant.
- The eight public tasks are sustained `/a/`, `/e/`, `/i/`, `/o/`, `/u/` and repeated `/pa/`, `/ta/`, `/ka/`.
- Every public recording filename includes a pseudonymized participant identifier and task label.
- The accompanying spreadsheet contains one participant row with participant ID, demographics, cohort category, clinical variables, task-level acoustic features, and a self-report questionnaire score.
- The published feature table includes five features for each of eight tasks: mean F0, F0 standard deviation, HNR, local jitter, and local shimmer (40 acoustic columns total).
- The recordings were sampled at 8,000 Hz with 16-bit resolution and saved as WAV.
- Acquisition occurred in one supervised clinic session, in a quiet outpatient room, using Vox4Health on a Samsung Galaxy S8+ running Android 9. The phone was held approximately 20 cm from the participant at about 45 degrees.
- Sustained vowels were produced for at least five seconds at continuous loudness. `/pa/`, `/ta/`, and `/ka/` were repeated as rapidly as possible in a single breath.
- Additional connected-speech tasks were collected but not released because they presented greater re-identification risk.

## Verified target definition

The primary cohort is the 102 participants already diagnosed with ALS.

- `ALSFRS-R_SpeechSubscore = 4`: normal speech process / no documented speech impairment.
- `ALSFRS-R_SpeechSubscore < 4`: clinically documented speech impairment.

The dataset paper reports 49 of 102 ALS participants with speech abnormality and therefore 53 without documented speech impairment. The severity counts in the proposal are 31 mild (score 3), 13 moderate (score 2), and 5 severe (score 1). No score-0 participant is reported.

This outcome assesses speech impairment within an ALS cohort. It must not be described as detecting or diagnosing ALS.

## Methodological consequences

1. **Participant grouping is mandatory.** All eight recordings and all derived features belonging to one participant must remain in the same fold.
2. **The analytical unit is the participant.** Recording rows cannot be treated as independent samples.
3. **The 8 kHz sample rate imposes a 4 kHz Nyquist limit.** Feature extraction must not assume information above 4 kHz, and any spectral/formant feature must be validated at this bandwidth.
4. **This is not an uncontrolled at-home dataset.** It supports research toward remote smartphone assessment, but external validity to home environments, newer phones, varying microphone placement, noise, language, and unsupervised use remains untested.
5. **Device variation cannot be estimated.** All recordings used one Samsung Galaxy S8+ acquisition setup.
6. **Task count is the defensible primary burden proxy.** Duration may be a secondary proxy after inspecting metadata and audio durations; neither directly measures fatigue or perceived effort.
7. **The public audio is pseudonymized, not inherently anonymous.** Raw voice, metadata, and derived features require careful handling and must not be committed to a public repository.
8. **Article licensing and dataset access are separate questions.** The article is CC BY 4.0, but the applicable Synapse access conditions and dataset-specific terms must be recorded before download or redistribution decisions.

## Open items to resolve before download

- Exact Synapse project identifier, access tier, terms of use, citation requirement, and redistribution prohibition, if any.
- Exact spreadsheet filename, schema, missing-value encodings, and whether every participant has all eight public recordings.
- Whether Auburn requires an institutional determination for analysis of existing pseudonymized public human-participant data.
- Whether sex, age, and onset-region variables are complete enough for the proposal's explicitly exploratory subgroup checks.
- Whether recording duration can serve as a stable secondary burden measure.
- Whether the authors' public Parselmouth extraction script reproduces the spreadsheet feature values within a documented tolerance.

## Primary sources

1. Dubbioso, R., Spisto, M., Verde, L., et al. (2024). *Voice signals database of ALS patients with different dysarthria severity and healthy controls*. Scientific Data, 11, 800. https://doi.org/10.1038/s41597-024-03597-2
2. VOC-ALS feature-extraction repository: https://github.com/giovannasannino/VOC-ALS
3. NINDS Common Data Element for ALSFRS-R speech score: https://cde.nlm.nih.gov/deView?tinyId=HnoNuF5mmvG
4. Cedarbaum, J. M., et al. (1999). *The ALSFRS-R: a revised ALS functional rating scale that incorporates assessments of respiratory function*. https://doi.org/10.1016/S0022-510X(99)00210-5
5. Synapse data-access documentation: https://docs.synapse.org/synapse-docs/data-access-types


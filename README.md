# Auburn ML Research — Minimal ALS Speech Assessment

## Authoritative Project Title

**How Much Speech Is Enough? Multi-Objective Selection of Smartphone Voice Tasks for ALS Dysarthria Assessment**

## Project Summary

This Auburn University graduate machine-learning research project asks whether a smaller subset of the eight smartphone speech exercises in VOC-ALS can preserve most of the predictive information of the complete assessment while reducing participant burden. The primary analysis concerns clinically documented speech impairment versus normal speech among participants already diagnosed with ALS. It is an assistive remote-monitoring study, not an ALS diagnostic or clinical-replacement system.

The required experiment will keep every participant's recordings together, use participant-disjoint nested cross-validation, and exhaustively evaluate all 255 nonempty combinations of five sustained vowels and three syllable-repetition tasks. Macro F1 is the primary selection metric. A minimal protocol is the smallest subset within a prespecified tolerance of the full eight-task protocol, with uncertainty and selection stability considered before practical equivalence is claimed.

The work continues the human-centered motivation behind EyeSpeak-Assist. The researcher's aunt lived with ALS and ultimately succumbed to the disease; her experience motivated a Raspberry Pi–based blink-to-speech communication system and now this lower-burden approach to monitoring speech earlier in the progression of impairment.

## Scope Authority

The proposal in `proposal/ALS_Dysarthria_Research_Proposal.md` is the authoritative specification. Its title, motivation, research questions, methodology, clinical boundaries, and scope must not change without Blake's explicit approval.

## Clinical and Ethical Guardrails

- Describe the work as assistive remote monitoring, not ALS diagnosis or screening.
- Keep all recordings from each participant in the same fold.
- Perform preprocessing, feature selection, tuning, and any evolutionary search only within training data at the appropriate nested-CV level.
- Do not claim replacement of neurologists, speech-language pathologists, or ALSFRS-R assessment.
- Treat ordinal severity analysis as secondary and limit claims about moderate and severe groups, especially the five severe participants.
- Treat demographic subgroup findings as exploratory, not strong fairness conclusions.
- Follow VOC-ALS licensing, attribution, audio-handling, and derived-feature requirements.
- State that task count and recording duration are imperfect proxies for participant effort and fatigue.

## Directory Map

| Folder | Intended contents |
|---|---|
| `proposal/` | Authoritative proposal and approved amendments |
| `literature/` | Papers, reading notes, evidence table, bibliography |
| `data/` | Dataset documentation and later local data artifacts; no dataset downloaded yet |
| `notebooks/` | Exploratory analyses and reproducible demonstrations |
| `src/` | Tested reusable preprocessing, feature, modeling, and evaluation code |
| `experiments/` | Frozen configurations, fold definitions, manifests, and run records |
| `results/` | Metrics, confidence intervals, statistical comparisons, and tables |
| `figures/` | Final plots, including performance–burden Pareto frontiers |
| `presentation/` | Auburn class-presentation materials |
| `paper/` | Manuscript, appendices, and submission-ready artifacts |

VOC-ALS metadata has been downloaded locally and audited, participant-disjoint folds have been frozen, the 40-feature participant-level table has been built, and baseline modeling is underway. Raw participant data and generated participant-level artifacts remain excluded from Git.

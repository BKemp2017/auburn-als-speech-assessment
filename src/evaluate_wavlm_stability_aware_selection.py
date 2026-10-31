"""Evaluate a stability-aware reduced WavLM speech assessment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata import TASKS
from bootstrap_wavlm_task_selection import (
    print_paired_wavlm_bootstrap_summary,
    run_paired_wavlm_bootstrap,
    summarize_paired_wavlm_bootstrap,
)
from evaluate_wavlm_task_selection import (
    evaluate_wavlm_outer_folds,
    print_wavlm_outer_summary,
)
from wavlm_task_selection_core import (
    EXPECTED_INNER_FOLDS_PER_OUTER_FOLD,
    EXPECTED_OUTER_FOLDS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_TASK_SUBSETS,
    PCA_COMPONENTS_PER_TASK,
    PRIMARY_DELTA,
    load_duplicate_filtered_design,
    load_wavlm_artifact,
    validate_wavlm_inputs,
)
from wavlm_task_selection_results import (
    INNER_RESULTS_PATH,
    convert_json_value,
)

MAX_SELECTED_TASKS = 3

OUTPUT_DIRECTORY = Path(
    "results/generated"
)
CANDIDATE_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_stability_aware_candidates.csv"
)
SELECTIONS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_stability_aware_selections.csv"
)
FOLD_METRICS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_stability_aware_fold_metrics.csv"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_stability_aware_predictions.csv"
)
BOOTSTRAP_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_stability_aware_bootstrap.csv"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/wavlm_stability_aware_selection.json"
)

INNER_SCORE_COLUMNS = tuple(
    f"inner_fold_{inner_fold}_macro_f1"
    for inner_fold in range(
        EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    )
)


def load_primary_inner_results() -> pd.DataFrame:
    """Load the locked primary WavLM inner-CV subset results."""

    if not INNER_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Run src/evaluate_wavlm_task_selection.py first. "
            f"Missing: {INNER_RESULTS_PATH}"
        )

    results = pd.read_csv(
        INNER_RESULTS_PATH
    )

    required_columns = {
        "outer_fold",
        "subset_id",
        "task_count",
        "predictor_count",
        "regularization_c",
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
        *INNER_SCORE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Primary WavLM inner results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_rows = (
        EXPECTED_OUTER_FOLDS
        * EXPECTED_TASK_SUBSETS
    )

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} inner-result rows, "
            f"found {len(results)}."
        )

    numeric_columns = [
        "outer_fold",
        "task_count",
        "predictor_count",
        "regularization_c",
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
        *INNER_SCORE_COLUMNS,
    ]

    if not np.isfinite(
        results[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Primary WavLM inner results contain a non-finite value."
        )

    full_subset_id = "+".join(
        TASKS
    )

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_results = results.loc[
            results["outer_fold"].eq(
                outer_fold
            )
        ]

        if len(
            fold_results
        ) != EXPECTED_TASK_SUBSETS:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain all "
                "255 task subsets."
            )

        if fold_results[
            "subset_id"
        ].nunique() != EXPECTED_TASK_SUBSETS:
            raise ValueError(
                f"Outer fold {outer_fold} contains duplicate subsets."
            )

        if fold_results[
            "regularization_c"
        ].nunique() != 1:
            raise ValueError(
                f"Outer fold {outer_fold} contains multiple "
                "regularization values."
            )

        if int(
            fold_results[
                "subset_id"
            ].eq(full_subset_id).sum()
        ) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain exactly "
                "one full-assessment result."
            )

    return results


def select_stability_aware_subsets(
    inner_results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Select competitive protocols using worst-inner-fold performance."""

    results = inner_results.copy()

    results[
        "inner_macro_f1_min"
    ] = results[
        list(INNER_SCORE_COLUMNS)
    ].min(
        axis=1
    )

    results[
        "stability_eligible"
    ] = False
    results[
        "stability_selected"
    ] = False

    selection_rows: list[
        dict[str, object]
    ] = []

    full_subset_id = "+".join(
        TASKS
    )

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_mask = results[
            "outer_fold"
        ].eq(
            outer_fold
        )
        fold_results = results.loc[
            fold_mask
        ]

        full_result = fold_results.loc[
            fold_results[
                "subset_id"
            ].eq(
                full_subset_id
            )
        ]

        if len(full_result) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain one "
                "full-assessment result."
            )

        full_reference = float(
            full_result.iloc[
                0
            ][
                "inner_macro_f1_mean"
            ]
        )
        competitive_threshold = (
            full_reference
            - PRIMARY_DELTA
        )

        eligible_mask = (
            fold_mask
            & results[
                "task_count"
            ].le(
                MAX_SELECTED_TASKS
            )
            & results[
                "inner_macro_f1_mean"
            ].ge(
                competitive_threshold
            )
        )

        eligible_results = results.loc[
            eligible_mask
        ]

        if eligible_results.empty:
            raise ValueError(
                f"Outer fold {outer_fold} has no eligible "
                "stability-aware protocol."
            )

        selected_index = (
            eligible_results
            .sort_values(
                by=[
                    "inner_macro_f1_min",
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                    "task_count",
                    "subset_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index[
                0
            ]
        )

        results.loc[
            eligible_mask,
            "stability_eligible",
        ] = True

        results.loc[
            selected_index,
            "stability_selected",
        ] = True

        selected_row = results.loc[
            selected_index
        ]

        selected_tasks = tuple(
            str(
                selected_row[
                    "subset_id"
                ]
            ).split(
                "+"
            )
        )

        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_subset": str(
                    selected_row[
                        "subset_id"
                    ]
                ),
                "selected_tasks": (
                    selected_tasks
                ),
                "task_count": int(
                    selected_row[
                        "task_count"
                    ]
                ),
                "predictor_count": int(
                    selected_row[
                        "predictor_count"
                    ]
                ),
                "regularization_c": float(
                    selected_row[
                        "regularization_c"
                    ]
                ),
                "selected_inner_macro_f1_min": float(
                    selected_row[
                        "inner_macro_f1_min"
                    ]
                ),
                "selected_inner_macro_f1_mean": float(
                    selected_row[
                        "inner_macro_f1_mean"
                    ]
                ),
                "selected_inner_macro_f1_std": float(
                    selected_row[
                        "inner_macro_f1_std"
                    ]
                ),
                "inner_full_reference": (
                    full_reference
                ),
                "competitive_threshold": (
                    competitive_threshold
                ),
                "competitive_subset_count": int(
                    eligible_mask.sum()
                ),
                "delta": float(
                    PRIMARY_DELTA
                ),
            }
        )

    selections = pd.DataFrame(
        selection_rows
    )

    validate_stability_aware_selections(
        results=results,
        selections=selections,
    )

    return (
        results,
        selections,
    )


def validate_stability_aware_selections(
    results: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    """Verify the frozen stability-aware rule was followed exactly."""

    if len(
        selections
    ) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Expected one stability-aware selection per outer fold."
        )

    if selections[
        "task_count"
    ].gt(
        MAX_SELECTED_TASKS
    ).any():
        raise ValueError(
            "A stability-aware selection exceeds the three-task cap."
        )

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_results = results.loc[
            results[
                "outer_fold"
            ].eq(
                outer_fold
            )
        ]

        eligible = fold_results.loc[
            fold_results[
                "stability_eligible"
            ]
        ]

        expected_index = (
            eligible
            .sort_values(
                by=[
                    "inner_macro_f1_min",
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                    "task_count",
                    "subset_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index[
                0
            ]
        )

        selected_rows = fold_results.loc[
            fold_results[
                "stability_selected"
            ]
        ]

        if len(
            selected_rows
        ) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not have exactly "
                "one selected stability-aware protocol."
            )

        if (
            selected_rows.index[
                0
            ]
            != expected_index
        ):
            raise ValueError(
                f"Outer fold {outer_fold} does not follow the "
                "stability-aware ranking rule."
            )


def build_experiment_record(
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    bootstrap_summary: dict[
        str,
        object,
    ],
) -> dict[
    str,
    object,
]:
    """Build the participant-free tracked experiment summary."""

    selected_macro_f1 = float(
        fold_metrics[
            "outer_test_selected_macro_f1"
        ].mean()
    )
    full_macro_f1 = float(
        fold_metrics[
            "outer_test_full_macro_f1"
        ].mean()
    )

    selected_recordings = float(
        (
            fold_metrics[
                "task_count"
            ]
            * fold_metrics[
                "test_participants"
            ]
        ).sum()
    )
    full_recordings = float(
        EXPECTED_PARTICIPANTS
        * len(
            TASKS
        )
    )

    mean_selected_tasks = (
        selected_recordings
        / EXPECTED_PARTICIPANTS
    )

    fold_columns = [
        column
        for column in fold_metrics.columns
        if column != "selected_tasks"
    ]

    return {
        "experiment": (
            "exploratory_wavlm_stability_aware_selection"
        ),
        "analysis_role": (
            "post_hoc_exploratory_follow_up"
        ),
        "participant_level_fields": [],
        "design": {
            "participants": (
                EXPECTED_PARTICIPANTS
            ),
            "outer_folds": (
                EXPECTED_OUTER_FOLDS
            ),
            "inner_folds_per_outer_fold": (
                EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
            ),
            "pca_components_per_task": (
                PCA_COMPONENTS_PER_TASK
            ),
            "primary_metric": (
                "macro_f1"
            ),
            "competitive_delta": float(
                PRIMARY_DELTA
            ),
            "maximum_selected_tasks": (
                MAX_SELECTED_TASKS
            ),
            "selection_rule": (
                "Among one-to-three-task protocols whose mean inner "
                "Macro F1 is within 0.02 of the full inner-CV "
                "reference, maximize the minimum Macro F1 across the "
                "four inner folds; ties prefer higher mean Macro F1, "
                "lower standard deviation, fewer tasks, and then "
                "lexicographic subset ID."
            ),
            "outer_test_rule": (
                "Each selected protocol is evaluated once on its "
                "outer-test fold after selection using inner-CV "
                "results only."
            ),
        },
        "selection_summary": {
            "selected_protocols": (
                selections
                .sort_values(
                    "outer_fold"
                )[
                    "selected_subset"
                ]
                .tolist()
            ),
            "mean_selected_tasks": float(
                mean_selected_tasks
            ),
            "selected_recordings": (
                selected_recordings
            ),
            "full_recordings": (
                full_recordings
            ),
            "recording_count_reduction_percent": float(
                100.0
                * (
                    1.0
                    - (
                        selected_recordings
                        / full_recordings
                    )
                )
            ),
        },
        "fold_results": (
            fold_metrics[
                fold_columns
            ]
            .sort_values(
                "outer_fold"
            )
            .to_dict(
                orient="records"
            )
        ),
        "selected_vs_full": {
            "selected_macro_f1_mean": (
                selected_macro_f1
            ),
            "full_macro_f1_mean": (
                full_macro_f1
            ),
            "difference_mean": float(
                selected_macro_f1
                - full_macro_f1
            ),
        },
        "paired_bootstrap": (
            bootstrap_summary
        ),
        "inference": {
            "exploratory_noninferiority_criterion_met": bool(
                bootstrap_summary[
                    "one_sided_bound_supports_noninferiority"
                ]
            ),
            "interpretation": (
                "This post hoc stability-aware procedure is an "
                "exploratory follow-up because its design was informed "
                "by earlier outer-test results. Meeting the statistical "
                "criterion would support future independent validation "
                "but would not independently establish noninferiority."
            ),
        },
    }


def save_results(
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    experiment_record: dict[
        str,
        object,
    ],
) -> None:
    """Save generated tables and the participant-free experiment record."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    saved_selections = (
        selections.copy()
    )
    saved_selections[
        "selected_tasks"
    ] = saved_selections[
        "selected_tasks"
    ].map(
        lambda tasks: ",".join(
            tasks
        )
    )
    saved_selections.to_csv(
        SELECTIONS_PATH,
        index=False,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
    bootstrap_results.to_csv(
        BOOTSTRAP_RESULTS_PATH,
        index=False,
    )

    with EXPERIMENT_RECORD_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            convert_json_value(
                experiment_record
            ),
            output_file,
            indent=2,
            sort_keys=True,
        )
        output_file.write(
            "\n"
        )


def print_selection_summary(
    selections: pd.DataFrame,
) -> None:
    """Print the frozen inner-only selections before outer evaluation."""

    print()
    print(
        "Stability-aware inner selection validated."
    )
    print(
        selections[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "selected_inner_macro_f1_min",
                "selected_inner_macro_f1_mean",
                "selected_inner_macro_f1_std",
                "inner_full_reference",
                "competitive_subset_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )


def main() -> None:
    """Run the final exploratory stability-aware WavLM experiment."""

    artifact, extraction_record = (
        load_wavlm_artifact()
    )

    (
        labels,
        nested_folds,
        excluded_ids,
    ) = load_duplicate_filtered_design()

    validate_wavlm_inputs(
        artifact=artifact,
        extraction_record=extraction_record,
        labels=labels,
        nested_folds=nested_folds,
        excluded_ids=excluded_ids,
    )

    inner_results = (
        load_primary_inner_results()
    )

    (
        candidate_results,
        selections,
    ) = select_stability_aware_subsets(
        inner_results
    )

    print_selection_summary(
        selections
    )

    print()
    print(
        "The selection rule is now fixed. "
        "Beginning one outer-test evaluation."
    )

    (
        fold_metrics,
        predictions,
    ) = evaluate_wavlm_outer_folds(
        artifact=artifact,
        labels=labels,
        selections=selections,
        pca_components_per_task=(
            PCA_COMPONENTS_PER_TASK
        ),
    )

    print_wavlm_outer_summary(
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    bootstrap_results = (
        run_paired_wavlm_bootstrap(
            predictions
        )
    )
    bootstrap_summary = (
        summarize_paired_wavlm_bootstrap(
            bootstrap_results=bootstrap_results,
            fold_metrics=fold_metrics,
            margin=PRIMARY_DELTA,
        )
    )

    print_paired_wavlm_bootstrap_summary(
        bootstrap_summary
    )

    experiment_record = (
        build_experiment_record(
            selections=selections,
            fold_metrics=fold_metrics,
            bootstrap_summary=(
                bootstrap_summary
            ),
        )
    )

    save_results(
        candidates=candidate_results,
        selections=selections,
        fold_metrics=fold_metrics,
        predictions=predictions,
        bootstrap_results=bootstrap_results,
        experiment_record=experiment_record,
    )

    print()
    print(
        "Final exploratory experiment saved."
    )
    print(
        EXPERIMENT_RECORD_PATH
    )


if __name__ == "__main__":
    main()
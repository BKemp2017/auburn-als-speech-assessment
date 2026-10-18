"""Prepare duplicate-excluded inputs for the nested sensitivity analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import json

from audit_duplicate_impact import normalize_participant_id
from evaluate_baselines import load_experiment_inputs

from audit_metadata import TASKS
from evaluate_nested_task_selection import (
    PRIMARY_DELTA,
    run_nested_task_selection,
)
from evaluate_task_subsets import (
    generate_task_subsets,
    load_frozen_configurations,
)

WAVEFORM_AUDIT_PATH = Path(
    "results/generated/raw_audio_waveform_audit.csv"
)

PARTICIPANT_ID_FIELDS = (
    "inner_train_ids",
    "inner_validation_ids",
    "outer_test_ids",
)

EXPECTED_ORIGINAL_PARTICIPANTS = 102
EXPECTED_EXCLUDED_PARTICIPANTS = 2
EXPECTED_FILTERED_PARTICIPANTS = 100
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS_PER_OUTER_FOLD = 4
RESULTS_DIRECTORY = Path(
    "results/generated"
)

INNER_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "duplicate_exclusion_inner_results.csv"
)
FOLD_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "duplicate_exclusion_fold_metrics.csv"
)
PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "duplicate_exclusion_predictions.csv"
)
FULL_FOLD_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "duplicate_exclusion_full_fold_metrics.csv"
)
SUMMARY_PATH = Path(
    "experiments/duplicate_exclusion_sensitivity.json"
)
PRIMARY_FOLD_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "nested_task_selection_fold_metrics.csv"
)

OUTER_METRICS = {
    "macro_f1": "outer_test_macro_f1",
    "balanced_accuracy": (
        "outer_test_balanced_accuracy"
    ),
    "roc_auc": "outer_test_roc_auc",
    "sensitivity": "outer_test_sensitivity",
    "specificity": "outer_test_specificity",
    "positive_class_f1": (
        "outer_test_positive_class_f1"
    ),
    "accuracy": "outer_test_accuracy",
}

def normalized_id_set(
    participant_ids,
) -> set[str]:
    """Return participant identifiers in one consistent representation."""

    return {
        normalize_participant_id(
            participant_id
        )
        for participant_id in participant_ids
    }


def identify_duplicate_connected_als_ids(
    waveform_audit_path: Path,
) -> set[str]:
    """Find ALS participants sharing an exact signal with another ALS case."""

    if not waveform_audit_path.exists():
        raise FileNotFoundError(
            "Run src/audit_raw_audio.py before the duplicate-exclusion "
            f"analysis. Missing: {waveform_audit_path}"
        )

    waveform_audit = pd.read_csv(
        waveform_audit_path
    )

    required_columns = {
        "participant_id",
        "category",
        "signal_sha256",
    }
    missing_columns = required_columns.difference(
        waveform_audit.columns
    )

    if missing_columns:
        raise ValueError(
            "Waveform audit is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    waveform_audit = waveform_audit.copy()
    waveform_audit["participant_id"] = (
        waveform_audit["participant_id"].map(
            normalize_participant_id
        )
    )

    duplicate_rows = waveform_audit.loc[
        waveform_audit.duplicated(
            subset="signal_sha256",
            keep=False,
        )
    ]

    duplicate_als_ids: set[str] = set()

    # A hash only affects the ALS-only experiment when at least two ALS
    # participants share that exact recording.
    for _, group in duplicate_rows.groupby(
        "signal_sha256",
        sort=True,
    ):
        als_ids = normalized_id_set(
            group.loc[
                group["category"].eq("ALS"),
                "participant_id",
            ]
        )

        if len(als_ids) >= 2:
            duplicate_als_ids.update(
                als_ids
            )

    if len(duplicate_als_ids) != (
        EXPECTED_EXCLUDED_PARTICIPANTS
    ):
        raise ValueError(
            "Expected to identify exactly "
            f"{EXPECTED_EXCLUDED_PARTICIPANTS} duplicate-connected ALS "
            f"participants, but found {len(duplicate_als_ids)}."
        )

    return duplicate_als_ids


def filter_modeling_inputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    excluded_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate-connected participants from frozen model inputs."""

    feature_ids = features["ID"].map(
        normalize_participant_id
    )
    label_ids = labels["participant_id"].map(
        normalize_participant_id
    )

    filtered_features = features.loc[
        ~feature_ids.isin(
            excluded_ids
        )
    ].copy()

    filtered_labels = labels.loc[
        ~label_ids.isin(
            excluded_ids
        )
    ].copy()

    filtered_features = filtered_features.reset_index(
        drop=True
    )
    filtered_labels = filtered_labels.reset_index(
        drop=True
    )

    return (
        filtered_features,
        filtered_labels,
    )


def filter_nested_folds(
    nested_folds: list[dict[str, object]],
    excluded_ids: set[str],
) -> list[dict[str, object]]:
    """Remove excluded participants while preserving frozen fold membership."""

    filtered_assignments: list[
        dict[str, object]
    ] = []

    for assignment in nested_folds:
        filtered_assignment = dict(
            assignment
        )

        for field in PARTICIPANT_ID_FIELDS:
            if field not in assignment:
                raise ValueError(
                    f"Nested-fold assignment is missing {field}."
                )

            # Retain the original identifier type so downstream evaluation
            # functions continue to receive the same representation.
            filtered_assignment[field] = [
                participant_id
                for participant_id in assignment[field]
                if normalize_participant_id(
                    participant_id
                )
                not in excluded_ids
            ]

        filtered_assignments.append(
            filtered_assignment
        )

    return filtered_assignments


def validate_filtered_inputs(
    original_features: pd.DataFrame,
    filtered_features: pd.DataFrame,
    filtered_labels: pd.DataFrame,
    filtered_nested_folds: list[dict[str, object]],
    excluded_ids: set[str],
) -> None:
    """Confirm the filtered folds remain participant-disjoint and usable."""

    if len(original_features) != (
        EXPECTED_ORIGINAL_PARTICIPANTS
    ):
        raise ValueError(
            "Expected the original feature table to contain "
            f"{EXPECTED_ORIGINAL_PARTICIPANTS} participants, "
            f"but found {len(original_features)}."
        )

    if len(filtered_features) != (
        EXPECTED_FILTERED_PARTICIPANTS
    ):
        raise ValueError(
            "Expected the filtered feature table to contain "
            f"{EXPECTED_FILTERED_PARTICIPANTS} participants, "
            f"but found {len(filtered_features)}."
        )

    if len(filtered_labels) != (
        EXPECTED_FILTERED_PARTICIPANTS
    ):
        raise ValueError(
            "Expected the filtered label table to contain "
            f"{EXPECTED_FILTERED_PARTICIPANTS} participants, "
            f"but found {len(filtered_labels)}."
        )

    feature_ids = normalized_id_set(
        filtered_features["ID"]
    )
    label_ids = normalized_id_set(
        filtered_labels["participant_id"]
    )

    if feature_ids != label_ids:
        raise ValueError(
            "Filtered feature and label tables contain different "
            "participants."
        )

    if feature_ids.intersection(
        excluded_ids
    ):
        raise ValueError(
            "An excluded participant remains in the filtered features."
        )

    if filtered_features["ID"].duplicated().any():
        raise ValueError(
            "Filtered feature table contains duplicate participant IDs."
        )

    if filtered_labels[
        "participant_id"
    ].duplicated().any():
        raise ValueError(
            "Filtered label table contains duplicate participant IDs."
        )

    expected_assignment_count = (
        EXPECTED_OUTER_FOLDS
        * EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    )

    if len(filtered_nested_folds) != (
        expected_assignment_count
    ):
        raise ValueError(
            f"Expected {expected_assignment_count} nested-fold "
            f"assignments, but found {len(filtered_nested_folds)}."
        )

    target_by_id = {
        normalize_participant_id(
            participant_id
        ): int(target)
        for participant_id, target in zip(
            filtered_labels["participant_id"],
            filtered_labels["target"],
            strict=True,
        )
    }

    outer_test_ids_by_fold: dict[
        int,
        set[str],
    ] = {}

    for assignment in filtered_nested_folds:
        outer_fold = int(
            assignment["outer_fold"]
        )

        inner_train_ids = normalized_id_set(
            assignment["inner_train_ids"]
        )
        inner_validation_ids = normalized_id_set(
            assignment["inner_validation_ids"]
        )
        outer_test_ids = normalized_id_set(
            assignment["outer_test_ids"]
        )

        assignment_ids = (
            inner_train_ids
            | inner_validation_ids
            | outer_test_ids
        )

        if assignment_ids.intersection(
            excluded_ids
        ):
            raise ValueError(
                "An excluded participant remains in a filtered fold."
            )

        if inner_train_ids.intersection(
            inner_validation_ids
        ):
            raise ValueError(
                "Inner training and validation participants overlap."
            )

        if inner_train_ids.intersection(
            outer_test_ids
        ):
            raise ValueError(
                "Inner training and outer-test participants overlap."
            )

        if inner_validation_ids.intersection(
            outer_test_ids
        ):
            raise ValueError(
                "Inner validation and outer-test participants overlap."
            )

        if assignment_ids != feature_ids:
            raise ValueError(
                "A filtered nested-fold assignment does not cover every "
                "remaining participant exactly once."
            )

        if len(
            {
                target_by_id[
                    participant_id
                ]
                for participant_id
                in inner_train_ids
            }
        ) != 2:
            raise ValueError(
                "An inner training split does not contain both targets."
            )

        if len(
            {
                target_by_id[
                    participant_id
                ]
                for participant_id
                in inner_validation_ids
            }
        ) != 2:
            raise ValueError(
                "An inner validation split does not contain both targets."
            )

        if len(
            {
                target_by_id[
                    participant_id
                ]
                for participant_id
                in outer_test_ids
            }
        ) != 2:
            raise ValueError(
                "An outer-test split does not contain both targets."
            )

        if outer_fold in outer_test_ids_by_fold:
            if (
                outer_test_ids_by_fold[
                    outer_fold
                ]
                != outer_test_ids
            ):
                raise ValueError(
                    "Outer-test membership changes across inner folds."
                )
        else:
            outer_test_ids_by_fold[
                outer_fold
            ] = outer_test_ids

    if len(outer_test_ids_by_fold) != (
        EXPECTED_OUTER_FOLDS
    ):
        raise ValueError(
            "Filtered assignments do not contain five outer folds."
        )

    seen_outer_test_ids: set[str] = set()

    for outer_fold in sorted(
        outer_test_ids_by_fold
    ):
        outer_test_ids = (
            outer_test_ids_by_fold[
                outer_fold
            ]
        )

        if seen_outer_test_ids.intersection(
            outer_test_ids
        ):
            raise ValueError(
                "A participant appears in more than one outer-test fold."
            )

        seen_outer_test_ids.update(outer_test_ids)

    if seen_outer_test_ids != feature_ids:
        raise ValueError(
            "The outer-test folds do not cover all remaining participants."
        )


def print_filtered_input_summary(
    original_features: pd.DataFrame,
    filtered_features: pd.DataFrame,
    filtered_labels: pd.DataFrame,
    filtered_nested_folds: list[dict[str, object]],
) -> None:
    """Print aggregate validation details without participant identifiers."""

    print("Duplicate-exclusion sensitivity inputs validated.")
    print(
        "Original participants:",
        len(original_features),
    )
    print(
        "Excluded participants:",
        len(original_features)
        - len(filtered_features),
    )
    print(
        "Remaining participants:",
        len(filtered_features),
    )

    print("\nRemaining target distribution:")
    print(
        filtered_labels["target"]
        .value_counts()
        .sort_index()
        .rename_axis("target")
        .reset_index(name="participants")
        .to_string(index=False)
    )

    fold_rows: list[dict[str, int]] = []

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        assignments = [
            assignment
            for assignment
            in filtered_nested_folds
            if int(
                assignment["outer_fold"]
            )
            == outer_fold
        ]

        outer_test_count = len(
            assignments[0][
                "outer_test_ids"
            ]
        )

        inner_train_counts = [
            len(
                assignment[
                    "inner_train_ids"
                ]
            )
            for assignment in assignments
        ]
        inner_validation_counts = [
            len(
                assignment[
                    "inner_validation_ids"
                ]
            )
            for assignment in assignments
        ]

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_test_participants": (
                    outer_test_count
                ),
                "minimum_inner_train": min(
                    inner_train_counts
                ),
                "maximum_inner_train": max(
                    inner_train_counts
                ),
                "minimum_inner_validation": min(
                    inner_validation_counts
                ),
                "maximum_inner_validation": max(
                    inner_validation_counts
                ),
            }
        )

    print("\nFiltered fold sizes:")
    print(
        pd.DataFrame(
            fold_rows
        ).to_string(
            index=False
        )
    )

def run_duplicate_excluded_full_baseline(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete eight-task baseline on the filtered cohort."""

    frozen_configurations = (
        load_frozen_configurations()
    )

    outer_fold_values = sorted(
        int(value)
        for value in outer_folds[
            "outer_fold"
        ].unique()
    )

    if set(outer_fold_values) != set(
        frozen_configurations
    ):
        raise ValueError(
            "Frozen configurations do not match the filtered outer folds."
        )

    # Using only the full subset exercises the same inner-selection and
    # outer-evaluation path without launching all 255 task combinations.
    return run_nested_task_selection(
        features=features,
        outer_folds=outer_folds,
        nested_folds=nested_folds,
        subsets=[
            tuple(TASKS),
        ],
        frozen_configurations=(
            frozen_configurations
        ),
        delta=PRIMARY_DELTA,
    )

def validate_duplicate_excluded_full_baseline(
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    filtered_labels: pd.DataFrame,
) -> None:
    """Validate the filtered full-assessment execution-path test."""

    if len(inner_results) != (
        EXPECTED_OUTER_FOLDS
    ):
        raise ValueError(
            "Expected one full-assessment inner result per outer fold, "
            f"but found {len(inner_results)} rows."
        )

    if len(fold_metrics) != (
        EXPECTED_OUTER_FOLDS
    ):
        raise ValueError(
            f"Expected {EXPECTED_OUTER_FOLDS} fold-metric rows, "
            f"but found {len(fold_metrics)}."
        )

    if len(predictions) != (
        EXPECTED_FILTERED_PARTICIPANTS
    ):
        raise ValueError(
            "Expected one prediction per remaining participant, "
            f"but found {len(predictions)}."
        )

    if predictions[
        "participant_id"
    ].duplicated().any():
        raise ValueError(
            "A participant received more than one outer-test prediction."
        )

    expected_ids = normalized_id_set(
        filtered_labels["participant_id"]
    )
    predicted_ids = normalized_id_set(
        predictions["participant_id"]
    )

    if predicted_ids != expected_ids:
        raise ValueError(
            "Full-baseline predictions do not cover the filtered cohort."
        )

    expected_fold_by_id = {
        normalize_participant_id(
            participant_id
        ): int(outer_fold)
        for participant_id, outer_fold in zip(
            filtered_labels["participant_id"],
            filtered_labels["outer_fold"],
            strict=True,
        )
    }

    for prediction in predictions.itertuples(
        index=False
    ):
        participant_id = normalize_participant_id(
            prediction.participant_id
        )

        if int(
            prediction.outer_fold
        ) != expected_fold_by_id[
            participant_id
        ]:
            raise ValueError(
                "A prediction was assigned to the wrong outer fold."
            )

    if not fold_metrics[
        "task_count"
    ].eq(len(TASKS)).all():
        raise ValueError(
            "The filtered full baseline did not use all eight tasks."
        )

    expected_subset_id = "+".join(
        TASKS
    )

    if not fold_metrics[
        "selected_subset"
    ].eq(expected_subset_id).all():
        raise ValueError(
            "The filtered full baseline selected something other than "
            "the complete eight-task assessment."
        )

    for table_name, table in (
        (
            "inner results",
            inner_results,
        ),
        (
            "fold metrics",
            fold_metrics,
        ),
        (
            "predictions",
            predictions,
        ),
    ):
        numeric_values = table.select_dtypes(
            include="number"
        ).to_numpy()

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                f"The filtered {table_name} contain a nonfinite value."
            )

    expected_outer_folds = set(
        range(
            EXPECTED_OUTER_FOLDS
        )
    )

    if set(
        fold_metrics["outer_fold"].astype(
            int
        )
    ) != expected_outer_folds:
        raise ValueError(
            "Full-baseline fold metrics do not cover all outer folds."
        )

    print()
    print("Duplicate-excluded full-assessment baseline validated.")
    print(
        "Inner-result rows:",
        len(inner_results),
    )
    print(
        "Fold-metric rows:",
        len(fold_metrics),
    )
    print(
        "Prediction rows:",
        len(predictions),
    )

    print("\nFull-assessment Macro F1 by outer fold:")
    print(
        fold_metrics[
            [
                "outer_fold",
                "outer_test_macro_f1",
            ]
        ]
        .sort_values("outer_fold")
        .to_string(index=False)
    )

    print(
        "\nFull-assessment mean Macro F1:",
        f"{fold_metrics['outer_test_macro_f1'].mean():.6f}",
    )

def run_duplicate_excluded_nested_selection(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all 255 task subsets after duplicate exclusion."""

    subsets = generate_task_subsets()
    frozen_configurations = (
        load_frozen_configurations()
    )

    if len(subsets) != 255:
        raise ValueError(
            f"Expected 255 task subsets, but found {len(subsets)}."
        )

    if subsets[-1] != tuple(TASKS):
        raise ValueError(
            "The final generated subset is not the full assessment."
        )

    return run_nested_task_selection(
        features=features,
        outer_folds=outer_folds,
        nested_folds=nested_folds,
        subsets=subsets,
        frozen_configurations=(
            frozen_configurations
        ),
        delta=PRIMARY_DELTA,
    )


def validate_duplicate_excluded_nested_results(
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    filtered_labels: pd.DataFrame,
) -> None:
    """Validate the complete duplicate-excluded nested experiment."""

    expected_inner_rows = (
        EXPECTED_OUTER_FOLDS * 255
    )

    if len(inner_results) != expected_inner_rows:
        raise ValueError(
            f"Expected {expected_inner_rows} inner-result rows, "
            f"but found {len(inner_results)}."
        )

    if len(fold_metrics) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            f"Expected {EXPECTED_OUTER_FOLDS} fold-metric rows, "
            f"but found {len(fold_metrics)}."
        )

    if len(predictions) != (
        EXPECTED_FILTERED_PARTICIPANTS
    ):
        raise ValueError(
            "Expected one prediction per remaining participant, "
            f"but found {len(predictions)}."
        )

    expected_ids = normalized_id_set(
        filtered_labels["participant_id"]
    )
    predicted_ids = normalized_id_set(
        predictions["participant_id"]
    )

    if predicted_ids != expected_ids:
        raise ValueError(
            "Nested predictions do not cover the filtered cohort."
        )

    if predictions[
        "participant_id"
    ].duplicated().any():
        raise ValueError(
            "A participant received more than one nested prediction."
        )

    target_by_id = {
        normalize_participant_id(
            participant_id
        ): int(target)
        for participant_id, target in zip(
            filtered_labels["participant_id"],
            filtered_labels["target"],
            strict=True,
        )
    }

    fold_by_id = {
        normalize_participant_id(
            participant_id
        ): int(outer_fold)
        for participant_id, outer_fold in zip(
            filtered_labels["participant_id"],
            filtered_labels["outer_fold"],
            strict=True,
        )
    }

    for prediction in predictions.itertuples(
        index=False
    ):
        participant_id = normalize_participant_id(
            prediction.participant_id
        )

        if int(
            prediction.outer_fold
        ) != fold_by_id[participant_id]:
            raise ValueError(
                "A prediction was assigned to the wrong outer fold."
            )

        if int(
            prediction.true_label
        ) != target_by_id[participant_id]:
            raise ValueError(
                "A prediction contains the wrong true target."
            )

    expected_outer_folds = set(
        range(
            EXPECTED_OUTER_FOLDS
        )
    )

    if set(
        fold_metrics["outer_fold"].astype(
            int
        )
    ) != expected_outer_folds:
        raise ValueError(
            "Fold metrics do not cover all five outer folds."
        )

    selected_rows = inner_results.loc[
        inner_results["selected"]
    ]

    if len(selected_rows) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Expected exactly one selected inner result per outer fold."
        )

    for outer_fold in sorted(
        expected_outer_folds
    ):
        fold_inner_results = inner_results.loc[
            inner_results["outer_fold"].eq(
                outer_fold
            )
        ]

        if len(fold_inner_results) != 255:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain 255 subsets."
            )

        if (
            fold_inner_results[
                "subset_id"
            ].nunique()
            != 255
        ):
            raise ValueError(
                f"Outer fold {outer_fold} contains duplicate subsets."
            )

        fold_metric = fold_metrics.loc[
            fold_metrics["outer_fold"].eq(
                outer_fold
            )
        ].iloc[0]

        selected_subset = str(
            fold_metric["selected_subset"]
        )

        fold_selected_rows = (
            fold_inner_results.loc[
                fold_inner_results[
                    "selected"
                ]
            ]
        )

        if len(fold_selected_rows) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not have one selection."
            )

        if str(
            fold_selected_rows[
                "subset_id"
            ].iloc[0]
        ) != selected_subset:
            raise ValueError(
                "The inner selected subset does not match the "
                "outer-fold metric."
            )

        prediction_subsets = set(
            predictions.loc[
                predictions[
                    "outer_fold"
                ].eq(outer_fold),
                "selected_subset",
            ].astype(str)
        )

        if prediction_subsets != {
            selected_subset
        }:
            raise ValueError(
                "Prediction rows do not match the selected fold protocol."
            )

    for table_name, table in (
        (
            "inner results",
            inner_results,
        ),
        (
            "fold metrics",
            fold_metrics,
        ),
        (
            "predictions",
            predictions,
        ),
    ):
        numeric_values = table.select_dtypes(
            include="number"
        ).to_numpy()

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                f"The sensitivity {table_name} contain a nonfinite value."
            )

    print()
    print(
        "Duplicate-excluded nested task selection validated."
    )
    print(
        "Inner-result rows:",
        len(inner_results),
    )
    print(
        "Fold-metric rows:",
        len(fold_metrics),
    )
    print(
        "Prediction rows:",
        len(predictions),
    )


def summarize_metric_table(
    fold_metrics: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Summarize outer-test metrics across the five folds."""

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric_name, column in (
        OUTER_METRICS.items()
    ):
        summary[metric_name] = {
            "mean": float(
                fold_metrics[column].mean()
            ),
            "std": float(
                fold_metrics[column].std(
                    ddof=1
                )
            ),
        }

    return summary


def build_duplicate_exclusion_summary(
    fold_metrics: pd.DataFrame,
    full_fold_metrics: pd.DataFrame,
    primary_fold_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Build the tracked participant-free sensitivity record."""

    selected_vs_full = (
        fold_metrics[
            [
                "outer_fold",
                "outer_test_macro_f1",
            ]
        ]
        .rename(
            columns={
                "outer_test_macro_f1": (
                    "nested_selected_macro_f1"
                ),
            }
        )
        .merge(
            full_fold_metrics[
                [
                    "outer_fold",
                    "outer_test_macro_f1",
                ]
            ].rename(
                columns={
                    "outer_test_macro_f1": (
                        "full_macro_f1"
                    ),
                }
            ),
            on="outer_fold",
            validate="one_to_one",
        )
        .sort_values("outer_fold")
    )

    selected_vs_full["difference"] = (
        selected_vs_full[
            "nested_selected_macro_f1"
        ]
        - selected_vs_full[
            "full_macro_f1"
        ]
    )

    primary_comparison = (
        primary_fold_metrics[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "outer_test_macro_f1",
            ]
        ]
        .rename(
            columns={
                "selected_subset": (
                    "primary_selected_subset"
                ),
                "task_count": (
                    "primary_task_count"
                ),
                "outer_test_macro_f1": (
                    "primary_macro_f1"
                ),
            }
        )
        .merge(
            fold_metrics[
                [
                    "outer_fold",
                    "selected_subset",
                    "task_count",
                    "outer_test_macro_f1",
                ]
            ].rename(
                columns={
                    "selected_subset": (
                        "sensitivity_selected_subset"
                    ),
                    "task_count": (
                        "sensitivity_task_count"
                    ),
                    "outer_test_macro_f1": (
                        "sensitivity_macro_f1"
                    ),
                }
            ),
            on="outer_fold",
            validate="one_to_one",
        )
        .sort_values("outer_fold")
    )

    primary_comparison["macro_f1_difference"] = (
        primary_comparison[
            "sensitivity_macro_f1"
        ]
        - primary_comparison[
            "primary_macro_f1"
        ]
    )

    selection_frequency = {
        str(subset_id): int(count)
        for subset_id, count in (
            fold_metrics["selected_subset"]
            .value_counts()
            .items()
        )
    }

    task_count_distribution = {
        str(int(task_count)): int(count)
        for task_count, count in (
            fold_metrics["task_count"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    task_frequency = {
        task: int(
            fold_metrics[
                "selected_subset"
            ]
            .astype(str)
            .map(
                lambda subset_id: (
                    task
                    in subset_id.split("+")
                )
            )
            .sum()
        )
        for task in TASKS
    }

    fold_selections = json.loads(
        fold_metrics[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "selected_inner_macro_f1_mean",
                "inner_full_reference",
                "outer_test_macro_f1",
            ]
        ]
        .sort_values("outer_fold")
        .to_json(
            orient="records"
        )
    )

    return {
        "experiment": (
            "duplicate_exclusion_sensitivity"
        ),
        "participant_level_fields": [],
        "primary_delta": PRIMARY_DELTA,
        "design": {
            "original_participants": (
                EXPECTED_ORIGINAL_PARTICIPANTS
            ),
            "excluded_participants": (
                EXPECTED_EXCLUDED_PARTICIPANTS
            ),
            "remaining_participants": (
                EXPECTED_FILTERED_PARTICIPANTS
            ),
            "exclusion_reason": (
                "Two ALS participants shared an exact A-task "
                "recording, had contradictory targets, appeared in "
                "different outer folds, and had identical A-task "
                "handcrafted predictors."
            ),
            "fold_policy": (
                "Preserve the frozen outer and inner assignments after "
                "filtering the two affected participants."
            ),
        },
        "fold_selections": fold_selections,
        "selection_frequency": (
            selection_frequency
        ),
        "task_count_distribution": (
            task_count_distribution
        ),
        "task_frequency": task_frequency,
        "nested_selected_outer_test_metrics": (
            summarize_metric_table(
                fold_metrics
            )
        ),
        "full_eight_task_outer_test_metrics": (
            summarize_metric_table(
                full_fold_metrics
            )
        ),
        "filtered_selected_vs_full": {
            "nested_selected_macro_f1_mean": float(
                fold_metrics[
                    "outer_test_macro_f1"
                ].mean()
            ),
            "full_macro_f1_mean": float(
                full_fold_metrics[
                    "outer_test_macro_f1"
                ].mean()
            ),
            "difference_mean": float(
                selected_vs_full[
                    "difference"
                ].mean()
            ),
            "fold_comparisons": json.loads(
                selected_vs_full.to_json(
                    orient="records"
                )
            ),
        },
        "comparison_with_primary_analysis": {
            "primary_macro_f1_mean": float(
                primary_fold_metrics[
                    "outer_test_macro_f1"
                ].mean()
            ),
            "sensitivity_macro_f1_mean": float(
                fold_metrics[
                    "outer_test_macro_f1"
                ].mean()
            ),
            "difference_mean": float(
                primary_comparison[
                    "macro_f1_difference"
                ].mean()
            ),
            "fold_comparisons": json.loads(
                primary_comparison.to_json(
                    orient="records"
                )
            ),
        },
    }

def save_duplicate_exclusion_results(
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    full_fold_metrics: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    """Save generated details and the tracked aggregate record."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inner_results.to_csv(
        INNER_RESULTS_PATH,
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
    full_fold_metrics.to_csv(
        FULL_FOLD_METRICS_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )
        output_file.write("\n")

def main() -> None:
    """Prepare and validate the duplicate-excluded sensitivity inputs."""

    (
        original_features,
        original_labels,
        nested_folds,
    ) = load_experiment_inputs()

    excluded_ids = (
        identify_duplicate_connected_als_ids(
            WAVEFORM_AUDIT_PATH
        )
    )

    (
        filtered_features,
        filtered_labels,
    ) = filter_modeling_inputs(
        features=original_features,
        labels=original_labels,
        excluded_ids=excluded_ids,
    )

    filtered_nested_folds = filter_nested_folds(
        nested_folds=nested_folds,
        excluded_ids=excluded_ids,
    )

    validate_filtered_inputs(
        original_features=original_features,
        filtered_features=filtered_features,
        filtered_labels=filtered_labels,
        filtered_nested_folds=filtered_nested_folds,
        excluded_ids=excluded_ids,
    )

    print_filtered_input_summary(
        original_features=original_features,
        filtered_features=filtered_features,
        filtered_labels=filtered_labels,
        filtered_nested_folds=filtered_nested_folds,
    )

    (
        full_inner_results,
        full_fold_metrics,
        full_predictions,
    ) = run_duplicate_excluded_full_baseline(
        features=filtered_features,
        outer_folds=filtered_labels,
        nested_folds=filtered_nested_folds,
    )

    validate_duplicate_excluded_full_baseline(
        inner_results=full_inner_results,
        fold_metrics=full_fold_metrics,
        predictions=full_predictions,
        filtered_labels=filtered_labels,
    )
    (
        inner_results,
        fold_metrics,
        predictions,
    ) = run_duplicate_excluded_nested_selection(
        features=filtered_features,
        outer_folds=filtered_labels,
        nested_folds=filtered_nested_folds,
    )

    validate_duplicate_excluded_nested_results(
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        filtered_labels=filtered_labels,
    )

    if not PRIMARY_FOLD_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Run the primary nested experiment before the sensitivity "
            f"analysis. Missing: {PRIMARY_FOLD_METRICS_PATH}"
        )

    primary_fold_metrics = pd.read_csv(
        PRIMARY_FOLD_METRICS_PATH
    )

    summary = build_duplicate_exclusion_summary(
        fold_metrics=fold_metrics,
        full_fold_metrics=full_fold_metrics,
        primary_fold_metrics=(
            primary_fold_metrics
        ),
    )

    save_duplicate_exclusion_results(
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        full_fold_metrics=full_fold_metrics,
        summary=summary,
    )

    print("\nSelected protocol by outer fold:")
    print(
        fold_metrics[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "selected_inner_macro_f1_mean",
                "inner_full_reference",
                "outer_test_macro_f1",
            ]
        ]
        .sort_values("outer_fold")
        .to_string(index=False)
    )

    selected_mean = float(
        fold_metrics[
            "outer_test_macro_f1"
        ].mean()
    )
    full_mean = float(
        full_fold_metrics[
            "outer_test_macro_f1"
        ].mean()
    )
    primary_mean = float(
        primary_fold_metrics[
            "outer_test_macro_f1"
        ].mean()
    )

    print("\nMacro F1 comparison:")
    print(
        "Duplicate-excluded selected mean:",
        f"{selected_mean:.6f}",
    )
    print(
        "Duplicate-excluded full mean:",
        f"{full_mean:.6f}",
    )
    print(
        "Original primary selected mean:",
        f"{primary_mean:.6f}",
    )
    print(
        "Selected minus filtered full:",
        f"{selected_mean - full_mean:+.6f}",
    )
    print(
        "Sensitivity minus primary:",
        f"{selected_mean - primary_mean:+.6f}",
    )

    print("\nGenerated artifacts:")
    print(INNER_RESULTS_PATH)
    print(FOLD_METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(FULL_FOLD_METRICS_PATH)
    print(SUMMARY_PATH)

if __name__ == "__main__":
    main()
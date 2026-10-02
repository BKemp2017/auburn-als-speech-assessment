"""Evaluate how the nested task selection changes across delta values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from audit_metadata import TASKS
from evaluate_baselines import (
    load_experiment_inputs,
    validate_experiment_inputs,
)
from evaluate_nested_task_selection import (
    FOLD_METRICS_PATH,
    INNER_SUBSET_RESULTS_PATH,
    OUTPUT_DIR,
    PREDICTIONS_PATH,
    PRIMARY_DELTA,
    evaluate_selected_subset_outer_fold,
    select_competitive_subset,
)
from evaluate_task_subsets import load_frozen_configurations


# Delta=0.02 remains the locked primary analysis. The other values are
# sensitivity checks and must not be chosen based on outer-test performance.
DELTAS = (0.01, PRIMARY_DELTA, 0.05)
OUTER_FOLDS = tuple(range(5))
SUBSETS_PER_FOLD = 255
DELTA_SELECTIONS_PATH = (
    OUTPUT_DIR / "delta_sensitivity_selections.csv"
)
DELTA_FOLD_METRICS_PATH = (
    OUTPUT_DIR / "delta_sensitivity_fold_metrics.csv"
)
DELTA_PREDICTIONS_PATH = (
    OUTPUT_DIR / "delta_sensitivity_predictions.csv"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/delta_sensitivity.json"
)

OUTER_METRIC_COLUMNS = [
    "outer_test_macro_f1",
    "outer_test_balanced_accuracy",
    "outer_test_roc_auc",
    "outer_test_sensitivity",
    "outer_test_specificity",
    "outer_test_positive_class_f1",
    "outer_test_accuracy",
]

PRIMARY_EVALUATION_REQUIRED_COLUMNS = {
    "outer_fold",
    "selected_subset",
    "selected_tasks",
    "task_count",
    "predictor_count",
    "training_participants",
    "test_participants",
    "inner_full_reference",
    "selected_inner_macro_f1_mean",
    "selected_inner_macro_f1_std",
    "competitive_threshold",
    "competitive_subset_count",
    "delta",
    "configuration",
    *OUTER_METRIC_COLUMNS,
}

PRIMARY_PREDICTION_REQUIRED_COLUMNS = {
    "participant_id",
    "outer_fold",
    "selected_subset",
    "true_label",
    "predicted_label",
    "score",
}

INNER_REQUIRED_COLUMNS = {
    "outer_fold",
    "tasks",
    "subset_id",
    "task_count",
    "predictor_count",
    "inner_macro_f1_mean",
    "inner_macro_f1_std",
}

PRIMARY_REQUIRED_COLUMNS = {
    "outer_fold",
    "selected_subset",
    "task_count",
    "delta",
}


def require_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Confirm that a saved result table contains the fields we reuse."""

    missing_columns = required_columns - set(table.columns)

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def parse_saved_tasks(value: object) -> tuple[str, ...]:
    """Convert a saved comma-separated task list back to a tuple."""

    return tuple(str(value).split(","))


def load_selection_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the saved nested-selection results."""

    if not INNER_SUBSET_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Nested inner-CV results were not found at "
            f"{INNER_SUBSET_RESULTS_PATH}."
        )

    if not FOLD_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Nested fold metrics were not found at "
            f"{FOLD_METRICS_PATH}."
        )

    inner_results = pd.read_csv(INNER_SUBSET_RESULTS_PATH)
    primary_metrics = pd.read_csv(FOLD_METRICS_PATH)

    require_columns(
        inner_results,
        INNER_REQUIRED_COLUMNS,
        "Nested inner-CV results",
    )
    require_columns(
        primary_metrics,
        PRIMARY_REQUIRED_COLUMNS,
        "Primary nested fold metrics",
    )

    expected_inner_rows = len(OUTER_FOLDS) * SUBSETS_PER_FOLD

    if len(inner_results) != expected_inner_rows:
        raise ValueError(
            "Expected "
            f"{expected_inner_rows} inner-result rows, "
            f"but found {len(inner_results)}."
        )

    if len(primary_metrics) != len(OUTER_FOLDS):
        raise ValueError(
            f"Expected {len(OUTER_FOLDS)} primary fold rows, "
            f"but found {len(primary_metrics)}."
        )

    expected_folds = set(OUTER_FOLDS)

    for table, table_name in (
        (inner_results, "Nested inner-CV results"),
        (primary_metrics, "Primary nested fold metrics"),
    ):
        recorded_folds = set(table["outer_fold"].astype(int).unique())

        if recorded_folds != expected_folds:
            raise ValueError(
                f"{table_name} contains outer folds "
                f"{sorted(recorded_folds)}, expected "
                f"{sorted(expected_folds)}."
            )

    fold_counts = inner_results.groupby("outer_fold").agg(
        rows=("subset_id", "size"),
        unique_subsets=("subset_id", "nunique"),
    )

    if not fold_counts["rows"].eq(SUBSETS_PER_FOLD).all():
        raise ValueError(
            "Each outer fold must contain exactly "
            f"{SUBSETS_PER_FOLD} inner-result rows."
        )

    if not fold_counts["unique_subsets"].eq(SUBSETS_PER_FOLD).all():
        raise ValueError(
            "Each outer fold must contain exactly "
            f"{SUBSETS_PER_FOLD} unique task subsets."
        )

    numeric_columns = [
        "task_count",
        "predictor_count",
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
    ]

    if not np.isfinite(
        inner_results[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nested inner-CV results contain non-finite values."
        )

    if not np.allclose(
        primary_metrics["delta"].to_numpy(dtype=float),
        PRIMARY_DELTA,
    ):
        raise ValueError(
            "Primary fold metrics do not consistently use the "
            f"locked delta of {PRIMARY_DELTA}."
        )

    # Parse the task field once so every later selection uses the same
    # validated tuple representation.
    inner_results = inner_results.copy()
    inner_results["tasks"] = inner_results["tasks"].map(
        parse_saved_tasks
    )

    valid_tasks = inner_results["tasks"].map(
        lambda tasks: (
            len(tasks) > 0
            and len(tasks) == len(set(tasks))
            and set(tasks).issubset(TASKS)
        )
    )

    if not valid_tasks.all():
        raise ValueError(
            "At least one inner-result row contains an invalid task list."
        )

    subset_ids_match = inner_results.apply(
        lambda row: "+".join(row["tasks"]) == row["subset_id"],
        axis=1,
    )

    if not subset_ids_match.all():
        raise ValueError(
            "At least one subset ID does not match its saved task list."
        )

    task_counts_match = inner_results["tasks"].map(len).eq(
        inner_results["task_count"]
    )

    if not task_counts_match.all():
        raise ValueError(
            "At least one task count does not match its saved task list."
        )

    # The full assessment provides the reference score used to define each
    # competitive threshold, so every fold must contain it exactly once.
    full_subset = tuple(TASKS)
    full_subset_counts = (
        inner_results["tasks"]
        .map(lambda tasks: tasks == full_subset)
        .groupby(inner_results["outer_fold"])
        .sum()
    )

    if not full_subset_counts.eq(1).all():
        raise ValueError(
            "Each outer fold must contain the full eight-task subset once."
        )

    return inner_results, primary_metrics


def build_selection_table(
    inner_results: pd.DataFrame,
) -> pd.DataFrame:
    """Reapply the selection rule to every fold and delta value."""

    candidate_columns = [
        "outer_fold",
        "tasks",
        "subset_id",
        "task_count",
        "predictor_count",
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
    ]

    selection_rows: list[dict[str, object]] = []

    # The inner-CV scores never change across delta values. Delta changes only
    # the competitive threshold and therefore which protocol is selected.
    for outer_fold, fold_rows in inner_results.groupby(
        "outer_fold",
        sort=True,
    ):
        candidates = fold_rows[candidate_columns].to_dict(
            orient="records"
        )

        for delta in DELTAS:
            selected = select_competitive_subset(
                candidates,
                delta=delta,
            )

            selection_rows.append(
                {
                    "delta": float(delta),
                    "outer_fold": int(outer_fold),
                    "selected_subset": selected["subset_id"],
                    "selected_tasks": ",".join(selected["tasks"]),
                    "task_count": int(selected["task_count"]),
                    "predictor_count": int(
                        selected["predictor_count"]
                    ),
                    "selected_inner_macro_f1_mean": float(
                        selected["inner_macro_f1_mean"]
                    ),
                    "selected_inner_macro_f1_std": float(
                        selected["inner_macro_f1_std"]
                    ),
                    "inner_full_reference": float(
                        selected["inner_full_reference"]
                    ),
                    "competitive_threshold": float(
                        selected["competitive_threshold"]
                    ),
                    "competitive_subset_count": int(
                        selected["competitive_subset_count"]
                    ),
                }
            )

    return (
        pd.DataFrame(selection_rows)
        .sort_values(["delta", "outer_fold"])
        .reset_index(drop=True)
    )


def validate_primary_reproduction(
    selections: pd.DataFrame,
    primary_metrics: pd.DataFrame,
) -> None:
    """Verify that delta=0.02 reproduces the locked primary selections."""

    observed = (
        selections.loc[
            np.isclose(selections["delta"], PRIMARY_DELTA),
            ["outer_fold", "selected_subset", "task_count"],
        ]
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    expected = (
        primary_metrics[
            ["outer_fold", "selected_subset", "task_count"]
        ]
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    if observed["outer_fold"].tolist() != expected["outer_fold"].tolist():
        raise ValueError(
            "Primary reproduction contains unexpected outer folds."
        )

    if (
        observed["selected_subset"].tolist()
        != expected["selected_subset"].tolist()
    ):
        raise ValueError(
            "Reapplying delta=0.02 did not reproduce the saved "
            "primary subset selections."
        )

    if (
        observed["task_count"].astype(int).tolist()
        != expected["task_count"].astype(int).tolist()
    ):
        raise ValueError(
            "Reapplying delta=0.02 did not reproduce the saved "
            "primary task counts."
        )
def load_primary_predictions(
    primary_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Load the participant predictions from the locked primary run."""

    require_columns(
        primary_metrics,
        PRIMARY_EVALUATION_REQUIRED_COLUMNS,
        "Primary nested fold metrics",
    )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Primary nested predictions were not found at "
            f"{PREDICTIONS_PATH}."
        )

    primary_predictions = pd.read_csv(PREDICTIONS_PATH)

    require_columns(
        primary_predictions,
        PRIMARY_PREDICTION_REQUIRED_COLUMNS,
        "Primary nested predictions",
    )

    if len(primary_predictions) != 102:
        raise ValueError(
            "Expected 102 primary predictions, "
            f"but found {len(primary_predictions)}."
        )

    if primary_predictions["participant_id"].nunique() != 102:
        raise ValueError(
            "Primary predictions do not contain 102 unique participants."
        )

    if primary_predictions["participant_id"].duplicated().any():
        raise ValueError(
            "A participant appears more than once in the primary predictions."
        )

    if not np.isfinite(
        primary_predictions[["score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Primary predictions contain a non-finite score."
        )

    return primary_predictions


def selection_from_row(row: object) -> dict[str, object]:
    """Convert one selection-table row into the nested evaluator format."""

    tasks = tuple(str(row.selected_tasks).split(","))

    return {
        "outer_fold": int(row.outer_fold),
        "tasks": tasks,
        "subset_id": str(row.selected_subset),
        "task_count": int(row.task_count),
        "predictor_count": int(row.predictor_count),
        "inner_macro_f1_mean": float(
            row.selected_inner_macro_f1_mean
        ),
        "inner_macro_f1_std": float(
            row.selected_inner_macro_f1_std
        ),
        "inner_full_reference": float(
            row.inner_full_reference
        ),
        "competitive_threshold": float(
            row.competitive_threshold
        ),
        "competitive_subset_count": int(
            row.competitive_subset_count
        ),
        "delta": float(row.delta),
    }


def seed_primary_evaluation_cache(
    primary_metrics: pd.DataFrame,
    primary_predictions: pd.DataFrame,
) -> dict[tuple[int, str], dict[str, object]]:
    """Create reusable model results from the completed primary run."""

    cache: dict[tuple[int, str], dict[str, object]] = {}

    for metric_record in primary_metrics.to_dict(orient="records"):
        outer_fold = int(metric_record["outer_fold"])
        selected_subset = str(metric_record["selected_subset"])
        key = (outer_fold, selected_subset)

        fold_predictions = primary_predictions.loc[
            primary_predictions["outer_fold"].eq(outer_fold)
        ].to_dict(orient="records")

        expected_count = int(metric_record["test_participants"])

        if len(fold_predictions) != expected_count:
            raise ValueError(
                f"Primary prediction count for outer fold {outer_fold} "
                f"does not match its recorded test-participant count."
            )

        prediction_subsets = {
            str(record["selected_subset"])
            for record in fold_predictions
        }

        if prediction_subsets != {selected_subset}:
            raise ValueError(
                f"Primary predictions for outer fold {outer_fold} "
                "do not match its selected subset."
            )

        cache[key] = {
            "metrics": metric_record,
            "predictions": fold_predictions,
        }

    return cache


def evaluate_delta_selections(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    selections: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    frozen_configurations: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each unique fold-specific protocol only once."""

    # The cache starts with all five completed primary evaluations.
    # Any sensitivity row selecting the same fold and protocol can reuse
    # those exact metrics and predictions instead of refitting the model.
    evaluation_cache = seed_primary_evaluation_cache(
        primary_metrics,
        primary_predictions,
    )

    metric_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for row in (
        selections
        .sort_values(["delta", "outer_fold"])
        .itertuples(index=False)
    ):
        selection = selection_from_row(row)
        outer_fold = int(selection["outer_fold"])
        selected_subset = str(selection["subset_id"])
        key = (outer_fold, selected_subset)

        fit_reused = key in evaluation_cache

        if fit_reused:
            print(
                f"Delta {row.delta:.2f}, outer fold {outer_fold}: "
                f"reusing {selected_subset}"
            )
        else:
            print(
                f"Delta {row.delta:.2f}, outer fold {outer_fold}: "
                f"fitting {selected_subset}"
            )

            outer_result = evaluate_selected_subset_outer_fold(
                features=features,
                outer_folds=outer_folds,
                outer_fold=outer_fold,
                selection=selection,
                configuration=frozen_configurations[outer_fold],
            )

            evaluation_cache[key] = outer_result

        cached_result = evaluation_cache[key]

        # Model performance comes from the cached evaluation, while the
        # threshold and competitive-count fields belong to this delta.
        metrics = dict(cached_result["metrics"])

        metrics.update(
            {
                "outer_fold": outer_fold,
                "selected_subset": selected_subset,
                "selected_tasks": ",".join(selection["tasks"]),
                "task_count": int(selection["task_count"]),
                "predictor_count": int(
                    selection["predictor_count"]
                ),
                "inner_full_reference": float(
                    selection["inner_full_reference"]
                ),
                "selected_inner_macro_f1_mean": float(
                    selection["inner_macro_f1_mean"]
                ),
                "selected_inner_macro_f1_std": float(
                    selection["inner_macro_f1_std"]
                ),
                "competitive_threshold": float(
                    selection["competitive_threshold"]
                ),
                "competitive_subset_count": int(
                    selection["competitive_subset_count"]
                ),
                "delta": float(selection["delta"]),
                "configuration": frozen_configurations[outer_fold],
                "fit_reused": fit_reused,
            }
        )

        metric_records.append(metrics)

        for cached_prediction in cached_result["predictions"]:
            prediction = dict(cached_prediction)

            prediction.update(
                {
                    "delta": float(selection["delta"]),
                    "outer_fold": outer_fold,
                    "selected_subset": selected_subset,
                    "fit_reused": fit_reused,
                }
            )

            prediction_records.append(prediction)

    fold_metrics = (
        pd.DataFrame(metric_records)
        .sort_values(["delta", "outer_fold"])
        .reset_index(drop=True)
    )

    predictions = (
        pd.DataFrame(prediction_records)
        .sort_values(["delta", "outer_fold", "participant_id"])
        .reset_index(drop=True)
    )

    return fold_metrics, predictions


def validate_delta_evaluation(
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Validate fold coverage, participant coverage, and metric values."""

    expected_metric_rows = len(DELTAS) * len(OUTER_FOLDS)
    expected_prediction_rows = len(DELTAS) * 102

    if len(fold_metrics) != expected_metric_rows:
        raise ValueError(
            f"Expected {expected_metric_rows} fold-metric rows, "
            f"but found {len(fold_metrics)}."
        )

    if fold_metrics.duplicated(
        subset=["delta", "outer_fold"]
    ).any():
        raise ValueError(
            "Duplicate delta and outer-fold metric rows were found."
        )

    if len(predictions) != expected_prediction_rows:
        raise ValueError(
            f"Expected {expected_prediction_rows} prediction rows, "
            f"but found {len(predictions)}."
        )

    if predictions.duplicated(
        subset=["delta", "participant_id"]
    ).any():
        raise ValueError(
            "A participant received multiple predictions for one delta."
        )

    for delta in DELTAS:
        delta_metrics = fold_metrics.loc[
            np.isclose(fold_metrics["delta"], delta)
        ]

        delta_predictions = predictions.loc[
            np.isclose(predictions["delta"], delta)
        ]

        metric_folds = set(
            delta_metrics["outer_fold"].astype(int)
        )

        if metric_folds != set(OUTER_FOLDS):
            raise ValueError(
                f"Delta {delta} does not contain all outer folds."
            )

        if len(delta_predictions) != 102:
            raise ValueError(
                f"Delta {delta} does not contain 102 predictions."
            )

        if delta_predictions["participant_id"].nunique() != 102:
            raise ValueError(
                f"Delta {delta} does not contain 102 unique participants."
            )

    if not np.isfinite(
        fold_metrics[OUTER_METRIC_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "A delta-sensitivity outer metric is non-finite."
        )

    if not np.isfinite(
        predictions[["score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "A delta-sensitivity prediction score is non-finite."
        )

    # Join the tables to prove that every evaluated protocol is exactly
    # the protocol selected from the corresponding inner-CV results.
    selection_check = fold_metrics[
        ["delta", "outer_fold", "selected_subset"]
    ].merge(
        selections[
            ["delta", "outer_fold", "selected_subset"]
        ],
        on=["delta", "outer_fold"],
        how="inner",
        suffixes=("_evaluated", "_selected"),
        validate="one_to_one",
    )

    if len(selection_check) != expected_metric_rows:
        raise ValueError(
            "Not every selected protocol has an outer evaluation."
        )

    if not selection_check["selected_subset_evaluated"].eq(
        selection_check["selected_subset_selected"]
    ).all():
        raise ValueError(
            "An outer fold evaluated a protocol different from the "
            "inner-selected protocol."
        )

    # A participant's true label and frozen outer-fold assignment must
    # remain identical across all three sensitivity conditions.
    if not predictions.groupby("participant_id")[
        "true_label"
    ].nunique().eq(1).all():
        raise ValueError(
            "A participant has inconsistent true labels across deltas."
        )

    if not predictions.groupby("participant_id")[
        "outer_fold"
    ].nunique().eq(1).all():
        raise ValueError(
            "A participant has inconsistent outer folds across deltas."
        )


def save_delta_evaluation(
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Save validated sensitivity artifacts using atomic replacement."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = (
        (selections, DELTA_SELECTIONS_PATH),
        (fold_metrics, DELTA_FOLD_METRICS_PATH),
        (predictions, DELTA_PREDICTIONS_PATH),
    )

    temporary_paths = [
        path.with_suffix(path.suffix + ".tmp")
        for _, path in artifacts
    ]

    try:
        for (table, final_path), temporary_path in zip(
            artifacts,
            temporary_paths,
        ):
            table.to_csv(temporary_path, index=False)
            temporary_path.replace(final_path)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

def build_experiment_record(
    fold_metrics: pd.DataFrame,
    frozen_configurations: dict[int, dict],
) -> dict[str, object]:
    """Build the aggregate record that is safe to track in Git."""

    results_by_delta: dict[str, object] = {}

    for delta in DELTAS:
        delta_rows = (
            fold_metrics.loc[
                np.isclose(fold_metrics["delta"], delta)
            ]
            .sort_values("outer_fold")
            .reset_index(drop=True)
        )

        fold_results: list[dict[str, object]] = []

        for row in delta_rows.itertuples(index=False):
            fold_results.append(
                {
                    "outer_fold": int(row.outer_fold),
                    "selected_subset": str(row.selected_subset),
                    "selected_tasks": str(
                        row.selected_tasks
                    ).split(","),
                    "task_count": int(row.task_count),
                    "predictor_count": int(row.predictor_count),
                    "inner_full_reference": float(
                        row.inner_full_reference
                    ),
                    "selected_inner_macro_f1_mean": float(
                        row.selected_inner_macro_f1_mean
                    ),
                    "selected_inner_macro_f1_std": float(
                        row.selected_inner_macro_f1_std
                    ),
                    "competitive_threshold": float(
                        row.competitive_threshold
                    ),
                    "competitive_subset_count": int(
                        row.competitive_subset_count
                    ),
                    "outer_test_macro_f1": float(
                        row.outer_test_macro_f1
                    ),
                    "outer_test_balanced_accuracy": float(
                        row.outer_test_balanced_accuracy
                    ),
                    "outer_test_roc_auc": float(
                        row.outer_test_roc_auc
                    ),
                    "outer_test_sensitivity": float(
                        row.outer_test_sensitivity
                    ),
                    "outer_test_specificity": float(
                        row.outer_test_specificity
                    ),
                    "outer_test_positive_class_f1": float(
                        row.outer_test_positive_class_f1
                    ),
                    "outer_test_accuracy": float(
                        row.outer_test_accuracy
                    ),
                    "fit_reused": bool(row.fit_reused),
                }
            )

        # Fold means and sample standard deviations use the same
        # convention as the primary nested-selection experiment.
        aggregate_metrics = {
            column.removeprefix("outer_test_"): {
                "mean": float(delta_rows[column].mean()),
                "std": float(delta_rows[column].std(ddof=1)),
            }
            for column in OUTER_METRIC_COLUMNS
        }

        results_by_delta[f"{delta:.2f}"] = {
            "analysis_role": (
                "primary"
                if np.isclose(delta, PRIMARY_DELTA)
                else "sensitivity"
            ),
            "fold_results": fold_results,
            "aggregate_outer_test_metrics": aggregate_metrics,
            "mean_selected_task_count": float(
                delta_rows["task_count"].mean()
            ),
            "selected_task_count_std": float(
                delta_rows["task_count"].std(ddof=1)
            ),
            "reused_fit_count": int(
                delta_rows["fit_reused"].sum()
            ),
            "new_fit_count": int(
                (~delta_rows["fit_reused"]).sum()
            ),
        }

    primary_by_fold = (
        fold_metrics.loc[
            np.isclose(
                fold_metrics["delta"],
                PRIMARY_DELTA,
            )
        ]
        .set_index("outer_fold")
        ["outer_test_macro_f1"]
    )

    paired_comparisons: dict[str, object] = {}

    for delta in DELTAS:
        if np.isclose(delta, PRIMARY_DELTA):
            continue

        sensitivity_by_fold = (
            fold_metrics.loc[
                np.isclose(
                    fold_metrics["delta"],
                    delta,
                )
            ]
            .set_index("outer_fold")
            ["outer_test_macro_f1"]
        )

        differences = (
            sensitivity_by_fold - primary_by_fold
        ).sort_index()

        paired_comparisons[f"{delta:.2f}"] = {
            "mean_macro_f1_difference": float(
                differences.mean()
            ),
            "std_macro_f1_difference": float(
                differences.std(ddof=1)
            ),
            "fold_differences": [
                {
                    "outer_fold": int(outer_fold),
                    "sensitivity_macro_f1": float(
                        sensitivity_by_fold.loc[outer_fold]
                    ),
                    "primary_macro_f1": float(
                        primary_by_fold.loc[outer_fold]
                    ),
                    "difference": float(difference),
                }
                for outer_fold, difference in differences.items()
            ],
        }

    return {
        "experiment": "nested_task_selection_delta_sensitivity",
        "population": {
            "als_participants": 102,
        },
        "evaluation": {
            "outer_folds": len(OUTER_FOLDS),
            "task_subsets_per_outer_fold": SUBSETS_PER_FOLD,
            "primary_metric": "macro_f1",
            "primary_delta": PRIMARY_DELTA,
            "sensitivity_deltas": [
                float(delta)
                for delta in DELTAS
                if not np.isclose(delta, PRIMARY_DELTA)
            ],
            "participant_disjoint": True,
            "outer_results_used_for_delta_selection": False,
        },
        "selection_rule": {
            "competitive_definition": (
                "inner_macro_f1_mean >= "
                "inner_full_reference - delta"
            ),
            "tie_break_order": [
                "fewer_tasks",
                "higher_inner_macro_f1_mean",
                "lower_inner_macro_f1_std",
                "lexicographic_task_order",
            ],
        },
        "frozen_configurations": {
            str(outer_fold): configuration
            for outer_fold, configuration in sorted(
                frozen_configurations.items()
            )
        },
        "results_by_delta": results_by_delta,
        "paired_macro_f1_comparisons_to_primary": (
            paired_comparisons
        ),
        "interpretation_status": (
            "Delta 0.02 remains the locked primary analysis. "
            "Delta 0.01 and delta 0.05 are prespecified sensitivity "
            "checks. Their outer-test performance must not be used "
            "to replace the primary delta. Results are based on "
            "participant-disjoint nested cross-validation within the "
            "VOC-ALS cohort and are not external clinical validation."
        ),
    }

def validate_experiment_record(
    record: dict[str, object],
) -> None:
    """Confirm that the tracked record is complete and aggregate-only."""

    expected_delta_keys = {
        f"{delta:.2f}"
        for delta in DELTAS
    }

    recorded_delta_keys = set(
        record["results_by_delta"]
    )

    if recorded_delta_keys != expected_delta_keys:
        raise ValueError(
            "The experiment record does not contain every delta."
        )

    for delta_key, delta_result in record[
        "results_by_delta"
    ].items():
        fold_results = delta_result["fold_results"]

        if len(fold_results) != len(OUTER_FOLDS):
            raise ValueError(
                f"Delta {delta_key} does not contain five fold results."
            )

        recorded_folds = {
            int(result["outer_fold"])
            for result in fold_results
        }

        if recorded_folds != set(OUTER_FOLDS):
            raise ValueError(
                f"Delta {delta_key} does not contain all outer folds."
            )

    # Participant-level information stays in ignored generated files.
    # The tracked experiment record contains only aggregate and fold data.
    forbidden_fields = {
        "participant_id",
        "true_label",
        "predicted_label",
        "score",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            keys = set(value)
            for nested_value in value.values():
                keys.update(collect_keys(nested_value))
            return keys

        if isinstance(value, list):
            keys: set[str] = set()
            for nested_value in value:
                keys.update(collect_keys(nested_value))
            return keys

        return set()

    present_forbidden_fields = (
        collect_keys(record) & forbidden_fields
    )

    if present_forbidden_fields:
        raise ValueError(
            "The tracked experiment record contains participant-level "
            f"fields: {sorted(present_forbidden_fields)}"
        )

    # Disallowing NaN and infinity here ensures the committed JSON is
    # valid for strict parsers outside Python.
    json.dumps(record, allow_nan=False)


def save_experiment_record(
    record: dict[str, object],
) -> None:
    """Atomically save the tracked aggregate experiment record."""

    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = EXPERIMENT_RECORD_PATH.with_suffix(
        ".json.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                record,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            EXPERIMENT_RECORD_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def main() -> None:
    """Run the complete delta-sensitivity evaluation."""

    inner_results, primary_metrics = load_selection_inputs()
    primary_predictions = load_primary_predictions(primary_metrics)

    selections = build_selection_table(inner_results)

    validate_primary_reproduction(
        selections,
        primary_metrics,
    )

    features, outer_folds, nested_folds = load_experiment_inputs()

    validate_experiment_inputs(
        features,
        outer_folds,
        nested_folds,
    )

    frozen_configurations = load_frozen_configurations()

    if set(frozen_configurations) != set(OUTER_FOLDS):
        raise ValueError(
            "Frozen configurations do not match the five outer folds."
        )

    print("Delta-sensitivity selection inputs validated.")
    print(f"Inner-result rows: {len(inner_results)}")
    print(f"Delta values: {list(DELTAS)}")
    print()

    fold_metrics, predictions = evaluate_delta_selections(
        features=features,
        outer_folds=outer_folds,
        selections=selections,
        primary_metrics=primary_metrics,
        primary_predictions=primary_predictions,
        frozen_configurations=frozen_configurations,
    )

    validate_delta_evaluation(
        selections=selections,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    experiment_record = build_experiment_record(
        fold_metrics=fold_metrics,
        frozen_configurations=frozen_configurations,
    )

    validate_experiment_record(
        experiment_record
    )

    save_delta_evaluation(
        selections=selections,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    save_experiment_record(
        experiment_record
    )

    summary = (
        fold_metrics.groupby("delta")
        .agg(
            macro_f1_mean=(
                "outer_test_macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "outer_test_macro_f1",
                "std",
            ),
            mean_task_count=(
                "task_count",
                "mean",
            ),
            reused_fits=(
                "fit_reused",
                "sum",
            ),
        )
        .reset_index()
    )

    print()
    print("Delta-sensitivity outer evaluation validated.")
    print()
    print("Selected protocol and outer-test Macro F1:")

    print(
        fold_metrics[
            [
                "delta",
                "outer_fold",
                "selected_subset",
                "task_count",
                "outer_test_macro_f1",
                "fit_reused",
            ]
        ].to_string(index=False)
    )

    print()
    print("Summary by delta:")
    print(summary.to_string(index=False))

    print()
    print("Generated artifacts:")
    print(DELTA_SELECTIONS_PATH)
    print(DELTA_FOLD_METRICS_PATH)
    print(DELTA_PREDICTIONS_PATH)
    print(EXPERIMENT_RECORD_PATH)

if __name__ == "__main__":
    main()
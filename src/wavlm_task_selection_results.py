"""Build and save reproducible WavLM task-selection result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_metadata import TASKS

OUTPUT_DIRECTORY = Path("results/generated")
REGULARIZATION_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_task_selection_regularization.csv"
)
INNER_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_task_selection_inner_results.csv"
)
FOLD_METRICS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_task_selection_fold_metrics.csv"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_task_selection_predictions.csv"
)
BOOTSTRAP_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_task_selection_bootstrap.csv"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/wavlm_task_selection.json"
)

METRIC_NAMES = (
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
    "positive_class_f1",
    "accuracy",
)


def summarize_metric_column(
    values: pd.Series,
) -> dict[str, float]:
    """Return the fold mean and sample standard deviation for one metric."""

    numeric_values = values.to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "A WavLM metric summary contains a non-finite value."
        )

    return {
        "mean": float(numeric_values.mean()),
        "std": float(
            numeric_values.std(ddof=1)
        ),
    }


def build_projection_summary(
    projection_cache: dict,
) -> dict[str, dict[str, float]]:
    """Summarize training-only PCA variance across the 20 inner folds."""

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for task in TASKS:
        values = np.asarray(
            [
                projected_fold.explained_variance_by_task[
                    task
                ]
                for projected_fold in projection_cache.values()
            ],
            dtype=float,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"PCA variance summary for task {task} is not finite."
            )

        summary[task] = {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
        }

    return summary


def build_metric_summary(
    fold_metrics: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize selected and full held-out metrics across outer folds."""

    summary: dict[
        str,
        dict[str, dict[str, float]],
    ] = {
        "selected": {},
        "full": {},
    }

    for protocol_name in (
        "selected",
        "full",
    ):
        for metric_name in METRIC_NAMES:
            column = (
                f"outer_test_{protocol_name}_{metric_name}"
            )

            if column not in fold_metrics.columns:
                raise ValueError(
                    f"Missing WavLM fold metric column {column}."
                )

            summary[protocol_name][metric_name] = (
                summarize_metric_column(
                    fold_metrics[column]
                )
            )

    return summary


def build_selection_frequency(
    selections: pd.DataFrame,
) -> dict[str, int]:
    """Count how often each speech task appears in selected protocols."""

    frequency = {
        task: 0
        for task in TASKS
    }

    for selected_tasks in selections[
        "selected_tasks"
    ]:
        for task in selected_tasks:
            frequency[str(task)] += 1

    return frequency


def build_wavlm_experiment_record(
    regularization_results: pd.DataFrame,
    selections: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    projection_cache: dict,
    bootstrap_summary: dict[str, object],
    participant_count: int,
    excluded_participant_count: int,
    model_name: str,
    model_revision: str,
    embedding_dimension: int,
    pca_components_per_task: int,
    delta: float,
) -> dict[str, object]:
    """Build the participant-free record committed with the experiment."""

    selected_regularization = (
        regularization_results.loc[
            regularization_results["selected"],
            [
                "outer_fold",
                "regularization_c",
                "inner_macro_f1_mean",
                "inner_macro_f1_std",
            ],
        ]
        .sort_values("outer_fold")
        .to_dict(orient="records")
    )

    fold_columns = [
        column
        for column in fold_metrics.columns
        if column != "selected_tasks"
    ]
    fold_results = (
        fold_metrics[fold_columns]
        .sort_values("outer_fold")
        .to_dict(orient="records")
    )
    metric_summary = build_metric_summary(
        fold_metrics
    )
    selected_macro_f1 = metric_summary[
        "selected"
    ]["macro_f1"]["mean"]
    full_macro_f1 = metric_summary[
        "full"
    ]["macro_f1"]["mean"]

    return {
        "experiment": (
            "duplicate_filtered_wavlm_nested_task_selection"
        ),
        "participant_level_fields": [],
        "design": {
            "participants": int(participant_count),
            "excluded_duplicate_connected_participants": int(
                excluded_participant_count
            ),
            "speech_tasks": list(TASKS),
            "task_subset_count": 255,
            "outer_folds": 5,
            "inner_folds_per_outer_fold": 4,
            "primary_metric": "macro_f1",
            "competitive_delta": float(delta),
            "selection_rule": (
                "fewest tasks within delta of the full inner-CV "
                "reference, then higher mean Macro F1, lower "
                "standard deviation, and subset identifier"
            ),
        },
        "representation": {
            "encoder": model_name,
            "encoder_revision": model_revision,
            "frozen_encoder": True,
            "embedding_dimension_per_task": int(
                embedding_dimension
            ),
            "pca_components_per_task": int(
                pca_components_per_task
            ),
            "pca_fit_scope": (
                "training participants only within each split"
            ),
            "inner_fold_explained_variance": (
                build_projection_summary(
                    projection_cache
                )
            ),
        },
        "classifier": {
            "type": "balanced_l2_logistic_regression",
            "solver": "liblinear",
            "regularization_candidates": [
                0.01,
                0.1,
                1.0,
                10.0,
            ],
            "regularization_selection_scope": (
                "full-assessment inner CV within each outer fold"
            ),
            "selected_regularization": (
                selected_regularization
            ),
        },
        "selection_summary": {
            "mean_selected_task_count": float(
                selections["task_count"].mean()
            ),
            "selected_protocols": (
                selections.sort_values("outer_fold")[
                    "selected_subset"
                ].tolist()
            ),
            "task_frequency": build_selection_frequency(
                selections
            ),
        },
        "fold_results": fold_results,
        "outer_test_metrics": metric_summary,
        "selected_vs_full": {
            "selected_macro_f1_mean": float(
                selected_macro_f1
            ),
            "full_macro_f1_mean": float(
                full_macro_f1
            ),
            "difference_mean": float(
                selected_macro_f1
                - full_macro_f1
            ),
        },
        "paired_bootstrap": bootstrap_summary,
        "inference": {
            "point_estimate_inside_margin": bool(
                bootstrap_summary[
                    "point_estimate_inside_margin"
                ]
            ),
            "formal_noninferiority_supported": bool(
                bootstrap_summary[
                    "one_sided_bound_supports_noninferiority"
                ]
            ),
            "interpretation": (
                "The reduced WavLM protocols retained similar average "
                "performance, but the paired bootstrap did not establish "
                "noninferiority at the prespecified margin."
            ),
        },
    }


def convert_json_value(
    value: Any,
) -> Any:
    """Convert NumPy and pandas scalar values into JSON-safe objects."""

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): convert_json_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            convert_json_value(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return convert_json_value(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def validate_wavlm_experiment_record(
    record: dict[str, object],
) -> None:
    """Protect the tracked record from participant data and stale values."""

    if record.get("participant_level_fields") != []:
        raise ValueError(
            "The tracked WavLM record must not contain participant fields."
        )

    serialized = json.dumps(
        convert_json_value(record),
        sort_keys=True,
    )
    forbidden_fields = (
        '"participant_id"',
        '"inner_train_ids"',
        '"inner_validation_ids"',
        '"outer_test_ids"',
    )

    for forbidden_field in forbidden_fields:
        if forbidden_field in serialized:
            raise ValueError(
                "The tracked WavLM record contains participant-level "
                f"field {forbidden_field}."
            )

    fold_results = record.get(
        "fold_results"
    )

    if not isinstance(
        fold_results,
        list,
    ) or len(fold_results) != 5:
        raise ValueError(
            "The tracked WavLM record must contain five fold results."
        )

    bootstrap = record.get(
        "paired_bootstrap"
    )

    if not isinstance(
        bootstrap,
        dict,
    ) or int(
        bootstrap.get(
            "bootstrap_replicates",
            0,
        )
    ) != 10_000:
        raise ValueError(
            "The tracked WavLM record has an incomplete bootstrap."
        )


def prepare_csv_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Serialize tuple-valued task columns consistently before CSV output."""

    prepared = frame.copy()

    for column in (
        "tasks",
        "selected_tasks",
    ):
        if column in prepared.columns:
            prepared[column] = prepared[
                column
            ].map(
                lambda tasks: ",".join(tasks)
                if isinstance(tasks, tuple)
                else tasks
            )

    return prepared


def save_wavlm_task_selection_results(
    regularization_results: pd.DataFrame,
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    experiment_record: dict[str, object],
) -> None:
    """Save private generated tables and the public aggregate record."""

    validate_wavlm_experiment_record(
        experiment_record
    )
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_csv_frame(
        regularization_results
    ).to_csv(
        REGULARIZATION_RESULTS_PATH,
        index=False,
    )
    prepare_csv_frame(
        inner_results
    ).to_csv(
        INNER_RESULTS_PATH,
        index=False,
    )
    prepare_csv_frame(
        fold_metrics
    ).to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )
    prepare_csv_frame(
        predictions
    ).to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
    prepare_csv_frame(
        bootstrap_results
    ).to_csv(
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
        output_file.write("\n")


def print_generated_artifacts() -> None:
    """List the generated and tracked artifacts produced by the run."""

    print()
    print("Generated artifacts:")
    print(REGULARIZATION_RESULTS_PATH)
    print(INNER_RESULTS_PATH)
    print(FOLD_METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(BOOTSTRAP_RESULTS_PATH)
    print(EXPERIMENT_RECORD_PATH)

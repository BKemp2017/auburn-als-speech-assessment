"""Run leakage-safe nested task selection with frozen WavLM embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from audit_duplicate_impact import normalize_participant_id
from audit_metadata import TASKS
from bootstrap_wavlm_task_selection import (
    print_paired_wavlm_bootstrap_summary,
    run_paired_wavlm_bootstrap,
    summarize_paired_wavlm_bootstrap,
)
from evaluate_task_subsets import generate_task_subsets
from wavlm_task_selection_results import (
    build_wavlm_experiment_record,
    print_generated_artifacts,
    save_wavlm_task_selection_results,
)
from wavlm_task_selection_core import (
    EXPECTED_INNER_FOLDS_PER_OUTER_FOLD,
    EXPECTED_OUTER_FOLDS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_TASK_SUBSETS,
    MODEL_NAME,
    MODEL_REVISION,
    PCA_COMPONENTS_PER_TASK,
    PRIMARY_DELTA,
    REGULARIZATION_VALUES,
    ProjectedInnerFold,
    WavLMEmbeddingArtifact,
    build_inner_projection_cache,
    build_wavlm_classifier,
    calculate_classification_metrics,
    concatenate_outer_tasks,
    concatenate_projected_tasks,
    load_duplicate_filtered_design,
    load_wavlm_artifact,
    print_projection_cache_summary,
    project_outer_fold_tasks,
    validate_wavlm_inputs,
)

def projection_component_count(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
) -> int:
    """Return the one PCA dimension shared by a validated cache."""

    component_counts = {
        projected_fold.pca_components_per_task
        for projected_fold in projection_cache.values()
    }

    if len(component_counts) != 1:
        raise ValueError(
            "Expected one PCA dimension across the projection cache."
        )

    return next(
        iter(component_counts)
    )

def evaluate_regularization_candidate(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
    outer_fold: int,
    regularization_c: float,
) -> dict[str, object]:
    """Score one C value on the full assessment's four inner folds."""

    pca_components_per_task = projection_component_count(
        projection_cache
    )

    fold_scores: list[float] = []
    score_fields: dict[str, float] = {}

    for inner_fold in range(
        EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    ):
        projected_fold = projection_cache[
            (outer_fold, inner_fold)
        ]

        training_features, validation_features = (
            concatenate_projected_tasks(
                projected_fold=projected_fold,
                tasks=tuple(TASKS),
            )
        )

        classifier = build_wavlm_classifier(
            regularization_c
        )
        classifier.fit(
            training_features,
            projected_fold.training_targets,
        )
        predictions = classifier.predict(
            validation_features
        )

        score = float(
            f1_score(
                projected_fold.validation_targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        if not np.isfinite(score):
            raise ValueError(
                "A regularization candidate produced a non-finite "
                "inner-fold Macro F1 score."
            )

        fold_scores.append(score)
        score_fields[
            f"inner_fold_{inner_fold}_macro_f1"
        ] = score

    score_series = pd.Series(
        fold_scores,
        dtype=float,
    )

    return {
        "outer_fold": outer_fold,
        "regularization_c": float(
            regularization_c
        ),
        "predictor_count": (
            len(TASKS)
            * pca_components_per_task
        ),
        **score_fields,
        "inner_macro_f1_mean": float(
            score_series.mean()
        ),
        "inner_macro_f1_std": float(
            score_series.std(ddof=1)
        ),
    }

def select_full_assessment_regularization(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Select one full-assessment C value inside each outer fold."""

    result_rows: list[
        dict[str, object]
    ] = []

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        print()
        print(
            "Selecting full-assessment regularization for outer fold "
            f"{outer_fold}..."
        )

        for regularization_c in REGULARIZATION_VALUES:
            result_rows.append(
                evaluate_regularization_candidate(
                    projection_cache=projection_cache,
                    outer_fold=outer_fold,
                    regularization_c=(
                        regularization_c
                    ),
                )
            )

    results = pd.DataFrame(
        result_rows
    )
    results["selected"] = False
    selected_by_outer_fold: dict[
        int,
        float,
    ] = {}

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_candidates = results.loc[
            results["outer_fold"].eq(
                outer_fold
            )
        ]

        # Prefer the highest mean Macro F1. Exact ties favor lower score
        # variability and then stronger regularization through smaller C.
        selected_index = (
            fold_candidates.sort_values(
                by=[
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                    "regularization_c",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index[0]
        )

        results.loc[
            selected_index,
            "selected",
        ] = True
        selected_by_outer_fold[outer_fold] = float(
            results.loc[
                selected_index,
                "regularization_c",
            ]
        )

    validate_regularization_results(
        results=results,
        selected_by_outer_fold=(
            selected_by_outer_fold
        ),
    )

    return (
        results,
        selected_by_outer_fold,
    )

def validate_regularization_results(
    results: pd.DataFrame,
    selected_by_outer_fold: dict[int, float],
) -> None:
    """Verify complete and finite full-assessment tuning results."""

    expected_rows = (
        EXPECTED_OUTER_FOLDS
        * len(REGULARIZATION_VALUES)
    )

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} regularization rows, found "
            f"{len(results)}."
        )

    if set(selected_by_outer_fold) != set(
        range(EXPECTED_OUTER_FOLDS)
    ):
        raise ValueError(
            "A regularization value was not selected for every outer "
            "fold."
        )

    numeric_columns = [
        column
        for column in results.columns
        if column != "selected"
    ]

    if not np.isfinite(
        results[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Regularization results contain a non-finite value."
        )

    score_columns = [
        column
        for column in results.columns
        if column.endswith("macro_f1")
        or column in {
            "inner_macro_f1_mean",
            "inner_macro_f1_std",
        }
    ]

    for column in score_columns:
        if results[column].lt(0.0).any() or results[
            column
        ].gt(1.0).any():
            raise ValueError(
                f"Regularization result column {column} is outside "
                "the valid score range."
            )

    expected_c_values = set(
        REGULARIZATION_VALUES
    )

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_results = results.loc[
            results["outer_fold"].eq(
                outer_fold
            )
        ]

        if set(
            fold_results["regularization_c"]
        ) != expected_c_values:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain the complete "
                "regularization grid."
            )

        if int(
            fold_results["selected"].sum()
        ) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not have exactly one "
                "selected regularization value."
            )

def print_regularization_summary(
    results: pd.DataFrame,
) -> None:
    """Display all full-assessment candidates and fold selections."""

    print()
    print("WavLM full-assessment regularization validated.")
    print(
        results[
            [
                "outer_fold",
                "regularization_c",
                "inner_macro_f1_mean",
                "inner_macro_f1_std",
                "selected",
            ]
        ]
        .sort_values(
            [
                "outer_fold",
                "regularization_c",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

def evaluate_wavlm_subset_inner_cv(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
    outer_fold: int,
    tasks: tuple[str, ...],
    regularization_c: float,
) -> dict[str, object]:
    """Evaluate one task subset across an outer fold's inner splits."""

    pca_components_per_task = projection_component_count(
        projection_cache
    )

    fold_scores: list[float] = []
    score_fields: dict[str, float] = {}

    for inner_fold in range(
        EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    ):
        projected_fold = projection_cache[
            (outer_fold, inner_fold)
        ]
        training_features, validation_features = (
            concatenate_projected_tasks(
                projected_fold=projected_fold,
                tasks=tasks,
            )
        )

        classifier = build_wavlm_classifier(
            regularization_c
        )
        classifier.fit(
            training_features,
            projected_fold.training_targets,
        )
        predictions = classifier.predict(
            validation_features
        )

        score = float(
            f1_score(
                projected_fold.validation_targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        if not np.isfinite(score):
            raise ValueError(
                "A WavLM task subset produced a non-finite inner "
                "Macro F1 score."
            )

        fold_scores.append(score)
        score_fields[
            f"inner_fold_{inner_fold}_macro_f1"
        ] = score

    score_series = pd.Series(
        fold_scores,
        dtype=float,
    )

    return {
        "outer_fold": outer_fold,
        "tasks": tasks,
        "subset_id": "+".join(tasks),
        "task_count": len(tasks),
        "predictor_count": (
            len(tasks)
            * pca_components_per_task
        ),
        "regularization_c": float(
            regularization_c
        ),
        **score_fields,
        "inner_macro_f1_mean": float(
            score_series.mean()
        ),
        "inner_macro_f1_std": float(
            score_series.std(ddof=1)
        ),
    }

def evaluate_all_wavlm_subsets(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
    subsets: list[tuple[str, ...]],
    selected_regularization: dict[int, float],
) -> pd.DataFrame:
    """Evaluate every task subset with fold-specific frozen settings."""

    if len(subsets) != EXPECTED_TASK_SUBSETS:
        raise ValueError(
            f"Expected {EXPECTED_TASK_SUBSETS} task subsets, found "
            f"{len(subsets)}."
        )

    if len(set(subsets)) != EXPECTED_TASK_SUBSETS:
        raise ValueError(
            "The generated task-subset list contains duplicates."
        )

    if subsets[-1] != tuple(TASKS):
        raise ValueError(
            "The final generated subset is not the full assessment."
        )

    result_rows: list[
        dict[str, object]
    ] = []

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        regularization_c = (
            selected_regularization[
                outer_fold
            ]
        )

        print()
        print(
            f"Evaluating {EXPECTED_TASK_SUBSETS} WavLM subsets for "
            f"outer fold {outer_fold} with C={regularization_c:g}..."
        )

        for subset_index, tasks in enumerate(
            subsets,
            start=1,
        ):
            if (
                subset_index == 1
                or subset_index % 25 == 0
                or subset_index == EXPECTED_TASK_SUBSETS
            ):
                print(
                    f"Outer fold {outer_fold}: "
                    f"{subset_index}/{EXPECTED_TASK_SUBSETS} subsets"
                )

            result_rows.append(
                evaluate_wavlm_subset_inner_cv(
                    projection_cache=projection_cache,
                    outer_fold=outer_fold,
                    tasks=tasks,
                    regularization_c=(
                        regularization_c
                    ),
                )
            )

    return pd.DataFrame(
        result_rows
    )

def select_competitive_wavlm_subsets(
    inner_results: pd.DataFrame,
    delta: float = PRIMARY_DELTA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the smallest competitive WavLM subset in each outer fold."""

    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(
            "The competitive-selection delta must be finite and "
            "nonnegative."
        )

    results = inner_results.copy()
    results["inner_full_reference"] = np.nan
    results["competitive_threshold"] = np.nan
    results["competitive"] = False
    results["selected"] = False
    results["competitive_subset_count"] = 0

    selection_rows: list[
        dict[str, object]
    ] = []
    full_subset_id = "+".join(TASKS)

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_mask = results[
            "outer_fold"
        ].eq(outer_fold)
        fold_results = results.loc[
            fold_mask
        ]
        full_results = fold_results.loc[
            fold_results["subset_id"].eq(
                full_subset_id
            )
        ]

        if len(full_results) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not have exactly one "
                "full-assessment result."
            )

        full_reference = float(
            full_results.iloc[0][
                "inner_macro_f1_mean"
            ]
        )
        competitive_threshold = (
            full_reference
            - delta
        )
        competitive_mask = (
            fold_mask
            & results[
                "inner_macro_f1_mean"
            ].ge(competitive_threshold)
        )
        competitive_results = results.loc[
            competitive_mask
        ]

        if competitive_results.empty:
            raise ValueError(
                f"Outer fold {outer_fold} has no competitive subset."
            )

        # The same deterministic rule used for handcrafted features
        # favors minimum burden, then performance, stability, and ID.
        selected_index = (
            competitive_results.sort_values(
                by=[
                    "task_count",
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                    "subset_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index[0]
        )
        competitive_subset_count = int(
            competitive_mask.sum()
        )

        results.loc[
            fold_mask,
            "inner_full_reference",
        ] = full_reference
        results.loc[
            fold_mask,
            "competitive_threshold",
        ] = competitive_threshold
        results.loc[
            competitive_mask,
            "competitive",
        ] = True
        results.loc[
            fold_mask,
            "competitive_subset_count",
        ] = competitive_subset_count
        results.loc[
            selected_index,
            "selected",
        ] = True

        selected_row = results.loc[
            selected_index
        ]
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_subset": selected_row[
                    "subset_id"
                ],
                "selected_tasks": selected_row[
                    "tasks"
                ],
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
                "inner_full_reference": full_reference,
                "competitive_threshold": (
                    competitive_threshold
                ),
                "competitive_subset_count": (
                    competitive_subset_count
                ),
                "delta": float(delta),
            }
        )

    selections = pd.DataFrame(
        selection_rows
    )

    validate_wavlm_subset_results(
        inner_results=results,
        selections=selections,
        delta=delta,
    )

    return results, selections

def validate_wavlm_subset_results(
    inner_results: pd.DataFrame,
    selections: pd.DataFrame,
    delta: float,
) -> None:
    """Verify complete subset coverage and deterministic selections."""

    expected_rows = (
        EXPECTED_OUTER_FOLDS
        * EXPECTED_TASK_SUBSETS
    )

    if len(inner_results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} WavLM subset rows, found "
            f"{len(inner_results)}."
        )

    if len(selections) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Expected exactly one WavLM selection per outer fold."
        )

    score_columns = [
        column
        for column in inner_results.columns
        if "macro_f1" in column
        or column
        in {
            "inner_full_reference",
            "competitive_threshold",
        }
    ]

    if not np.isfinite(
        inner_results[
            score_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "WavLM subset results contain a non-finite score."
        )

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        fold_results = inner_results.loc[
            inner_results["outer_fold"].eq(
                outer_fold
            )
        ]

        if len(fold_results) != EXPECTED_TASK_SUBSETS:
            raise ValueError(
                f"Outer fold {outer_fold} does not have all task "
                "subsets."
            )

        if fold_results[
            "subset_id"
        ].nunique() != EXPECTED_TASK_SUBSETS:
            raise ValueError(
                f"Outer fold {outer_fold} contains duplicate subset IDs."
            )

        if int(
            fold_results["selected"].sum()
        ) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} does not have exactly one "
                "selected subset."
            )

        full_references = fold_results[
            "inner_full_reference"
        ].unique()
        thresholds = fold_results[
            "competitive_threshold"
        ].unique()

        if len(full_references) != 1 or len(thresholds) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} has inconsistent reference "
                "values."
            )

        if not np.isclose(
            thresholds[0],
            full_references[0] - delta,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Outer fold {outer_fold} has an incorrect competitive "
                "threshold."
            )

        expected_competitive = fold_results[
            "inner_macro_f1_mean"
        ].ge(thresholds[0])

        if not np.array_equal(
            fold_results[
                "competitive"
            ].to_numpy(dtype=bool),
            expected_competitive.to_numpy(
                dtype=bool
            ),
        ):
            raise ValueError(
                f"Outer fold {outer_fold} has incorrect competitive "
                "flags."
            )

        recorded_competitive_counts = fold_results[
            "competitive_subset_count"
        ].unique()
        expected_competitive_count = int(
            expected_competitive.sum()
        )

        if (
            len(recorded_competitive_counts) != 1
            or int(recorded_competitive_counts[0])
            != expected_competitive_count
        ):
            raise ValueError(
                f"Outer fold {outer_fold} has an incorrect competitive "
                "subset count."
            )

        expected_selected_index = (
            fold_results.loc[
                expected_competitive
            ]
            .sort_values(
                by=[
                    "task_count",
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                    "subset_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index[0]
        )
        recorded_selected_index = (
            fold_results.loc[
                fold_results["selected"]
            ].index[0]
        )

        if recorded_selected_index != expected_selected_index:
            raise ValueError(
                f"Outer fold {outer_fold} does not follow the locked "
                "selection rule."
            )

def validate_full_subset_reproduction(
    inner_results: pd.DataFrame,
    regularization_results: pd.DataFrame,
) -> None:
    """Confirm subset evaluation reproduces the selected full baseline."""

    full_subset_id = "+".join(TASKS)

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        full_result = inner_results.loc[
            inner_results["outer_fold"].eq(
                outer_fold
            )
            & inner_results["subset_id"].eq(
                full_subset_id
            )
        ]
        tuned_result = regularization_results.loc[
            regularization_results["outer_fold"].eq(
                outer_fold
            )
            & regularization_results["selected"]
        ]

        if len(full_result) != 1 or len(tuned_result) != 1:
            raise ValueError(
                "Could not find one full-subset and one selected tuning "
                f"row for outer fold {outer_fold}."
            )

        full_row = full_result.iloc[0]
        tuned_row = tuned_result.iloc[0]

        for field in (
            "regularization_c",
            "inner_macro_f1_mean",
            "inner_macro_f1_std",
        ):
            if not np.isclose(
                float(full_row[field]),
                float(tuned_row[field]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Full-subset field {field} does not reproduce the "
                    f"selected tuning result for outer fold {outer_fold}."
                )

def evaluate_wavlm_outer_folds(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    selections: pd.DataFrame,
    pca_components_per_task: int = PCA_COMPONENTS_PER_TASK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected and full WavLM protocols on untouched folds."""

    metric_rows: list[
        dict[str, object]
    ] = []
    prediction_rows: list[
        dict[str, object]
    ] = []

    for outer_fold in range(
        EXPECTED_OUTER_FOLDS
    ):
        selection_rows = selections.loc[
            selections["outer_fold"].eq(
                outer_fold
            )
        ]

        if len(selection_rows) != 1:
            raise ValueError(
                f"Expected one selected protocol for outer fold "
                f"{outer_fold}."
            )

        selection = selection_rows.iloc[0]
        selected_tasks = tuple(
            selection["selected_tasks"]
        )
        regularization_c = float(
            selection["regularization_c"]
        )

        print()
        print(
            f"Evaluating untouched WavLM outer fold {outer_fold}: "
            f"{selection['selected_subset']} versus full assessment..."
        )

        projected_fold = project_outer_fold_tasks(
            artifact=artifact,
            labels=labels,
            outer_fold=outer_fold,
            pca_components_per_task=(
                pca_components_per_task
            ),
        )

        (
            selected_training_features,
            selected_test_features,
        ) = concatenate_outer_tasks(
            projected_fold=projected_fold,
            tasks=selected_tasks,
        )
        (
            full_training_features,
            full_test_features,
        ) = concatenate_outer_tasks(
            projected_fold=projected_fold,
            tasks=tuple(TASKS),
        )

        selected_classifier = build_wavlm_classifier(
            regularization_c
        )
        full_classifier = build_wavlm_classifier(
            regularization_c
        )

        selected_classifier.fit(
            selected_training_features,
            projected_fold.training_targets,
        )
        full_classifier.fit(
            full_training_features,
            projected_fold.training_targets,
        )

        selected_predictions = selected_classifier.predict(
            selected_test_features
        )
        full_predictions = full_classifier.predict(
            full_test_features
        )
        selected_scores = selected_classifier.predict_proba(
            selected_test_features
        )[:, 1]
        full_scores = full_classifier.predict_proba(
            full_test_features
        )[:, 1]

        selected_metrics = calculate_classification_metrics(
            true_labels=projected_fold.test_targets,
            predicted_labels=selected_predictions,
            scores=selected_scores,
        )
        full_metrics = calculate_classification_metrics(
            true_labels=projected_fold.test_targets,
            predicted_labels=full_predictions,
            scores=full_scores,
        )

        metric_row: dict[str, object] = {
            "outer_fold": outer_fold,
            "selected_subset": selection[
                "selected_subset"
            ],
            "selected_tasks": ",".join(
                selected_tasks
            ),
            "task_count": int(
                selection["task_count"]
            ),
            "selected_predictor_count": (
                len(selected_tasks)
                * projected_fold.pca_components_per_task
            ),
            "full_predictor_count": (
                len(TASKS)
                * projected_fold.pca_components_per_task
            ),
            "training_participants": len(
                projected_fold.training_ids
            ),
            "test_participants": len(
                projected_fold.test_ids
            ),
            "regularization_c": regularization_c,
            "selected_inner_macro_f1_mean": float(
                selection[
                    "selected_inner_macro_f1_mean"
                ]
            ),
            "selected_inner_macro_f1_std": float(
                selection[
                    "selected_inner_macro_f1_std"
                ]
            ),
            "inner_full_reference": float(
                selection[
                    "inner_full_reference"
                ]
            ),
            "competitive_threshold": float(
                selection[
                    "competitive_threshold"
                ]
            ),
            "competitive_subset_count": int(
                selection[
                    "competitive_subset_count"
                ]
            ),
            "delta": float(
                selection["delta"]
            ),
        }

        for metric_name, value in selected_metrics.items():
            metric_row[
                f"outer_test_selected_{metric_name}"
            ] = value

        for metric_name, value in full_metrics.items():
            metric_row[
                f"outer_test_full_{metric_name}"
            ] = value

        metric_row[
            "outer_test_macro_f1_difference"
        ] = (
            selected_metrics["macro_f1"]
            - full_metrics["macro_f1"]
        )
        metric_rows.append(metric_row)

        for (
            participant_id,
            true_label,
            selected_prediction,
            selected_score,
            full_prediction,
            full_score,
        ) in zip(
            projected_fold.test_ids,
            projected_fold.test_targets,
            selected_predictions,
            selected_scores,
            full_predictions,
            full_scores,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "participant_id": participant_id,
                    "outer_fold": outer_fold,
                    "selected_subset": selection[
                        "selected_subset"
                    ],
                    "true_label": int(
                        true_label
                    ),
                    "selected_predicted_label": int(
                        selected_prediction
                    ),
                    "selected_score": float(
                        selected_score
                    ),
                    "full_predicted_label": int(
                        full_prediction
                    ),
                    "full_score": float(
                        full_score
                    ),
                }
            )

    fold_metrics = pd.DataFrame(
        metric_rows
    )
    predictions = pd.DataFrame(
        prediction_rows
    )

    validate_wavlm_outer_results(
        artifact=artifact,
        labels=labels,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    return fold_metrics, predictions

def validate_wavlm_outer_results(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Verify paired outer predictions cover every participant once."""

    if len(fold_metrics) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Expected one WavLM metric row per outer fold."
        )

    if len(predictions) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} paired predictions, "
            f"found {len(predictions)}."
        )

    if predictions["participant_id"].duplicated().any():
        raise ValueError(
            "A participant has more than one WavLM outer prediction."
        )

    if set(predictions["participant_id"]) != set(
        artifact.participant_ids
    ):
        raise ValueError(
            "WavLM outer predictions do not cover the retained cohort."
        )

    label_lookup = {
        normalize_participant_id(participant_id): (
            int(target),
            int(outer_fold),
        )
        for participant_id, target, outer_fold in zip(
            labels["participant_id"],
            labels["target"],
            labels["outer_fold"],
            strict=True,
        )
    }

    for prediction in predictions.itertuples(
        index=False
    ):
        expected_target, expected_outer_fold = (
            label_lookup[
                prediction.participant_id
            ]
        )

        if prediction.true_label != expected_target:
            raise ValueError(
                "A WavLM outer prediction has an incorrect true label."
            )

        if prediction.outer_fold != expected_outer_fold:
            raise ValueError(
                "A WavLM prediction was assigned to the wrong outer fold."
            )

    predicted_columns = (
        "selected_predicted_label",
        "full_predicted_label",
    )

    for column in predicted_columns:
        if not set(predictions[column]).issubset(
            {0, 1}
        ):
            raise ValueError(
                f"Prediction column {column} is not binary."
            )

    score_values = predictions[
        [
            "selected_score",
            "full_score",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(score_values).all() or (
        score_values < 0.0
    ).any() or (
        score_values > 1.0
    ).any():
        raise ValueError(
            "WavLM outer probability scores are invalid."
        )

    metric_columns = [
        column
        for column in fold_metrics.columns
        if column.startswith("outer_test_")
    ]

    if not np.isfinite(
        fold_metrics[metric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "WavLM outer metrics contain a non-finite value."
        )

    reproduced_differences = (
        fold_metrics[
            "outer_test_selected_macro_f1"
        ]
        - fold_metrics[
            "outer_test_full_macro_f1"
        ]
    )

    if not np.allclose(
        fold_metrics[
            "outer_test_macro_f1_difference"
        ],
        reproduced_differences,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Stored WavLM Macro F1 differences are inconsistent."
        )

def print_wavlm_outer_summary(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Display the held-out WavLM selected-versus-full comparison."""

    print()
    print("WavLM outer-test evaluation validated.")
    print(
        f"Fold-metric rows: {len(fold_metrics)}"
    )
    print(
        f"Paired prediction rows: {len(predictions)}"
    )
    print()
    print("Held-out Macro F1 by outer fold:")
    print(
        fold_metrics[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "outer_test_selected_macro_f1",
                "outer_test_full_macro_f1",
                "outer_test_macro_f1_difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    selected_mean = float(
        fold_metrics[
            "outer_test_selected_macro_f1"
        ].mean()
    )
    full_mean = float(
        fold_metrics[
            "outer_test_full_macro_f1"
        ].mean()
    )

    print()
    print(
        "Selected WavLM mean Macro F1: "
        f"{selected_mean:.6f}"
    )
    print(
        "Full WavLM mean Macro F1: "
        f"{full_mean:.6f}"
    )
    print(
        "Mean paired fold difference: "
        f"{selected_mean - full_mean:+.6f}"
    )

def print_wavlm_selection_summary(
    inner_results: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    """Display the selected WavLM protocol for every outer fold."""

    print()
    print("WavLM inner task-subset selection validated.")
    print(
        f"Inner-result rows: {len(inner_results)}"
    )
    print(
        selections[
            [
                "outer_fold",
                "selected_subset",
                "task_count",
                "regularization_c",
                "selected_inner_macro_f1_mean",
                "inner_full_reference",
                "competitive_subset_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

def print_input_summary(
    artifact: WavLMEmbeddingArtifact,
    nested_folds: list[dict[str, object]],
    excluded_ids: set[str],
) -> None:
    """Print a concise confirmation before modeling is implemented."""

    outer_folds = {
        int(assignment["outer_fold"])
        for assignment in nested_folds
    }
    inner_fold_counts = {
        outer_fold: sum(
            int(assignment["outer_fold"])
            == outer_fold
            for assignment in nested_folds
        )
        for outer_fold in sorted(
            outer_folds
        )
    }

    print("WavLM task-selection inputs validated.")
    print(
        f"Participants: {artifact.participant_count}"
    )
    print(
        f"Excluded participants: {len(excluded_ids)}"
    )
    print(
        f"Recordings: {artifact.recording_count}"
    )
    print(
        "Embedding dimension per task: "
        f"{artifact.embedding_dimension}"
    )
    print(
        f"Speech tasks: {len(TASKS)}"
    )
    print(
        f"Outer folds: {len(outer_folds)}"
    )
    print(
        "Inner folds per outer fold: "
        f"{sorted(set(inner_fold_counts.values()))}"
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Model revision: {MODEL_REVISION}")

def main() -> None:
    """Validate WavLM embeddings against the frozen study design."""

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

    print_input_summary(
        artifact=artifact,
        nested_folds=nested_folds,
        excluded_ids=excluded_ids,
    )

    # PCA is the expensive preprocessing step, so each task is projected
    # once per inner fold and then reused for all 255 candidate subsets.
    projection_cache = build_inner_projection_cache(
        artifact=artifact,
        labels=labels,
        nested_folds=nested_folds,
        pca_components_per_task=(
            PCA_COMPONENTS_PER_TASK
        ),
    )

    print_projection_cache_summary(
        projection_cache
    )

    (
        regularization_results,
        selected_regularization,
    ) = select_full_assessment_regularization(
        projection_cache
    )

    print_regularization_summary(
        regularization_results
    )

    subsets = generate_task_subsets()
    raw_inner_results = evaluate_all_wavlm_subsets(
        projection_cache=projection_cache,
        subsets=subsets,
        selected_regularization=(
            selected_regularization
        ),
    )
    inner_results, selections = (
        select_competitive_wavlm_subsets(
            raw_inner_results,
            delta=PRIMARY_DELTA,
        )
    )

    validate_full_subset_reproduction(
        inner_results=inner_results,
        regularization_results=(
            regularization_results
        ),
    )

    print_wavlm_selection_summary(
        inner_results=inner_results,
        selections=selections,
    )

    fold_metrics, predictions = (
        evaluate_wavlm_outer_folds(
            artifact=artifact,
            labels=labels,
            selections=selections,
            pca_components_per_task=(
                PCA_COMPONENTS_PER_TASK
            ),
        )
    )

    print_wavlm_outer_summary(
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    bootstrap_results = run_paired_wavlm_bootstrap(
        predictions
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

    # Detailed tables stay in the ignored generated-results directory.
    # The tracked JSON contains aggregate values and no participant IDs.
    experiment_record = build_wavlm_experiment_record(
        regularization_results=regularization_results,
        selections=selections,
        fold_metrics=fold_metrics,
        projection_cache=projection_cache,
        bootstrap_summary=bootstrap_summary,
        participant_count=artifact.participant_count,
        excluded_participant_count=len(
            excluded_ids
        ),
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        embedding_dimension=(
            artifact.embedding_dimension
        ),
        pca_components_per_task=(
            PCA_COMPONENTS_PER_TASK
        ),
        delta=PRIMARY_DELTA,
    )

    save_wavlm_task_selection_results(
        regularization_results=regularization_results,
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        bootstrap_results=bootstrap_results,
        experiment_record=experiment_record,
    )
    print_generated_artifacts()

if __name__ == "__main__":
    main()

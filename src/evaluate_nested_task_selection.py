"""Run leakage-safe nested task-subset selection for VOC-ALS."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from audit_metadata import TASKS
from evaluate_baselines import (
    build_pipeline,
    load_experiment_inputs,
    validate_experiment_inputs,
)
from evaluate_task_subsets import (
    feature_columns_for_tasks,
    generate_task_subsets,
    load_frozen_configurations,
)


# A reduced protocol is competitive when its inner-CV Macro F1 is
# no more than 0.02 below the complete eight-task assessment.
PRIMARY_DELTA = 0.02


# Generated results stay outside version control because they may
# contain participant-level predictions or reproducible run outputs.
OUTPUT_DIR = Path("results/generated")

INNER_SUBSET_RESULTS_PATH = (
    OUTPUT_DIR / "nested_task_selection_inner_results.csv"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIR / "nested_task_selection_fold_metrics.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "nested_task_selection_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "nested_task_selection_summary.json"
)

# The nested procedure is compared against the previously frozen
# eight-task outer-fold baseline using the same participants and folds.
FULL_BASELINE_FOLD_METRICS_PATH = (
    OUTPUT_DIR / "full_eight_task_fold_metrics.csv"
)

def inner_assignments_for_outer_fold(
    nested_folds: list[dict],
    outer_fold: int,
) -> list[dict]:
    # Pull out the four inner folds that belong to one outer fold.
    # Sorting them makes every run process the folds in the same order.
    assignments = sorted(
        (
            assignment
            for assignment in nested_folds
            if assignment["outer_fold"] == outer_fold
        ),
        key=lambda assignment: assignment["inner_fold"],
    )

    # Each outer-training partition was divided into exactly four
    # inner folds when the frozen assignments were created.
    if len(assignments) != 4:
        raise ValueError(
            f"Expected 4 inner folds for outer fold {outer_fold}, "
            f"found {len(assignments)}."
        )

    inner_fold_values = [
        assignment["inner_fold"]
        for assignment in assignments
    ]

    # Checking the identifiers protects against missing, duplicated,
    # or unexpectedly numbered inner-fold records.
    if inner_fold_values != [0, 1, 2, 3]:
        raise ValueError(
            f"Unexpected inner-fold identifiers for outer fold "
            f"{outer_fold}: {inner_fold_values}."
        )

    # Every inner fold must reserve the same participants as the
    # untouched outer-test set.
    reference_outer_test_ids = set(
        assignments[0]["outer_test_ids"]
    )

    # The union of the first inner train and validation partitions
    # defines the complete outer-training participant set.
    first_train_ids = set(
        assignments[0]["inner_train_ids"]
    )

    first_validation_ids = set(
        assignments[0]["inner_validation_ids"]
    )

    reference_outer_training_ids = (
        first_train_ids | first_validation_ids
    )

    # We collect all validation IDs to confirm that each outer-training
    # participant is used for validation exactly once.
    all_validation_ids: list = []

    for assignment in assignments:
        train_ids = set(
            assignment["inner_train_ids"]
        )

        validation_ids = set(
            assignment["inner_validation_ids"]
        )

        outer_test_ids = set(
            assignment["outer_test_ids"]
        )

        # These checks are the main leakage barriers. No participant
        # may belong to two partitions during the same evaluation.
        if train_ids & validation_ids:
            raise ValueError(
                "Inner training and validation participants overlap."
            )

        if train_ids & outer_test_ids:
            raise ValueError(
                "Inner training and outer-test participants overlap."
            )

        if validation_ids & outer_test_ids:
            raise ValueError(
                "Inner validation and outer-test participants overlap."
            )

        # All four inner folds must protect the same outer-test group.
        if outer_test_ids != reference_outer_test_ids:
            raise ValueError(
                f"Outer-test assignments differ within outer fold "
                f"{outer_fold}."
            )

        # Each inner split should cover the same outer-training group,
        # even though the train and validation roles rotate.
        if (
            train_ids | validation_ids
        ) != reference_outer_training_ids:
            raise ValueError(
                f"Inner folds do not cover a consistent outer-training "
                f"set for outer fold {outer_fold}."
            )

        all_validation_ids.extend(validation_ids)

    # In four-fold cross-validation, each outer-training participant
    # should appear in an inner-validation fold exactly once.
    if len(all_validation_ids) != len(
        set(all_validation_ids)
    ):
        raise ValueError(
            f"An outer-training participant appears in multiple "
            f"inner-validation folds for outer fold {outer_fold}."
        )

    # The combined validation folds must cover the entire outer-training
    # set. Otherwise, some participants would never contribute to the
    # task-subset selection score.
    if set(all_validation_ids) != (
        reference_outer_training_ids
    ):
        raise ValueError(
            f"Inner-validation folds do not partition the complete "
            f"outer-training set for outer fold {outer_fold}."
        )

    return assignments

def evaluate_subset_inner_fold(
    features: pd.DataFrame,
    assignment: dict,
    tasks: tuple[str, ...],
    configuration: dict,
) -> float:
    # Convert the selected speech tasks into their corresponding
    # acoustic-feature columns. Each task contributes five predictors.
    columns = feature_columns_for_tasks(tasks)

    train_ids = set(
        assignment["inner_train_ids"]
    )

    validation_ids = set(
        assignment["inner_validation_ids"]
    )

    outer_test_ids = set(
        assignment["outer_test_ids"]
    )

    # Repeat the leakage checks at the modeling boundary. This makes
    # the function safe even if it is called independently later.
    if train_ids & validation_ids:
        raise ValueError(
            "Inner training and validation participants overlap."
        )

    if train_ids & outer_test_ids:
        raise ValueError(
            "Inner training and outer-test participants overlap."
        )

    if validation_ids & outer_test_ids:
        raise ValueError(
            "Inner validation and outer-test participants overlap."
        )

    # Build participant-level training and validation tables from the
    # frozen ID assignments.
    train = features.loc[
        features["ID"].isin(train_ids)
    ].copy()

    validation = features.loc[
        features["ID"].isin(validation_ids)
    ].copy()

    # Confirm that every frozen participant ID was found in the
    # feature table before fitting a model.
    if set(train["ID"]) != train_ids:
        raise ValueError(
            "Not all inner-training participants were found."
        )

    if set(validation["ID"]) != validation_ids:
        raise ValueError(
            "Not all inner-validation participants were found."
        )

    X_train = train[columns]
    y_train = train["target"]

    X_validation = validation[columns]
    y_validation = validation["target"]

    # build_pipeline keeps standardization inside the training process.
    # The scaler is fitted on X_train and then applied to validation,
    # which prevents preprocessing leakage.
    pipeline = build_pipeline(configuration)

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_validation
    )

    # Macro F1 gives equal weight to the normal-speech and documented-
    # impairment classes and is the locked primary selection metric.
    return float(
        f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

def evaluate_subset_inner_cv(
    features: pd.DataFrame,
    assignments: list[dict],
    tasks: tuple[str, ...],
    configuration: dict,
) -> dict:
    # One subset must be evaluated across all four inner folds before
    # it can be compared with the full eight-task assessment.
    if len(assignments) != 4:
        raise ValueError(
            f"Expected 4 inner-fold assignments, "
            f"found {len(assignments)}."
        )

    outer_fold_values = {
        int(assignment["outer_fold"])
        for assignment in assignments
    }

    # Mixing assignments from different outer folds would combine
    # different training populations and invalidate the selection.
    if len(outer_fold_values) != 1:
        raise ValueError(
            "Inner assignments belong to multiple outer folds."
        )

    inner_scores: list[float] = []

    for assignment in assignments:
        score = evaluate_subset_inner_fold(
            features=features,
            assignment=assignment,
            tasks=tasks,
            configuration=configuration,
        )

        inner_scores.append(score)

    score_series = pd.Series(
        inner_scores,
        dtype=float,
    )

    # Keep both the individual scores and their summary statistics.
    # The individual values support auditing, while the mean and SD
    # drive the deterministic task-subset selection rule.
    return {
        "outer_fold": outer_fold_values.pop(),
        "tasks": tasks,
        "subset_id": "+".join(tasks),
        "task_count": len(tasks),
        "predictor_count": len(
            feature_columns_for_tasks(tasks)
        ),
        "inner_scores": inner_scores,
        "inner_macro_f1_mean": float(
            score_series.mean()
        ),
        "inner_macro_f1_std": float(
            score_series.std(ddof=1)
        ),
    }

def evaluate_subsets_inner_cv(
    features: pd.DataFrame,
    assignments: list[dict],
    subsets: list[tuple[str, ...]],
    configuration: dict,
) -> list[dict]:
    # This function applies the same inner-CV procedure to every task
    # subset while keeping the outer fold's model configuration fixed.
    if not subsets:
        raise ValueError(
            "At least one task subset is required."
        )

    # Duplicate subsets would waste model fits and could distort later
    # result counts or selection-frequency summaries.
    if len(subsets) != len(set(subsets)):
        raise ValueError(
            "Duplicate task subsets were provided."
        )

    if not assignments:
        raise ValueError(
            "Inner-fold assignments are required."
        )

    outer_fold = int(
        assignments[0]["outer_fold"]
    )

    total_subsets = len(subsets)
    results: list[dict] = []

    for subset_index, tasks in enumerate(
        subsets,
        start=1,
    ):
        # The progress message is useful because the full run performs
        # 255 subsets times four inner folds for each outer fold.
        print(
            f"Outer fold {outer_fold} "
            f"[{subset_index}/{total_subsets}] "
            f"Evaluating {tasks}"
        )

        result = evaluate_subset_inner_cv(
            features=features,
            assignments=assignments,
            tasks=tasks,
            configuration=configuration,
        )

        results.append(result)

    return results

def select_competitive_subset(
    inner_results: list[dict],
    delta: float = PRIMARY_DELTA,
) -> dict:
    # Selection must use only inner-CV results from one outer fold.
    # Outer-test performance cannot influence this decision.
    if not inner_results:
        raise ValueError(
            "Inner-CV subset results are required."
        )

    if delta < 0:
        raise ValueError(
            "The competitive tolerance cannot be negative."
        )

    outer_fold_values = {
        int(result["outer_fold"])
        for result in inner_results
    }

    if len(outer_fold_values) != 1:
        raise ValueError(
            "Inner-CV results belong to multiple outer folds."
        )

    # The complete assessment provides the fold-specific reference
    # score against which all reduced protocols are compared.
    full_results = [
        result
        for result in inner_results
        if result["tasks"] == tuple(TASKS)
    ]

    if len(full_results) != 1:
        raise ValueError(
            "Expected exactly one full-assessment result."
        )

    inner_full_reference = float(
        full_results[0]["inner_macro_f1_mean"]
    )

    competitive_threshold = (
        inner_full_reference - delta
    )

    # A subset is competitive when its inner mean Macro F1 is within
    # delta of the full eight-task reference.
    competitive_results = [
        result
        for result in inner_results
        if result["inner_macro_f1_mean"]
        >= competitive_threshold
    ]

    # The full protocol should always meet its own threshold, so an
    # empty candidate list indicates corrupted or invalid results.
    if not competitive_results:
        raise ValueError(
            "No competitive task subset was found."
        )

    # Python compares tuple fields from left to right. This implements
    # the locked selection rule in the required priority order:
    #   1. fewer tasks
    #   2. higher mean Macro F1
    #   3. lower Macro F1 standard deviation
    #   4. lexicographic task order
    selected = min(
        competitive_results,
        key=lambda result: (
            result["task_count"],
            -result["inner_macro_f1_mean"],
            result["inner_macro_f1_std"],
            result["tasks"],
        ),
    )

    # Copy the selected result so adding selection metadata does not
    # mutate the original inner-CV result.
    selection = dict(selected)

    selection.update(
        {
            "delta": float(delta),
            "inner_full_reference": (
                inner_full_reference
            ),
            "competitive_threshold": float(
                competitive_threshold
            ),
            "competitive_subset_count": len(
                competitive_results
            ),
        }
    )

    return selection

def evaluate_selected_subset_outer_fold(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    outer_fold: int,
    selection: dict,
    configuration: dict,
) -> dict:
    # The selection must have been produced from the inner folds that
    # belong to this same outer fold.
    if int(selection["outer_fold"]) != outer_fold:
        raise ValueError(
            "The selected subset belongs to a different outer fold."
        )

    tasks = selection["tasks"]
    columns = feature_columns_for_tasks(tasks)

    # Outer-test participants have remained untouched throughout model
    # and task-subset selection. They are identified only after the
    # inner selection has been completed.
    test_ids = set(
        outer_folds.loc[
            outer_folds["outer_fold"].eq(outer_fold),
            "participant_id",
        ]
    )

    if not test_ids:
        raise ValueError(
            f"No test participants found for outer fold "
            f"{outer_fold}."
        )

    # Every participant outside the test fold becomes part of the final
    # outer-training partition.
    train = features.loc[
        ~features["ID"].isin(test_ids)
    ].copy()

    test = features.loc[
        features["ID"].isin(test_ids)
    ].copy()

    if set(train["ID"]) & set(test["ID"]):
        raise ValueError(
            "Outer training and test participants overlap."
        )

    if set(test["ID"]) != test_ids:
        raise ValueError(
            "Not all outer-test participants were found."
        )

    if len(train) + len(test) != len(features):
        raise ValueError(
            "Outer train/test partitions do not cover all participants."
        )

    X_train = train[columns]
    y_train = train["target"]

    X_test = test[columns]
    y_test = test["target"]

    # Refit the frozen model configuration on the complete outer-
    # training partition using only the selected task features.
    pipeline = build_pipeline(configuration)

    pipeline.fit(
        X_train,
        y_train,
    )

    # The outer-test partition is evaluated exactly once after task
    # selection is finished.
    predictions = pipeline.predict(X_test)

    # Probability-producing classifiers use positive-class
    # probabilities. The RBF SVM uses its decision-function score.
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Store the inner-selection evidence beside the untouched outer-test
    # metrics. This makes each fold-level decision reproducible.
    metrics = {
        "outer_fold": outer_fold,
        "selected_subset": selection["subset_id"],
        "selected_tasks": ",".join(tasks),
        "task_count": len(tasks),
        "predictor_count": len(columns),
        "training_participants": len(train),
        "test_participants": len(test),
        "inner_full_reference": selection[
            "inner_full_reference"
        ],
        "selected_inner_macro_f1_mean": selection[
            "inner_macro_f1_mean"
        ],
        "selected_inner_macro_f1_std": selection[
            "inner_macro_f1_std"
        ],
        "competitive_threshold": selection[
            "competitive_threshold"
        ],
        "competitive_subset_count": selection[
            "competitive_subset_count"
        ],
        "delta": selection["delta"],
        "outer_test_macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "outer_test_balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "outer_test_roc_auc": float(
            roc_auc_score(
                y_test,
                scores,
            )
        ),
        "outer_test_sensitivity": float(
            sensitivity
        ),
        "outer_test_specificity": float(
            specificity
        ),
        "outer_test_positive_class_f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "outer_test_accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "configuration": configuration,
    }

    # Keep one record per outer-test participant so later analyses can
    # perform paired comparisons and audit individual predictions.
    prediction_records: list[dict] = []

    for (
        participant_id,
        true_label,
        predicted_label,
        score,
    ) in zip(
        test["ID"],
        y_test,
        predictions,
        scores,
    ):
        prediction_records.append(
            {
                "participant_id": participant_id,
                "outer_fold": outer_fold,
                "selected_subset": selection[
                    "subset_id"
                ],
                "true_label": int(true_label),
                "predicted_label": int(
                    predicted_label
                ),
                "score": float(score),
            }
        )

    return {
        "metrics": metrics,
        "predictions": prediction_records,
    }

def run_nested_task_selection(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict],
    subsets: list[tuple[str, ...]],
    frozen_configurations: dict[int, dict],
    delta: float = PRIMARY_DELTA,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    # These collections become the three main experiment artifacts:
    # all inner-CV subset results, one metric record per outer fold,
    # and one prediction record per outer-test participant.
    all_inner_records: list[dict] = []
    fold_metric_records: list[dict] = []
    prediction_records: list[dict] = []

    for outer_fold in sorted(frozen_configurations):
        print()
        print(
            f"Starting nested task selection for outer fold "
            f"{outer_fold}..."
        )

        # Only the four inner folds belonging to this outer-training
        # partition may influence which task subset is selected.
        assignments = inner_assignments_for_outer_fold(
            nested_folds=nested_folds,
            outer_fold=outer_fold,
        )

                # Confirm that the nested-fold file protects the exact same
        # outer-test participants recorded in outer_folds.csv. Internal
        # consistency alone would not catch two files that disagree.
        expected_outer_test_ids = set(
            outer_folds.loc[
                outer_folds["outer_fold"].eq(
                    outer_fold
                ),
                "participant_id",
            ]
        )

        nested_outer_test_ids = set(
            assignments[0]["outer_test_ids"]
        )

        if (
            nested_outer_test_ids
            != expected_outer_test_ids
        ):
            raise ValueError(
                f"Nested and outer test assignments disagree for "
                f"outer fold {outer_fold}."
            )

        # The inner train/validation union must also equal the complete
        # participant set remaining after removal of the outer-test fold.
        expected_outer_training_ids = (
            set(features["ID"])
            - expected_outer_test_ids
        )

        nested_outer_training_ids = (
            set(assignments[0]["inner_train_ids"])
            | set(
                assignments[0][
                    "inner_validation_ids"
                ]
            )
        )

        if (
            nested_outer_training_ids
            != expected_outer_training_ids
        ):
            raise ValueError(
                f"Nested folds do not cover the complete outer-training "
                f"partition for outer fold {outer_fold}."
            )

        # This configuration was previously selected using the complete
        # eight-task assessment. Every subset in this outer fold must
        # reuse it so subsets cannot independently tune their models.
        configuration = frozen_configurations[
            outer_fold
        ]

        # Evaluate all 255 task subsets using only the four inner
        # train/validation splits for this outer fold.
        inner_results = evaluate_subsets_inner_cv(
            features=features,
            assignments=assignments,
            subsets=subsets,
            configuration=configuration,
        )

        # Apply the prespecified tolerance and deterministic tie-breakers
        # before any outer-test participant is evaluated.
        selection = select_competitive_subset(
            inner_results=inner_results,
            delta=delta,
        )

        # Convert the internal result objects into flat records that can
        # be saved cleanly as a CSV file.
        for result in inner_results:
            record = dict(result)

            inner_scores = record.pop(
                "inner_scores"
            )

            # Store task names as a readable comma-separated value
            # instead of Python's tuple representation.
            record["tasks"] = ",".join(
                result["tasks"]
            )

            # Keeping each fold score in its own column makes later
            # auditing and fold-level comparisons straightforward.
            for inner_fold, score in enumerate(
                inner_scores
            ):
                record[
                    f"inner_fold_{inner_fold}_macro_f1"
                ] = float(score)

            record["delta"] = float(delta)
            record["inner_full_reference"] = selection[
                "inner_full_reference"
            ]
            record["competitive_threshold"] = selection[
                "competitive_threshold"
            ]

            # Mark every competitive candidate and the one subset that
            # ultimately won the deterministic selection procedure.
            record["competitive"] = bool(
                result["inner_macro_f1_mean"]
                >= selection["competitive_threshold"]
            )

            record["selected"] = bool(
                result["tasks"] == selection["tasks"]
            )

            all_inner_records.append(record)

        print(
            f"Selected subset for outer fold {outer_fold}: "
            f"{selection['subset_id']}"
        )

        print(
            "Selected inner Macro F1: "
            f"{selection['inner_macro_f1_mean']:.6f}"
        )

        print(
            "Inner full-reference Macro F1: "
            f"{selection['inner_full_reference']:.6f}"
        )

        # Only now, after selection is finalized, may the chosen subset
        # be fitted on the complete outer-training partition and tested
        # once on the untouched outer-test participants.
        outer_result = (
            evaluate_selected_subset_outer_fold(
                features=features,
                outer_folds=outer_folds,
                outer_fold=outer_fold,
                selection=selection,
                configuration=configuration,
            )
        )

        fold_metric_records.append(
            outer_result["metrics"]
        )

        prediction_records.extend(
            outer_result["predictions"]
        )

        print(
            "Outer-test Macro F1: "
            f"{outer_result['metrics']['outer_test_macro_f1']:.6f}"
        )

    return (
        pd.DataFrame(all_inner_records),
        pd.DataFrame(fold_metric_records),
        pd.DataFrame(prediction_records),
    )

def validate_nested_selection_results(
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    # The complete experiment should produce 255 subset results for
    # each of the five outer folds.
    expected_outer_folds = set(range(5))
    expected_inner_rows = 255 * 5

    if len(inner_results) != expected_inner_rows:
        raise ValueError(
            f"Expected {expected_inner_rows} inner subset results, "
            f"found {len(inner_results)}."
        )

    actual_inner_folds = set(
        int(value)
        for value in inner_results["outer_fold"].unique()
    )

    if actual_inner_folds != expected_outer_folds:
        raise ValueError(
            "Inner results do not contain all five outer folds."
        )

    # Each outer fold must contain every task subset exactly once.
    subset_counts = (
        inner_results["outer_fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_subset_counts = {
        outer_fold: 255
        for outer_fold in expected_outer_folds
    }

    if subset_counts != expected_subset_counts:
        raise ValueError(
            "Each outer fold must contain 255 subset results. "
            f"Found: {subset_counts}"
        )

    if inner_results.duplicated(
        subset=[
            "outer_fold",
            "subset_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate task-subset results were found within an "
            "outer fold."
        )

    # The full eight-task reference must appear once in every outer
    # fold because its inner score defines the competitive threshold.
    full_subset_id = "+".join(TASKS)

    full_rows = inner_results.loc[
        inner_results["subset_id"].eq(
            full_subset_id
        )
    ]

    if len(full_rows) != 5:
        raise ValueError(
            "Expected one full-assessment result per outer fold."
        )

    if full_rows["outer_fold"].nunique() != 5:
        raise ValueError(
            "The full assessment is missing from an outer fold."
        )

    # The deterministic procedure must select exactly one protocol
    # inside each outer fold.
    selected_rows = inner_results.loc[
        inner_results["selected"].eq(True)
    ]

    if len(selected_rows) != 5:
        raise ValueError(
            f"Expected 5 selected subset records, "
            f"found {len(selected_rows)}."
        )

    selected_counts = (
        selected_rows["outer_fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_selected_counts = {
        outer_fold: 1
        for outer_fold in expected_outer_folds
    }

    if selected_counts != expected_selected_counts:
        raise ValueError(
            "Each outer fold must select exactly one subset. "
            f"Found: {selected_counts}"
        )

    # A selected protocol must also satisfy the competitive threshold.
    if not selected_rows["competitive"].all():
        raise ValueError(
            "A selected subset was not marked competitive."
        )

    # These columns contain the numeric evidence used during inner-CV
    # selection. NaN or infinite values would invalidate the ranking.
    inner_numeric_columns = [
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
        "inner_fold_0_macro_f1",
        "inner_fold_1_macro_f1",
        "inner_fold_2_macro_f1",
        "inner_fold_3_macro_f1",
        "inner_full_reference",
        "competitive_threshold",
    ]

    if not np.isfinite(
        inner_results[
            inner_numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A non-finite inner-CV result was found."
        )

    # One outer-test metric record should be produced for each fold.
    if len(fold_metrics) != 5:
        raise ValueError(
            f"Expected 5 outer-fold metric records, "
            f"found {len(fold_metrics)}."
        )

    actual_metric_folds = set(
        int(value)
        for value in fold_metrics["outer_fold"].unique()
    )

    if actual_metric_folds != expected_outer_folds:
        raise ValueError(
            "Outer metrics do not contain all five folds."
        )

    if fold_metrics["outer_fold"].duplicated().any():
        raise ValueError(
            "Duplicate outer-fold metric records were found."
        )

    # Confirm that the subset evaluated on each outer-test fold is the
    # same subset selected from that fold's inner-CV results.
    selected_by_fold = (
        selected_rows.set_index(
            "outer_fold"
        )["subset_id"].to_dict()
    )

    evaluated_by_fold = (
        fold_metrics.set_index(
            "outer_fold"
        )["selected_subset"].to_dict()
    )

    if selected_by_fold != evaluated_by_fold:
        raise ValueError(
            "An outer fold evaluated a subset different from its "
            "inner-selected subset."
        )

    outer_metric_columns = [
        "outer_test_macro_f1",
        "outer_test_balanced_accuracy",
        "outer_test_roc_auc",
        "outer_test_sensitivity",
        "outer_test_specificity",
        "outer_test_positive_class_f1",
        "outer_test_accuracy",
    ]

    if not np.isfinite(
        fold_metrics[
            outer_metric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A non-finite outer-test metric was found."
        )

    # Every ALS participant should receive exactly one out-of-fold
    # prediction because each participant belongs to one outer fold.
    if len(predictions) != 102:
        raise ValueError(
            f"Expected 102 outer-test predictions, "
            f"found {len(predictions)}."
        )

    if predictions["participant_id"].nunique() != 102:
        raise ValueError(
            "Outer-test predictions do not contain 102 unique "
            "participants."
        )

    if predictions["participant_id"].duplicated().any():
        raise ValueError(
            "A participant received multiple outer-test predictions."
        )

    actual_prediction_folds = set(
        int(value)
        for value in predictions["outer_fold"].unique()
    )

    if actual_prediction_folds != expected_outer_folds:
        raise ValueError(
            "Predictions do not contain all five outer folds."
        )

    # Prediction counts should match the recorded test-set size for
    # every outer fold.
    expected_predictions_by_fold = (
        fold_metrics.set_index(
            "outer_fold"
        )["test_participants"]
        .astype(int)
        .to_dict()
    )

    actual_predictions_by_fold = (
        predictions["outer_fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if (
        actual_predictions_by_fold
        != expected_predictions_by_fold
    ):
        raise ValueError(
            "Prediction counts do not match the outer-test fold sizes."
        )

    valid_labels = {0, 1}

    if not set(
        predictions["true_label"].unique()
    ).issubset(valid_labels):
        raise ValueError(
            "An invalid true label was found."
        )

    if not set(
        predictions["predicted_label"].unique()
    ).issubset(valid_labels):
        raise ValueError(
            "An invalid predicted label was found."
        )

    if not np.isfinite(
        predictions[["score"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A non-finite prediction score was found."
        )

    # Every prediction in a fold must carry the protocol selected for
    # that fold. This prevents records from being mislabeled later.
    for outer_fold, selected_subset in (
        evaluated_by_fold.items()
    ):
        prediction_subsets = set(
            predictions.loc[
                predictions["outer_fold"].eq(
                    outer_fold
                ),
                "selected_subset",
            ]
        )

        if prediction_subsets != {
            selected_subset
        }:
            raise ValueError(
                f"Prediction subset labels do not match the selected "
                f"protocol for outer fold {outer_fold}."
            )
def build_nested_selection_summary(
    fold_metrics: pd.DataFrame,
    full_baseline_metrics: pd.DataFrame,
) -> dict:
    # The existing full-assessment artifact must contain one result for
    # each frozen outer fold so the comparison remains paired by fold.
    if len(full_baseline_metrics) != 5:
        raise ValueError(
            f"Expected 5 full-baseline fold records, "
            f"found {len(full_baseline_metrics)}."
        )

    required_baseline_columns = {
        "outer_fold",
        "macro_f1",
    }

    missing_baseline_columns = (
        required_baseline_columns
        - set(full_baseline_metrics.columns)
    )

    if missing_baseline_columns:
        raise ValueError(
            "The full-baseline metrics are missing columns: "
            f"{sorted(missing_baseline_columns)}"
        )

    if full_baseline_metrics[
        "outer_fold"
    ].duplicated().any():
        raise ValueError(
            "Duplicate full-baseline outer-fold results were found."
        )

    nested_outer_folds = set(
        int(value)
        for value in fold_metrics["outer_fold"].unique()
    )

    baseline_outer_folds = set(
        int(value)
        for value in full_baseline_metrics[
            "outer_fold"
        ].unique()
    )

    if nested_outer_folds != baseline_outer_folds:
        raise ValueError(
            "Nested-selection and full-baseline folds do not match."
        )

    # Pair the selected-protocol result with the complete assessment
    # result from the same outer fold.
    comparison = fold_metrics[
        [
            "outer_fold",
            "outer_test_macro_f1",
        ]
    ].merge(
        full_baseline_metrics[
            [
                "outer_fold",
                "macro_f1",
            ]
        ],
        on="outer_fold",
        how="inner",
        validate="one_to_one",
    )

    comparison = comparison.rename(
        columns={
            "macro_f1": "full_macro_f1",
        }
    )

    comparison["macro_f1_difference"] = (
        comparison["outer_test_macro_f1"]
        - comparison["full_macro_f1"]
    )

    # Summarize every required outer-test metric using the same
    # five-fold mean and sample standard deviation convention.
    metric_columns = {
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

    outer_test_summary = {
        metric_name: {
            "mean": float(
                fold_metrics[column_name].mean()
            ),
            "std": float(
                fold_metrics[column_name].std(
                    ddof=1
                )
            ),
        }
        for metric_name, column_name in (
            metric_columns.items()
        )
    }

    # Count how often the exact same protocol was selected across the
    # five independent outer-training partitions.
    selection_frequency = {
        str(subset_id): int(count)
        for subset_id, count in (
            fold_metrics["selected_subset"]
            .value_counts()
            .items()
        )
    }

    # This distribution shows whether the selection procedure
    # consistently prefers protocols of a similar size.
    selected_task_count_distribution = {
        str(int(task_count)): int(count)
        for task_count, count in (
            fold_metrics["task_count"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    # Count individual task appearances even when the exact selected
    # task combinations differ between outer folds.
    task_counter: Counter = Counter()

    for selected_tasks in fold_metrics[
        "selected_tasks"
    ]:
        task_counter.update(
            selected_tasks.split(",")
        )

    task_frequency = {
        task: int(task_counter[task])
        for task in TASKS
    }

    # Keep the complete selection record for each outer fold so readers
    # can see both the inner evidence and untouched outer performance.
    fold_selections: list[dict] = []

    for row in (
        fold_metrics
        .sort_values("outer_fold")
        .itertuples(index=False)
    ):
        fold_selections.append(
            {
                "outer_fold": int(row.outer_fold),
                "selected_subset": (
                    row.selected_subset
                ),
                "selected_tasks": (
                    row.selected_tasks.split(",")
                ),
                "task_count": int(row.task_count),
                "predictor_count": int(
                    row.predictor_count
                ),
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
                "configuration": row.configuration,
            }
        )

    fold_comparisons: list[dict] = []

    for row in (
        comparison
        .sort_values("outer_fold")
        .itertuples(index=False)
    ):
        fold_comparisons.append(
            {
                "outer_fold": int(row.outer_fold),
                "nested_selected_macro_f1": float(
                    row.outer_test_macro_f1
                ),
                "full_macro_f1": float(
                    row.full_macro_f1
                ),
                "difference": float(
                    row.macro_f1_difference
                ),
            }
        )

    delta_values = fold_metrics[
        "delta"
    ].unique()

    if len(delta_values) != 1:
        raise ValueError(
            "Outer folds were evaluated with different delta values."
        )

    return {
        "experiment": (
            "leakage_safe_nested_task_selection"
        ),
        "population": {
            "als_participants": 102,
        },
        "evaluation": {
            "outer_folds": 5,
            "inner_folds": 4,
            "task_subsets_per_outer_fold": 255,
            "primary_metric": "macro_f1",
            "delta": float(delta_values[0]),
            "participant_disjoint": True,
            "model_configuration_frozen_per_outer_fold": True,
        },
        "fold_selections": fold_selections,
        "selection_frequency": selection_frequency,
        "selected_task_count_distribution": (
            selected_task_count_distribution
        ),
        "task_frequency": task_frequency,
        "nested_selected_outer_test_metrics": (
            outer_test_summary
        ),
        "full_eight_task_comparison": {
            "nested_selected_macro_f1_mean": float(
                comparison[
                    "outer_test_macro_f1"
                ].mean()
            ),
            "nested_selected_macro_f1_std": float(
                comparison[
                    "outer_test_macro_f1"
                ].std(ddof=1)
            ),
            "full_macro_f1_mean": float(
                comparison[
                    "full_macro_f1"
                ].mean()
            ),
            "full_macro_f1_std": float(
                comparison[
                    "full_macro_f1"
                ].std(ddof=1)
            ),
            "difference_mean": float(
                comparison[
                    "macro_f1_difference"
                ].mean()
            ),
            "difference_std": float(
                comparison[
                    "macro_f1_difference"
                ].std(ddof=1)
            ),
            "fold_comparisons": fold_comparisons,
        },
        "interpretation_status": (
            "Leakage-safe nested-CV estimate of the task-selection "
            "procedure on the VOC-ALS cohort. This is not external "
            "clinical validation and does not establish that any "
            "selected task is universally sufficient."
        ),
    }


def save_nested_selection_results(
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: dict,
) -> None:
    # Create the generated-results directory only after the complete
    # experiment has passed validation.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write each artifact to a temporary file first. Replacing the final
    # path is atomic on the same filesystem, so an interrupted write
    # cannot leave a half-written CSV or JSON document.
    inner_temp_path = Path(
        f"{INNER_SUBSET_RESULTS_PATH}.tmp"
    )

    metrics_temp_path = Path(
        f"{FOLD_METRICS_PATH}.tmp"
    )

    predictions_temp_path = Path(
        f"{PREDICTIONS_PATH}.tmp"
    )

    summary_temp_path = Path(
        f"{SUMMARY_PATH}.tmp"
    )

    temporary_paths = [
        inner_temp_path,
        metrics_temp_path,
        predictions_temp_path,
        summary_temp_path,
    ]

    try:
        inner_results.to_csv(
            inner_temp_path,
            index=False,
        )

        fold_metrics.to_csv(
            metrics_temp_path,
            index=False,
        )

        predictions.to_csv(
            predictions_temp_path,
            index=False,
        )

        summary_temp_path.write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # The summary is replaced last because it acts as the compact
        # record indicating that the complete result set was produced.
        inner_temp_path.replace(
            INNER_SUBSET_RESULTS_PATH
        )

        metrics_temp_path.replace(
            FOLD_METRICS_PATH
        )

        predictions_temp_path.replace(
            PREDICTIONS_PATH
        )

        summary_temp_path.replace(
            SUMMARY_PATH
        )

    finally:
        # Clean up temporary files if writing or replacement fails.
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

def main() -> None:
    # Load the private participant-level feature table and the frozen
    # outer and inner cross-validation assignments.
    features, outer_folds, nested_folds = (
        load_experiment_inputs()
    )

    # Stop before model fitting if the participant IDs, predictors,
    # targets, or fold assignments do not match the locked experiment.
    validate_experiment_inputs(
        features,
        outer_folds,
        nested_folds,
    )

    # Generate all 255 nonempty combinations of the eight speech tasks.
    subsets = generate_task_subsets()

    # Each outer fold reuses the model configuration previously selected
    # from that fold's complete eight-task inner-CV baseline.
    frozen_configurations = load_frozen_configurations()

    if len(subsets) != 255:
        raise ValueError(
            f"Expected 255 task subsets, found {len(subsets)}."
        )

    # The subset generator orders combinations by size, making the
    # complete eight-task assessment the final generated subset.
    if subsets[-1] != tuple(TASKS):
        raise ValueError(
            "The final generated subset is not the full assessment."
        )

    outer_fold_values = sorted(
        int(value)
        for value in outer_folds[
            "outer_fold"
        ].unique()
    )

    # Every frozen outer fold must have exactly one corresponding model
    # configuration before subset selection begins.
    if set(outer_fold_values) != set(
        frozen_configurations
    ):
        raise ValueError(
            "Frozen configurations do not match the outer folds."
        )

    print("Nested task-selection inputs validated.")
    print(f"Participants: {len(features)}")
    print(f"Task subsets: {len(subsets)}")
    print(f"Outer folds: {len(outer_fold_values)}")
    print(f"Inner folds per outer fold: 4")
    print(f"Primary delta: {PRIMARY_DELTA}")

    # Run the complete leakage-safe procedure. Each outer fold performs
    # task selection using only its four inner folds before evaluating
    # the selected protocol once on untouched outer-test participants.
    (
        inner_results,
        fold_metrics,
        predictions,
    ) = run_nested_task_selection(
        features=features,
        outer_folds=outer_folds,
        nested_folds=nested_folds,
        subsets=subsets,
        frozen_configurations=(
            frozen_configurations
        ),
        delta=PRIMARY_DELTA,
    )

    # Do not write any output until the complete experiment has passed
    # structural, participant-coverage, and finite-value validation.
    validate_nested_selection_results(
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    print()
    print("Nested task-selection results validated.")

    # The earlier full eight-task baseline uses the same frozen outer
    # folds and provides the reference for the final comparison.
    if not FULL_BASELINE_FOLD_METRICS_PATH.exists():
        raise FileNotFoundError(
            "The frozen full-baseline fold metrics were not found at "
            f"{FULL_BASELINE_FOLD_METRICS_PATH}."
        )

    full_baseline_metrics = pd.read_csv(
        FULL_BASELINE_FOLD_METRICS_PATH
    )

    summary = build_nested_selection_summary(
        fold_metrics=fold_metrics,
        full_baseline_metrics=(
            full_baseline_metrics
        ),
    )

    # Saving occurs only after modeling, result validation, and summary
    # construction have all completed successfully.
    save_nested_selection_results(
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        summary=summary,
    )

    print()
    print("Selected protocol by outer fold:")

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

    print()
    print("Nested-selection outer-test summary:")

    for metric_name, values in summary[
        "nested_selected_outer_test_metrics"
    ].items():
        print(
            f"{metric_name}: "
            f"{values['mean']:.6f} "
            f"+/- {values['std']:.6f}"
        )

    comparison = summary[
        "full_eight_task_comparison"
    ]

    print()
    print("Macro F1 comparison:")

    print(
        "Nested selected mean: "
        f"{comparison['nested_selected_macro_f1_mean']:.6f}"
    )

    print(
        "Full eight-task mean: "
        f"{comparison['full_macro_f1_mean']:.6f}"
    )

    print(
        "Mean paired fold difference: "
        f"{comparison['difference_mean']:+.6f}"
    )

    print()
    print("Generated artifacts:")
    print(INNER_SUBSET_RESULTS_PATH)
    print(FOLD_METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)

if __name__ == "__main__":
    main()
# Evaluate all non empty VOC-ALS speech task subsets

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from evaluate_baselines import (
    FEATURE_TABLE_PATH,
    OUTER_FOLDS_PATH,
    build_pipeline,
)

from audit_metadata import FEATURES, TASKS

BASELINE_SELECTIONS_PATH = Path(
    "results/generated/full_eight_task_selections.json"
)

OUTPUT_DIR = Path(
    "results/generated"
)

SUBSET_FOLD_RESULTS_PATH = OUTPUT_DIR / (
    "task_subset_fold_metrics.csv"
)

SUBSET_SUMMARY_PATH = OUTPUT_DIR / (
    "task_subset_summary.csv"
)

def generate_task_subsets() -> list[tuple[str, ...]]:
    # Return all 255 non-empty combinations of the eight tasks

    subsets: list[tuple[str, ...]]= []

    for subset_size in range(1, len(TASKS) + 1):
        subsets.extend(
            combinations(
                TASKS,
                subset_size,
            )
        )
    return subsets

def feature_columns_for_tasks(
        tasks: tuple[str, ...],
) -> list[str]:
    # return the acoustic feature columns belonging to a task subset
    return [
        f"{feature}_{task}"
        for task in tasks
        for feature in FEATURES
    ]

def load_frozen_configurations() -> dict[list, dict]:
    # Load the model config selected for each outer fold

    selections = json.loads(
        BASELINE_SELECTIONS_PATH.read_text(
            encoding = "utf-8"
        )
    )

    configurations = {
        int (selection["outer_fold"]): selection["configuration"]
        for selection in selections
    }

    if len(configurations) != 5:
        raise ValueError(
            "Expected one frozen configuration for each 5 outer folds"
        )
    
    return configurations

def evaluate_subset_outer_fold(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    tasks: tuple[str, ...],
    outer_fold: int,
    configuration: dict,
) -> dict:
    # check one task subset on one frozen outer fold.

    columns = feature_columns_for_tasks(tasks)

    test_ids = set(
        outer_folds.loc[
            outer_folds["outer_fold"].eq(outer_fold),
            "participant_id",
        ]
    )

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

    X_train = train[columns]
    y_train = train["target"]

    X_test = test[columns]
    y_test = test["target"]

    pipeline = build_pipeline(configuration)

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

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

    return {
        "outer_fold": outer_fold,
        "tasks": tasks,
        "task_count": len(tasks),
        "predictor_count": len(columns),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                scores,
            )
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "positive_class_f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
    }

def evaluate_subset_all_outer_folds(
        features: pd.DataFrame,
        outer_folds: pd.DataFrame,
        tasks: tuple[str, ...],
        frozen_configurations: dict[int, dict],
) -> pd.DataFrame:
    # evaluate one task subset across all five frozen outer tables

    fold_results: list[dict] = []

    for outer_fold in sorted(frozen_configurations):
        result = evaluate_subset_outer_fold(
            features = features,
            outer_folds = outer_folds,
            tasks = tasks,
            outer_fold = outer_fold,
            configuration = frozen_configurations[outer_fold],
        )

        fold_results.append(result)
    return pd.DataFrame(fold_results)

def summarize_subset(
        fold_results: pd.DataFrame,
) -> dict:
    # Summarize one task subset across the five outer folds.

    return {
        "macro_f1_mean": float(
            fold_results["macro_f1"].mean()
        ),
        "macro_f1_std": float(
            fold_results["macro_f1"].std(ddof=1)
        ),
    }

def evaluate_all_subsets(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    subsets: list[tuple[str, ...]],
    frozen_configurations: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all 255 task subsets across all five outer folds."""

    all_fold_results: list[pd.DataFrame] = []
    summary_records: list[dict] = []

    total_subsets = len(subsets)

    for subset_index, tasks in enumerate(subsets, start=1):
        print(
            f"[{subset_index}/{total_subsets}] "
            f"Evaluating {tasks}"
        )

        fold_results = evaluate_subset_all_outer_folds(
            features=features,
            outer_folds=outer_folds,
            tasks=tasks,
            frozen_configurations=frozen_configurations,
        )

        subset_id = "+".join(tasks)

        fold_results["subset_id"] = subset_id

        all_fold_results.append(
            fold_results
        )

        summary = summarize_subset(
            fold_results
        )

        summary_records.append(
            {
                "subset_id": subset_id,
                "tasks": ",".join(tasks),
                "task_count": len(tasks),
                "predictor_count": len(
                    feature_columns_for_tasks(tasks)
                ),
                **summary,
            }
        )

    return (
        pd.concat(
            all_fold_results,
            ignore_index=True,
        ),
        pd.DataFrame(summary_records),
    )

def validate_exhaustive_results(
    fold_results: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    # Verify that the exhaustive subset experiment is complete.

    if len(fold_results) != 1275:
        raise ValueError(
            f"Expected 1275 fold-level results, "
            f"found {len(fold_results)}."
        )

    if len(summaries) != 255:
        raise ValueError(
            f"Expected 255 subset summaries, "
            f"found {len(summaries)}."
        )

    if summaries["subset_id"].nunique() != 255:
        raise ValueError(
            "Expected 255 unique task subsets."
        )

    expected_counts = {
        1: 8,
        2: 28,
        3: 56,
        4: 70,
        5: 56,
        6: 28,
        7: 8,
        8: 1,
    }

    actual_counts = (
        summaries["task_count"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if actual_counts != expected_counts:
        raise ValueError(
            "Task-count distribution is incorrect. "
            f"Found: {actual_counts}"
        )

def main() -> None:
    # Generate all 255 non-empty task subsets
    subsets = generate_task_subsets()

    # Load the five model configurations selected by the
    # official eight-task baseline
    frozen_configurations = load_frozen_configurations()

    print(f"Tasks: {len(TASKS)}")
    print(f"Task subsets: {len(subsets)}")

    print()
    print("First 10 subsets:")

    for subset in subsets[:10]:
        print(
            subset,
            "->",
            len(feature_columns_for_tasks(subset)),
            "features",
        )

    print()
    print("Final subset:")
    print(subsets[-1])

    print()
    print(
        "Full-subset predictors:",
        len(
            feature_columns_for_tasks(
                subsets[-1]
            )
        ),
    )

    print()
    print("Frozen outer-fold configurations:")

    for outer_fold in sorted(frozen_configurations):
        print(
            f"Fold {outer_fold}:",
            frozen_configurations[outer_fold],
        )

    # Load the participant-level feature table
    # and frozen outer-fold assignments
    features = pd.read_csv(
        FEATURE_TABLE_PATH
    )

    outer_folds = pd.read_csv(
        OUTER_FOLDS_PATH
    )

    # The final generated subset is the complete
    # eight-task assessment
    full_subset = subsets[-1]

    # Evaluate the complete eight-task assessment
    # across all five outer folds using the frozen
    # configuration belonging to each fold
    full_subset_results = evaluate_subset_all_outer_folds(
        features=features,
        outer_folds=outer_folds,
        tasks=full_subset,
        frozen_configurations=frozen_configurations,
    )

    # Calculate mean and standard deviation of Macro F1
    full_subset_summary = summarize_subset(
        full_subset_results
    )

    print()
    print("Full eight-task reproduction — all outer folds:")

    print(
        full_subset_results[
            [
                "outer_fold",
                "macro_f1",
            ]
        ].to_string(index=False)
    )

    print()
    print("Reproduced full-subset summary:")

    print(
        "Macro F1 mean:",
        full_subset_summary["macro_f1_mean"],
    )

    print(
        "Macro F1 std:",
        full_subset_summary["macro_f1_std"],
    )

    print()
    print("Starting exhaustive 255-subset evaluation...")

    all_fold_results, subset_summaries = evaluate_all_subsets(
        features=features,
        outer_folds=outer_folds,
        subsets=subsets,
        frozen_configurations=frozen_configurations,
    )

    validate_exhaustive_results(
        all_fold_results,
        subset_summaries,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_fold_results.to_csv(
        SUBSET_FOLD_RESULTS_PATH,
        index=False,
    )

    subset_summaries.to_csv(
        SUBSET_SUMMARY_PATH,
        index=False,
    )

    print()
    print("Exhaustive evaluation complete.")
    print(
        f"Fold-level results: {len(all_fold_results)}"
    )
    print(
        f"Subset summaries: {len(subset_summaries)}"
    )

    print()
    print("Subsets by task count:")

    print(
        subset_summaries[
            "task_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Fold results written to:",
        SUBSET_FOLD_RESULTS_PATH,
    )

    print(
        "Subset summaries written to:",
        SUBSET_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
"""Leakage-safe nested task selection (primary handcrafted result) and its delta sensitivity.

Delta 0.02 is the locked primary rule. Deltas 0.01 and 0.05 reuse the same
inner-CV scores and only change which subset counts as competitive.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from common import (
    DELTA,
    EXPERIMENTS,
    GENERATED,
    TASKS,
    all_subsets,
    frozen_configurations,
    load_inputs,
    mean_std,
    write_csv,
    write_json,
)
from handcrafted import evaluate_selections, outer_metric_summary, run_nested_selection, select_subsets

DELTAS = (0.01, DELTA, 0.05)
INNER_SCORE_COLUMNS = [
    "outer_fold", "tasks", "subset_id", "task_count", "predictor_count",
    "inner_macro_f1_mean", "inner_macro_f1_std",
]


def fold_selection_record(row) -> dict:
    return {
        "outer_fold": int(row["outer_fold"]),
        "selected_subset": row["selected_subset"],
        "selected_tasks": row["selected_tasks"].split(","),
        "task_count": int(row["task_count"]),
        "predictor_count": int(row["predictor_count"]),
        "inner_full_reference": float(row["inner_full_reference"]),
        "selected_inner_macro_f1_mean": float(row["selected_inner_macro_f1_mean"]),
        "selected_inner_macro_f1_std": float(row["selected_inner_macro_f1_std"]),
        "competitive_threshold": float(row["competitive_threshold"]),
        "competitive_subset_count": int(row["competitive_subset_count"]),
        "outer_test_macro_f1": float(row["outer_test_macro_f1"]),
    }


def primary_summary(fold_metrics: pd.DataFrame, full_metrics: pd.DataFrame, participants: int) -> dict:
    comparison = fold_metrics[["outer_fold", "outer_test_macro_f1"]].merge(
        full_metrics[["outer_fold", "macro_f1"]].rename(columns={"macro_f1": "full_macro_f1"}), on="outer_fold"
    )
    comparison["difference"] = comparison["outer_test_macro_f1"] - comparison["full_macro_f1"]
    task_counter = Counter(task for tasks in fold_metrics["selected_tasks"] for task in tasks.split(","))
    selected, full, difference = (mean_std(comparison[c]) for c in ("outer_test_macro_f1", "full_macro_f1", "difference"))
    return {
        "experiment": "leakage_safe_nested_task_selection",
        "population": {"als_participants": participants},
        "evaluation": {
            "outer_folds": len(fold_metrics),
            "inner_folds": 4,
            "task_subsets_per_outer_fold": 255,
            "primary_metric": "macro_f1",
            "delta": DELTA,
            "participant_disjoint": True,
            "model_configuration_frozen_per_outer_fold": True,
        },
        "fold_selections": [
            {**fold_selection_record(row), "configuration": row["configuration"]} for _, row in fold_metrics.iterrows()
        ],
        "selection_frequency": {str(k): int(v) for k, v in fold_metrics["selected_subset"].value_counts().items()},
        "selected_task_count_distribution": {
            str(int(k)): int(v) for k, v in fold_metrics["task_count"].value_counts().sort_index().items()
        },
        "task_frequency": {task: int(task_counter[task]) for task in TASKS},
        "nested_selected_outer_test_metrics": outer_metric_summary(fold_metrics),
        "full_eight_task_comparison": {
            "nested_selected_macro_f1_mean": selected["mean"],
            "nested_selected_macro_f1_std": selected["std"],
            "full_macro_f1_mean": full["mean"],
            "full_macro_f1_std": full["std"],
            "difference_mean": difference["mean"],
            "difference_std": difference["std"],
            "fold_comparisons": [
                {
                    "outer_fold": int(row.outer_fold),
                    "nested_selected_macro_f1": float(row.outer_test_macro_f1),
                    "full_macro_f1": float(row.full_macro_f1),
                    "difference": float(row.difference),
                }
                for row in comparison.itertuples(index=False)
            ],
        },
        "interpretation_status": "Leakage-safe nested-CV estimate of the task-selection procedure on the VOC-ALS cohort. This is not external clinical validation and does not establish that any selected task is universally sufficient.",
    }


def delta_sensitivity(features, outer_folds, inner: pd.DataFrame, configurations):
    scores = inner[INNER_SCORE_COLUMNS + [f"inner_fold_{i}_macro_f1" for i in range(4)]]
    selections = pd.concat([select_subsets(scores, delta)[1] for delta in DELTAS], ignore_index=True)
    fold_metrics, predictions = [], []
    for delta, rows in selections.groupby("delta", sort=True):
        metrics, preds = evaluate_selections(features, outer_folds, rows, configurations)
        fold_metrics.append(metrics)
        predictions.append(preds.assign(delta=delta))
    fold_metrics = pd.concat(fold_metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True).sort_values(["delta", "outer_fold", "participant_id"])
    selections = selections[
        ["delta", "outer_fold", "selected_subset", "selected_tasks", "task_count", "predictor_count",
         "selected_inner_macro_f1_mean", "selected_inner_macro_f1_std", "inner_full_reference",
         "competitive_threshold", "competitive_subset_count"]
    ]
    return selections, fold_metrics, predictions


def delta_record(fold_metrics: pd.DataFrame, configurations, participants: int) -> dict:
    by_delta = {delta: rows.sort_values("outer_fold") for delta, rows in fold_metrics.groupby("delta")}
    primary = by_delta[DELTA].set_index("outer_fold")["outer_test_macro_f1"]
    metric_columns = [c for c in fold_metrics.columns if c.startswith("outer_test_")]
    results, comparisons = {}, {}
    for delta, rows in by_delta.items():
        task_counts = mean_std(rows["task_count"])
        results[f"{delta:.2f}"] = {
            "analysis_role": "primary" if delta == DELTA else "sensitivity",
            "fold_results": [
                {**fold_selection_record(row), **{c: float(row[c]) for c in metric_columns}}
                for _, row in rows.iterrows()
            ],
            "aggregate_outer_test_metrics": outer_metric_summary(rows),
            "mean_selected_task_count": task_counts["mean"],
            "selected_task_count_std": task_counts["std"],
        }
        if delta != DELTA:
            sensitivity = rows.set_index("outer_fold")["outer_test_macro_f1"]
            differences = sensitivity - primary
            comparisons[f"{delta:.2f}"] = {
                "mean_macro_f1_difference": float(differences.mean()),
                "std_macro_f1_difference": float(differences.std(ddof=1)),
                "fold_differences": [
                    {
                        "outer_fold": int(fold),
                        "sensitivity_macro_f1": float(sensitivity[fold]),
                        "primary_macro_f1": float(primary[fold]),
                        "difference": float(differences[fold]),
                    }
                    for fold in differences.index
                ],
            }
    return {
        "experiment": "nested_task_selection_delta_sensitivity",
        "population": {"als_participants": participants},
        "evaluation": {
            "outer_folds": len(primary),
            "task_subsets_per_outer_fold": 255,
            "primary_metric": "macro_f1",
            "primary_delta": DELTA,
            "sensitivity_deltas": [d for d in DELTAS if d != DELTA],
            "participant_disjoint": True,
            "outer_results_used_for_delta_selection": False,
        },
        "selection_rule": {
            "competitive_definition": "inner_macro_f1_mean >= inner_full_reference - delta",
            "tie_break_order": ["fewer_tasks", "higher_inner_macro_f1_mean", "lower_inner_macro_f1_std", "lexicographic_task_order"],
        },
        "frozen_configurations": {str(fold): config for fold, config in sorted(configurations.items())},
        "results_by_delta": results,
        "paired_macro_f1_comparisons_to_primary": comparisons,
        "interpretation_status": "Delta 0.02 remains the locked primary analysis. Delta 0.01 and delta 0.05 are prespecified sensitivity checks. Their outer-test performance must not be used to replace the primary delta. Results are based on participant-disjoint nested cross-validation within the VOC-ALS cohort and are not external clinical validation.",
    }


def main() -> None:
    features, outer_folds, nested_folds = load_inputs()
    configurations = frozen_configurations()
    inner_predictions, inner, fold_metrics, predictions = run_nested_selection(
        features, outer_folds, nested_folds, all_subsets(), configurations
    )
    write_csv(inner_predictions, "protocol_stability_inner_predictions.csv")
    write_csv(inner, "nested_task_selection_inner_results.csv")
    write_csv(fold_metrics, "nested_task_selection_fold_metrics.csv")
    write_csv(predictions, "nested_task_selection_predictions.csv")
    full_metrics = pd.read_csv(GENERATED / "full_eight_task_fold_metrics.csv")
    summary = primary_summary(fold_metrics, full_metrics, len(features))
    write_json(GENERATED / "nested_task_selection_summary.json", summary)
    write_json(EXPERIMENTS / "nested_task_selection.json", summary)

    selections, delta_metrics, delta_predictions = delta_sensitivity(features, outer_folds, inner, configurations)
    write_csv(selections, "delta_sensitivity_selections.csv")
    write_csv(delta_metrics, "delta_sensitivity_fold_metrics.csv")
    write_csv(delta_predictions, "delta_sensitivity_predictions.csv")
    write_json(EXPERIMENTS / "delta_sensitivity.json", delta_record(delta_metrics, configurations, len(features)))

    print(fold_metrics[["outer_fold", "selected_subset", "outer_test_macro_f1"]].to_string(index=False))
    comparison = summary["full_eight_task_comparison"]
    print(f"Nested selected {comparison['nested_selected_macro_f1_mean']:.4f} vs full {comparison['full_macro_f1_mean']:.4f}")
    print(delta_metrics.groupby("delta")[["outer_test_macro_f1", "task_count"]].mean().to_string())


if __name__ == "__main__":
    main()

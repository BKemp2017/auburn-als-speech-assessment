"""Nested selection rerun under different handling of the duplicated ALS recording.

Two ALS participants share an exact task-A recording but have opposite
targets. Each policy keeps the frozen fold assignments:
  exclusion                  drop both participants (the WavLM cohort)
  exclude_task_a             keep everyone, drop task A from every subset
  remove_target_{0,1}_duplicate  drop only one of the two participants
"""

from __future__ import annotations

import pandas as pd

from common import (
    DELTA,
    EXPERIMENTS,
    GENERATED,
    TASKS,
    all_subsets,
    duplicate_als_ids,
    frozen_configurations,
    load_inputs,
    without_participants,
    write_csv,
    write_json,
)
from handcrafted import (
    evaluate_selections,
    inner_scores,
    outer_metric_summary,
    run_nested_selection,
    select_subsets,
)

POLICY_DIR = "duplicate_policy_sensitivity"


def reference_evaluation(features, outer_folds, inner, configurations, reference):
    """Score the reference task set itself on each outer fold."""
    reference_rows = inner.loc[inner["subset_id"].eq("+".join(reference))]
    _, selections = select_subsets(reference_rows, DELTA, reference)
    return evaluate_selections(features, outer_folds, selections, configurations)


def fold_selections(fold_metrics: pd.DataFrame) -> list[dict]:
    columns = ["outer_fold", "selected_subset", "task_count", "selected_inner_macro_f1_mean", "inner_full_reference", "outer_test_macro_f1"]
    return fold_metrics[columns].sort_values("outer_fold").to_dict(orient="records")


def task_frequency(fold_metrics: pd.DataFrame, tasks) -> dict:
    return {task: int(fold_metrics["selected_subset"].map(lambda s: task in s.split("+")).sum()) for task in tasks}


def paired(selected: pd.DataFrame, reference: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    left, right = names
    table = selected[["outer_fold", "outer_test_macro_f1"]].rename(columns={"outer_test_macro_f1": left}).merge(
        reference[["outer_fold", "outer_test_macro_f1"]].rename(columns={"outer_test_macro_f1": right}), on="outer_fold"
    ).sort_values("outer_fold")
    table["difference"] = table[left] - table[right]
    return table


def exclusion(features, outer_folds, nested_folds, configurations, excluded, primary_fold_metrics):
    original_participants = len(features)
    features, outer_folds, nested_folds = without_participants(features, outer_folds, nested_folds, excluded)
    _, inner, fold_metrics, predictions = run_nested_selection(
        features, outer_folds, nested_folds, all_subsets(), configurations
    )
    full_metrics, _ = reference_evaluation(features, outer_folds, inner, configurations, TASKS)
    write_csv(inner, "duplicate_exclusion_inner_results.csv")
    write_csv(fold_metrics, "duplicate_exclusion_fold_metrics.csv")
    write_csv(predictions, "duplicate_exclusion_predictions.csv")
    write_csv(full_metrics, "duplicate_exclusion_full_fold_metrics.csv")

    versus_full = paired(fold_metrics, full_metrics, ("nested_selected_macro_f1", "full_macro_f1"))
    versus_primary = primary_fold_metrics[["outer_fold", "selected_subset", "task_count", "outer_test_macro_f1"]].rename(
        columns={"selected_subset": "primary_selected_subset", "task_count": "primary_task_count", "outer_test_macro_f1": "primary_macro_f1"}
    ).merge(
        fold_metrics[["outer_fold", "selected_subset", "task_count", "outer_test_macro_f1"]].rename(
            columns={"selected_subset": "sensitivity_selected_subset", "task_count": "sensitivity_task_count", "outer_test_macro_f1": "sensitivity_macro_f1"}
        ),
        on="outer_fold",
    ).sort_values("outer_fold")
    versus_primary["macro_f1_difference"] = versus_primary["sensitivity_macro_f1"] - versus_primary["primary_macro_f1"]
    write_json(
        EXPERIMENTS / "duplicate_exclusion_sensitivity.json",
        {
            "experiment": "duplicate_exclusion_sensitivity",
            "participant_level_fields": [],
            "primary_delta": DELTA,
            "design": {
                "original_participants": original_participants,
                "excluded_participants": len(excluded),
                "remaining_participants": len(features),
                "exclusion_reason": "Two ALS participants shared an exact A-task recording, had contradictory targets, appeared in different outer folds, and had identical A-task handcrafted predictors.",
                "fold_policy": "Preserve the frozen outer and inner assignments after filtering the two affected participants.",
            },
            "fold_selections": fold_selections(fold_metrics),
            "selection_frequency": {str(k): int(v) for k, v in fold_metrics["selected_subset"].value_counts().items()},
            "task_count_distribution": {str(int(k)): int(v) for k, v in fold_metrics["task_count"].value_counts().sort_index().items()},
            "task_frequency": task_frequency(fold_metrics, TASKS),
            "nested_selected_outer_test_metrics": outer_metric_summary(fold_metrics),
            "full_eight_task_outer_test_metrics": outer_metric_summary(full_metrics),
            "filtered_selected_vs_full": {
                "nested_selected_macro_f1_mean": float(fold_metrics["outer_test_macro_f1"].mean()),
                "full_macro_f1_mean": float(full_metrics["outer_test_macro_f1"].mean()),
                "difference_mean": float(versus_full["difference"].mean()),
                "fold_comparisons": versus_full.to_dict(orient="records"),
            },
            "comparison_with_primary_analysis": {
                "primary_macro_f1_mean": float(primary_fold_metrics["outer_test_macro_f1"].mean()),
                "sensitivity_macro_f1_mean": float(fold_metrics["outer_test_macro_f1"].mean()),
                "difference_mean": float(versus_primary["macro_f1_difference"].mean()),
                "fold_comparisons": versus_primary.to_dict(orient="records"),
            },
        },
    )
    print(f"Duplicate-filtered: selected {fold_metrics['outer_test_macro_f1'].mean():.4f} vs full {full_metrics['outer_test_macro_f1'].mean():.4f}")


def policy(name, features, outer_folds, nested_folds, configurations, excluded, reference, primary_inner_predictions):
    features, outer_folds, nested_folds = without_participants(features, outer_folds, nested_folds, excluded)
    subsets = [s for s in all_subsets() if set(s) <= set(reference)]
    if excluded:
        _, inner, selected_metrics, selected_predictions = run_nested_selection(
            features, outer_folds, nested_folds, subsets, configurations, reference
        )
    else:
        # Same participants and folds as the primary run, so its inner fits are reused.
        allowed = {"+".join(s) for s in subsets}
        inner, selections = select_subsets(
            inner_scores(primary_inner_predictions.loc[primary_inner_predictions["subset_id"].isin(allowed)]), DELTA, reference
        )
        selected_metrics, selected_predictions = evaluate_selections(features, outer_folds, selections, configurations)
    reference_metrics, reference_predictions = reference_evaluation(features, outer_folds, inner, configurations, reference)
    for label, table in [
        ("inner_results", inner),
        ("selected_fold_metrics", selected_metrics),
        ("selected_predictions", selected_predictions),
        ("reference_fold_metrics", reference_metrics),
        ("reference_predictions", reference_predictions),
    ]:
        write_csv(table, f"{POLICY_DIR}/{name}_{label}.csv")

    comparison = paired(selected_metrics, reference_metrics, ("selected_macro_f1", "reference_macro_f1"))
    print(f"{name}: selected {comparison['selected_macro_f1'].mean():.4f} vs reference {comparison['reference_macro_f1'].mean():.4f}")
    return {
        "excluded_participants": len(excluded),
        "remaining_participants": len(features),
        "available_tasks": list(reference),
        "task_subsets_per_outer_fold": len(subsets),
        "reference_assessment": "+".join(reference),
        "fold_selections": fold_selections(selected_metrics),
        "selection_frequency": {str(k): int(v) for k, v in selected_metrics["selected_subset"].value_counts().items()},
        "task_frequency": task_frequency(selected_metrics, reference),
        "mean_selected_task_count": float(selected_metrics["task_count"].mean()),
        "selected_outer_test_metrics": outer_metric_summary(selected_metrics),
        "reference_outer_test_metrics": outer_metric_summary(reference_metrics),
        "selected_vs_reference": {
            "selected_macro_f1_mean": float(comparison["selected_macro_f1"].mean()),
            "reference_macro_f1_mean": float(comparison["reference_macro_f1"].mean()),
            "difference_mean": float(comparison["difference"].mean()),
            "folds_selected_at_least_reference": int(comparison["difference"].ge(0.0).sum()),
            "fold_comparisons": comparison.to_dict(orient="records"),
        },
    }


def main() -> None:
    features, outer_folds, nested_folds = load_inputs()
    configurations = frozen_configurations()
    excluded = duplicate_als_ids()
    primary_fold_metrics = pd.read_csv(GENERATED / "nested_task_selection_fold_metrics.csv")
    exclusion(features, outer_folds, nested_folds, configurations, excluded, primary_fold_metrics)

    target = outer_folds.set_index("participant_id")["target"]
    by_target = {t: {pid for pid in excluded if target[pid] == t} for t in (0, 1)}
    primary_inner_predictions = pd.read_csv(GENERATED / "protocol_stability_inner_predictions.csv")
    without_a = tuple(t for t in TASKS if t != "A")
    policies = {
        "exclude_task_a": (set(), without_a),
        "remove_target_0_duplicate": (by_target[0], TASKS),
        "remove_target_1_duplicate": (by_target[1], TASKS),
    }
    write_json(
        EXPERIMENTS / "duplicate_policy_sensitivity.json",
        {
            "experiment": "duplicate_policy_sensitivity",
            "participant_level_fields": [],
            "primary_delta": DELTA,
            "outer_folds": 5,
            "inner_folds_per_outer_fold": 4,
            "policies": {
                name: policy(name, features, outer_folds, nested_folds, configurations, ids, reference, primary_inner_predictions)
                for name, (ids, reference) in policies.items()
            },
            "interpretation_status": "Sensitivity analysis of three defensible policies for an exact duplicated ALS recording with contradictory targets. The experiment tests whether the low-burden conclusion depends on one arbitrary duplicate-removal decision.",
        },
    )


if __name__ == "__main__":
    main()

# Here we analyze exhaustive VOC-ALS speech-task subset results.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_metadata import TASKS


SUBSET_SUMMARY_PATH = Path(
    "results/generated/task_subset_summary.csv"
)

PRIMARY_TOLERANCE = 0.02

TOLERANCES = (
    0.01,
    0.02,
    0.05,
)

def load_subset_summaries() -> pd.DataFrame:
        # Load and validate the task subset summaries

        summaries = pd.read_csv(
            SUBSET_SUMMARY_PATH
        )

        if len(summaries) != 255:
            raise ValueError(
                  f"Expected 255 subset summaries, "
                  f"found {len(summaries)}."
            )
        if summaries["subset_id"].nunique() != 255:
              raise ValueError(
                    "Expected 255 unique subset IDs."
              )

        return summaries

def get_full_reference(summaries: pd.DataFrame,) -> pd.Series:
    # Return the complete eight-task reference result.

    full_rows = summaries.loc[
        summaries["task_count"].eq(8)
    ]

    if len(full_rows) != 1:
        raise ValueError(
            "Expected exactly one eight-task reference subset."
        )

    return full_rows.iloc[0]

def calculate_thresholds(full_macro_f1: float,) -> dict[float, float]:
    # Calculate competitive Macro F1 thresholds.

    return {
        tolerance: full_macro_f1 - tolerance
        for tolerance in TOLERANCES
    }

def best_subset_per_task_count(summaries: pd.DataFrame,) -> pd.DataFrame:
    # Return the best-performing subset for each task count.

    ranked = summaries.sort_values(
        by=[
            "task_count",
            "macro_f1_mean",
            "macro_f1_std",
            "subset_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    return (
        ranked
        .groupby(
            "task_count",
            as_index=False,
            sort=True,
        )
        .first()
    )

def select_minimal_competitive_subset(summaries: pd.DataFrame,threshold: float,
) -> pd.Series:
    # Select the preferred subset meeting a Macro F1 threshold.

    competitive = summaries.loc[
        summaries["macro_f1_mean"].ge(threshold)
    ].copy()

    if competitive.empty:
        raise ValueError(
            "No subset meets the competitive threshold."
        )

    ranked = competitive.sort_values(
        by=[
            "task_count",
            "macro_f1_mean",
            "macro_f1_std",
            "subset_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    return ranked.iloc[0]

def add_reference_comparisons(summaries: pd.DataFrame,full_macro_f1: float,
) -> pd.DataFrame:
    # Add performance differences and competitive flags.

    results = summaries.copy()

    results["macro_f1_difference"] = (
        results["macro_f1_mean"]
        - full_macro_f1
    )

    for tolerance in TOLERANCES:
        column_name = (
            f"competitive_{tolerance:.2f}"
        )

        threshold = (
            full_macro_f1
            - tolerance
        )

        results[column_name] = (
            results["macro_f1_mean"]
            .ge(threshold)
        )

    return results

def task_frequency_analysis(summaries: pd.DataFrame,top_n: int = 25,
) -> pd.DataFrame:
    # Count task appearances among the top-performing subsets.

    top_subsets = (
        summaries
        .sort_values(
            "macro_f1_mean",
            ascending=False,
        )
        .head(top_n)
    )

    records: list[dict] = []

    for task in TASKS:
        appearances = (
            top_subsets["tasks"]
            .str.split(",")
            .apply(
                lambda subset: task in subset
            )
            .sum()
        )

        records.append(
            {
                "task": task,
                "appearances": int(appearances),
                "top_n": top_n,
                "frequency": float(
                    appearances / top_n
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "appearances",
            ascending=False,
        )
        .reset_index(drop=True)
    )

def main() -> None:
    summaries = load_subset_summaries()

    full_reference = get_full_reference(
        summaries
    )

    full_macro_f1 = float(
        full_reference["macro_f1_mean"]
    )

    thresholds = calculate_thresholds(
        full_macro_f1
    )

    results = add_reference_comparisons(
        summaries,
        full_macro_f1,
    )

    print("Complete eight-task reference:")
    print(
        f"Subset: {full_reference['subset_id']}"
    )
    print(
        f"Macro F1: {full_macro_f1:.6f}"
    )
    print(
        f"Macro F1 SD: "
        f"{full_reference['macro_f1_std']:.6f}"
    )

    print()
    print("Competitive thresholds:")

    for tolerance, threshold in thresholds.items():
        print(
            f"Delta {tolerance:.2f}: "
            f"{threshold:.6f}"
        )

    best_by_count = best_subset_per_task_count(
        results
    )

    print()
    print("Best subset by task count:")

    print(
        best_by_count[
            [
                "task_count",
                "subset_id",
                "macro_f1_mean",
                "macro_f1_std",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Minimal competitive subsets:")

    for tolerance in TOLERANCES:
        threshold = thresholds[tolerance]

        selected = select_minimal_competitive_subset(
            results,
            threshold,
        )

        print()
        print(
            f"Delta {tolerance:.2f}:"
        )
        print(
            f"  Subset: {selected['subset_id']}"
        )
        print(
            f"  Tasks: {int(selected['task_count'])}"
        )
        print(
            f"  Macro F1: "
            f"{selected['macro_f1_mean']:.6f}"
        )
        print(
            f"  Macro F1 SD: "
            f"{selected['macro_f1_std']:.6f}"
        )

    print()
    print("Single-task results:")

    single_tasks = (
        results.loc[
            results["task_count"].eq(1)
        ]
        .sort_values(
            "macro_f1_mean",
            ascending=False,
        )
    )

    print(
        single_tasks[
            [
                "subset_id",
                "macro_f1_mean",
                "macro_f1_std",
                "macro_f1_difference",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Competitive subset counts:")

    for tolerance in TOLERANCES:
        column_name = (
            f"competitive_{tolerance:.2f}"
        )

        count = int(
            results[column_name].sum()
        )

        print(
            f"Delta {tolerance:.2f}: "
            f"{count} / {len(results)} subsets"
        )
    task_frequency = task_frequency_analysis(
        results,
        top_n=25,
    )

    print()
    print("Task frequency among top 25 subsets:")

    print(
        task_frequency.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
"""Summarize participant-resampled protocol stability results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata import TASKS
from evaluate_nested_task_selection import (
    INNER_SUBSET_RESULTS_PATH,
    PRIMARY_DELTA,
)
from evaluate_protocol_stability import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BOOTSTRAP_SELECTIONS_PATH,
    select_bootstrap_protocols,
    validate_bootstrap_selections,
)
from evaluate_task_subsets import generate_task_subsets


EXPERIMENT_RECORD_PATH = Path(
    "experiments/protocol_stability.json"
)


def load_stability_inputs(
    subsets: list[tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate bootstrap selections and original results."""

    if not BOOTSTRAP_SELECTIONS_PATH.exists():
        raise FileNotFoundError(
            "Bootstrap selections were not found at "
            f"{BOOTSTRAP_SELECTIONS_PATH}."
        )

    if not INNER_SUBSET_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Nested inner results were not found at "
            f"{INNER_SUBSET_RESULTS_PATH}."
        )

    selections = pd.read_csv(
        BOOTSTRAP_SELECTIONS_PATH
    )

    inner_results = pd.read_csv(
        INNER_SUBSET_RESULTS_PATH
    )

    validate_bootstrap_selections(
        selections=selections,
        subsets=subsets,
    )

    required_inner_columns = {
        "outer_fold",
        "tasks",
        "subset_id",
        "task_count",
        "inner_macro_f1_mean",
        "inner_macro_f1_std",
        "selected",
    }

    missing_columns = (
        required_inner_columns
        - set(inner_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Nested inner results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    return selections, inner_results


def selected_primary_rows(
    inner_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return the original delta=0.02 selection from each outer fold."""

    selected_mask = (
        inner_results["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    selected_rows = (
        inner_results.loc[selected_mask]
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    if len(selected_rows) != 5:
        raise ValueError(
            "Expected one original selected protocol per outer fold."
        )

    if set(
        selected_rows["outer_fold"].astype(int)
    ) != set(range(5)):
        raise ValueError(
            "Original selections do not contain all outer folds."
        )

    return selected_rows


def validate_original_selection_rule(
    inner_results: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> None:
    """Confirm that the vectorized selector reproduces the primary run."""

    subset_ids = [
        "+".join(tasks)
        for tasks in subsets
    ]

    task_counts = np.asarray(
        [
            len(tasks)
            for tasks in subsets
        ],
        dtype=int,
    )

    lexicographic_order = sorted(
        range(len(subsets)),
        key=lambda index: subsets[index],
    )

    lexicographic_ranks = np.empty(
        len(subsets),
        dtype=int,
    )

    lexicographic_ranks[
        lexicographic_order
    ] = np.arange(len(subsets))

    full_subset_id = "+".join(TASKS)
    full_subset_index = subset_ids.index(
        full_subset_id
    )

    expected_rows = selected_primary_rows(
        inner_results
    )

    expected_by_fold = (
        expected_rows.set_index(
            "outer_fold"
        )["subset_id"].to_dict()
    )

    for outer_fold in range(5):
        fold_rows = (
            inner_results.loc[
                inner_results["outer_fold"].eq(
                    outer_fold
                )
            ]
            .set_index("subset_id")
            .reindex(subset_ids)
        )

        if fold_rows[
            "inner_macro_f1_mean"
        ].isna().any():
            raise ValueError(
                f"Outer fold {outer_fold} is missing a task subset."
            )

        mean_scores = fold_rows[
            "inner_macro_f1_mean"
        ].to_numpy(dtype=float)[
            np.newaxis,
            :,
        ]

        score_standard_deviations = fold_rows[
            "inner_macro_f1_std"
        ].to_numpy(dtype=float)[
            np.newaxis,
            :,
        ]

        full_reference = mean_scores[
            0,
            full_subset_index,
        ]

        competitive = (
            mean_scores
            >= full_reference - PRIMARY_DELTA
        )

        selected_index = select_bootstrap_protocols(
            mean_scores=mean_scores,
            score_standard_deviations=(
                score_standard_deviations
            ),
            competitive=competitive,
            task_counts=task_counts,
            lexicographic_ranks=(
                lexicographic_ranks
            ),
        )[0]

        reproduced_subset = subset_ids[
            selected_index
        ]

        if reproduced_subset != expected_by_fold[
            outer_fold
        ]:
            raise ValueError(
                f"Vectorized selection did not reproduce outer fold "
                f"{outer_fold}: expected "
                f"{expected_by_fold[outer_fold]}, found "
                f"{reproduced_subset}."
            )


def protocol_frequency_records(
    selected_subsets: pd.Series,
) -> list[dict[str, object]]:
    """Convert exact-protocol counts into JSON-ready records."""

    counts = selected_subsets.value_counts()
    total = int(counts.sum())

    return [
        {
            "subset_id": str(subset_id),
            "tasks": str(subset_id).split("+"),
            "selection_count": int(count),
            "selection_rate": float(
                count / total
            ),
        }
        for subset_id, count in counts.items()
    ]


def task_frequency_record(
    selected_tasks: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Count how often each individual task appears."""

    task_counts = (
        selected_tasks
        .str.split(",")
        .explode()
        .value_counts()
        .reindex(TASKS, fill_value=0)
    )

    selection_count = len(selected_tasks)

    return {
        task: {
            "selection_count": int(
                task_counts[task]
            ),
            "selection_rate": float(
                task_counts[task]
                / selection_count
            ),
        }
        for task in TASKS
    }


def task_count_distribution(
    task_counts: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Summarize the selected assessment burden."""

    counts = (
        task_counts
        .value_counts()
        .sort_index()
    )

    total = int(counts.sum())

    return {
        str(int(task_count)): {
            "selection_count": int(count),
            "selection_rate": float(
                count / total
            ),
        }
        for task_count, count in counts.items()
    }


def jaccard_similarity(
    selected_tasks: str,
    original_tasks: tuple[str, ...],
) -> float:
    """Measure task overlap with the original fold selection."""

    selected = set(
        str(selected_tasks).split(",")
    )

    original = set(original_tasks)

    return float(
        len(selected & original)
        / len(selected | original)
    )


def build_stability_record(
    selections: pd.DataFrame,
    inner_results: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> dict[str, object]:
    """Build the aggregate stability record tracked by Git."""

    original_rows = selected_primary_rows(
        inner_results
    )

    original_by_fold = {
        int(row.outer_fold): {
            "subset_id": str(row.subset_id),
            "tasks": str(row.tasks).split(","),
            "task_count": int(row.task_count),
        }
        for row in original_rows.itertuples(
            index=False
        )
    }

    per_outer_fold: list[dict[str, object]] = []

    for outer_fold in range(5):
        fold_selections = selections.loc[
            selections["outer_fold"].eq(
                outer_fold
            )
        ].copy()

        original = original_by_fold[
            outer_fold
        ]

        exact_match_rate = float(
            fold_selections[
                "selected_subset"
            ].eq(
                original["subset_id"]
            ).mean()
        )

        jaccard_values = fold_selections[
            "selected_tasks"
        ].map(
            lambda tasks: jaccard_similarity(
                selected_tasks=tasks,
                original_tasks=tuple(
                    original["tasks"]
                ),
            )
        )

        frequencies = (
            fold_selections[
                "selected_subset"
            ].value_counts(
                normalize=True
            )
        )

        entropy = float(
            -np.sum(
                frequencies
                * np.log(frequencies)
            )
        )

        normalized_entropy = float(
            entropy / np.log(len(subsets))
        )

        per_outer_fold.append(
            {
                "outer_fold": outer_fold,
                "original_selection": original,
                "original_selection_retention_rate": (
                    exact_match_rate
                ),
                "mean_jaccard_similarity_to_original": float(
                    jaccard_values.mean()
                ),
                "unique_selected_protocols": int(
                    fold_selections[
                        "selected_subset"
                    ].nunique()
                ),
                "selection_entropy": entropy,
                "normalized_selection_entropy": (
                    normalized_entropy
                ),
                "mean_selected_task_count": float(
                    fold_selections[
                        "task_count"
                    ].mean()
                ),
                "exact_protocol_frequency": (
                    protocol_frequency_records(
                        fold_selections[
                            "selected_subset"
                        ]
                    )
                ),
                "task_frequency": task_frequency_record(
                    fold_selections[
                        "selected_tasks"
                    ]
                ),
                "task_count_distribution": (
                    task_count_distribution(
                        fold_selections[
                            "task_count"
                        ]
                    )
                ),
            }
        )

    total_selections = len(selections)

    one_or_two_task_rate = float(
        selections["task_count"]
        .le(2)
        .mean()
    )

    one_to_three_task_rate = float(
        selections["task_count"]
        .le(3)
        .mean()
    )

    full_subset_id = "+".join(TASKS)

    full_selection_count = int(
        selections["selected_subset"]
        .eq(full_subset_id)
        .sum()
    )

    return {
        "experiment": "participant_resampled_protocol_stability",
        "population": {
            "als_participants": 102,
        },
        "evaluation": {
            "outer_folds": 5,
            "inner_folds_per_outer_fold": 4,
            "task_subsets": len(subsets),
            "primary_metric": "macro_f1",
            "delta": PRIMARY_DELTA,
            "bootstrap_replicates_per_outer_fold": (
                BOOTSTRAP_REPLICATES
            ),
            "total_bootstrap_selections": (
                total_selections
            ),
            "random_seed": BOOTSTRAP_SEED,
            "resampling_unit": "participant",
            "resampling_design": (
                "paired within inner fold with replacement"
            ),
            "selection_uses_outer_test_results": False,
        },
        "overall_stability": {
            "mean_selected_task_count": float(
                selections["task_count"].mean()
            ),
            "one_or_two_task_selection_rate": (
                one_or_two_task_rate
            ),
            "one_to_three_task_selection_rate": (
                one_to_three_task_rate
            ),
            "full_assessment_selection_count": (
                full_selection_count
            ),
            "full_assessment_selection_rate": float(
                full_selection_count
                / total_selections
            ),
            "unique_selected_protocols": int(
                selections[
                    "selected_subset"
                ].nunique()
            ),
            "exact_protocol_frequency": (
                protocol_frequency_records(
                    selections[
                        "selected_subset"
                    ]
                )
            ),
            "task_frequency": task_frequency_record(
                selections[
                    "selected_tasks"
                ]
            ),
            "task_count_distribution": (
                task_count_distribution(
                    selections[
                        "task_count"
                    ]
                )
            ),
        },
        "per_outer_fold": per_outer_fold,
        "interpretation_status": (
            "Participant resampling measures selection stability "
            "conditional on the frozen folds, fitted inner models, "
            "published acoustic features, and VOC-ALS cohort. Exact "
            "protocol instability does not invalidate a stable burden "
            "conclusion. These results are not external clinical "
            "validation and do not establish that one task is "
            "universally sufficient."
        ),
    }


def validate_stability_record(
    record: dict[str, object],
) -> None:
    """Validate aggregate totals and exclude participant-level data."""

    evaluation = record["evaluation"]
    overall = record["overall_stability"]

    expected_total = (
        5 * BOOTSTRAP_REPLICATES
    )

    if (
        evaluation[
            "total_bootstrap_selections"
        ]
        != expected_total
    ):
        raise ValueError(
            "The experiment record contains the wrong selection total."
        )

    protocol_total = sum(
        record["selection_count"]
        for record in overall[
            "exact_protocol_frequency"
        ]
    )

    if protocol_total != expected_total:
        raise ValueError(
            "Exact-protocol frequencies do not sum to the total."
        )

    task_count_total = sum(
        values["selection_count"]
        for values in overall[
            "task_count_distribution"
        ].values()
    )

    if task_count_total != expected_total:
        raise ValueError(
            "Task-count frequencies do not sum to the total."
        )

    if len(record["per_outer_fold"]) != 5:
        raise ValueError(
            "The record does not contain all outer folds."
        )

    forbidden_fields = {
        "participant_id",
        "true_label",
        "predicted_label",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            keys = set(value)

            for nested_value in value.values():
                keys.update(
                    collect_keys(nested_value)
                )

            return keys

        if isinstance(value, list):
            keys: set[str] = set()

            for nested_value in value:
                keys.update(
                    collect_keys(nested_value)
                )

            return keys

        return set()

    present_forbidden_fields = (
        collect_keys(record)
        & forbidden_fields
    )

    if present_forbidden_fields:
        raise ValueError(
            "The tracked record contains participant-level fields: "
            f"{sorted(present_forbidden_fields)}"
        )

    json.dumps(
        record,
        allow_nan=False,
    )


def save_stability_record(
    record: dict[str, object],
) -> None:
    """Atomically save the tracked protocol-stability record."""

    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        EXPERIMENT_RECORD_PATH.with_suffix(
            ".json.tmp"
        )
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
    """Validate and summarize protocol stability results."""

    subsets = generate_task_subsets()

    selections, inner_results = (
        load_stability_inputs(
            subsets
        )
    )

    validate_original_selection_rule(
        inner_results=inner_results,
        subsets=subsets,
    )

    record = build_stability_record(
        selections=selections,
        inner_results=inner_results,
        subsets=subsets,
    )

    validate_stability_record(
        record
    )

    save_stability_record(
        record
    )

    overall = record[
        "overall_stability"
    ]

    print("Protocol-stability record validated.")
    print(
        "Mean selected task count:",
        f"{overall['mean_selected_task_count']:.6f}",
    )
    print(
        "One-or-two-task selection rate:",
        f"{overall['one_or_two_task_selection_rate']:.6f}",
    )
    print(
        "One-to-three-task selection rate:",
        f"{overall['one_to_three_task_selection_rate']:.6f}",
    )
    print(
        "Full-assessment selection rate:",
        f"{overall['full_assessment_selection_rate']:.6f}",
    )
    print()

    print("Per-fold stability:")

    stability_rows = pd.DataFrame(
        [
            {
                "outer_fold": result["outer_fold"],
                "original_subset": result[
                    "original_selection"
                ]["subset_id"],
                "retention_rate": result[
                    "original_selection_retention_rate"
                ],
                "mean_jaccard": result[
                    "mean_jaccard_similarity_to_original"
                ],
                "unique_protocols": result[
                    "unique_selected_protocols"
                ],
                "normalized_entropy": result[
                    "normalized_selection_entropy"
                ],
            }
            for result in record[
                "per_outer_fold"
            ]
        ]
    )

    print(
        stability_rows.to_string(
            index=False
        )
    )

    print()
    print(
        "Tracked experiment record:",
        EXPERIMENT_RECORD_PATH,
    )


if __name__ == "__main__":
    main()
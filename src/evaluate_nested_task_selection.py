"""Run leakage-safe nested task-subset selection for VOC-ALS."""

from __future__ import annotations

from pathlib import Path

from audit_metadata import TASKS
from evaluate_baselines import (
    load_experiment_inputs,
    validate_experiment_inputs,
)
from evaluate_task_subsets import (
    generate_task_subsets,
    load_frozen_configurations,
)

PRIMARY_DELTA = 0.02

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


def inner_assignments_for_outer_fold(
    nested_folds: list[dict],
    outer_fold: int,
) -> list[dict]:
    """Return and validate the four inner folds for one outer fold."""

    assignments = sorted(
        (
            assignment
            for assignment in nested_folds
            if assignment["outer_fold"] == outer_fold
        ),
        key=lambda assignment: assignment["inner_fold"],
    )

    if len(assignments) != 4:
        raise ValueError(
            f"Expected 4 inner folds for outer fold {outer_fold}, "
            f"found {len(assignments)}."
        )

    inner_fold_values = [
        assignment["inner_fold"]
        for assignment in assignments
    ]

    if inner_fold_values != [0, 1, 2, 3]:
        raise ValueError(
            f"Unexpected inner-fold identifiers for outer fold "
            f"{outer_fold}: {inner_fold_values}."
        )

    reference_outer_test_ids = set(
        assignments[0]["outer_test_ids"]
    )

    first_train_ids = set(
        assignments[0]["inner_train_ids"]
    )
    first_validation_ids = set(
        assignments[0]["inner_validation_ids"]
    )
    reference_outer_training_ids = (
        first_train_ids | first_validation_ids
    )

    all_validation_ids: list = []

    for assignment in assignments:
        train_ids = set(assignment["inner_train_ids"])
        validation_ids = set(
            assignment["inner_validation_ids"]
        )
        outer_test_ids = set(
            assignment["outer_test_ids"]
        )

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

        if outer_test_ids != reference_outer_test_ids:
            raise ValueError(
                f"Outer-test assignments differ within outer fold "
                f"{outer_fold}."
            )

        if train_ids | validation_ids != (
            reference_outer_training_ids
        ):
            raise ValueError(
                f"Inner folds do not cover a consistent outer-training "
                f"set for outer fold {outer_fold}."
            )

        all_validation_ids.extend(validation_ids)

    if len(all_validation_ids) != len(
        set(all_validation_ids)
    ):
        raise ValueError(
            f"An outer-training participant appears in multiple "
            f"inner-validation folds for outer fold {outer_fold}."
        )

    if set(all_validation_ids) != (
        reference_outer_training_ids
    ):
        raise ValueError(
            f"Inner-validation folds do not partition the complete "
            f"outer-training set for outer fold {outer_fold}."
        )

    return assignments


def main() -> None:
    features, outer_folds, nested_folds = (
        load_experiment_inputs()
    )

    validate_experiment_inputs(
        features,
        outer_folds,
        nested_folds,
    )

    subsets = generate_task_subsets()
    frozen_configurations = load_frozen_configurations()

    if len(subsets) != 255:
        raise ValueError(
            f"Expected 255 task subsets, found {len(subsets)}."
        )

    if subsets[-1] != tuple(TASKS):
        raise ValueError(
            "The final generated subset is not the full assessment."
        )

    outer_fold_values = sorted(
        int(value)
        for value in outer_folds["outer_fold"].unique()
    )

    if set(outer_fold_values) != set(
        frozen_configurations
    ):
        raise ValueError(
            "Frozen configurations do not match the outer folds."
        )

    print("Nested task-selection inputs validated.")
    print(f"Participants: {len(features)}")
    print(f"Task subsets: {len(subsets)}")
    print(f"Primary delta: {PRIMARY_DELTA}")

    for outer_fold in outer_fold_values:
        assignments = inner_assignments_for_outer_fold(
            nested_folds,
            outer_fold,
        )

        print(
            f"Outer fold {outer_fold}: "
            f"{len(assignments)} validated inner folds"
        )


if __name__ == "__main__":
    main()

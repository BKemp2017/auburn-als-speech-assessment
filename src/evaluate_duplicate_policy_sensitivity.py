"""Evaluate alternative policies for handling the duplicate ALS recording."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from audit_duplicate_impact import normalize_participant_id
from audit_metadata import TASKS
from evaluate_baselines import load_experiment_inputs
from evaluate_duplicate_exclusion_sensitivity import (
    filter_modeling_inputs,
    filter_nested_folds,
    identify_duplicate_connected_als_ids,
    normalized_id_set,
    summarize_metric_table,
)
from evaluate_nested_task_selection import (
    PRIMARY_DELTA,
    evaluate_selected_subset_outer_fold,
    run_nested_task_selection,
)
from evaluate_task_subsets import (
    generate_task_subsets,
    load_frozen_configurations,
)


WAVEFORM_AUDIT_PATH = Path(
    "results/generated/raw_audio_waveform_audit.csv"
)
RESULTS_DIRECTORY = Path(
    "results/generated/duplicate_policy_sensitivity"
)
SUMMARY_PATH = Path(
    "experiments/duplicate_policy_sensitivity.json"
)

EXPECTED_ORIGINAL_PARTICIPANTS = 102
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS_PER_OUTER_FOLD = 4

POLICY_EXCLUDE_TASK_A = "exclude_task_a"
POLICY_REMOVE_TARGET_0 = "remove_target_0_duplicate"
POLICY_REMOVE_TARGET_1 = "remove_target_1_duplicate"

@dataclass(frozen=True)
class PolicyScenario:
    """Hold one duplicate policy and its filtered experiment inputs."""

    name: str
    excluded_ids: frozenset[str]
    features: pd.DataFrame
    outer_folds: pd.DataFrame
    nested_folds: list[dict[str, object]]
    subsets: list[tuple[str, ...]]
    reference_tasks: tuple[str, ...]

    @property
    def expected_participants(self) -> int:
        """Return the cohort size implied by this exclusion policy."""

        return (
            EXPECTED_ORIGINAL_PARTICIPANTS
            - len(self.excluded_ids)
        )

    @property
    def expected_subset_count(self) -> int:
        """Return the number of nonempty subsets in the task universe."""

        return (2 ** len(self.reference_tasks)) - 1

@dataclass(frozen=True)
class ScenarioResult:
    """Hold the selected and reference results for one policy."""

    inner_results: pd.DataFrame
    selected_fold_metrics: pd.DataFrame
    selected_predictions: pd.DataFrame
    reference_fold_metrics: pd.DataFrame
    reference_predictions: pd.DataFrame

def identify_duplicate_ids_by_target(
    outer_folds: pd.DataFrame,
) -> dict[int, frozenset[str]]:
    """Separate the affected ALS participants by severity target."""

    duplicate_ids = identify_duplicate_connected_als_ids(
        WAVEFORM_AUDIT_PATH
    )

    target_by_id = {
        normalize_participant_id(participant_id): int(target)
        for participant_id, target in zip(
            outer_folds["participant_id"],
            outer_folds["target"],
            strict=True,
        )
    }

    ids_by_target = {
        target: frozenset(
            participant_id
            for participant_id in duplicate_ids
            if target_by_id.get(participant_id) == target
        )
        for target in (0, 1)
    }

    for target, participant_ids in ids_by_target.items():
        if len(participant_ids) != 1:
            raise ValueError(
                "Expected exactly one duplicate-connected participant "
                f"with target {target}, but found {len(participant_ids)}."
            )

    if set().union(*ids_by_target.values()) != duplicate_ids:
        raise ValueError(
            "A duplicate-connected participant has an unexpected target."
        )

    return ids_by_target

def subsets_for_tasks(
    all_subsets: Iterable[tuple[str, ...]],
    allowed_tasks: tuple[str, ...],
) -> list[tuple[str, ...]]:
    """Keep subsets composed entirely of the allowed speech tasks."""

    allowed_task_set = set(allowed_tasks)

    return [
        subset
        for subset in all_subsets
        if set(subset).issubset(allowed_task_set)
    ]

def build_policy_scenario(
    name: str,
    excluded_ids: frozenset[str],
    reference_tasks: tuple[str, ...],
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict[str, object]],
    all_subsets: list[tuple[str, ...]],
) -> PolicyScenario:
    """Filter the cohort, folds, and task space for one policy."""

    scenario_features, scenario_outer_folds = filter_modeling_inputs(
        features=features,
        labels=outer_folds,
        excluded_ids=set(excluded_ids),
    )

    scenario_nested_folds = filter_nested_folds(
        nested_folds=nested_folds,
        excluded_ids=set(excluded_ids),
    )

    return PolicyScenario(
        name=name,
        excluded_ids=excluded_ids,
        features=scenario_features,
        outer_folds=scenario_outer_folds,
        nested_folds=scenario_nested_folds,
        subsets=subsets_for_tasks(
            all_subsets=all_subsets,
            allowed_tasks=reference_tasks,
        ),
        reference_tasks=reference_tasks,
    )

def build_policy_scenarios(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict[str, object]],
) -> list[PolicyScenario]:
    """Build the three plausible duplicate-handling scenarios."""

    duplicate_ids_by_target = identify_duplicate_ids_by_target(
        outer_folds
    )
    all_subsets = generate_task_subsets()
    tasks_without_a = tuple(
        task
        for task in TASKS
        if task != "A"
    )

    # Each definition changes one decision at a time: remove only the
    # contaminated task, or retain one side of the contradictory pair.
    definitions = (
        (
            POLICY_EXCLUDE_TASK_A,
            frozenset(),
            tasks_without_a,
        ),
        (
            POLICY_REMOVE_TARGET_0,
            duplicate_ids_by_target[0],
            tuple(TASKS),
        ),
        (
            POLICY_REMOVE_TARGET_1,
            duplicate_ids_by_target[1],
            tuple(TASKS),
        ),
    )

    return [
        build_policy_scenario(
            name=name,
            excluded_ids=excluded_ids,
            reference_tasks=reference_tasks,
            features=features,
            outer_folds=outer_folds,
            nested_folds=nested_folds,
            all_subsets=all_subsets,
        )
        for name, excluded_ids, reference_tasks in definitions
    ]

def targets_for_ids(
    participant_ids: set[str],
    target_by_id: dict[str, int],
) -> set[int]:
    """Return the targets represented by one participant split."""

    return {
        target_by_id[participant_id]
        for participant_id in participant_ids
    }

def validate_policy_scenario(
    scenario: PolicyScenario,
) -> None:
    """Validate one policy without exposing participant identifiers."""

    if len(scenario.features) != scenario.expected_participants:
        raise ValueError(
            f"Policy {scenario.name} expected "
            f"{scenario.expected_participants} feature rows, but found "
            f"{len(scenario.features)}."
        )

    if len(scenario.outer_folds) != scenario.expected_participants:
        raise ValueError(
            f"Policy {scenario.name} expected "
            f"{scenario.expected_participants} label rows, but found "
            f"{len(scenario.outer_folds)}."
        )

    feature_ids = normalized_id_set(
        scenario.features["ID"]
    )
    outer_fold_ids = normalized_id_set(
        scenario.outer_folds["participant_id"]
    )

    if feature_ids != outer_fold_ids:
        raise ValueError(
            f"Policy {scenario.name} has mismatched feature and label IDs."
        )

    if feature_ids.intersection(scenario.excluded_ids):
        raise ValueError(
            f"Policy {scenario.name} still contains an excluded participant."
        )

    expected_assignments = (
        EXPECTED_OUTER_FOLDS
        * EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    )

    if len(scenario.nested_folds) != expected_assignments:
        raise ValueError(
            f"Policy {scenario.name} expected {expected_assignments} "
            f"nested assignments, but found {len(scenario.nested_folds)}."
        )

    target_by_id = {
        normalize_participant_id(participant_id): int(target)
        for participant_id, target in zip(
            scenario.outer_folds["participant_id"],
            scenario.outer_folds["target"],
            strict=True,
        )
    }
    outer_test_ids_by_fold: dict[int, set[str]] = {}

    for assignment in scenario.nested_folds:
        outer_fold = int(assignment["outer_fold"])
        split_ids = {
            "inner training": normalized_id_set(
                assignment["inner_train_ids"]
            ),
            "inner validation": normalized_id_set(
                assignment["inner_validation_ids"]
            ),
            "outer test": normalized_id_set(
                assignment["outer_test_ids"]
            ),
        }

        combined_ids = set().union(*split_ids.values())
        combined_size = sum(
            len(participant_ids)
            for participant_ids in split_ids.values()
        )

        if combined_ids != feature_ids or combined_size != len(feature_ids):
            raise ValueError(
                f"Policy {scenario.name} has overlapping or incomplete "
                "nested-fold membership."
            )

        for split_name, participant_ids in split_ids.items():
            if targets_for_ids(participant_ids, target_by_id) != {0, 1}:
                raise ValueError(
                    f"Policy {scenario.name} {split_name} split does not "
                    "contain both targets."
                )

        outer_test_ids = split_ids["outer test"]

        if outer_fold in outer_test_ids_by_fold:
            if outer_test_ids_by_fold[outer_fold] != outer_test_ids:
                raise ValueError(
                    f"Policy {scenario.name} changes outer-test membership "
                    "across inner folds."
                )
        else:
            outer_test_ids_by_fold[outer_fold] = outer_test_ids

    if set(outer_test_ids_by_fold) != set(range(EXPECTED_OUTER_FOLDS)):
        raise ValueError(
            f"Policy {scenario.name} does not contain five outer folds."
        )

    all_outer_test_ids = set().union(*outer_test_ids_by_fold.values())
    total_outer_test_ids = sum(
        len(participant_ids)
        for participant_ids in outer_test_ids_by_fold.values()
    )

    if (
        all_outer_test_ids != feature_ids
        or total_outer_test_ids != len(feature_ids)
    ):
        raise ValueError(
            f"Policy {scenario.name} outer folds overlap or miss participants."
        )

    if len(scenario.subsets) != scenario.expected_subset_count:
        raise ValueError(
            f"Policy {scenario.name} expected "
            f"{scenario.expected_subset_count} subsets, but found "
            f"{len(scenario.subsets)}."
        )

    if len(set(scenario.subsets)) != scenario.expected_subset_count:
        raise ValueError(
            f"Policy {scenario.name} contains duplicate task subsets."
        )

    if scenario.subsets[-1] != scenario.reference_tasks:
        raise ValueError(
            f"Policy {scenario.name} does not end with its full reference."
        )

def print_policy_scenario_summary(
    scenarios: list[PolicyScenario],
) -> None:
    """Print participant-free details for the validated policies."""

    rows = []

    for scenario in scenarios:
        target_counts = scenario.outer_folds["target"].value_counts()

        rows.append(
            {
                "policy": scenario.name,
                "excluded_participants": len(scenario.excluded_ids),
                "remaining_participants": len(scenario.features),
                "target_0": int(target_counts.get(0, 0)),
                "target_1": int(target_counts.get(1, 0)),
                "available_tasks": len(scenario.reference_tasks),
                "task_subsets": len(scenario.subsets),
                "full_reference": "+".join(scenario.reference_tasks),
            }
        )

    print("Duplicate-policy sensitivity inputs validated.")
    print()
    print(
        pd.DataFrame(rows).to_string(
            index=False
        )
    )

def evaluate_reference_outer_folds(
    scenario: PolicyScenario,
    inner_results: pd.DataFrame,
    frozen_configurations: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each scenario's complete reference assessment."""

    reference_subset = "+".join(
        scenario.reference_tasks
    )
    metric_records: list[dict] = []
    prediction_records: list[dict] = []

    # The nested search already evaluated the reference subset in every
    # inner fold. Reuse those scores and fit only the five outer models
    # needed for the selected-versus-reference comparison.
    for outer_fold in sorted(frozen_configurations):
        reference_rows = inner_results.loc[
            inner_results["outer_fold"].eq(outer_fold)
            & inner_results["subset_id"].eq(reference_subset)
        ]

        if len(reference_rows) != 1:
            raise ValueError(
                f"Policy {scenario.name} expected one reference row "
                f"for outer fold {outer_fold}, but found "
                f"{len(reference_rows)}."
            )

        reference_row = reference_rows.iloc[0]
        reference_mean = float(
            reference_row["inner_macro_f1_mean"]
        )

        # This selection-shaped record lets the existing outer evaluator
        # run the reference protocol without repeating its inner-CV fits.
        reference_selection = {
            "outer_fold": outer_fold,
            "tasks": scenario.reference_tasks,
            "subset_id": reference_subset,
            "task_count": len(scenario.reference_tasks),
            "predictor_count": int(
                reference_row["predictor_count"]
            ),
            "inner_macro_f1_mean": reference_mean,
            "inner_macro_f1_std": float(
                reference_row["inner_macro_f1_std"]
            ),
            "inner_full_reference": reference_mean,
            "competitive_threshold": (
                reference_mean - PRIMARY_DELTA
            ),
            "competitive_subset_count": 1,
            "delta": PRIMARY_DELTA,
        }

        outer_result = evaluate_selected_subset_outer_fold(
            features=scenario.features,
            outer_folds=scenario.outer_folds,
            outer_fold=outer_fold,
            selection=reference_selection,
            configuration=frozen_configurations[outer_fold],
        )

        metric_records.append(
            outer_result["metrics"]
        )
        prediction_records.extend(
            outer_result["predictions"]
        )

    return (
        pd.DataFrame(metric_records),
        pd.DataFrame(prediction_records),
    )

def run_policy_scenario(
    scenario: PolicyScenario,
    frozen_configurations: dict[int, dict],
) -> ScenarioResult:
    """Run nested selection and the paired reference for one policy."""

    print()
    print(f"Running policy: {scenario.name}")

    (
        inner_results,
        selected_fold_metrics,
        selected_predictions,
    ) = run_nested_task_selection(
        features=scenario.features,
        outer_folds=scenario.outer_folds,
        nested_folds=scenario.nested_folds,
        subsets=scenario.subsets,
        frozen_configurations=frozen_configurations,
        delta=PRIMARY_DELTA,
        reference_tasks=scenario.reference_tasks,
    )

    (
        reference_fold_metrics,
        reference_predictions,
    ) = evaluate_reference_outer_folds(
        scenario=scenario,
        inner_results=inner_results,
        frozen_configurations=frozen_configurations,
    )

    return ScenarioResult(
        inner_results=inner_results,
        selected_fold_metrics=selected_fold_metrics,
        selected_predictions=selected_predictions,
        reference_fold_metrics=reference_fold_metrics,
        reference_predictions=reference_predictions,
    )

def validate_prediction_table(
    predictions: pd.DataFrame,
    scenario: PolicyScenario,
    table_name: str,
) -> None:
    """Validate one participant-level outer-test prediction table."""

    if len(predictions) != scenario.expected_participants:
        raise ValueError(
            f"Policy {scenario.name} {table_name} expected "
            f"{scenario.expected_participants} rows, but found "
            f"{len(predictions)}."
        )

    if predictions["participant_id"].duplicated().any():
        raise ValueError(
            f"Policy {scenario.name} {table_name} contains duplicate "
            "participant predictions."
        )

    expected_ids = normalized_id_set(
        scenario.outer_folds["participant_id"]
    )
    predicted_ids = normalized_id_set(
        predictions["participant_id"]
    )

    if predicted_ids != expected_ids:
        raise ValueError(
            f"Policy {scenario.name} {table_name} does not cover the "
            "complete scenario cohort."
        )

    expected_target_by_id = {
        normalize_participant_id(participant_id): int(target)
        for participant_id, target in zip(
            scenario.outer_folds["participant_id"],
            scenario.outer_folds["target"],
            strict=True,
        )
    }
    expected_fold_by_id = {
        normalize_participant_id(participant_id): int(outer_fold)
        for participant_id, outer_fold in zip(
            scenario.outer_folds["participant_id"],
            scenario.outer_folds["outer_fold"],
            strict=True,
        )
    }

    for prediction in predictions.itertuples(index=False):
        participant_id = normalize_participant_id(
            prediction.participant_id
        )

        if int(prediction.true_label) != expected_target_by_id[
            participant_id
        ]:
            raise ValueError(
                f"Policy {scenario.name} {table_name} contains an "
                "incorrect true target."
            )

        if int(prediction.outer_fold) != expected_fold_by_id[
            participant_id
        ]:
            raise ValueError(
                f"Policy {scenario.name} {table_name} contains an "
                "incorrect outer-fold assignment."
            )

def validate_policy_result(
    scenario: PolicyScenario,
    result: ScenarioResult,
) -> None:
    """Validate all generated results for one duplicate policy."""

    expected_outer_folds = set(
        range(EXPECTED_OUTER_FOLDS)
    )
    expected_inner_rows = (
        EXPECTED_OUTER_FOLDS
        * scenario.expected_subset_count
    )

    if len(result.inner_results) != expected_inner_rows:
        raise ValueError(
            f"Policy {scenario.name} expected {expected_inner_rows} "
            f"inner rows, but found {len(result.inner_results)}."
        )

    inner_counts = (
        result.inner_results["outer_fold"]
        .value_counts()
        .to_dict()
    )
    expected_inner_counts = {
        outer_fold: scenario.expected_subset_count
        for outer_fold in expected_outer_folds
    }

    if inner_counts != expected_inner_counts:
        raise ValueError(
            f"Policy {scenario.name} has an incorrect subset count in "
            "at least one outer fold."
        )

    unique_subset_counts = (
        result.inner_results.groupby("outer_fold")["subset_id"]
        .nunique()
    )

    if not unique_subset_counts.eq(
        scenario.expected_subset_count
    ).all():
        raise ValueError(
            f"Policy {scenario.name} contains a missing or duplicate "
            "inner-CV subset."
        )

    selected_rows = result.inner_results.loc[
        result.inner_results["selected"]
    ]

    if len(selected_rows) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            f"Policy {scenario.name} did not select exactly one subset "
            "per outer fold."
        )

    reference_subset = "+".join(
        scenario.reference_tasks
    )
    reference_rows = result.inner_results.loc[
        result.inner_results["subset_id"].eq(reference_subset)
    ]

    if len(reference_rows) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            f"Policy {scenario.name} does not contain one reference "
            "result per outer fold."
        )

    for table_name, fold_metrics in (
        ("selected fold metrics", result.selected_fold_metrics),
        ("reference fold metrics", result.reference_fold_metrics),
    ):
        if len(fold_metrics) != EXPECTED_OUTER_FOLDS:
            raise ValueError(
                f"Policy {scenario.name} {table_name} expected "
                f"{EXPECTED_OUTER_FOLDS} rows, but found "
                f"{len(fold_metrics)}."
            )

        if set(fold_metrics["outer_fold"].astype(int)) != (
            expected_outer_folds
        ):
            raise ValueError(
                f"Policy {scenario.name} {table_name} does not cover "
                "all outer folds."
            )

    selected_by_fold = dict(
        zip(
            result.selected_fold_metrics["outer_fold"].astype(int),
            result.selected_fold_metrics["selected_subset"].astype(str),
            strict=True,
        )
    )

    for row in selected_rows.itertuples(index=False):
        if str(row.subset_id) != selected_by_fold[int(row.outer_fold)]:
            raise ValueError(
                f"Policy {scenario.name} inner and outer selections "
                "do not match."
            )

    if not result.reference_fold_metrics[
        "selected_subset"
    ].eq(reference_subset).all():
        raise ValueError(
            f"Policy {scenario.name} reference evaluation used the "
            "wrong task subset."
        )

    validate_prediction_table(
        predictions=result.selected_predictions,
        scenario=scenario,
        table_name="selected predictions",
    )
    validate_prediction_table(
        predictions=result.reference_predictions,
        scenario=scenario,
        table_name="reference predictions",
    )

    for table_name, table in (
        ("inner results", result.inner_results),
        ("selected fold metrics", result.selected_fold_metrics),
        ("selected predictions", result.selected_predictions),
        ("reference fold metrics", result.reference_fold_metrics),
        ("reference predictions", result.reference_predictions),
    ):
        numeric_values = table.select_dtypes(
            include="number"
        ).to_numpy()

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"Policy {scenario.name} {table_name} contains a "
                "nonfinite value."
            )

    print(
        f"Policy {scenario.name} results validated: "
        f"{len(result.inner_results)} inner rows, "
        f"{len(result.selected_predictions)} selected predictions."
    )

def scenario_artifact_paths(
    scenario_name: str,
) -> dict[str, Path]:
    """Return isolated generated-result paths for one policy."""

    return {
        "inner_results": RESULTS_DIRECTORY / (
            f"{scenario_name}_inner_results.csv"
        ),
        "selected_fold_metrics": RESULTS_DIRECTORY / (
            f"{scenario_name}_selected_fold_metrics.csv"
        ),
        "selected_predictions": RESULTS_DIRECTORY / (
            f"{scenario_name}_selected_predictions.csv"
        ),
        "reference_fold_metrics": RESULTS_DIRECTORY / (
            f"{scenario_name}_reference_fold_metrics.csv"
        ),
        "reference_predictions": RESULTS_DIRECTORY / (
            f"{scenario_name}_reference_predictions.csv"
        ),
    }

def save_policy_result(
    scenario: PolicyScenario,
    result: ScenarioResult,
) -> None:
    """Checkpoint one validated policy without replacing primary results."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = scenario_artifact_paths(
        scenario.name
    )
    tables = {
        "inner_results": result.inner_results,
        "selected_fold_metrics": result.selected_fold_metrics,
        "selected_predictions": result.selected_predictions,
        "reference_fold_metrics": result.reference_fold_metrics,
        "reference_predictions": result.reference_predictions,
    }
    temporary_paths: list[Path] = []

    try:
        for artifact_name, table in tables.items():
            final_path = paths[artifact_name]
            temporary_path = Path(
                f"{final_path}.tmp"
            )
            temporary_paths.append(
                temporary_path
            )
            table.to_csv(
                temporary_path,
                index=False,
            )
            temporary_path.replace(
                final_path
            )
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

def build_policy_comparison(
    result: ScenarioResult,
) -> pd.DataFrame:
    """Pair selected and reference Macro F1 within each outer fold."""

    comparison = (
        result.selected_fold_metrics[
            ["outer_fold", "outer_test_macro_f1"]
        ]
        .rename(
            columns={
                "outer_test_macro_f1": "selected_macro_f1",
            }
        )
        .merge(
            result.reference_fold_metrics[
                ["outer_fold", "outer_test_macro_f1"]
            ].rename(
                columns={
                    "outer_test_macro_f1": "reference_macro_f1",
                }
            ),
            on="outer_fold",
            validate="one_to_one",
        )
        .sort_values("outer_fold")
    )
    comparison["difference"] = (
        comparison["selected_macro_f1"]
        - comparison["reference_macro_f1"]
    )

    return comparison

def build_policy_summary(
    scenario: PolicyScenario,
    result: ScenarioResult,
) -> dict[str, object]:
    """Build one participant-free policy summary."""

    comparison = build_policy_comparison(
        result
    )
    selected_metrics = result.selected_fold_metrics
    reference_metrics = result.reference_fold_metrics

    fold_selections = json.loads(
        selected_metrics[
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
        .to_json(orient="records")
    )

    selection_frequency = {
        str(subset_id): int(count)
        for subset_id, count in (
            selected_metrics["selected_subset"]
            .value_counts()
            .items()
        )
    }
    task_frequency = {
        task: int(
            selected_metrics["selected_subset"]
            .astype(str)
            .map(
                lambda subset_id: task in subset_id.split("+")
            )
            .sum()
        )
        for task in scenario.reference_tasks
    }

    return {
        "excluded_participants": len(scenario.excluded_ids),
        "remaining_participants": len(scenario.features),
        "available_tasks": list(scenario.reference_tasks),
        "task_subsets_per_outer_fold": len(scenario.subsets),
        "reference_assessment": "+".join(
            scenario.reference_tasks
        ),
        "fold_selections": fold_selections,
        "selection_frequency": selection_frequency,
        "task_frequency": task_frequency,
        "mean_selected_task_count": float(
            selected_metrics["task_count"].mean()
        ),
        "selected_outer_test_metrics": summarize_metric_table(
            selected_metrics
        ),
        "reference_outer_test_metrics": summarize_metric_table(
            reference_metrics
        ),
        "selected_vs_reference": {
            "selected_macro_f1_mean": float(
                comparison["selected_macro_f1"].mean()
            ),
            "reference_macro_f1_mean": float(
                comparison["reference_macro_f1"].mean()
            ),
            "difference_mean": float(
                comparison["difference"].mean()
            ),
            "folds_selected_at_least_reference": int(
                comparison["difference"].ge(0.0).sum()
            ),
            "fold_comparisons": json.loads(
                comparison.to_json(orient="records")
            ),
        },
    }

def build_experiment_summary(
    scenarios: list[PolicyScenario],
    results: dict[str, ScenarioResult],
) -> dict[str, object]:
    """Build the tracked aggregate record for all policies."""

    return {
        "experiment": "duplicate_policy_sensitivity",
        "participant_level_fields": [],
        "primary_delta": PRIMARY_DELTA,
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "inner_folds_per_outer_fold": (
            EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
        ),
        "policies": {
            scenario.name: build_policy_summary(
                scenario=scenario,
                result=results[scenario.name],
            )
            for scenario in scenarios
        },
        "interpretation_status": (
            "Sensitivity analysis of three defensible policies for an "
            "exact duplicated ALS recording with contradictory targets. "
            "The experiment tests whether the low-burden conclusion "
            "depends on one arbitrary duplicate-removal decision."
        ),
    }

def save_experiment_summary(
    summary: dict[str, object],
) -> None:
    """Save the participant-free aggregate record atomically."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = Path(
        f"{SUMMARY_PATH}.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(
            SUMMARY_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def print_experiment_results(
    scenarios: list[PolicyScenario],
    results: dict[str, ScenarioResult],
) -> None:
    """Print the final aggregate policy comparison."""

    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        result = results[scenario.name]
        comparison = build_policy_comparison(
            result
        )

        rows.append(
            {
                "policy": scenario.name,
                "participants": len(scenario.features),
                "available_tasks": len(scenario.reference_tasks),
                "mean_selected_tasks": float(
                    result.selected_fold_metrics["task_count"].mean()
                ),
                "selected_macro_f1": float(
                    comparison["selected_macro_f1"].mean()
                ),
                "reference_macro_f1": float(
                    comparison["reference_macro_f1"].mean()
                ),
                "difference": float(
                    comparison["difference"].mean()
                ),
                "selected_protocols": ", ".join(
                    result.selected_fold_metrics
                    .sort_values("outer_fold")["selected_subset"]
                    .astype(str)
                ),
            }
        )

    print()
    print("Duplicate-policy sensitivity results:")
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Tracked experiment record:")
    print(SUMMARY_PATH)

def main() -> None:
    """Run and compare all alternative duplicate policies."""

    features, outer_folds, nested_folds = load_experiment_inputs()

    if len(features) != EXPECTED_ORIGINAL_PARTICIPANTS:
        raise ValueError(
            "Expected the frozen experiment to contain "
            f"{EXPECTED_ORIGINAL_PARTICIPANTS} participants, but found "
            f"{len(features)}."
        )

    scenarios = build_policy_scenarios(
        features=features,
        outer_folds=outer_folds,
        nested_folds=nested_folds,
    )

    for scenario in scenarios:
        validate_policy_scenario(scenario)

    print_policy_scenario_summary(scenarios)

    frozen_configurations = (
        load_frozen_configurations()
    )
    expected_outer_folds = set(
        range(EXPECTED_OUTER_FOLDS)
    )

    if set(frozen_configurations) != expected_outer_folds:
        raise ValueError(
            "Frozen model configurations do not match the five "
            "outer folds."
        )

    results: dict[str, ScenarioResult] = {}

    # Run every policy through the same modeling path. Saving each result
    # immediately provides a checkpoint without exposing participant IDs
    # in the tracked aggregate experiment record.
    for scenario in scenarios:
        result = run_policy_scenario(
            scenario=scenario,
            frozen_configurations=frozen_configurations,
        )
        validate_policy_result(
            scenario=scenario,
            result=result,
        )
        save_policy_result(
            scenario=scenario,
            result=result,
        )
        results[scenario.name] = result

    summary = build_experiment_summary(
        scenarios=scenarios,
        results=results,
    )
    save_experiment_summary(summary)
    print_experiment_results(
        scenarios=scenarios,
        results=results,
    )

if __name__ == "__main__":
    main()

"""Test whether WavLM task-selection results depend on PCA dimension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bootstrap_wavlm_task_selection import (
    BOOTSTRAP_REPLICATES,
    print_paired_wavlm_bootstrap_summary,
    run_paired_wavlm_bootstrap,
    summarize_paired_wavlm_bootstrap,
)
from evaluate_task_subsets import generate_task_subsets
from evaluate_wavlm_task_selection import (
    evaluate_all_wavlm_subsets,
    evaluate_wavlm_outer_folds,
    print_input_summary,
    print_regularization_summary,
    print_wavlm_outer_summary,
    print_wavlm_selection_summary,
    select_competitive_wavlm_subsets,
    select_full_assessment_regularization,
    validate_full_subset_reproduction,
)
from wavlm_task_selection_core import (
    EXPECTED_OUTER_FOLDS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_TASK_SUBSETS,
    MODEL_NAME,
    MODEL_REVISION,
    PCA_COMPONENTS_PER_TASK,
    PRIMARY_DELTA,
    WavLMEmbeddingArtifact,
    build_inner_projection_cache,
    load_duplicate_filtered_design,
    load_wavlm_artifact,
    print_projection_cache_summary,
    validate_wavlm_inputs,
)
from wavlm_task_selection_results import (
    build_wavlm_experiment_record,
    convert_json_value,
    prepare_csv_frame,
    validate_wavlm_experiment_record,
)

SENSITIVITY_PCA_COMPONENTS = (
    8,
    32,
)
PRIMARY_RECORD_PATH = Path(
    "experiments/wavlm_task_selection.json"
)
OUTPUT_DIRECTORY = Path(
    "results/generated"
)
REGULARIZATION_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_pca_sensitivity_regularization.csv"
)
INNER_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_pca_sensitivity_inner_results.csv"
)
FOLD_METRICS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_pca_sensitivity_fold_metrics.csv"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_pca_sensitivity_predictions.csv"
)
BOOTSTRAP_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "wavlm_pca_sensitivity_bootstrap.csv"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/wavlm_pca_sensitivity.json"
)

@dataclass(frozen=True)
class PCASensitivityResult:
    """Keep every output from one prespecified PCA sensitivity run."""

    pca_components_per_task: int
    regularization_results: pd.DataFrame
    inner_results: pd.DataFrame
    selections: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    bootstrap_results: pd.DataFrame
    experiment_record: dict[str, Any]

def load_primary_wavlm_record() -> dict[str, Any]:
    """Load and validate the locked 16-component primary result."""

    if not PRIMARY_RECORD_PATH.exists():
        raise FileNotFoundError(
            "The primary WavLM experiment record was not found at "
            f"{PRIMARY_RECORD_PATH}."
        )

    with PRIMARY_RECORD_PATH.open(
        encoding="utf-8"
    ) as input_file:
        record = json.load(
            input_file
        )

    validate_wavlm_experiment_record(
        record
    )

    if record.get("experiment") != (
        "duplicate_filtered_wavlm_nested_task_selection"
    ):
        raise ValueError(
            "The primary WavLM record has an unexpected experiment name."
        )

    representation = record.get(
        "representation"
    )
    design = record.get(
        "design"
    )

    if not isinstance(
        representation,
        dict,
    ) or not isinstance(
        design,
        dict,
    ):
        raise ValueError(
            "The primary WavLM record is missing design metadata."
        )

    expected_values = {
        "pca_components_per_task": (
            PCA_COMPONENTS_PER_TASK
        ),
        "encoder": MODEL_NAME,
        "encoder_revision": MODEL_REVISION,
    }

    for field, expected_value in expected_values.items():
        if representation.get(field) != expected_value:
            raise ValueError(
                f"Primary WavLM representation field {field!r} changed."
            )

    if int(
        design.get(
            "participants",
            0,
        )
    ) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            "The primary WavLM participant count changed."
        )

    if float(
        design.get(
            "competitive_delta",
            np.nan,
        )
    ) != PRIMARY_DELTA:
        raise ValueError(
            "The primary WavLM competitive delta changed."
        )

    return record

def build_scenario_interpretation(
    bootstrap_summary: dict[str, object],
    pca_components_per_task: int,
) -> str:
    """Describe one sensitivity result without overstating evidence."""

    if bool(
        bootstrap_summary[
            "one_sided_bound_supports_noninferiority"
        ]
    ):
        conclusion = (
            "supported noninferiority at the prespecified margin"
        )
    else:
        conclusion = (
            "did not establish noninferiority at the prespecified margin"
        )

    return (
        f"With {pca_components_per_task} PCA components per task, "
        f"the paired participant bootstrap {conclusion}. This is a "
        "prespecified representation-sensitivity result, not a new "
        "choice made from outer-test performance."
    )

def run_pca_sensitivity_scenario(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    nested_folds: list[dict[str, object]],
    excluded_participant_count: int,
    pca_components_per_task: int,
) -> PCASensitivityResult:
    """Run the complete leakage-safe analysis for one PCA dimension."""

    print()
    print(
        "=" * 72
    )
    print(
        "Running WavLM PCA sensitivity with "
        f"{pca_components_per_task} components per task"
    )
    print(
        "=" * 72
    )

    # Every PCA fit remains inside its training partition. The cache is
    # reused only across candidate task subsets from that same split.
    projection_cache = build_inner_projection_cache(
        artifact=artifact,
        labels=labels,
        nested_folds=nested_folds,
        pca_components_per_task=(
            pca_components_per_task
        ),
    )
    print_projection_cache_summary(
        projection_cache
    )

    # Regularization is selected from the complete assessment using only
    # inner folds, exactly as it was in the locked primary analysis.
    (
        regularization_results,
        selected_regularization,
    ) = select_full_assessment_regularization(
        projection_cache
    )
    print_regularization_summary(
        regularization_results
    )

    raw_inner_results = evaluate_all_wavlm_subsets(
        projection_cache=projection_cache,
        subsets=generate_task_subsets(),
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

    # The selected and complete protocols are evaluated together on each
    # untouched outer fold so the participant-level comparison is paired.
    fold_metrics, predictions = (
        evaluate_wavlm_outer_folds(
            artifact=artifact,
            labels=labels,
            selections=selections,
            pca_components_per_task=(
                pca_components_per_task
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

    experiment_record: dict[str, Any] = (
        build_wavlm_experiment_record(
            regularization_results=(
                regularization_results
            ),
            selections=selections,
            fold_metrics=fold_metrics,
            projection_cache=projection_cache,
            bootstrap_summary=bootstrap_summary,
            participant_count=(
                artifact.participant_count
            ),
            excluded_participant_count=(
                excluded_participant_count
            ),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            embedding_dimension=(
                artifact.embedding_dimension
            ),
            pca_components_per_task=(
                pca_components_per_task
            ),
            delta=PRIMARY_DELTA,
        )
    )
    experiment_record["experiment"] = (
        "duplicate_filtered_wavlm_pca_sensitivity_scenario"
    )
    experiment_record["sensitivity_role"] = (
        "prespecified_representation_dimension_check"
    )
    experiment_record["inference"]["interpretation"] = (
        build_scenario_interpretation(
            bootstrap_summary=bootstrap_summary,
            pca_components_per_task=(
                pca_components_per_task
            ),
        )
    )
    validate_wavlm_experiment_record(
        experiment_record
    )

    return PCASensitivityResult(
        pca_components_per_task=(
            pca_components_per_task
        ),
        regularization_results=(
            regularization_results
        ),
        inner_results=inner_results,
        selections=selections,
        fold_metrics=fold_metrics,
        predictions=predictions,
        bootstrap_results=bootstrap_results,
        experiment_record=experiment_record,
    )

def add_scenario_dimension(
    frame: pd.DataFrame,
    pca_components_per_task: int,
) -> pd.DataFrame:
    """Add the scenario key as the first column of a generated table."""

    result = frame.copy()
    result.insert(
        0,
        "pca_components_per_task",
        pca_components_per_task,
    )

    return result

def combine_scenario_frames(
    results: list[PCASensitivityResult],
    field: str,
) -> pd.DataFrame:
    """Combine one table type across all sensitivity dimensions."""

    frames = [
        add_scenario_dimension(
            frame=getattr(
                result,
                field,
            ),
            pca_components_per_task=(
                result.pca_components_per_task
            ),
        )
        for result in results
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )

def primary_comparison_row(
    primary_record: dict[str, Any],
) -> dict[str, object]:
    """Extract the primary result into the sensitivity comparison schema."""

    selection_summary = primary_record[
        "selection_summary"
    ]
    selected_vs_full = primary_record[
        "selected_vs_full"
    ]
    bootstrap = primary_record[
        "paired_bootstrap"
    ]
    inference = primary_record[
        "inference"
    ]

    return {
        "pca_components_per_task": (
            PCA_COMPONENTS_PER_TASK
        ),
        "analysis_role": "locked_primary",
        "mean_selected_task_count": float(
            selection_summary[
                "mean_selected_task_count"
            ]
        ),
        "selected_protocols": ",".join(
            selection_summary[
                "selected_protocols"
            ]
        ),
        "selected_macro_f1_mean": float(
            selected_vs_full[
                "selected_macro_f1_mean"
            ]
        ),
        "full_macro_f1_mean": float(
            selected_vs_full[
                "full_macro_f1_mean"
            ]
        ),
        "difference_mean": float(
            selected_vs_full[
                "difference_mean"
            ]
        ),
        "difference_one_sided_95_lower": float(
            bootstrap[
                "difference_one_sided_95_lower"
            ]
        ),
        "noninferiority_probability": float(
            bootstrap[
                "noninferiority_probability"
            ]
        ),
        "point_estimate_inside_margin": bool(
            bootstrap[
                "point_estimate_inside_margin"
            ]
        ),
        "formal_noninferiority_supported": bool(
            inference[
                "formal_noninferiority_supported"
            ]
        ),
        "folds_inside_margin": int(
            bootstrap[
                "folds_inside_margin"
            ]
        ),
    }

def sensitivity_comparison_row(
    result: PCASensitivityResult,
) -> dict[str, object]:
    """Extract one rerun into the final comparison schema."""

    record = result.experiment_record
    selection_summary = record[
        "selection_summary"
    ]
    selected_vs_full = record[
        "selected_vs_full"
    ]
    bootstrap = record[
        "paired_bootstrap"
    ]
    inference = record[
        "inference"
    ]

    return {
        "pca_components_per_task": (
            result.pca_components_per_task
        ),
        "analysis_role": "prespecified_sensitivity",
        "mean_selected_task_count": float(
            selection_summary[
                "mean_selected_task_count"
            ]
        ),
        "selected_protocols": ",".join(
            selection_summary[
                "selected_protocols"
            ]
        ),
        "selected_macro_f1_mean": float(
            selected_vs_full[
                "selected_macro_f1_mean"
            ]
        ),
        "full_macro_f1_mean": float(
            selected_vs_full[
                "full_macro_f1_mean"
            ]
        ),
        "difference_mean": float(
            selected_vs_full[
                "difference_mean"
            ]
        ),
        "difference_one_sided_95_lower": float(
            bootstrap[
                "difference_one_sided_95_lower"
            ]
        ),
        "noninferiority_probability": float(
            bootstrap[
                "noninferiority_probability"
            ]
        ),
        "point_estimate_inside_margin": bool(
            bootstrap[
                "point_estimate_inside_margin"
            ]
        ),
        "formal_noninferiority_supported": bool(
            inference[
                "formal_noninferiority_supported"
            ]
        ),
        "folds_inside_margin": int(
            bootstrap[
                "folds_inside_margin"
            ]
        ),
    }

def build_comparison_table(
    primary_record: dict[str, Any],
    results: list[PCASensitivityResult],
) -> pd.DataFrame:
    """Compare the locked primary result with both sensitivity reruns."""

    rows = [
        primary_comparison_row(
            primary_record
        ),
        *[
            sensitivity_comparison_row(
                result
            )
            for result in results
        ],
    ]

    return (
        pd.DataFrame(rows)
        .sort_values(
            "pca_components_per_task"
        )
        .reset_index(drop=True)
    )

def build_protocol_agreement(
    primary_record: dict[str, Any],
    results: list[PCASensitivityResult],
) -> dict[str, dict[str, object]]:
    """Compare exact fold selections with the locked primary protocols."""

    primary_protocols = list(
        primary_record[
            "selection_summary"
        ]["selected_protocols"]
    )
    agreement: dict[
        str,
        dict[str, object],
    ] = {}

    for result in results:
        sensitivity_protocols = list(
            result.experiment_record[
                "selection_summary"
            ]["selected_protocols"]
        )
        matching_folds = sum(
            primary_protocol == sensitivity_protocol
            for (
                primary_protocol,
                sensitivity_protocol,
            ) in zip(
                primary_protocols,
                sensitivity_protocols,
                strict=True,
            )
        )

        agreement[
            str(result.pca_components_per_task)
        ] = {
            "matching_outer_folds": int(
                matching_folds
            ),
            "total_outer_folds": (
                EXPECTED_OUTER_FOLDS
            ),
            "exact_match_rate": float(
                matching_folds
                / EXPECTED_OUTER_FOLDS
            ),
            "selected_protocols": (
                sensitivity_protocols
            ),
        }

    return agreement

def build_final_inference(
    primary_record: dict[str, Any],
    comparison: pd.DataFrame,
    results: list[PCASensitivityResult],
) -> dict[str, object]:
    """State what the PCA sensitivity can and cannot support."""

    maximum_selected_tasks = max(
        [
            int(
                fold_result[
                    "task_count"
                ]
            )
            for fold_result in primary_record[
                "fold_results"
            ]
        ]
        + [
            int(
                fold_result[
                    "task_count"
                ]
            )
            for result in results
            for fold_result in result.experiment_record[
                "fold_results"
            ]
        ]
    )
    low_burden_robust = (
        maximum_selected_tasks <= 3
    )
    point_estimates_inside_margin = bool(
        comparison[
            "point_estimate_inside_margin"
        ].all()
    )
    formal_support_count = int(
        comparison[
            "formal_noninferiority_supported"
        ].sum()
    )

    if low_burden_robust:
        burden_statement = (
            "The low-recording-count selection pattern persisted across "
            "the locked 16-component analysis and both prespecified PCA "
            "sensitivity settings."
        )
    else:
        burden_statement = (
            "At least one PCA sensitivity setting did not preserve the "
            "one-to-three-task selection pattern."
        )

    inference_statement = (
        f"Formal noninferiority was supported in {formal_support_count} "
        f"of {len(comparison)} PCA settings. The sensitivity analysis "
        "therefore describes representation robustness but does not "
        "replace the locked primary inference."
    )

    return {
        "low_burden_selection_robust": (
            low_burden_robust
        ),
        "maximum_selected_tasks_across_pca_settings": (
            maximum_selected_tasks
        ),
        "all_point_estimates_inside_margin": (
            point_estimates_inside_margin
        ),
        "formal_noninferiority_supported_setting_count": (
            formal_support_count
        ),
        "total_pca_settings": int(
            len(comparison)
        ),
        "interpretation": (
            f"{burden_statement} {inference_statement}"
        ),
    }

def build_sensitivity_record(
    primary_record: dict[str, Any],
    results: list[PCASensitivityResult],
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Build the participant-free tracked record for the full check."""

    return {
        "experiment": (
            "wavlm_pca_dimension_sensitivity"
        ),
        "participant_level_fields": [],
        "design": {
            "primary_pca_components_per_task": (
                PCA_COMPONENTS_PER_TASK
            ),
            "sensitivity_pca_components_per_task": list(
                SENSITIVITY_PCA_COMPONENTS
            ),
            "participants": (
                EXPECTED_PARTICIPANTS
            ),
            "outer_folds": (
                EXPECTED_OUTER_FOLDS
            ),
            "task_subset_count": (
                EXPECTED_TASK_SUBSETS
            ),
            "competitive_delta": (
                PRIMARY_DELTA
            ),
            "bootstrap_replicates_per_setting": (
                BOOTSTRAP_REPLICATES
            ),
            "analysis_rule": (
                "The 16-component analysis remains primary. The 8- and "
                "32-component settings are prespecified sensitivity "
                "checks and are not selected using outer-test results."
            ),
        },
        "representation": {
            "encoder": MODEL_NAME,
            "encoder_revision": MODEL_REVISION,
            "frozen_encoder": True,
            "pca_fit_scope": (
                "training participants only within each split"
            ),
        },
        "comparison": comparison.to_dict(
            orient="records"
        ),
        "protocol_agreement_with_primary": (
            build_protocol_agreement(
                primary_record=primary_record,
                results=results,
            )
        ),
        "sensitivity_scenarios": [
            result.experiment_record
            for result in results
        ],
        "inference": build_final_inference(
            primary_record=primary_record,
            comparison=comparison,
            results=results,
        ),
    }

def validate_combined_results(
    results: list[PCASensitivityResult],
    regularization_results: pd.DataFrame,
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    comparison: pd.DataFrame,
    experiment_record: dict[str, Any],
) -> None:
    """Protect the final sensitivity artifacts from missing or stale data."""

    expected_components = set(
        SENSITIVITY_PCA_COMPONENTS
    )

    if {
        result.pca_components_per_task
        for result in results
    } != expected_components:
        raise ValueError(
            "The PCA sensitivity did not run both locked dimensions."
        )

    expected_rows = {
        "regularization": (
            len(expected_components)
            * EXPECTED_OUTER_FOLDS
            * 4
        ),
        "inner": (
            len(expected_components)
            * EXPECTED_OUTER_FOLDS
            * EXPECTED_TASK_SUBSETS
        ),
        "fold_metrics": (
            len(expected_components)
            * EXPECTED_OUTER_FOLDS
        ),
        "predictions": (
            len(expected_components)
            * EXPECTED_PARTICIPANTS
        ),
        "bootstrap": (
            len(expected_components)
            * BOOTSTRAP_REPLICATES
        ),
    }
    frames = {
        "regularization": regularization_results,
        "inner": inner_results,
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "bootstrap": bootstrap_results,
    }

    for name, frame in frames.items():
        if len(frame) != expected_rows[name]:
            raise ValueError(
                f"Expected {expected_rows[name]} {name} rows, found "
                f"{len(frame)}."
            )

        if set(
            frame["pca_components_per_task"]
        ) != expected_components:
            raise ValueError(
                f"The {name} table has incorrect PCA scenario keys."
            )

    for pca_components_per_task in sorted(
        expected_components
    ):
        scenario_predictions = predictions.loc[
            predictions[
                "pca_components_per_task"
            ].eq(pca_components_per_task)
        ]

        if scenario_predictions[
            "participant_id"
        ].nunique() != EXPECTED_PARTICIPANTS:
            raise ValueError(
                "A PCA scenario does not contain one prediction per "
                "retained participant."
            )

    numeric_frames = (
        regularization_results,
        inner_results,
        fold_metrics,
        bootstrap_results,
    )

    for frame in numeric_frames:
        numeric_values = frame.select_dtypes(
            include=[np.number]
        ).to_numpy(dtype=float)

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                "A PCA sensitivity result contains a non-finite value."
            )

    if set(
        comparison[
            "pca_components_per_task"
        ]
    ) != {
        PCA_COMPONENTS_PER_TASK,
        *SENSITIVITY_PCA_COMPONENTS,
    }:
        raise ValueError(
            "The final comparison does not contain 8, 16, and 32 "
            "components."
        )

    if experiment_record.get(
        "participant_level_fields"
    ) != []:
        raise ValueError(
            "The tracked sensitivity record must not contain participant "
            "fields."
        )

    serialized = json.dumps(
        convert_json_value(
            experiment_record
        ),
        sort_keys=True,
    )

    for forbidden_field in (
        '"participant_id"',
        '"inner_train_ids"',
        '"inner_validation_ids"',
        '"outer_test_ids"',
    ):
        if forbidden_field in serialized:
            raise ValueError(
                "The tracked sensitivity record contains participant "
                f"field {forbidden_field}."
            )

    scenarios = experiment_record.get(
        "sensitivity_scenarios"
    )

    if not isinstance(
        scenarios,
        list,
    ) or len(scenarios) != len(
        SENSITIVITY_PCA_COMPONENTS
    ):
        raise ValueError(
            "The tracked record is missing a PCA sensitivity scenario."
        )

def save_sensitivity_results(
    regularization_results: pd.DataFrame,
    inner_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    experiment_record: dict[str, Any],
) -> None:
    """Save private detailed tables and the public aggregate record."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_frames = (
        (
            regularization_results,
            REGULARIZATION_RESULTS_PATH,
        ),
        (
            inner_results,
            INNER_RESULTS_PATH,
        ),
        (
            fold_metrics,
            FOLD_METRICS_PATH,
        ),
        (
            predictions,
            PREDICTIONS_PATH,
        ),
        (
            bootstrap_results,
            BOOTSTRAP_RESULTS_PATH,
        ),
    )

    for frame, path in output_frames:
        prepare_csv_frame(
            frame
        ).to_csv(
            path,
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

def print_final_summary(
    comparison: pd.DataFrame,
    experiment_record: dict[str, Any],
) -> None:
    """Display the final primary-versus-sensitivity comparison."""

    print()
    print("WavLM PCA sensitivity validated.")
    print()
    print("Comparison across PCA dimensions:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(
        experiment_record[
            "inference"
        ]["interpretation"]
    )
    print()
    print("Generated artifacts:")
    print(REGULARIZATION_RESULTS_PATH)
    print(INNER_RESULTS_PATH)
    print(FOLD_METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(BOOTSTRAP_RESULTS_PATH)
    print(EXPERIMENT_RECORD_PATH)

def main() -> None:
    """Run the prespecified 8- and 32-component WavLM checks."""

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
    primary_record = load_primary_wavlm_record()
    print_input_summary(
        artifact=artifact,
        nested_folds=nested_folds,
        excluded_ids=excluded_ids,
    )
    print(
        "Primary PCA components per task: "
        f"{PCA_COMPONENTS_PER_TASK}"
    )
    print(
        "Sensitivity PCA components per task: "
        f"{list(SENSITIVITY_PCA_COMPONENTS)}"
    )

    results = [
        run_pca_sensitivity_scenario(
            artifact=artifact,
            labels=labels,
            nested_folds=nested_folds,
            excluded_participant_count=len(
                excluded_ids
            ),
            pca_components_per_task=(
                pca_components_per_task
            ),
        )
        for pca_components_per_task in (
            SENSITIVITY_PCA_COMPONENTS
        )
    ]

    regularization_results = combine_scenario_frames(
        results=results,
        field="regularization_results",
    )
    inner_results = combine_scenario_frames(
        results=results,
        field="inner_results",
    )
    fold_metrics = combine_scenario_frames(
        results=results,
        field="fold_metrics",
    )
    predictions = combine_scenario_frames(
        results=results,
        field="predictions",
    )
    bootstrap_results = combine_scenario_frames(
        results=results,
        field="bootstrap_results",
    )
    comparison = build_comparison_table(
        primary_record=primary_record,
        results=results,
    )
    experiment_record = build_sensitivity_record(
        primary_record=primary_record,
        results=results,
        comparison=comparison,
    )

    validate_combined_results(
        results=results,
        regularization_results=(
            regularization_results
        ),
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        bootstrap_results=bootstrap_results,
        comparison=comparison,
        experiment_record=experiment_record,
    )
    save_sensitivity_results(
        regularization_results=(
            regularization_results
        ),
        inner_results=inner_results,
        fold_metrics=fold_metrics,
        predictions=predictions,
        bootstrap_results=bootstrap_results,
        experiment_record=experiment_record,
    )
    print_final_summary(
        comparison=comparison,
        experiment_record=experiment_record,
    )

if __name__ == "__main__":
    main()

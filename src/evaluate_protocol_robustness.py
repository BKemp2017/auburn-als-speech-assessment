"""Evaluate how reliably fixed reduced protocols match the full assessment."""

from __future__ import annotations

import json 
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata import TASKS
from evaluate_baselines import (
    load_experiment_inputs,
    validate_experiment_inputs,
)
from evaluate_nested_task_selection import PRIMARY_DELTA
from evaluate_protocol_stability import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    INNER_OOF_PREDICTIONS_PATH,
    bootstrap_macro_f1,
    bootstrap_weights,
    validate_inner_oof_predictions,
)
from evaluate_task_subsets import (
    generate_task_subsets,
    load_frozen_configurations,
)


OUTPUT_DIR = Path("results/generated")

FOLD_ROBUSTNESS_PATH = (
    OUTPUT_DIR / "protocol_robustness_fold_results.csv"
)

CANDIDATE_SUMMARY_PATH = (
    OUTPUT_DIR / "protocol_robustness_candidate_summary.csv"
)

EXPERIMENT_RECORD_PATH = (
    Path("experiments") / "protocol_robustness.json"
)


def load_protocol_robustness_inputs(
) -> tuple[pd.DataFrame, list[tuple[str, ...]]]:
    """Load and validate the predictions needed for paired comparisons."""

    features, outer_folds, nested_folds = load_experiment_inputs()

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

    # Keeping the complete protocol last matches the established subset
    # ordering and gives us an unambiguous reference for every comparison.
    if subsets[-1] != tuple(TASKS):
        raise ValueError(
            "The final task subset is not the full assessment."
        )

    if not INNER_OOF_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Validated inner out-of-fold predictions were not found at "
            f"{INNER_OOF_PREDICTIONS_PATH}. Run "
            "src/evaluate_protocol_stability.py first."
        )

    predictions = pd.read_csv(
        INNER_OOF_PREDICTIONS_PATH
    )

    # This validator checks participant coverage, leakage boundaries,
    # task-subset coverage, and reproduction of all saved inner scores.
    validate_inner_oof_predictions(
        predictions=predictions,
        features=features,
        nested_folds=nested_folds,
        subsets=subsets,
        frozen_configurations=frozen_configurations,
    )

    return predictions, subsets

def evaluate_outer_fold_robustness(
    predictions: pd.DataFrame,
    outer_fold: int,
    subsets: list[tuple[str, ...]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compare every fixed subset with the full protocol in one outer fold."""

    fold_predictions = predictions.loc[
        predictions["outer_fold"].eq(
            outer_fold
        )
    ].copy()

    # Each outer-training participant has one inner out-of-fold
    # prediction per subset and belongs to exactly one inner fold.
    participant_information = (
        fold_predictions[
            [
                "participant_id",
                "inner_fold",
                "true_label",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "inner_fold",
                "participant_id",
            ]
        )
        .reset_index(drop=True)
    )

    if participant_information[
        "participant_id"
    ].duplicated().any():
        raise ValueError(
            f"A participant appears in multiple inner folds for "
            f"outer fold {outer_fold}."
        )

    subset_ids = [
        "+".join(tasks)
        for tasks in subsets
    ]

    participant_ids = participant_information[
        "participant_id"
    ].tolist()

    # Rows represent protocols and columns represent the same participants
    # for every protocol. That shared participant order is what makes the
    # reduced-versus-full comparisons paired.
    prediction_matrix = (
        fold_predictions.pivot(
            index="subset_id",
            columns="participant_id",
            values="predicted_label",
        )
        .reindex(
            index=subset_ids,
            columns=participant_ids,
        )
    )

    if prediction_matrix.isna().any().any():
        raise ValueError(
            f"Outer fold {outer_fold} has incomplete paired predictions."
        )

    predicted_labels = prediction_matrix.to_numpy(
        dtype=float
    )

    true_labels = participant_information[
        "true_label"
    ].to_numpy(dtype=int)

    inner_fold_labels = participant_information[
        "inner_fold"
    ].to_numpy(dtype=int)

    bootstrap_fold_scores = np.empty(
        (
            BOOTSTRAP_REPLICATES,
            4,
            len(subsets),
        ),
        dtype=float,
    )

    observed_fold_scores = np.empty(
        (
            4,
            len(subsets),
        ),
        dtype=float,
    )

    for inner_fold in range(4):
        inner_mask = (
            inner_fold_labels == inner_fold
        )

        participant_count = int(
            inner_mask.sum()
        )

        if participant_count == 0:
            raise ValueError(
                f"Outer fold {outer_fold} has no participants in "
                f"inner fold {inner_fold}."
            )

        # The same participant-resampling weights are applied to all 255
        # subsets, including the full protocol. This preserves pairing.
        weights = bootstrap_weights(
            rng=rng,
            replicates=BOOTSTRAP_REPLICATES,
            participant_count=participant_count,
        )

        bootstrap_fold_scores[
            :,
            inner_fold,
            :,
        ] = bootstrap_macro_f1(
            weights=weights,
            true_labels=true_labels[
                inner_mask
            ],
            predicted_labels=predicted_labels[
                :,
                inner_mask,
            ],
        )

        # One unit of weight per participant reproduces the original,
        # non-bootstrapped Macro F1 for this inner validation fold.
        observed_weights = np.ones(
            (
                1,
                participant_count,
            ),
            dtype=float,
        )

        observed_fold_scores[
            inner_fold,
            :,
        ] = bootstrap_macro_f1(
            weights=observed_weights,
            true_labels=true_labels[
                inner_mask
            ],
            predicted_labels=predicted_labels[
                :,
                inner_mask,
            ],
        )[0]

    bootstrap_mean_scores = (
        bootstrap_fold_scores.mean(
            axis=1
        )
    )

    observed_mean_scores = (
        observed_fold_scores.mean(
            axis=0
        )
    )

    full_subset_id = "+".join(TASKS)
    full_subset_index = subset_ids.index(
        full_subset_id
    )

    bootstrap_full_scores = (
        bootstrap_mean_scores[
            :,
            full_subset_index,
        ]
    )

    observed_full_score = float(
        observed_mean_scores[
            full_subset_index
        ]
    )

    # Positive differences favor the reduced protocol. Differences above
    # -0.02 remain inside the project's locked noninferiority margin.
    bootstrap_differences = (
        bootstrap_mean_scores
        - bootstrap_full_scores[
            :,
            np.newaxis,
        ]
    )

    observed_differences = (
        observed_mean_scores
        - observed_full_score
    )

    confidence_bounds = np.quantile(
        bootstrap_differences,
        q=[0.025, 0.975],
        axis=0,
    )

    noninferiority_probabilities = (
        bootstrap_differences
        >= -PRIMARY_DELTA
    ).mean(axis=0)

    superiority_probabilities = (
        bootstrap_differences > 0.0
    ).mean(axis=0)

    lower_bounds = confidence_bounds[0]
    upper_bounds = confidence_bounds[1]

    return pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "tasks": [
                ",".join(tasks)
                for tasks in subsets
            ],
            "subset_id": subset_ids,
            "task_count": [
                len(tasks)
                for tasks in subsets
            ],
            "observed_subset_macro_f1_mean": (
                observed_mean_scores
            ),
            "observed_full_macro_f1_mean": (
                observed_full_score
            ),
            "observed_difference": (
                observed_differences
            ),
            "bootstrap_difference_mean": (
                bootstrap_differences.mean(
                    axis=0
                )
            ),
            "bootstrap_difference_std": (
                bootstrap_differences.std(
                    axis=0,
                    ddof=1,
                )
            ),
            "difference_ci_lower": lower_bounds,
            "difference_ci_upper": upper_bounds,
            "noninferiority_probability": (
                noninferiority_probabilities
            ),
            "superiority_probability": (
                superiority_probabilities
            ),
            "supports_noninferiority": (
                lower_bounds
                >= -PRIMARY_DELTA
            ),
            "delta": PRIMARY_DELTA,
        }
    )


def run_protocol_robustness_bootstrap(
    predictions: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> pd.DataFrame:
    """Evaluate every protocol in every outer-fold training population."""

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    fold_results: list[pd.DataFrame] = []

    for outer_fold in sorted(
        predictions["outer_fold"].unique()
    ):
        print(
            f"Evaluating protocol robustness for outer fold "
            f"{outer_fold}..."
        )

        fold_results.append(
            evaluate_outer_fold_robustness(
                predictions=predictions,
                outer_fold=int(outer_fold),
                subsets=subsets,
                rng=rng,
            )
        )

    return (
        pd.concat(
            fold_results,
            ignore_index=True,
        )
        .sort_values(
            [
                "outer_fold",
                "task_count",
                "subset_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_fold_robustness_results(
    results: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> None:
    """Validate coverage, numeric values, and full-protocol comparisons."""

    required_columns = {
        "outer_fold",
        "tasks",
        "subset_id",
        "task_count",
        "observed_subset_macro_f1_mean",
        "observed_full_macro_f1_mean",
        "observed_difference",
        "bootstrap_difference_mean",
        "bootstrap_difference_std",
        "difference_ci_lower",
        "difference_ci_upper",
        "noninferiority_probability",
        "superiority_probability",
        "supports_noninferiority",
        "delta",
    }

    missing_columns = (
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold robustness results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_rows = (
        5 * len(subsets)
    )

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} fold-result rows, "
            f"but found {len(results)}."
        )

    if results.duplicated(
        subset=[
            "outer_fold",
            "subset_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate outer-fold and subset results were found."
        )

    expected_subset_ids = {
        "+".join(tasks)
        for tasks in subsets
    }

    for outer_fold, fold_results in results.groupby(
        "outer_fold"
    ):
        if set(
            fold_results["subset_id"]
        ) != expected_subset_ids:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain all subsets."
            )

    numeric_columns = [
        "observed_subset_macro_f1_mean",
        "observed_full_macro_f1_mean",
        "observed_difference",
        "bootstrap_difference_mean",
        "bootstrap_difference_std",
        "difference_ci_lower",
        "difference_ci_upper",
        "noninferiority_probability",
        "superiority_probability",
        "delta",
    ]

    if not np.isfinite(
        results[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Fold robustness results contain non-finite values."
        )

    probability_columns = [
        "noninferiority_probability",
        "superiority_probability",
    ]

    if not results[
        probability_columns
    ].apply(
        lambda column: column.between(
            0.0,
            1.0,
        )
    ).all().all():
        raise ValueError(
            "A bootstrap probability is outside the range zero to one."
        )

    if (
        results["difference_ci_lower"]
        > results["difference_ci_upper"]
    ).any():
        raise ValueError(
            "A confidence interval has reversed bounds."
        )

    expected_support = (
        results["difference_ci_lower"]
        >= -PRIMARY_DELTA
    )

    if not results[
        "supports_noninferiority"
    ].eq(expected_support).all():
        raise ValueError(
            "A noninferiority flag does not match its confidence bound."
        )

    full_subset_id = "+".join(TASKS)

    full_rows = results.loc[
        results["subset_id"].eq(
            full_subset_id
        )
    ]

    if len(full_rows) != 5:
        raise ValueError(
            "The full assessment does not appear once per outer fold."
        )

    full_difference_columns = [
        "observed_difference",
        "bootstrap_difference_mean",
        "bootstrap_difference_std",
        "difference_ci_lower",
        "difference_ci_upper",
    ]

    if not np.allclose(
        full_rows[
            full_difference_columns
        ].to_numpy(dtype=float),
        0.0,
    ):
        raise ValueError(
            "The full assessment does not have zero paired difference."
        )

    if not np.allclose(
        results["delta"],
        PRIMARY_DELTA,
    ):
        raise ValueError(
            "A fold comparison used the wrong noninferiority margin."
        )

def save_fold_robustness_results(
    results: pd.DataFrame,
) -> None:
    """Atomically save the validated fold-level comparisons."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)

    temporary_path = (
        FOLD_ROBUSTNESS_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        results.to_csv(temporary_path, index=False,)

        temporary_path.replace(FOLD_ROBUSTNESS_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def build_candidate_summary(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each fixed protocol across all five outer folds."""

    summary = (
        fold_results.groupby(
            [
                "subset_id",
                "tasks",
                "task_count",
            ],
            as_index=False,
        )
        .agg(
            outer_fold_count=(
                "outer_fold",
                "nunique",
            ),
            supporting_fold_count=(
                "supports_noninferiority",
                "sum",
            ),
            observed_difference_mean=(
                "observed_difference",
                "mean",
            ),
            observed_difference_std=(
                "observed_difference",
                "std",
            ),
            observed_difference_min=(
                "observed_difference",
                "min",
            ),
            observed_difference_max=(
                "observed_difference",
                "max",
            ),
            bootstrap_difference_mean=(
                "bootstrap_difference_mean",
                "mean",
            ),
            difference_ci_lower_min=(
                "difference_ci_lower",
                "min",
            ),
            difference_ci_lower_mean=(
                "difference_ci_lower",
                "mean",
            ),
            difference_ci_upper_max=(
                "difference_ci_upper",
                "max",
            ),
            noninferiority_probability_mean=(
                "noninferiority_probability",
                "mean",
            ),
            noninferiority_probability_min=(
                "noninferiority_probability",
                "min",
            ),
            superiority_probability_mean=(
                "superiority_probability",
                "mean",
            ),
        )
    )

    # A point estimate inside the margin is weaker evidence than a complete
    # confidence interval inside the margin, so both counts are retained.
    point_noninferior_counts = (
        fold_results.assign(
            point_noninferior=(
                fold_results[
                    "observed_difference"
                ]
                >= -PRIMARY_DELTA
            )
        )
        .groupby("subset_id")[
            "point_noninferior"
        ]
        .sum()
    )

    above_full_counts = (
        fold_results.assign(
            above_full=(
                fold_results[
                    "observed_difference"
                ]
                > 0.0
            )
        )
        .groupby("subset_id")[
            "above_full"
        ]
        .sum()
    )

    summary[
        "point_noninferior_fold_count"
    ] = summary["subset_id"].map(
        point_noninferior_counts
    ).astype(int)

    summary[
        "above_full_fold_count"
    ] = summary["subset_id"].map(
        above_full_counts
    ).astype(int)

    summary[
        "all_folds_support_noninferiority"
    ] = summary[
        "supporting_fold_count"
    ].eq(5)

    summary[
        "all_point_estimates_noninferior"
    ] = summary[
        "point_noninferior_fold_count"
    ].eq(5)

    summary["is_reduced"] = (
        summary["task_count"] < len(TASKS)
    )

    # The ranking is exploratory. It prioritizes consistency across folds,
    # then worst-fold bootstrap support, average support, and lower burden.
    reduced_ranking = (
        summary.loc[
            summary["is_reduced"]
        ]
        .sort_values(
            [
                "point_noninferior_fold_count",
                "noninferiority_probability_min",
                "noninferiority_probability_mean",
                "task_count",
                "observed_difference_mean",
                "subset_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                False,
                True,
            ],
        )
    )

    summary["candidate_rank"] = pd.array(
        [pd.NA] * len(summary),
        dtype="Int64",
    )

    summary.loc[
        reduced_ranking.index,
        "candidate_rank",
    ] = np.arange(
        1,
        len(reduced_ranking) + 1,
    )

    return (
        summary.sort_values(
            [
                "is_reduced",
                "candidate_rank",
                "subset_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def validate_candidate_summary(
    summary: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> None:
    """Validate the one-row-per-protocol aggregate table."""

    if len(summary) != len(subsets):
        raise ValueError(
            f"Expected {len(subsets)} candidate rows, "
            f"but found {len(summary)}."
        )

    if summary["subset_id"].duplicated().any():
        raise ValueError(
            "Duplicate protocols were found in the candidate summary."
        )

    expected_subset_ids = {
        "+".join(tasks)
        for tasks in subsets
    }

    if set(summary["subset_id"]) != expected_subset_ids:
        raise ValueError(
            "The candidate summary does not contain all task subsets."
        )

    if not summary[
        "outer_fold_count"
    ].eq(5).all():
        raise ValueError(
            "A candidate was not evaluated in all five outer folds."
        )

    count_columns = [
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "above_full_fold_count",
    ]

    if not summary[
        count_columns
    ].apply(
        lambda column: column.between(
            0,
            5,
        )
    ).all().all():
        raise ValueError(
            "A candidate fold count is outside the valid range."
        )

    numeric_columns = [
        "observed_difference_mean",
        "observed_difference_std",
        "observed_difference_min",
        "observed_difference_max",
        "bootstrap_difference_mean",
        "difference_ci_lower_min",
        "difference_ci_lower_mean",
        "difference_ci_upper_max",
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
        "superiority_probability_mean",
    ]

    if not np.isfinite(
        summary[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "The candidate summary contains non-finite values."
        )

    probability_columns = [
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
        "superiority_probability_mean",
    ]

    if not summary[
        probability_columns
    ].apply(
        lambda column: column.between(
            0.0,
            1.0,
        )
    ).all().all():
        raise ValueError(
            "A candidate probability is outside zero to one."
        )

    reduced_ranks = (
        summary.loc[
            summary["is_reduced"],
            "candidate_rank",
        ]
        .dropna()
        .astype(int)
        .sort_values()
        .to_numpy()
    )

    expected_ranks = np.arange(
        1,
        len(subsets),
    )

    if not np.array_equal(
        reduced_ranks,
        expected_ranks,
    ):
        raise ValueError(
            "Reduced candidate ranks are incomplete or duplicated."
        )

    full_subset_id = "+".join(TASKS)

    full_row = summary.loc[
        summary["subset_id"].eq(
            full_subset_id
        )
    ]

    if len(full_row) != 1:
        raise ValueError(
            "The full assessment does not appear exactly once."
        )

    if full_row[
        "candidate_rank"
    ].notna().any():
        raise ValueError(
            "The full assessment received a reduced-candidate rank."
        )

def save_candidate_summary(
    summary: pd.DataFrame,
) -> None:
    """Atomically save the cross-fold candidate summary."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)

    temporary_path = (
        CANDIDATE_SUMMARY_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        summary.to_csv(temporary_path, index=False,)

        temporary_path.replace(CANDIDATE_SUMMARY_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def candidate_to_record(
    candidate: pd.Series,
) -> dict[str, object]:
    """Convert one aggregate candidate row into JSON-safe values."""

    return {
        "candidate_rank": int(
            candidate["candidate_rank"]
        ),
        "subset_id": str(
            candidate["subset_id"]
        ),
        "tasks": str(
            candidate["tasks"]
        ).split(","),
        "task_count": int(
            candidate["task_count"]
        ),
        "supporting_fold_count": int(
            candidate["supporting_fold_count"]
        ),
        "point_noninferior_fold_count": int(
            candidate[
                "point_noninferior_fold_count"
            ]
        ),
        "above_full_fold_count": int(
            candidate["above_full_fold_count"]
        ),
        "observed_difference_mean": float(
            candidate["observed_difference_mean"]
        ),
        "observed_difference_min": float(
            candidate["observed_difference_min"]
        ),
        "difference_ci_lower_min": float(
            candidate["difference_ci_lower_min"]
        ),
        "noninferiority_probability_mean": float(
            candidate[
                "noninferiority_probability_mean"
            ]
        ),
        "noninferiority_probability_min": float(
            candidate[
                "noninferiority_probability_min"
            ]
        ),
    }


def build_experiment_record(
    fold_results: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> dict[str, object]:
    """Build a privacy-safe aggregate record of the robustness analysis."""

    reduced_summary = candidate_summary.loc[
        candidate_summary["is_reduced"]
    ].sort_values(
        "candidate_rank"
    )

    best_candidate = reduced_summary.iloc[0]

    key_subset_ids = [
        "KA",
        "TA",
        "PA",
        "O+TA",
        "PA+KA",
        "TA+KA",
        "A+PA+KA",
        "O+TA+KA",
    ]

    key_candidates: list[dict[str, object]] = []

    for subset_id in key_subset_ids:
        matching_candidate = reduced_summary.loc[
            reduced_summary["subset_id"].eq(
                subset_id
            )
        ]

        if len(matching_candidate) != 1:
            raise ValueError(
                f"Expected one summary row for {subset_id}, "
                f"but found {len(matching_candidate)}."
            )

        key_candidates.append(
            candidate_to_record(
                matching_candidate.iloc[0]
            )
        )

    supported_by_fold = (
        fold_results.loc[
            fold_results[
                "supports_noninferiority"
            ]
            & fold_results[
                "task_count"
            ].lt(len(TASKS))
        ]
        .groupby("outer_fold")
        .size()
        .reindex(
            range(5),
            fill_value=0,
        )
    )

    return {
        "experiment": "fixed_protocol_robustness",
        "analysis_role": (
            "Exploratory cross-fold robustness assessment of fixed "
            "reduced speech-task protocols."
        ),
        "primary_metric": "macro_f1",
        "reference_protocol": "+".join(TASKS),
        "noninferiority_margin": PRIMARY_DELTA,
        "bootstrap": {
            "replicates_per_outer_fold": (
                BOOTSTRAP_REPLICATES
            ),
            "random_seed": BOOTSTRAP_SEED,
            "resampling_unit": "participant",
            "resampling_scope": (
                "Participants were resampled independently within each "
                "inner validation fold. Identical weights were applied "
                "to every task subset and the full protocol."
            ),
            "confidence_interval": (
                "Two-sided percentile 95% interval for the paired "
                "Macro F1 difference."
            ),
        },
        "coverage": {
            "outer_folds": 5,
            "task_subsets": len(candidate_summary),
            "reduced_task_subsets": int(
                candidate_summary[
                    "is_reduced"
                ].sum()
            ),
            "fold_comparisons": len(fold_results),
            "reduced_fold_comparisons": int(
                fold_results[
                    "task_count"
                ].lt(len(TASKS)).sum()
            ),
        },
        "results": {
            "supported_reduced_comparisons": int(
                (
                    fold_results[
                        "supports_noninferiority"
                    ]
                    & fold_results[
                        "task_count"
                    ].lt(len(TASKS))
                ).sum()
            ),
            "supported_reduced_comparisons_by_fold": {
                str(outer_fold): int(count)
                for outer_fold, count
                in supported_by_fold.items()
            },
            "reduced_protocols_supported_in_all_folds": int(
                (
                    reduced_summary[
                        "all_folds_support_noninferiority"
                    ]
                ).sum()
            ),
            "reduced_protocols_point_noninferior_in_all_folds": int(
                (
                    reduced_summary[
                        "all_point_estimates_noninferior"
                    ]
                ).sum()
            ),
        },
        "exploratory_best_fixed_candidate": (
            candidate_to_record(
                best_candidate
            )
        ),
        "top_ranked_candidates": [
            candidate_to_record(candidate)
            for _, candidate
            in reduced_summary.head(15).iterrows()
        ],
        "key_candidates": key_candidates,
        "ranking_rule": [
            "More outer folds with point differences inside the margin.",
            "Higher worst-fold noninferiority probability.",
            "Higher mean noninferiority probability.",
            "Fewer speech tasks.",
            "Higher mean paired Macro F1 difference.",
            "Lexicographic subset identifier as the final tie-break.",
        ],
        "interpretation": {
            "universal_fixed_reduced_protocol_supported": False,
            "best_exploratory_subset": str(
                best_candidate["subset_id"]
            ),
            "statement": (
                "No fixed reduced protocol had all five point estimates "
                "inside the noninferiority margin, and no fixed reduced "
                "protocol had confidence-bound support in all five outer "
                "folds. The evidence supports a low-burden adaptive "
                "protocol policy rather than one universal task subset."
            ),
            "limitation": (
                "Candidate ranking is exploratory because the same inner "
                "out-of-fold predictions were used to compare and rank "
                "the fixed protocols."
            ),
        },
        "participant_level_fields": [],
        "source_artifacts": [
            str(INNER_OOF_PREDICTIONS_PATH),
            str(FOLD_ROBUSTNESS_PATH),
            str(CANDIDATE_SUMMARY_PATH),
        ],
    }


def validate_experiment_record(
    record: dict[str, object],
) -> None:
    """Check the tracked record before writing it to the repository."""

    if record.get(
        "experiment"
    ) != "fixed_protocol_robustness":
        raise ValueError(
            "The experiment record has the wrong experiment name."
        )

    coverage = record["coverage"]

    if coverage["outer_folds"] != 5:
        raise ValueError(
            "The experiment record has the wrong outer-fold count."
        )

    if coverage["task_subsets"] != 255:
        raise ValueError(
            "The experiment record has the wrong subset count."
        )

    if coverage["fold_comparisons"] != 1275:
        raise ValueError(
            "The experiment record has the wrong comparison count."
        )

    if record["participant_level_fields"]:
        raise ValueError(
            "Participant-level fields must not appear in the "
            "tracked experiment record."
        )

    results = record["results"]

    if (
        results[
            "reduced_protocols_supported_in_all_folds"
        ]
        != 0
    ):
        raise ValueError(
            "The aggregate record unexpectedly claims that a fixed "
            "reduced protocol was supported in every fold."
        )

    if (
        results[
            "reduced_protocols_point_noninferior_in_all_folds"
        ]
        != 0
    ):
        raise ValueError(
            "The aggregate record unexpectedly claims complete "
            "point-estimate consistency."
        )

def save_experiment_record(
    record: dict[str, object],
) -> None:
    """Atomically save the aggregate protocol-robustness record."""

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
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                record,
                output_file,
                indent=2,
                sort_keys=True,
            )

            output_file.write("\n")

        temporary_path.replace(EXPERIMENT_RECORD_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def main() -> None:
    """Run paired bootstrap comparisons against the full assessment."""

    predictions, subsets = load_protocol_robustness_inputs()

    print("Protocol-robustness inputs validated.")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Task subsets: {len(subsets)}")
    print(
        "Outer folds: "
        f"{predictions['outer_fold'].nunique()}"
    )
    print(
        "Bootstrap replicates per outer fold: "
        f"{BOOTSTRAP_REPLICATES}"
    )
    print(f"Random seed: {BOOTSTRAP_SEED}")
    print(f"Noninferiority margin: {PRIMARY_DELTA}")
    print()

    results = run_protocol_robustness_bootstrap(
        predictions=predictions,
        subsets=subsets,
    )

    validate_fold_robustness_results(results=results, subsets=subsets,)

    save_fold_robustness_results(results)

    candidate_summary = build_candidate_summary(fold_results=results)

    validate_candidate_summary(summary=candidate_summary,subsets=subsets,)

    save_candidate_summary(candidate_summary)

    experiment_record = build_experiment_record(
        fold_results=results,
        candidate_summary=candidate_summary,
    )

    validate_experiment_record(experiment_record)

    save_experiment_record(experiment_record)

    supported = results.loc[
        results["supports_noninferiority"]
        & results["task_count"].lt(
            len(TASKS)
        )
    ]

    reduced_summary = candidate_summary.loc[
        candidate_summary["is_reduced"]
    ]

    strict_candidates = reduced_summary.loc[
        reduced_summary[
            "all_folds_support_noninferiority"
        ]
    ]

    point_consistent_candidates = (
        reduced_summary.loc[
            reduced_summary[
                "all_point_estimates_noninferior"
            ]
        ]
    )

    display_columns = [
        "candidate_rank",
        "subset_id",
        "task_count",
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "observed_difference_mean",
        "observed_difference_min",
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
    ]

    print()
    print("Protocol-robustness bootstrap validated.")
    print(f"Fold-result rows: {len(results)}")
    print(f"Candidate-summary rows: {len(candidate_summary)}")
    print(
        "Reduced protocol/fold comparisons supporting "
        f"noninferiority: {len(supported)}"
    )
    print(
        "Reduced protocols with confidence-bound support "
        f"in every fold: {len(strict_candidates)}"
    )
    print(
        "Reduced protocols with point estimates inside the "
        f"margin in every fold: {len(point_consistent_candidates)}"
    )

    print()
    print("Highest-ranked fixed reduced protocols:")
    print(
        reduced_summary[
            display_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Generated artifacts:")
    print(FOLD_ROBUSTNESS_PATH)
    print(CANDIDATE_SUMMARY_PATH)
    print(EXPERIMENT_RECORD_PATH)

if __name__ == "__main__":
    main()
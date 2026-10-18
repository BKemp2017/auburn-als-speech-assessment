"""Paired participant bootstrap for WavLM selected-versus-full results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EXPECTED_OUTER_FOLDS = 5
PRIMARY_DELTA = 0.02
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
def run_paired_wavlm_bootstrap(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap paired selected-versus-full fold-mean Macro F1."""

    random_generator = np.random.default_rng(
        BOOTSTRAP_SEED
    )
    fold_tables = {
        int(outer_fold): (
            predictions.loc[
                predictions["outer_fold"].eq(
                    outer_fold
                ),
                [
                    "true_label",
                    "selected_predicted_label",
                    "full_predicted_label",
                ],
            ]
            .reset_index(drop=True)
        )
        for outer_fold in sorted(
            predictions["outer_fold"].unique()
        )
    }

    if len(fold_tables) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "The paired bootstrap did not receive all outer folds."
        )

    bootstrap_rows: list[
        dict[str, float | int]
    ] = []

    for bootstrap_id in range(
        BOOTSTRAP_REPLICATES
    ):
        selected_fold_scores: list[
            float
        ] = []
        full_fold_scores: list[
            float
        ] = []

        for fold_table in fold_tables.values():
            # Resample paired participant rows within each outer fold.
            # Rejection only prevents an undefined single-class replicate.
            while True:
                sample_positions = random_generator.integers(
                    low=0,
                    high=len(fold_table),
                    size=len(fold_table),
                )
                sampled = fold_table.iloc[
                    sample_positions
                ]

                if sampled[
                    "true_label"
                ].nunique() == 2:
                    break

            true_labels = sampled[
                "true_label"
            ].to_numpy(dtype=np.int64)
            selected_predictions = sampled[
                "selected_predicted_label"
            ].to_numpy(dtype=np.int64)
            full_predictions = sampled[
                "full_predicted_label"
            ].to_numpy(dtype=np.int64)

            selected_fold_scores.append(
                float(
                    f1_score(
                        true_labels,
                        selected_predictions,
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            full_fold_scores.append(
                float(
                    f1_score(
                        true_labels,
                        full_predictions,
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                )
            )

        selected_score = float(
            np.mean(selected_fold_scores)
        )
        full_score = float(
            np.mean(full_fold_scores)
        )

        bootstrap_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "selected_macro_f1": (
                    selected_score
                ),
                "full_macro_f1": full_score,
                "difference": (
                    selected_score
                    - full_score
                ),
            }
        )

    bootstrap_results = pd.DataFrame(
        bootstrap_rows
    )

    validate_paired_wavlm_bootstrap(
        bootstrap_results
    )

    return bootstrap_results

def validate_paired_wavlm_bootstrap(
    bootstrap_results: pd.DataFrame,
) -> None:
    """Confirm the paired bootstrap is complete and internally consistent."""

    if len(bootstrap_results) != BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"Expected {BOOTSTRAP_REPLICATES} bootstrap rows, found "
            f"{len(bootstrap_results)}."
        )

    if bootstrap_results[
        "bootstrap_id"
    ].nunique() != BOOTSTRAP_REPLICATES:
        raise ValueError(
            "Bootstrap replicate identifiers are not unique."
        )

    metric_columns = [
        "selected_macro_f1",
        "full_macro_f1",
        "difference",
    ]
    metric_values = bootstrap_results[
        metric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(metric_values).all():
        raise ValueError(
            "The paired WavLM bootstrap contains a non-finite value."
        )

    reproduced_differences = (
        bootstrap_results[
            "selected_macro_f1"
        ]
        - bootstrap_results[
            "full_macro_f1"
        ]
    )

    if not np.allclose(
        bootstrap_results["difference"],
        reproduced_differences,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Paired bootstrap differences are internally inconsistent."
        )

def summarize_paired_wavlm_bootstrap(
    bootstrap_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    margin: float = PRIMARY_DELTA,
) -> dict[str, float | bool | int]:
    """Summarize the paired difference and noninferiority evidence."""

    difference_values = bootstrap_results[
        "difference"
    ].to_numpy(dtype=float)
    selected_point = float(
        fold_metrics[
            "outer_test_selected_macro_f1"
        ].mean()
    )
    full_point = float(
        fold_metrics[
            "outer_test_full_macro_f1"
        ].mean()
    )
    difference_point = (
        selected_point
        - full_point
    )
    confidence_lower, confidence_upper = np.quantile(
        difference_values,
        [
            0.025,
            0.975,
        ],
    )
    one_sided_lower = float(
        np.quantile(
            difference_values,
            0.05,
        )
    )
    noninferiority_probability = float(
        np.mean(
            difference_values
            >= -margin
        )
    )

    return {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "noninferiority_margin": float(margin),
        "selected_macro_f1_point": selected_point,
        "full_macro_f1_point": full_point,
        "difference_point": difference_point,
        "difference_bootstrap_mean": float(
            difference_values.mean()
        ),
        "difference_ci_95_lower": float(
            confidence_lower
        ),
        "difference_ci_95_upper": float(
            confidence_upper
        ),
        "difference_one_sided_95_lower": (
            one_sided_lower
        ),
        "noninferiority_probability": (
            noninferiority_probability
        ),
        "point_estimate_inside_margin": bool(
            difference_point >= -margin
        ),
        "two_sided_ci_supports_noninferiority": bool(
            confidence_lower >= -margin
        ),
        "one_sided_bound_supports_noninferiority": bool(
            one_sided_lower >= -margin
        ),
        "folds_inside_margin": int(
            fold_metrics[
                "outer_test_macro_f1_difference"
            ].ge(-margin).sum()
        ),
    }

def print_paired_wavlm_bootstrap_summary(
    summary: dict[str, float | bool | int],
) -> None:
    """Display the final paired WavLM noninferiority evidence."""

    print()
    print("Paired WavLM bootstrap validated.")
    print(
        "Bootstrap replicates: "
        f"{summary['bootstrap_replicates']}"
    )
    print(
        "Selected minus full point estimate: "
        f"{summary['difference_point']:+.6f}"
    )
    print(
        "Bootstrap mean difference: "
        f"{summary['difference_bootstrap_mean']:+.6f}"
    )
    print(
        "Two-sided 95% percentile interval: "
        f"[{summary['difference_ci_95_lower']:+.6f}, "
        f"{summary['difference_ci_95_upper']:+.6f}]"
    )
    print(
        "One-sided 95% lower bound: "
        f"{summary['difference_one_sided_95_lower']:+.6f}"
    )
    print(
        "Noninferiority probability: "
        f"{summary['noninferiority_probability']:.6f}"
    )
    print(
        "Point estimate inside margin: "
        f"{summary['point_estimate_inside_margin']}"
    )
    print(
        "Two-sided CI supports noninferiority: "
        f"{summary['two_sided_ci_supports_noninferiority']}"
    )
    print(
        "One-sided bound supports noninferiority: "
        f"{summary['one_sided_bound_supports_noninferiority']}"
    )
    print(
        "Outer folds inside margin: "
        f"{summary['folds_inside_margin']} / "
        f"{EXPECTED_OUTER_FOLDS}"
    )

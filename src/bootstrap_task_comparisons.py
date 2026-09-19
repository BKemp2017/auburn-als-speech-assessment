# Paired particpant level bootstrap comparisons for reduced ALS speech protocol

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd 

from sklearn.metrics import f1_score
from audit_metadata import TASKS
from evaluate_baselines import (FEATURE_TABLE_PATH, OUTER_FOLDS_PATH, build_pipeline)
from evaluate_task_subsets import (feature_columns_for_tasks, load_frozen_configurations)

OUTPUT_DIR = Path("results/generated")
PREDICTIONS_PATH = OUTPUT_DIR / (
    "bootstrap_protocol_predictions.csv"
)
PROTOCOLS = {
    "TA": (
        "TA",
    ),
    "E+TA+KA": (
        "E",
        "TA",
        "KA",
    ),
    "FULL": tuple(TASKS),
}

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 2026

BOOTSTRAP_RESULTS_PATH = OUTPUT_DIR / (
    "bootstrap_task_comparisons.csv"
)

BOOTSTRAP_SUMMARY_PATH = OUTPUT_DIR / (
    "bootstrap_task_comparison_summary.csv"
)

def predict_protocol_outer_fold(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    protocol_name: str,
    tasks: tuple[str, ...],
    outer_fold: int,
    configuration: dict,
) -> pd.DataFrame:
    # Generate participant predictions for one protocol
    # and one outer fold.

    columns = feature_columns_for_tasks(
        tasks
    )

    test_ids = set(
        outer_folds.loc[
            outer_folds["outer_fold"].eq(
                outer_fold
            ),
            "participant_id",
        ]
    )

    train = features.loc[
        ~features["ID"].isin(
            test_ids
        )
    ].copy()

    test = features.loc[
        features["ID"].isin(
            test_ids
        )
    ].copy()

    X_train = train[
        columns
    ]

    y_train = train[
        "target"
    ]

    X_test = test[
        columns
    ]

    y_test = test[
        "target"
    ]

    if not np.isfinite(
        X_train.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite training predictor detected."
        )

    if not np.isfinite(
        X_test.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite test predictor detected."
        )

    pipeline = build_pipeline(
        configuration
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    return pd.DataFrame(
        {
            "participant_id": test["ID"].to_numpy(),
            "outer_fold": outer_fold,
            "protocol": protocol_name,
            "true_label": y_test.to_numpy(
                dtype=int
            ),
            "predicted_label": predictions.astype(
                int
            ),
        }
    )

def generate_protocol_predictions(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    frozen_configurations: dict[int, dict],
) -> pd.DataFrame:
    # Generate out-of-fold predictions for all selected protocols.

    prediction_frames: list[pd.DataFrame] = []

    for protocol_name, tasks in PROTOCOLS.items():

        for outer_fold in sorted(
            frozen_configurations
        ):

            fold_predictions = (
                predict_protocol_outer_fold(
                    features=features,
                    outer_folds=outer_folds,
                    protocol_name=protocol_name,
                    tasks=tasks,
                    outer_fold=outer_fold,
                    configuration=(
                        frozen_configurations[
                            outer_fold
                        ]
                    ),
                )
            )

            prediction_frames.append(
                fold_predictions
            )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )

def validate_predictions(
        predictions: pd.DataFrame
) -> None:
    # verify paired out of fold prediction structure

    expected_rows = (
        len(PROTOCOLS)
        * 102
    )

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions, "
            f"found {len(predictions)}."
        )

    expected_participants = 102

    for protocol_name in PROTOCOLS:
        protocol_rows = predictions.loc[
            predictions["protocol"].eq(
                protocol_name
            )
        ]

        if len(protocol_rows) != expected_participants:
            raise ValueError(
                f"{protocol_name} does not contain "
                f"{expected_participants} predictions."
            )

        if (
            protocol_rows[
                "participant_id"
            ]
            .nunique()
            != expected_participants
        ):
            raise ValueError(
                f"{protocol_name} does not contain "
                f"{expected_participants} unique participants."
            )

        if (
            protocol_rows[
                "outer_fold"
            ]
            .nunique()
            != 5
        ):
            raise ValueError(
                f"{protocol_name} does not contain "
                "predictions from all five outer folds."
            )
def validate_protocol_pairing(
    predictions: pd.DataFrame,
) -> None:
    # Verify that every protocol contains the same participants,
    # outer-fold assignments, and true labels.

    reference = (
        predictions.loc[
            predictions["protocol"].eq(
                "FULL"
            ),
            [
                "participant_id",
                "outer_fold",
                "true_label",
            ],
        ]
        .sort_values(
            "participant_id"
        )
        .reset_index(
            drop=True
        )
    )

    for protocol_name in PROTOCOLS:

        protocol_rows = (
            predictions.loc[
                predictions["protocol"].eq(
                    protocol_name
                ),
                [
                    "participant_id",
                    "outer_fold",
                    "true_label",
                ],
            ]
            .sort_values(
                "participant_id"
            )
            .reset_index(
                drop=True
            )
        )

        if not protocol_rows.equals(
            reference
        ):
            raise ValueError(
                f"{protocol_name} is not aligned "
                "with the FULL protocol."
            )
def run_paired_bootstrap(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    # Run fold-stratified paired participant bootstrap resampling.

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    bootstrap_records: list[dict] = []

    # Build one aligned participant table for each outer fold.
    fold_tables: dict[int, pd.DataFrame] = {}

    for outer_fold in sorted(
        predictions["outer_fold"].unique()
    ):

        fold_rows = predictions.loc[
            predictions["outer_fold"].eq(
                outer_fold
            )
        ]

        fold_table = (
            fold_rows.pivot(
                index=[
                    "participant_id",
                    "true_label",
                ],
                columns="protocol",
                values="predicted_label",
            )
            .reset_index()
        )

        if fold_table[list(PROTOCOLS)].isna().any().any():
            raise ValueError(
                f"Missing paired prediction in outer fold "
                f"{outer_fold}."
            )

        fold_tables[
            int(outer_fold)
        ] = fold_table

    # Generate bootstrap replicates.
    for bootstrap_id in range(
        N_BOOTSTRAP
    ):

        protocol_fold_scores = {
            protocol_name: []
            for protocol_name in PROTOCOLS
        }

        for outer_fold, fold_table in fold_tables.items():

            while True:
                sample_positions = rng.integers(
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

            y_true = sampled[
                "true_label"
            ].to_numpy()

            for protocol_name in PROTOCOLS:

                y_pred = sampled[
                    protocol_name
                ].to_numpy()

                score = f1_score(
                    y_true,
                    y_pred,
                    labels=[
                        0,
                        1,
                    ],
                    average="macro",
                    zero_division=0,
                )

                protocol_fold_scores[
                    protocol_name
                ].append(
                    float(score)
                )

        ta_score = float(
            np.mean(
                protocol_fold_scores[
                    "TA"
                ]
            )
        )

        three_task_score = float(
            np.mean(
                protocol_fold_scores[
                    "E+TA+KA"
                ]
            )
        )

        full_score = float(
            np.mean(
                protocol_fold_scores[
                    "FULL"
                ]
            )
        )

        bootstrap_records.append(
            {
                "bootstrap_id": bootstrap_id,
                "TA": ta_score,
                "E+TA+KA": three_task_score,
                "FULL": full_score,
                "TA_minus_FULL": (
                    ta_score
                    - full_score
                ),
                "E+TA+KA_minus_FULL": (
                    three_task_score
                    - full_score
                ),
                "E+TA+KA_minus_TA":(
                    three_task_score
                    - ta_score
                ),
            }
        )

    return pd.DataFrame(
        bootstrap_records
    )

def summarize_bootstrap_results(
    bootstrap_results: pd.DataFrame,
    protocol_summary: pd.DataFrame,
) -> pd.DataFrame:
    # Summarize point estimates and percentile bootstrap intervals.

    protocol_points = (
        protocol_summary
        .set_index(
            "protocol"
        )[
            "fold_mean_macro_f1"
        ]
        .to_dict()
    )

    point_estimates = {
        "TA": protocol_points[
            "TA"
        ],
        "E+TA+KA": protocol_points[
            "E+TA+KA"
        ],
        "FULL": protocol_points[
            "FULL"
        ],
        "TA_minus_FULL": (
            protocol_points["TA"] - protocol_points["FULL"]
        ),
        "E+TA+KA_minus_FULL": (
            protocol_points["E+TA+KA"] - protocol_points["FULL"]
        ),
        "E+TA+KA_minus_TA": (
            protocol_points["E+TA+KA"] - protocol_points["TA"]
        ),
    }

    records: list[dict] = []

    for metric_name, point_estimate in (
        point_estimates.items()
    ):

        values = bootstrap_results[
            metric_name
        ].to_numpy()

        lower, upper = np.quantile(
            values,
            [
                0.025,
                0.975,
            ],
        )

        records.append(
            {
                "metric": metric_name,
                "point_estimate": float(
                    point_estimate
                ),
                "bootstrap_mean": float(
                    np.mean(
                        values
                    )
                ),
                "ci_95_lower": float(
                    lower
                ),
                "ci_95_upper": float(
                    upper
                ),
            }
        )

    return pd.DataFrame(
        records
    )

def summarize_fold_mean_macro_f1(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    # Recalculate Macro F1 in each outer fold,
    # then summarize across the five folds.

    records: list[dict] = []

    for protocol_name in PROTOCOLS:

        protocol_rows = predictions.loc[
            predictions["protocol"].eq(
                protocol_name
            )
        ]

        fold_scores: list[float] = []

        for outer_fold in sorted(
            protocol_rows[
                "outer_fold"
            ].unique()
        ):

            fold_rows = protocol_rows.loc[
                protocol_rows[
                    "outer_fold"
                ].eq(
                    outer_fold
                )
            ]

            score = f1_score(
                fold_rows[
                    "true_label"
                ],
                fold_rows[
                    "predicted_label"
                ],
                average="macro",
                zero_division=0,
            )

            fold_scores.append(
                float(score)
            )

        records.append(
            {
                "protocol": protocol_name,
                "fold_mean_macro_f1": float(
                    np.mean(
                        fold_scores
                    )
                ),
                "fold_std_macro_f1": float(
                    np.std(
                        fold_scores,
                        ddof=1,
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )

def main() -> None:
    # load the participant feature tabls and frozen outer fold assignments.

    features = pd.read_csv(
        FEATURE_TABLE_PATH
    )

    outer_folds = pd.read_csv(
        OUTER_FOLDS_PATH
    )

    # Load the model configuration previously selected
    # for each outer fold using the full eight-task baseline.

    frozen_configurations = (
        load_frozen_configurations()
    )

    # Generate one out-of-fold prediction per participant
    # for each selected speech protocol.

    predictions = generate_protocol_predictions(
        features=features,
        outer_folds=outer_folds,
        frozen_configurations=(
            frozen_configurations
        ),
    )

    # Make sure the prediction table contains the expected
    # participants, protocols, and outer folds.

    validate_predictions(
        predictions
    )

    validate_protocol_pairing(
        predictions
    )

    # Reconstruct the five-fold Macro F1 results
    # from the participant-level predictions.

    summary = summarize_fold_mean_macro_f1(
        predictions
    )

    bootstrap_results = run_paired_bootstrap(
        predictions
    )

    bootstrap_summary = (
        summarize_bootstrap_results(
            bootstrap_results = (
                bootstrap_results
            ),
            protocol_summary = summary,
        )
    )

    # Create the generated-results directory if needed.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save participant-level predictions locally.
    # This path is ignored by Git.

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    bootstrap_results.to_csv(
    BOOTSTRAP_RESULTS_PATH,
    index=False,
    )

    bootstrap_summary.to_csv(
        BOOTSTRAP_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Out-of-fold predictions validated."
    )

    print(
        f"Prediction rows: {len(predictions)}"
    )

    print()
    print(
        "Reproduced protocol performance:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Predictions written to:",
        PREDICTIONS_PATH,
    )

    print()
    print(
        f"Bootstrap replicates: "
        f"{len(bootstrap_results)}"
    )

    print()
    print(
        "Paired bootstrap summary:"
    )

    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Bootstrap results written to:",
        BOOTSTRAP_RESULTS_PATH,
    )

    print(
        "Bootstrap summary written to:",
        BOOTSTRAP_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
"""Generate inner out-of-fold predictions for the stability analysis"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from audit_metadata import TASKS
from evaluate_baselines import (
    build_pipeline,
    load_experiment_inputs,
    validate_experiment_inputs,
)
from evaluate_nested_task_selection import (
    INNER_SUBSET_RESULTS_PATH,
    PRIMARY_DELTA,
    inner_assignments_for_outer_fold,
)
from evaluate_task_subsets import (
    feature_columns_for_tasks,
    generate_task_subsets,
    load_frozen_configurations,
)

OUTPUT_DIR = Path("results/generated")

INNER_OOF_PREDICTIONS_PATH = (
    OUTPUT_DIR / "protocol_stability_inner_predictions.csv"
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026

BOOTSTRAP_SELECTIONS_PATH = (
    OUTPUT_DIR / "protocol_stability_bootstrap_selections.csv"
)

def predict_subset_inner_fold(
        features: pd.DataFrame,
        assignment: dict,
        tasks: tuple[str, ...],
        configuration: dict,
) -> list[dict[str, object]]:
    # Generate participant predictions for one subset and inner fold

    train_ids = set(assignment["inner_train_ids"])
    validation_ids = set(assignment["inner_validation_ids"])
    outer_test_ids = set(assignment["outer_test_ids"])

    # These partitions must remain disjoint because the predictions 
    # will later be treated as genuine inner out of fold predictions.

    if train_ids & validation_ids:
        raise ValueError("Inner training and validation participants overlap")

    if train_ids & outer_test_ids:
        raise ValueError("Inner training and outer test participants overlap")

    if validation_ids & outer_test_ids:
        raise ValueError("Inner validation and outer test participants overlap")

    train = features.loc[features["ID"].isin(train_ids)].copy()
    validation = features.loc[features["ID"].isin(validation_ids)].copy()

    if set(train["ID"]) != train_ids:
        raise ValueError("Not all inner training participants were found")

    if set(validation["ID"]) != validation_ids:
        raise ValueError("Not all inner validation participants were found")

    columns = feature_columns_for_tasks(tasks)
    pipeline = build_pipeline(configuration)

    pipeline.fit(
        train[columns],
        train["target"],
    )

    predicted_labels = pipeline.predict(validation[columns])

    records: list[dict[str, object]] = []
    for participant_id, true_label, predicted_label in zip(
        validation["ID"],
        validation["target"],
        predicted_labels,
    ):
        records.append(
            {
                "participant_id": participant_id,
                "outer_fold": int (
                    assignment["outer_fold"]
                ),
                "inner_fold": int (
                    assignment["inner_fold"]
                ),
                "tasks": ",".join(tasks),
                "subset_id": "+".join(tasks),
                "task_count": len(tasks),
                "predictor_count": len(columns),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
            }
        )

    return records

def generate_inner_oof_predictions(
    features: pd.DataFrame,
    nested_folds: list[dict],
    subsets: list[tuple[str, ...]],
    frozen_configurations: dict[int, dict],
) -> pd.DataFrame:
    """Generate paired inner out-of-fold predictions for every subset."""

    prediction_records: list[dict[str, object]] = []
    total_subsets = len(subsets)

    for outer_fold in sorted(frozen_configurations):
        assignments = inner_assignments_for_outer_fold(
            nested_folds=nested_folds,
            outer_fold=outer_fold,
        )

        configuration = frozen_configurations[
            outer_fold
        ]

        print()
        print(
            f"Generating inner predictions for outer fold "
            f"{outer_fold}..."
        )

        for subset_index, tasks in enumerate(
            subsets,
            start=1,
        ):
            # Print periodic progress without producing more than one
            # thousand nearly identical terminal messages.
            if (
                subset_index == 1
                or subset_index % 25 == 0
                or subset_index == total_subsets
            ):
                print(
                    f"Outer fold {outer_fold}: "
                    f"{subset_index}/{total_subsets} subsets"
                )

            for assignment in assignments:
                prediction_records.extend(
                    predict_subset_inner_fold(
                        features=features,
                        assignment=assignment,
                        tasks=tasks,
                        configuration=configuration,
                    )
                )

    return pd.DataFrame(prediction_records)


def validate_inner_oof_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    nested_folds: list[dict],
    subsets: list[tuple[str, ...]],
    frozen_configurations: dict[int, dict],
) -> None:
    """Validate prediction coverage and reproduce saved inner scores."""

    required_columns = {
        "participant_id",
        "outer_fold",
        "inner_fold",
        "tasks",
        "subset_id",
        "task_count",
        "predictor_count",
        "true_label",
        "predicted_label",
    }

    missing_columns = required_columns - set(
        predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Inner predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_folds = set(frozen_configurations)

    if set(
        predictions["outer_fold"].astype(int).unique()
    ) != expected_folds:
        raise ValueError(
            "Inner predictions do not contain all outer folds."
        )

    if predictions.duplicated(
        subset=[
            "outer_fold",
            "subset_id",
            "participant_id",
        ]
    ).any():
        raise ValueError(
            "A participant received duplicate predictions for the "
            "same outer fold and task subset."
        )

    valid_labels = {0, 1}

    if not set(
        predictions["true_label"].unique()
    ).issubset(valid_labels):
        raise ValueError(
            "An invalid true label was found."
        )

    if not set(
        predictions["predicted_label"].unique()
    ).issubset(valid_labels):
        raise ValueError(
            "An invalid predicted label was found."
        )

    expected_rows = 0

    for outer_fold in sorted(expected_folds):
        assignments = inner_assignments_for_outer_fold(
            nested_folds=nested_folds,
            outer_fold=outer_fold,
        )

        outer_test_ids = set(
            assignments[0]["outer_test_ids"]
        )

        outer_training_ids = (
            set(features["ID"]) - outer_test_ids
        )

        fold_predictions = predictions.loc[
            predictions["outer_fold"].eq(
                outer_fold
            )
        ]

        observed_participants = set(
            fold_predictions["participant_id"]
        )

        if observed_participants != outer_training_ids:
            raise ValueError(
                f"Inner predictions for outer fold {outer_fold} "
                "do not cover the complete outer-training population."
            )

        if observed_participants & outer_test_ids:
            raise ValueError(
                f"Outer-test participants appear in inner predictions "
                f"for outer fold {outer_fold}."
            )

        expected_rows += (
            len(outer_training_ids) * len(subsets)
        )

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} inner predictions, "
            f"but found {len(predictions)}."
        )

    # Every participant belongs to the outer-training population in four
    # of five outer folds and therefore receives four predictions per
    # task subset.
    expected_predictions_per_participant = (
        (len(expected_folds) - 1) * len(subsets)
    )

    participant_counts = predictions[
        "participant_id"
    ].value_counts()

    if not participant_counts.eq(
        expected_predictions_per_participant
    ).all():
        raise ValueError(
            "Participant prediction counts do not match the expected "
            "four outer-training appearances per task subset."
        )

    subset_counts = predictions.groupby(
        "outer_fold"
    )["subset_id"].nunique()

    if not subset_counts.eq(len(subsets)).all():
        raise ValueError(
            "An outer fold does not contain all task subsets."
        )

    full_subset_id = "+".join(TASKS)

    if (
        predictions.loc[
            predictions["subset_id"].eq(
                full_subset_id
            )
        ]["outer_fold"].nunique()
        != len(expected_folds)
    ):
        raise ValueError(
            "The full assessment is missing from an outer fold."
        )

    if not INNER_SUBSET_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Saved nested inner results were not found at "
            f"{INNER_SUBSET_RESULTS_PATH}."
        )

    saved_inner_results = pd.read_csv(
        INNER_SUBSET_RESULTS_PATH
    )

    score_records: list[dict[str, object]] = []

    # Recomputing the 5,100 inner-fold scores proves that prediction
    # capture did not change the established nested experiment.
    for (
        outer_fold,
        subset_id,
        inner_fold,
    ), group in predictions.groupby(
        [
            "outer_fold",
            "subset_id",
            "inner_fold",
        ],
        sort=True,
    ):
        score_records.append(
            {
                "outer_fold": int(outer_fold),
                "subset_id": str(subset_id),
                "inner_fold": int(inner_fold),
                "macro_f1": float(
                    f1_score(
                        group["true_label"],
                        group["predicted_label"],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    recomputed_scores = pd.DataFrame(
        score_records
    )

    score_table = recomputed_scores.pivot(
        index=["outer_fold", "subset_id"],
        columns="inner_fold",
        values="macro_f1",
    ).reset_index()

    score_table = score_table.rename(
        columns={
            inner_fold: (
                f"inner_fold_{inner_fold}_macro_f1"
            )
            for inner_fold in range(4)
        }
    )

    saved_score_columns = [
        f"inner_fold_{inner_fold}_macro_f1"
        for inner_fold in range(4)
    ]

    comparison = saved_inner_results[
        [
            "outer_fold",
            "subset_id",
            *saved_score_columns,
        ]
    ].merge(
        score_table,
        on=["outer_fold", "subset_id"],
        how="inner",
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )

    expected_subset_rows = (
        len(expected_folds) * len(subsets)
    )

    if len(comparison) != expected_subset_rows:
        raise ValueError(
            "Recomputed scores do not match every saved subset result."
        )

    for column in saved_score_columns:
        if not np.allclose(
            comparison[f"{column}_saved"],
            comparison[f"{column}_recomputed"],
        ):
            raise ValueError(
                f"Recomputed predictions do not reproduce {column}."
            )


def save_inner_oof_predictions(
    predictions: pd.DataFrame,
) -> None:
    # Atomically save validated inner out-of-fold predictions.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        INNER_OOF_PREDICTIONS_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        predictions.to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(
            INNER_OOF_PREDICTIONS_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def load_or_generate_inner_oof_predictions(
    features: pd.DataFrame,
    nested_folds: list[dict],
    subsets: list[tuple[str, ...]],
    frozen_configurations: dict[int, dict],
) -> pd.DataFrame:
    """Reuse validated predictions instead of repeating 5,100 fits."""

    if INNER_OOF_PREDICTIONS_PATH.exists():
        print(
            "Reusing existing inner out-of-fold predictions from "
            f"{INNER_OOF_PREDICTIONS_PATH}."
        )

        predictions = pd.read_csv(
            INNER_OOF_PREDICTIONS_PATH
        )

        generated = False
    else:
        predictions = generate_inner_oof_predictions(
            features=features,
            nested_folds=nested_folds,
            subsets=subsets,
            frozen_configurations=frozen_configurations,
        )

        generated = True

    validate_inner_oof_predictions(
        predictions=predictions,
        features=features,
        nested_folds=nested_folds,
        subsets=subsets,
        frozen_configurations=frozen_configurations,
    )

    if generated:
        save_inner_oof_predictions(
            predictions
        )

    return predictions


def bootstrap_weights(
    rng: np.random.Generator,
    replicates: int,
    participant_count: int,
) -> np.ndarray:
    """Create participant bootstrap counts for one inner fold."""

    # Each row contains one bootstrap sample represented as participant
    # multiplicities. This is faster than copying prediction rows.
    draws = rng.integers(
        0,
        participant_count,
        size=(replicates, participant_count),
    )

    weights = np.zeros(
        (replicates, participant_count),
        dtype=float,
    )

    bootstrap_rows = np.repeat(
        np.arange(replicates),
        participant_count,
    )

    np.add.at(
        weights,
        (
            bootstrap_rows,
            draws.reshape(-1),
        ),
        1.0,
    )

    return weights


def bootstrap_macro_f1(
    weights: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> np.ndarray:
    """Calculate Macro F1 for every replicate and task subset."""

    positive_mask = true_labels == 1
    negative_mask = true_labels == 0

    if not positive_mask.any() or not negative_mask.any():
        raise ValueError(
            "An inner fold does not contain both target classes."
        )

    # predicted_labels has one row per subset and one column per
    # participant. Matrix multiplication applies every paired bootstrap
    # sample to all 255 subsets simultaneously.
    positive_predictions = predicted_labels[
        :,
        positive_mask,
    ]

    negative_predictions = predicted_labels[
        :,
        negative_mask,
    ]

    positive_weights = weights[
        :,
        positive_mask,
    ]

    negative_weights = weights[
        :,
        negative_mask,
    ]

    true_positives = (
        positive_weights
        @ positive_predictions.T
    )

    false_positives = (
        negative_weights
        @ negative_predictions.T
    )

    positive_totals = positive_weights.sum(
        axis=1,
        keepdims=True,
    )

    negative_totals = negative_weights.sum(
        axis=1,
        keepdims=True,
    )

    false_negatives = (
        positive_totals - true_positives
    )

    true_negatives = (
        negative_totals - false_positives
    )

    positive_denominator = (
        2.0 * true_positives
        + false_positives
        + false_negatives
    )

    negative_denominator = (
        2.0 * true_negatives
        + false_positives
        + false_negatives
    )

    positive_f1 = np.divide(
        2.0 * true_positives,
        positive_denominator,
        out=np.zeros_like(true_positives),
        where=positive_denominator != 0,
    )

    negative_f1 = np.divide(
        2.0 * true_negatives,
        negative_denominator,
        out=np.zeros_like(true_negatives),
        where=negative_denominator != 0,
    )

    return (
        positive_f1 + negative_f1
    ) / 2.0


def select_bootstrap_protocols(
    mean_scores: np.ndarray,
    score_standard_deviations: np.ndarray,
    competitive: np.ndarray,
    task_counts: np.ndarray,
    lexicographic_ranks: np.ndarray,
) -> np.ndarray:
    # Apply the locked deterministic selection rule to every replicate.

    # First retain competitive protocols with the fewest tasks.
    candidate_task_counts = np.where(
        competitive,
        task_counts[np.newaxis, :],
        np.inf,
    )

    minimum_task_counts = candidate_task_counts.min(
        axis=1
    )

    eligible = (
        competitive
        & (
            task_counts[np.newaxis, :]
            == minimum_task_counts[:, np.newaxis]
        )
    )

    # Among equally small protocols, prefer the highest mean Macro F1.
    eligible_means = np.where(
        eligible,
        mean_scores,
        -np.inf,
    )

    best_means = eligible_means.max(
        axis=1
    )

    eligible &= mean_scores == best_means[
        :,
        np.newaxis,
    ]

    # Remaining ties are resolved by lower fold-to-fold variability.
    eligible_standard_deviations = np.where(
        eligible,
        score_standard_deviations,
        np.inf,
    )

    best_standard_deviations = (
        eligible_standard_deviations.min(
            axis=1
        )
    )

    eligible &= (
        score_standard_deviations
        == best_standard_deviations[
            :,
            np.newaxis,
        ]
    )

    # The final tie-break reproduces Python's tuple ordering used by the
    # primary nested-selection implementation.
    eligible_lexicographic_ranks = np.where(
        eligible,
        lexicographic_ranks[
            np.newaxis,
            :,
        ],
        len(lexicographic_ranks) + 1,
    )

    return eligible_lexicographic_ranks.argmin(
        axis=1
    )


def bootstrap_outer_fold_selections(
    predictions: pd.DataFrame,
    outer_fold: int,
    subsets: list[tuple[str, ...]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap the selection procedure within one outer fold."""

    fold_predictions = predictions.loc[
        predictions["outer_fold"].eq(
            outer_fold
        )
    ].copy()

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

    # Four score matrices are retained because the primary selection
    # rule uses both the mean and sample SD across inner folds.
    fold_scores = np.empty(
        (
            BOOTSTRAP_REPLICATES,
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

        weights = bootstrap_weights(
            rng=rng,
            replicates=BOOTSTRAP_REPLICATES,
            participant_count=participant_count,
        )

        fold_scores[:, inner_fold, :] = (
            bootstrap_macro_f1(
                weights=weights,
                true_labels=true_labels[
                    inner_mask
                ],
                predicted_labels=predicted_labels[
                    :,
                    inner_mask,
                ],
            )
        )

    mean_scores = fold_scores.mean(
        axis=1
    )

    score_standard_deviations = fold_scores.std(
        axis=1,
        ddof=1,
    )

    full_subset_id = "+".join(TASKS)
    full_subset_index = subset_ids.index(
        full_subset_id
    )

    full_references = mean_scores[
        :,
        full_subset_index,
    ]

    competitive_thresholds = (
        full_references - PRIMARY_DELTA
    )

    competitive = (
        mean_scores
        >= competitive_thresholds[
            :,
            np.newaxis,
        ]
    )

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

    selected_indices = select_bootstrap_protocols(
        mean_scores=mean_scores,
        score_standard_deviations=(
            score_standard_deviations
        ),
        competitive=competitive,
        task_counts=task_counts,
        lexicographic_ranks=(
            lexicographic_ranks
        ),
    )

    selected_subsets = np.asarray(
        subset_ids,
        dtype=object,
    )[selected_indices]

    selected_tasks = np.asarray(
        [
            ",".join(tasks)
            for tasks in subsets
        ],
        dtype=object,
    )[selected_indices]

    return pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "bootstrap_replicate": np.arange(
                BOOTSTRAP_REPLICATES
            ),
            "delta": PRIMARY_DELTA,
            "selected_subset": selected_subsets,
            "selected_tasks": selected_tasks,
            "task_count": task_counts[
                selected_indices
            ],
            "selected_inner_macro_f1_mean": (
                mean_scores[
                    np.arange(BOOTSTRAP_REPLICATES),
                    selected_indices,
                ]
            ),
            "selected_inner_macro_f1_std": (
                score_standard_deviations[
                    np.arange(BOOTSTRAP_REPLICATES),
                    selected_indices,
                ]
            ),
            "inner_full_reference": full_references,
            "competitive_threshold": (
                competitive_thresholds
            ),
            "competitive_subset_count": (
                competitive.sum(axis=1)
            ),
        }
    )


def run_protocol_stability_bootstrap(
    predictions: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> pd.DataFrame:
    """Run paired participant resampling in all five outer folds."""

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    fold_results: list[pd.DataFrame] = []

    for outer_fold in range(5):
        print(
            f"Bootstrapping protocol selection for outer fold "
            f"{outer_fold}..."
        )

        fold_results.append(
            bootstrap_outer_fold_selections(
                predictions=predictions,
                outer_fold=outer_fold,
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
                "bootstrap_replicate",
            ]
        )
        .reset_index(drop=True)
    )


def validate_bootstrap_selections(
    selections: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> None:
    """Validate bootstrap coverage and selected protocol metadata."""

    expected_rows = (
        5 * BOOTSTRAP_REPLICATES
    )

    if len(selections) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} bootstrap selections, "
            f"but found {len(selections)}."
        )

    if selections.duplicated(
        subset=[
            "outer_fold",
            "bootstrap_replicate",
        ]
    ).any():
        raise ValueError(
            "Duplicate outer-fold bootstrap replicates were found."
        )

    fold_counts = selections[
        "outer_fold"
    ].value_counts()

    if not fold_counts.eq(
        BOOTSTRAP_REPLICATES
    ).all():
        raise ValueError(
            "An outer fold does not contain every bootstrap replicate."
        )

    valid_subset_ids = {
        "+".join(tasks)
        for tasks in subsets
    }

    if not set(
        selections["selected_subset"]
    ).issubset(valid_subset_ids):
        raise ValueError(
            "An invalid task subset was selected."
        )

    expected_task_counts = selections[
        "selected_tasks"
    ].map(
        lambda tasks: len(
            str(tasks).split(",")
        )
    )

    if not selections["task_count"].eq(
        expected_task_counts
    ).all():
        raise ValueError(
            "A selected task count does not match its task list."
        )

    if not np.allclose(
        selections["delta"],
        PRIMARY_DELTA,
    ):
        raise ValueError(
            "A bootstrap selection used the wrong delta."
        )

    numeric_columns = [
        "selected_inner_macro_f1_mean",
        "selected_inner_macro_f1_std",
        "inner_full_reference",
        "competitive_threshold",
        "competitive_subset_count",
    ]

    if not np.isfinite(
        selections[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A bootstrap selection contains a non-finite value."
        )

    if not selections[
        "competitive_subset_count"
    ].between(1, len(subsets)).all():
        raise ValueError(
            "A competitive-subset count is outside the valid range."
        )


def save_bootstrap_selections(
    selections: pd.DataFrame,
) -> None:
    """Atomically save participant-resampled protocol selections."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        BOOTSTRAP_SELECTIONS_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        selections.to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(
            BOOTSTRAP_SELECTIONS_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def main() -> None:
    """Run participant-resampled protocol stability analysis."""

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
            "The final subset is not the full assessment."
        )

    if set(frozen_configurations) != {
        0,
        1,
        2,
        3,
        4,
    }:
        raise ValueError(
            "Frozen configurations do not match the outer folds."
        )

    print("Protocol-stability inputs validated.")
    print(f"Participants: {len(features)}")
    print(f"Task subsets: {len(subsets)}")
    print(f"Bootstrap replicates per outer fold: {BOOTSTRAP_REPLICATES}")
    print(f"Random seed: {BOOTSTRAP_SEED}")
    print()

    predictions = (
        load_or_generate_inner_oof_predictions(
            features=features,
            nested_folds=nested_folds,
            subsets=subsets,
            frozen_configurations=frozen_configurations,
        )
    )

    print()
    print("Inner prediction artifact validated.")
    print(f"Prediction rows: {len(predictions)}")
    print()

    selections = run_protocol_stability_bootstrap(
        predictions=predictions,
        subsets=subsets,
    )

    validate_bootstrap_selections(
        selections=selections,
        subsets=subsets,
    )

    save_bootstrap_selections(
        selections
    )

    print()
    print("Protocol-stability bootstrap validated.")
    print(f"Selection rows: {len(selections)}")
    print()

    print("Most frequently selected exact protocols:")

    print(
        selections["selected_subset"]
        .value_counts()
        .head(15)
        .rename_axis("selected_subset")
        .reset_index(name="selection_count")
        .to_string(index=False)
    )

    print()
    print("Selected task-count distribution:")

    print(
        selections["task_count"]
        .value_counts()
        .sort_index()
        .rename_axis("task_count")
        .reset_index(name="selection_count")
        .to_string(index=False)
    )

    task_frequency = (
        selections["selected_tasks"]
        .str.split(",")
        .explode()
        .value_counts()
        .reindex(TASKS, fill_value=0)
    )

    print()
    print("Individual task selection frequency:")

    print(
        task_frequency
        .rename_axis("task")
        .reset_index(name="selection_count")
        .to_string(index=False)
    )

    print()
    print("Generated artifact:")
    print(BOOTSTRAP_SELECTIONS_PATH)


if __name__ == "__main__":
    main()
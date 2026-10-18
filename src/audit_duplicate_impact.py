"""Audit whether exact audio duplicates affect the ALS-only experiment."""

from pathlib import Path

import json
import numpy as np
import pandas as pd

from audit_metadata import TASKS
from evaluate_baselines import load_experiment_inputs


WAVEFORM_AUDIT_PATH = Path(
    "results/generated/raw_audio_waveform_audit.csv"
)
OUTPUT_PATH = Path(
    "results/generated/raw_audio_duplicate_impact.csv"
)

EXPERIMENT_RECORD_PATH = Path(
    "experiments/raw_audio_audit.json"
)

# These are the same thresholds used when interpreting the waveform audit.
NEAR_SILENT_RMS_DBFS = -50.0
HIGH_SILENCE_FRAME_FRACTION = 0.50
MATERIAL_CLIPPING_FRACTION = 0.001

def normalize_participant_id(value: object) -> str:
    """Normalize participant IDs before joining generated artifacts."""

    participant_id = str(value).strip()

    # Excel sometimes converts integer identifiers to strings ending in
    # ".0". Remove that suffix without changing nonnumeric identifiers.
    if participant_id.endswith(".0"):
        possible_integer = participant_id[:-2]

        if possible_integer.isdigit():
            participant_id = possible_integer

    return participant_id


def load_duplicate_impact_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the waveform audit and the frozen ALS modeling inputs."""

    if not WAVEFORM_AUDIT_PATH.exists():
        raise FileNotFoundError(
            "Run src/audit_raw_audio.py before auditing duplicate impact. "
            f"Missing: {WAVEFORM_AUDIT_PATH}"
        )

    waveform_audit = pd.read_csv(
        WAVEFORM_AUDIT_PATH
    )

    features, labels, _ = load_experiment_inputs()

    waveform_audit = waveform_audit.copy()
    features = features.copy()
    labels = labels.copy()

    waveform_audit["participant_id"] = (
        waveform_audit["participant_id"].map(
            normalize_participant_id
        )
    )
    features["ID"] = features["ID"].map(
        normalize_participant_id
    )
    labels["participant_id"] = labels[
        "participant_id"
    ].map(normalize_participant_id)

    return (
        waveform_audit,
        features,
        labels,
    )


def validate_duplicate_impact_inputs(
    waveform_audit: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    """Validate identifiers and columns before comparing duplicate records."""

    required_waveform_columns = {
        "participant_id",
        "task",
        "category",
        "signal_sha256",
    }
    missing_waveform_columns = (
        required_waveform_columns.difference(
            waveform_audit.columns
        )
    )

    if missing_waveform_columns:
        raise ValueError(
            "Waveform audit is missing required columns: "
            f"{sorted(missing_waveform_columns)}"
        )

    required_feature_columns = {
        "ID",
        "target",
    }
    missing_feature_columns = (
        required_feature_columns.difference(
            features.columns
        )
    )

    if missing_feature_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            f"{sorted(missing_feature_columns)}"
        )

    required_label_columns = {
        "participant_id",
        "target",
        "outer_fold",
    }
    missing_label_columns = (
        required_label_columns.difference(
            labels.columns
        )
    )

    if missing_label_columns:
        raise ValueError(
            "Label table is missing required columns: "
            f"{sorted(missing_label_columns)}"
        )

    if features["ID"].duplicated().any():
        raise ValueError(
            "The feature table contains duplicate participant IDs."
        )

    if labels["participant_id"].duplicated().any():
        raise ValueError(
            "The label table contains duplicate participant IDs."
        )

    feature_ids = set(features["ID"])
    label_ids = set(labels["participant_id"])

    if feature_ids != label_ids:
        raise ValueError(
            "The feature and label tables do not contain the same "
            "ALS participants."
        )

    if len(feature_ids) != 102:
        raise ValueError(
            "Expected 102 participants in the frozen ALS experiment, "
            f"but found {len(feature_ids)}."
        )

    # Confirm that each task still has the five frozen acoustic predictors.
    for task in TASKS:
        task_columns = [
            column
            for column in features.columns
            if column.endswith(
                f"_{task}"
            )
        ]

        if len(task_columns) != 5:
            raise ValueError(
                f"Expected five predictors for task {task}, "
                f"but found {len(task_columns)}."
            )


def build_duplicate_impact_table(
    waveform_audit: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Compare ALS participants connected by exact duplicate recordings."""

    duplicate_rows = waveform_audit.loc[
        waveform_audit.duplicated(
            subset="signal_sha256",
            keep=False,
        )
    ].copy()

    als_duplicate_rows = duplicate_rows.loc[
        duplicate_rows["category"].eq("ALS")
    ].copy()

    duplicate_connected_als_count = (
        als_duplicate_rows["participant_id"].nunique()
    )

    features_by_id = features.set_index("ID")
    labels_by_id = labels.set_index(
        "participant_id"
    )

    task_order = {
        task: index
        for index, task in enumerate(TASKS)
    }

    comparison_records: list[dict[str, object]] = []

    # Only hashes shared by at least two ALS participants can directly
    # duplicate information inside the ALS-only modeling experiment.
    for _, group in duplicate_rows.groupby(
        "signal_sha256",
        sort=True,
    ):
        tasks = sorted(
            group["task"].unique(),
            key=task_order.get,
        )

        if len(tasks) != 1:
            raise ValueError(
                "An exact signal hash appeared under multiple tasks."
            )

        task = tasks[0]

        als_participants = sorted(
            group.loc[
                group["category"].eq("ALS"),
                "participant_id",
            ].unique()
        )

        if len(als_participants) < 2:
            continue

        task_feature_columns = [
            column
            for column in features.columns
            if column.endswith(
                f"_{task}"
            )
        ]

        for left_index, left_participant in enumerate(
            als_participants
        ):
            for right_participant in als_participants[
                left_index + 1:
            ]:
                if left_participant not in features_by_id.index:
                    raise ValueError(
                        "A duplicate-connected ALS participant is missing "
                        "from the frozen feature table."
                    )

                if right_participant not in features_by_id.index:
                    raise ValueError(
                        "A duplicate-connected ALS participant is missing "
                        "from the frozen feature table."
                    )

                left_label = labels_by_id.loc[
                    left_participant
                ]
                right_label = labels_by_id.loc[
                    right_participant
                ]

                left_features = (
                    features_by_id.loc[
                        left_participant,
                        task_feature_columns,
                    ]
                    .astype(float)
                )
                right_features = (
                    features_by_id.loc[
                        right_participant,
                        task_feature_columns,
                    ]
                    .astype(float)
                )

                # Series.equals treats matching NaN positions as equal and
                # therefore provides a strict stored-value comparison.
                features_exactly_equal = (
                    left_features.equals(
                        right_features
                    )
                )

                features_numerically_equal = bool(
                    np.allclose(
                        left_features.to_numpy(),
                        right_features.to_numpy(),
                        rtol=0.0,
                        atol=1e-12,
                        equal_nan=True,
                    )
                )

                absolute_differences = np.abs(
                    left_features.to_numpy()
                    - right_features.to_numpy()
                )

                if np.isnan(
                    absolute_differences
                ).all():
                    maximum_difference = 0.0
                else:
                    maximum_difference = float(
                        np.nanmax(
                            absolute_differences
                        )
                    )

                comparison_records.append(
                    {
                        # IDs are retained only while grouping repeated
                        # pairs and are removed before saving or printing.
                        "participant_a": left_participant,
                        "participant_b": right_participant,
                        "task": task,
                        "target_a": int(
                            left_label["target"]
                        ),
                        "target_b": int(
                            right_label["target"]
                        ),
                        "outer_fold_a": int(
                            left_label["outer_fold"]
                        ),
                        "outer_fold_b": int(
                            right_label["outer_fold"]
                        ),
                        "features_exactly_equal": (
                            features_exactly_equal
                        ),
                        "features_numerically_equal": (
                            features_numerically_equal
                        ),
                        "maximum_absolute_feature_difference": (
                            maximum_difference
                        ),
                    }
                )

    if not comparison_records:
        return (
            pd.DataFrame(),
            duplicate_connected_als_count,
        )

    comparison_details = pd.DataFrame(
        comparison_records
    )

    pair_records: list[dict[str, object]] = []

    for pair_number, (
        _,
        pair_rows,
    ) in enumerate(
        comparison_details.groupby(
            [
                "participant_a",
                "participant_b",
            ],
            sort=True,
        ),
        start=1,
    ):
        pair_rows = pair_rows.sort_values(
            "task",
            key=lambda values: values.map(
                task_order
            ),
        )

        left_target = int(
            pair_rows["target_a"].iloc[0]
        )
        right_target = int(
            pair_rows["target_b"].iloc[0]
        )
        left_fold = int(
            pair_rows["outer_fold_a"].iloc[0]
        )
        right_fold = int(
            pair_rows["outer_fold_b"].iloc[0]
        )

        pair_records.append(
            {
                "duplicate_als_pair": pair_number,
                "shared_task_count": (
                    pair_rows["task"].nunique()
                ),
                "shared_tasks": "+".join(
                    pair_rows["task"]
                ),
                "target_pair": "+".join(
                    str(value)
                    for value in sorted(
                        [
                            left_target,
                            right_target,
                        ]
                    )
                ),
                "same_target": (
                    left_target == right_target
                ),
                "outer_fold_pair": "+".join(
                    str(value)
                    for value in sorted(
                        [
                            left_fold,
                            right_fold,
                        ]
                    )
                ),
                "same_outer_fold": (
                    left_fold == right_fold
                ),
                "all_task_features_exactly_equal": bool(
                    pair_rows[
                        "features_exactly_equal"
                    ].all()
                ),
                "all_task_features_numerically_equal": bool(
                    pair_rows[
                        "features_numerically_equal"
                    ].all()
                ),
                "maximum_absolute_feature_difference": float(
                    pair_rows[
                        "maximum_absolute_feature_difference"
                    ].max()
                ),
            }
        )

    # No participant identifiers leave this function.
    impact_table = pd.DataFrame(
        pair_records
    )

    return (
        impact_table,
        duplicate_connected_als_count,
    )


def save_duplicate_impact_table(
    impact_table: pd.DataFrame,
) -> None:
    """Save the participant-free duplicate impact summary."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    impact_table.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def print_duplicate_impact_summary(
    impact_table: pd.DataFrame,
    duplicate_connected_als_count: int,
) -> None:
    """Print the findings that determine the duplicate-handling policy."""

    print("Raw-audio duplicate impact audit validated.")
    print(
        "Duplicate-connected ALS participants:",
        duplicate_connected_als_count,
    )
    print(
        "ALS duplicate participant pairs:",
        len(impact_table),
    )

    if impact_table.empty:
        print(
            "\nNo exact audio duplicate is shared by two ALS "
            "participants."
        )
        return

    print("\nALS duplicate impact:")
    print(
        impact_table.to_string(
            index=False
        )
    )

    if (~impact_table["same_target"]).any():
        print(
            "\nWARNING: At least one exact recording is assigned to "
            "different ALS speech-severity targets."
        )
    else:
        print(
            "\nThe duplicate-connected ALS participants have the same "
            "speech-severity target."
        )

    if (~impact_table["same_outer_fold"]).any():
        print(
            "WARNING: At least one duplicate ALS pair crosses outer-fold "
            "boundaries, creating potential train/test duplication."
        )
    else:
        print(
            "Duplicate-connected ALS participants are assigned to the "
            "same outer fold."
        )

    if impact_table[
        "all_task_features_numerically_equal"
    ].any():
        print(
            "WARNING: At least one duplicated recording also has "
            "matching handcrafted predictors."
        )
    else:
        print(
            "The duplicated recordings do not have identical "
            "handcrafted predictor values."
        )

    print("\nGenerated artifact:")
    print(OUTPUT_PATH)

def build_experiment_record(
    waveform_audit: pd.DataFrame,
    impact_table: pd.DataFrame,
    duplicate_connected_als_count: int,
) -> dict[str, object]:
    """Build a participant-free record of the raw-audio audit."""

    participant_categories = (
        waveform_audit[
            [
                "participant_id",
                "category",
            ]
        ]
        .drop_duplicates()
    )

    category_counts = {
        str(category): int(count)
        for category, count in (
            participant_categories["category"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    recordings_by_task = {
        str(task): int(count)
        for task, count in (
            waveform_audit["task"]
            .value_counts()
            .reindex(TASKS)
            .items()
        )
    }

    sample_rate_distribution = {
        str(int(sample_rate)): int(count)
        for sample_rate, count in (
            waveform_audit["sample_rate_hz"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    channel_distribution = {
        str(int(channels)): int(count)
        for channels, count in (
            waveform_audit["channels"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    duration_by_task: dict[str, dict[str, float]] = {}

    for task in TASKS:
        task_durations = waveform_audit.loc[
            waveform_audit["task"].eq(task),
            "duration_seconds",
        ]

        duration_by_task[task] = {
            "minimum_seconds": float(
                task_durations.min()
            ),
            "median_seconds": float(
                task_durations.median()
            ),
            "maximum_seconds": float(
                task_durations.max()
            ),
        }

    signal_counts = waveform_audit[
        "signal_sha256"
    ].value_counts()
    duplicate_signal_counts = signal_counts.loc[
        signal_counts.gt(1)
    ]

    duplicate_rows = waveform_audit.loc[
        waveform_audit["signal_sha256"].isin(
            duplicate_signal_counts.index
        )
    ]

    # A profile signature represents the ordered set of all eight task
    # recordings belonging to one participant.
    participant_profiles = (
        waveform_audit.pivot(
            index="participant_id",
            columns="task",
            values="signal_sha256",
        )
        .reindex(columns=TASKS)
    )

    if participant_profiles.isna().any().any():
        raise ValueError(
            "Cannot summarize complete audio profiles because at least "
            "one participant is missing a task recording."
        )

    profile_signatures = participant_profiles.astype(
        str
    ).agg(
        "|".join,
        axis=1,
    )
    profile_counts = profile_signatures.value_counts()
    duplicate_profile_counts = profile_counts.loc[
        profile_counts.gt(1)
    ]

    quality_summary = {
        "near_silent_rms_dbfs_threshold": (
            NEAR_SILENT_RMS_DBFS
        ),
        "near_silent_recordings": int(
            waveform_audit["rms_dbfs"]
            .le(NEAR_SILENT_RMS_DBFS)
            .sum()
        ),
        "high_silence_frame_fraction_threshold": (
            HIGH_SILENCE_FRAME_FRACTION
        ),
        "high_silence_recordings": int(
            waveform_audit["silent_frame_fraction"]
            .gt(HIGH_SILENCE_FRAME_FRACTION)
            .sum()
        ),
        "recordings_with_any_endpoint_clipping": int(
            waveform_audit["clipped_sample_count"]
            .gt(0)
            .sum()
        ),
        "material_clipping_fraction_threshold": (
            MATERIAL_CLIPPING_FRACTION
        ),
        "recordings_with_material_clipping": int(
            waveform_audit["clipped_sample_fraction"]
            .ge(MATERIAL_CLIPPING_FRACTION)
            .sum()
        ),
    }

    # Converting through JSON replaces NumPy scalar types with ordinary
    # Python values while retaining the participant-free comparison rows.
    als_pair_comparisons = json.loads(
        impact_table.to_json(
            orient="records"
        )
    )

    return {
        "experiment": "voc_als_raw_audio_audit",
        "participant_level_fields": [],
        "dataset": {
            "recordings": int(
                len(waveform_audit)
            ),
            "participants": int(
                waveform_audit[
                    "participant_id"
                ].nunique()
            ),
            "participants_by_category": (
                category_counts
            ),
            "recordings_by_task": (
                recordings_by_task
            ),
        },
        "audio_format": {
            "sample_rate_hz_distribution": (
                sample_rate_distribution
            ),
            "channel_distribution": (
                channel_distribution
            ),
            "formats": sorted(
                waveform_audit["format"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "subtypes": sorted(
                waveform_audit["subtype"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "duration_by_task": duration_by_task,
        "quality_summary": quality_summary,
        "exact_duplicate_summary": {
            "duplicate_signal_groups": int(
                len(duplicate_signal_counts)
            ),
            "files_in_duplicate_signal_groups": int(
                duplicate_signal_counts.sum()
            ),
            "duplicate_connected_participants": int(
                duplicate_rows[
                    "participant_id"
                ].nunique()
            ),
            "duplicate_connected_categories": sorted(
                duplicate_rows["category"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "full_duplicate_profile_groups": int(
                len(duplicate_profile_counts)
            ),
            "participants_in_full_duplicate_profiles": int(
                duplicate_profile_counts.sum()
            ),
        },
        "als_primary_analysis_impact": {
            "duplicate_connected_als_participants": int(
                duplicate_connected_als_count
            ),
            "duplicate_als_pairs": int(
                len(impact_table)
            ),
            "contradictory_target_pairs": int(
                (
                    ~impact_table["same_target"]
                ).sum()
            ),
            "cross_outer_fold_pairs": int(
                (
                    ~impact_table["same_outer_fold"]
                ).sum()
            ),
            "pairs_with_matching_handcrafted_features": int(
                impact_table[
                    "all_task_features_numerically_equal"
                ].sum()
            ),
            "pair_comparisons": als_pair_comparisons,
        },
        "interpretation": (
            "One complete eight-task audio profile is duplicated across "
            "an ALS participant and a healthy-control participant. A "
            "second ALS participant shares the same A-task recording. "
            "The ALS pair has contradictory severity targets, appears "
            "in different outer folds, and has identical A-task "
            "handcrafted predictors."
        ),
        "required_follow_up": [
            (
                "Run a duplicate-exclusion sensitivity analysis after "
                "removing both affected ALS participants."
            ),
            (
                "Exclude or group the entire three-participant duplicate "
                "component before any ALS-versus-control audio model."
            ),
            (
                "Do not treat the current raw-audio dataset as suitable "
                "for neural modeling until duplicate handling is frozen."
            ),
        ],
    }

def save_experiment_record(
    experiment_record: dict[str, object],
) -> None:
    """Save the participant-free audit record tracked by Git."""

    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EXPERIMENT_RECORD_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            experiment_record,
            output_file,
            indent=2,
        )
        output_file.write("\n")

def main() -> None:
    """Measure how exact audio duplicates affect the ALS experiment."""

    (
        waveform_audit,
        features,
        labels,
    ) = load_duplicate_impact_inputs()

    validate_duplicate_impact_inputs(
        waveform_audit=waveform_audit,
        features=features,
        labels=labels,
    )

    (
        impact_table,
        duplicate_connected_als_count,
    ) = build_duplicate_impact_table(
        waveform_audit=waveform_audit,
        features=features,
        labels=labels,
    )

    save_duplicate_impact_table(
        impact_table
    )

    experiment_record = build_experiment_record(
        waveform_audit=waveform_audit,
        impact_table=impact_table,
        duplicate_connected_als_count=(
            duplicate_connected_als_count
        ),
    )

    save_experiment_record(experiment_record)

    print_duplicate_impact_summary(
        impact_table=impact_table,
        duplicate_connected_als_count=(
            duplicate_connected_als_count
        ),
    )
    print("\nTracked experiment record:")
    print(EXPERIMENT_RECORD_PATH)

if __name__ == "__main__":
    main()
"""Quantify recording burden for the selected WavLM task policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_duplicate_impact import normalize_participant_id
from audit_metadata import TASKS


WAVEFORM_AUDIT_PATH = Path(
    "results/generated/raw_audio_waveform_audit.csv"
)
WAVLM_FOLD_METRICS_PATH = Path(
    "results/generated/wavlm_task_selection_fold_metrics.csv"
)
WAVLM_PREDICTIONS_PATH = Path(
    "results/generated/wavlm_task_selection_predictions.csv"
)

OUTPUT_DIRECTORY = Path("results/generated")
PARTICIPANT_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "recording_duration_burden_participants.csv"
)
FOLD_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "recording_duration_burden_by_fold.csv"
)
PROTOCOL_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "recording_duration_burden_by_protocol.csv"
)
TASK_RESULTS_PATH = OUTPUT_DIRECTORY / (
    "recording_duration_burden_by_task.csv"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/recording_duration_burden.json"
)

EXPECTED_PARTICIPANTS = 100
EXPECTED_OUTER_FOLDS = 5
EXPECTED_RECORDINGS_PER_PARTICIPANT = len(TASKS)

# These values come from the waveform audit and define the active-speech
# proxy used here. The proxy is useful for relative burden comparisons,
# but it is not a direct measurement of speaking effort or fatigue.
FRAME_DURATION_SECONDS = 0.025
FRAME_HOP_SECONDS = 0.010
SILENCE_THRESHOLD_DBFS = -50.0

BURDEN_PAIRS = {
    "recording_count": (
        "full_recording_count",
        "selected_recording_count",
    ),
    "recorded_seconds": (
        "full_recorded_seconds",
        "selected_recorded_seconds",
    ),
    "endpoint_trimmed_seconds": (
        "full_endpoint_trimmed_seconds",
        "selected_endpoint_trimmed_seconds",
    ),
    "nonsilent_seconds_proxy": (
        "full_nonsilent_seconds_proxy",
        "selected_nonsilent_seconds_proxy",
    ),
}

NUMERIC_WAVEFORM_COLUMNS = (
    "duration_seconds",
    "silent_frame_fraction",
    "leading_silence_seconds",
    "trailing_silence_seconds",
)

REQUIRED_WAVEFORM_COLUMNS = {
    "participant_id",
    "category",
    "task",
    *NUMERIC_WAVEFORM_COLUMNS,
}

REQUIRED_PREDICTION_COLUMNS = {
    "participant_id",
    "outer_fold",
    "selected_subset",
}

REQUIRED_FOLD_COLUMNS = {
    "outer_fold",
    "selected_subset",
    "selected_tasks",
    "task_count",
}

def parse_task_subset(subset_id: object) -> tuple[str, ...]:
    """Convert a saved subset identifier into validated task names."""

    tasks = tuple(
        task.strip()
        for task in str(subset_id).split("+")
        if task.strip()
    )

    if not tasks:
        raise ValueError(
            "A selected protocol does not contain any speech tasks."
        )

    unknown_tasks = set(tasks).difference(TASKS)

    if unknown_tasks:
        raise ValueError(
            "A selected protocol contains unknown tasks: "
            f"{sorted(unknown_tasks)}"
        )

    if len(tasks) != len(set(tasks)):
        raise ValueError(
            f"A selected protocol repeats a task: {subset_id}"
        )

    # Put every protocol in the repository's canonical task order. This
    # keeps joins and tracked summaries deterministic.
    return tuple(
        task
        for task in TASKS
        if task in tasks
    )

def load_burden_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load waveform measurements and finalized WavLM selections."""

    required_paths = (
        WAVEFORM_AUDIT_PATH,
        WAVLM_FOLD_METRICS_PATH,
        WAVLM_PREDICTIONS_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required burden-analysis input was not found: {path}"
            )

    waveform_audit = pd.read_csv(
        WAVEFORM_AUDIT_PATH
    )
    fold_metrics = pd.read_csv(
        WAVLM_FOLD_METRICS_PATH
    )
    predictions = pd.read_csv(
        WAVLM_PREDICTIONS_PATH
    )

    return waveform_audit, fold_metrics, predictions

def validate_required_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Stop early when an upstream artifact has changed shape."""

    missing_columns = required_columns.difference(
        table.columns
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

def prepare_predictions(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the held-out protocol assignment table."""

    validate_required_columns(
        table=predictions,
        required_columns=REQUIRED_PREDICTION_COLUMNS,
        table_name="WavLM predictions",
    )
    validate_required_columns(
        table=fold_metrics,
        required_columns=REQUIRED_FOLD_COLUMNS,
        table_name="WavLM fold metrics",
    )

    prepared = predictions[
        [
            "participant_id",
            "outer_fold",
            "selected_subset",
        ]
    ].copy()
    prepared["participant_id"] = prepared[
        "participant_id"
    ].map(normalize_participant_id)
    prepared["outer_fold"] = pd.to_numeric(
        prepared["outer_fold"],
        errors="raise",
    ).astype(int)
    prepared["selected_subset"] = (
        prepared["selected_subset"]
        .astype(str)
        .str.strip()
    )

    if len(prepared) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} held-out predictions, "
            f"but found {len(prepared)}."
        )

    if prepared["participant_id"].duplicated().any():
        raise ValueError(
            "A participant has more than one held-out WavLM prediction."
        )

    expected_folds = set(range(EXPECTED_OUTER_FOLDS))
    observed_folds = set(prepared["outer_fold"])

    if observed_folds != expected_folds:
        raise ValueError(
            "Held-out predictions do not cover outer folds 0 through 4."
        )

    fold_reference = fold_metrics[
        [
            "outer_fold",
            "selected_subset",
            "selected_tasks",
            "task_count",
        ]
    ].copy()
    fold_reference["outer_fold"] = pd.to_numeric(
        fold_reference["outer_fold"],
        errors="raise",
    ).astype(int)
    fold_reference["task_count"] = pd.to_numeric(
        fold_reference["task_count"],
        errors="raise",
    ).astype(int)

    if len(fold_reference) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Expected one WavLM metric row for each of five outer folds."
        )

    if fold_reference["outer_fold"].duplicated().any():
        raise ValueError(
            "WavLM fold metrics repeat an outer fold."
        )

    fold_reference["parsed_tasks"] = fold_reference[
        "selected_subset"
    ].map(parse_task_subset)

    for fold_row in fold_reference.itertuples(index=False):
        saved_tasks = tuple(
            task.strip()
            for task in str(fold_row.selected_tasks).split(",")
            if task.strip()
        )
        canonical_saved_tasks = tuple(
            task
            for task in TASKS
            if task in saved_tasks
        )

        if canonical_saved_tasks != fold_row.parsed_tasks:
            raise ValueError(
                "A WavLM fold metric has inconsistent selected task "
                f"fields for outer fold {fold_row.outer_fold}."
            )

        if fold_row.task_count != len(fold_row.parsed_tasks):
            raise ValueError(
                "A WavLM fold metric has an incorrect task count for "
                f"outer fold {fold_row.outer_fold}."
            )

        participant_subsets = set(
            prepared.loc[
                prepared["outer_fold"].eq(
                    fold_row.outer_fold
                ),
                "selected_subset",
            ]
        )

        if participant_subsets != {fold_row.selected_subset}:
            raise ValueError(
                "Participant predictions disagree with the finalized "
                f"selection for outer fold {fold_row.outer_fold}."
            )

    prepared["selected_tasks"] = prepared[
        "selected_subset"
    ].map(parse_task_subset)
    prepared["selected_task_count"] = prepared[
        "selected_tasks"
    ].map(len)

    return prepared

def prepare_waveform_audit(
    waveform_audit: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the 100 retained ALS participants and derive burden proxies."""

    validate_required_columns(
        table=waveform_audit,
        required_columns=REQUIRED_WAVEFORM_COLUMNS,
        table_name="Raw-audio waveform audit",
    )

    prepared = waveform_audit.copy()
    prepared["participant_id"] = prepared[
        "participant_id"
    ].map(normalize_participant_id)
    prepared["category"] = (
        prepared["category"]
        .astype(str)
        .str.strip()
    )
    prepared["task"] = (
        prepared["task"]
        .astype(str)
        .str.strip()
    )

    for column in NUMERIC_WAVEFORM_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    retained_ids = set(
        predictions["participant_id"]
    )
    prepared = prepared.loc[
        prepared["participant_id"].isin(
            retained_ids
        )
    ].copy()

    if len(prepared) != (
        EXPECTED_PARTICIPANTS
        * EXPECTED_RECORDINGS_PER_PARTICIPANT
    ):
        raise ValueError(
            "Expected 800 waveform rows for the retained cohort, but "
            f"found {len(prepared)}."
        )

    if set(prepared["participant_id"]) != retained_ids:
        raise ValueError(
            "Waveform audit does not cover every retained WavLM participant."
        )

    if set(prepared["category"]) != {"ALS"}:
        raise ValueError(
            "Retained burden rows must all belong to ALS participants."
        )

    if set(prepared["task"]) != set(TASKS):
        raise ValueError(
            "Retained waveform rows do not cover all eight speech tasks."
        )

    duplicate_recordings = prepared.duplicated(
        subset=["participant_id", "task"]
    )

    if duplicate_recordings.any():
        raise ValueError(
            "A retained participant has duplicate rows for one task."
        )

    task_counts = prepared.groupby(
        "participant_id"
    )["task"].nunique()

    if not task_counts.eq(
        EXPECTED_RECORDINGS_PER_PARTICIPANT
    ).all():
        raise ValueError(
            "Every retained participant must have one recording for each "
            "of the eight speech tasks."
        )

    numeric_values = prepared[
        list(NUMERIC_WAVEFORM_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Retained waveform burden fields contain non-finite values."
        )

    if not prepared["duration_seconds"].gt(0.0).all():
        raise ValueError(
            "Every retained recording must have a positive duration."
        )

    if not prepared["silent_frame_fraction"].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise ValueError(
            "Silent-frame fractions must remain between zero and one."
        )

    if not prepared[
        [
            "leading_silence_seconds",
            "trailing_silence_seconds",
        ]
    ].ge(0.0).all().all():
        raise ValueError(
            "Endpoint-silence durations cannot be negative."
        )

    # Endpoint trimming estimates the captured utterance span after only
    # leading and trailing silence are removed. It does not remove pauses
    # occurring inside the recording.
    prepared["endpoint_trimmed_seconds"] = (
        prepared["duration_seconds"]
        - prepared["leading_silence_seconds"]
        - prepared["trailing_silence_seconds"]
    ).clip(lower=0.0)

    # Multiplying duration by the fraction of nonsilent frames yields a
    # transparent active-speech proxy under the audit's fixed -50 dBFS
    # threshold. Overlapping frames make this an estimate, not a stopwatch.
    prepared["nonsilent_seconds_proxy"] = (
        prepared["duration_seconds"]
        * (
            1.0
            - prepared["silent_frame_fraction"]
        )
    )

    return prepared

def reduction_fraction(
    full_values: pd.Series,
    selected_values: pd.Series,
) -> pd.Series:
    """Calculate the fraction of burden removed for each participant."""

    if not full_values.gt(0.0).all():
        raise ValueError(
            "Full-assessment burden must be positive before reduction."
        )

    return 1.0 - (
        selected_values / full_values
    )

def build_participant_burden(
    waveform_audit: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each participant's selected tasks with all eight tasks."""

    joined = waveform_audit.merge(
        predictions,
        on="participant_id",
        how="inner",
        validate="many_to_one",
    )
    joined["selected_recording"] = [
        task in selected_tasks
        for task, selected_tasks in zip(
            joined["task"],
            joined["selected_tasks"],
            strict=True,
        )
    ]

    full_summary = (
        joined.groupby(
            [
                "participant_id",
                "outer_fold",
                "selected_subset",
                "selected_task_count",
            ],
            as_index=False,
        )
        .agg(
            full_recording_count=("task", "size"),
            full_recorded_seconds=(
                "duration_seconds",
                "sum",
            ),
            full_endpoint_trimmed_seconds=(
                "endpoint_trimmed_seconds",
                "sum",
            ),
            full_nonsilent_seconds_proxy=(
                "nonsilent_seconds_proxy",
                "sum",
            ),
        )
    )

    selected_summary = (
        joined.loc[joined["selected_recording"]]
        .groupby(
            "participant_id",
            as_index=False,
        )
        .agg(
            selected_recording_count=("task", "size"),
            selected_recorded_seconds=(
                "duration_seconds",
                "sum",
            ),
            selected_endpoint_trimmed_seconds=(
                "endpoint_trimmed_seconds",
                "sum",
            ),
            selected_nonsilent_seconds_proxy=(
                "nonsilent_seconds_proxy",
                "sum",
            ),
        )
    )

    results = full_summary.merge(
        selected_summary,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    for burden_name, (
        full_column,
        selected_column,
    ) in BURDEN_PAIRS.items():
        results[
            f"{burden_name}_reduction_fraction"
        ] = reduction_fraction(
            full_values=results[full_column],
            selected_values=results[selected_column],
        )

    return results.sort_values(
        ["outer_fold", "participant_id"]
    ).reset_index(drop=True)

def validate_participant_burden(
    results: pd.DataFrame,
) -> None:
    """Verify that the comparison is complete and internally consistent."""

    if len(results) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} burden rows, "
            f"but found {len(results)}."
        )

    if results["participant_id"].duplicated().any():
        raise ValueError(
            "Participant burden results contain duplicate identifiers."
        )

    if not results["full_recording_count"].eq(
        EXPECTED_RECORDINGS_PER_PARTICIPANT
    ).all():
        raise ValueError(
            "Full-assessment burden must include all eight recordings."
        )

    if not results["selected_recording_count"].eq(
        results["selected_task_count"]
    ).all():
        raise ValueError(
            "Selected recording counts do not match the selected protocols."
        )

    for burden_name, (
        full_column,
        selected_column,
    ) in BURDEN_PAIRS.items():
        full_values = results[full_column].to_numpy(
            dtype=float
        )
        selected_values = results[selected_column].to_numpy(
            dtype=float
        )
        reduction_values = results[
            f"{burden_name}_reduction_fraction"
        ].to_numpy(dtype=float)

        if not (
            np.isfinite(full_values).all()
            and np.isfinite(selected_values).all()
            and np.isfinite(reduction_values).all()
        ):
            raise ValueError(
                f"{burden_name} results contain non-finite values."
            )

        if not np.less_equal(
            selected_values,
            full_values + 1e-12,
        ).all():
            raise ValueError(
                f"Selected {burden_name} exceeds full-assessment burden."
            )

        if not (
            (reduction_values >= -1e-12).all()
            and (reduction_values < 1.0).all()
        ):
            raise ValueError(
                f"{burden_name} reduction fractions are out of range."
            )

def summarize_participant_group(
    group: pd.DataFrame,
) -> dict[str, int | float]:
    """Build one participant-free burden summary for a cohort slice."""

    summary: dict[str, int | float] = {
        "participants": int(len(group)),
        "mean_selected_tasks": float(
            group["selected_task_count"].mean()
        ),
        "median_selected_tasks": float(
            group["selected_task_count"].median()
        ),
    }

    for burden_name, (
        full_column,
        selected_column,
    ) in BURDEN_PAIRS.items():
        full_total = float(group[full_column].sum())
        selected_total = float(
            group[selected_column].sum()
        )

        if full_total <= 0.0:
            raise ValueError(
                f"Cannot summarize nonpositive full {burden_name}."
            )

        summary[f"full_{burden_name}_total"] = full_total
        summary[f"selected_{burden_name}_total"] = selected_total
        summary[
            f"full_{burden_name}_mean_per_participant"
        ] = float(group[full_column].mean())
        summary[
            f"selected_{burden_name}_mean_per_participant"
        ] = float(group[selected_column].mean())
        summary[
            f"{burden_name}_reduction_fraction"
        ] = 1.0 - (selected_total / full_total)
        summary[
            f"{burden_name}_reduction_percent"
        ] = 100.0 * (
            1.0 - (selected_total / full_total)
        )

    return summary

def build_fold_results(
    participant_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize burden for each held-out outer fold."""

    rows: list[dict[str, object]] = []

    for outer_fold, fold_group in participant_results.groupby(
        "outer_fold",
        sort=True,
    ):
        selected_subsets = set(
            fold_group["selected_subset"]
        )

        if len(selected_subsets) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} has multiple selected protocols."
            )

        rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_subset": next(
                    iter(selected_subsets)
                ),
                **summarize_participant_group(
                    fold_group
                ),
            }
        )

    results = pd.DataFrame(rows)

    if len(results) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            "Fold burden results do not contain all five outer folds."
        )

    return results

def build_protocol_results(
    participant_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize burden after pooling identical selected protocols."""

    rows = [
        {
            "selected_subset": selected_subset,
            **summarize_participant_group(
                protocol_group
            ),
        }
        for selected_subset, protocol_group in (
            participant_results.groupby(
                "selected_subset",
                sort=True,
            )
        )
    ]

    return pd.DataFrame(rows).sort_values(
        ["mean_selected_tasks", "selected_subset"]
    ).reset_index(drop=True)

def build_task_results(
    waveform_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Describe why recording-count and duration savings can differ."""

    results = (
        waveform_audit.groupby(
            "task",
            as_index=False,
        )
        .agg(
            recordings=("participant_id", "size"),
            recorded_seconds_mean=(
                "duration_seconds",
                "mean",
            ),
            recorded_seconds_median=(
                "duration_seconds",
                "median",
            ),
            endpoint_trimmed_seconds_mean=(
                "endpoint_trimmed_seconds",
                "mean",
            ),
            nonsilent_seconds_proxy_mean=(
                "nonsilent_seconds_proxy",
                "mean",
            ),
            silent_frame_fraction_mean=(
                "silent_frame_fraction",
                "mean",
            ),
        )
    )
    task_order = {
        task: index
        for index, task in enumerate(TASKS)
    }
    results["task_order"] = results["task"].map(
        task_order
    )
    results = (
        results.sort_values("task_order")
        .drop(columns="task_order")
        .reset_index(drop=True)
    )

    if len(results) != len(TASKS):
        raise ValueError(
            "Task burden results do not contain all eight speech tasks."
        )

    if not results["recordings"].eq(
        EXPECTED_PARTICIPANTS
    ).all():
        raise ValueError(
            "Each task must contain 100 retained recordings."
        )

    return results

def dataframe_records(
    table: pd.DataFrame,
) -> list[dict[str, object]]:
    """Convert aggregate tables into JSON-safe native Python records."""

    records: list[dict[str, object]] = []

    for record in table.to_dict(orient="records"):
        clean_record: dict[str, object] = {}

        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean_record[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean_record[key] = float(value)
            else:
                clean_record[key] = value

        records.append(clean_record)

    return records

def build_experiment_record(
    participant_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    protocol_results: pd.DataFrame,
    task_results: pd.DataFrame,
) -> dict[str, object]:
    """Build the participant-free record committed with the experiment."""

    aggregate = summarize_participant_group(
        participant_results
    )

    return {
        "experiment": "wavlm_selected_protocol_recording_burden",
        "participant_level_fields": [],
        "design": {
            "participants": EXPECTED_PARTICIPANTS,
            "speech_tasks": list(TASKS),
            "full_assessment_task_count": len(TASKS),
            "selection_source": (
                "duplicate_filtered_wavlm_nested_task_selection"
            ),
            "selection_policy": (
                "fold-specific protocol selected using inner CV, then "
                "applied to that fold's held-out participants"
            ),
            "excluded_duplicate_connected_participants": 2,
        },
        "burden_definitions": {
            "recording_count": (
                "number of required task recordings"
            ),
            "recorded_seconds": (
                "WAV header duration summed across required tasks"
            ),
            "endpoint_trimmed_seconds": (
                "recorded duration after subtracting detected leading "
                "and trailing silence"
            ),
            "nonsilent_seconds_proxy": (
                "recorded duration multiplied by one minus the fraction "
                "of silent analysis frames"
            ),
            "waveform_frame_duration_seconds": (
                FRAME_DURATION_SECONDS
            ),
            "waveform_frame_hop_seconds": FRAME_HOP_SECONDS,
            "silence_threshold_dbfs": SILENCE_THRESHOLD_DBFS,
        },
        "aggregate_results": aggregate,
        "fold_results": dataframe_records(
            fold_results
        ),
        "protocol_results": dataframe_records(
            protocol_results
        ),
        "task_duration_results": dataframe_records(
            task_results
        ),
        "interpretation": {
            "supported": (
                "The evaluated fold-specific WavLM policy requires fewer "
                "recordings and less observed recording time than the "
                "complete eight-task assessment."
            ),
            "not_supported": (
                "These retrospective duration proxies do not establish "
                "reduced patient fatigue, do not validate one universal "
                "task protocol, and do not replace prospective clinical "
                "burden measurement."
            ),
        },
    }

def validate_experiment_record(
    record: dict[str, object],
) -> None:
    """Prevent participant identifiers from entering the tracked record."""

    if record["participant_level_fields"] != []:
        raise ValueError(
            "Tracked burden record must not list participant-level fields."
        )

    serialized = json.dumps(
        record,
        allow_nan=False,
    )

    private_field_names = (
        "participant_id",
        "inner_train_ids",
        "inner_validation_ids",
        "outer_test_ids",
    )

    for field_name in private_field_names:
        if field_name in serialized:
            raise ValueError(
                "Tracked burden record contains a participant-level field: "
                f"{field_name}"
            )

def atomic_save_csv(
    table: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV completely before replacing an older generated file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
    )

    try:
        table.to_csv(
            temporary_path,
            index=False,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def save_burden_results(
    participant_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    protocol_results: pd.DataFrame,
    task_results: pd.DataFrame,
    experiment_record: dict[str, object],
) -> None:
    """Save private generated rows and the public aggregate record."""

    atomic_save_csv(
        participant_results,
        PARTICIPANT_RESULTS_PATH,
    )
    atomic_save_csv(
        fold_results,
        FOLD_RESULTS_PATH,
    )
    atomic_save_csv(
        protocol_results,
        PROTOCOL_RESULTS_PATH,
    )
    atomic_save_csv(
        task_results,
        TASK_RESULTS_PATH,
    )

    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = EXPERIMENT_RECORD_PATH.with_suffix(
        ".json.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                experiment_record,
                output_file,
                indent=2,
                allow_nan=False,
            )
            output_file.write("\n")

        temporary_path.replace(
            EXPERIMENT_RECORD_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def print_burden_summary(
    participant_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    experiment_record: dict[str, object],
) -> None:
    """Print the values needed for the final report and slides."""

    aggregate = experiment_record[
        "aggregate_results"
    ]

    print("Recording-duration burden analysis validated.")
    print(f"Participants: {len(participant_results)}")
    print(
        "Mean selected tasks: "
        f"{aggregate['mean_selected_tasks']:.3f} / {len(TASKS)}"
    )
    print(
        "Recording-count reduction: "
        f"{aggregate['recording_count_reduction_percent']:.2f}%"
    )
    print(
        "Recorded-duration reduction: "
        f"{aggregate['recorded_seconds_reduction_percent']:.2f}%"
    )
    print(
        "Endpoint-trimmed duration reduction: "
        f"{aggregate['endpoint_trimmed_seconds_reduction_percent']:.2f}%"
    )
    print(
        "Nonsilent-frame duration reduction: "
        f"{aggregate['nonsilent_seconds_proxy_reduction_percent']:.2f}%"
    )

    print()
    print("Burden by held-out outer fold:")
    print(
        fold_results[
            [
                "outer_fold",
                "selected_subset",
                "participants",
                "mean_selected_tasks",
                "recording_count_reduction_percent",
                "recorded_seconds_reduction_percent",
                "nonsilent_seconds_proxy_reduction_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("Generated artifacts:")
    print(PARTICIPANT_RESULTS_PATH)
    print(FOLD_RESULTS_PATH)
    print(PROTOCOL_RESULTS_PATH)
    print(TASK_RESULTS_PATH)
    print(EXPERIMENT_RECORD_PATH)

def main() -> None:
    """Run the final recording-duration burden analysis."""

    (
        waveform_audit,
        fold_metrics,
        predictions,
    ) = load_burden_inputs()

    prepared_predictions = prepare_predictions(
        predictions=predictions,
        fold_metrics=fold_metrics,
    )
    prepared_waveforms = prepare_waveform_audit(
        waveform_audit=waveform_audit,
        predictions=prepared_predictions,
    )

    print("Recording-duration burden inputs validated.")
    print(f"Retained participants: {len(prepared_predictions)}")
    print(f"Retained recordings: {len(prepared_waveforms)}")
    print(f"Speech tasks: {len(TASKS)}")
    print(f"Outer folds: {EXPECTED_OUTER_FOLDS}")
    print(
        "Selected protocols: "
        f"{sorted(prepared_predictions['selected_subset'].unique())}"
    )
    print(
        "Active-speech proxy: 25 ms frames, 10 ms hop, "
        "-50 dBFS silence threshold"
    )
    print()

    participant_results = build_participant_burden(
        waveform_audit=prepared_waveforms,
        predictions=prepared_predictions,
    )
    validate_participant_burden(
        participant_results
    )

    fold_results = build_fold_results(
        participant_results
    )
    protocol_results = build_protocol_results(
        participant_results
    )
    task_results = build_task_results(
        prepared_waveforms
    )
    experiment_record = build_experiment_record(
        participant_results=participant_results,
        fold_results=fold_results,
        protocol_results=protocol_results,
        task_results=task_results,
    )
    validate_experiment_record(
        experiment_record
    )

    save_burden_results(
        participant_results=participant_results,
        fold_results=fold_results,
        protocol_results=protocol_results,
        task_results=task_results,
        experiment_record=experiment_record,
    )
    print_burden_summary(
        participant_results=participant_results,
        fold_results=fold_results,
        experiment_record=experiment_record,
    )

if __name__ == "__main__":
    main()

"""Audit VOC-ALS recording coverage and audio-file headers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audit_metadata import TASKS, load_metadata


RAW_AUDIO_DIR = Path("data/raw/VOC-ALS")
METADATA_PATH = Path("data/raw/VOC-ALS.xlsx")

OUTPUT_DIR = Path("results/generated")

RECORDING_INVENTORY_PATH = (
    OUTPUT_DIR / "raw_audio_inventory.csv"
)

WAVEFORM_AUDIT_PATH = (
    OUTPUT_DIR / "raw_audio_waveform_audit.csv"
)

FRAME_DURATION_SECONDS = 0.025
FRAME_HOP_SECONDS = 0.010
SILENCE_THRESHOLD_DBFS = -50.0
MATERIAL_CLIPPING_FRACTION = 0.001
HIGH_SILENCE_FRAME_FRACTION = 0.50

TASK_DIRECTORY_MAP = {
    "phonationA": "A",
    "phonationE": "E",
    "phonationI": "I",
    "phonationO": "O",
    "phonationU": "U",
    "rhythmPA": "PA",
    "rhythmTA": "TA",
    "rhythmKA": "KA",
}

EXPECTED_PARTICIPANTS = 153
EXPECTED_ALS_PARTICIPANTS = 102
EXPECTED_HEALTHY_CONTROLS = 51
EXPECTED_RECORDINGS_PER_TASK = 153
EXPECTED_RECORDINGS_PER_PARTICIPANT = 8
EXPECTED_TOTAL_RECORDINGS = 1224
EXPECTED_ALS_RECORDINGS = 816
EXPECTED_HEALTHY_RECORDINGS = 408


def load_participant_metadata(
    path: Path,
) -> pd.DataFrame:
    """Load only the fields needed to validate recording ownership."""

    metadata = load_metadata(path)

    required_columns = {
        "ID",
        "Category",
        "ALSFRS-R_SpeechSubscore",
    }

    missing_columns = (required_columns - set(metadata.columns))

    if missing_columns:
        raise ValueError(
            "Metadata is missing columns required by the audio audit: "
            f"{sorted(missing_columns)}"
        )

    participants = metadata[
        [
            "ID",
            "Category",
            "ALSFRS-R_SpeechSubscore",
        ]
    ].copy()

    if participants["ID"].isna().any():
        raise ValueError(
            "Metadata contains a missing participant ID."
        )

    participants["ID"] = (
        participants["ID"]
        .astype(str)
        .str.strip()
    )

    participants["Category"] = (
        participants["Category"]
        .astype(str)
        .str.strip()
    )

    participants[
        "ALSFRS-R_SpeechSubscore"
    ] = pd.to_numeric(
        participants[
            "ALSFRS-R_SpeechSubscore"
        ],
        errors="coerce",
    )

    if participants["ID"].duplicated().any():
        raise ValueError(
            "Metadata contains a duplicate participant ID."
        )

    if len(participants) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} metadata participants, "
            f"but found {len(participants)}."
        )

    category_counts = participants["Category"].value_counts()

    if (
        category_counts.get("ALS", 0)
        != EXPECTED_ALS_PARTICIPANTS
    ):
        raise ValueError(
            "Metadata does not contain the expected 102 ALS participants."
        )

    if (
        category_counts.get("HC", 0)
        != EXPECTED_HEALTHY_CONTROLS
    ):
        raise ValueError(
            "Metadata does not contain the expected 51 healthy controls."
        )

    return participants

def find_audio_files(
    audio_directory: Path,
) -> list[Path]:
    """Find WAV files without depending on filename case."""

    if not audio_directory.exists():
        raise FileNotFoundError(
            "The VOC-ALS audio directory was not found at "
            f"{audio_directory}."
        )

    return sorted(
        path
        for path in audio_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == ".wav"
        )
    )

def parse_recording_identity(
    path: Path,
) -> tuple[str, str]:
    """Extract participant and task identities from an official filename."""

    task_directory = path.parent.name

    if task_directory not in TASK_DIRECTORY_MAP:
        raise ValueError(
            f"Recording appears under an unexpected task directory: "
            f"{task_directory}"
        )

    expected_suffix = (f"_{task_directory}")

    if not path.stem.endswith(
        expected_suffix
    ):
        raise ValueError(
            "Recording filename does not end with its task-directory "
            f"label: {path.name}"
        )

    participant_id = path.stem[
        :-len(expected_suffix)
    ].strip()

    if not participant_id:
        raise ValueError(
            f"Recording has an empty participant ID: {path.name}"
        )

    task = TASK_DIRECTORY_MAP[task_directory]

    return participant_id, task

def inspect_audio_header(
    path: Path,
    audio_directory: Path,
) -> dict[str, object]:
    """Read one audio header without modifying or resampling its signal."""

    participant_id, task = (
        parse_recording_identity(
            path
        )
    )

    record: dict[str, object] = {
        "relative_path": str(
            path.relative_to(
                audio_directory
            )
        ),
        "participant_id": participant_id,
        "task": task,
        "task_directory": path.parent.name,
        "file_size_bytes": int(
            path.stat().st_size
        ),
        "readable": False,
        "sample_rate_hz": np.nan,
        "channels": np.nan,
        "frames": np.nan,
        "duration_seconds": np.nan,
        "format": None,
        "subtype": None,
        "endian": None,
        "header_error": None,
    }

    try:
        information = sf.info(
            path
        )
    except (OSError, RuntimeError) as error:
        # Keep the failed row in the generated inventory so every bad file
        # can be diagnosed without hiding failures from the final count.
        record["header_error"] = str(
            error
        )

        return record

    record.update(
        {
            "readable": True,
            "sample_rate_hz": int(
                information.samplerate
            ),
            "channels": int(
                information.channels
            ),
            "frames": int(
                information.frames
            ),
            "duration_seconds": float(
                information.duration
            ),
            "format": str(
                information.format
            ),
            "subtype": str(
                information.subtype
            ),
            "endian": str(
                information.endian
            ),
        }
    )

    return record

def build_audio_inventory(
    audio_directory: Path,
    participants: pd.DataFrame,
) -> pd.DataFrame:
    """Build a participant-level inventory in the ignored results folder."""

    audio_paths = find_audio_files(audio_directory)

    records = [
        inspect_audio_header(
            path=path,
            audio_directory=audio_directory,
        )
        for path in audio_paths
    ]

    inventory = pd.DataFrame(records)

    if inventory.empty:
        raise ValueError(
            "No WAV recordings were found."
        )

    participant_fields = participants.rename(
        columns={
            "ID": "participant_id",
            "Category": "category",
            "ALSFRS-R_SpeechSubscore": (
                "speech_subscore"
            ),
        }
    )

    inventory = inventory.merge(
        participant_fields,
        on="participant_id",
        how="left",
        validate="many_to_one",
    )

    task_order = {
        task: index
        for index, task in enumerate(TASKS)
    }

    inventory["_task_order"] = (
        inventory["task"].map(
            task_order
        )
    )

    return (
        inventory.sort_values(
            [
                "participant_id",
                "_task_order",
            ]
        )
        .drop(
            columns="_task_order"
        )
        .reset_index(drop=True)
    )

def save_audio_inventory(
    inventory: pd.DataFrame,
) -> None:
    """Atomically save the participant-level inventory outside Git."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        RECORDING_INVENTORY_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        inventory.to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(
            RECORDING_INVENTORY_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def validate_audio_inventory(
    inventory: pd.DataFrame,
    participants: pd.DataFrame,
) -> None:
    """Validate file coverage, metadata linkage, and audio headers."""

    if len(inventory) != EXPECTED_TOTAL_RECORDINGS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RECORDINGS} recordings, "
            f"but found {len(inventory)}."
        )

    if inventory.duplicated(
        subset=[
            "participant_id",
            "task",
        ]
    ).any():
        raise ValueError(
            "A participant has duplicate recordings for the same task."
        )

    expected_participant_ids = set(
        participants["ID"]
    )

    observed_participant_ids = set(
        inventory["participant_id"]
    )

    missing_participants = (
        expected_participant_ids
        - observed_participant_ids
    )

    unknown_participants = (
        observed_participant_ids
        - expected_participant_ids
    )

    if missing_participants:
        raise ValueError(
            f"{len(missing_participants)} metadata participants "
            "have no recordings."
        )

    if unknown_participants:
        raise ValueError(
            f"{len(unknown_participants)} recording participants "
            "do not appear in metadata."
        )

    participant_counts = inventory[
        "participant_id"
    ].value_counts()

    if not participant_counts.eq(
        EXPECTED_RECORDINGS_PER_PARTICIPANT
    ).all():
        raise ValueError(
            "A participant does not have exactly eight recordings."
        )

    task_counts = (
        inventory["task"]
        .value_counts()
        .reindex(
            TASKS,
            fill_value=0,
        )
    )

    if not task_counts.eq(
        EXPECTED_RECORDINGS_PER_TASK
    ).all():
        raise ValueError(
            "A speech task does not contain exactly 153 recordings."
        )

    expected_directories = set(
        TASK_DIRECTORY_MAP
    )

    observed_directories = set(
        inventory["task_directory"]
    )

    if observed_directories != expected_directories:
        raise ValueError(
            "Observed task directories do not match the official "
            "VOC-ALS task directories."
        )

    if inventory["category"].isna().any():
        raise ValueError(
            "A recording could not be linked to a metadata category."
        )

    category_counts = inventory[
        "category"
    ].value_counts()

    if (
        category_counts.get("ALS", 0)
        != EXPECTED_ALS_RECORDINGS
    ):
        raise ValueError(
            "The inventory does not contain the expected "
            "816 ALS recordings."
        )

    if (
        category_counts.get("HC", 0)
        != EXPECTED_HEALTHY_RECORDINGS
    ):
        raise ValueError(
            "The inventory does not contain the expected "
            "408 healthy-control recordings."
        )

    if not inventory["readable"].all():
        unreadable_count = int(
            (~inventory["readable"]).sum()
        )

        raise ValueError(
            f"{unreadable_count} recordings have unreadable headers."
        )

    if inventory[
        "file_size_bytes"
    ].le(0).any():
        raise ValueError(
            "An audio file is empty."
        )

    numeric_header_columns = [
        "sample_rate_hz",
        "channels",
        "frames",
        "duration_seconds",
    ]

    if not np.isfinite(
        inventory[
            numeric_header_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "An audio header contains a non-finite numeric value."
        )

    positive_columns = [
        "sample_rate_hz",
        "channels",
        "frames",
        "duration_seconds",
    ]

    if inventory[
        positive_columns
    ].le(0).any().any():
        raise ValueError(
            "An audio recording has an invalid non-positive "
            "header value."
        )

def calculate_frame_rms(
    samples: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate frame RMS without constructing a large frame matrix."""

    frame_length = max(
        1,
        int(
            round(
                FRAME_DURATION_SECONDS
                * sample_rate
            )
        ),
    )

    hop_length = max(
        1,
        int(
            round(
                FRAME_HOP_SECONDS
                * sample_rate
            )
        ),
    )

    starts = np.arange(
        0,
        len(samples),
        hop_length,
        dtype=int,
    )

    ends = np.minimum(
        starts + frame_length,
        len(samples),
    )

    # Cumulative energy gives each frame's sum of squares without copying
    # thousands of overlapping waveform windows into memory.
    cumulative_energy = np.concatenate(
        [
            np.zeros(
                1,
                dtype=float,
            ),
            np.cumsum(
                np.square(
                    samples,
                    dtype=float,
                )
            ),
        ]
    )

    frame_energy = (
        cumulative_energy[ends]
        - cumulative_energy[starts]
    )

    frame_lengths = (
        ends - starts
    )

    frame_rms = np.sqrt(
        frame_energy / frame_lengths
    )

    return frame_rms, starts, ends


def inspect_waveform(
    inventory_row,
    audio_directory: Path,
) -> dict[str, object]:
    """Decode one complete waveform and calculate quality indicators."""

    path = (
        audio_directory
        / inventory_row.relative_path
    )

    record: dict[str, object] = {
        "relative_path": (
            inventory_row.relative_path
        ),
        "participant_id": (
            inventory_row.participant_id
        ),
        "task": inventory_row.task,
        "category": inventory_row.category,
        "decoded": False,
        "decoded_sample_rate_hz": np.nan,
        "decoded_samples": np.nan,
        "finite_samples": False,
        "all_zero": False,
        "peak_absolute_amplitude": np.nan,
        "rms_amplitude": np.nan,
        "rms_dbfs": np.nan,
        "absolute_dc_offset": np.nan,
        "zero_sample_fraction": np.nan,
        "silent_frame_fraction": np.nan,
        "leading_silence_seconds": np.nan,
        "trailing_silence_seconds": np.nan,
        "clipped_sample_count": np.nan,
        "clipped_sample_fraction": np.nan,
        "any_clipping": False,
        "material_clipping": False,
        "near_silent_recording": False,
        "signal_sha256": None,
        "waveform_error": None,
    }

    try:
        integer_samples, sample_rate = sf.read(
            path,
            dtype="int16",
            always_2d=False,
        )
    except (OSError, RuntimeError) as error:
        record["waveform_error"] = str(
            error
        )

        return record

    if integer_samples.ndim != 1:
        record["waveform_error"] = (
            "Decoded waveform is not mono."
        )

        return record

    if len(integer_samples) == 0:
        record["waveform_error"] = (
            "Decoded waveform contains no samples."
        )

        return record

    # Dividing by 32768 converts signed PCM-16 samples into the conventional
    # floating-point range while retaining the exact original sample values.
    samples = (
        integer_samples.astype(
            np.float64
        )
        / 32768.0
    )

    finite_samples = bool(
        np.isfinite(samples).all()
    )

    peak_amplitude = float(
        np.max(
            np.abs(samples)
        )
    )

    rms_amplitude = float(
        np.sqrt(
            np.mean(
                np.square(samples)
            )
        )
    )

    all_zero = bool(
        peak_amplitude == 0.0
    )

    rms_dbfs = (
        float(
            20.0
            * np.log10(
                rms_amplitude
            )
        )
        if rms_amplitude > 0.0
        else float("-inf")
    )

    frame_rms, frame_starts, frame_ends = (
        calculate_frame_rms(
            samples=samples,
            sample_rate=int(sample_rate),
        )
    )

    silence_amplitude = float(
        10.0 ** (
            SILENCE_THRESHOLD_DBFS
            / 20.0
        )
    )

    silent_frames = (
        frame_rms < silence_amplitude
    )

    active_frame_indices = np.flatnonzero(
        ~silent_frames
    )

    if len(active_frame_indices) == 0:
        leading_silence_seconds = (
            len(samples) / sample_rate
        )

        trailing_silence_seconds = (
            len(samples) / sample_rate
        )
    else:
        first_active = int(
            active_frame_indices[0]
        )

        last_active = int(
            active_frame_indices[-1]
        )

        leading_silence_seconds = float(
            frame_starts[first_active]
            / sample_rate
        )

        trailing_silence_seconds = float(
            (
                len(samples)
                - frame_ends[last_active]
            )
            / sample_rate
        )

    clipped_mask = (
        (integer_samples == -32768)
        | (integer_samples == 32767)
    )

    clipped_sample_count = int(
        clipped_mask.sum()
    )

    clipped_sample_fraction = float(
        clipped_sample_count
        / len(integer_samples)
    )

    # Hashing decoded PCM values detects recordings with identical signals
    # even if their WAV container metadata differs.
    signal_sha256 = hashlib.sha256(
        integer_samples.tobytes()
    ).hexdigest()

    record.update(
        {
            "decoded": True,
            "decoded_sample_rate_hz": int(
                sample_rate
            ),
            "decoded_samples": int(
                len(integer_samples)
            ),
            "finite_samples": finite_samples,
            "all_zero": all_zero,
            "peak_absolute_amplitude": (
                peak_amplitude
            ),
            "rms_amplitude": rms_amplitude,
            "rms_dbfs": rms_dbfs,
            "absolute_dc_offset": float(
                abs(
                    np.mean(samples)
                )
            ),
            "zero_sample_fraction": float(
                np.mean(
                    integer_samples == 0
                )
            ),
            "silent_frame_fraction": float(
                np.mean(
                    silent_frames
                )
            ),
            "leading_silence_seconds": float(
                leading_silence_seconds
            ),
            "trailing_silence_seconds": float(
                trailing_silence_seconds
            ),
            "clipped_sample_count": (
                clipped_sample_count
            ),
            "clipped_sample_fraction": (
                clipped_sample_fraction
            ),
            "any_clipping": (
                clipped_sample_count > 0
            ),
            "material_clipping": (
                clipped_sample_fraction
                >= MATERIAL_CLIPPING_FRACTION
            ),
            "near_silent_recording": (
                rms_dbfs
                <= SILENCE_THRESHOLD_DBFS
            ),
            "signal_sha256": signal_sha256,
        }
    )

    return record


def build_waveform_audit(
    inventory: pd.DataFrame,
    audio_directory: Path,
) -> pd.DataFrame:
    """Decode every recording and build local quality-control results."""

    waveform_records: list[
        dict[str, object]
    ] = []

    total_recordings = len(
        inventory
    )

    for recording_index, inventory_row in enumerate(
        inventory.itertuples(
            index=False
        ),
        start=1,
    ):
        if (
            recording_index == 1
            or recording_index % 100 == 0
            or recording_index
            == total_recordings
        ):
            print(
                "Decoding recording "
                f"{recording_index}/{total_recordings}..."
            )

        waveform_records.append(
            inspect_waveform(
                inventory_row=inventory_row,
                audio_directory=audio_directory,
            )
        )

    waveform_audit = pd.DataFrame(
        waveform_records
    )

    header_fields = inventory[
        [
            "relative_path",
            "sample_rate_hz",
            "frames",
            "duration_seconds",
            "channels",
            "format",
            "subtype",
        ]
    ]

    return waveform_audit.merge(
        header_fields,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )


def save_waveform_audit(
    waveform_audit: pd.DataFrame,
) -> None:
    """Atomically save participant-level waveform quality results."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        WAVEFORM_AUDIT_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        waveform_audit.to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(
            WAVEFORM_AUDIT_PATH
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_waveform_audit(
    waveform_audit: pd.DataFrame,
) -> None:
    """Validate complete decoding and agreement with WAV headers."""

    if len(
        waveform_audit
    ) != EXPECTED_TOTAL_RECORDINGS:
        raise ValueError(
            "Waveform audit does not contain all 1,224 recordings."
        )

    if waveform_audit[
        "relative_path"
    ].duplicated().any():
        raise ValueError(
            "A recording appears more than once in the waveform audit."
        )

    if not waveform_audit[
        "decoded"
    ].all():
        failed_count = int(
            (
                ~waveform_audit["decoded"]
            ).sum()
        )

        raise ValueError(
            f"{failed_count} recordings could not be decoded."
        )

    if not waveform_audit[
        "finite_samples"
    ].all():
        raise ValueError(
            "A decoded recording contains non-finite samples."
        )

    if waveform_audit[
        "all_zero"
    ].any():
        raise ValueError(
            "An all-zero recording was found."
        )

    if not np.array_equal(
        waveform_audit[
            "decoded_sample_rate_hz"
        ].to_numpy(dtype=int),
        waveform_audit[
            "sample_rate_hz"
        ].to_numpy(dtype=int),
    ):
        raise ValueError(
            "A decoded sample rate does not match its WAV header."
        )

    if not np.array_equal(
        waveform_audit[
            "decoded_samples"
        ].to_numpy(dtype=int),
        waveform_audit[
            "frames"
        ].to_numpy(dtype=int),
    ):
        raise ValueError(
            "A decoded sample count does not match its WAV header."
        )

    finite_numeric_columns = [
        "peak_absolute_amplitude",
        "rms_amplitude",
        "rms_dbfs",
        "absolute_dc_offset",
        "zero_sample_fraction",
        "silent_frame_fraction",
        "leading_silence_seconds",
        "trailing_silence_seconds",
        "clipped_sample_count",
        "clipped_sample_fraction",
    ]

    if not np.isfinite(
        waveform_audit[
            finite_numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "A waveform quality metric is non-finite."
        )

    fraction_columns = [
        "zero_sample_fraction",
        "silent_frame_fraction",
        "clipped_sample_fraction",
    ]

    if not waveform_audit[
        fraction_columns
    ].apply(
        lambda column: column.between(
            0.0,
            1.0,
        )
    ).all().all():
        raise ValueError(
            "A waveform fraction is outside zero to one."
        )

    if not waveform_audit[
        "peak_absolute_amplitude"
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "A waveform peak amplitude is outside zero to one."
        )

    if waveform_audit[
        "rms_amplitude"
    ].le(0.0).any():
        raise ValueError(
            "A waveform has non-positive RMS amplitude."
        )

    if waveform_audit[
        "signal_sha256"
    ].isna().any():
        raise ValueError(
            "A decoded waveform is missing its signal hash."
        )

def print_waveform_summary(
    waveform_audit: pd.DataFrame,
) -> None:
    """Print aggregate signal-quality findings without participant IDs."""

    hash_counts = waveform_audit[
        "signal_sha256"
    ].value_counts()

    duplicate_hash_counts = hash_counts.loc[hash_counts.gt(1)]

    duplicate_groups = len(duplicate_hash_counts)

    duplicate_files = int(duplicate_hash_counts.sum())

    print()
    print("Raw-audio waveform audit validated.")
    print(f"Decoded recordings: {len(waveform_audit)}")
    print(
        "All-zero recordings: "
        f"{waveform_audit['all_zero'].sum()}"
    )
    print(
        "Near-silent recordings: "
        f"{waveform_audit['near_silent_recording'].sum()}"
    )
    print(
        "Recordings with any endpoint clipping: "
        f"{waveform_audit['any_clipping'].sum()}"
    )
    print(
        "Recordings with at least 0.1% clipped samples: "
        f"{waveform_audit['material_clipping'].sum()}"
    )
    print(f"Exact duplicate signal groups: {duplicate_groups}")
    print(f"Files belonging to duplicate groups: {duplicate_files}")

    task_summary = (
        waveform_audit.groupby(
            "task"
        )
        .agg(
            recordings=(
                "relative_path",
                "size",
            ),
            minimum_rms_dbfs=(
                "rms_dbfs",
                "min",
            ),
            median_rms_dbfs=(
                "rms_dbfs",
                "median",
            ),
            maximum_rms_dbfs=(
                "rms_dbfs",
                "max",
            ),
            median_silent_frame_fraction=(
                "silent_frame_fraction",
                "median",
            ),
            maximum_silent_frame_fraction=(
                "silent_frame_fraction",
                "max",
            ),
            files_with_any_clipping=(
                "any_clipping",
                "sum",
            ),
            files_with_material_clipping=(
                "material_clipping",
                "sum",
            ),
        )
        .reindex(TASKS)
        .reset_index()
    )

    print()
    print("Waveform quality by task:")
    print(
        task_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Generated waveform audit:")
    print(WAVEFORM_AUDIT_PATH)

def print_flag_diagnostics(
    waveform_audit: pd.DataFrame,
) -> None:
    """Describe quality flags and duplicate structure without showing IDs."""

    diagnostic_table = (
        waveform_audit.assign(
            high_silence_fraction=(
                waveform_audit[
                    "silent_frame_fraction"
                ]
                >= HIGH_SILENCE_FRAME_FRACTION
            )
        )
        .groupby("task")
        .agg(
            near_silent_recordings=(
                "near_silent_recording",
                "sum",
            ),
            high_silence_recordings=(
                "high_silence_fraction",
                "sum",
            ),
            files_with_any_clipping=(
                "any_clipping",
                "sum",
            ),
            files_with_material_clipping=(
                "material_clipping",
                "sum",
            ),
        )
        .reindex(TASKS)
        .reset_index()
    )

    print()
    print("Quality flags by task:")
    print(
        diagnostic_table.to_string(
            index=False
        )
    )

    near_silent = waveform_audit.loc[
        waveform_audit[
            "near_silent_recording"
        ],
        [
            "task",
            "category",
            "duration_seconds",
            "rms_dbfs",
            "silent_frame_fraction",
            "leading_silence_seconds",
            "trailing_silence_seconds",
        ],
    ]

    print()
    print("Near-silent recording summary:")
    if near_silent.empty:
        print("None")
    else:
        print(
            near_silent.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

    material_clipping = waveform_audit.loc[
        waveform_audit[
            "material_clipping"
        ],
        [
            "task",
            "category",
            "duration_seconds",
            "rms_dbfs",
            "clipped_sample_count",
            "clipped_sample_fraction",
        ],
    ]

    print()
    print("Material-clipping recording summary:")
    if material_clipping.empty:
        print("None")
    else:
        print(
            material_clipping.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    duplicate_files = waveform_audit.loc[
        waveform_audit.duplicated(
            subset="signal_sha256",
            keep=False,
        )
    ].copy()

    print()
    print("Exact duplicate structure:")

    if duplicate_files.empty:
        print("No exact duplicate signals were found.")
        return

    duplicate_groups = (
        duplicate_files.groupby(
            "signal_sha256",
            sort=True,
        )
        .agg(
            file_count=(
                "relative_path",
                "size",
            ),
            participant_count=(
                "participant_id",
                "nunique",
            ),
            task_count=(
                "task",
                "nunique",
            ),
            category_count=(
                "category",
                "nunique",
            ),
            duration_seconds=(
                "duration_seconds",
                "first",
            ),
            rms_dbfs=(
                "rms_dbfs",
                "first",
            ),
        )
        .reset_index(drop=True)
    )

    task_membership = (
        duplicate_files.groupby(
            "signal_sha256",
            sort=True,
        )["task"]
        .agg(
            lambda values: "+".join(
                sorted(
                    set(values),
                    key=lambda task: (
                        TASKS.index(task)
                    ),
                )
            )
        )
        .reset_index(
            name="tasks"
        )
    )

    category_membership = (
        duplicate_files.groupby(
            "signal_sha256",
            sort=True,
        )["category"]
        .agg(
            lambda values: "+".join(
                sorted(
                    set(values)
                )
            )
        )
        .reset_index(
            name="categories"
        )
    )

    duplicate_groups["tasks"] = (
        task_membership["tasks"]
    )

    duplicate_groups["categories"] = (
        category_membership[
            "categories"
        ]
    )

    duplicate_groups.insert(
        0,
        "duplicate_group",
        np.arange(
            1,
            len(duplicate_groups) + 1,
        ),
    )

    duplicate_groups[
        "cross_participant"
    ] = duplicate_groups[
        "participant_count"
    ].gt(1)

    duplicate_groups[
        "cross_task"
    ] = duplicate_groups[
        "task_count"
    ].gt(1)

    print(
        duplicate_groups[
            [
                "duplicate_group",
                "file_count",
                "participant_count",
                "task_count",
                "tasks",
                "category_count",
                "categories",
                "cross_participant",
                "cross_task",
                "duration_seconds",
                "rms_dbfs",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Duplicate groups crossing participant boundaries: "
        f"{duplicate_groups['cross_participant'].sum()}"
    )
    print(
        "Duplicate groups crossing task boundaries: "
        f"{duplicate_groups['cross_task'].sum()}"
    )
    print(
        "Duplicate groups containing multiple categories: "
        f"{duplicate_groups['category_count'].gt(1).sum()}"
    )

def print_inventory_summary(
    inventory: pd.DataFrame,
) -> None:
    """Print aggregate properties without exposing participant identities."""

    print("Raw-audio header audit validated.")
    print(f"Recordings: {len(inventory)}")
    print(
        "Participants: "
        f"{inventory['participant_id'].nunique()}"
    )
    print(
        "ALS recordings: "
        f"{inventory['category'].eq('ALS').sum()}"
    )
    print(
        "Healthy-control recordings: "
        f"{inventory['category'].eq('HC').sum()}"
    )

    print()
    print("Recordings by task:")
    print(
        inventory["task"]
        .value_counts()
        .reindex(TASKS)
        .rename_axis("task")
        .reset_index(name="recordings")
        .to_string(index=False)
    )

    print()
    print("Sample-rate distribution:")
    print(
        inventory[
            "sample_rate_hz"
        ]
        .value_counts()
        .sort_index()
        .rename_axis("sample_rate_hz")
        .reset_index(name="recordings")
        .to_string(index=False)
    )

    print()
    print("Channel-count distribution:")
    print(
        inventory[
            "channels"
        ]
        .value_counts()
        .sort_index()
        .rename_axis("channels")
        .reset_index(name="recordings")
        .to_string(index=False)
    )

    print()
    print("Audio format and subtype:")
    print(
        inventory.groupby(
            [
                "format",
                "subtype",
                "endian",
            ]
        )
        .size()
        .reset_index(name="recordings")
        .to_string(index=False)
    )

    duration_summary = (
        inventory.groupby(
            "task"
        )["duration_seconds"]
        .agg(
            recordings="size",
            minimum_seconds="min",
            median_seconds="median",
            maximum_seconds="max",
        )
        .reindex(TASKS)
        .reset_index()
    )

    print()
    print("Duration by task:")
    print(
        duration_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("Generated inventory:")
    print(RECORDING_INVENTORY_PATH)

def summarize_duplicate_participant_topology(
    waveform_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe how exact duplicate signals connect participants across tasks."""

    required_columns = {
        "participant_id",
        "task",
        "category",
        "signal_sha256",
    }
    missing_columns = required_columns.difference(
        waveform_audit.columns
    )

    if missing_columns:
        raise ValueError(
            "Waveform audit is missing columns required for duplicate "
            f"topology analysis: {sorted(missing_columns)}"
        )

    audit = waveform_audit.copy()
    audit["participant_id"] = (
        audit["participant_id"].astype(str)
    )

    # Keep every occurrence of a signal hash that appears more than once.
    duplicate_rows = audit.loc[
        audit.duplicated(
            subset="signal_sha256",
            keep=False,
        )
    ].copy()

    if duplicate_rows.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    task_order = {
        task: index
        for index, task in enumerate(TASKS)
    }

    participant_category_counts = (
        audit.groupby("participant_id")["category"]
        .nunique()
    )

    if participant_category_counts.max() != 1:
        raise ValueError(
            "At least one participant has inconsistent category labels."
        )

    participant_categories = (
        audit[
            [
                "participant_id",
                "category",
            ]
        ]
        .drop_duplicates()
        .set_index("participant_id")["category"]
    )

    # First count how many tasks are shared by each duplicate-connected
    # participant pair. Participant identifiers stay internal and are not
    # included in the returned diagnostic table.
    pair_records: list[dict[str, object]] = []

    for _, group in duplicate_rows.groupby(
        "signal_sha256",
        sort=True,
    ):
        participants = sorted(
            group["participant_id"].unique()
        )
        duplicate_tasks = sorted(
            group["task"].unique(),
            key=task_order.get,
        )

        if len(duplicate_tasks) != 1:
            raise ValueError(
                "An exact signal hash appeared under more than one task."
            )

        task = duplicate_tasks[0]

        for left_index, left_participant in enumerate(
            participants
        ):
            for right_participant in participants[
                left_index + 1:
            ]:
                pair_records.append(
                    {
                        "participant_a": left_participant,
                        "participant_b": right_participant,
                        "task": task,
                        "category_pair": "+".join(
                            sorted(
                                {
                                    participant_categories.loc[
                                        left_participant
                                    ],
                                    participant_categories.loc[
                                        right_participant
                                    ],
                                }
                            )
                        ),
                    }
                )

    pair_details = pd.DataFrame(pair_records)

    pair_summary_internal = (
        pair_details.groupby(
            [
                "participant_a",
                "participant_b",
                "category_pair",
            ],
            as_index=False,
        )
        .agg(
            shared_task_count=(
                "task",
                "nunique",
            ),
            shared_tasks=(
                "task",
                lambda values: "+".join(
                    sorted(
                        set(values),
                        key=task_order.get,
                    )
                ),
            ),
        )
        .sort_values(
            [
                "shared_task_count",
                "shared_tasks",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    pair_summary = pair_summary_internal[
        [
            "category_pair",
            "shared_task_count",
            "shared_tasks",
        ]
    ].copy()
    pair_summary.insert(
        0,
        "duplicate_pair_group",
        range(
            1,
            len(pair_summary) + 1,
        ),
    )

    # Build connected components. A component contains participants joined
    # through one or more exact duplicate recordings.
    parent: dict[str, str] = {}

    def find_root(participant_id: str) -> str:
        parent.setdefault(
            participant_id,
            participant_id,
        )

        while parent[participant_id] != participant_id:
            parent[participant_id] = parent[
                parent[participant_id]
            ]
            participant_id = parent[participant_id]

        return participant_id

    def join_participants(
        left_participant: str,
        right_participant: str,
    ) -> None:
        left_root = find_root(left_participant)
        right_root = find_root(right_participant)

        if left_root != right_root:
            parent[right_root] = left_root

    for _, group in duplicate_rows.groupby(
        "signal_sha256",
        sort=True,
    ):
        participants = sorted(
            group["participant_id"].unique()
        )

        for participant in participants:
            find_root(participant)

        for participant in participants[1:]:
            join_participants(
                participants[0],
                participant,
            )

    component_members: dict[str, list[str]] = {}

    for participant in parent:
        root = find_root(participant)
        component_members.setdefault(
            root,
            [],
        ).append(participant)

    # A full profile is duplicated only when all eight task recordings are
    # identical between two or more participants.
    participant_profiles = (
        audit.pivot(
            index="participant_id",
            columns="task",
            values="signal_sha256",
        )
        .reindex(columns=TASKS)
    )

    if participant_profiles.isna().any().any():
        raise ValueError(
            "Cannot compare complete participant profiles because at least "
            "one participant is missing a task hash."
        )

    profile_signatures = participant_profiles.apply(
        lambda row: hashlib.sha256(
            "|".join(
                row.astype(str)
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    profile_frame = profile_signatures.rename(
        "profile_signature"
    ).reset_index()

    duplicate_profile_rows = profile_frame.loc[
        profile_frame.duplicated(
            subset="profile_signature",
            keep=False,
        )
    ]

    full_profile_records: list[dict[str, object]] = []
    full_profile_member_sets: list[set[str]] = []

    for profile_group, (_, group) in enumerate(
        duplicate_profile_rows.groupby(
            "profile_signature",
            sort=True,
        ),
        start=1,
    ):
        participants = sorted(
            group["participant_id"].astype(str)
        )
        member_set = set(participants)
        full_profile_member_sets.append(member_set)

        category_counts = (
            participant_categories.loc[participants]
            .value_counts()
            .sort_index()
        )

        full_profile_records.append(
            {
                "full_profile_group": profile_group,
                "participant_count": len(participants),
                "categories": "+".join(
                    f"{category}:{count}"
                    for category, count
                    in category_counts.items()
                ),
                "identical_task_count": len(TASKS),
            }
        )

    full_profile_summary = pd.DataFrame(
        full_profile_records
    )

    component_records: list[dict[str, object]] = []

    ordered_components = sorted(
        component_members.values(),
        key=lambda members: (
            -len(members),
            sorted(members),
        ),
    )

    for component_number, members in enumerate(
        ordered_components,
        start=1,
    ):
        member_set = set(members)
        component_rows = duplicate_rows.loc[
            duplicate_rows["participant_id"].isin(
                member_set
            )
        ]

        component_pairs = pair_summary_internal.loc[
            pair_summary_internal["participant_a"].isin(
                member_set
            )
            & pair_summary_internal["participant_b"].isin(
                member_set
            )
        ]

        category_counts = (
            participant_categories.loc[members]
            .value_counts()
            .sort_index()
        )

        component_tasks = sorted(
            component_rows["task"].unique(),
            key=task_order.get,
        )

        component_records.append(
            {
                "duplicate_component": component_number,
                "participant_count": len(member_set),
                "categories": "+".join(
                    f"{category}:{count}"
                    for category, count
                    in category_counts.items()
                ),
                "duplicate_signal_groups": (
                    component_rows[
                        "signal_sha256"
                    ].nunique()
                ),
                "duplicate_file_rows": len(
                    component_rows
                ),
                "duplicate_task_count": len(
                    component_tasks
                ),
                "duplicate_tasks": "+".join(
                    component_tasks
                ),
                "maximum_shared_tasks_for_one_pair": int(
                    component_pairs[
                        "shared_task_count"
                    ].max()
                ),
                "contains_full_duplicate_profile": any(
                    profile_members.issubset(
                        member_set
                    )
                    for profile_members
                    in full_profile_member_sets
                ),
            }
        )

    component_summary = pd.DataFrame(
        component_records
    )

    return (
        component_summary,
        pair_summary,
        full_profile_summary,
    )

def main() -> None:
    """Run the raw-audio header and coverage audit."""

    participants = load_participant_metadata(METADATA_PATH)

    inventory = build_audio_inventory(
        audio_directory=RAW_AUDIO_DIR,
        participants=participants,
    )

    # Save the generated participant-level inventory before aggregate
    # validation so any failure can be diagnosed locally.
    save_audio_inventory(inventory)

    validate_audio_inventory(
        inventory=inventory,
        participants=participants,
    )

    print_inventory_summary(inventory)

    print()

    waveform_audit = build_waveform_audit(
        inventory=inventory,
        audio_directory=RAW_AUDIO_DIR,
    )

    # This detailed artifact contains participant IDs and therefore stays
    # inside the ignored generated-results directory.
    save_waveform_audit(waveform_audit)
    validate_waveform_audit(waveform_audit)
    print_waveform_summary(waveform_audit)
    print_flag_diagnostics(waveform_audit)

    (
        duplicate_components,
        duplicate_pairs,
        duplicate_profiles,
    ) = summarize_duplicate_participant_topology(
        waveform_audit
    )

    print("\nDuplicate participant components:")
    if duplicate_components.empty:
        print("None")
    else:
        print(
            duplicate_components.to_string(
                index=False
            )
        )

    print("\nRepeated duplicate participant pairs:")
    if duplicate_pairs.empty:
        print("None")
    else:
        print(
            duplicate_pairs.to_string(
                index=False
            )
        )

    print("\nFully duplicated eight-task participant profiles:")
    if duplicate_profiles.empty:
        print("None")
    else:
        print(
            duplicate_profiles.to_string(
                index=False
            )
        )

if __name__ == "__main__":
    main()
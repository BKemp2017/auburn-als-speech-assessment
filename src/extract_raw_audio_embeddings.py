"""Extract frozen WavLM embeddings from duplicate-filtered VOC-ALS audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import transformers
from huggingface_hub import HfApi
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

from audit_duplicate_impact import normalize_participant_id
from audit_metadata import TASKS
from audit_raw_audio import (
    METADATA_PATH,
    RAW_AUDIO_DIR,
    build_audio_inventory,
    load_participant_metadata,
    validate_audio_inventory,
)
from evaluate_baselines import load_experiment_inputs
from evaluate_duplicate_exclusion_sensitivity import (
    WAVEFORM_AUDIT_PATH,
    identify_duplicate_connected_als_ids,
)


MODEL_NAME = "microsoft/wavlm-base-plus"
SOURCE_SAMPLE_RATE = 8_000
TARGET_SAMPLE_RATE = 16_000
EXPECTED_HIDDEN_SIZE = 768

EXPECTED_PARTICIPANTS = 100
EXPECTED_RECORDINGS = EXPECTED_PARTICIPANTS * len(TASKS)
EXPECTED_TARGET_COUNTS = {0: 52, 1: 48}

OUTPUT_DIRECTORY = Path("results/generated")
EMBEDDINGS_PATH = (
    OUTPUT_DIRECTORY / "raw_audio_wavlm_embeddings.npz"
)
EXPERIMENT_RECORD_PATH = Path(
    "experiments/raw_audio_wavlm_embeddings.json"
)

def parse_arguments() -> argparse.Namespace:
    """Read the optional validation and smoke-test modes."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen WavLM embeddings from the conservative "
            "duplicate-filtered ALS cohort."
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate recording coverage without loading WavLM.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Extract only the longest recording without saving outputs.",
    )

    return parser.parse_args()

def load_extraction_inventory() -> tuple[pd.DataFrame, set[str]]:
    """Build the ALS recording inventory and apply the locked exclusion."""

    participants = load_participant_metadata(
        METADATA_PATH
    )
    inventory = build_audio_inventory(
        audio_directory=RAW_AUDIO_DIR,
        participants=participants,
    )

    # Validate the complete public dataset before filtering it. This keeps a
    # missing or misplaced recording from being hidden by the ALS-only slice.
    validate_audio_inventory(
        inventory=inventory,
        participants=participants,
    )

    excluded_ids = identify_duplicate_connected_als_ids(
        WAVEFORM_AUDIT_PATH
    )

    features, _, _ = load_experiment_inputs()
    target_table = features[
        ["ID", "target"]
    ].copy()
    target_table["participant_id"] = target_table[
        "ID"
    ].map(normalize_participant_id)
    target_table = target_table.drop(
        columns="ID"
    )

    als_inventory = inventory.loc[
        inventory["category"].eq("ALS")
    ].copy()
    als_inventory["participant_id"] = als_inventory[
        "participant_id"
    ].map(normalize_participant_id)

    # Neural encoders can memorize exact waveforms. Remove both members of
    # the contradictory ALS duplicate pair before loading any audio signal.
    extraction_inventory = als_inventory.loc[
        ~als_inventory["participant_id"].isin(
            excluded_ids
        )
    ].copy()

    extraction_inventory = extraction_inventory.merge(
        target_table,
        on="participant_id",
        how="left",
        validate="many_to_one",
    )

    task_order = {
        task: index
        for index, task in enumerate(TASKS)
    }
    extraction_inventory["_task_order"] = (
        extraction_inventory["task"].map(
            task_order
        )
    )

    extraction_inventory = (
        extraction_inventory.sort_values(
            ["participant_id", "_task_order"]
        )
        .drop(columns="_task_order")
        .reset_index(drop=True)
    )

    return extraction_inventory, excluded_ids

def validate_extraction_inventory(
    inventory: pd.DataFrame,
    excluded_ids: set[str],
) -> None:
    """Confirm the encoder will receive the intended 100-person cohort."""

    required_columns = {
        "relative_path",
        "participant_id",
        "task",
        "category",
        "target",
        "sample_rate_hz",
        "channels",
        "duration_seconds",
    }
    missing_columns = required_columns.difference(
        inventory.columns
    )

    if missing_columns:
        raise ValueError(
            "Extraction inventory is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(excluded_ids) != 2:
        raise ValueError(
            "Expected exactly two duplicate-connected ALS participants, "
            f"but found {len(excluded_ids)}."
        )

    if len(inventory) != EXPECTED_RECORDINGS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDINGS} recordings, "
            f"but found {len(inventory)}."
        )

    if inventory["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} participants after "
            "duplicate exclusion."
        )

    if inventory.duplicated(
        subset=["participant_id", "task"]
    ).any():
        raise ValueError(
            "A participant has more than one recording for a task."
        )

    recordings_per_participant = inventory.groupby(
        "participant_id"
    ).size()

    if not recordings_per_participant.eq(len(TASKS)).all():
        raise ValueError(
            "Every retained participant must have all eight tasks."
        )

    task_counts = (
        inventory["task"]
        .value_counts()
        .reindex(TASKS, fill_value=0)
    )

    if not task_counts.eq(EXPECTED_PARTICIPANTS).all():
        raise ValueError(
            "Every task must contain one recording for each retained "
            "participant."
        )

    if not inventory["category"].eq("ALS").all():
        raise ValueError(
            "The extraction inventory contains a healthy control."
        )

    if inventory["target"].isna().any():
        raise ValueError(
            "A retained recording is missing its frozen severity target."
        )

    target_counts = {
        int(target): int(count)
        for target, count in (
            inventory.drop_duplicates("participant_id")["target"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    if target_counts != EXPECTED_TARGET_COUNTS:
        raise ValueError(
            "Filtered target counts do not match the locked sensitivity "
            f"cohort: {target_counts}."
        )

    if not inventory["sample_rate_hz"].eq(
        SOURCE_SAMPLE_RATE
    ).all():
        raise ValueError(
            "A recording does not use the expected 8 kHz source rate."
        )

    if not inventory["channels"].eq(1).all():
        raise ValueError(
            "A recording is not mono."
        )

def choose_device() -> torch.device:
    """Use the NVIDIA GPU when PyTorch can access it."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def resolve_model_revision() -> str:
    """Resolve the model repository to an immutable commit hash."""

    model_information = HfApi().model_info(
        MODEL_NAME,
        revision="main",
    )

    if not model_information.sha:
        raise ValueError(
            "Hugging Face did not return a model commit hash."
        )

    return str(model_information.sha)

def load_encoder(
    device: torch.device,
) -> tuple[object, torch.nn.Module, str]:
    """Load one frozen WavLM checkpoint for deterministic inference."""

    revision = resolve_model_revision()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME,
        revision=revision,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=revision,
    )

    if int(feature_extractor.sampling_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(
            "The WavLM feature extractor does not expect 16 kHz audio."
        )

    if int(model.config.hidden_size) != EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            "Unexpected WavLM hidden size: "
            f"{model.config.hidden_size}."
        )

    model.requires_grad_(False)
    model.eval()
    model.to(device)

    return feature_extractor, model, revision

def load_resampled_waveform(path: Path) -> np.ndarray:
    """Decode one mono WAV file and deterministically resample to 16 kHz."""

    waveform, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected mono audio at {path}."
        )

    if sample_rate != SOURCE_SAMPLE_RATE:
        raise ValueError(
            f"Expected {SOURCE_SAMPLE_RATE} Hz audio at {path}, "
            f"but found {sample_rate} Hz."
        )

    if waveform.size == 0:
        raise ValueError(
            f"Decoded an empty recording at {path}."
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            f"Decoded a non-finite audio sample at {path}."
        )

    resampled = resample_poly(
        waveform,
        up=TARGET_SAMPLE_RATE,
        down=sample_rate,
    ).astype(np.float32, copy=False)

    if not np.isfinite(resampled).all():
        raise ValueError(
            f"Resampling produced a non-finite value at {path}."
        )

    return resampled

def extract_recording_embedding(
    waveform: np.ndarray,
    feature_extractor,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Pool WavLM frame representations into one recording vector."""

    model_inputs = feature_extractor(
        waveform,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = model_inputs["input_values"].to(
        device
    )

    # Each recording is processed alone, so there is no padding to mask.
    # Mean pooling captures its central representation while standard
    # deviation pooling preserves how the representation varies over time.
    with torch.inference_mode():
        hidden_states = model(
            input_values=input_values
        ).last_hidden_state.squeeze(0)

    pooled = torch.cat(
        (
            hidden_states.mean(dim=0),
            hidden_states.std(dim=0, unbiased=False),
        ),
        dim=0,
    )

    return (
        pooled.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )

def extract_embeddings(
    inventory: pd.DataFrame,
    feature_extractor,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Extract one pooled embedding for every retained recording."""

    embeddings: list[np.ndarray] = []
    total_recordings = len(inventory)

    for position, row in enumerate(
        inventory.itertuples(index=False),
        start=1,
    ):
        if position == 1 or position % 25 == 0:
            print(
                f"Extracting recording {position}/{total_recordings}..."
            )

        waveform = load_resampled_waveform(
            RAW_AUDIO_DIR / row.relative_path
        )
        embedding = extract_recording_embedding(
            waveform=waveform,
            feature_extractor=feature_extractor,
            model=model,
            device=device,
        )
        embeddings.append(embedding)

    return np.stack(
        embeddings
    ).astype(np.float32, copy=False)

def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    """Reject incomplete, non-finite, or degenerate neural features."""

    expected_shape = (
        expected_rows,
        EXPECTED_HIDDEN_SIZE * 2,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected embedding shape {expected_shape}, "
            f"but found {embeddings.shape}."
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, found {embeddings.dtype}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "The extracted embedding matrix contains a non-finite value."
        )

    if np.any(np.linalg.norm(embeddings, axis=1) == 0.0):
        raise ValueError(
            "At least one recording produced an all-zero embedding."
        )

def save_embeddings(
    inventory: pd.DataFrame,
    embeddings: np.ndarray,
    revision: str,
    device: torch.device,
) -> None:
    """Atomically save private embeddings and reproducibility metadata."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The tracked record contains only aggregate extraction details and no
    # participant identifiers, recording paths, or embedding values.
    EXPERIMENT_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_embeddings_path = Path(
        f"{EMBEDDINGS_PATH}.tmp"
    )
    temporary_metadata_path = Path(
        f"{EXPERIMENT_RECORD_PATH}.tmp"
    )

    metadata = {
        "experiment": "duplicate_filtered_wavlm_embedding_extraction",
        "participant_level_fields": [],
        "model_name": MODEL_NAME,
        "model_revision": revision,
        "source_sample_rate_hz": SOURCE_SAMPLE_RATE,
        "encoder_sample_rate_hz": TARGET_SAMPLE_RATE,
        "pooling": "last_hidden_state_mean_and_standard_deviation",
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "embedding_dimension": int(embeddings.shape[1]),
        "recordings": int(len(inventory)),
        "participants": int(inventory["participant_id"].nunique()),
        "tasks": list(TASKS),
        "target_counts": {
            str(target): int(count)
            for target, count in (
                inventory.drop_duplicates("participant_id")["target"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "duplicate_policy": (
            "exclude_both_duplicate_connected_als_participants"
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "extraction_device": str(device),
    }

    try:
        # Unicode arrays keep loading safe with allow_pickle=False.
        with temporary_embeddings_path.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                embeddings=embeddings,
                participant_ids=inventory[
                    "participant_id"
                ].astype(str).to_numpy(dtype=str),
                tasks=inventory["task"].astype(str).to_numpy(dtype=str),
                targets=inventory["target"].to_numpy(dtype=np.int64),
                relative_paths=inventory[
                    "relative_path"
                ].astype(str).to_numpy(dtype=str),
            )

        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_embeddings_path.replace(
            EMBEDDINGS_PATH
        )
        temporary_metadata_path.replace(
            EXPERIMENT_RECORD_PATH
        )
    finally:
        if temporary_embeddings_path.exists():
            temporary_embeddings_path.unlink()

        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()

def print_inventory_summary(
    inventory: pd.DataFrame,
    excluded_ids: set[str],
) -> None:
    """Print aggregate checks without exposing participant identifiers."""

    print("Raw-audio embedding inputs validated.")
    print(f"Retained participants: {inventory['participant_id'].nunique()}")
    print(f"Excluded participants: {len(excluded_ids)}")
    print(f"Recordings: {len(inventory)}")
    print(f"Tasks: {inventory['task'].nunique()}")
    print(f"Source sample rate: {SOURCE_SAMPLE_RATE} Hz")
    print(f"Encoder sample rate: {TARGET_SAMPLE_RATE} Hz")

def main() -> None:
    """Validate inputs, extract frozen embeddings, and save the artifact."""

    arguments = parse_arguments()
    inventory, excluded_ids = load_extraction_inventory()

    validate_extraction_inventory(
        inventory=inventory,
        excluded_ids=excluded_ids,
    )
    print_inventory_summary(
        inventory=inventory,
        excluded_ids=excluded_ids,
    )

    if arguments.validate_only:
        return

    device = choose_device()
    print(f"Extraction device: {device}")

    feature_extractor, model, revision = load_encoder(
        device=device
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Model revision: {revision}")

    if arguments.smoke_test:
        # The longest retained file is the safest early memory check because
        # it creates the longest transformer sequence in the experiment.
        smoke_inventory = (
            inventory.sort_values(
                "duration_seconds",
                ascending=False,
            )
            .head(1)
            .reset_index(drop=True)
        )
        embeddings = extract_embeddings(
            inventory=smoke_inventory,
            feature_extractor=feature_extractor,
            model=model,
            device=device,
        )
        validate_embeddings(
            embeddings=embeddings,
            expected_rows=1,
        )

        print("WavLM smoke test validated.")
        print(f"Embedding shape: {embeddings.shape}")
        print("No output artifact was written.")
        return

    embeddings = extract_embeddings(
        inventory=inventory,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
    )
    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(inventory),
    )
    save_embeddings(
        inventory=inventory,
        embeddings=embeddings,
        revision=revision,
        device=device,
    )

    print()
    print("Raw-audio WavLM embeddings validated.")
    print(f"Embedding rows: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("Generated artifacts:")
    print(EMBEDDINGS_PATH)
    print(EXPERIMENT_RECORD_PATH)


if __name__ == "__main__":
    main()

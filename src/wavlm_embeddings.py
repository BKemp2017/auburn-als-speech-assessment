"""Extract frozen WavLM embeddings for every task recording in the duplicate-filtered ALS cohort.

Each 8 kHz recording is resampled to 16 kHz and passed through
microsoft/wavlm-base-plus at a pinned revision. The last hidden layer is
pooled over time as mean and standard deviation, giving 1,536 values.

The published embeddings were extracted on CUDA. CPU extraction is close but
not bit-identical, so downstream WavLM numbers can shift slightly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import transformers
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

from common import (
    AUDIO_DIR,
    EMBEDDINGS_PATH,
    EXPERIMENTS,
    GENERATED,
    TASKS,
    WAVLM_MODEL,
    WAVLM_REVISION,
    duplicate_als_ids,
    load_inputs,
    write_json,
)

SOURCE_RATE = 8000
ENCODER_RATE = 16000


def recordings() -> pd.DataFrame:
    """Audio rows for the 100 retained ALS participants, ordered by participant then task."""
    inventory = pd.read_csv(GENERATED / "raw_audio_inventory.csv")
    features, _, _ = load_inputs()
    rows = inventory.loc[inventory["category"].eq("ALS") & ~inventory["participant_id"].isin(duplicate_als_ids())]
    rows = rows.merge(features[["ID", "target"]].rename(columns={"ID": "participant_id"}), on="participant_id", validate="many_to_one")
    task_order = {task: i for i, task in enumerate(TASKS)}
    return rows.sort_values(["participant_id", "task"], key=lambda c: c.map(task_order) if c.name == "task" else c).reset_index(drop=True)


def embed(path, extractor, model, device) -> np.ndarray:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SOURCE_RATE or waveform.ndim != 1:
        raise ValueError(f"{path}: expected mono {SOURCE_RATE} Hz audio, got {rate} Hz with shape {waveform.shape}")
    waveform = resample_poly(waveform, up=ENCODER_RATE, down=rate).astype(np.float32, copy=False)
    inputs = extractor(waveform, sampling_rate=ENCODER_RATE, return_tensors="pt")["input_values"].to(device)
    with torch.inference_mode():
        hidden = model(input_values=inputs).last_hidden_state.squeeze(0)
    pooled = torch.cat((hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)))
    return pooled.to(device="cpu", dtype=torch.float32).numpy()


def main() -> None:
    rows = recordings()
    if len(rows) != 800 or rows["participant_id"].nunique() != 100:
        raise ValueError(f"Expected 800 recordings from 100 participants, found {len(rows)} from {rows['participant_id'].nunique()}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL, revision=WAVLM_REVISION)
    model = AutoModel.from_pretrained(WAVLM_MODEL, revision=WAVLM_REVISION).eval().to(device)
    print(f"Extracting {len(rows)} recordings on {device}")

    embeddings = []
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        embeddings.append(embed(AUDIO_DIR / row.relative_path, extractor, model, device))
        if position % 100 == 0:
            print(f"  {position}/{len(rows)}")
    embeddings = np.stack(embeddings).astype(np.float32)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        participant_ids=rows["participant_id"].to_numpy(dtype=str),
        tasks=rows["task"].to_numpy(dtype=str),
        targets=rows["target"].to_numpy(dtype=np.int64),
        relative_paths=rows["relative_path"].to_numpy(dtype=str),
    )
    record = {
        "experiment": "duplicate_filtered_wavlm_embedding_extraction",
        "participant_level_fields": [],
        "model_name": WAVLM_MODEL,
        "model_revision": WAVLM_REVISION,
        "source_sample_rate_hz": SOURCE_RATE,
        "encoder_sample_rate_hz": ENCODER_RATE,
        "pooling": "last_hidden_state_mean_and_standard_deviation",
        "hidden_size": int(model.config.hidden_size),
        "embedding_dimension": int(embeddings.shape[1]),
        "recordings": len(rows),
        "participants": int(rows["participant_id"].nunique()),
        "tasks": list(TASKS),
        "target_counts": {str(k): int(v) for k, v in rows.drop_duplicates("participant_id")["target"].value_counts().sort_index().items()},
        "duplicate_policy": "exclude_both_duplicate_connected_als_participants",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "extraction_device": str(device),
    }
    write_json(EXPERIMENTS / "raw_audio_wavlm_embeddings.json", record)
    write_json(GENERATED / "raw_audio_wavlm_embeddings_metadata.json", record)


if __name__ == "__main__":
    main()

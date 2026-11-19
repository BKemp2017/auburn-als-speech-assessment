"""Audit the raw VOC-ALS WAV files: headers, signal quality, durations, exact duplicates.

The waveform audit's endpoint-silence and duration columns feed the recording
burden analysis, and its signal hashes define the duplicate-filtered cohort.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import soundfile as sf

from common import AUDIO_DIR, EXPERIMENTS, TASKS, load_inputs, load_metadata, write_csv, write_json

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.01
SILENCE_DBFS = -50.0
MATERIAL_CLIPPING_FRACTION = 0.001
HIGH_SILENCE_FRACTION = 0.5
TASK_DIRECTORIES = {
    "phonationA": "A", "phonationE": "E", "phonationI": "I", "phonationO": "O", "phonationU": "U",
    "rhythmPA": "PA", "rhythmTA": "TA", "rhythmKA": "KA",
}
TASK_ORDER = {task: index for index, task in enumerate(TASKS)}


def inventory() -> pd.DataFrame:
    records = []
    for path in sorted(p for p in AUDIO_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"):
        task_directory = path.parent.name
        record = {
            "relative_path": str(path.relative_to(AUDIO_DIR)),
            "participant_id": path.stem.removesuffix(f"_{task_directory}").strip(),
            "task": TASK_DIRECTORIES[task_directory],
            "task_directory": task_directory,
            "file_size_bytes": path.stat().st_size,
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
            info = sf.info(path)
            record.update(
                readable=True,
                sample_rate_hz=info.samplerate,
                channels=info.channels,
                frames=info.frames,
                duration_seconds=float(info.duration),
                format=info.format,
                subtype=info.subtype,
                endian=info.endian,
            )
        except (OSError, RuntimeError) as error:
            record["header_error"] = str(error)
        records.append(record)
    table = pd.DataFrame(records)
    metadata = load_metadata()[["ID", "Category", "ALSFRS-R_SpeechSubscore"]].rename(
        columns={"ID": "participant_id", "Category": "category", "ALSFRS-R_SpeechSubscore": "speech_subscore"}
    )
    metadata["speech_subscore"] = pd.to_numeric(metadata["speech_subscore"], errors="coerce")  # controls have "-"
    table = table.merge(metadata, on="participant_id", how="left", validate="many_to_one")
    return table.sort_values(["participant_id", "task"], key=lambda c: c.map(TASK_ORDER) if c.name == "task" else c).reset_index(drop=True)


def waveform_stats(row) -> dict:
    record = {
        "relative_path": row.relative_path,
        "participant_id": row.participant_id,
        "task": row.task,
        "category": row.category,
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
        pcm, rate = sf.read(AUDIO_DIR / row.relative_path, dtype="int16", always_2d=False)
    except (OSError, RuntimeError) as error:
        record["waveform_error"] = str(error)
        return record
    if pcm.ndim != 1 or len(pcm) == 0:
        record["waveform_error"] = "Decoded waveform is not a nonempty mono signal."
        return record

    samples = pcm.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    rms_dbfs = float(20.0 * np.log10(rms)) if rms > 0.0 else float("-inf")

    frame, hop = max(1, round(FRAME_SECONDS * rate)), max(1, round(HOP_SECONDS * rate))
    starts = np.arange(0, len(samples), hop)
    ends = np.minimum(starts + frame, len(samples))
    energy = np.concatenate([[0.0], np.cumsum(np.square(samples))])
    silent = np.sqrt((energy[ends] - energy[starts]) / (ends - starts)) < 10.0 ** (SILENCE_DBFS / 20.0)
    active = np.flatnonzero(~silent)
    if len(active):
        leading = starts[active[0]] / rate
        trailing = (len(samples) - ends[active[-1]]) / rate
    else:
        leading = trailing = len(samples) / rate

    clipped = int(((pcm == -32768) | (pcm == 32767)).sum())
    record.update(
        decoded=True,
        decoded_sample_rate_hz=int(rate),
        decoded_samples=len(pcm),
        finite_samples=bool(np.isfinite(samples).all()),
        all_zero=peak == 0.0,
        peak_absolute_amplitude=peak,
        rms_amplitude=rms,
        rms_dbfs=rms_dbfs,
        absolute_dc_offset=float(abs(np.mean(samples))),
        zero_sample_fraction=float(np.mean(pcm == 0)),
        silent_frame_fraction=float(np.mean(silent)),
        leading_silence_seconds=float(leading),
        trailing_silence_seconds=float(trailing),
        clipped_sample_count=clipped,
        clipped_sample_fraction=clipped / len(pcm),
        any_clipping=clipped > 0,
        material_clipping=clipped / len(pcm) >= MATERIAL_CLIPPING_FRACTION,
        near_silent_recording=rms_dbfs <= SILENCE_DBFS,
        signal_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
    )
    return record


def duplicate_als_pairs(audit: pd.DataFrame, features: pd.DataFrame, outer_folds: pd.DataFrame) -> pd.DataFrame:
    """Every pair of ALS participants sharing an identical recording, with their targets and folds."""
    features = features.set_index("ID")
    folds = outer_folds.set_index("participant_id")
    comparisons = []
    for _, group in audit.loc[audit.duplicated("signal_sha256", keep=False)].groupby("signal_sha256", sort=True):
        task = group["task"].iloc[0]
        als = sorted(group.loc[group["category"].eq("ALS"), "participant_id"].unique())
        columns = [c for c in features.columns if c.endswith(f"_{task}")]
        for i, a in enumerate(als):
            for b in als[i + 1:]:
                difference = np.abs(features.loc[a, columns].to_numpy(float) - features.loc[b, columns].to_numpy(float))
                comparisons.append(
                    {
                        "participant_a": a, "participant_b": b, "task": task,
                        "target_a": int(folds.loc[a, "target"]), "target_b": int(folds.loc[b, "target"]),
                        "fold_a": int(folds.loc[a, "outer_fold"]), "fold_b": int(folds.loc[b, "outer_fold"]),
                        "exact": bool(features.loc[a, columns].astype(float).equals(features.loc[b, columns].astype(float))),
                        "max_difference": float(difference.max()),
                    }
                )
    pairs = []
    for number, (_, rows) in enumerate(pd.DataFrame(comparisons).groupby(["participant_a", "participant_b"], sort=True), start=1):
        rows = rows.sort_values("task", key=lambda c: c.map(TASK_ORDER))
        targets = sorted([rows["target_a"].iloc[0], rows["target_b"].iloc[0]])
        folds_pair = sorted([rows["fold_a"].iloc[0], rows["fold_b"].iloc[0]])
        pairs.append(
            {
                "duplicate_als_pair": number,
                "shared_task_count": rows["task"].nunique(),
                "shared_tasks": "+".join(rows["task"]),
                "target_pair": "+".join(map(str, targets)),
                "same_target": targets[0] == targets[1],
                "outer_fold_pair": "+".join(map(str, folds_pair)),
                "same_outer_fold": folds_pair[0] == folds_pair[1],
                "all_task_features_exactly_equal": bool(rows["exact"].all()),
                "all_task_features_numerically_equal": bool(rows["max_difference"].le(1e-12).all()),
                "maximum_absolute_feature_difference": float(rows["max_difference"].max()),
            }
        )
    return pd.DataFrame(pairs)


def audit_record(audit: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    signal_counts = audit["signal_sha256"].value_counts()
    duplicated = signal_counts[signal_counts.gt(1)]
    duplicate_rows = audit.loc[audit["signal_sha256"].isin(duplicated.index)]
    profiles = audit.pivot(index="participant_id", columns="task", values="signal_sha256")[list(TASKS)].agg("|".join, axis=1)
    profile_counts = profiles.value_counts()
    duplicated_profiles = profile_counts[profile_counts.gt(1)]
    categories = audit[["participant_id", "category"]].drop_duplicates()["category"].value_counts().sort_index()
    return {
        "experiment": "voc_als_raw_audio_audit",
        "participant_level_fields": [],
        "dataset": {
            "recordings": len(audit),
            "participants": int(audit["participant_id"].nunique()),
            "participants_by_category": {str(k): int(v) for k, v in categories.items()},
            "recordings_by_task": {t: int(c) for t, c in audit["task"].value_counts().reindex(TASKS).items()},
        },
        "audio_format": {
            "sample_rate_hz_distribution": {str(int(k)): int(v) for k, v in audit["sample_rate_hz"].value_counts().sort_index().items()},
            "channel_distribution": {str(int(k)): int(v) for k, v in audit["channels"].value_counts().sort_index().items()},
            "formats": sorted(audit["format"].astype(str).unique()),
            "subtypes": sorted(audit["subtype"].astype(str).unique()),
        },
        "duration_by_task": {
            task: {
                "minimum_seconds": float(d.min()),
                "median_seconds": float(d.median()),
                "maximum_seconds": float(d.max()),
            }
            for task in TASKS
            for d in [audit.loc[audit["task"].eq(task), "duration_seconds"]]
        },
        "quality_summary": {
            "near_silent_rms_dbfs_threshold": SILENCE_DBFS,
            "near_silent_recordings": int(audit["rms_dbfs"].le(SILENCE_DBFS).sum()),
            "high_silence_frame_fraction_threshold": HIGH_SILENCE_FRACTION,
            "high_silence_recordings": int(audit["silent_frame_fraction"].gt(HIGH_SILENCE_FRACTION).sum()),
            "recordings_with_any_endpoint_clipping": int(audit["clipped_sample_count"].gt(0).sum()),
            "material_clipping_fraction_threshold": MATERIAL_CLIPPING_FRACTION,
            "recordings_with_material_clipping": int(audit["clipped_sample_fraction"].ge(MATERIAL_CLIPPING_FRACTION).sum()),
        },
        "exact_duplicate_summary": {
            "duplicate_signal_groups": len(duplicated),
            "files_in_duplicate_signal_groups": int(duplicated.sum()),
            "duplicate_connected_participants": int(duplicate_rows["participant_id"].nunique()),
            "duplicate_connected_categories": sorted(duplicate_rows["category"].astype(str).unique()),
            "full_duplicate_profile_groups": len(duplicated_profiles),
            "participants_in_full_duplicate_profiles": int(duplicated_profiles.sum()),
        },
        "als_primary_analysis_impact": {
            "duplicate_connected_als_participants": int(duplicate_rows.loc[duplicate_rows["category"].eq("ALS"), "participant_id"].nunique()),
            "duplicate_als_pairs": len(pairs),
            "contradictory_target_pairs": int((~pairs["same_target"]).sum()),
            "cross_outer_fold_pairs": int((~pairs["same_outer_fold"]).sum()),
            "pairs_with_matching_handcrafted_features": int(pairs["all_task_features_numerically_equal"].sum()),
            "pair_comparisons": pairs.to_dict(orient="records"),
        },
        "interpretation": "One complete eight-task audio profile is duplicated across an ALS participant and a healthy-control participant. A second ALS participant shares the same A-task recording. The ALS pair has contradictory severity targets, appears in different outer folds, and has identical A-task handcrafted predictors.",
        "required_follow_up": [
            "Run a duplicate-exclusion sensitivity analysis after removing both affected ALS participants.",
            "Exclude or group the entire three-participant duplicate component before any ALS-versus-control audio model.",
            "Do not treat the current raw-audio dataset as suitable for neural modeling until duplicate handling is frozen.",
        ],
    }


def main() -> None:
    files = inventory()
    unreadable = files.loc[~files["readable"], "relative_path"].tolist()
    if len(files) != 1224 or unreadable or files["category"].isna().any():
        raise ValueError(f"Expected 1,224 readable recordings with metadata; got {len(files)}, unreadable: {unreadable}")
    write_csv(files, "raw_audio_inventory.csv")

    audit = pd.DataFrame([waveform_stats(row) for row in files.itertuples(index=False)])
    audit = audit.merge(
        files[["relative_path", "sample_rate_hz", "frames", "duration_seconds", "channels", "format", "subtype"]],
        on="relative_path",
        validate="one_to_one",
    )
    write_csv(audit, "raw_audio_waveform_audit.csv")

    features, outer_folds, _ = load_inputs()
    pairs = duplicate_als_pairs(audit, features, outer_folds)
    write_csv(pairs, "raw_audio_duplicate_impact.csv")
    record = audit_record(audit, pairs)
    write_json(EXPERIMENTS / "raw_audio_audit.json", record)
    print(f"Recordings: {len(audit)}  decoded: {int(audit['decoded'].sum())}")
    print(f"Exact duplicate signal groups: {record['exact_duplicate_summary']['duplicate_signal_groups']}")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()

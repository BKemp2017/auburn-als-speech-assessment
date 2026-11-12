"""Recording count and duration needed by the WavLM-selected protocols versus the full assessment.

Durations come from the raw-audio waveform audit. Endpoint-trimmed time drops
leading and trailing silence; the nonsilent proxy scales duration by the
fraction of non-silent frames. These are retrospective proxies, not measured
patient effort.
"""

from __future__ import annotations

import pandas as pd

from common import EXPERIMENTS, GENERATED, TASKS, write_csv, write_json
from audit_raw_audio import FRAME_SECONDS, HOP_SECONDS, SILENCE_DBFS

BURDENS = ("recording_count", "recorded_seconds", "endpoint_trimmed_seconds", "nonsilent_seconds_proxy")


def participant_burden(audit: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[["participant_id", "outer_fold", "selected_subset"]].copy()
    selected["selected_task_count"] = selected["selected_subset"].str.split("+").map(len)
    rows = audit.merge(selected, on="participant_id", validate="many_to_one")
    rows["recording_count"] = 1
    rows["recorded_seconds"] = rows["duration_seconds"]
    rows["is_selected"] = [task in subset.split("+") for task, subset in zip(rows["task"], rows["selected_subset"])]

    keys = ["participant_id", "outer_fold", "selected_subset", "selected_task_count"]
    full = rows.groupby(keys, as_index=False)[list(BURDENS)].sum().rename(columns={b: f"full_{b}" for b in BURDENS})
    chosen = rows.loc[rows["is_selected"]].groupby("participant_id", as_index=False)[list(BURDENS)].sum()
    table = full.merge(chosen.rename(columns={b: f"selected_{b}" for b in BURDENS}), on="participant_id", validate="one_to_one")
    for b in BURDENS:
        table[f"{b}_reduction_fraction"] = 1.0 - table[f"selected_{b}"] / table[f"full_{b}"]
    return table.sort_values(["outer_fold", "participant_id"]).reset_index(drop=True)


def summarize(group: pd.DataFrame) -> dict:
    summary = {
        "participants": len(group),
        "mean_selected_tasks": float(group["selected_task_count"].mean()),
        "median_selected_tasks": float(group["selected_task_count"].median()),
    }
    for b in BURDENS:
        full, selected = float(group[f"full_{b}"].sum()), float(group[f"selected_{b}"].sum())
        summary |= {
            f"full_{b}_total": full,
            f"selected_{b}_total": selected,
            f"full_{b}_mean_per_participant": float(group[f"full_{b}"].mean()),
            f"selected_{b}_mean_per_participant": float(group[f"selected_{b}"].mean()),
            f"{b}_reduction_fraction": 1.0 - selected / full,
            f"{b}_reduction_percent": 100.0 * (1.0 - selected / full),
        }
    return summary


def main() -> None:
    predictions = pd.read_csv(GENERATED / "wavlm_task_selection_predictions.csv")
    audit = pd.read_csv(GENERATED / "raw_audio_waveform_audit.csv")
    audit = audit.loc[audit["participant_id"].isin(predictions["participant_id"])].copy()
    if len(audit) != len(predictions) * len(TASKS):
        raise ValueError(f"Expected {len(predictions) * len(TASKS)} recordings for the WavLM cohort, found {len(audit)}.")
    audit["endpoint_trimmed_seconds"] = (
        audit["duration_seconds"] - audit["leading_silence_seconds"] - audit["trailing_silence_seconds"]
    ).clip(lower=0.0)
    audit["nonsilent_seconds_proxy"] = audit["duration_seconds"] * (1.0 - audit["silent_frame_fraction"])

    participants = participant_burden(audit, predictions)
    folds = pd.DataFrame(
        [{"outer_fold": int(f), "selected_subset": g["selected_subset"].iloc[0], **summarize(g)} for f, g in participants.groupby("outer_fold")]
    )
    protocols = pd.DataFrame(
        [{"selected_subset": s, **summarize(g)} for s, g in participants.groupby("selected_subset")]
    ).sort_values(["mean_selected_tasks", "selected_subset"]).reset_index(drop=True)
    tasks = (
        audit.groupby("task", as_index=False)
        .agg(
            recordings=("participant_id", "size"),
            recorded_seconds_mean=("duration_seconds", "mean"),
            recorded_seconds_median=("duration_seconds", "median"),
            endpoint_trimmed_seconds_mean=("endpoint_trimmed_seconds", "mean"),
            nonsilent_seconds_proxy_mean=("nonsilent_seconds_proxy", "mean"),
            silent_frame_fraction_mean=("silent_frame_fraction", "mean"),
        )
        .sort_values("task", key=lambda c: c.map({t: i for i, t in enumerate(TASKS)}))
        .reset_index(drop=True)
    )

    write_csv(participants, "recording_duration_burden_participants.csv")
    write_csv(folds, "recording_duration_burden_by_fold.csv")
    write_csv(protocols, "recording_duration_burden_by_protocol.csv")
    write_csv(tasks, "recording_duration_burden_by_task.csv")
    aggregate = summarize(participants)
    write_json(
        EXPERIMENTS / "recording_duration_burden.json",
        {
            "experiment": "wavlm_selected_protocol_recording_burden",
            "participant_level_fields": [],
            "design": {
                "participants": len(participants),
                "speech_tasks": list(TASKS),
                "full_assessment_task_count": len(TASKS),
                "selection_source": "duplicate_filtered_wavlm_nested_task_selection",
                "selection_policy": "fold-specific protocol selected using inner CV, then applied to that fold's held-out participants",
                "excluded_duplicate_connected_participants": 2,
            },
            "burden_definitions": {
                "recording_count": "number of required task recordings",
                "recorded_seconds": "WAV header duration summed across required tasks",
                "endpoint_trimmed_seconds": "recorded duration after subtracting detected leading and trailing silence",
                "nonsilent_seconds_proxy": "recorded duration multiplied by one minus the fraction of silent analysis frames",
                "waveform_frame_duration_seconds": FRAME_SECONDS,
                "waveform_frame_hop_seconds": HOP_SECONDS,
                "silence_threshold_dbfs": SILENCE_DBFS,
            },
            "aggregate_results": aggregate,
            "fold_results": folds.to_dict(orient="records"),
            "protocol_results": protocols.to_dict(orient="records"),
            "task_duration_results": tasks.to_dict(orient="records"),
            "interpretation": {
                "supported": "The evaluated fold-specific WavLM policy requires fewer recordings and less observed recording time than the complete eight-task assessment.",
                "not_supported": "These retrospective duration proxies do not establish reduced patient fatigue, do not validate one universal task protocol, and do not replace prospective clinical burden measurement.",
            },
        },
    )
    print(
        f"Per participant: {aggregate['full_recording_count_mean_per_participant']:.1f} -> {aggregate['selected_recording_count_mean_per_participant']:.1f} recordings, "
        f"{aggregate['full_recorded_seconds_mean_per_participant']:.2f} -> {aggregate['selected_recorded_seconds_mean_per_participant']:.2f} s "
        f"({aggregate['recorded_seconds_reduction_percent']:.1f}% less)"
    )


if __name__ == "__main__":
    main()

"""Audit VOC-ALS metadata without exposing participant-level records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("A", "E", "I", "O", "U", "PA", "TA", "KA")
FEATURES = (
    "meanF0Hz",
    "stdevF0Hz",
    "HNR",
    "localJitter",
    "localShimmer",
)

EXPECTED_TOTAL_PARTICIPANTS = 153
EXPECTED_ALS_PARTICIPANTS = 102
EXPECTED_HEALTHY_CONTROLS = 51
EXPECTED_NORMAL_SPEECH = 53
EXPECTED_IMPAIRED_SPEECH = 49


def sha256(path: Path) -> str:
    """Calculate a reproducible checksum for the source workbook."""
    digest = hashlib.sha256()

    with path.open("rb") as workbook:
        for chunk in iter(lambda: workbook.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the participant table using the workbook's actual header row."""
    frame = pd.read_excel(
        path,
        sheet_name="VOC-ALS_Data",
        header=1,
    )

    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def build_audit(path: Path, frame: pd.DataFrame) -> dict:
    """Build an aggregate audit containing no participant-level records."""
    public_feature_columns = [
        f"{feature}_{task}"
        for task in TASKS
        for feature in FEATURES
    ]

    required_columns = {
        "ID",
        "Category",
        "ALSFRS-R_SpeechSubscore",
        *public_feature_columns,
    }

    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    numeric_features = frame[public_feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    als = frame.loc[frame["Category"].eq("ALS")].copy()
    speech_score = pd.to_numeric(
        als["ALSFRS-R_SpeechSubscore"],
        errors="coerce",
    )

    category_counts = {
        str(category): int(count)
        for category, count in frame["Category"].value_counts().items()
    }

    speech_score_counts = {
        str(int(score)): int(count)
        for score, count in speech_score.value_counts().sort_index().items()
    }

    report = {
        "source": {
            "path": str(path),
            "sha256": sha256(path),
            "sheet": "VOC-ALS_Data",
            "header_row_zero_based": 1,
        },
        "participants": {
            "total_rows": int(len(frame)),
            "unique_ids": int(frame["ID"].nunique()),
            "duplicate_ids": int(frame["ID"].duplicated().sum()),
            "missing_ids": int(frame["ID"].isna().sum()),
            "category_counts": category_counts,
        },
        "primary_target": {
            "population": "Participants diagnosed with ALS",
            "normal_speech_score_4": int(speech_score.eq(4).sum()),
            "documented_impairment_score_below_4": int(
                speech_score.lt(4).sum()
            ),
            "missing_scores": int(speech_score.isna().sum()),
            "speech_score_counts": speech_score_counts,
        },
        "public_features": {
            "tasks": list(TASKS),
            "features_per_task": len(FEATURES),
            "total_feature_columns": len(public_feature_columns),
            "missing_values": int(numeric_features.isna().sum().sum()),
            "participants_with_missing_features": int(
                numeric_features.isna().any(axis=1).sum()
            ),
            "infinite_values": int(
                np.isinf(numeric_features.to_numpy()).sum()
            ),
        },
    }

    report["validation"] = {
        "total_participants_match": (
            report["participants"]["total_rows"]
            == EXPECTED_TOTAL_PARTICIPANTS
        ),
        "unique_participants_match": (
            report["participants"]["unique_ids"]
            == EXPECTED_TOTAL_PARTICIPANTS
        ),
        "als_count_matches": (
            category_counts.get("ALS") == EXPECTED_ALS_PARTICIPANTS
        ),
        "healthy_control_count_matches": (
            category_counts.get("HC") == EXPECTED_HEALTHY_CONTROLS
        ),
        "normal_speech_count_matches": (
            report["primary_target"]["normal_speech_score_4"]
            == EXPECTED_NORMAL_SPEECH
        ),
        "impaired_speech_count_matches": (
            report["primary_target"][
                "documented_impairment_score_below_4"
            ]
            == EXPECTED_IMPAIRED_SPEECH
        ),
        "all_public_features_present": (
            report["public_features"]["total_feature_columns"] == 40
        ),
        "public_features_complete": (
            report["public_features"]["missing_values"] == 0
        ),
        "public_features_finite": (
            report["public_features"]["infinite_values"] == 0
        ),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/VOC-ALS.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metadata_audit.json"),
    )
    args = parser.parse_args()

    frame = load_metadata(args.input)
    report = build_audit(args.input, frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if not all(report["validation"].values()):
        raise SystemExit("Metadata validation failed.")

    print(f"\nMetadata audit passed. Report written to {args.output}")


if __name__ == "__main__":
    main()
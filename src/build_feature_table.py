# Build the participant-level ALS Modeling table

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from audit_metadata import FEATURES, TASKS, load_metadata

INPUT_PATH = Path("data/raw/VOC-ALS.xlsx")
OUTPUT_PATH = Path("data/processed/als_features.csv")

EXPECTED_ALS_PARTICIPANTS = 102  # Expected number of participants in the ALS dataset
EXPECTED_NORMAL_SPEECH = 53
EXPECTED_IMPAIRED_SPEECH = 49
EXPECTED_FEATURE_COUNT = 40

def feature_columns() -> list[str]:
    # Return the 40 public acoustic predictor columns

    return [
        f"{feature}_{task}"
        for task in TASKS
        for feature in FEATURES
    ]

def build_feature_table(path: Path) -> pd.DataFrame:
    #  Build the ALS only participant level modeling table
    
    metadata = load_metadata(path)
    columns = feature_columns()

    als = metadata.loc[
        metadata["Category"].eq("ALS"),
        ["ID", "ALSFRS-R_SpeechSubscore", *columns]
    ].copy()

    speech_score = pd.to_numeric(
        als["ALSFRS-R_SpeechSubscore"], 
        errors = "raise"
    )

    als["target"] = speech_score.lt(4).astype(int)

    als[columns] = als[columns].apply(
        pd.to_numeric,
        errors = "raise"
    )

    modeling_table = als[
        ["ID", "target", *columns]
    ].copy()

    return modeling_table.reset_index(drop=True)

def validate_feature_table(table: pd.DataFrame) -> None:
    """Validate the participant-level modeling table."""

    columns = feature_columns()

    if len(columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(columns)}."
        )

    if len(table) != EXPECTED_ALS_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_ALS_PARTICIPANTS} ALS participants, "
            f"found {len(table)}."
        )

    if table["ID"].isna().any():
        raise ValueError("Missing participant ID detected.")

    if table["ID"].duplicated().any():
        raise ValueError("Duplicate participant ID detected.")

    normal_count = int(table["target"].eq(0).sum())
    impaired_count = int(table["target"].eq(1).sum())

    if normal_count != EXPECTED_NORMAL_SPEECH:
        raise ValueError(
            f"Expected {EXPECTED_NORMAL_SPEECH} normal-speech participants, "
            f"found {normal_count}."
        )

    if impaired_count != EXPECTED_IMPAIRED_SPEECH:
        raise ValueError(
            f"Expected {EXPECTED_IMPAIRED_SPEECH} impaired participants, "
            f"found {impaired_count}."
        )

    if table[columns].isna().any().any():
        raise ValueError("Missing acoustic feature value detected.")

    if not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError("Non-finite acoustic feature value detected.")




def main() -> None:
    table = build_feature_table(INPUT_PATH)

    validate_feature_table(table)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Feature table validation passed.")
    print(f"Participants: {len(table)}")
    print(f"Predictors: {len(feature_columns())}")
    print()
    print("Target counts:")
    print(table["target"].value_counts().sort_index())
    print()
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
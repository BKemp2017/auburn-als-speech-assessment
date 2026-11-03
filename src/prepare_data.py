"""Audit the VOC-ALS metadata, build the ALS feature table, and freeze the CV folds.

Outputs:
  results/metadata_audit.json, results/fold_audit.json (tracked aggregates)
  data/processed/als_features.csv, outer_folds.csv, nested_folds.json (private)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import (
    FEATURES,
    FEATURES_PATH,
    METADATA_PATH,
    NESTED_FOLDS_PATH,
    OUTER_FOLDS_PATH,
    SEED,
    TASKS,
    feature_columns,
    load_metadata,
    write_json,
)

OUTER_SPLITS = 5
INNER_SPLITS = 4


def audit_metadata(frame: pd.DataFrame) -> dict:
    columns = feature_columns()
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    als = frame.loc[frame["Category"].eq("ALS")]
    speech = pd.to_numeric(als["ALSFRS-R_SpeechSubscore"], errors="coerce")
    categories = {str(key): int(value) for key, value in frame["Category"].value_counts().items()}
    report = {
        "source": {
            "path": str(METADATA_PATH),
            "sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
            "sheet": "VOC-ALS_Data",
            "header_row_zero_based": 1,
        },
        "participants": {
            "total_rows": len(frame),
            "unique_ids": int(frame["ID"].nunique()),
            "duplicate_ids": int(frame["ID"].duplicated().sum()),
            "missing_ids": int(frame["ID"].isna().sum()),
            "category_counts": categories,
        },
        "primary_target": {
            "population": "Participants diagnosed with ALS",
            "normal_speech_score_4": int(speech.eq(4).sum()),
            "documented_impairment_score_below_4": int(speech.lt(4).sum()),
            "missing_scores": int(speech.isna().sum()),
            "speech_score_counts": {str(int(k)): int(v) for k, v in speech.value_counts().sort_index().items()},
        },
        "public_features": {
            "tasks": list(TASKS),
            "features_per_task": len(FEATURES),
            "total_feature_columns": len(columns),
            "missing_values": int(numeric.isna().sum().sum()),
            "participants_with_missing_features": int(numeric.isna().any(axis=1).sum()),
            "infinite_values": int(np.isinf(numeric.to_numpy()).sum()),
        },
    }
    report["validation"] = {
        "total_participants_match": report["participants"]["total_rows"] == 153,
        "unique_participants_match": report["participants"]["unique_ids"] == 153,
        "als_count_matches": categories.get("ALS") == 102,
        "healthy_control_count_matches": categories.get("HC") == 51,
        "normal_speech_count_matches": report["primary_target"]["normal_speech_score_4"] == 53,
        "impaired_speech_count_matches": report["primary_target"]["documented_impairment_score_below_4"] == 49,
        "all_public_features_present": len(columns) == 40,
        "public_features_complete": report["public_features"]["missing_values"] == 0,
        "public_features_finite": report["public_features"]["infinite_values"] == 0,
    }
    return report


def build_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns()
    als = frame.loc[frame["Category"].eq("ALS"), ["ID", "ALSFRS-R_SpeechSubscore", *columns]].copy()
    als["target"] = pd.to_numeric(als["ALSFRS-R_SpeechSubscore"], errors="raise").lt(4).astype(int)
    als[columns] = als[columns].apply(pd.to_numeric, errors="raise")
    return als[["ID", "target", *columns]].reset_index(drop=True)


def class_counts(target: pd.Series) -> dict[str, int]:
    return {"normal_speech": int(target.eq(0).sum()), "documented_impairment": int(target.eq(1).sum())}


def create_folds(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Stratified participant-disjoint 5x4 nested folds. Each inner split reseeds per outer fold."""
    outer_rows, nested_rows, outer_audit = [], [], []
    outer_splitter = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    for outer_fold, (train_idx, test_idx) in enumerate(outer_splitter.split(table, table["target"])):
        outer_train = table.iloc[train_idx].reset_index(drop=True)
        outer_test = table.iloc[test_idx].reset_index(drop=True)
        outer_rows += [
            {"participant_id": row.ID, "target": int(row.target), "outer_fold": outer_fold}
            for row in outer_test.itertuples(index=False)
        ]
        inner_splitter = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + outer_fold)
        inner_audit = []
        for inner_fold, (inner_train_idx, validation_idx) in enumerate(
            inner_splitter.split(outer_train, outer_train["target"])
        ):
            inner_train = outer_train.iloc[inner_train_idx]
            validation = outer_train.iloc[validation_idx]
            nested_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "inner_train_ids": sorted(inner_train["ID"]),
                    "inner_validation_ids": sorted(validation["ID"]),
                    "outer_test_ids": sorted(outer_test["ID"]),
                }
            )
            inner_audit.append(
                {
                    "inner_fold": inner_fold,
                    "training_participants": len(inner_train),
                    "validation_participants": len(validation),
                    "training_classes": class_counts(inner_train["target"]),
                    "validation_classes": class_counts(validation["target"]),
                }
            )
        outer_audit.append(
            {
                "outer_fold": outer_fold,
                "training_participants": len(outer_train),
                "test_participants": len(outer_test),
                "training_classes": class_counts(outer_train["target"]),
                "test_classes": class_counts(outer_test["target"]),
                "inner_folds": inner_audit,
            }
        )
    outer_folds = pd.DataFrame(outer_rows).sort_values(["outer_fold", "participant_id"])
    leakage = any(
        set(row["inner_train_ids"]) & set(row["inner_validation_ids"])
        or (set(row["inner_train_ids"]) | set(row["inner_validation_ids"])) & set(row["outer_test_ids"])
        for row in nested_rows
    )
    duplicates = int(outer_folds["participant_id"].duplicated().sum())
    audit = {
        "design": {
            "random_seed": SEED,
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "stratification_target": "ALSFRS-R speech score < 4 versus score = 4",
            "participant_disjoint": True,
        },
        "population": {"participants": len(table), "classes": class_counts(table["target"])},
        "outer_folds": outer_audit,
        "validation": {
            "all_participants_assigned_once": duplicates == 0 and set(outer_folds["participant_id"]) == set(table["ID"]),
            "duplicate_outer_assignments": duplicates,
            "participant_leakage_detected": leakage,
        },
    }
    if leakage or duplicates:
        raise ValueError("Fold construction produced overlapping participants.")
    return outer_folds, nested_rows, audit


def main() -> None:
    metadata = load_metadata()
    report = audit_metadata(metadata)
    write_json(Path("results/metadata_audit.json"), report)
    failed = [name for name, passed in report["validation"].items() if not passed]
    if failed:
        raise SystemExit(f"Metadata audit failed: {failed}")

    table = build_feature_table(metadata)
    if not np.isfinite(table[feature_columns()].to_numpy()).all():
        raise ValueError("Feature table contains missing or non-finite values.")
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)

    outer_folds, nested_folds, fold_audit = create_folds(table)
    outer_folds.to_csv(OUTER_FOLDS_PATH, index=False)
    NESTED_FOLDS_PATH.write_text(json.dumps(nested_folds, indent=2) + "\n")
    write_json(Path("results/fold_audit.json"), fold_audit)

    print(f"Participants: {len(table)}  (normal {int(table['target'].eq(0).sum())}, impaired {int(table['target'].eq(1).sum())})")
    print(f"Predictors: {len(feature_columns())}")
    print(f"Outer folds: {OUTER_SPLITS}, inner folds per outer fold: {INNER_SPLITS}")


if __name__ == "__main__":
    main()

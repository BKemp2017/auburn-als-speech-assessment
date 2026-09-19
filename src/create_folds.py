""" Create reproducible participant-disjoint nested CV folds """

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from audit_metadata import load_metadata

INPUT_PATH = Path("data/raw/VOC-ALS.xlsx")
OUTER_FOLDS_PATH = Path("data/processed/outer_folds.csv")
NESTED_FOLDS_PATH = Path("data/processed/nested_folds.json")
AUDIT_PATH = Path("results/fold_audit.json")

RANDOM_SEED = 2026
OUTER_SPLITS = 5
INNER_SPLITS = 4

def class_counts(target: pd.Series) -> dict[str, int]:
    # return aggregate binart-target counts
    return {
        "normal_speech": int(target.eq(0).sum()),
        "documented_impairment": int(target.eq(1).sum()),
    }

def main() -> None:
    metadata = load_metadata(INPUT_PATH)

    als = metadata.loc[
        metadata["Category"].eq("ALS"),
        ["ID", "ALSFRS-R_SpeechSubscore"],
        ].copy()

    als["speech_score"] = pd.to_numeric(als["ALSFRS-R_SpeechSubscore"], errors="raise")

    als["target"] = als["speech_score"].lt(4).astype(int)
    als = als.reset_index(drop=True)

    outer_splitter = StratifiedKFold(
        n_splits=OUTER_SPLITS, 
        shuffle=True, 
        random_state=RANDOM_SEED)

    outer_assignments: list[dict] = []
    nested_assignments: list[dict] = []
    outer_audit: list[dict] = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(als, als["target"])
    ):
        outer_train = als.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = als.iloc[outer_test_idx].reset_index(drop=True)

        for row in outer_test.itertuples(index=False):
            outer_assignments.append(
                {
                    "participant_id": row.ID,
                    "target": int(row.target),
                    "outer_fold": outer_fold,
                }
            )
        inner_splitter = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + outer_fold,
        )

        inner_audit: list[dict] = []

        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(
            inner_splitter.split(outer_train, outer_train["target"])
        ):
            inner_train = outer_train.iloc[inner_train_idx]
            inner_validation = outer_train.iloc[inner_validation_idx]

            train_ids = set(inner_train["ID"])
            validation_ids = set(inner_validation["ID"])
            outer_test_ids = set(outer_test["ID"])

            if train_ids & validation_ids:
                raise ValueError("Inner train/validation participant leakage detected")
            
            if train_ids & outer_test_ids:
                raise ValueError("Inner train/outer test participant leakage detected")
            
            if validation_ids & outer_test_ids:
                raise ValueError("Inner validation/outer test participant leakage detected")
            
            nested_assignments.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "inner_train_ids": sorted(train_ids),
                    "inner_validation_ids": sorted(validation_ids),
                    "outer_test_ids": sorted(outer_test_ids)
                }
            )

            inner_audit.append(
                {
                    "inner_fold": inner_fold,
                    "training_participants": int(len(inner_train)),
                    "validation_participants": int(len(inner_validation)),
                    "training_classes": class_counts(inner_train["target"]),
                    "validation_classes": class_counts(inner_validation["target"]),
                }
            )

        outer_audit.append(
            {
                "outer_fold": outer_fold,
                "training_participants": int(len(outer_train)),
                "test_participants": int(len(outer_test)),
                "training_classes": class_counts(outer_train["target"]),
                "test_classes": class_counts(outer_test["target"]),
                "inner_folds": inner_audit,
            }
        )

    outer_frame = pd.DataFrame(outer_assignments).sort_values(
        ["outer_fold", "participant_id"]
    )

    if len(outer_frame) != len(als):
        raise ValueError("Not every ALS participant received an outer fold.")

    if outer_frame["participant_id"].duplicated().any():
        raise ValueError("A participant was assigned to multiple outer folds.")

    audit = {
            "design": {
            "random_seed": RANDOM_SEED,
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "stratification_target": (
                "ALSFRS-R speech score < 4 versus score = 4"
            ),
            "participant_disjoint": True,
        },
        "population": {
            "participants": int(len(als)),
            "classes": class_counts(als["target"]),
        },
        "outer_folds": outer_audit,
        "validation": {
            "all_participants_assigned_once": True,
            "duplicate_outer_assignments": 0,
            "participant_leakage_detected": False,
        },
    }

    OUTER_FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    outer_frame.to_csv(OUTER_FOLDS_PATH, index=False)
    NESTED_FOLDS_PATH.write_text(
        json.dumps(nested_assignments, indent=2) + "\n",
        encoding="utf-8",
    )
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8"
    )

    print(json.dumps(audit, indent=2))
    print("\nFold generation passed.")
    print(f"Private outer assignments: {OUTER_FOLDS_PATH}")
    print(f"Private nested assignments: {NESTED_FOLDS_PATH}")
    print(f"Aggregate audit: {AUDIT_PATH}")

if __name__ == "__main__":
    main()
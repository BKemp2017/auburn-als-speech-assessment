"""Shared constants, data loading, models, and metrics for the VOC-ALS pipeline."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

METADATA_PATH = Path("data/raw/VOC-ALS.xlsx")
AUDIO_DIR = Path("data/raw/VOC-ALS")
FEATURES_PATH = Path("data/processed/als_features.csv")
OUTER_FOLDS_PATH = Path("data/processed/outer_folds.csv")
NESTED_FOLDS_PATH = Path("data/processed/nested_folds.json")
GENERATED = Path("results/generated")
EMBEDDINGS_PATH = GENERATED / "raw_audio_wavlm_embeddings.npz"
EXPERIMENTS = Path("experiments")
FIGURES = Path("figures")

TASKS = ("A", "E", "I", "O", "U", "PA", "TA", "KA")
FULL_ID = "+".join(TASKS)
FEATURES = ("meanF0Hz", "stdevF0Hz", "HNR", "localJitter", "localShimmer")
SEED = 2026
DELTA = 0.02
BOOTSTRAP_REPLICATES = 10_000
WAVLM_MODEL = "microsoft/wavlm-base-plus"
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
    "positive_class_f1",
    "accuracy",
)


def load_metadata() -> pd.DataFrame:
    frame = pd.read_excel(METADATA_PATH, sheet_name="VOC-ALS_Data", header=1)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def feature_columns(tasks: tuple[str, ...] = TASKS) -> list[str]:
    return [f"{feature}_{task}" for task in tasks for feature in FEATURES]


def all_subsets(tasks: tuple[str, ...] = TASKS) -> list[tuple[str, ...]]:
    """Every nonempty task subset, ordered by size and then by task order."""
    return [subset for size in range(1, len(tasks) + 1) for subset in combinations(tasks, size)]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    features = pd.read_csv(FEATURES_PATH)
    outer_folds = pd.read_csv(OUTER_FOLDS_PATH)
    nested_folds = json.loads(NESTED_FOLDS_PATH.read_text())
    return features, outer_folds, nested_folds


def frozen_configurations() -> dict[int, dict]:
    """Model configuration chosen per outer fold by the eight-task baseline."""
    selections = json.loads((GENERATED / "full_eight_task_selections.json").read_text())
    return {int(row["outer_fold"]): row["configuration"] for row in selections}


def inner_assignments(nested_folds: list[dict], outer_fold: int) -> list[dict]:
    return sorted(
        (row for row in nested_folds if row["outer_fold"] == outer_fold),
        key=lambda row: row["inner_fold"],
    )


def build_pipeline(configuration: dict) -> Pipeline:
    name = configuration["model"]
    if name == "logistic_regression":
        classifier = LogisticRegression(
            C=configuration["C"],
            class_weight=configuration["class_weight"],
            max_iter=5000,
            solver="liblinear",
            random_state=SEED,
        )
    elif name == "rbf_svm":
        classifier = SVC(
            kernel="rbf",
            C=configuration["C"],
            gamma=configuration["gamma"],
            class_weight=configuration["class_weight"],
        )
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=configuration["n_estimators"],
            max_depth=configuration["max_depth"],
            min_samples_leaf=configuration["min_samples_leaf"],
            class_weight=configuration["class_weight"],
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=configuration["n_estimators"],
            learning_rate=configuration["learning_rate"],
            max_depth=configuration["max_depth"],
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model type: {name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def classification_metrics(y_true, y_pred, scores) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "positive_class_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def fit_and_score(configuration: dict, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    """Fit on train, return test predictions and a continuous score for ROC AUC.

    The RBF SVM has no predict_proba, so its decision function is used instead.
    """
    pipeline = build_pipeline(configuration)
    pipeline.fit(train[columns], train["target"])
    predictions = pipeline.predict(test[columns])
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(test[columns])[:, 1]
    else:
        scores = pipeline.decision_function(test[columns])
    return predictions, scores


def outer_split(features: pd.DataFrame, outer_folds: pd.DataFrame, outer_fold: int):
    test_ids = set(outer_folds.loc[outer_folds["outer_fold"].eq(outer_fold), "participant_id"])
    is_test = features["ID"].isin(test_ids)
    return features.loc[~is_test], features.loc[is_test]


def mean_std(values) -> dict[str, float]:
    values = pd.Series(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def paired_fold_bootstrap(fold_tables: list[pd.DataFrame], columns: list[str]) -> np.ndarray:
    """Resample participants within each outer fold and average Macro F1 over folds.

    Every column in `columns` is scored on the same resampled participants, so
    differences between columns stay paired. Draws with only one class are
    redrawn because Macro F1 is not meaningful there. Returns an array of shape
    (BOOTSTRAP_REPLICATES, len(columns)).
    """
    rng = np.random.default_rng(SEED)
    results = np.empty((BOOTSTRAP_REPLICATES, len(columns)))
    for replicate in range(BOOTSTRAP_REPLICATES):
        fold_scores = []
        for table in fold_tables:
            while True:
                sampled = table.iloc[rng.integers(0, len(table), size=len(table))]
                if sampled["true_label"].nunique() == 2:
                    break
            fold_scores.append(
                [
                    f1_score(sampled["true_label"], sampled[column], labels=[0, 1], average="macro", zero_division=0)
                    for column in columns
                ]
            )
        results[replicate] = np.mean(fold_scores, axis=0)
    return results


def duplicate_als_ids() -> set[str]:
    """ALS participants who share an exact recording with another ALS participant."""
    audit = pd.read_csv(GENERATED / "raw_audio_waveform_audit.csv")
    duplicates = audit.loc[audit.duplicated("signal_sha256", keep=False) & audit["category"].eq("ALS")]
    shared = duplicates.groupby("signal_sha256")["participant_id"].nunique()
    ids = set(duplicates.loc[duplicates["signal_sha256"].isin(shared[shared >= 2].index), "participant_id"])
    if len(ids) != 2:
        raise ValueError(f"Expected 2 duplicate-connected ALS participants, found {len(ids)}: {sorted(ids)}")
    return ids


def without_participants(features, outer_folds, nested_folds, excluded: set[str]):
    """Drop participants while keeping every remaining fold assignment unchanged."""
    keep_features = features.loc[~features["ID"].isin(excluded)].reset_index(drop=True)
    keep_folds = outer_folds.loc[~outer_folds["participant_id"].isin(excluded)].reset_index(drop=True)
    keep_nested = [
        {
            **row,
            **{
                field: [pid for pid in row[field] if pid not in excluded]
                for field in ("inner_train_ids", "inner_validation_ids", "outer_test_ids")
            },
        }
        for row in nested_folds
    ]
    return keep_features, keep_folds, keep_nested


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, record, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(record), indent=2, allow_nan=False, sort_keys=sort_keys) + "\n")


def write_csv(table: pd.DataFrame, name: str) -> None:
    path = GENERATED / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

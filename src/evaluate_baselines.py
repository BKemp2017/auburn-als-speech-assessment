"""Evaluate the full eight-task ALS 
speech baseline with nested cross-validation."""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

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

from build_feature_table import feature_columns

FEATURE_TABLE_PATH = Path("data/processed/als_features.csv")
OUTER_FOLDS_PATH = Path("data/processed/outer_folds.csv")
NESTED_FOLDS_PATH = Path("data/processed/nested_folds.json")
OUTPUT_DIR = Path("results/generated")

FOLD_METRICS_PATH = OUTPUT_DIR / "full_eight_task_fold_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "full_eight_task_predictions.csv"
SELECTIONS_PATH = OUTPUT_DIR / "full_eight_task_selections.json"
SUMMARY_PATH = OUTPUT_DIR / "full_eight_task_summary.json"

RANDOM_SEED = 2026

def load_experiment_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
]:
    # Load the modeling table and frozen nested-CV assignments

    features = pd.read_csv(FEATURE_TABLE_PATH)
    outer_folds = pd.read_csv(OUTER_FOLDS_PATH)

    nested_folds = json.loads(
        NESTED_FOLDS_PATH.read_text(
            encoding="utf-8",
        )
    )

    return features, outer_folds, nested_folds

def validate_experiment_inputs(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict],
) -> None:
    
    # Verify that the frozen CV assignments match the modeling table.

    expected_features = feature_columns()

    if len(features) != 102:
        raise ValueError(
            f"Expected 102 participants, found {len(features)}."
        )

    if len(expected_features) != 40:
        raise ValueError(
            f"Expected 40 predictors, found {len(expected_features)}."
        )

    feature_ids = set(features["ID"])
    fold_ids = set(outer_folds["participant_id"])

    if feature_ids != fold_ids:
        raise ValueError(
            "Participant IDs in the feature table and outer folds do not match."
        )

    if outer_folds["participant_id"].duplicated().any():
        raise ValueError(
            "Duplicate participant assignment detected in outer folds."
        )

    if outer_folds["outer_fold"].nunique() != 5:
        raise ValueError(
            "Expected exactly 5 outer folds."
        )

    if len(nested_folds) != 20:
        raise ValueError(
            f"Expected 20 nested fold records, found {len(nested_folds)}."
        )

    merged = features[
        ["ID", "target"]
    ].merge(
        outer_folds[
            ["participant_id", "target"]
        ],
        left_on="ID",
        right_on="participant_id",
        suffixes=("_features", "_folds"),
        validate="one_to_one",
    )

    if not merged["target_features"].equals(
        merged["target_folds"]
    ):
        raise ValueError(
            "Target values disagree between the feature table and fold assignments."
        )

def candidate_configuration() -> list[dict]:
    # return al prespecifiedd baseline model configs

    configurations: list[dict] = []

    for c_value in (0.01, 0.1, 1, 10, 100):
        for class_weight in (None, "balanced"):
            configurations.append(
                {
                    "model": "logistic_regression",
                    "C": c_value,
                    "class_weight": class_weight
                }
            )

    for c_value in (0.1, 1, 10, 100): 
        for gamma in ("scale", 0.001, 0.01, 0.1, 1):
            for class_weight in (None, "balanced"):
                configurations.append(
                    {
                        "model": "rbf_svm",
                        "C": c_value,
                        "gamma": gamma,
                        "class_weight": class_weight,
                    }
                )

    return configurations

def build_pipeline(configuration: dict) -> Pipeline:
    # Standard classifier pipeline for one configuration

    model_name = configuration["model"]

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C = configuration["C"],
            class_weight = configuration["class_weight"],
            max_iter = 5000,
            solver = "liblinear",
            random_state = RANDOM_SEED,
        )
    elif model_name == "rbf_svm":
        classifier = SVC(
            kernel = "rbf",
            C = configuration["C"],
            gamma = configuration["gamma"],
            class_weight = configuration["class_weight"],
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_name}"
        )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

def evaluate_inner_fold(
        features: pd.DataFrame,
        assignment: dict,
        configuration: dict,
) -> float:
    # Train and evaluate one config on one inner CV Fold

    columns = feature_columns()

    train_ids = set(assignment["inner_train_ids"])
    validation_ids = set(assignment["inner_validation_ids"])
    outer_test_ids = set(assignment["outer_test_ids"])

    if train_ids & validation_ids:
        raise ValueError(
            "Inner training and validation particpants overlap"
        )

    train = features.loc[
        features["ID"].isin(train_ids)
    ].copy()

    validation = features.loc[
        features["ID"].isin(validation_ids)
    ].copy()

    if set(train["ID"]) != train_ids:
        raise ValueError(
            "Not all inner-training participants were found."
        )

    if set(validation["ID"]) != validation_ids:
        raise ValueError(
            "Not all inner-validation participants were found."
        )

    X_train = train[columns]
    y_train = train["target"]

    X_validation = validation[columns]
    y_validation = validation["target"]

    pipeline = build_pipeline(configuration)

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_validation
    )

    score = balanced_accuracy_score(
        y_validation,
        predictions,
    )

    return float(score)

def evaluate_configuration(
        features: pd.DataFrame,
        nested_folds: list[dict],
        outer_fold: int,
        configuration: dict,
) -> dict:
    # Evaluate one model config across all inner folds

    assignments = [
        assignment
        for assignment in nested_folds
        if assignment["outer_fold"] == outer_fold
    ]

    assignments = sorted(
        assignments,
        key=lambda assignment: assignment["inner_fold"],
    )

    if len(assignments) != 4:
        raise ValueError(
            f"Expected 4 inner folds for outer fold {outer_fold}, "
            f"found {len(assignments)}"
        )

    scores: list[float] = []

    for assignment in assignments:
        score = evaluate_inner_fold(
            features,
            assignment,
            configuration,
        )

        scores.append(score)
    mean_score = sum(scores) / len(scores)

    return {
        "outer_fold": outer_fold,
        "configuration": configuration,
        "inner_scores": scores,
        "mean_balanced_accuracy": float(mean_score),
    }

def selection_key(result: dict) -> tuple:
    # Return the deterministic inner-loop model-selection key.

    configuration = result["configuration"]

    model_preference = (
        0 if configuration["model"] == "logistic_regression"
        else 1
    )

    return (
        -result["mean_balanced_accuracy"],
        configuration["C"],
        model_preference,
        result["grid_index"],
    )


def select_best_configuration(
    features: pd.DataFrame,
    nested_folds: list[dict],
    outer_fold: int,
    configurations: list[dict],
) -> tuple[dict, list[dict]]:
    # Evaluate all candidates and select the best inner-loop configuration.

    results: list[dict] = []

    for grid_index, configuration in enumerate(configurations):
        result = evaluate_configuration(
            features,
            nested_folds,
            outer_fold,
            configuration,
        )

        result["grid_index"] = grid_index
        results.append(result)

    ranked_results = sorted(
        results,
        key=selection_key,
    )

    return ranked_results[0], ranked_results

def evaluate_outer_fold(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    outer_fold: int,
    configuration: dict,
) -> dict:
    # Refit the selected model and evaluate one untouched outer test fold.

    columns = feature_columns()

    test_ids = set(
        outer_folds.loc[
            outer_folds["outer_fold"].eq(outer_fold),
            "participant_id",
        ]
    )

    train = features.loc[
        ~features["ID"].isin(test_ids)
    ].copy()

    test = features.loc[
        features["ID"].isin(test_ids)
    ].copy()

    if set(train["ID"]) & set(test["ID"]):
        raise ValueError(
            "Outer training and test participants overlap."
        )

    if set(test["ID"]) != test_ids:
        raise ValueError(
            "Not all outer-test participants were found."
        )

    if len(train) + len(test) != len(features):
        raise ValueError(
            "Outer train/test partition does not cover all participants."
        )

    X_train = train[columns]
    y_train = train["target"]

    X_test = test[columns]
    y_test = test["target"]

    pipeline = build_pipeline(configuration)

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    if configuration["model"] == "logistic_regression":
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    metrics = {
        "outer_fold": outer_fold,
        "training_participants": len(train),
        "test_participants": len(test),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                scores,
            )
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_test,
                predictions,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "configuration": configuration,
    }

    prediction_records = []

    for participant_id, true_label, predicted_label, score in zip(
        test["ID"],
        y_test,
        predictions,
        scores,
    ):
        prediction_records.append(
            {
                "participant_id": participant_id,
                "outer_fold": outer_fold,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "score": float(score),
            }
        )

    return {
        "metrics": metrics,
        "predictions": prediction_records,
    }

def run_full_baseline(
    features: pd.DataFrame,
    outer_folds: pd.DataFrame,
    nested_folds: list[dict],
    configurations: list[dict],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
]:
    # Run the complete five-outer-fold eight-task baseline.

    fold_metrics: list[dict] = []
    prediction_records: list[dict] = []
    selection_records: list[dict] = []

    outer_fold_values = sorted(
        outer_folds["outer_fold"].unique()
    )

    for outer_fold in outer_fold_values:
        print()
        print(
            f"Selecting configuration for outer fold "
            f"{outer_fold}..."
        )

        best_result, _ = select_best_configuration(
            features,
            nested_folds,
            outer_fold=outer_fold,
            configurations=configurations,
        )

        configuration = best_result["configuration"]

        print(
            "Selected: "
            f"{configuration}"
        )

        print(
            "Mean inner balanced accuracy: "
            f"{best_result['mean_balanced_accuracy']:.4f}"
        )

        outer_result = evaluate_outer_fold(
            features,
            outer_folds,
            outer_fold=outer_fold,
            configuration=configuration,
        )

        metrics = outer_result["metrics"]

        metrics["selected_inner_balanced_accuracy"] = (
            best_result["mean_balanced_accuracy"]
        )

        fold_metrics.append(metrics)

        prediction_records.extend(
            outer_result["predictions"]
        )

        selection_records.append(
            {
                "outer_fold": int(outer_fold),
                "configuration": configuration,
                "mean_inner_balanced_accuracy": (
                    best_result["mean_balanced_accuracy"]
                ),
                "inner_scores": best_result["inner_scores"],
            }
        )

        print(
            "Outer balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.DataFrame(prediction_records),
        selection_records,
    )

def summarize_outer_results(
    fold_metrics: pd.DataFrame,
) -> dict:
    # Summarize generalized performance across outer folds.

    metric_names = [
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "specificity",
        "f1",
        "accuracy",
    ]

    summary = {
        "task_count": 8,
        "predictor_count": len(feature_columns()),
        "outer_folds": len(fold_metrics),
        "metrics": {},
    }

    for metric_name in metric_names:
        summary["metrics"][metric_name] = {
            "mean": float(
                fold_metrics[metric_name].mean()
            ),
            "std": float(
                fold_metrics[metric_name].std(ddof=1)
            ),
        }

    return summary

def save_full_baseline_results(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    selections: list[dict],
    summary: dict,
) -> None:
    """Save reproducibility artifacts for the full baseline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    SELECTIONS_PATH.write_text(
        json.dumps(
            selections,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

def main() -> None:
    features, outer_folds, nested_folds = load_experiment_inputs()

    validate_experiment_inputs(
        features,
        outer_folds,
        nested_folds,
    )

    configurations = candidate_configuration()

    print("Baseline experiment inputs validated.")
    print(f"Participants: {len(features)}")
    print(f"Predictors: {len(feature_columns())}")
    print(f"Candidate configurations: {len(configurations)}")

    (
        fold_metrics,
        predictions,
        selections,
    ) = run_full_baseline(
        features,
        outer_folds,
        nested_folds,
        configurations,
    )

    summary = summarize_outer_results(
        fold_metrics
    )

    save_full_baseline_results(
        fold_metrics,
        predictions,
        selections,
        summary,
    )

    print()
    print("Outer-fold results:")
    print(
        fold_metrics[
            [
                "outer_fold",
                "balanced_accuracy",
                "roc_auc",
                "sensitivity",
                "specificity",
                "f1",
                "accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print("Eight-task baseline summary:")

    for metric_name, values in summary["metrics"].items():
        print(
            f"{metric_name}: "
            f"{values['mean']:.4f} "
            f"+/- {values['std']:.4f}"
        )

if __name__ == "__main__":
    main()
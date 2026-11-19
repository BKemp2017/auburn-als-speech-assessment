"""Majority-class baseline and the full eight-task nested-CV baseline.

The full baseline tunes over 70 model configurations inside each outer
training split. The configuration it picks per outer fold is frozen and reused
by every later handcrafted-feature analysis.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.dummy import DummyClassifier

from common import (
    EXPERIMENTS,
    GENERATED,
    METRICS,
    SEED,
    build_pipeline,
    classification_metrics,
    feature_columns,
    fit_and_score,
    inner_assignments,
    load_inputs,
    macro_f1,
    mean_std,
    outer_split,
    write_csv,
    write_json,
)
from handcrafted import check_folds

MODEL_PREFERENCE = {"logistic_regression": 0, "rbf_svm": 1, "random_forest": 2, "gradient_boosting": 3}


def candidate_configurations() -> list[dict]:
    grid = []
    for c in (0.01, 0.1, 1, 10, 100):
        for class_weight in (None, "balanced"):
            grid.append({"model": "logistic_regression", "C": c, "class_weight": class_weight})
    for c in (0.1, 1, 10, 100):
        for gamma in ("scale", 0.001, 0.01, 0.1, 1):
            for class_weight in (None, "balanced"):
                grid.append({"model": "rbf_svm", "C": c, "gamma": gamma, "class_weight": class_weight})
    for max_depth in (None, 5, 10):
        for min_samples_leaf in (1, 2):
            for class_weight in (None, "balanced"):
                grid.append(
                    {
                        "model": "random_forest",
                        "n_estimators": 500,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "class_weight": class_weight,
                    }
                )
    for n_estimators in (50, 100):
        for learning_rate in (0.05, 0.1):
            for max_depth in (1, 2):
                grid.append(
                    {
                        "model": "gradient_boosting",
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                    }
                )
    return grid


def selection_key(candidate: dict) -> tuple:
    """Highest inner Macro F1; ties go to the simpler model family, then lower complexity."""
    configuration = candidate["configuration"]
    model = configuration["model"]
    if model in ("logistic_regression", "rbf_svm"):
        complexity = (float(configuration["C"]),)
    elif model == "random_forest":
        depth = configuration["max_depth"]
        complexity = (float("inf") if depth is None else float(depth), float(configuration["min_samples_leaf"]))
    else:
        complexity = (
            float(configuration["n_estimators"]),
            float(configuration["max_depth"]),
            float(configuration["learning_rate"]),
        )
    return (-candidate["mean_macro_f1"], MODEL_PREFERENCE[model], complexity, candidate["grid_index"])


def tune_outer_fold(features, nested_folds, outer_fold: int) -> dict:
    columns = feature_columns()
    candidates = []
    for grid_index, configuration in enumerate(candidate_configurations()):
        scores = []
        for assignment in inner_assignments(nested_folds, outer_fold):
            train = features.loc[features["ID"].isin(assignment["inner_train_ids"])]
            validation = features.loc[features["ID"].isin(assignment["inner_validation_ids"])]
            pipeline = build_pipeline(configuration).fit(train[columns], train["target"])
            scores.append(macro_f1(validation["target"], pipeline.predict(validation[columns])))
        candidates.append(
            {
                "configuration": configuration,
                "inner_scores": scores,
                "mean_macro_f1": float(sum(scores) / len(scores)),
                "grid_index": grid_index,
            }
        )
    return min(candidates, key=selection_key)


def majority_baseline(features, outer_folds) -> pd.DataFrame:
    columns = feature_columns()
    rows = []
    for outer_fold in sorted(outer_folds["outer_fold"].unique()):
        train, test = outer_split(features, outer_folds, outer_fold)
        model = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(train[columns], train["target"])
        metrics = classification_metrics(
            test["target"], model.predict(test[columns]), model.predict_proba(test[columns])[:, 1]
        )
        rows.append({"outer_fold": int(outer_fold), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    features, outer_folds, nested_folds = load_inputs()
    check_folds(features, outer_folds, nested_folds)
    columns = feature_columns()

    fold_rows, prediction_frames, selections = [], [], []
    for outer_fold in sorted(outer_folds["outer_fold"].unique()):
        best = tune_outer_fold(features, nested_folds, outer_fold)
        train, test = outer_split(features, outer_folds, outer_fold)
        predicted, scores = fit_and_score(best["configuration"], train, test, columns)
        fold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "training_participants": len(train),
                "test_participants": len(test),
                **classification_metrics(test["target"], predicted, scores),
                "configuration": best["configuration"],
                "selected_inner_macro_f1": best["mean_macro_f1"],
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "participant_id": test["ID"].to_numpy(),
                    "outer_fold": int(outer_fold),
                    "true_label": test["target"].to_numpy(dtype=int),
                    "predicted_label": predicted.astype(int),
                    "score": scores.astype(float),
                }
            )
        )
        selections.append(
            {
                "outer_fold": int(outer_fold),
                "configuration": best["configuration"],
                "mean_inner_macro_f1": best["mean_macro_f1"],
                "inner_scores": best["inner_scores"],
            }
        )
        print(f"Outer fold {outer_fold}: {best['configuration']}  outer Macro F1 {fold_rows[-1]['macro_f1']:.4f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "task_count": 8,
        "predictor_count": len(columns),
        "outer_folds": len(fold_metrics),
        "metrics": {name: mean_std(fold_metrics[name]) for name in METRICS},
    }
    write_csv(fold_metrics, "full_eight_task_fold_metrics.csv")
    write_csv(pd.concat(prediction_frames, ignore_index=True), "full_eight_task_predictions.csv")
    (GENERATED / "full_eight_task_selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    write_json(GENERATED / "full_eight_task_summary.json", summary)

    majority = majority_baseline(features, outer_folds)
    majority_summary = {
        "strategy": "most_frequent",
        "outer_folds": len(majority),
        "metrics": {name: mean_std(majority[name]) for name in METRICS},
    }
    write_csv(majority, "majority_baseline_fold_metrics.csv")
    write_json(GENERATED / "majority_baseline_summary.json", majority_summary)

    write_json(
        EXPERIMENTS / "core_baseline.json",
        {
            "experiment": "core_eight_task_baseline",
            "population": {
                "als_participants": len(features),
                "normal_speech": int(features["target"].eq(0).sum()),
                "documented_speech_impairment": int(features["target"].eq(1).sum()),
            },
            "predictors": {"tasks": 8, "features_per_task": 5, "total_features": len(columns)},
            "evaluation": {
                "outer_folds": 5,
                "inner_folds": 4,
                "random_seed": SEED,
                "primary_metric": "macro_f1",
                "candidate_configurations": len(candidate_configurations()),
                "participant_disjoint": True,
            },
            "majority_class_baseline": {
                "macro_f1_mean": majority_summary["metrics"]["macro_f1"]["mean"],
                "balanced_accuracy_mean": majority_summary["metrics"]["balanced_accuracy"]["mean"],
                "roc_auc_mean": majority_summary["metrics"]["roc_auc"]["mean"],
            },
            "full_eight_task_baseline": {
                "macro_f1_mean": summary["metrics"]["macro_f1"]["mean"],
                "macro_f1_std": summary["metrics"]["macro_f1"]["std"],
            },
            "outer_fold_selected_models": {str(row["outer_fold"]): row["configuration"] for row in selections},
        },
    )
    print(f"Full eight-task Macro F1: {summary['metrics']['macro_f1']['mean']:.4f}")
    print(f"Majority baseline Macro F1: {majority_summary['metrics']['macro_f1']['mean']:.4f}")


if __name__ == "__main__":
    main()

"""Nested task selection with the handcrafted acoustic features.

Inside each outer fold, every candidate task subset is fit on each inner
training split with that fold's frozen model configuration. The smallest
subset whose mean inner Macro F1 is within `delta` of the reference subset is
then refit on the full outer training split and scored once on the outer test.
"""

from __future__ import annotations

import pandas as pd

from common import (
    DELTA,
    METRICS,
    TASKS,
    classification_metrics,
    feature_columns,
    fit_and_score,
    inner_assignments,
    macro_f1,
    mean_std,
    outer_split,
    build_pipeline,
)


def check_folds(features: pd.DataFrame, outer_folds: pd.DataFrame, nested_folds: list[dict]) -> None:
    """Fail if any participant leaks between inner train, inner validation and outer test."""
    ids = set(features["ID"])
    if ids != set(outer_folds["participant_id"]) or outer_folds["participant_id"].duplicated().any():
        raise ValueError("Feature table and outer folds do not cover the same participants once each.")
    for row in nested_folds:
        train, validation, test = (set(row[key]) for key in ("inner_train_ids", "inner_validation_ids", "outer_test_ids"))
        expected_test = set(outer_folds.loc[outer_folds["outer_fold"].eq(row["outer_fold"]), "participant_id"])
        if train & validation or train & test or validation & test:
            raise ValueError(f"Participant leakage in outer fold {row['outer_fold']}, inner fold {row['inner_fold']}.")
        if test != expected_test or train | validation != ids - test:
            raise ValueError(f"Nested folds disagree with outer fold {row['outer_fold']}.")


def inner_predictions(features, nested_folds, subsets, configurations) -> pd.DataFrame:
    """Inner-validation predictions for every outer fold, subset and inner fold."""
    frames = []
    for outer_fold, configuration in sorted(configurations.items()):
        assignments = inner_assignments(nested_folds, outer_fold)
        for tasks in subsets:
            columns = feature_columns(tasks)
            for assignment in assignments:
                train = features.loc[features["ID"].isin(assignment["inner_train_ids"])]
                validation = features.loc[features["ID"].isin(assignment["inner_validation_ids"])]
                pipeline = build_pipeline(configuration).fit(train[columns], train["target"])
                frames.append(
                    pd.DataFrame(
                        {
                            "participant_id": validation["ID"].to_numpy(),
                            "outer_fold": outer_fold,
                            "inner_fold": assignment["inner_fold"],
                            "tasks": ",".join(tasks),
                            "subset_id": "+".join(tasks),
                            "task_count": len(tasks),
                            "predictor_count": len(columns),
                            "true_label": validation["target"].to_numpy(dtype=int),
                            "predicted_label": pipeline.predict(validation[columns]).astype(int),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def inner_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per outer fold and subset: Macro F1 on each inner fold, plus mean and std."""
    records = []
    for (outer_fold, subset_id), group in predictions.groupby(["outer_fold", "subset_id"], sort=False):
        scores = [macro_f1(fold["true_label"], fold["predicted_label"]) for _, fold in group.groupby("inner_fold")]
        summary = mean_std(scores)
        first = group.iloc[0]
        records.append(
            {
                "outer_fold": outer_fold,
                "tasks": first["tasks"],
                "subset_id": subset_id,
                "task_count": int(first["task_count"]),
                "predictor_count": int(first["predictor_count"]),
                "inner_macro_f1_mean": summary["mean"],
                "inner_macro_f1_std": summary["std"],
                **{f"inner_fold_{index}_macro_f1": score for index, score in enumerate(scores)},
            }
        )
    return pd.DataFrame(records)


def select_subsets(inner: pd.DataFrame, delta: float = DELTA, reference: tuple[str, ...] = TASKS):
    """Pick the competitive subset per outer fold.

    Competitive means a mean inner Macro F1 of at least reference - delta. Ties
    break toward fewer tasks, a higher mean, a lower std, then task order.
    Returns the inner table annotated with the selection and one row per fold.
    """
    reference_id = "+".join(reference)
    annotated, selections = [], []
    for outer_fold, rows in inner.groupby("outer_fold", sort=True):
        rows = rows.copy()
        full = float(rows.loc[rows["subset_id"].eq(reference_id), "inner_macro_f1_mean"].iloc[0])
        threshold = full - delta
        competitive = rows.loc[rows["inner_macro_f1_mean"].ge(threshold)]
        chosen = min(
            competitive.itertuples(index=False),
            key=lambda row: (row.task_count, -row.inner_macro_f1_mean, row.inner_macro_f1_std, tuple(row.tasks.split(","))),
        )
        rows["delta"] = float(delta)
        rows["inner_full_reference"] = full
        rows["competitive_threshold"] = threshold
        rows["competitive"] = rows["inner_macro_f1_mean"].ge(threshold)
        rows["selected"] = rows["subset_id"].eq(chosen.subset_id)
        annotated.append(rows)
        selections.append(
            {
                "outer_fold": int(outer_fold),
                "selected_subset": chosen.subset_id,
                "selected_tasks": chosen.tasks,
                "task_count": int(chosen.task_count),
                "predictor_count": int(chosen.predictor_count),
                "selected_inner_macro_f1_mean": float(chosen.inner_macro_f1_mean),
                "selected_inner_macro_f1_std": float(chosen.inner_macro_f1_std),
                "inner_full_reference": full,
                "competitive_threshold": threshold,
                "competitive_subset_count": len(competitive),
                "delta": float(delta),
            }
        )
    return pd.concat(annotated, ignore_index=True), pd.DataFrame(selections)


def evaluate_selections(features, outer_folds, selections: pd.DataFrame, configurations):
    """Refit each fold's selected subset on outer training data and score the outer test."""
    metric_rows, prediction_frames = [], []
    for selection in selections.to_dict(orient="records"):
        outer_fold = selection["outer_fold"]
        configuration = configurations[outer_fold]
        columns = feature_columns(tuple(selection["selected_tasks"].split(",")))
        train, test = outer_split(features, outer_folds, outer_fold)
        predicted, scores = fit_and_score(configuration, train, test, columns)
        metrics = classification_metrics(test["target"], predicted, scores)
        metric_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_subset": selection["selected_subset"],
                "selected_tasks": selection["selected_tasks"],
                "task_count": selection["task_count"],
                "predictor_count": len(columns),
                "training_participants": len(train),
                "test_participants": len(test),
                "inner_full_reference": selection["inner_full_reference"],
                "selected_inner_macro_f1_mean": selection["selected_inner_macro_f1_mean"],
                "selected_inner_macro_f1_std": selection["selected_inner_macro_f1_std"],
                "competitive_threshold": selection["competitive_threshold"],
                "competitive_subset_count": selection["competitive_subset_count"],
                "delta": selection["delta"],
                **{f"outer_test_{name}": metrics[name] for name in METRICS},
                "configuration": configuration,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "participant_id": test["ID"].to_numpy(),
                    "outer_fold": outer_fold,
                    "selected_subset": selection["selected_subset"],
                    "true_label": test["target"].to_numpy(dtype=int),
                    "predicted_label": predicted.astype(int),
                    "score": scores.astype(float),
                }
            )
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def run_nested_selection(features, outer_folds, nested_folds, subsets, configurations, reference=TASKS):
    """Full nested procedure. Returns inner predictions, annotated inner results, fold metrics, outer predictions."""
    check_folds(features, outer_folds, nested_folds)
    predictions = inner_predictions(features, nested_folds, subsets, configurations)
    inner, selections = select_subsets(inner_scores(predictions), DELTA, reference)
    fold_metrics, outer_predictions = evaluate_selections(features, outer_folds, selections, configurations)
    return predictions, inner, fold_metrics, outer_predictions


def outer_metric_summary(fold_metrics: pd.DataFrame) -> dict:
    return {name: mean_std(fold_metrics[f"outer_test_{name}"]) for name in METRICS}

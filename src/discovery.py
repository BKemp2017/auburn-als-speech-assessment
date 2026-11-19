"""Exhaustive 255-subset search and the paired bootstrap of its standout protocols.

This is a discovery analysis: subsets are compared on outer-test scores, so
the best subset's score is optimistic. nested_selection.py gives the
leakage-safe estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BOOTSTRAP_REPLICATES,
    DELTA,
    EXPERIMENTS,
    METRICS,
    SEED,
    TASKS,
    all_subsets,
    classification_metrics,
    feature_columns,
    fit_and_score,
    frozen_configurations,
    load_inputs,
    macro_f1,
    outer_split,
    paired_fold_bootstrap,
    write_csv,
    write_json,
)

TOLERANCES = (0.01, 0.02, 0.05)
BOOTSTRAP_PROTOCOLS = {"TA": ("TA",), "E+TA+KA": ("E", "TA", "KA"), "FULL": TASKS}


def evaluate_all_subsets(features, outer_folds, configurations):
    metric_rows, prediction_frames = [], []
    for tasks in all_subsets():
        columns = feature_columns(tasks)
        for outer_fold, configuration in sorted(configurations.items()):
            train, test = outer_split(features, outer_folds, outer_fold)
            predicted, scores = fit_and_score(configuration, train, test, columns)
            metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "tasks": ",".join(tasks),
                    "task_count": len(tasks),
                    "predictor_count": len(columns),
                    **classification_metrics(test["target"], predicted, scores),
                    "subset_id": "+".join(tasks),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "participant_id": test["ID"].to_numpy(),
                        "outer_fold": outer_fold,
                        "subset_id": "+".join(tasks),
                        "true_label": test["target"].to_numpy(dtype=int),
                        "predicted_label": predicted.astype(int),
                    }
                )
            )
    fold_metrics = pd.DataFrame(metric_rows)
    summary = (
        fold_metrics.groupby("subset_id", sort=False)
        .agg(
            tasks=("tasks", "first"),
            task_count=("task_count", "first"),
            predictor_count=("predictor_count", "first"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )
    return fold_metrics, summary, pd.concat(prediction_frames, ignore_index=True)


def ranked(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["task_count", "macro_f1_mean", "macro_f1_std", "subset_id"], ascending=[True, False, True, True]
    )


def best_by_task_count(summary: pd.DataFrame) -> pd.DataFrame:
    return ranked(summary).groupby("task_count", as_index=False, sort=True).first()


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """Best subset at each task count, kept only if it beats every smaller count."""
    best = best_by_task_count(summary)
    return best.loc[best["macro_f1_mean"] > best["macro_f1_mean"].cummax().shift(fill_value=-np.inf)]


def subset_entry(row) -> dict:
    return {"tasks": row["tasks"].split(","), "task_count": int(row["task_count"]), "macro_f1_mean": float(row["macro_f1_mean"])}


def bootstrap_protocols(predictions: pd.DataFrame):
    selected = pd.concat(
        [
            predictions.loc[predictions["subset_id"].eq("+".join(tasks))].assign(protocol=name)
            for name, tasks in BOOTSTRAP_PROTOCOLS.items()
        ],
        ignore_index=True,
    )[["participant_id", "outer_fold", "protocol", "true_label", "predicted_label"]]

    names = list(BOOTSTRAP_PROTOCOLS)
    fold_tables = [
        rows.pivot(index=["participant_id", "true_label"], columns="protocol", values="predicted_label").reset_index()
        for _, rows in selected.groupby("outer_fold", sort=True)
    ]
    draws = paired_fold_bootstrap(fold_tables, names)
    boot = pd.DataFrame(draws, columns=names)
    boot.insert(0, "bootstrap_id", np.arange(BOOTSTRAP_REPLICATES))
    boot["TA_minus_FULL"] = boot["TA"] - boot["FULL"]
    boot["E+TA+KA_minus_FULL"] = boot["E+TA+KA"] - boot["FULL"]
    boot["E+TA+KA_minus_TA"] = boot["E+TA+KA"] - boot["TA"]

    points = {
        name: float(
            np.mean(
                [
                    macro_f1(rows["true_label"], rows["predicted_label"])
                    for _, rows in selected.loc[selected["protocol"].eq(name)].groupby("outer_fold", sort=True)
                ]
            )
        )
        for name in names
    }
    points["TA_minus_FULL"] = points["TA"] - points["FULL"]
    points["E+TA+KA_minus_FULL"] = points["E+TA+KA"] - points["FULL"]
    points["E+TA+KA_minus_TA"] = points["E+TA+KA"] - points["TA"]
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "point_estimate": point,
                "bootstrap_mean": float(boot[metric].mean()),
                "ci_95_lower": float(np.quantile(boot[metric], 0.025)),
                "ci_95_upper": float(np.quantile(boot[metric], 0.975)),
            }
            for metric, point in points.items()
        ]
    )
    return selected, boot, summary


def main() -> None:
    features, outer_folds, _ = load_inputs()
    configurations = frozen_configurations()
    fold_metrics, summary, predictions = evaluate_all_subsets(features, outer_folds, configurations)
    write_csv(fold_metrics, "task_subset_fold_metrics.csv")
    write_csv(summary, "task_subset_summary.csv")

    full = float(summary.loc[summary["task_count"].eq(8), "macro_f1_mean"].iloc[0])
    competitive = ranked(summary.loc[summary["macro_f1_mean"].ge(full - DELTA)])
    top25 = summary.sort_values("macro_f1_mean", ascending=False).head(25)
    top25_counts = {task: int(top25["tasks"].str.split(",").map(lambda tasks: task in tasks).sum()) for task in TASKS}
    write_json(
        EXPERIMENTS / "exhaustive_task_search.json",
        {
            "experiment": "exhaustive_task_subset_search",
            "research_question": "How much structured speech is needed to retain predictive information about documented speech impairment among participants already diagnosed with ALS?",
            "task_search": {
                "available_tasks": list(TASKS),
                "nonempty_subsets": len(summary),
                "evaluations": len(fold_metrics),
                "outer_folds_per_subset": len(configurations),
            },
            "evaluation": {
                "primary_metric": "macro_f1",
                "full_reference_macro_f1": full,
                "primary_tolerance": DELTA,
                "sensitivity_tolerances": [t for t in TOLERANCES if t != DELTA],
                "model_configuration": "Frozen independently within each outer fold from the complete eight-task inner-CV baseline",
            },
            "descriptive_results": {
                "minimal_competitive_subset": subset_entry(competitive.iloc[0]),
                "best_observed_subset": subset_entry(summary.loc[summary["macro_f1_mean"].idxmax()]),
                "pareto_frontier": [subset_entry(row) for _, row in pareto_frontier(summary).iterrows()],
                "competitive_subset_counts": {
                    f"delta_{t:.2f}": int(summary["macro_f1_mean"].ge(full - t).sum()) for t in TOLERANCES
                },
                "top_25_task_frequency": dict(sorted(top25_counts.items(), key=lambda item: -item[1])),
            },
            "interpretation_status": "Discovery analysis. Subsets were compared on outer-test performance, so the best observed subset is optimistic; the nested task-selection analysis gives the leakage-safe estimate.",
        },
    )

    protocol_predictions, boot, boot_summary = bootstrap_protocols(predictions)
    write_csv(protocol_predictions, "bootstrap_protocol_predictions.csv")
    write_csv(boot, "bootstrap_task_comparisons.csv")
    write_csv(boot_summary, "bootstrap_task_comparison_summary.csv")
    by_metric = boot_summary.set_index("metric")
    write_json(
        EXPERIMENTS / "bootstrap_task_comparisons.json",
        {
            "experiment": "paired_bootstrap_task_comparisons",
            "population": {
                "als_participants": len(features),
                "normal_speech": int(features["target"].eq(0).sum()),
                "documented_speech_impairment": int(features["target"].eq(1).sum()),
            },
            "protocols": {name: list(tasks) for name, tasks in BOOTSTRAP_PROTOCOLS.items()},
            "evaluation": {
                "primary_metric": "macro_f1",
                "outer_folds": len(configurations),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "random_seed": SEED,
                "resampling_unit": "participant",
                "resampling_design": "paired within outer fold with replacement",
                "confidence_interval": "95% percentile bootstrap",
            },
            "observed_results": {
                name: {
                    "macro_f1": by_metric.loc[name, "point_estimate"],
                    "ci_95": [by_metric.loc[name, "ci_95_lower"], by_metric.loc[name, "ci_95_upper"]],
                }
                for name in BOOTSTRAP_PROTOCOLS
            },
            "paired_differences": {
                name: {
                    "point_estimate": by_metric.loc[name, "point_estimate"],
                    "ci_95": [by_metric.loc[name, "ci_95_lower"], by_metric.loc[name, "ci_95_upper"]],
                }
                for name in ("TA_minus_FULL", "E+TA+KA_minus_FULL", "E+TA+KA_minus_TA")
            },
            "interpretation_status": "Post-selection discovery uncertainty. These bootstrap intervals condition on the selected protocols and do not remove task-selection optimism.",
        },
    )

    print(f"Full eight-task Macro F1: {full:.4f}")
    print(best_by_task_count(summary)[["task_count", "subset_id", "macro_f1_mean"]].to_string(index=False))
    print(boot_summary.to_string(index=False))


if __name__ == "__main__":
    main()

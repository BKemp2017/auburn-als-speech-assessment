"""Participant-resampled stability of nested task selection, and fixed-protocol robustness.

Both analyses resample participants within each inner validation fold
(10,000 replicates per outer fold, seed 2026) and rescore the saved
inner-validation predictions; no models are refit. Stability reruns the
selection rule on every replicate. Robustness asks whether each fixed
reduced subset stays within the 0.02 margin of the full protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BOOTSTRAP_REPLICATES,
    DELTA,
    EXPERIMENTS,
    FULL_ID,
    GENERATED,
    SEED,
    TASKS,
    all_subsets,
    write_csv,
    write_json,
)

KEY_CANDIDATES = ["KA", "TA", "PA", "O+TA", "PA+KA", "TA+KA", "A+PA+KA", "O+TA+KA"]


def weighted_macro_f1(weights: np.ndarray, true_labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Macro F1 for every (replicate, subset) pair, with participants counted by weight.

    weights: (replicates, participants); predicted: (subsets, participants).
    """
    positive, negative = true_labels == 1, true_labels == 0
    tp = weights[:, positive] @ predicted[:, positive].T
    fp = weights[:, negative] @ predicted[:, negative].T
    fn = weights[:, positive].sum(axis=1, keepdims=True) - tp
    tn = weights[:, negative].sum(axis=1, keepdims=True) - fp
    positive_denominator = 2.0 * tp + fp + fn
    negative_denominator = 2.0 * tn + fp + fn
    positive_f1 = np.divide(2.0 * tp, positive_denominator, out=np.zeros_like(tp), where=positive_denominator != 0)
    negative_f1 = np.divide(2.0 * tn, negative_denominator, out=np.zeros_like(tn), where=negative_denominator != 0)
    return (positive_f1 + negative_f1) / 2.0


def resampled_inner_scores(predictions: pd.DataFrame, outer_fold: int, subset_ids: list[str], rng):
    """Bootstrap and observed Macro F1 per inner fold and subset for one outer fold.

    Returns arrays of shape (replicates, 4, subsets) and (4, subsets).
    """
    fold = predictions.loc[predictions["outer_fold"].eq(outer_fold)]
    participants = (
        fold[["participant_id", "inner_fold", "true_label"]].drop_duplicates().sort_values(["inner_fold", "participant_id"])
    )
    predicted = (
        fold.pivot(index="subset_id", columns="participant_id", values="predicted_label")
        .reindex(index=subset_ids, columns=participants["participant_id"])
        .to_numpy(dtype=float)
    )
    true_labels = participants["true_label"].to_numpy(dtype=int)
    inner_folds = participants["inner_fold"].to_numpy(dtype=int)
    boot = np.empty((BOOTSTRAP_REPLICATES, 4, len(subset_ids)))
    observed = np.empty((4, len(subset_ids)))
    for inner_fold in range(4):
        mask = inner_folds == inner_fold
        count = int(mask.sum())
        draws = rng.integers(0, count, size=(BOOTSTRAP_REPLICATES, count))
        weights = np.zeros((BOOTSTRAP_REPLICATES, count))
        np.add.at(weights, (np.repeat(np.arange(BOOTSTRAP_REPLICATES), count), draws.reshape(-1)), 1.0)
        boot[:, inner_fold, :] = weighted_macro_f1(weights, true_labels[mask], predicted[:, mask])
        observed[inner_fold, :] = weighted_macro_f1(np.ones((1, count)), true_labels[mask], predicted[:, mask])[0]
    return boot, observed


def bootstrap_selections(boot: np.ndarray, subsets, outer_fold: int) -> pd.DataFrame:
    """Apply the nested selection rule to every bootstrap replicate at once."""
    subset_ids = ["+".join(tasks) for tasks in subsets]
    task_counts = np.array([len(tasks) for tasks in subsets])
    lexicographic_rank = np.empty(len(subsets), dtype=int)
    lexicographic_rank[sorted(range(len(subsets)), key=lambda i: subsets[i])] = np.arange(len(subsets))

    means = boot.mean(axis=1)
    stds = boot.std(axis=1, ddof=1)
    reference = means[:, subset_ids.index(FULL_ID)]
    eligible = means >= (reference - DELTA)[:, None]
    competitive_count = eligible.sum(axis=1)
    eligible &= task_counts == np.where(eligible, task_counts, np.inf).min(axis=1, keepdims=True)
    eligible &= means == np.where(eligible, means, -np.inf).max(axis=1, keepdims=True)
    eligible &= stds == np.where(eligible, stds, np.inf).min(axis=1, keepdims=True)
    chosen = np.where(eligible, lexicographic_rank, len(subsets)).argmin(axis=1)

    rows = np.arange(BOOTSTRAP_REPLICATES)
    return pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "bootstrap_replicate": rows,
            "delta": DELTA,
            "selected_subset": np.array(subset_ids, dtype=object)[chosen],
            "selected_tasks": np.array([",".join(t) for t in subsets], dtype=object)[chosen],
            "task_count": task_counts[chosen],
            "selected_inner_macro_f1_mean": means[rows, chosen],
            "selected_inner_macro_f1_std": stds[rows, chosen],
            "inner_full_reference": reference,
            "competitive_threshold": reference - DELTA,
            "competitive_subset_count": competitive_count,
        }
    )


def robustness(boot: np.ndarray, observed: np.ndarray, subsets, outer_fold: int) -> pd.DataFrame:
    """Paired difference of each subset against the full protocol."""
    full_index = [len(tasks) for tasks in subsets].index(len(TASKS))
    boot_means = boot.mean(axis=1)
    observed_means = observed.mean(axis=0)
    differences = boot_means - boot_means[:, [full_index]]
    lower, upper = np.quantile(differences, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "tasks": [",".join(t) for t in subsets],
            "subset_id": ["+".join(t) for t in subsets],
            "task_count": [len(t) for t in subsets],
            "observed_subset_macro_f1_mean": observed_means,
            "observed_full_macro_f1_mean": float(observed_means[full_index]),
            "observed_difference": observed_means - observed_means[full_index],
            "bootstrap_difference_mean": differences.mean(axis=0),
            "bootstrap_difference_std": differences.std(axis=0, ddof=1),
            "difference_ci_lower": lower,
            "difference_ci_upper": upper,
            "noninferiority_probability": (differences >= -DELTA).mean(axis=0),
            "superiority_probability": (differences > 0.0).mean(axis=0),
            "supports_noninferiority": lower >= -DELTA,
            "delta": DELTA,
        }
    )


def frequency_records(subset_ids: pd.Series) -> list[dict]:
    counts = subset_ids.value_counts()
    return [
        {"subset_id": s, "tasks": s.split("+"), "selection_count": int(c), "selection_rate": float(c / counts.sum())}
        for s, c in counts.items()
    ]


def task_frequency(selected_tasks: pd.Series) -> dict:
    counts = selected_tasks.str.split(",").explode().value_counts().reindex(TASKS, fill_value=0)
    return {t: {"selection_count": int(counts[t]), "selection_rate": float(counts[t] / len(selected_tasks))} for t in TASKS}


def task_count_distribution(task_counts: pd.Series) -> dict:
    counts = task_counts.value_counts().sort_index()
    return {str(int(k)): {"selection_count": int(v), "selection_rate": float(v / counts.sum())} for k, v in counts.items()}


def stability_record(selections: pd.DataFrame, inner: pd.DataFrame, subset_count: int, participants: int) -> dict:
    original = inner.loc[inner["selected"]].set_index("outer_fold")
    per_fold = []
    for outer_fold, rows in selections.groupby("outer_fold", sort=True):
        chosen = original.loc[outer_fold]
        original_tasks = set(chosen["tasks"].split(","))
        jaccard = rows["selected_tasks"].map(
            lambda tasks: len(set(tasks.split(",")) & original_tasks) / len(set(tasks.split(",")) | original_tasks)
        )
        frequencies = rows["selected_subset"].value_counts(normalize=True)
        entropy = float(-np.sum(frequencies * np.log(frequencies)))
        per_fold.append(
            {
                "outer_fold": int(outer_fold),
                "original_selection": {
                    "subset_id": chosen["subset_id"],
                    "tasks": chosen["tasks"].split(","),
                    "task_count": int(chosen["task_count"]),
                },
                "original_selection_retention_rate": float(rows["selected_subset"].eq(chosen["subset_id"]).mean()),
                "mean_jaccard_similarity_to_original": float(jaccard.mean()),
                "unique_selected_protocols": int(rows["selected_subset"].nunique()),
                "selection_entropy": entropy,
                "normalized_selection_entropy": float(entropy / np.log(subset_count)),
                "mean_selected_task_count": float(rows["task_count"].mean()),
                "exact_protocol_frequency": frequency_records(rows["selected_subset"]),
                "task_frequency": task_frequency(rows["selected_tasks"]),
                "task_count_distribution": task_count_distribution(rows["task_count"]),
            }
        )
    full_count = int(selections["selected_subset"].eq(FULL_ID).sum())
    return {
        "experiment": "participant_resampled_protocol_stability",
        "population": {"als_participants": participants},
        "evaluation": {
            "outer_folds": len(per_fold),
            "inner_folds_per_outer_fold": 4,
            "task_subsets": subset_count,
            "primary_metric": "macro_f1",
            "delta": DELTA,
            "bootstrap_replicates_per_outer_fold": BOOTSTRAP_REPLICATES,
            "total_bootstrap_selections": len(selections),
            "random_seed": SEED,
            "resampling_unit": "participant",
            "resampling_design": "paired within inner fold with replacement",
            "selection_uses_outer_test_results": False,
        },
        "overall_stability": {
            "mean_selected_task_count": float(selections["task_count"].mean()),
            "one_or_two_task_selection_rate": float(selections["task_count"].le(2).mean()),
            "one_to_three_task_selection_rate": float(selections["task_count"].le(3).mean()),
            "full_assessment_selection_count": full_count,
            "full_assessment_selection_rate": float(full_count / len(selections)),
            "unique_selected_protocols": int(selections["selected_subset"].nunique()),
            "exact_protocol_frequency": frequency_records(selections["selected_subset"]),
            "task_frequency": task_frequency(selections["selected_tasks"]),
            "task_count_distribution": task_count_distribution(selections["task_count"]),
        },
        "per_outer_fold": per_fold,
        "interpretation_status": "Participant resampling measures selection stability conditional on the frozen folds, fitted inner models, published acoustic features, and VOC-ALS cohort. Exact protocol instability does not invalidate a stable burden conclusion. These results are not external clinical validation and do not establish that one task is universally sufficient.",
    }


def candidate_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    summary = fold_results.groupby(["subset_id", "tasks", "task_count"], as_index=False).agg(
        outer_fold_count=("outer_fold", "nunique"),
        supporting_fold_count=("supports_noninferiority", "sum"),
        observed_difference_mean=("observed_difference", "mean"),
        observed_difference_std=("observed_difference", "std"),
        observed_difference_min=("observed_difference", "min"),
        observed_difference_max=("observed_difference", "max"),
        bootstrap_difference_mean=("bootstrap_difference_mean", "mean"),
        difference_ci_lower_min=("difference_ci_lower", "min"),
        difference_ci_lower_mean=("difference_ci_lower", "mean"),
        difference_ci_upper_max=("difference_ci_upper", "max"),
        noninferiority_probability_mean=("noninferiority_probability", "mean"),
        noninferiority_probability_min=("noninferiority_probability", "min"),
        superiority_probability_mean=("superiority_probability", "mean"),
    )
    by_subset = fold_results.groupby("subset_id")["observed_difference"]
    summary["point_noninferior_fold_count"] = summary["subset_id"].map(by_subset.apply(lambda d: int((d >= -DELTA).sum())))
    summary["above_full_fold_count"] = summary["subset_id"].map(by_subset.apply(lambda d: int((d > 0.0).sum())))
    summary["all_folds_support_noninferiority"] = summary["supporting_fold_count"].eq(5)
    summary["all_point_estimates_noninferior"] = summary["point_noninferior_fold_count"].eq(5)
    summary["is_reduced"] = summary["task_count"] < len(TASKS)
    reduced = summary.loc[summary["is_reduced"]].sort_values(
        ["point_noninferior_fold_count", "noninferiority_probability_min", "noninferiority_probability_mean",
         "task_count", "observed_difference_mean", "subset_id"],
        ascending=[False, False, False, True, False, True],
    )
    summary["candidate_rank"] = pd.array([pd.NA] * len(summary), dtype="Int64")
    summary.loc[reduced.index, "candidate_rank"] = np.arange(1, len(reduced) + 1)
    return summary.sort_values(
        ["is_reduced", "candidate_rank", "subset_id"], ascending=[False, True, True], na_position="last"
    ).reset_index(drop=True)


CANDIDATE_FIELDS = [
    "candidate_rank", "subset_id", "tasks", "task_count", "supporting_fold_count", "point_noninferior_fold_count",
    "above_full_fold_count", "observed_difference_mean", "observed_difference_min", "difference_ci_lower_min",
    "noninferiority_probability_mean", "noninferiority_probability_min",
]


def candidate_record(row) -> dict:
    record = {field: row[field] for field in CANDIDATE_FIELDS}
    record["candidate_rank"] = int(record["candidate_rank"])
    record["tasks"] = row["tasks"].split(",")
    return record


def robustness_record(fold_results: pd.DataFrame, summary: pd.DataFrame) -> dict:
    reduced = summary.loc[summary["is_reduced"]].sort_values("candidate_rank")
    reduced_folds = fold_results.loc[fold_results["task_count"].lt(len(TASKS))]
    supported = reduced_folds.loc[reduced_folds["supports_noninferiority"]]
    best = reduced.iloc[0]
    return {
        "experiment": "fixed_protocol_robustness",
        "analysis_role": "Exploratory cross-fold robustness assessment of fixed reduced speech-task protocols.",
        "primary_metric": "macro_f1",
        "reference_protocol": FULL_ID,
        "noninferiority_margin": DELTA,
        "bootstrap": {
            "replicates_per_outer_fold": BOOTSTRAP_REPLICATES,
            "random_seed": SEED,
            "resampling_unit": "participant",
            "resampling_scope": "Participants were resampled independently within each inner validation fold. Identical weights were applied to every task subset and the full protocol.",
            "confidence_interval": "Two-sided percentile 95% interval for the paired Macro F1 difference.",
        },
        "coverage": {
            "outer_folds": 5,
            "task_subsets": len(summary),
            "reduced_task_subsets": int(summary["is_reduced"].sum()),
            "fold_comparisons": len(fold_results),
            "reduced_fold_comparisons": len(reduced_folds),
        },
        "results": {
            "supported_reduced_comparisons": len(supported),
            "supported_reduced_comparisons_by_fold": {
                str(f): int(c) for f, c in supported.groupby("outer_fold").size().reindex(range(5), fill_value=0).items()
            },
            "reduced_protocols_supported_in_all_folds": int(reduced["all_folds_support_noninferiority"].sum()),
            "reduced_protocols_point_noninferior_in_all_folds": int(reduced["all_point_estimates_noninferior"].sum()),
        },
        "exploratory_best_fixed_candidate": candidate_record(best),
        "top_ranked_candidates": [candidate_record(row) for _, row in reduced.head(15).iterrows()],
        "key_candidates": [
            candidate_record(reduced.loc[reduced["subset_id"].eq(s)].iloc[0]) for s in KEY_CANDIDATES
        ],
        "ranking_rule": [
            "More outer folds with point differences inside the margin.",
            "Higher worst-fold noninferiority probability.",
            "Higher mean noninferiority probability.",
            "Fewer speech tasks.",
            "Higher mean paired Macro F1 difference.",
            "Lexicographic subset identifier as the final tie-break.",
        ],
        "interpretation": {
            "universal_fixed_reduced_protocol_supported": bool(reduced["all_folds_support_noninferiority"].any()),
            "best_exploratory_subset": best["subset_id"],
            "statement": "No fixed reduced protocol had all five point estimates inside the noninferiority margin, and no fixed reduced protocol had confidence-bound support in all five outer folds. The evidence supports a low-burden adaptive protocol policy rather than one universal task subset.",
            "limitation": "Candidate ranking is exploratory because the same inner out-of-fold predictions were used to compare and rank the fixed protocols.",
        },
        "participant_level_fields": [],
        "source_artifacts": [
            str(GENERATED / "protocol_stability_inner_predictions.csv"),
            str(GENERATED / "protocol_robustness_fold_results.csv"),
            str(GENERATED / "protocol_robustness_candidate_summary.csv"),
        ],
    }


def main() -> None:
    predictions = pd.read_csv(GENERATED / "protocol_stability_inner_predictions.csv")
    inner = pd.read_csv(GENERATED / "nested_task_selection_inner_results.csv")
    subsets = all_subsets()
    subset_ids = ["+".join(tasks) for tasks in subsets]

    # One generator for both analyses: they share the same bootstrap weights.
    rng = np.random.default_rng(SEED)
    selections, fold_results = [], []
    for outer_fold in range(5):
        boot, observed = resampled_inner_scores(predictions, outer_fold, subset_ids, rng)
        selections.append(bootstrap_selections(boot, subsets, outer_fold))
        fold_results.append(robustness(boot, observed, subsets, outer_fold))
    selections = pd.concat(selections, ignore_index=True)
    fold_results = pd.concat(fold_results, ignore_index=True).sort_values(["outer_fold", "task_count", "subset_id"])

    write_csv(selections, "protocol_stability_bootstrap_selections.csv")
    write_json(EXPERIMENTS / "protocol_stability.json", stability_record(selections, inner, len(subsets), predictions["participant_id"].nunique()))

    summary = candidate_summary(fold_results)
    write_csv(fold_results, "protocol_robustness_fold_results.csv")
    write_csv(summary, "protocol_robustness_candidate_summary.csv")
    record = robustness_record(fold_results, summary)
    write_json(EXPERIMENTS / "protocol_robustness.json", record, sort_keys=True)
    key = pd.DataFrame(record["key_candidates"]).sort_values("candidate_rank")
    key["tasks"] = key["tasks"].map("+".join)
    key[CANDIDATE_FIELDS].to_csv(EXPERIMENTS / "protocol_robustness_key_candidates.csv", index=False)

    counts = selections["task_count"]
    print(f"Selections: {len(selections)}; one task {counts.eq(1).mean():.2%}, <=2 {counts.le(2).mean():.2%}, <=3 {counts.le(3).mean():.2%}")
    print(f"Best fixed reduced candidate: {record['exploratory_best_fixed_candidate']['subset_id']}")


if __name__ == "__main__":
    main()

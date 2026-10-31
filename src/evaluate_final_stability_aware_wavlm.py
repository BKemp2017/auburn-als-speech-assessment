"""Final exploratory experiment: stability-aware WavLM speech-task selection.

Reuses the ORIGINAL 16-PC inner-CV table and frozen regularization choices.
The new rule uses ONLY inner-validation scores: among subsets with at most
three tasks whose mean inner Macro F1 is within 0.02 of the full assessment,
maximize the WORST paired inner-fold Macro F1 difference.

Then fit/evaluate once per original outer fold and run the SAME 10,000-
replicate paired bootstrap used in the primary WavLM analysis.

IMPORTANT: Outer-test results from this cohort have been inspected in the
project already. This experiment is exploratory, even if its confidence bound
happens to exceed the noninferiority boundary. Do not use it as independent
confirmatory evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_wavlm_task_selection import (
    run_paired_wavlm_bootstrap,
    summarize_paired_wavlm_bootstrap,
)
from evaluate_wavlm_task_selection import evaluate_wavlm_outer_folds
from wavlm_task_selection_core import (
    load_duplicate_filtered_design,
    load_wavlm_artifact,
    validate_wavlm_inputs,
)

ROOT = Path(".")
PRIMARY = ROOT / "experiments/wavlm_task_selection.json"
INNER = ROOT / "results/generated/wavlm_task_selection_inner_results.csv"
OUTPUT = ROOT / "results/generated"
MANIFEST = ROOT / "experiments/final_stability_aware_wavlm.json"
FIGURE = ROOT / "figures/final_stability_aware_noninferiority.pdf"

TASKS = ("A", "E", "I", "O", "U", "PA", "TA", "KA")
FULL_ID = "+".join(TASKS)
FOLDS = 5
INNER_FOLDS = 4
SUBSETS = 255
MARGIN = 0.02
MAX_TASKS = 3
INNER_COLUMNS = [f"inner_fold_{i}_macro_f1" for i in range(INNER_FOLDS)]


def load_and_validate() -> tuple[dict, pd.DataFrame]:
    if not PRIMARY.is_file():
        raise FileNotFoundError(f"Missing frozen WavLM record: {PRIMARY}")
    if not INNER.is_file():
        raise FileNotFoundError(
            f"Missing original inner-validation results: {INNER}\n"
            "Regenerate these only if necessary using: "
            "python src/evaluate_wavlm_task_selection.py"
        )
    original = json.loads(PRIMARY.read_text(encoding="utf-8"))
    design = original["design"]
    if (design["participants"], design["outer_folds"],
            design["task_subset_count"]) != (100, FOLDS, SUBSETS):
        raise ValueError("Unexpected participant/fold/subset design.")
    if design["primary_metric"] != "macro_f1":
        raise ValueError("Primary metric changed.")
    if not np.isclose(design["competitive_delta"], MARGIN, atol=1e-12):
        raise ValueError("Original margin changed.")
    if original["representation"]["pca_components_per_task"] != 16:
        raise ValueError("This experiment requires the locked 16-PC WavLM setup.")

    table = pd.read_csv(INNER)
    required = {
        "outer_fold", "subset_id", "task_count", "regularization_c",
        "inner_macro_f1_mean", "inner_macro_f1_std", *INNER_COLUMNS,
    }
    if not required.issubset(table.columns):
        raise ValueError(f"Missing inner-CV columns: {sorted(required - set(table.columns))}")
    if len(table) != FOLDS * SUBSETS:
        raise ValueError("Expected exactly 1,275 inner-CV rows.")
    if table.duplicated(["outer_fold", "subset_id"]).any():
        raise ValueError("Duplicate fold/subset combinations.")
    numeric = table[list(required - {"subset_id"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Inner-CV table contains non-finite values.")

    expected_regularizations = {
        int(r["outer_fold"]): float(r["regularization_c"])
        for r in original["classifier"]["selected_regularization"]
    }
    if set(expected_regularizations) != set(range(FOLDS)):
        raise ValueError("Missing frozen regularization values.")
    for fold in range(FOLDS):
        rows = table.loc[table["outer_fold"].eq(fold)]
        if len(rows) != SUBSETS or rows["subset_id"].nunique() != SUBSETS:
            raise ValueError(f"Fold {fold} does not contain 255 unique subsets.")
        if len(rows.loc[rows["subset_id"].eq(FULL_ID)]) != 1:
            raise ValueError(f"Missing full-reference result in fold {fold}.")
        if not np.allclose(rows["regularization_c"], expected_regularizations[fold]):
            raise ValueError(f"Fold {fold} regularization does not match the frozen model.")
        if not np.allclose(rows[INNER_COLUMNS].mean(axis=1),
                           rows["inner_macro_f1_mean"], atol=1e-9):
            raise ValueError(f"Fold {fold} inner means are inconsistent.")
    return original, table


def select_stability_aware(original: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Pick ONE protocol per fold without consulting any outer-test results."""
    selected = []
    regularizations = {
        int(r["outer_fold"]): float(r["regularization_c"])
        for r in original["classifier"]["selected_regularization"]
    }
    for fold in range(FOLDS):
        rows = table.loc[table["outer_fold"].eq(fold)].copy()
        full = rows.loc[rows["subset_id"].eq(FULL_ID)].iloc[0]
        full_scores = full[INNER_COLUMNS].to_numpy(dtype=float)
        ref_mean = float(full["inner_macro_f1_mean"])
        competitive = rows.loc[
            rows["task_count"].between(1, MAX_TASKS)
            & rows["inner_macro_f1_mean"].ge(ref_mean - MARGIN - 1e-12)
        ].copy()
        if competitive.empty:
            raise ValueError(
                f"No <= {MAX_TASKS}-task competitive subset in outer fold {fold}."
                " Do not silently substitute the full protocol."
            )
        differences = competitive[INNER_COLUMNS].to_numpy(dtype=float) - full_scores
        competitive["worst_inner_difference"] = differences.min(axis=1)
        competitive["mean_inner_difference"] = differences.mean(axis=1)
        # Maximin first, then favor fewer recordings, stronger mean, lower std.
        chosen = competitive.sort_values(
            by=["worst_inner_difference", "task_count",
                "inner_macro_f1_mean", "inner_macro_f1_std", "subset_id"],
            ascending=[False, True, False, True, True],
            kind="mergesort",
        ).iloc[0]
        subset = str(chosen["subset_id"])
        tasks = tuple(subset.split("+"))
        if not 1 <= len(tasks) <= MAX_TASKS or not set(tasks).issubset(TASKS):
            raise ValueError(f"Invalid selected subset: {subset}")
        selected.append({
            "outer_fold": fold,
            "selected_subset": subset,
            "selected_tasks": tasks,
            "task_count": len(tasks),
            "regularization_c": regularizations[fold],
            "selected_inner_macro_f1_mean": float(chosen["inner_macro_f1_mean"]),
            "selected_inner_macro_f1_std": float(chosen["inner_macro_f1_std"]),
            "inner_full_reference": ref_mean,
            "competitive_threshold": ref_mean - MARGIN,
            "competitive_subset_count": int(len(competitive)),
            "worst_inner_difference": float(chosen["worst_inner_difference"]),
            "delta": MARGIN,
        })
    return pd.DataFrame(selected)


def plot_comparison(original: dict, summary: dict) -> None:
    old = original["paired_bootstrap"]
    entries = [
        ("New stability-aware rule (exploratory)",
         summary["difference_point"],
         summary["difference_ci_95_lower"],
         summary["difference_ci_95_upper"]),
        ("Original WavLM rule (locked)",
         old["difference_point"],
         old["difference_ci_95_lower"],
         old["difference_ci_95_upper"]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for y, (_, point, lower, upper) in enumerate(entries[::-1]):
        ax.hlines(y, lower, upper, linewidth=3)
        ax.plot(point, y, "o")
    ax.axvline(-MARGIN, linestyle="--", label="Noninferiority boundary (−0.02)")
    ax.axvline(0, linestyle=":", label="Equal performance")
    ax.set_yticks([0, 1], [entries[1][0], entries[0][0]])
    ax.set_xlabel("Mean outer-fold Macro F1 difference (reduced − full)")
    ax.set_title("Paired 95% bootstrap intervals (same previously studied cohort)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original, inner_results = load_and_validate()
    selections = select_stability_aware(original, inner_results)
    print("Frozen stability-aware selections (inner-CV only):")
    print(selections[["outer_fold", "selected_subset", "task_count",
                      "worst_inner_difference"]].to_string(index=False))

    artifact, extraction_record = load_wavlm_artifact()
    labels, nested_folds, excluded_ids = load_duplicate_filtered_design()
    validate_wavlm_inputs(
        artifact=artifact, extraction_record=extraction_record,
        labels=labels, nested_folds=nested_folds, excluded_ids=excluded_ids,
    )
    fold_metrics, predictions = evaluate_wavlm_outer_folds(
        artifact=artifact, labels=labels, selections=selections,
        pca_components_per_task=16,
    )

    # Verify we really used the unchanged eight-task comparator.
    old_full = {
        int(x["outer_fold"]): float(x["outer_test_full_macro_f1"])
        for x in original["fold_results"]
    }
    for row in fold_metrics.itertuples(index=False):
        if not np.isclose(row.outer_test_full_macro_f1,
                          old_full[int(row.outer_fold)], atol=1e-6):
            raise ValueError("Full-assessment result differs from the frozen run.")

    boots = run_paired_wavlm_bootstrap(predictions)
    stats = summarize_paired_wavlm_bootstrap(
        bootstrap_results=boots, fold_metrics=fold_metrics, margin=MARGIN,
    )
    # This is an exploratory test because prior outer outcomes have been inspected.
    reduced_task_count = float(fold_metrics["task_count"].mean())
    public = {
        "experiment": "final_exploratory_stability_aware_wavlm_selection",
        "status": "EXPLORATORY_ONLY_NOT_INDEPENDENT_CONFIRMATION",
        "population": {"als_participants": 100, "excluded_duplicate_participants": 2},
        "design": {
            "outer_folds": FOLDS, "inner_folds": INNER_FOLDS, "pca_components_per_task": 16,
            "original_margin": MARGIN, "maximum_reduced_tasks": MAX_TASKS,
            "primary_metric": "mean_outer_fold_macro_f1",
            "fixed_selection_rule": (
                "Among 1-to-3-task subsets within 0.02 mean inner Macro F1 "
                "of full, maximize the worst of four inner-fold paired "
                "differences; tie-break by fewer tasks, higher inner mean, "
                "lower inner standard deviation, then subset identifier."
            ),
            "original_outer_folds_reused": True,
            "original_outer_test_results_previously_inspected": True,
        },
        "selection": {
            "subsets": selections["selected_subset"].tolist(),
            "task_counts": selections["task_count"].astype(int).tolist(),
            "mean_selected_task_count": reduced_task_count,
            "recording_count_reduction_percent": float((1 - reduced_task_count / 8) * 100),
            "new_protocol_recording_duration_not_measured": True,
        },
        "performance": {
            "selected_macro_f1": float(fold_metrics["outer_test_selected_macro_f1"].mean()),
            "full_macro_f1": float(fold_metrics["outer_test_full_macro_f1"].mean()),
            "original_locked_selected_macro_f1": float(
                original["outer_test_metrics"]["selected"]["macro_f1"]["mean"]
            ),
        },
        "paired_bootstrap": stats,
        "interpretation": (
            "Passing an exploratory bound does not establish confirmatory "
            "noninferiority on previously inspected outer-test participants. "
            "If the one-sided lower bound is at or below -0.02, even this "
            "exploratory analysis fails to support noninferiority."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selections.assign(selected_tasks=selections["selected_tasks"].map(
        lambda t: "+".join(t)
    )).to_csv(OUTPUT / "final_stability_aware_selections.csv", index=False)
    fold_metrics.to_csv(OUTPUT / "final_stability_aware_fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "final_stability_aware_predictions.csv", index=False)
    boots.to_csv(OUTPUT / "final_stability_aware_bootstrap.csv", index=False)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    plot_comparison(original, stats)
    print("\nFINAL RESULT (EXPLORATORY):")
    print(f"Selected Macro F1: {public['performance']['selected_macro_f1']:.6f}")
    print(f"Full Macro F1:     {public['performance']['full_macro_f1']:.6f}")
    print(f"Difference:        {stats['difference_point']:+.6f}")
    print(f"One-sided lower:   {stats['difference_one_sided_95_lower']:+.6f}")
    print(f"Mean tasks:        {reduced_task_count:.2f}/8")
    print(f"Exploratory bound above -0.02: "
          f"{stats['one_sided_bound_supports_noninferiority']}")
    print(f"Saved: {MANIFEST}, {FIGURE}, {OUTPUT}/final_stability_aware_*.csv")
    print("STOP: Do not tune this rule on these outer-test results.")


if __name__ == "__main__":
    main()

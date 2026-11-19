"""Paper figures, written to figures/ as PDF and 300-dpi PNG under the names the LaTeX source uses."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DELTA, EXPERIMENTS, FIGURES, GENERATED, TASKS

FULL_LABEL = "Full 8-task"
SELECTED_LABEL = "Nested-selected"


def load(name: str) -> dict:
    return json.loads((EXPERIMENTS / f"{name}.json").read_text())


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def y_grid(ax) -> None:
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def fold_bars(full, selected, title: str, note: str, name: str) -> None:
    x = np.arange(len(full))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, full, 0.36, label=FULL_LABEL)
    ax.bar(x + 0.18, selected, 0.36, label=SELECTED_LABEL)
    ax.set_xticks(x, [f"Fold {i}" for i in x])
    ax.set_ylim(0, max(max(full), max(selected)) + 0.15)
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    ax.text(0.5, 0.95, note, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.legend(loc="upper right")
    y_grid(ax)
    save(fig, name)


def discovery_by_task_count() -> None:
    summary = pd.read_csv(GENERATED / "task_subset_summary.csv")
    best = summary.sort_values("macro_f1_mean", ascending=False).groupby("task_count").first()
    full = float(summary.loc[summary["task_count"].eq(8), "macro_f1_mean"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best.index, best["macro_f1_mean"], marker="o")
    ax.axhline(full, color="C0", linestyle="--", label="Full 8-task reference")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0.46, 0.73)
    ax.set_xlabel("Number of speech tasks")
    ax.set_ylabel("Mean Macro F1")
    ax.set_title("Discovery Analysis: Best Observed Subset by Task Count")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "handcrafted_discovery_by_task_count")


def handcrafted_nested() -> None:
    comparison = load("nested_task_selection")["full_eight_task_comparison"]
    folds = pd.DataFrame(comparison["fold_comparisons"])
    fold_bars(
        folds["full_macro_f1"],
        folds["nested_selected_macro_f1"],
        "Handcrafted Features: Leakage-Safe Nested Selection",
        f"Mean: selected {comparison['nested_selected_macro_f1_mean']:.3f} vs full {comparison['full_macro_f1_mean']:.3f}",
        "handcrafted_nested_selected_vs_full",
    )


def stability_task_count() -> None:
    distribution = load("protocol_stability")["overall_stability"]["task_count_distribution"]
    rates = {int(k): v["selection_rate"] * 100 for k, v in distribution.items()}
    labels = ["1 task", "2 tasks", "3 tasks", "4–8 tasks"]
    values = [rates.get(1, 0), rates.get(2, 0), rates.get(3, 0), sum(v for k, v in rates.items() if k >= 4)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.bar_label(bars, [f"{v:.2f}%" for v in values], padding=3)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Selection frequency (%)")
    ax.set_title("Protocol Stability Across 50,000 Participant-Resampled Selections")
    y_grid(ax)
    save(fig, "protocol_stability_task_count")


def wavlm_nested() -> None:
    record = load("wavlm_task_selection")
    folds = pd.DataFrame(record["fold_results"]).sort_values("outer_fold")
    comparison = record["selected_vs_full"]
    fold_bars(
        folds["outer_test_full_macro_f1"],
        folds["outer_test_selected_macro_f1"],
        "WavLM: Leakage-Safe Nested Selection",
        f"Mean: selected {comparison['selected_macro_f1_mean']:.3f} vs full {comparison['full_macro_f1_mean']:.3f}\n"
        f"mean selected task count = {record['selection_summary']['mean_selected_task_count']:.1f}",
        "wavlm_selected_vs_full",
    )


def wavlm_noninferiority() -> None:
    bootstrap = load("wavlm_task_selection")["paired_bootstrap"]
    lower, upper = bootstrap["difference_ci_95_lower"], bootstrap["difference_ci_95_upper"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot([lower, upper], [1, 1], color="C0", linewidth=5, label="Two-sided 95% paired bootstrap interval")
    ax.plot(bootstrap["difference_point"], 1, "o", color="C0", markersize=10, label="Observed selected − full difference")
    ax.plot(bootstrap["difference_one_sided_95_lower"], 0.6, "|", color="C1", markersize=22, markeredgewidth=2, label="One-sided 95% lower bound")
    ax.axvline(-DELTA, color="C0", linestyle="--", label=f"Noninferiority boundary (−{DELTA})")
    ax.axvline(0, color="C0", linestyle=":", label="No difference")
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.set_xlim(min(lower, -DELTA) - 0.02, max(upper, 0) + 0.02)
    ax.set_xlabel("Macro F1 difference (nested-selected − full)")
    ax.set_title("Locked WavLM Noninferiority Analysis")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)
    save(fig, "wavlm_noninferiority_interval")


def wavlm_pca() -> None:
    comparison = pd.DataFrame(load("wavlm_pca_sensitivity")["comparison"]).sort_values("pca_components_per_task")
    x = comparison["pca_components_per_task"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, comparison["full_macro_f1_mean"], marker="o", label=FULL_LABEL)
    ax.plot(x, comparison["selected_macro_f1_mean"], marker="o", label=SELECTED_LABEL)
    for _, row in comparison.iterrows():
        ax.annotate(f"{row['mean_selected_task_count']:.1f} tasks", (row["pca_components_per_task"], row["selected_macro_f1_mean"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xticks(x)
    ax.set_ylim(0.58, 0.74)
    ax.set_xlabel("PCA components per task")
    ax.set_ylabel("Mean held-out Macro F1")
    ax.set_title("WavLM Representation-Dimension Sensitivity")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save(fig, "wavlm_pca_sensitivity")


def wavlm_tradeoff() -> None:
    primary = load("wavlm_task_selection")
    stability = load("wavlm_stability_aware_selection")
    full = primary["selected_vs_full"]["full_macro_f1_mean"]
    points = [
        ("Original nested\nWavLM policy", primary["selection_summary"]["mean_selected_task_count"], primary["selected_vs_full"]["selected_macro_f1_mean"]),
        ("Stability-aware\nfollow-up", stability["selection_summary"]["mean_selected_tasks"], stability["selected_vs_full"]["selected_macro_f1_mean"]),
        (FULL_LABEL, 8, full),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, tasks, score in points:
        ax.scatter(tasks, score, s=120, color="C0", zorder=3)
        ax.annotate(label, (tasks, score), xytext=(12, 8), textcoords="offset points")
    ax.axhline(full - DELTA, color="C0", linestyle="--", label=f"Full-assessment point margin (−{DELTA})")
    ax.set_xticks(range(1, 9))
    ax.set_xlim(0.6, 8.6)
    ax.set_ylim(0.60, 0.69)
    ax.set_xlabel("Mean tasks per participant")
    ax.set_ylabel("Mean held-out Macro F1")
    ax.set_title("WavLM Performance–Burden Tradeoff")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "wavlm_stability_aware_tradeoff")


def recording_burden() -> None:
    aggregate = load("recording_duration_burden")["aggregate_results"]
    measures = [("Recorded duration", "recorded_seconds"), ("Endpoint-trimmed", "endpoint_trimmed_seconds"), ("Nonsilent proxy", "nonsilent_seconds_proxy")]
    full = [aggregate[f"full_{key}_mean_per_participant"] for _, key in measures]
    selected = [aggregate[f"selected_{key}_mean_per_participant"] for _, key in measures]
    x = np.arange(len(measures))
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, values, label in ((-0.18, full, "Full 8-task assessment"), (0.18, selected, "Nested-selected policy")):
        bars = ax.bar(x + offset, values, 0.36, label=label)
        ax.bar_label(bars, [f"{v:.1f}s" for v in values], padding=4)
    ax.set_xticks(x, [label for label, _ in measures])
    ax.set_ylim(0, max(full) * 1.12)
    ax.set_ylabel("Mean seconds per participant")
    ax.set_title("Observed Recording-Time Burden")
    ax.legend(loc="upper right")
    y_grid(ax)
    save(fig, "recording_duration_burden")


def recording_by_task() -> None:
    tasks = pd.read_csv(GENERATED / "recording_duration_burden_by_task.csv").set_index("task").loc[list(TASKS)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(tasks.index, tasks["recorded_seconds_mean"])
    ax.bar_label(bars, [f"{v:.1f}" for v in tasks["recorded_seconds_mean"]], padding=3, fontsize=8)
    ax.set_xlabel("Speech task")
    ax.set_ylabel("Mean recorded seconds")
    ax.set_title("Observed Recording Duration by VOC-ALS Task")
    y_grid(ax)
    save(fig, "recording_duration_by_task")


def representation_comparison() -> None:
    handcrafted = load("duplicate_exclusion_sensitivity")["filtered_selected_vs_full"]
    wavlm = load("wavlm_task_selection")["selected_vs_full"]
    full = [handcrafted["full_macro_f1_mean"], wavlm["full_macro_f1_mean"]]
    selected = [handcrafted["nested_selected_macro_f1_mean"], wavlm["selected_macro_f1_mean"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.bar(x - 0.18, full, 0.36, label=FULL_LABEL)
    ax.bar(x + 0.18, selected, 0.36, label=SELECTED_LABEL)
    ax.set_xticks(x, ["Handcrafted", "WavLM"])
    ax.set_ylim(0, max(full + selected) + 0.09)
    ax.set_ylabel("Mean held-out Macro F1")
    ax.set_title("Duplicate-Filtered Same-Cohort Representation Comparison")
    ax.legend(loc="upper left")
    y_grid(ax)
    save(fig, "representation_comparison")


def main() -> None:
    discovery_by_task_count()
    handcrafted_nested()
    stability_task_count()
    wavlm_nested()
    wavlm_noninferiority()
    wavlm_pca()
    wavlm_tradeoff()
    recording_burden()
    recording_by_task()
    representation_comparison()
    print(f"Wrote 10 figures to {FIGURES}/")


if __name__ == "__main__":
    main()

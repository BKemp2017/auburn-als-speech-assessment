# Generate figures for the VOC-ALS task-subset experiment.

from __future__ import annotations

from audit_metadata import TASKS

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_task_subsets import (
    load_subset_summaries,
    task_frequency_analysis,
)


FIGURE_DIR = Path("figures")

TOP_N = 25

PRIMARY_TOLERANCE = 0.02


def save_figure(fig: plt.Figure,filename: str,
) -> None:
    # Save one figure as both PNG and PDF.

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def best_subset_per_task_count(results: pd.DataFrame,
) -> pd.DataFrame:
    # Return the best subset at each task count.

    ranked = results.sort_values(
        by=[
            "task_count",
            "macro_f1_mean",
            "macro_f1_std",
            "subset_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    return (
        ranked.groupby(
            "task_count",
            as_index=False,
            sort=True,
        )
        .first()
    )


def calculate_pareto_frontier(results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return subsets on the performance-versus-burden Pareto frontier.

    Lower task count is better.
    Higher Macro F1 is better.
    """

    best_by_count = best_subset_per_task_count(
        results
    )

    frontier_rows = []

    best_performance_so_far = float("-inf")

    for _, row in best_by_count.iterrows():
        performance = float(
            row["macro_f1_mean"]
        )

        if performance > best_performance_so_far:
            frontier_rows.append(
                row.to_dict()
            )

            best_performance_so_far = performance

    return pd.DataFrame(
        frontier_rows
    )


def plot_single_task_performance(results: pd.DataFrame,
) -> None:
    # Plot performance of the eight individual speech tasks.

    single_tasks = (
        results.loc[
            results["task_count"].eq(1)
        ]
        .sort_values(
            "macro_f1_mean",
            ascending=False,
        )
    )

    full_reference = float(
        results.loc[
            results["task_count"].eq(8),
            "macro_f1_mean",
        ].iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        single_tasks["subset_id"],
        single_tasks["macro_f1_mean"],
        yerr=single_tasks["macro_f1_std"],
        capsize=4,
    )

    ax.axhline(
        full_reference,
        linestyle="--",
        label=(
            f"Full 8-task reference "
            f"({full_reference:.3f})"
        ),
    )

    for bar, value in zip(
        bars,
        single_tasks["macro_f1_mean"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(
        "Speech task"
    )

    ax.set_ylabel(
        "Mean Macro F1 (±1 SD across outer folds)"
    )

    ax.set_title(
        "Single-Task Speech Impairment Performance"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "single_task_performance",
    )


def plot_task_frequency(
    results: pd.DataFrame,
) -> None:
    # Plot task appearances among the top-performing subsets.

    frequency = task_frequency_analysis(
        results,
        top_n=TOP_N,
    )

    top_subsets = (
        results
        .sort_values(
            "macro_f1_mean",
            ascending=False,
        )
        .head(TOP_N)
    )

    expected_appearances = (
        top_subsets["task_count"].sum()
        / len(TASKS)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        frequency["task"],
        frequency["appearances"],
    )

    ax.axhline(
        expected_appearances,
        linestyle="--",
        label=(
            "Expected from top-25 "
            "task-count mix "
            f"({expected_appearances:.1f})"
        ),
    )

    for bar, value in zip(
        bars,
        frequency["appearances"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(value),
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(
        "Speech task"
    )

    ax.set_ylabel(
        f"Appearances among top {TOP_N} subsets"
    )

    ax.set_title(
        "Task Frequency Among High-Performing Protocols"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "task_frequency_top25",
    )

def plot_pareto_performance_vs_burden(results: pd.DataFrame,
) -> None:
    # Plot Macro F1 against speech-assessment burden.

    full_reference = float(
        results.loc[
            results["task_count"].eq(8),
            "macro_f1_mean",
        ].iloc[0]
    )

    primary_threshold = (
        full_reference
        - PRIMARY_TOLERANCE
    )

    best_by_count = best_subset_per_task_count(
        results
    )

    frontier = calculate_pareto_frontier(
        results
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        results["task_count"],
        results["macro_f1_mean"],
        alpha=0.25,
        s=25,
        label="All task subsets",
    )

    ax.plot(
        best_by_count["task_count"],
        best_by_count["macro_f1_mean"],
        marker="o",
        linestyle="--",
        label="Best subset at each task count",
    )

    ax.plot(
        frontier["task_count"],
        frontier["macro_f1_mean"],
        marker="o",
        linewidth=2.5,
        label="Pareto frontier",
    )

    ax.axhline(
        full_reference,
        linestyle="--",
        label=(
            f"Full 8-task reference "
            f"({full_reference:.3f})"
        ),
    )

    ax.axhline(
        primary_threshold,
        linestyle=":",
        label=(
            f"Δ = 0.02 threshold "
            f"({primary_threshold:.3f})"
        ),
    )

    for _, row in frontier.iterrows():
        label = (
            f"{row['subset_id']} "
            f"({row['macro_f1_mean']:.3f})"
        )

        ax.annotate(
            label,
            (
                row["task_count"],
                row["macro_f1_mean"],
            ),
            xytext=(6, 7),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Number of speech tasks"
    )

    ax.set_ylabel(
        "Mean Macro F1"
    )

    ax.set_title(
        "Predictive Performance vs Speech-Assessment Burden"
    )

    ax.set_xticks(
        range(1, 9)
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "pareto_performance_vs_burden",
    )


def main() -> None:
    results = load_subset_summaries()

    frequency = task_frequency_analysis(
        results,
        top_n=TOP_N,
    )

    frontier = calculate_pareto_frontier(
        results
    )

    print("Task frequency among top 25 subsets:")
    print(
        frequency.to_string(
            index=False
        )
    )

    print()
    print("Pareto frontier:")

    print(
        frontier[
            [
                "task_count",
                "subset_id",
                "macro_f1_mean",
                "macro_f1_std",
            ]
        ].to_string(
            index=False
        )
    )

    plot_single_task_performance(
        results
    )

    plot_task_frequency(
        results
    )

    plot_pareto_performance_vs_burden(
        results
    )

    print()
    print("Figures generated:")
    print(
        FIGURE_DIR / "single_task_performance.png"
    )
    print(
        FIGURE_DIR / "task_frequency_top25.png"
    )
    print(
        FIGURE_DIR / "pareto_performance_vs_burden.png"
    )


if __name__ == "__main__":
    main()
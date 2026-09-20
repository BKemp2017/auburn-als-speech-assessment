"""Create figures for the leakage-safe nested task-selection analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from audit_metadata import TASKS


# The tracked experiment record contains only aggregate results. Keeping it
# under version control lets the figures be reproduced without exposing
# participant-level predictions or rerunning model training.
EXPERIMENT_RECORD_PATH = Path(
    "experiments/nested_task_selection.json"
)

FIGURE_DIR = Path("figures")

def load_plot_inputs() -> tuple[pd.DataFrame, dict]:
    """Load and validate the tracked aggregate experiment record."""

    if not EXPERIMENT_RECORD_PATH.exists():
        raise FileNotFoundError(
            "The nested task-selection experiment record was not found at "
            f"{EXPERIMENT_RECORD_PATH}."
        )

    summary = json.loads(
        EXPERIMENT_RECORD_PATH.read_text(
            encoding="utf-8"
        )
    )

    if summary.get("experiment") != (
        "leakage_safe_nested_task_selection"
    ):
        raise ValueError(
            "The experiment record does not describe the expected "
            "nested-selection analysis."
        )

    fold_selections = summary.get(
        "fold_selections"
    )

    if not isinstance(
        fold_selections,
        list,
    ):
        raise ValueError(
            "The experiment record does not contain fold selections."
        )

    # Reconstruct the small aggregate table used by the figures. This table
    # contains one row per outer fold and no participant-level information.
    fold_metrics = pd.DataFrame(
        fold_selections
    )

    required_columns = {
        "outer_fold",
        "selected_subset",
        "task_count",
        "outer_test_macro_f1",
    }

    missing_columns = sorted(
        required_columns
        - set(fold_metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold selections are missing required columns: "
            f"{missing_columns}"
        )

    fold_metrics = (
        fold_metrics
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    if len(fold_metrics) != 5:
        raise ValueError(
            "Expected one result for each of the five outer folds, "
            f"but found {len(fold_metrics)} rows."
        )

    expected_outer_folds = set(
        range(5)
    )

    recorded_outer_folds = set(
        int(value)
        for value in fold_metrics[
            "outer_fold"
        ]
    )

    if recorded_outer_folds != expected_outer_folds:
        raise ValueError(
            "The experiment record does not contain outer folds 0 through 4."
        )

    numeric_columns = [
        "outer_fold",
        "task_count",
        "outer_test_macro_f1",
    ]

    if not np.isfinite(
        fold_metrics[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A non-finite value was found in the aggregate fold results."
        )

    return (
        fold_metrics,
        summary,
    )

def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    """Save one publication-ready figure as both PNG and PDF."""

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

def build_fold_comparison_table(
    fold_metrics: pd.DataFrame,
    summary: dict,
) -> pd.DataFrame:
    """Build and validate the paired full-versus-selected fold results."""

    # The summary contains the full eight-task reference for each fold.
    # Keeping the pairing matters because every comparison must use the
    # same held-out participants.
    comparison_records = (
        summary.get(
            "full_eight_task_comparison",
            {},
        ).get(
            "fold_comparisons"
        )
    )

    if not isinstance(
        comparison_records,
        list,
    ):
        raise ValueError(
            "The summary does not contain fold-level comparisons."
        )

    comparison = pd.DataFrame(
        comparison_records
    )

    required_columns = {
        "outer_fold",
        "nested_selected_macro_f1",
        "full_macro_f1",
        "difference",
    }

    missing_columns = sorted(
        required_columns
        - set(comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold comparisons are missing required columns: "
            f"{missing_columns}"
        )

    comparison = (
        comparison
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    if len(comparison) != 5:
        raise ValueError(
            "Expected five paired fold comparisons, "
            f"but found {len(comparison)}."
        )

    # Cross-check the fold-selection and fold-comparison sections of the
    # tracked record so inconsistent aggregate values cannot be plotted.
    recorded_nested_scores = (
        fold_metrics[
            "outer_test_macro_f1"
        ].to_numpy()
    )

    summarized_nested_scores = (
        comparison[
            "nested_selected_macro_f1"
        ].to_numpy()
    )

    if not np.allclose(
        recorded_nested_scores,
        summarized_nested_scores,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Nested Macro F1 values disagree between the fold metrics "
            "and summary."
        )

    calculated_differences = (
        comparison[
            "nested_selected_macro_f1"
        ]
        - comparison["full_macro_f1"]
    )

    if not np.allclose(
        calculated_differences,
        comparison["difference"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Stored fold differences do not match the plotted scores."
        )

    return comparison

def plot_fold_macro_f1_comparison(
    fold_metrics: pd.DataFrame,
    summary: dict,
) -> None:
    """Plot paired Macro F1 results for the five outer folds."""

    comparison = build_fold_comparison_table(
        fold_metrics,
        summary,
    )

    full_scores = comparison[
        "full_macro_f1"
    ].to_numpy()

    selected_scores = comparison[
        "nested_selected_macro_f1"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # A thin line connects the two results from the same outer fold. This
    # makes fold-to-fold variation visible instead of hiding it in averages.
    for row in comparison.itertuples(
        index=False
    ):
        ax.plot(
            [0, 1],
            [
                row.full_macro_f1,
                row.nested_selected_macro_f1,
            ],
            color="#8C8C8C",
            linewidth=1.3,
            alpha=0.75,
            zorder=1,
        )

        ax.annotate(
            f"Fold {row.outer_fold}",
            (
                1,
                row.nested_selected_macro_f1,
            ),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.scatter(
        np.zeros(len(full_scores)),
        full_scores,
        color="#4C78A8",
        s=55,
        label="Full eight-task folds",
        zorder=2,
    )

    ax.scatter(
        np.ones(len(selected_scores)),
        selected_scores,
        color="#F58518",
        s=55,
        label="Nested-selected folds",
        zorder=2,
    )

    # The heavier black line shows the average direction without replacing
    # the five individual paired observations.
    mean_scores = [
        float(full_scores.mean()),
        float(selected_scores.mean()),
    ]

    ax.plot(
        [0, 1],
        mean_scores,
        color="#222222",
        linewidth=3,
        marker="D",
        markersize=7,
        label="Mean across folds",
        zorder=3,
    )

    all_scores = np.concatenate(
        [
            full_scores,
            selected_scores,
        ]
    )

    score_range = float(
        all_scores.max()
        - all_scores.min()
    )

    padding = max(
        0.04,
        score_range * 0.15,
    )

    ax.set_ylim(
        max(
            0.0,
            float(all_scores.min()) - padding,
        ),
        min(
            1.0,
            float(all_scores.max()) + padding,
        ),
    )

    ax.set_xlim(
        -0.25,
        1.25,
    )

    ax.set_xticks(
        [0, 1],
        [
            "Full eight-task",
            "Nested selected",
        ],
    )

    ax.set_ylabel(
        "Macro F1 on held-out outer fold"
    )

    ax.set_title(
        "Nested-Selected Protocol vs Full Assessment"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "nested_selected_vs_full_macro_f1",
    )
def build_task_frequency_table(
    fold_metrics: pd.DataFrame,
    summary: dict,
) -> pd.DataFrame:
    """Build and validate task-selection counts across outer folds."""

    task_frequency = summary.get(
        "task_frequency"
    )

    if not isinstance(
        task_frequency,
        dict,
    ):
        raise ValueError(
            "The summary does not contain task-selection frequencies."
        )

    missing_tasks = sorted(
        set(TASKS)
        - set(task_frequency)
    )

    unexpected_tasks = sorted(
        set(task_frequency)
        - set(TASKS)
    )

    if missing_tasks or unexpected_tasks:
        raise ValueError(
            "Task-frequency labels do not match the expected tasks. "
            f"Missing: {missing_tasks}; "
            f"unexpected: {unexpected_tasks}."
        )

    frequency_table = pd.DataFrame(
        {
            "task": list(TASKS),
            "selected_fold_count": [
                int(task_frequency[task])
                for task in TASKS
            ],
        }
    )

    outer_fold_count = len(
        fold_metrics
    )

    invalid_counts = frequency_table.loc[
        ~frequency_table[
            "selected_fold_count"
        ].between(
            0,
            outer_fold_count,
        )
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Task-selection frequencies must be between zero and "
            f"{outer_fold_count}."
        )

    # Adding all task frequencies should produce the same total as adding
    # the number of tasks selected independently in each outer fold.
    recorded_task_appearances = int(
        frequency_table[
            "selected_fold_count"
        ].sum()
    )

    expected_task_appearances = int(
        fold_metrics[
            "task_count"
        ].sum()
    )

    if (
        recorded_task_appearances
        != expected_task_appearances
    ):
        raise ValueError(
            "Task frequencies do not match the selected task counts "
            "recorded across the outer folds."
        )

    return frequency_table

def plot_task_selection_frequency(
    fold_metrics: pd.DataFrame,
    summary: dict,
) -> None:
    """Plot how often each speech task was selected."""

    frequency_table = (
        build_task_frequency_table(
            fold_metrics,
            summary,
        )
    )

    outer_fold_count = len(
        fold_metrics
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        frequency_table["task"],
        frequency_table[
            "selected_fold_count"
        ],
        color="#4C78A8",
    )

    # Showing the count above every bar makes zero-frequency tasks explicit
    # instead of making them look like missing data.
    for bar, count in zip(
        bars,
        frequency_table[
            "selected_fold_count"
        ],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            float(count) + 0.1,
            str(count),
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(
        "Speech task"
    )

    ax.set_ylabel(
        "Number of outer folds selected"
    )

    ax.set_title(
        "Task Selection Frequency Across Outer Folds"
    )

    ax.set_ylim(
        0,
        outer_fold_count + 0.6,
    )

    ax.set_yticks(
        range(
            0,
            outer_fold_count + 1,
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.set_axisbelow(
        True
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    fig.tight_layout()

    save_figure(
        fig,
        "nested_task_selection_frequency",
    )

def build_selected_protocol_matrix(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a binary task-inclusion matrix for the selected protocols."""

    protocol_records = []

    for row in fold_metrics.itertuples(
        index=False
    ):
        subset_id = str(
            row.selected_subset
        )

        selected_tasks = subset_id.split(
            "+"
        )

        if (
            not subset_id
            or any(
                not task
                for task in selected_tasks
            )
        ):
            raise ValueError(
                f"Outer fold {row.outer_fold} has an invalid subset ID."
            )

        if len(selected_tasks) != len(
            set(selected_tasks)
        ):
            raise ValueError(
                f"Outer fold {row.outer_fold} contains a duplicate task."
            )

        unexpected_tasks = sorted(
            set(selected_tasks)
            - set(TASKS)
        )

        if unexpected_tasks:
            raise ValueError(
                f"Outer fold {row.outer_fold} contains unknown tasks: "
                f"{unexpected_tasks}."
            )

        if len(selected_tasks) != int(
            row.task_count
        ):
            raise ValueError(
                f"Outer fold {row.outer_fold} has a task-count mismatch."
            )

        record = {
            "outer_fold": int(
                row.outer_fold
            )
        }

        # Each cell records whether the task was part of that fold's
        # independently selected reduced protocol.
        for task in TASKS:
            record[task] = int(
                task in selected_tasks
            )

        protocol_records.append(
            record
        )

    matrix = (
        pd.DataFrame(
            protocol_records
        )
        .set_index("outer_fold")
        .sort_index()
    )

    matrix = matrix.loc[
        :,
        list(TASKS),
    ]

    if matrix.shape != (
        len(fold_metrics),
        len(TASKS),
    ):
        raise ValueError(
            "The selected-protocol matrix has an unexpected shape."
        )

    return matrix

def plot_selected_protocol_matrix(
    fold_metrics: pd.DataFrame,
) -> None:
    """Plot the exact task composition selected in every outer fold."""

    matrix = (
        build_selected_protocol_matrix(
            fold_metrics
        )
    )

    task_counts = matrix.sum(
        axis=1
    )

    row_labels = []

    for outer_fold, task_count in (
        task_counts.items()
    ):
        task_word = (
            "task"
            if int(task_count) == 1
            else "tasks"
        )

        row_labels.append(
            f"Fold {outer_fold} "
            f"({int(task_count)} {task_word})"
        )

    # Light cells are excluded tasks, while blue cells are tasks included
    # in the protocol selected using only that outer fold's training data.
    selection_colors = ListedColormap(
        [
            "#F1F3F5",
            "#4C78A8",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.imshow(
        matrix.to_numpy(
            dtype=int
        ),
        cmap=selection_colors,
        vmin=0,
        vmax=1,
        interpolation="none",
        aspect="auto",
    )

    # Check marks let the figure remain understandable when printed without
    # relying only on the difference between the two cell colors.
    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            if (
                matrix.iloc[
                    row_index,
                    column_index,
                ]
                == 1
            ):
                ax.text(
                    column_index,
                    row_index,
                    "✓",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=16,
                    fontweight="bold",
                )

    ax.set_xticks(
        range(
            len(TASKS)
        ),
        list(TASKS),
    )

    ax.set_yticks(
        range(
            len(row_labels)
        ),
        row_labels,
    )

    ax.set_xlabel(
        "Speech task"
    )

    ax.set_ylabel(
        "Outer fold and selected burden"
    )

    ax.set_title(
        "Selected Protocol Composition by Outer Fold"
    )

    # White boundaries keep adjacent selected tasks from visually merging
    # into one wide block.
    ax.set_xticks(
        np.arange(
            -0.5,
            len(TASKS),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(matrix),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )

    fig.tight_layout()

    save_figure(
        fig,
        "nested_selected_protocols",
    )

def main() -> None:
    """Generate figures from the validated aggregate experiment record."""

    fold_metrics, summary = load_plot_inputs()

    print("Nested plotting inputs validated.")
    print(f"Outer-fold rows: {len(fold_metrics)}")
    print(
        "Experiment: "
        f"{summary['experiment']}"
    )

    plot_fold_macro_f1_comparison(
        fold_metrics,
        summary,
    )

    plot_task_selection_frequency(
        fold_metrics,
        summary,
    )
    plot_selected_protocol_matrix(
        fold_metrics
    )

    print("Figures generated:")
    print(
        "figures/nested_selected_vs_full_macro_f1.png"
    )
    print(
        "figures/nested_task_selection_frequency.png"
    )
    print(
        "figures/nested_selected_protocols.png"
    )



if __name__ == "__main__":
    main()
"""Create the protocol-stability summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_metadata import TASKS


EXPERIMENT_RECORD_PATH = Path(
    "experiments/protocol_stability.json"
)

FIGURE_DIR = Path("figures")
FIGURE_NAME = "protocol_stability_summary"

PRIMARY_COLOR = "#4C78A8"
SECONDARY_COLOR = "#F58518"
MUTED_COLOR = "#C7CDD4"


def load_stability_record() -> dict:
    """Load and validate the tracked aggregate experiment record."""

    if not EXPERIMENT_RECORD_PATH.exists():
        raise FileNotFoundError(
            "The protocol-stability record was not found at "
            f"{EXPERIMENT_RECORD_PATH}."
        )

    record = json.loads(
        EXPERIMENT_RECORD_PATH.read_text(
            encoding="utf-8"
        )
    )

    if record.get("experiment") != (
        "participant_resampled_protocol_stability"
    ):
        raise ValueError(
            "The experiment record does not describe the expected "
            "protocol-stability analysis."
        )

    evaluation = record.get("evaluation", {})

    if evaluation.get(
        "bootstrap_replicates_per_outer_fold"
    ) != 10_000:
        raise ValueError(
            "The experiment record does not contain 10,000 "
            "bootstrap replicates per outer fold."
        )

    if evaluation.get(
        "total_bootstrap_selections"
    ) != 50_000:
        raise ValueError(
            "The experiment record does not contain 50,000 "
            "total bootstrap selections."
        )

    if evaluation.get("delta") != 0.02:
        raise ValueError(
            "The experiment record does not use the locked "
            "primary delta of 0.02."
        )

    if not isinstance(
        record.get("overall_stability"),
        dict,
    ):
        raise ValueError(
            "The experiment record is missing overall stability results."
        )

    per_outer_fold = record.get(
        "per_outer_fold"
    )

    if not isinstance(
        per_outer_fold,
        list,
    ) or len(per_outer_fold) != 5:
        raise ValueError(
            "The experiment record must contain five outer-fold results."
        )

    return record


def build_task_count_table(
    record: dict,
) -> pd.DataFrame:
    """Build the task-count distribution used by the first panel."""

    distribution = record[
        "overall_stability"
    ]["task_count_distribution"]

    rows = []

    for task_count in range(1, 9):
        values = distribution.get(
            str(task_count),
            {
                "selection_count": 0,
                "selection_rate": 0.0,
            },
        )

        rows.append(
            {
                "task_count": task_count,
                "selection_count": int(
                    values["selection_count"]
                ),
                "selection_percentage": float(
                    values["selection_rate"]
                    * 100.0
                ),
            }
        )

    table = pd.DataFrame(rows)

    if table["selection_count"].sum() != 50_000:
        raise ValueError(
            "Task-count selections do not sum to 50,000."
        )

    if not np.isclose(
        table["selection_percentage"].sum(),
        100.0,
    ):
        raise ValueError(
            "Task-count percentages do not sum to 100."
        )

    return table


def build_task_frequency_table(
    record: dict,
) -> pd.DataFrame:
    """Build individual-task appearance rates for the second panel."""

    task_frequency = record[
        "overall_stability"
    ]["task_frequency"]

    rows = []

    for task in TASKS:
        if task not in task_frequency:
            raise ValueError(
                f"Task {task} is missing from the stability record."
            )

        values = task_frequency[task]

        rows.append(
            {
                "task": task,
                "selection_count": int(
                    values["selection_count"]
                ),
                "selection_percentage": float(
                    values["selection_rate"]
                    * 100.0
                ),
            }
        )

    table = pd.DataFrame(rows)

    if not table[
        "selection_percentage"
    ].between(0.0, 100.0).all():
        raise ValueError(
            "A task-selection percentage is outside 0 to 100."
        )

    return table


def build_fold_stability_table(
    record: dict,
) -> pd.DataFrame:
    """Build fold-level retention and task-overlap results."""

    rows = []

    for result in record["per_outer_fold"]:
        original = result[
            "original_selection"
        ]

        rows.append(
            {
                "outer_fold": int(
                    result["outer_fold"]
                ),
                "original_subset": str(
                    original["subset_id"]
                ),
                "retention_percentage": float(
                    result[
                        "original_selection_retention_rate"
                    ]
                    * 100.0
                ),
                "jaccard_percentage": float(
                    result[
                        "mean_jaccard_similarity_to_original"
                    ]
                    * 100.0
                ),
                "normalized_entropy": float(
                    result[
                        "normalized_selection_entropy"
                    ]
                ),
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )

    if set(table["outer_fold"]) != set(range(5)):
        raise ValueError(
            "Fold stability results do not contain folds 0 through 4."
        )

    percentage_columns = [
        "retention_percentage",
        "jaccard_percentage",
    ]

    if not table[
        percentage_columns
    ].apply(
        lambda column: column.between(
            0.0,
            100.0,
        )
    ).all().all():
        raise ValueError(
            "A fold-stability percentage is outside 0 to 100."
        )

    if not table[
        "normalized_entropy"
    ].between(0.0, 1.0).all():
        raise ValueError(
            "A normalized entropy value is outside 0 to 1."
        )

    return table


def label_bars(
    axis: plt.Axes,
    bars,
) -> None:
    """Place compact percentage labels above vertical bars."""

    for bar in bars:
        height = float(
            bar.get_height()
        )

        label = (
            "<0.1%"
            if 0.0 < height < 0.1
            else f"{height:.1f}%"
            )
        axis.annotate(
            label,
            xy=(
                bar.get_x()
                + bar.get_width() / 2.0,
                height,
            ),  
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_figure(
    figure: plt.Figure,
) -> None:
    """Save the figure as both a high-resolution PNG and PDF."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_DIR / f"{FIGURE_NAME}.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR / f"{FIGURE_NAME}.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_protocol_stability_summary(
    task_counts: pd.DataFrame,
    task_frequency: pd.DataFrame,
    fold_stability: pd.DataFrame,
) -> None:
    """Create the three-panel protocol-stability figure."""

    plt.style.use(
        "seaborn-v0_8-whitegrid"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
    )

    task_count_colors = [
        PRIMARY_COLOR
        if task_count <= 2
        else MUTED_COLOR
        for task_count in task_counts[
            "task_count"
        ]
    ]

    task_count_bars = axes[0].bar(
        task_counts["task_count"],
        task_counts[
            "selection_percentage"
        ],
        color=task_count_colors,
        edgecolor="white",
    )

    axes[0].set_title(
        "A. Selected assessment burden"
    )

    axes[0].set_xlabel(
        "Number of selected speech tasks"
    )

    axes[0].set_ylabel(
        "Bootstrap selections (%)"
    )

    axes[0].set_xticks(
        task_counts["task_count"]
    )

    axes[0].set_ylim(0, 100)

    label_bars(
        axes[0],
        task_count_bars,
    )

    task_bars = axes[1].bar(
        task_frequency["task"],
        task_frequency[
            "selection_percentage"
        ],
        color=PRIMARY_COLOR,
        edgecolor="white",
    )

    axes[1].set_title(
        "B. Individual task appearance"
    )

    axes[1].set_xlabel(
        "Speech task"
    )

    axes[1].set_ylabel(
        "Bootstrap selections containing task (%)"
    )

    axes[1].set_ylim(0, 100)

    label_bars(
        axes[1],
        task_bars,
    )

    fold_positions = np.arange(
        len(fold_stability)
    )

    bar_width = 0.36

    retention_bars = axes[2].bar(
        fold_positions - bar_width / 2.0,
        fold_stability[
            "retention_percentage"
        ],
        width=bar_width,
        color=PRIMARY_COLOR,
        label="Exact original protocol",
        edgecolor="white",
    )

    jaccard_bars = axes[2].bar(
        fold_positions + bar_width / 2.0,
        fold_stability[
            "jaccard_percentage"
        ],
        width=bar_width,
        color=SECONDARY_COLOR,
        label="Mean task overlap",
        edgecolor="white",
    )

    fold_labels = [
        f"Fold {row.outer_fold}\n{row.original_subset}"
        for row in fold_stability.itertuples(
            index=False
        )
    ]

    axes[2].set_title(
        "C. Stability of original fold selections"
    )

    axes[2].set_xlabel(
        "Outer fold and original protocol"
    )

    axes[2].set_ylabel(
        "Bootstrap stability (%)"
    )

    axes[2].set_xticks(
        fold_positions
    )

    axes[2].set_xticklabels(
        fold_labels
    )

    axes[2].set_ylim(0, 100)

    axes[2].legend(
        loc="upper right",
        frameon=True,
    )

    label_bars(
        axes[2],
        retention_bars,
    )

    label_bars(
        axes[2],
        jaccard_bars,
    )

    for axis in axes:
        axis.spines["top"].set_visible(
            False
        )

        axis.spines["right"].set_visible(
            False
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

        axis.grid(
            axis="x",
            visible=False,
        )

    figure.suptitle(
        "Protocol Stability Under Participant Resampling",
        fontsize=18,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.925,
        (
            "10,000 paired bootstrap resamples within each inner fold; "
            "locked selection tolerance \u03b4 = 0.02"
        ),
        ha="center",
        fontsize=10,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.89)
    )

    save_figure(
        figure
    )


def main() -> None:
    """Validate aggregate inputs and generate the stability figure."""

    record = load_stability_record()

    task_counts = build_task_count_table(
        record
    )

    task_frequency = build_task_frequency_table(
        record
    )

    fold_stability = build_fold_stability_table(
        record
    )

    plot_protocol_stability_summary(
        task_counts=task_counts,
        task_frequency=task_frequency,
        fold_stability=fold_stability,
    )

    print("Protocol-stability plotting inputs validated.")
    print("Bootstrap selections: 50000")
    print("Outer folds: 5")
    print("Figures generated:")
    print(
        FIGURE_DIR
        / f"{FIGURE_NAME}.png"
    )
    print(
        FIGURE_DIR
        / f"{FIGURE_NAME}.pdf"
    )


if __name__ == "__main__":
    main()
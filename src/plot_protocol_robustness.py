"""Create figures and tables for the fixed-protocol robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_RECORD_PATH = Path(
    "experiments/protocol_robustness.json"
)

KEY_CANDIDATE_TABLE_PATH = Path(
    "experiments/protocol_robustness_key_candidates.csv"
)

FIGURE_DIR = Path("figures")
FIGURE_NAME = "protocol_robustness_summary"

PRIMARY_COLOR = "#4C78A8"
SECONDARY_COLOR = "#F58518"
MUTED_COLOR = "#C7CDD4"
MARGIN_COLOR = "#C44E52"
REFERENCE_COLOR = "#333333"


def load_robustness_record() -> dict:
    """Load and validate the tracked aggregate experiment record."""

    if not EXPERIMENT_RECORD_PATH.exists():
        raise FileNotFoundError(
            "The protocol-robustness record was not found at "
            f"{EXPERIMENT_RECORD_PATH}."
        )

    record = json.loads(
        EXPERIMENT_RECORD_PATH.read_text(
            encoding="utf-8"
        )
    )

    if record.get(
        "experiment"
    ) != "fixed_protocol_robustness":
        raise ValueError(
            "The experiment record does not describe the expected "
            "fixed-protocol robustness analysis."
        )

    if record.get(
        "participant_level_fields"
    ) != []:
        raise ValueError(
            "The tracked experiment record unexpectedly contains "
            "participant-level fields."
        )

    if record.get(
        "noninferiority_margin"
    ) != 0.02:
        raise ValueError(
            "The experiment record does not use the locked "
            "noninferiority margin of 0.02."
        )

    coverage = record.get(
        "coverage"
    )

    if not isinstance(
        coverage,
        dict,
    ):
        raise ValueError(
            "The experiment record is missing coverage information."
        )

    expected_coverage = {
        "outer_folds": 5,
        "task_subsets": 255,
        "reduced_task_subsets": 254,
        "fold_comparisons": 1275,
        "reduced_fold_comparisons": 1270,
    }

    for field, expected_value in expected_coverage.items():
        if coverage.get(field) != expected_value:
            raise ValueError(
                f"Expected {field} to equal {expected_value}, "
                f"but found {coverage.get(field)}."
            )

    bootstrap = record.get(
        "bootstrap"
    )

    if not isinstance(
        bootstrap,
        dict,
    ):
        raise ValueError(
            "The experiment record is missing bootstrap information."
        )

    if bootstrap.get(
        "replicates_per_outer_fold"
    ) != 10_000:
        raise ValueError(
            "The experiment record does not contain 10,000 "
            "bootstrap replicates per outer fold."
        )

    if bootstrap.get(
        "random_seed"
    ) != 2026:
        raise ValueError(
            "The experiment record contains the wrong random seed."
        )

    top_candidates = record.get(
        "top_ranked_candidates"
    )

    if not isinstance(
        top_candidates,
        list,
    ) or len(top_candidates) != 15:
        raise ValueError(
            "The experiment record must contain 15 top-ranked candidates."
        )

    key_candidates = record.get(
        "key_candidates"
    )

    if not isinstance(
        key_candidates,
        list,
    ) or len(key_candidates) != 8:
        raise ValueError(
            "The experiment record must contain eight key candidates."
        )

    interpretation = record.get(
        "interpretation"
    )

    if not isinstance(
        interpretation,
        dict,
    ):
        raise ValueError(
            "The experiment record is missing its interpretation."
        )

    if interpretation.get(
        "universal_fixed_reduced_protocol_supported"
    ) is not False:
        raise ValueError(
            "The record unexpectedly claims support for a universal "
            "fixed reduced protocol."
        )

    return record


def candidate_table_from_records(
    candidates: list[dict],
) -> pd.DataFrame:
    """Convert aggregate candidate records into a plotting table."""

    table = pd.DataFrame(
        candidates
    )

    required_columns = {
        "candidate_rank",
        "subset_id",
        "tasks",
        "task_count",
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "above_full_fold_count",
        "observed_difference_mean",
        "observed_difference_min",
        "difference_ci_lower_min",
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
    }

    missing_columns = (
        required_columns - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate records are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if table["subset_id"].duplicated().any():
        raise ValueError(
            "Duplicate task subsets were found in a candidate table."
        )

    # Store task lists as a readable protocol identifier instead of a
    # Python list representation in the exported CSV.
    table["tasks"] = table["tasks"].map(
        lambda tasks: "+".join(tasks)
    )

    integer_columns = [
        "candidate_rank",
        "task_count",
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "above_full_fold_count",
    ]

    numeric_columns = [
        *integer_columns,
        "observed_difference_mean",
        "observed_difference_min",
        "difference_ci_lower_min",
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
    ]

    if not np.isfinite(
        table[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "A candidate table contains a non-finite value."
        )

    for column in integer_columns:
        table[column] = table[column].astype(
            int
        )

    probability_columns = [
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
    ]

    if not table[
        probability_columns
    ].apply(
        lambda column: column.between(
            0.0,
            1.0,
        )
    ).all().all():
        raise ValueError(
            "A candidate probability is outside zero to one."
        )

    fold_count_columns = [
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "above_full_fold_count",
    ]

    if not table[
        fold_count_columns
    ].apply(
        lambda column: column.between(
            0,
            5,
        )
    ).all().all():
        raise ValueError(
            "A candidate fold count is outside zero to five."
        )

    return (
        table.sort_values(
            "candidate_rank"
        )
        .reset_index(drop=True)
    )


def build_plot_tables(
    record: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the top-ranked and clinically relevant candidate tables."""

    top_candidates = candidate_table_from_records(
        record["top_ranked_candidates"]
    )

    key_candidates = candidate_table_from_records(
        record["key_candidates"]
    )

    expected_key_subsets = {
        "KA",
        "TA",
        "PA",
        "O+TA",
        "PA+KA",
        "TA+KA",
        "A+PA+KA",
        "O+TA+KA",
    }

    if set(
        key_candidates["subset_id"]
    ) != expected_key_subsets:
        raise ValueError(
            "The key-candidate table does not contain the expected "
            "protocols."
        )

    if top_candidates.iloc[0][
        "subset_id"
    ] != "O+TA+KA":
        raise ValueError(
            "The highest-ranked exploratory candidate is not O+TA+KA."
        )

    return top_candidates, key_candidates


def save_key_candidate_table(
    key_candidates: pd.DataFrame,
) -> None:
    """Atomically save the compact aggregate candidate table."""

    output_columns = [
        "candidate_rank",
        "subset_id",
        "tasks",
        "task_count",
        "supporting_fold_count",
        "point_noninferior_fold_count",
        "above_full_fold_count",
        "observed_difference_mean",
        "observed_difference_min",
        "difference_ci_lower_min",
        "noninferiority_probability_mean",
        "noninferiority_probability_min",
    ]

    temporary_path = (
        KEY_CANDIDATE_TABLE_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    try:
        key_candidates[
            output_columns
        ].to_csv(
            temporary_path,
            index=False,
        )

        temporary_path.replace(KEY_CANDIDATE_TABLE_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save one publication-ready figure as PNG and PDF."""

    FIGURE_DIR.mkdir(parents=True,exist_ok=True,)

    figure.savefig(
        FIGURE_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(FIGURE_DIR / f"{filename}.pdf",bbox_inches="tight",)

def configure_plot_style() -> None:
    """Apply a consistent publication-friendly plotting style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "figure.titlesize": 17,
            "figure.titleweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def label_vertical_bars(
    axis: plt.Axes,
    bars,
) -> None:
    """Place compact integer labels above vertical bars."""

    for bar in bars:
        height = float(
            bar.get_height()
        )

        axis.annotate(
            f"{height:.0f}",
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


def plot_candidate_difference_ranges(
    axis: plt.Axes,
    top_candidates: pd.DataFrame,
    margin: float,
) -> None:
    """Show mean and worst-fold differences for top fixed candidates."""

    plot_data = (
        top_candidates.sort_values(
            "candidate_rank"
        )
        .reset_index(drop=True)
    )

    y_positions = np.arange(
        len(plot_data)
    )

    # The connecting lines make the gap between average behavior and the
    # worst observed fold visible without treating that range as a formal
    # confidence interval.
    for position, candidate in plot_data.iterrows():
        axis.plot(
            [
                candidate[
                    "observed_difference_min"
                ],
                candidate[
                    "observed_difference_mean"
                ],
            ],
            [
                position,
                position,
            ],
            color=MUTED_COLOR,
            linewidth=2.5,
            zorder=1,
        )

    axis.scatter(
        plot_data[
            "observed_difference_min"
        ],
        y_positions,
        color=SECONDARY_COLOR,
        s=45,
        label="Worst outer fold",
        zorder=3,
    )

    axis.scatter(
        plot_data[
            "observed_difference_mean"
        ],
        y_positions,
        color=PRIMARY_COLOR,
        s=55,
        label="Mean across outer folds",
        zorder=4,
    )

    axis.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=1.4,
        label="Equal to full protocol",
    )

    axis.axvline(
        -margin,
        color=MARGIN_COLOR,
        linewidth=1.4,
        linestyle="--",
        label=f"Noninferiority margin (−{margin:.2f})",
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        [
            f"{int(row.candidate_rank)}. "
            f"{row.subset_id} "
            f"({int(row.task_count)} tasks)"
            for row in plot_data.itertuples()
        ]
    )

    axis.invert_yaxis()

    axis.set_title(
        "A. Cross-fold performance of top fixed candidates"
    )

    axis.set_xlabel(
        "Macro F1 difference versus full eight-task protocol"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.16,
        ),
        frameon=True,
        ncol=4,
    )


def plot_burden_vs_robustness(
    axis: plt.Axes,
    figure: plt.Figure,
    top_candidates: pd.DataFrame,
    margin: float,
) -> None:
    """Relate protocol burden to average and worst-fold robustness."""

    marker_sizes = (
        60
        + 35
        * top_candidates[
            "point_noninferior_fold_count"
        ]
    )

    scatter = axis.scatter(
        top_candidates["task_count"],
        top_candidates[
            "observed_difference_mean"
        ],
        c=top_candidates[
            "noninferiority_probability_min"
        ],
        s=marker_sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    axis.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=1.3,
    )

    axis.axhline(
        -margin,
        color=MARGIN_COLOR,
        linewidth=1.3,
        linestyle="--",
    )

    labeled_subsets = {
        "O+TA+KA",
        "I+O+U+TA+KA",
        "I+TA+KA",
        "TA+KA",
    }

    for candidate in top_candidates.itertuples():
        if candidate.subset_id not in labeled_subsets:
            continue

        axis.annotate(
            candidate.subset_id,
            xy=(
                candidate.task_count,
                candidate.observed_difference_mean,
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label(
        "Worst-fold noninferiority probability"
    )

    axis.set_title(
        "B. Performance–burden tradeoff"
    )

    axis.set_xlabel(
        "Number of speech tasks"
    )

    axis.set_ylabel(
        "Mean Macro F1 difference versus full protocol"
    )

    axis.set_xticks(
        range(1, 8)
    )

    axis.grid(
        alpha=0.25,
    )


def plot_key_candidate_fold_counts(
    axis: plt.Axes,
    key_candidates: pd.DataFrame,
) -> None:
    """Compare point-estimate and confidence-bound support by protocol."""

    plot_data = (
        key_candidates.sort_values(
            "candidate_rank"
        )
        .reset_index(drop=True)
    )

    x_positions = np.arange(
        len(plot_data)
    )

    bar_width = 0.38

    point_bars = axis.bar(
        x_positions - bar_width / 2.0,
        plot_data[
            "point_noninferior_fold_count"
        ],
        width=bar_width,
        color=PRIMARY_COLOR,
        label="Point estimate inside margin",
    )

    confidence_bars = axis.bar(
        x_positions + bar_width / 2.0,
        plot_data[
            "supporting_fold_count"
        ],
        width=bar_width,
        color=SECONDARY_COLOR,
        label="Confidence-bound support",
    )

    label_vertical_bars(axis,point_bars,)

    label_vertical_bars(axis,confidence_bars,)

    axis.set_title(
        "C. Cross-fold support for key protocols"
    )

    axis.set_ylabel("Number of outer folds")

    axis.set_ylim(0.0,5.6,)

    axis.set_yticks(range(0, 6))

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        plot_data["subset_id"],
        rotation=40,
        ha="right",
    )

    axis.grid(axis="y",alpha=0.25,)

    axis.legend(loc="upper right",frameon=True,)

def plot_protocol_robustness_summary(
    top_candidates: pd.DataFrame,
    key_candidates: pd.DataFrame,
    margin: float,
) -> None:
    """Create the multi-panel fixed-protocol robustness figure."""

    configure_plot_style()

    figure = plt.figure(
        figsize=(16, 12)
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[
            1.25,
            1.0,
        ],
        hspace=0.48,
        wspace=0.25,
    )

    difference_axis = figure.add_subplot(
        grid[0, :]
    )

    burden_axis = figure.add_subplot(
        grid[1, 0]
    )

    support_axis = figure.add_subplot(
        grid[1, 1]
    )

    plot_candidate_difference_ranges(
        axis=difference_axis,
        top_candidates=top_candidates,
        margin=margin,
    )

    plot_burden_vs_robustness(
        axis=burden_axis,
        figure=figure,
        top_candidates=top_candidates,
        margin=margin,
    )

    plot_key_candidate_fold_counts(
        axis=support_axis,
        key_candidates=key_candidates,
    )

    figure.suptitle(
        "Fixed Reduced-Protocol Robustness Across Outer Folds"
    )

    figure.text(
        0.5,
        0.012,
        (
            "Gray segments connect each candidate's worst outer fold "
            "with its cross-fold mean; they are not confidence intervals.\n"
            "Differences are reduced protocol minus full eight-task "
            "protocol. No fixed reduced protocol remained inside the "
            "0.02 margin in all five outer folds."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.subplots_adjust(top=0.92,bottom=0.11,)

    save_figure(figure=figure,filename=FIGURE_NAME,)

    plt.close(figure)

def main() -> None:
    """Validate plotting inputs and export the key-candidate table."""

    record = load_robustness_record()

    top_candidates, key_candidates = (
        build_plot_tables(
            record
        )
    )

    save_key_candidate_table(key_candidates)

    plot_protocol_robustness_summary(
        top_candidates=top_candidates,
        key_candidates=key_candidates,
        margin=float(
            record["noninferiority_margin"]
        ),
    )

    print("Protocol-robustness plotting inputs validated.")
    print(f"Top-ranked candidates: {len(top_candidates)}")
    print(f"Key candidates: {len(key_candidates)}")
    print(
        "Noninferiority margin: "
        f"{record['noninferiority_margin']}"
    )
    print()
    print("Key-candidate table:")
    print(
        key_candidates[
            [
                "candidate_rank",
                "subset_id",
                "task_count",
                "point_noninferior_fold_count",
                "observed_difference_mean",
                "observed_difference_min",
                "noninferiority_probability_min",
            ]
        ].to_string(index=False)
    )
    print()
    print("Generated artifacts:")
    print(KEY_CANDIDATE_TABLE_PATH)
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
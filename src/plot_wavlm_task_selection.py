"""Create paper-ready figures for the WavLM task-selection experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_metadata import TASKS

WAVLM_RECORD_PATH = Path(
    "experiments/wavlm_task_selection.json"
)
WAVLM_BOOTSTRAP_PATH = Path(
    "results/generated/wavlm_task_selection_bootstrap.csv"
)
HANDCRAFTED_RECORD_PATH = Path(
    "experiments/duplicate_exclusion_sensitivity.json"
)

FIGURE_DIRECTORY = Path("figures")
COMPARISON_TABLE_PATH = Path(
    "experiments/wavlm_vs_handcrafted_comparison.csv"
)

FOLD_COMPARISON_STEM = (
    "wavlm_selected_vs_full_macro_f1"
)
BOOTSTRAP_STEM = (
    "wavlm_macro_f1_bootstrap_difference"
)
PROTOCOL_MATRIX_STEM = (
    "wavlm_selected_protocols"
)
TASK_FREQUENCY_STEM = (
    "wavlm_task_selection_frequency"
)
BURDEN_PERFORMANCE_STEM = (
    "wavlm_burden_vs_performance"
)
REPRESENTATION_COMPARISON_STEM = (
    "wavlm_vs_handcrafted_macro_f1"
)

EXPECTED_OUTER_FOLDS = 5
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
NONINFERIORITY_MARGIN = 0.02

FULL_COLOR = "#4C78A8"
SELECTED_COLOR = "#F58518"
MARGIN_COLOR = "#C44E52"
NEUTRAL_COLOR = "#333333"
CONNECTOR_COLOR = "#AAAAAA"
GRID_COLOR = "#DDDDDD"


def configure_plot_style() -> None:
    """Apply one consistent publication-friendly style to every figure."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_json_record(
    path: Path,
) -> dict[str, object]:
    """Load one tracked aggregate experiment record."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required experiment record was not found at {path}."
        )

    with path.open(
        encoding="utf-8",
    ) as input_file:
        record = json.load(
            input_file
        )

    if not isinstance(record, dict):
        raise ValueError(
            f"Experiment record {path} is not a JSON object."
        )

    return record


def load_plot_inputs() -> tuple[
    dict[str, object],
    pd.DataFrame,
    dict[str, object],
]:
    """Load the tracked records and private bootstrap distribution."""

    wavlm_record = load_json_record(
        WAVLM_RECORD_PATH
    )
    handcrafted_record = load_json_record(
        HANDCRAFTED_RECORD_PATH
    )

    if not WAVLM_BOOTSTRAP_PATH.exists():
        raise FileNotFoundError(
            "The WavLM bootstrap artifact is missing. Run "
            "src/evaluate_wavlm_task_selection.py first."
        )

    bootstrap = pd.read_csv(
        WAVLM_BOOTSTRAP_PATH
    )

    return (
        wavlm_record,
        bootstrap,
        handcrafted_record,
    )


def build_wavlm_fold_table(
    record: dict[str, object],
) -> pd.DataFrame:
    """Convert the tracked WavLM fold results into a plotting table."""

    fold_results = record.get(
        "fold_results"
    )

    if not isinstance(
        fold_results,
        list,
    ):
        raise ValueError(
            "The WavLM record does not contain fold results."
        )

    fold_table = (
        pd.DataFrame(fold_results)
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )
    required_columns = {
        "outer_fold",
        "selected_subset",
        "task_count",
        "outer_test_selected_macro_f1",
        "outer_test_full_macro_f1",
        "outer_test_macro_f1_difference",
    }
    missing_columns = required_columns.difference(
        fold_table.columns
    )

    if missing_columns:
        raise ValueError(
            "The WavLM fold results are missing columns: "
            f"{sorted(missing_columns)}."
        )

    return fold_table


def validate_plot_inputs(
    wavlm_record: dict[str, object],
    fold_table: pd.DataFrame,
    bootstrap: pd.DataFrame,
    handcrafted_record: dict[str, object],
) -> None:
    """Cross-check all tracked values before producing any figure."""

    if wavlm_record.get(
        "participant_level_fields"
    ) != []:
        raise ValueError(
            "The WavLM record unexpectedly contains participant fields."
        )

    if handcrafted_record.get(
        "participant_level_fields"
    ) != []:
        raise ValueError(
            "The handcrafted record unexpectedly contains participant "
            "fields."
        )

    if len(fold_table) != EXPECTED_OUTER_FOLDS:
        raise ValueError(
            f"Expected {EXPECTED_OUTER_FOLDS} WavLM fold rows, found "
            f"{len(fold_table)}."
        )

    if fold_table["outer_fold"].tolist() != list(
        range(EXPECTED_OUTER_FOLDS)
    ):
        raise ValueError(
            "The WavLM outer folds are incomplete or out of order."
        )

    if not fold_table[
        "task_count"
    ].between(
        1,
        len(TASKS),
    ).all():
        raise ValueError(
            "A selected WavLM task count is outside the valid range."
        )

    metric_columns = [
        "outer_test_selected_macro_f1",
        "outer_test_full_macro_f1",
        "outer_test_macro_f1_difference",
    ]
    metric_values = fold_table[
        metric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(metric_values).all():
        raise ValueError(
            "The WavLM fold table contains a non-finite metric."
        )

    reproduced_difference = (
        fold_table[
            "outer_test_selected_macro_f1"
        ]
        - fold_table[
            "outer_test_full_macro_f1"
        ]
    )

    if not np.allclose(
        reproduced_difference,
        fold_table[
            "outer_test_macro_f1_difference"
        ],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The stored WavLM fold differences are inconsistent."
        )

    required_bootstrap_columns = {
        "bootstrap_id",
        "selected_macro_f1",
        "full_macro_f1",
        "difference",
    }

    if not required_bootstrap_columns.issubset(
        bootstrap.columns
    ):
        raise ValueError(
            "The WavLM bootstrap table is missing required columns."
        )

    if len(bootstrap) != EXPECTED_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"Expected {EXPECTED_BOOTSTRAP_REPLICATES} bootstrap rows, "
            f"found {len(bootstrap)}."
        )

    if bootstrap[
        "bootstrap_id"
    ].nunique() != EXPECTED_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "The WavLM bootstrap identifiers are not unique."
        )

    bootstrap_values = bootstrap[
        [
            "selected_macro_f1",
            "full_macro_f1",
            "difference",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(
        bootstrap_values
    ).all():
        raise ValueError(
            "The WavLM bootstrap contains a non-finite value."
        )

    reproduced_bootstrap_difference = (
        bootstrap["selected_macro_f1"]
        - bootstrap["full_macro_f1"]
    )

    if not np.allclose(
        reproduced_bootstrap_difference,
        bootstrap["difference"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The WavLM bootstrap differences are inconsistent."
        )

    bootstrap_record = wavlm_record.get(
        "paired_bootstrap"
    )

    if not isinstance(
        bootstrap_record,
        dict,
    ):
        raise ValueError(
            "The WavLM record is missing its bootstrap summary."
        )

    lower, upper = np.quantile(
        bootstrap["difference"].to_numpy(
            dtype=float
        ),
        [
            0.025,
            0.975,
        ],
    )
    comparisons = {
        "difference_bootstrap_mean": float(
            bootstrap["difference"].mean()
        ),
        "difference_ci_95_lower": float(lower),
        "difference_ci_95_upper": float(upper),
        "difference_one_sided_95_lower": float(
            bootstrap["difference"].quantile(
                0.05
            )
        ),
        "noninferiority_probability": float(
            bootstrap["difference"].ge(
                -NONINFERIORITY_MARGIN
            ).mean()
        ),
    }

    for field, reproduced_value in comparisons.items():
        if not np.isclose(
            float(bootstrap_record[field]),
            reproduced_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Tracked bootstrap field {field} is stale."
            )

    design = handcrafted_record.get(
        "design"
    )

    if not isinstance(
        design,
        dict,
    ) or int(
        design.get(
            "remaining_participants",
            0,
        )
    ) != 100:
        raise ValueError(
            "The comparison requires the duplicate-filtered 100-person "
            "handcrafted experiment."
        )


def save_figure(
    figure: plt.Figure,
    stem: str,
) -> None:
    """Save matching high-resolution PNG and vector PDF versions."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in (
        "png",
        "pdf",
    ):
        figure.savefig(
            FIGURE_DIRECTORY
            / f"{stem}.{extension}"
        )

    plt.close(figure)


def plot_fold_comparison(
    fold_table: pd.DataFrame,
) -> None:
    """Plot paired selected-versus-full Macro F1 for every outer fold."""

    figure, axis = plt.subplots(
        figsize=(8.4, 6.2)
    )
    full_scores = fold_table[
        "outer_test_full_macro_f1"
    ].to_numpy(dtype=float)
    selected_scores = fold_table[
        "outer_test_selected_macro_f1"
    ].to_numpy(dtype=float)
    label_offsets = {
        1: (8, 9),
        4: (8, -9),
    }

    for row in fold_table.itertuples(
        index=False
    ):
        axis.plot(
            [0, 1],
            [
                row.outer_test_full_macro_f1,
                row.outer_test_selected_macro_f1,
            ],
            color=CONNECTOR_COLOR,
            linewidth=1.5,
            zorder=1,
        )
        axis.annotate(
            f"Fold {row.outer_fold}: {row.selected_subset}",
            xy=(
                1,
                row.outer_test_selected_macro_f1,
            ),
            xytext=label_offsets.get(
                int(row.outer_fold),
                (8, 0),
            ),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    axis.scatter(
        np.zeros(len(fold_table)),
        full_scores,
        color=FULL_COLOR,
        s=62,
        label="Full eight-task folds",
        zorder=3,
    )
    axis.scatter(
        np.ones(len(fold_table)),
        selected_scores,
        color=SELECTED_COLOR,
        s=62,
        label="Nested-selected folds",
        zorder=3,
    )
    axis.plot(
        [0, 1],
        [
            full_scores.mean(),
            selected_scores.mean(),
        ],
        color=NEUTRAL_COLOR,
        marker="D",
        markersize=8,
        linewidth=3,
        label="Mean across folds",
        zorder=4,
    )
    axis.set_xticks(
        [0, 1],
        [
            "Full eight-task",
            "Nested selected",
        ],
    )
    axis.set_xlim(-0.25, 1.35)
    axis.set_ylabel(
        "Macro F1 on held-out outer fold"
    )
    axis.set_title(
        "Frozen WavLM: Reduced Protocol vs Full Assessment"
    )
    axis.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower left",
        frameon=True,
    )

    save_figure(
        figure,
        FOLD_COMPARISON_STEM,
    )


def plot_bootstrap_distribution(
    bootstrap: pd.DataFrame,
    record: dict[str, object],
) -> None:
    """Show paired uncertainty relative to equality and the NI margin."""

    bootstrap_summary = record[
        "paired_bootstrap"
    ]
    differences = bootstrap[
        "difference"
    ].to_numpy(dtype=float)
    lower = float(
        bootstrap_summary[
            "difference_ci_95_lower"
        ]
    )
    upper = float(
        bootstrap_summary[
            "difference_ci_95_upper"
        ]
    )
    point = float(
        bootstrap_summary[
            "difference_point"
        ]
    )
    probability = float(
        bootstrap_summary[
            "noninferiority_probability"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 5.8)
    )
    axis.hist(
        differences,
        bins=55,
        density=True,
        color=SELECTED_COLOR,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )
    axis.axvspan(
        -NONINFERIORITY_MARGIN,
        axis.get_xlim()[1],
        color="#55A868",
        alpha=0.10,
        label="Noninferiority region",
    )
    axis.axvline(
        -NONINFERIORITY_MARGIN,
        color=MARGIN_COLOR,
        linestyle="--",
        linewidth=2,
        label="Margin = -0.02",
    )
    axis.axvline(
        0.0,
        color=NEUTRAL_COLOR,
        linestyle=":",
        linewidth=2,
        label="Equal performance",
    )
    axis.axvline(
        point,
        color=FULL_COLOR,
        linewidth=2.5,
        label=f"Point estimate = {point:+.3f}",
    )
    axis.axvline(
        lower,
        color="#777777",
        linewidth=1.2,
    )
    axis.axvline(
        upper,
        color="#777777",
        linewidth=1.2,
        label=(
            f"95% interval [{lower:+.3f}, {upper:+.3f}]"
        ),
    )
    axis.text(
        0.02,
        0.95,
        (
            "P(difference ≥ -0.02) = "
            f"{probability:.1%}\n"
            "Formal noninferiority: not supported"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.95,
        },
    )
    axis.set_xlabel(
        "Selected minus full fold-mean Macro F1"
    )
    axis.set_ylabel("Bootstrap density")
    axis.set_title(
        "Paired Bootstrap Uncertainty for WavLM Task Reduction"
    )
    axis.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    save_figure(
        figure,
        BOOTSTRAP_STEM,
    )


def selected_task_matrix(
    fold_table: pd.DataFrame,
) -> np.ndarray:
    """Build a binary outer-fold by task matrix from subset identifiers."""

    matrix = np.zeros(
        (
            len(fold_table),
            len(TASKS),
        ),
        dtype=int,
    )

    for row_index, subset_id in enumerate(
        fold_table["selected_subset"]
    ):
        selected_tasks = set(
            str(subset_id).split("+")
        )

        if not selected_tasks.issubset(
            set(TASKS)
        ):
            raise ValueError(
                f"Unknown task in selected subset {subset_id}."
            )

        for task_index, task in enumerate(
            TASKS
        ):
            matrix[
                row_index,
                task_index,
            ] = int(
                task in selected_tasks
            )

    return matrix


def plot_protocol_matrix(
    fold_table: pd.DataFrame,
) -> None:
    """Display the selected speech-task composition of every outer fold."""

    matrix = selected_task_matrix(
        fold_table
    )
    figure, axis = plt.subplots(
        figsize=(9.0, 5.2)
    )
    axis.imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(
            [
                "#F0F2F4",
                FULL_COLOR,
            ]
        ),
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for task_index in range(
            matrix.shape[1]
        ):
            if matrix[
                row_index,
                task_index,
            ]:
                axis.text(
                    task_index,
                    row_index,
                    "✓",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=18,
                )

    y_labels = [
        (
            f"Fold {int(row.outer_fold)} "
            f"({int(row.task_count)} "
            f"{'task' if int(row.task_count) == 1 else 'tasks'})"
        )
        for row in fold_table.itertuples(
            index=False
        )
    ]
    axis.set_xticks(
        range(len(TASKS)),
        TASKS,
    )
    axis.set_yticks(
        range(len(fold_table)),
        y_labels,
    )
    axis.set_xlabel("Speech task")
    axis.set_ylabel("Outer fold and selected burden")
    axis.set_title(
        "WavLM Selected Protocol Composition by Outer Fold"
    )
    axis.set_xticks(
        np.arange(-0.5, len(TASKS), 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(fold_table), 1),
        minor=True,
    )
    axis.grid(
        which="minor",
        color="white",
        linewidth=2,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    save_figure(
        figure,
        PROTOCOL_MATRIX_STEM,
    )


def plot_task_frequency(
    fold_table: pd.DataFrame,
) -> None:
    """Count task appearances across the five nested selections."""

    matrix = selected_task_matrix(
        fold_table
    )
    frequencies = matrix.sum(axis=0)
    figure, axis = plt.subplots(
        figsize=(8.8, 5.3)
    )
    bars = axis.bar(
        TASKS,
        frequencies,
        color=FULL_COLOR,
        width=0.72,
    )

    for bar, frequency in zip(
        bars,
        frequencies,
        strict=True,
    ):
        axis.annotate(
            str(int(frequency)),
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                frequency,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    axis.set_ylim(0, EXPECTED_OUTER_FOLDS + 0.6)
    axis.set_xlabel("Speech task")
    axis.set_ylabel("Number of outer folds selected")
    axis.set_title(
        "WavLM Task Selection Frequency Across Outer Folds"
    )
    axis.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)

    save_figure(
        figure,
        TASK_FREQUENCY_STEM,
    )


def plot_burden_performance(
    fold_table: pd.DataFrame,
) -> None:
    """Relate selected task burden to held-out performance difference."""

    figure, axis = plt.subplots(
        figsize=(8.5, 5.8)
    )
    task_counts = fold_table[
        "task_count"
    ].to_numpy(dtype=float)
    differences = fold_table[
        "outer_test_macro_f1_difference"
    ].to_numpy(dtype=float)
    label_offsets = {
        3: (6, 9),
        4: (6, -10),
    }

    axis.scatter(
        task_counts,
        differences,
        color=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=95,
        zorder=3,
    )

    for row in fold_table.itertuples(
        index=False
    ):
        axis.annotate(
            f"Fold {row.outer_fold}: {row.selected_subset}",
            xy=(
                row.task_count,
                row.outer_test_macro_f1_difference,
            ),
            xytext=label_offsets.get(
                int(row.outer_fold),
                (6, 5),
            ),
            textcoords="offset points",
            fontsize=8,
        )

    axis.axhline(
        0.0,
        color=NEUTRAL_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="Equal performance",
    )
    axis.axhline(
        -NONINFERIORITY_MARGIN,
        color=MARGIN_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Noninferiority margin",
    )
    axis.set_xticks(
        sorted(
            fold_table["task_count"].unique()
        )
    )
    axis.set_xlim(0.7, 2.3)
    axis.set_xlabel(
        "Number of tasks in nested-selected protocol"
    )
    axis.set_ylabel(
        "Selected minus full held-out Macro F1"
    )
    axis.set_title(
        "WavLM Protocol Burden and Held-Out Performance"
    )
    axis.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower right",
        frameon=True,
    )

    save_figure(
        figure,
        BURDEN_PERFORMANCE_STEM,
    )


def build_representation_comparison(
    wavlm_record: dict[str, object],
    handcrafted_record: dict[str, object],
) -> pd.DataFrame:
    """Build the same-cohort summary used by the comparison figure."""

    wavlm_comparison = wavlm_record[
        "selected_vs_full"
    ]
    handcrafted_comparison = handcrafted_record[
        "filtered_selected_vs_full"
    ]
    handcrafted_selections = handcrafted_record[
        "fold_selections"
    ]
    handcrafted_mean_tasks = float(
        np.mean(
            [
                float(row["task_count"])
                for row in handcrafted_selections
            ]
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "representation": (
                    "Handcrafted acoustic features"
                ),
                "participants": 100,
                "mean_selected_tasks": (
                    handcrafted_mean_tasks
                ),
                "selected_macro_f1_mean": float(
                    handcrafted_comparison[
                        "nested_selected_macro_f1_mean"
                    ]
                ),
                "full_macro_f1_mean": float(
                    handcrafted_comparison[
                        "full_macro_f1_mean"
                    ]
                ),
                "difference_mean": float(
                    handcrafted_comparison[
                        "difference_mean"
                    ]
                ),
                "point_inside_0_02_margin": (
                    float(
                        handcrafted_comparison[
                            "difference_mean"
                        ]
                    )
                    >= -NONINFERIORITY_MARGIN
                ),
                "formal_noninferiority_supported": (
                    "not_evaluated"
                ),
            },
            {
                "representation": "Frozen WavLM embeddings",
                "participants": 100,
                "mean_selected_tasks": float(
                    wavlm_record[
                        "selection_summary"
                    ]["mean_selected_task_count"]
                ),
                "selected_macro_f1_mean": float(
                    wavlm_comparison[
                        "selected_macro_f1_mean"
                    ]
                ),
                "full_macro_f1_mean": float(
                    wavlm_comparison[
                        "full_macro_f1_mean"
                    ]
                ),
                "difference_mean": float(
                    wavlm_comparison[
                        "difference_mean"
                    ]
                ),
                "point_inside_0_02_margin": bool(
                    wavlm_record[
                        "inference"
                    ]["point_estimate_inside_margin"]
                ),
                "formal_noninferiority_supported": str(
                    wavlm_record[
                        "inference"
                    ][
                        "formal_noninferiority_supported"
                    ]
                ).lower(),
            },
        ]
    )

    return comparison


def validate_representation_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Ensure the cross-representation summary is finite and complete."""

    if len(comparison) != 2:
        raise ValueError(
            "Expected two representation-comparison rows."
        )

    numeric_columns = [
        "participants",
        "mean_selected_tasks",
        "selected_macro_f1_mean",
        "full_macro_f1_mean",
        "difference_mean",
    ]

    if not np.isfinite(
        comparison[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "The representation comparison contains a non-finite value."
        )

    reproduced = (
        comparison["selected_macro_f1_mean"]
        - comparison["full_macro_f1_mean"]
    )

    if not np.allclose(
        reproduced,
        comparison["difference_mean"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The representation differences are inconsistent."
        )


def plot_representation_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Compare selected and full performance for both representations."""

    figure, axis = plt.subplots(
        figsize=(9.2, 5.9)
    )
    positions = np.arange(
        len(comparison)
    )
    width = 0.32
    full_bars = axis.bar(
        positions - width / 2,
        comparison["full_macro_f1_mean"],
        width=width,
        color=FULL_COLOR,
        label="Full assessment",
    )
    selected_bars = axis.bar(
        positions + width / 2,
        comparison[
            "selected_macro_f1_mean"
        ],
        width=width,
        color=SELECTED_COLOR,
        label="Nested selected",
    )

    for bars in (
        full_bars,
        selected_bars,
    ):
        for bar in bars:
            value = float(
                bar.get_height()
            )
            axis.annotate(
                f"{value:.3f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    burden_labels = [
        (
            f"{row.representation}\n"
            f"Mean selected tasks: "
            f"{row.mean_selected_tasks:.1f}"
        )
        for row in comparison.itertuples(
            index=False
        )
    ]
    axis.set_xticks(
        positions,
        burden_labels,
    )
    axis.set_ylim(
        0.0,
        max(
            comparison[
                "selected_macro_f1_mean"
            ].max(),
            comparison[
                "full_macro_f1_mean"
            ].max(),
        )
        + 0.12,
    )
    axis.set_ylabel(
        "Mean held-out Macro F1"
    )
    axis.set_title(
        "Duplicate-Filtered Results Across Feature Representations"
    )
    axis.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper left",
        frameon=True,
    )

    save_figure(
        figure,
        REPRESENTATION_COMPARISON_STEM,
    )


def save_comparison_table(
    comparison: pd.DataFrame,
) -> None:
    """Write the aggregate same-cohort table used by the paper figure."""

    COMPARISON_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison.to_csv(
        COMPARISON_TABLE_PATH,
        index=False,
    )


def print_plot_summary(
    fold_table: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Display the exact aggregate values represented by the figures."""

    print("WavLM plotting inputs validated.")
    print(
        f"Outer folds: {len(fold_table)}"
    )
    print(
        "Mean selected tasks: "
        f"{fold_table['task_count'].mean():.6f}"
    )
    print(
        "Selected WavLM Macro F1: "
        f"{fold_table['outer_test_selected_macro_f1'].mean():.6f}"
    )
    print(
        "Full WavLM Macro F1: "
        f"{fold_table['outer_test_full_macro_f1'].mean():.6f}"
    )
    print()
    print("Representation comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def print_generated_outputs() -> None:
    """List all figures and the aggregate comparison table."""

    print()
    print("Generated figures:")

    for stem in (
        FOLD_COMPARISON_STEM,
        BOOTSTRAP_STEM,
        PROTOCOL_MATRIX_STEM,
        TASK_FREQUENCY_STEM,
        BURDEN_PERFORMANCE_STEM,
        REPRESENTATION_COMPARISON_STEM,
    ):
        print(
            FIGURE_DIRECTORY
            / f"{stem}.png"
        )
        print(
            FIGURE_DIRECTORY
            / f"{stem}.pdf"
        )

    print()
    print("Generated aggregate table:")
    print(COMPARISON_TABLE_PATH)


def main() -> None:
    """Validate the experiment records and generate all WavLM figures."""

    configure_plot_style()
    (
        wavlm_record,
        bootstrap,
        handcrafted_record,
    ) = load_plot_inputs()
    fold_table = build_wavlm_fold_table(
        wavlm_record
    )

    validate_plot_inputs(
        wavlm_record=wavlm_record,
        fold_table=fold_table,
        bootstrap=bootstrap,
        handcrafted_record=handcrafted_record,
    )

    comparison = build_representation_comparison(
        wavlm_record=wavlm_record,
        handcrafted_record=handcrafted_record,
    )
    validate_representation_comparison(
        comparison
    )
    print_plot_summary(
        fold_table=fold_table,
        comparison=comparison,
    )

    plot_fold_comparison(
        fold_table
    )
    plot_bootstrap_distribution(
        bootstrap=bootstrap,
        record=wavlm_record,
    )
    plot_protocol_matrix(
        fold_table
    )
    plot_task_frequency(
        fold_table
    )
    plot_burden_performance(
        fold_table
    )
    plot_representation_comparison(
        comparison
    )
    save_comparison_table(
        comparison
    )
    print_generated_outputs()


if __name__ == "__main__":
    main()

"""Audit the frozen WavLM noninferiority result without rerunning model selection.

Uses the participant-paired bootstrap result already computed by
src/bootstrap_wavlm_task_selection.py. This is a reporting/consistency
check, NOT an independent confirmatory analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MANIFEST = Path("experiments/wavlm_task_selection.json")
DEFAULT_OUTPUT = Path("results/generated/wavlm_noninferiority_audit.json")
DEFAULT_FIGURE = Path("figures/wavlm_noninferiority_interval.pdf")
LOCKED_MARGIN = 0.02
TOLERANCE = 1e-9


def audit(record: dict) -> dict:
    design = record["design"]
    bootstrap = record["paired_bootstrap"]
    comparison = record["selected_vs_full"]
    if design["participants"] != 100 or design["outer_folds"] != 5:
        raise ValueError("Unexpected WavLM cohort or validation design.")
    if design["primary_metric"] != "macro_f1":
        raise ValueError("Expected Macro F1 as the primary outcome.")
    if abs(float(design["competitive_delta"]) - LOCKED_MARGIN) > TOLERANCE:
        raise ValueError("The predeclared selection tolerance changed.")
    if abs(float(bootstrap["noninferiority_margin"]) - LOCKED_MARGIN) > TOLERANCE:
        raise ValueError("The noninferiority margin does not match the locked design.")
    if int(bootstrap["bootstrap_replicates"]) != 10000:
        raise ValueError("Unexpected bootstrap replicate count.")

    point = float(bootstrap["difference_point"])
    lower = float(bootstrap["difference_one_sided_95_lower"])
    ci_low = float(bootstrap["difference_ci_95_lower"])
    ci_high = float(bootstrap["difference_ci_95_upper"])
    selected = float(bootstrap["selected_macro_f1_point"])
    full = float(bootstrap["full_macro_f1_point"])
    if not ci_low <= ci_high or not ci_low <= lower <= ci_high:
        raise ValueError("Bootstrap interval bounds are inconsistent.")
    if abs((selected - full) - point) > TOLERANCE:
        raise ValueError("The paired point estimate does not reconcile.")
    if abs(point - float(comparison["difference_mean"])) > TOLERANCE:
        raise ValueError("The manifest comparison does not reconcile.")
    supported = lower > -LOCKED_MARGIN
    if supported != bool(bootstrap["one_sided_bound_supports_noninferiority"]):
        raise ValueError("Existing noninferiority decision is inconsistent.")
    if supported != bool(record["inference"]["formal_noninferiority_supported"]):
        raise ValueError("Manifest inference status is inconsistent.")

    return {
        "analysis": "locked_wavlm_paired_bootstrap_noninferiority_audit",
        "role": "reproduced_decision_from_existing_manifest_not_new_test",
        "participants": 100,
        "outer_folds": 5,
        "metric": "mean_outer_fold_macro_f1",
        "selected_macro_f1": selected,
        "full_macro_f1": full,
        "difference_selected_minus_full": point,
        "margin": LOCKED_MARGIN,
        "one_sided_alpha": 0.05,
        "one_sided_95_lower_bound": lower,
        "two_sided_95_percentile_ci": [ci_low, ci_high],
        "noninferiority_supported": supported,
        "interpretation": (
            "Noninferiority at the locked margin is supported."
            if supported else
            "Noninferiority at the locked margin is NOT established. "
            "A point estimate inside the margin is insufficient."
        ),
        "limitations": [
            "A fixed dataset cannot yield additional independent patients by resampling.",
            "Exploring margins/models after seeing outer-test outcomes cannot "
            "turn this into a new confirmatory test.",
            "Bootstrap inference concerns this dataset and evaluation policy; "
            "external clinical validation is still required.",
        ],
    }


def plot_interval(summary: dict, destination: Path) -> None:
    lo, hi = summary["two_sided_95_percentile_ci"]
    point = summary["difference_selected_minus_full"]
    lower = summary["one_sided_95_lower_bound"]
    margin = -summary["margin"]
    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    ax.axvline(margin, ls="--", lw=1.5, color="darkred",
               label="Locked noninferiority boundary (−0.02)")
    ax.axvline(0, ls=":", lw=1, color="gray", label="No difference")
    ax.plot([lo, hi], [1, 1], linewidth=3, color="steelblue",
            label="Two-sided 95% paired bootstrap interval")
    ax.scatter([point], [1], s=45, color="navy", zorder=3,
               label="Observed paired difference")
    ax.scatter([lower], [0.75], marker="|", s=180, color="darkorange",
               label="One-sided 95% lower bound")
    ax.set_yticks([])
    ax.set_ylim(0.4, 1.4)
    ax.set_xlim(min(lo, margin) - .02, max(hi, 0) + .02)
    ax.set_xlabel("Macro F1 difference (selected − full)")
    ax.set_title("WavLM: statistical noninferiority not established"
                 if not summary["noninferiority_supported"]
                 else "WavLM: noninferiority supported")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29),
              fontsize=8, frameon=False, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    summary = audit(json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_interval(summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

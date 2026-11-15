"""Collect every number the paper cites into results/paper_numbers.md and results/paper_tables.tex.

Reads only the experiment records and generated tables, so it reflects the
latest run. The LaTeX tables follow the layout of the current report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import DELTA, EXPERIMENTS, GENERATED, TASKS

NUMBERS_PATH = Path("results/paper_numbers.md")
TABLES_PATH = Path("results/paper_tables.tex")


def load(name: str) -> dict:
    return json.loads((EXPERIMENTS / f"{name}.json").read_text())


def latex_table(label: str, caption: str, columns: str, header: list[str], rows: list[list[str]], footer: list[str] | None = None) -> str:
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\small",
        f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule",
        *[" & ".join(row) + " \\\\" for row in rows],
    ]
    if footer:
        lines += ["\\midrule", " & ".join(footer) + " \\\\"]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{value:.2f}\\%"


def main() -> None:
    baseline = load("core_baseline")
    discovery = load("exhaustive_task_search")
    discovery_boot = load("bootstrap_task_comparisons")
    nested = load("nested_task_selection")
    delta = load("delta_sensitivity")
    stability = load("protocol_stability")
    robustness = load("protocol_robustness")
    exclusion = load("duplicate_exclusion_sensitivity")
    policies = load("duplicate_policy_sensitivity")
    audit = load("raw_audio_audit")
    wavlm = load("wavlm_task_selection")
    pca = load("wavlm_pca_sensitivity")
    stable = load("wavlm_stability_aware_selection")
    final = load("final_stability_aware_wavlm")
    burden = load("recording_duration_burden")

    nested_folds = pd.DataFrame(nested["fold_selections"])
    full_folds = pd.DataFrame(nested["full_eight_task_comparison"]["fold_comparisons"])
    comparison = nested["full_eight_task_comparison"]
    overall = stability["overall_stability"]
    one_task = overall["task_count_distribution"]["1"]["selection_rate"]
    wb = wavlm["paired_bootstrap"]
    sb = stable["paired_bootstrap"]
    wavlm_folds = pd.DataFrame(wavlm["fold_results"]).sort_values("outer_fold")
    stable_folds = pd.DataFrame(stable["fold_results"]).sort_values("outer_fold")
    stable_selections = pd.read_csv(GENERATED / "wavlm_stability_aware_selections.csv")
    agg = burden["aggregate_results"]
    task_durations = pd.DataFrame(burden["task_duration_results"]).set_index("task")
    primary_reduction = 100 * (1 - wavlm["selection_summary"]["mean_selected_task_count"] / len(TASKS))

    tables = [
        latex_table(
            "nested", "Leakage-safe handcrafted nested task selection.", "clccc",
            ["Fold", "Selected tasks", "Count", "Selected F1", "Full F1"],
            [[str(r.outer_fold), r.selected_subset, str(r.task_count), f"{r.outer_test_macro_f1:.3f}", f"{f:.3f}"]
             for r, f in zip(nested_folds.itertuples(), full_folds["full_macro_f1"])],
            ["Mean", "--", f"{nested_folds['task_count'].mean():.1f}", f"{comparison['nested_selected_macro_f1_mean']:.3f}", f"{comparison['full_macro_f1_mean']:.3f}"],
        ),
        latex_table(
            "pca", "WavLM PCA sensitivity. The 16-component setting is the locked primary analysis.", "cccccc",
            ["PCA/task", "Mean tasks", "Selected F1", "Full F1", "Difference", "NI"],
            [[str(r["pca_components_per_task"]), f"{r['mean_selected_task_count']:.1f}", f"{r['selected_macro_f1_mean']:.3f}",
              f"{r['full_macro_f1_mean']:.3f}", f"{r['difference_mean']:+.3f}", "Yes" if r["formal_noninferiority_supported"] else "No"]
             for r in pca["comparison"]],
        ),
        latex_table(
            "stabilityaware", "Exploratory stability-aware WavLM follow-up.", "clccc",
            ["Fold", "Selected tasks", "Count", "Selected F1", "Full F1"],
            [[str(r.outer_fold), r.selected_subset, str(r.task_count), f"{r.outer_test_selected_macro_f1:.3f}", f"{r.outer_test_full_macro_f1:.3f}"]
             for r in stable_folds.itertuples()],
            ["Mean", "--", f"{stable['selection_summary']['mean_selected_tasks']:.2f}",
             f"{stable['selected_vs_full']['selected_macro_f1_mean']:.3f}", f"{stable['selected_vs_full']['full_macro_f1_mean']:.3f}"],
        ),
        latex_table(
            "wavlmpolicies", "Primary WavLM policy versus the stability-aware exploratory follow-up.", "lrrrrr",
            ["Policy", "Mean tasks", "Selected F1", "Difference", "1-sided lower", "Count reduction"],
            [
                ["Primary nested", f"{wavlm['selection_summary']['mean_selected_task_count']:.2f}", f"{wb['selected_macro_f1_point']:.3f}",
                 f"{wb['difference_point']:.4f}", f"{wb['difference_one_sided_95_lower']:.4f}", pct(primary_reduction)],
                ["Stability-aware", f"{stable['selection_summary']['mean_selected_tasks']:.2f}", f"{sb['selected_macro_f1_point']:.3f}",
                 f"{sb['difference_point']:.4f}", f"{sb['difference_one_sided_95_lower']:.4f}", pct(stable['selection_summary']['recording_count_reduction_percent'])],
            ],
        ),
        latex_table(
            "burden", "Observed burden for the full assessment and the WavLM nested-selected policy.", "lrrr",
            ["Measure", "Full", "Selected", "Reduction"],
            [
                [name, f"{agg[f'full_{key}_mean_per_participant']:.2f}", f"{agg[f'selected_{key}_mean_per_participant']:.2f}", pct(agg[f"{key}_reduction_percent"])]
                for name, key in [("Recordings / participant", "recording_count"), ("Recorded seconds", "recorded_seconds"),
                                  ("Endpoint-trimmed seconds", "endpoint_trimmed_seconds"), ("Nonsilent-seconds proxy", "nonsilent_seconds_proxy")]
            ],
        ),
    ]
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.write_text("% Generated by src/paper_numbers.py\n\n" + "\n".join(tables))

    representation = pd.DataFrame(
        [
            {
                "representation": "Handcrafted acoustic features",
                "participants": exclusion["design"]["remaining_participants"],
                "mean_selected_tasks": pd.DataFrame(exclusion["fold_selections"])["task_count"].mean(),
                "selected_macro_f1_mean": exclusion["filtered_selected_vs_full"]["nested_selected_macro_f1_mean"],
                "full_macro_f1_mean": exclusion["filtered_selected_vs_full"]["full_macro_f1_mean"],
                "difference_mean": exclusion["filtered_selected_vs_full"]["difference_mean"],
                "point_inside_0_02_margin": exclusion["filtered_selected_vs_full"]["difference_mean"] >= -DELTA,
                "formal_noninferiority_supported": "not_evaluated",
            },
            {
                "representation": "Frozen WavLM embeddings",
                "participants": wavlm["design"]["participants"],
                "mean_selected_tasks": wavlm["selection_summary"]["mean_selected_task_count"],
                "selected_macro_f1_mean": wavlm["selected_vs_full"]["selected_macro_f1_mean"],
                "full_macro_f1_mean": wavlm["selected_vs_full"]["full_macro_f1_mean"],
                "difference_mean": wavlm["selected_vs_full"]["difference_mean"],
                "point_inside_0_02_margin": wavlm["inference"]["point_estimate_inside_margin"],
                "formal_noninferiority_supported": str(wavlm["inference"]["formal_noninferiority_supported"]).lower(),
            },
        ]
    )
    representation.to_csv(EXPERIMENTS / "wavlm_vs_handcrafted_comparison.csv", index=False)

    ka_rate = overall["task_frequency"]["KA"]["selection_rate"]
    retention = [f"{f['original_selection']['subset_id']} {f['original_selection_retention_rate']:.1%}" for f in stability["per_outer_fold"]]
    boot = {k: v for k, v in discovery_boot["observed_results"].items()} | discovery_boot["paired_differences"]
    duplicate = audit["exact_duplicate_summary"]
    lines = [
        "# Numbers for the paper",
        "",
        "Generated by `src/paper_numbers.py` from `experiments/*.json`. LaTeX tables are in `results/paper_tables.tex`.",
        "",
        "## Cohort and baselines",
        f"- ALS participants: {baseline['population']['als_participants']} ({baseline['population']['normal_speech']} normal speech, {baseline['population']['documented_speech_impairment']} impaired)",
        f"- Majority-class Macro F1: {baseline['majority_class_baseline']['macro_f1_mean']:.4f}",
        f"- Full eight-task Macro F1: {baseline['full_eight_task_baseline']['macro_f1_mean']:.4f} (SD {baseline['full_eight_task_baseline']['macro_f1_std']:.4f})",
        "",
        "## Discovery (optimistic, outer-test comparison)",
        f"- Minimal competitive subset: {'+'.join(discovery['descriptive_results']['minimal_competitive_subset']['tasks'])} {discovery['descriptive_results']['minimal_competitive_subset']['macro_f1_mean']:.4f}",
        f"- Best observed subset: {'+'.join(discovery['descriptive_results']['best_observed_subset']['tasks'])} {discovery['descriptive_results']['best_observed_subset']['macro_f1_mean']:.4f}",
        f"- Competitive subsets at delta 0.01/0.02/0.05: " + "/".join(str(v) for v in discovery["descriptive_results"]["competitive_subset_counts"].values()),
        *[f"- {name}: {v.get('macro_f1', v.get('point_estimate')):.4f} [{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}]" for name, v in boot.items()],
        "",
        "## Handcrafted nested selection (primary)",
        f"- Selected {comparison['nested_selected_macro_f1_mean']:.4f} vs full {comparison['full_macro_f1_mean']:.4f}; difference {comparison['difference_mean']:+.4f}",
        f"- Selected protocols: {', '.join(nested_folds['selected_subset'])}",
        "- Delta sensitivity: " + "; ".join(
            f"delta {d}: F1 {r['aggregate_outer_test_metrics']['macro_f1']['mean']:.4f}, {r['mean_selected_task_count']:.1f} tasks"
            for d, r in delta["results_by_delta"].items()
        ),
        "",
        "## Protocol stability (50,000 resampled selections)",
        f"- One task {one_task:.2%}; at most two {overall['one_or_two_task_selection_rate']:.2%}; at most three {overall['one_to_three_task_selection_rate']:.2%}",
        f"- Full assessment selected {overall['full_assessment_selection_count']} times; {overall['unique_selected_protocols']} distinct protocols; KA in {ka_rate:.2%} of selections",
        f"- Mean retention of the original selection: {pd.Series([f['original_selection_retention_rate'] for f in stability['per_outer_fold']]).mean():.1%} ({', '.join(retention)})",
        f"- Fixed-protocol robustness: best exploratory candidate {robustness['interpretation']['best_exploratory_subset']}; reduced protocols supported in all folds: {robustness['results']['reduced_protocols_supported_in_all_folds']}",
        "",
        "## Duplicates",
        f"- {duplicate['duplicate_signal_groups']} duplicate signal groups, {duplicate['files_in_duplicate_signal_groups']} files, {duplicate['duplicate_connected_participants']} participants",
        f"- Duplicate-filtered handcrafted (n={exclusion['design']['remaining_participants']}): selected {exclusion['filtered_selected_vs_full']['nested_selected_macro_f1_mean']:.4f} vs full {exclusion['filtered_selected_vs_full']['full_macro_f1_mean']:.4f}",
        *[f"- Policy {name}: selected {p['selected_vs_reference']['selected_macro_f1_mean']:.4f} vs reference {p['selected_vs_reference']['reference_macro_f1_mean']:.4f}, {p['mean_selected_task_count']:.1f} tasks"
          for name, p in policies["policies"].items()],
        "",
        "## WavLM primary (16 PCs per task)",
        f"- Selected protocols: {', '.join(wavlm['selection_summary']['selected_protocols'])}; mean {wavlm['selection_summary']['mean_selected_task_count']:.1f} tasks",
        f"- Selected {wb['selected_macro_f1_point']:.4f} vs full {wb['full_macro_f1_point']:.4f}; difference {wb['difference_point']:+.4f}",
        f"- 95% CI [{wb['difference_ci_95_lower']:.4f}, {wb['difference_ci_95_upper']:.4f}]; one-sided lower {wb['difference_one_sided_95_lower']:.4f}; P(diff >= -{DELTA}) {wb['noninferiority_probability']:.4f}",
        f"- Formal noninferiority supported: {wavlm['inference']['formal_noninferiority_supported']}",
        f"- PCA settings with formal noninferiority: {pca['inference']['formal_noninferiority_supported_setting_count']} of {pca['inference']['total_pca_settings']}",
        "",
        "## WavLM stability-aware follow-up (exploratory)",
        f"- Selected protocols: {', '.join(stable['selection_summary']['selected_protocols'])}; {stable['selection_summary']['mean_selected_tasks']:.2f} tasks; count reduction {stable['selection_summary']['recording_count_reduction_percent']:.2f}%",
        f"- Worst inner-fold Macro F1 of selected protocols: {stable_selections['selected_inner_macro_f1_min'].min():.3f} to {stable_selections['selected_inner_macro_f1_min'].max():.3f}",
        f"- Selected {sb['selected_macro_f1_point']:.4f}; difference {sb['difference_point']:+.4f}; 95% CI [{sb['difference_ci_95_lower']:.4f}, {sb['difference_ci_95_upper']:.4f}]; one-sided lower {sb['difference_one_sided_95_lower']:.4f}; P(diff >= -{DELTA}) {sb['noninferiority_probability']:.4f}",
        f"- Second rule (worst gap to full, not in the paper): {', '.join(final['selection']['subsets'])}; selected {final['performance']['selected_macro_f1']:.4f}; one-sided lower {final['paired_bootstrap']['difference_one_sided_95_lower']:.4f}",
        "",
        "## Recording burden (WavLM policy, 100 participants)",
        f"- Recordings: {agg['full_recording_count_total']:.0f} -> {agg['selected_recording_count_total']:.0f} ({agg['recording_count_reduction_percent']:.2f}% fewer)",
        f"- Recorded seconds per participant: {agg['full_recorded_seconds_mean_per_participant']:.2f} -> {agg['selected_recorded_seconds_mean_per_participant']:.2f} ({agg['recorded_seconds_reduction_percent']:.2f}% less)",
        f"- Endpoint-trimmed: {agg['full_endpoint_trimmed_seconds_mean_per_participant']:.2f} -> {agg['selected_endpoint_trimmed_seconds_mean_per_participant']:.2f} ({agg['endpoint_trimmed_seconds_reduction_percent']:.2f}% less)",
        f"- Nonsilent proxy: {agg['full_nonsilent_seconds_proxy_mean_per_participant']:.2f} -> {agg['selected_nonsilent_seconds_proxy_mean_per_participant']:.2f} ({agg['nonsilent_seconds_proxy_reduction_percent']:.2f}% less)",
        "- Mean recorded seconds by task: " + ", ".join(f"{t} {task_durations.loc[t, 'recorded_seconds_mean']:.2f}" for t in TASKS),
        "",
    ]
    NUMBERS_PATH.write_text("\n".join(lines))
    print(f"Wrote {NUMBERS_PATH} and {TABLES_PATH}")


if __name__ == "__main__":
    main()

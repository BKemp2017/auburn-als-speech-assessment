"""Nested task selection on frozen WavLM embeddings (duplicate-filtered cohort of 100).

Per split and per task: standardize the 1,536-d embedding and fit PCA on the
training participants only. Concatenated task components feed a balanced L2
logistic regression whose C is tuned on the full eight-task inner CV.

Analyses:
  primary            16 components per task, the paper's locked result
  PCA sensitivity    8 and 32 components per task
  stability-aware    post hoc rule maximizing the worst inner-fold Macro F1 (paper)
  final stability    post hoc rule maximizing the worst inner-fold gap to full (not in paper)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BOOTSTRAP_REPLICATES,
    DELTA,
    EMBEDDINGS_PATH,
    EXPERIMENTS,
    FULL_ID,
    GENERATED,
    METRICS,
    SEED,
    TASKS,
    WAVLM_MODEL,
    WAVLM_REVISION,
    all_subsets,
    classification_metrics,
    duplicate_als_ids,
    inner_assignments,
    load_inputs,
    macro_f1,
    paired_fold_bootstrap,
    without_participants,
    write_csv,
    write_json,
)

PRIMARY_COMPONENTS = 16
SENSITIVITY_COMPONENTS = (8, 32)
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
MAX_STABILITY_TASKS = 3
INNER_COLUMNS = [f"inner_fold_{i}_macro_f1" for i in range(4)]


class Embeddings:
    def __init__(self) -> None:
        with np.load(EMBEDDINGS_PATH, allow_pickle=False) as saved:
            self.matrix = saved["embeddings"].astype(np.float32)
            self.row = {(p, t): i for i, (p, t) in enumerate(zip(saved["participant_ids"], saved["tasks"]))}
            self.participants = set(saved["participant_ids"].tolist())

    def project(self, train_ids, test_ids, components: int):
        """Per-task standardize + PCA fit on train_ids. Returns {task: (train, test)} and explained variance."""
        projected, explained = {}, {}
        for task in TASKS:
            train = self.matrix[[self.row[p, task] for p in train_ids]]
            test = self.matrix[[self.row[p, task] for p in test_ids]]
            scaler = StandardScaler()
            pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED)
            projected[task] = (
                pca.fit_transform(scaler.fit_transform(train)).astype(np.float32),
                pca.transform(scaler.transform(test)).astype(np.float32),
            )
            explained[task] = float(pca.explained_variance_ratio_.sum())
        return projected, explained


def classifier(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=SEED)),
        ]
    )


def stack(projected, tasks):
    return (
        np.concatenate([projected[t][0] for t in tasks], axis=1),
        np.concatenate([projected[t][1] for t in tasks], axis=1),
    )


class Design:
    """The duplicate-filtered cohort with its frozen folds."""

    def __init__(self) -> None:
        features, outer_folds, nested_folds = load_inputs()
        self.excluded = duplicate_als_ids()
        _, self.labels, self.nested_folds = without_participants(features, outer_folds, nested_folds, self.excluded)
        self.target = dict(zip(self.labels["participant_id"], self.labels["target"]))

    def targets(self, ids) -> np.ndarray:
        return np.array([self.target[p] for p in ids], dtype=np.int64)

    def outer_ids(self, outer_fold: int):
        is_test = self.labels["outer_fold"].eq(outer_fold)
        return tuple(self.labels.loc[~is_test, "participant_id"]), tuple(self.labels.loc[is_test, "participant_id"])


def inner_cv(design: Design, embeddings: Embeddings, components: int):
    """Tune C on the full assessment, then score every subset with that C. Returns tables and explained variance."""
    projections, explained = {}, []
    for outer_fold in range(5):
        for assignment in inner_assignments(design.nested_folds, outer_fold):
            train_ids, validation_ids = assignment["inner_train_ids"], assignment["inner_validation_ids"]
            projected, variance = embeddings.project(train_ids, validation_ids, components)
            projections[outer_fold, assignment["inner_fold"]] = (projected, design.targets(train_ids), design.targets(validation_ids))
            explained.append(variance)

    def scores(outer_fold, tasks, c):
        values = []
        for inner_fold in range(4):
            projected, y_train, y_validation = projections[outer_fold, inner_fold]
            x_train, x_validation = stack(projected, tasks)
            values.append(macro_f1(y_validation, classifier(c).fit(x_train, y_train).predict(x_validation)))
        series = pd.Series(values, dtype=float)
        return {**dict(zip(INNER_COLUMNS, values)), "inner_macro_f1_mean": float(series.mean()), "inner_macro_f1_std": float(series.std(ddof=1))}

    regularization = pd.DataFrame(
        [
            {"outer_fold": f, "regularization_c": c, "predictor_count": len(TASKS) * components, **scores(f, TASKS, c)}
            for f in range(5)
            for c in REGULARIZATION
        ]
    )
    chosen = {
        f: rows.sort_values(["inner_macro_f1_mean", "inner_macro_f1_std", "regularization_c"], ascending=[False, True, True], kind="mergesort").index[0]
        for f, rows in regularization.groupby("outer_fold")
    }
    regularization["selected"] = regularization.index.isin(list(chosen.values()))
    selected_c = {f: float(regularization.loc[i, "regularization_c"]) for f, i in chosen.items()}

    inner = pd.DataFrame(
        [
            {
                "outer_fold": f,
                "tasks": ",".join(tasks),
                "subset_id": "+".join(tasks),
                "task_count": len(tasks),
                "predictor_count": len(tasks) * components,
                "regularization_c": selected_c[f],
                **scores(f, tasks, selected_c[f]),
            }
            for f in range(5)
            for tasks in all_subsets()
        ]
    )
    return regularization, inner, explained


def select_primary(inner: pd.DataFrame):
    """Fewest tasks within DELTA of the full inner reference; then higher mean, lower std, subset id."""
    inner = inner.copy()
    rows = []
    for outer_fold, fold in inner.groupby("outer_fold", sort=True):
        reference = float(fold.loc[fold["subset_id"].eq(FULL_ID), "inner_macro_f1_mean"].iloc[0])
        competitive = fold["inner_macro_f1_mean"].ge(reference - DELTA)
        chosen = fold.loc[competitive].sort_values(
            ["task_count", "inner_macro_f1_mean", "inner_macro_f1_std", "subset_id"], ascending=[True, False, True, True], kind="mergesort"
        ).index[0]
        inner.loc[fold.index, "inner_full_reference"] = reference
        inner.loc[fold.index, "competitive_threshold"] = reference - DELTA
        inner.loc[fold.index, "competitive"] = competitive
        inner.loc[fold.index, "selected"] = fold.index == chosen
        inner.loc[fold.index, "competitive_subset_count"] = int(competitive.sum())
        rows.append(selection_row(inner.loc[chosen], reference, int(competitive.sum())))
    inner["competitive"] = inner["competitive"].astype(bool)
    inner["selected"] = inner["selected"].astype(bool)
    inner["competitive_subset_count"] = inner["competitive_subset_count"].astype(int)
    return inner, pd.DataFrame(rows)


def selection_row(row, reference: float, competitive_count: int, **extra) -> dict:
    return {
        "outer_fold": int(row["outer_fold"]),
        "selected_subset": row["subset_id"],
        "selected_tasks": row["tasks"],
        "task_count": int(row["task_count"]),
        "predictor_count": int(row["predictor_count"]),
        "regularization_c": float(row["regularization_c"]),
        **extra,
        "selected_inner_macro_f1_mean": float(row["inner_macro_f1_mean"]),
        "selected_inner_macro_f1_std": float(row["inner_macro_f1_std"]),
        "inner_full_reference": reference,
        "competitive_threshold": reference - DELTA,
        "competitive_subset_count": competitive_count,
        "delta": DELTA,
    }


def select_stability_aware(inner: pd.DataFrame):
    """Among 1-3 task competitive subsets, maximize the worst inner-fold Macro F1."""
    candidates = inner.copy()
    candidates["inner_macro_f1_min"] = candidates[INNER_COLUMNS].min(axis=1)
    candidates["stability_eligible"] = False
    candidates["stability_selected"] = False
    rows = []
    for outer_fold, fold in candidates.groupby("outer_fold", sort=True):
        reference = float(fold.loc[fold["subset_id"].eq(FULL_ID), "inner_macro_f1_mean"].iloc[0])
        eligible = fold.loc[fold["task_count"].le(MAX_STABILITY_TASKS) & fold["inner_macro_f1_mean"].ge(reference - DELTA)]
        chosen = eligible.sort_values(
            ["inner_macro_f1_min", "inner_macro_f1_mean", "inner_macro_f1_std", "task_count", "subset_id"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        ).index[0]
        candidates.loc[eligible.index, "stability_eligible"] = True
        candidates.loc[chosen, "stability_selected"] = True
        row = candidates.loc[chosen]
        rows.append(selection_row(row, reference, len(eligible), selected_inner_macro_f1_min=float(row["inner_macro_f1_min"])))
    return candidates, pd.DataFrame(rows)


def select_final_stability(inner: pd.DataFrame) -> pd.DataFrame:
    """Among 1-3 task competitive subsets, maximize the worst inner-fold difference from the full protocol."""
    rows = []
    for outer_fold, fold in inner.groupby("outer_fold", sort=True):
        full = fold.loc[fold["subset_id"].eq(FULL_ID)].iloc[0]
        reference = float(full["inner_macro_f1_mean"])
        competitive = fold.loc[fold["task_count"].between(1, MAX_STABILITY_TASKS) & fold["inner_macro_f1_mean"].ge(reference - DELTA - 1e-12)].copy()
        differences = competitive[INNER_COLUMNS].to_numpy(float) - full[INNER_COLUMNS].to_numpy(float)
        competitive["worst_inner_difference"] = differences.min(axis=1)
        row = competitive.sort_values(
            ["worst_inner_difference", "task_count", "inner_macro_f1_mean", "inner_macro_f1_std", "subset_id"],
            ascending=[False, True, False, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_subset": row["subset_id"],
                "selected_tasks": row["subset_id"],
                "task_count": int(row["task_count"]),
                "regularization_c": float(row["regularization_c"]),
                "selected_inner_macro_f1_mean": float(row["inner_macro_f1_mean"]),
                "selected_inner_macro_f1_std": float(row["inner_macro_f1_std"]),
                "inner_full_reference": reference,
                "competitive_threshold": reference - DELTA,
                "competitive_subset_count": len(competitive),
                "worst_inner_difference": float(row["worst_inner_difference"]),
                "delta": DELTA,
            }
        )
    return pd.DataFrame(rows)


def evaluate_outer(design: Design, embeddings: Embeddings, selections: pd.DataFrame, components: int):
    """Fit the selected and the full protocol on each outer training split and score the outer test."""
    metric_rows, prediction_frames = [], []
    for selection in selections.sort_values("outer_fold").to_dict(orient="records"):
        outer_fold = selection["outer_fold"]
        tasks = tuple(t for t in TASKS if t in selection["selected_subset"].split("+"))
        train_ids, test_ids = design.outer_ids(outer_fold)
        y_train, y_test = design.targets(train_ids), design.targets(test_ids)
        projected, _ = embeddings.project(train_ids, test_ids, components)
        results = {}
        for name, protocol in (("selected", tasks), ("full", TASKS)):
            x_train, x_test = stack(projected, protocol)
            model = classifier(selection["regularization_c"]).fit(x_train, y_train)
            results[name] = (model.predict(x_test), model.predict_proba(x_test)[:, 1])
        selected_metrics = classification_metrics(y_test, *results["selected"])
        full_metrics = classification_metrics(y_test, *results["full"])
        metric_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_subset": selection["selected_subset"],
                "selected_tasks": ",".join(tasks),
                "task_count": len(tasks),
                "selected_predictor_count": len(tasks) * components,
                "full_predictor_count": len(TASKS) * components,
                "training_participants": len(train_ids),
                "test_participants": len(test_ids),
                "regularization_c": selection["regularization_c"],
                **{k: selection[k] for k in ("selected_inner_macro_f1_mean", "selected_inner_macro_f1_std", "inner_full_reference", "competitive_threshold", "competitive_subset_count", "delta")},
                **{f"outer_test_selected_{m}": selected_metrics[m] for m in METRICS},
                **{f"outer_test_full_{m}": full_metrics[m] for m in METRICS},
                "outer_test_macro_f1_difference": selected_metrics["macro_f1"] - full_metrics["macro_f1"],
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "participant_id": test_ids,
                    "outer_fold": outer_fold,
                    "selected_subset": selection["selected_subset"],
                    "true_label": y_test,
                    "selected_predicted_label": results["selected"][0].astype(int),
                    "selected_score": results["selected"][1],
                    "full_predicted_label": results["full"][0].astype(int),
                    "full_score": results["full"][1],
                }
            )
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def paired_bootstrap(fold_metrics: pd.DataFrame, predictions: pd.DataFrame):
    columns = ["selected_predicted_label", "full_predicted_label"]
    tables = [rows.reset_index(drop=True) for _, rows in predictions.groupby("outer_fold", sort=True)]
    draws = paired_fold_bootstrap(tables, columns)
    boot = pd.DataFrame({"bootstrap_id": np.arange(BOOTSTRAP_REPLICATES), "selected_macro_f1": draws[:, 0], "full_macro_f1": draws[:, 1]})
    boot["difference"] = boot["selected_macro_f1"] - boot["full_macro_f1"]

    differences = boot["difference"].to_numpy()
    selected = float(fold_metrics["outer_test_selected_macro_f1"].mean())
    full = float(fold_metrics["outer_test_full_macro_f1"].mean())
    lower, upper = np.quantile(differences, [0.025, 0.975])
    one_sided = float(np.quantile(differences, 0.05))
    summary = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "noninferiority_margin": DELTA,
        "selected_macro_f1_point": selected,
        "full_macro_f1_point": full,
        "difference_point": selected - full,
        "difference_bootstrap_mean": float(differences.mean()),
        "difference_ci_95_lower": float(lower),
        "difference_ci_95_upper": float(upper),
        "difference_one_sided_95_lower": one_sided,
        "noninferiority_probability": float(np.mean(differences >= -DELTA)),
        "point_estimate_inside_margin": bool(selected - full >= -DELTA),
        "two_sided_ci_supports_noninferiority": bool(lower >= -DELTA),
        "one_sided_bound_supports_noninferiority": bool(one_sided >= -DELTA),
        "folds_inside_margin": int(fold_metrics["outer_test_macro_f1_difference"].ge(-DELTA).sum()),
    }
    return boot, summary


def metric_summary(fold_metrics: pd.DataFrame) -> dict:
    return {
        protocol: {
            m: {
                "mean": float(fold_metrics[f"outer_test_{protocol}_{m}"].mean()),
                "std": float(fold_metrics[f"outer_test_{protocol}_{m}"].to_numpy(float).std(ddof=1)),
            }
            for m in METRICS
        }
        for protocol in ("selected", "full")
    }


def selection_record(design, components, regularization, fold_metrics, explained, bootstrap) -> dict:
    metrics = metric_summary(fold_metrics)
    selected, full = metrics["selected"]["macro_f1"]["mean"], metrics["full"]["macro_f1"]["mean"]
    task_frequency = {t: int(fold_metrics["selected_tasks"].str.split(",").map(lambda ts: t in ts).sum()) for t in TASKS}
    return {
        "experiment": "duplicate_filtered_wavlm_nested_task_selection",
        "participant_level_fields": [],
        "design": {
            "participants": len(design.labels),
            "excluded_duplicate_connected_participants": len(design.excluded),
            "speech_tasks": list(TASKS),
            "task_subset_count": 255,
            "outer_folds": 5,
            "inner_folds_per_outer_fold": 4,
            "primary_metric": "macro_f1",
            "competitive_delta": DELTA,
            "selection_rule": "fewest tasks within delta of the full inner-CV reference, then higher mean Macro F1, lower standard deviation, and subset identifier",
        },
        "representation": {
            "encoder": WAVLM_MODEL,
            "encoder_revision": WAVLM_REVISION,
            "frozen_encoder": True,
            "embedding_dimension_per_task": 1536,
            "pca_components_per_task": components,
            "pca_fit_scope": "training participants only within each split",
            "inner_fold_explained_variance": {
                t: {
                    "minimum": float(np.min([v[t] for v in explained])),
                    "mean": float(np.mean([v[t] for v in explained])),
                    "maximum": float(np.max([v[t] for v in explained])),
                }
                for t in TASKS
            },
        },
        "classifier": {
            "type": "balanced_l2_logistic_regression",
            "solver": "liblinear",
            "regularization_candidates": list(REGULARIZATION),
            "regularization_selection_scope": "full-assessment inner CV within each outer fold",
            "selected_regularization": regularization.loc[
                regularization["selected"], ["outer_fold", "regularization_c", "inner_macro_f1_mean", "inner_macro_f1_std"]
            ].to_dict(orient="records"),
        },
        "selection_summary": {
            "mean_selected_task_count": float(fold_metrics["task_count"].mean()),
            "selected_protocols": fold_metrics["selected_subset"].tolist(),
            "task_frequency": task_frequency,
        },
        "fold_results": fold_metrics.drop(columns="selected_tasks").to_dict(orient="records"),
        "outer_test_metrics": metrics,
        "selected_vs_full": {"selected_macro_f1_mean": selected, "full_macro_f1_mean": full, "difference_mean": selected - full},
        "paired_bootstrap": bootstrap,
        "inference": {
            "point_estimate_inside_margin": bootstrap["point_estimate_inside_margin"],
            "formal_noninferiority_supported": bootstrap["one_sided_bound_supports_noninferiority"],
            "interpretation": "The reduced WavLM protocols retained similar average performance, but the paired bootstrap did not establish noninferiority at the prespecified margin.",
        },
    }


def run_setting(design, embeddings, components):
    regularization, inner, explained = inner_cv(design, embeddings, components)
    inner, selections = select_primary(inner)
    fold_metrics, predictions = evaluate_outer(design, embeddings, selections, components)
    boot, bootstrap = paired_bootstrap(fold_metrics, predictions)
    record = selection_record(design, components, regularization, fold_metrics, explained, bootstrap)
    print(f"{components} PCs: {record['selection_summary']['selected_protocols']}  "
          f"selected {bootstrap['selected_macro_f1_point']:.4f} vs full {bootstrap['full_macro_f1_point']:.4f}  "
          f"one-sided lower {bootstrap['difference_one_sided_95_lower']:+.4f}")
    return {"regularization": regularization, "inner": inner, "fold_metrics": fold_metrics, "predictions": predictions, "bootstrap": boot, "record": record}


def comparison_row(components: int, role: str, record: dict) -> dict:
    bootstrap = record["paired_bootstrap"]
    return {
        "pca_components_per_task": components,
        "analysis_role": role,
        "mean_selected_task_count": record["selection_summary"]["mean_selected_task_count"],
        "selected_protocols": ",".join(record["selection_summary"]["selected_protocols"]),
        "selected_macro_f1_mean": record["selected_vs_full"]["selected_macro_f1_mean"],
        "full_macro_f1_mean": record["selected_vs_full"]["full_macro_f1_mean"],
        "difference_mean": record["selected_vs_full"]["difference_mean"],
        "difference_one_sided_95_lower": bootstrap["difference_one_sided_95_lower"],
        "noninferiority_probability": bootstrap["noninferiority_probability"],
        "point_estimate_inside_margin": bootstrap["point_estimate_inside_margin"],
        "formal_noninferiority_supported": record["inference"]["formal_noninferiority_supported"],
        "folds_inside_margin": bootstrap["folds_inside_margin"],
    }


def pca_sensitivity(design, embeddings, primary: dict) -> None:
    results = {c: run_setting(design, embeddings, c) for c in SENSITIVITY_COMPONENTS}
    for name in ("regularization", "inner", "fold_metrics", "predictions", "bootstrap"):
        combined = pd.concat([r[name].assign(**{"pca_components_per_task": c}) for c, r in results.items()], ignore_index=True)
        combined = combined[["pca_components_per_task", *[col for col in combined.columns if col != "pca_components_per_task"]]]
        write_csv(combined, f"wavlm_pca_sensitivity_{'inner_results' if name == 'inner' else name}.csv")

    comparison = pd.DataFrame(
        [comparison_row(PRIMARY_COMPONENTS, "locked_primary", primary["record"])]
        + [comparison_row(c, "prespecified_sensitivity", r["record"]) for c, r in results.items()]
    ).sort_values("pca_components_per_task")
    scenarios = []
    for c, r in results.items():
        record = r["record"]
        supported = record["inference"]["formal_noninferiority_supported"]
        record["experiment"] = "duplicate_filtered_wavlm_pca_sensitivity_scenario"
        record["sensitivity_role"] = "prespecified_representation_dimension_check"
        record["inference"]["interpretation"] = (
            f"With {c} PCA components per task, the paired participant bootstrap "
            f"{'supported noninferiority' if supported else 'did not establish noninferiority'} at the prespecified margin. "
            "This is a prespecified representation-sensitivity result, not a new choice made from outer-test performance."
        )
        scenarios.append(record)

    primary_protocols = primary["record"]["selection_summary"]["selected_protocols"]
    max_tasks = max(int(x) for r in [primary, *results.values()] for x in r["fold_metrics"]["task_count"])
    supported_count = int(comparison["formal_noninferiority_supported"].sum())
    burden = (
        "The low-recording-count selection pattern persisted across the locked 16-component analysis and both prespecified PCA sensitivity settings."
        if max_tasks <= MAX_STABILITY_TASKS
        else "At least one PCA sensitivity setting did not preserve the one-to-three-task selection pattern."
    )
    write_json(
        EXPERIMENTS / "wavlm_pca_sensitivity.json",
        {
            "experiment": "wavlm_pca_dimension_sensitivity",
            "participant_level_fields": [],
            "design": {
                "primary_pca_components_per_task": PRIMARY_COMPONENTS,
                "sensitivity_pca_components_per_task": list(SENSITIVITY_COMPONENTS),
                "participants": len(design.labels),
                "outer_folds": 5,
                "task_subset_count": 255,
                "competitive_delta": DELTA,
                "bootstrap_replicates_per_setting": BOOTSTRAP_REPLICATES,
                "analysis_rule": "The 16-component analysis remains primary. The 8- and 32-component settings are prespecified sensitivity checks and are not selected using outer-test results.",
            },
            "representation": {
                "encoder": WAVLM_MODEL,
                "encoder_revision": WAVLM_REVISION,
                "frozen_encoder": True,
                "pca_fit_scope": "training participants only within each split",
            },
            "comparison": comparison.to_dict(orient="records"),
            "protocol_agreement_with_primary": {
                str(c): {
                    "matching_outer_folds": sum(a == b for a, b in zip(primary_protocols, r["record"]["selection_summary"]["selected_protocols"])),
                    "total_outer_folds": 5,
                    "exact_match_rate": sum(a == b for a, b in zip(primary_protocols, r["record"]["selection_summary"]["selected_protocols"])) / 5,
                    "selected_protocols": r["record"]["selection_summary"]["selected_protocols"],
                }
                for c, r in results.items()
            },
            "sensitivity_scenarios": scenarios,
            "inference": {
                "low_burden_selection_robust": max_tasks <= MAX_STABILITY_TASKS,
                "maximum_selected_tasks_across_pca_settings": max_tasks,
                "all_point_estimates_inside_margin": bool(comparison["point_estimate_inside_margin"].all()),
                "formal_noninferiority_supported_setting_count": supported_count,
                "total_pca_settings": len(comparison),
                "interpretation": f"{burden} Formal noninferiority was supported in {supported_count} of {len(comparison)} PCA settings. The sensitivity analysis therefore describes representation robustness but does not replace the locked primary inference.",
            },
        },
        sort_keys=True,
    )


def stability_aware(design, embeddings, primary: dict) -> None:
    candidates, selections = select_stability_aware(primary["inner"])
    fold_metrics, predictions = evaluate_outer(design, embeddings, selections, PRIMARY_COMPONENTS)
    boot, bootstrap = paired_bootstrap(fold_metrics, predictions)
    write_csv(candidates, "wavlm_stability_aware_candidates.csv")
    write_csv(selections, "wavlm_stability_aware_selections.csv")
    write_csv(fold_metrics, "wavlm_stability_aware_fold_metrics.csv")
    write_csv(predictions, "wavlm_stability_aware_predictions.csv")
    write_csv(boot, "wavlm_stability_aware_bootstrap.csv")

    participants = len(design.labels)
    selected_recordings = float((fold_metrics["task_count"] * fold_metrics["test_participants"]).sum())
    full_recordings = float(participants * len(TASKS))
    selected, full = float(fold_metrics["outer_test_selected_macro_f1"].mean()), float(fold_metrics["outer_test_full_macro_f1"].mean())
    write_json(
        EXPERIMENTS / "wavlm_stability_aware_selection.json",
        {
            "experiment": "exploratory_wavlm_stability_aware_selection",
            "analysis_role": "post_hoc_exploratory_follow_up",
            "participant_level_fields": [],
            "design": {
                "participants": participants,
                "outer_folds": 5,
                "inner_folds_per_outer_fold": 4,
                "pca_components_per_task": PRIMARY_COMPONENTS,
                "primary_metric": "macro_f1",
                "competitive_delta": DELTA,
                "maximum_selected_tasks": MAX_STABILITY_TASKS,
                "selection_rule": "Among one-to-three-task protocols whose mean inner Macro F1 is within 0.02 of the full inner-CV reference, maximize the minimum Macro F1 across the four inner folds; ties prefer higher mean Macro F1, lower standard deviation, fewer tasks, and then lexicographic subset ID.",
                "outer_test_rule": "Each selected protocol is evaluated once on its outer-test fold after selection using inner-CV results only.",
            },
            "selection_summary": {
                "selected_protocols": fold_metrics["selected_subset"].tolist(),
                "mean_selected_tasks": selected_recordings / participants,
                "selected_recordings": selected_recordings,
                "full_recordings": full_recordings,
                "recording_count_reduction_percent": 100.0 * (1.0 - selected_recordings / full_recordings),
            },
            "fold_results": fold_metrics.drop(columns="selected_tasks").to_dict(orient="records"),
            "selected_vs_full": {"selected_macro_f1_mean": selected, "full_macro_f1_mean": full, "difference_mean": selected - full},
            "paired_bootstrap": bootstrap,
            "inference": {
                "exploratory_noninferiority_criterion_met": bootstrap["one_sided_bound_supports_noninferiority"],
                "interpretation": "This post hoc stability-aware procedure is an exploratory follow-up because its design was informed by earlier outer-test results. Meeting the statistical criterion would support future independent validation but would not independently establish noninferiority.",
            },
        },
        sort_keys=True,
    )
    print(f"Stability-aware: {fold_metrics['selected_subset'].tolist()}  selected {selected:.4f}  one-sided lower {bootstrap['difference_one_sided_95_lower']:+.4f}")


def final_stability(design, embeddings, primary: dict) -> None:
    selections = select_final_stability(primary["inner"])
    fold_metrics, predictions = evaluate_outer(design, embeddings, selections, PRIMARY_COMPONENTS)
    boot, bootstrap = paired_bootstrap(fold_metrics, predictions)
    write_csv(selections, "final_stability_aware_selections.csv")
    write_csv(fold_metrics, "final_stability_aware_fold_metrics.csv")
    write_csv(predictions, "final_stability_aware_predictions.csv")
    write_csv(boot, "final_stability_aware_bootstrap.csv")
    mean_tasks = float(fold_metrics["task_count"].mean())
    write_json(
        EXPERIMENTS / "final_stability_aware_wavlm.json",
        {
            "experiment": "final_exploratory_stability_aware_wavlm_selection",
            "status": "EXPLORATORY_ONLY_NOT_INDEPENDENT_CONFIRMATION",
            "population": {"als_participants": len(design.labels), "excluded_duplicate_participants": len(design.excluded)},
            "design": {
                "outer_folds": 5,
                "inner_folds": 4,
                "pca_components_per_task": PRIMARY_COMPONENTS,
                "original_margin": DELTA,
                "maximum_reduced_tasks": MAX_STABILITY_TASKS,
                "primary_metric": "mean_outer_fold_macro_f1",
                "fixed_selection_rule": "Among 1-to-3-task subsets within 0.02 mean inner Macro F1 of full, maximize the worst of four inner-fold paired differences; tie-break by fewer tasks, higher inner mean, lower inner standard deviation, then subset identifier.",
                "original_outer_folds_reused": True,
                "original_outer_test_results_previously_inspected": True,
            },
            "selection": {
                "subsets": selections["selected_subset"].tolist(),
                "task_counts": selections["task_count"].tolist(),
                "mean_selected_task_count": mean_tasks,
                "recording_count_reduction_percent": (1 - mean_tasks / 8) * 100,
                "new_protocol_recording_duration_not_measured": True,
            },
            "performance": {
                "selected_macro_f1": float(fold_metrics["outer_test_selected_macro_f1"].mean()),
                "full_macro_f1": float(fold_metrics["outer_test_full_macro_f1"].mean()),
                "original_locked_selected_macro_f1": primary["record"]["outer_test_metrics"]["selected"]["macro_f1"]["mean"],
            },
            "paired_bootstrap": bootstrap,
            "interpretation": "Passing an exploratory bound does not establish confirmatory noninferiority on previously inspected outer-test participants. If the one-sided lower bound is at or below -0.02, even this exploratory analysis fails to support noninferiority.",
        },
    )
    print(f"Final stability-aware: {selections['selected_subset'].tolist()}  one-sided lower {bootstrap['difference_one_sided_95_lower']:+.4f}")


def noninferiority_audit(record: dict) -> None:
    bootstrap = record["paired_bootstrap"]
    supported = bootstrap["difference_one_sided_95_lower"] > -DELTA
    write_json(
        GENERATED / "wavlm_noninferiority_audit.json",
        {
            "analysis": "locked_wavlm_paired_bootstrap_noninferiority_audit",
            "role": "reproduced_decision_from_existing_manifest_not_new_test",
            "participants": record["design"]["participants"],
            "outer_folds": 5,
            "metric": "mean_outer_fold_macro_f1",
            "selected_macro_f1": bootstrap["selected_macro_f1_point"],
            "full_macro_f1": bootstrap["full_macro_f1_point"],
            "difference_selected_minus_full": bootstrap["difference_point"],
            "margin": DELTA,
            "one_sided_alpha": 0.05,
            "one_sided_95_lower_bound": bootstrap["difference_one_sided_95_lower"],
            "two_sided_95_percentile_ci": [bootstrap["difference_ci_95_lower"], bootstrap["difference_ci_95_upper"]],
            "noninferiority_supported": supported,
            "interpretation": "Noninferiority at the locked margin is supported." if supported else "Noninferiority at the locked margin is NOT established. A point estimate inside the margin is insufficient.",
            "limitations": [
                "A fixed dataset cannot yield additional independent patients by resampling.",
                "Exploring margins/models after seeing outer-test outcomes cannot turn this into a new confirmatory test.",
                "Bootstrap inference concerns this dataset and evaluation policy; external clinical validation is still required.",
            ],
        },
    )


def main() -> None:
    design = Design()
    embeddings = Embeddings()
    if embeddings.participants != set(design.labels["participant_id"]):
        raise ValueError("WavLM embeddings do not cover exactly the duplicate-filtered cohort; rerun wavlm_embeddings.py.")

    primary = run_setting(design, embeddings, PRIMARY_COMPONENTS)
    write_csv(primary["regularization"], "wavlm_task_selection_regularization.csv")
    write_csv(primary["inner"], "wavlm_task_selection_inner_results.csv")
    write_csv(primary["fold_metrics"], "wavlm_task_selection_fold_metrics.csv")
    write_csv(primary["predictions"], "wavlm_task_selection_predictions.csv")
    write_csv(primary["bootstrap"], "wavlm_task_selection_bootstrap.csv")
    write_json(EXPERIMENTS / "wavlm_task_selection.json", primary["record"], sort_keys=True)
    noninferiority_audit(primary["record"])

    pca_sensitivity(design, embeddings, primary)
    stability_aware(design, embeddings, primary)
    final_stability(design, embeddings, primary)


if __name__ == "__main__":
    main()

"""Shared leakage-safe mechanics for frozen WavLM task selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_duplicate_impact import normalize_participant_id
from audit_metadata import TASKS
from evaluate_baselines import load_experiment_inputs
from evaluate_duplicate_exclusion_sensitivity import (
    WAVEFORM_AUDIT_PATH,
    filter_modeling_inputs,
    filter_nested_folds,
    identify_duplicate_connected_als_ids,
    validate_filtered_inputs,
)

EMBEDDINGS_PATH = Path(
    "results/generated/raw_audio_wavlm_embeddings.npz"
)
EMBEDDING_RECORD_PATH = Path(
    "experiments/raw_audio_wavlm_embeddings.json"
)

MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_REVISION = (
    "4c66d4806a428f2e922ccfa1a962776e232d487b"
)

EXPECTED_PARTICIPANTS = 100
EXPECTED_RECORDINGS = EXPECTED_PARTICIPANTS * len(TASKS)
EXPECTED_EMBEDDING_DIMENSION = 1_536
EXPECTED_TARGET_COUNTS = {0: 52, 1: 48}
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS_PER_OUTER_FOLD = 4
PCA_COMPONENTS_PER_TASK = 16
RANDOM_SEED = 2026
REGULARIZATION_VALUES = (
    0.01,
    0.1,
    1.0,
    10.0,
)
MAXIMUM_CLASSIFIER_ITERATIONS = 10_000
PRIMARY_DELTA = 0.02
EXPECTED_TASK_SUBSETS = 255
@dataclass(frozen=True)
class WavLMEmbeddingArtifact:
    """Keep the aligned recording-level arrays in one validated object."""

    embeddings: np.ndarray
    participant_ids: np.ndarray
    tasks: np.ndarray
    targets: np.ndarray
    relative_paths: np.ndarray

    @property
    def recording_count(self) -> int:
        """Return the number of recording rows in the artifact."""

        return int(self.embeddings.shape[0])

    @property
    def embedding_dimension(self) -> int:
        """Return the number of learned features per recording."""

        return int(self.embeddings.shape[1])

    @property
    def participant_count(self) -> int:
        """Return the number of distinct retained participants."""

        return int(np.unique(self.participant_ids).size)

@dataclass(frozen=True)
class ProjectedInnerFold:
    """Hold leakage-safe task projections for one inner-CV split."""

    outer_fold: int
    inner_fold: int
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    training_targets: np.ndarray
    validation_targets: np.ndarray
    training_features_by_task: dict[str, np.ndarray]
    validation_features_by_task: dict[str, np.ndarray]
    explained_variance_by_task: dict[str, float]

@dataclass(frozen=True)
class ProjectedOuterFold:
    """Hold task projections fitted on one complete outer-training fold."""

    outer_fold: int
    training_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    training_targets: np.ndarray
    test_targets: np.ndarray
    training_features_by_task: dict[str, np.ndarray]
    test_features_by_task: dict[str, np.ndarray]
    explained_variance_by_task: dict[str, float]

def load_wavlm_artifact() -> tuple[
    WavLMEmbeddingArtifact,
    dict[str, object],
]:
    """Load the private embeddings and their tracked aggregate record."""

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Run src/extract_raw_audio_embeddings.py before this "
            f"analysis. Missing: {EMBEDDINGS_PATH}"
        )

    if not EMBEDDING_RECORD_PATH.exists():
        raise FileNotFoundError(
            "The tracked WavLM extraction record was not found at "
            f"{EMBEDDING_RECORD_PATH}."
        )

    expected_arrays = {
        "embeddings",
        "participant_ids",
        "tasks",
        "targets",
        "relative_paths",
    }

    # Pickled object arrays are unnecessary here and would make loading
    # participant-linked data less safe than fixed numeric and text arrays.
    with np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ) as saved_artifact:
        if set(saved_artifact.files) != expected_arrays:
            raise ValueError(
                "The WavLM artifact contains unexpected arrays: "
                f"{sorted(saved_artifact.files)}"
            )

        artifact = WavLMEmbeddingArtifact(
            embeddings=np.asarray(
                saved_artifact["embeddings"],
                dtype=np.float32,
            ).copy(),
            participant_ids=np.asarray(
                [
                    normalize_participant_id(value)
                    for value in saved_artifact[
                        "participant_ids"
                    ]
                ],
                dtype=str,
            ),
            tasks=np.asarray(
                saved_artifact["tasks"],
                dtype=str,
            ).copy(),
            targets=np.asarray(
                saved_artifact["targets"],
                dtype=np.int64,
            ).copy(),
            relative_paths=np.asarray(
                saved_artifact["relative_paths"],
                dtype=str,
            ).copy(),
        )

    with EMBEDDING_RECORD_PATH.open(
        encoding="utf-8"
    ) as input_file:
        extraction_record = json.load(
            input_file
        )

    return artifact, extraction_record

def load_duplicate_filtered_design() -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    set[str],
]:
    """Rebuild the duplicate-filtered labels and frozen nested folds."""

    features, labels, nested_folds = (
        load_experiment_inputs()
    )

    excluded_ids = identify_duplicate_connected_als_ids(
        WAVEFORM_AUDIT_PATH
    )

    filtered_features, filtered_labels = (
        filter_modeling_inputs(
            features=features,
            labels=labels,
            excluded_ids=excluded_ids,
        )
    )

    filtered_nested_folds = filter_nested_folds(
        nested_folds=nested_folds,
        excluded_ids=excluded_ids,
    )

    # Reuse the established sensitivity-analysis validation instead of
    # maintaining a second implementation of the same fold invariants.
    validate_filtered_inputs(
        original_features=features,
        filtered_features=filtered_features,
        filtered_labels=filtered_labels,
        filtered_nested_folds=filtered_nested_folds,
        excluded_ids=excluded_ids,
    )

    return (
        filtered_labels,
        filtered_nested_folds,
        excluded_ids,
    )

def validate_wavlm_inputs(
    artifact: WavLMEmbeddingArtifact,
    extraction_record: dict[str, object],
    labels: pd.DataFrame,
    nested_folds: list[dict[str, object]],
    excluded_ids: set[str],
) -> None:
    """Confirm embeddings, labels, and frozen folds describe one cohort."""

    if artifact.embeddings.ndim != 2:
        raise ValueError(
            "The WavLM embedding matrix must be two-dimensional."
        )

    expected_shape = (
        EXPECTED_RECORDINGS,
        EXPECTED_EMBEDDING_DIMENSION,
    )

    if artifact.embeddings.shape != expected_shape:
        raise ValueError(
            "Expected WavLM embedding shape "
            f"{expected_shape}, found {artifact.embeddings.shape}."
        )

    aligned_arrays = {
        "participant_ids": artifact.participant_ids,
        "tasks": artifact.tasks,
        "targets": artifact.targets,
        "relative_paths": artifact.relative_paths,
    }

    for name, values in aligned_arrays.items():
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional array."
            )

        if len(values) != EXPECTED_RECORDINGS:
            raise ValueError(
                f"{name} contains {len(values)} rows; expected "
                f"{EXPECTED_RECORDINGS}."
            )

    if not np.isfinite(
        artifact.embeddings
    ).all():
        raise ValueError(
            "The WavLM artifact contains a non-finite embedding value."
        )

    if not np.issubdtype(
        artifact.embeddings.dtype,
        np.floating,
    ):
        raise ValueError(
            "The WavLM embeddings must use a floating-point dtype."
        )

    if artifact.participant_count != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, found "
            f"{artifact.participant_count}."
        )

    if set(artifact.tasks) != set(TASKS):
        raise ValueError(
            "The embedding artifact does not contain the expected tasks."
        )

    participant_task_pairs = list(
        zip(
            artifact.participant_ids,
            artifact.tasks,
            strict=True,
        )
    )

    if len(set(participant_task_pairs)) != EXPECTED_RECORDINGS:
        raise ValueError(
            "A participant-task pair is duplicated in the embeddings."
        )

    if len(set(artifact.relative_paths)) != EXPECTED_RECORDINGS:
        raise ValueError(
            "A recording path is duplicated in the embeddings."
        )

    if any(
        Path(path).is_absolute()
        for path in artifact.relative_paths
    ):
        raise ValueError(
            "The private embedding artifact contains an absolute path."
        )

    artifact_ids = set(
        artifact.participant_ids
    )
    label_ids = {
        normalize_participant_id(value)
        for value in labels["participant_id"]
    }

    if artifact_ids != label_ids:
        raise ValueError(
            "WavLM embeddings and filtered labels contain different "
            "participants."
        )

    if artifact_ids.intersection(
        excluded_ids
    ):
        raise ValueError(
            "A duplicate-connected participant remains in the WavLM "
            "artifact."
        )

    label_targets = {
        normalize_participant_id(participant_id): int(target)
        for participant_id, target in zip(
            labels["participant_id"],
            labels["target"],
            strict=True,
        )
    }

    participant_targets: dict[str, int] = {}

    for participant_id in sorted(
        artifact_ids
    ):
        participant_mask = (
            artifact.participant_ids
            == participant_id
        )
        participant_tasks = set(
            artifact.tasks[participant_mask]
        )
        unique_targets = np.unique(
            artifact.targets[participant_mask]
        )

        if int(participant_mask.sum()) != len(TASKS):
            raise ValueError(
                f"Participant {participant_id} does not have exactly "
                f"{len(TASKS)} embedding rows."
            )

        if participant_tasks != set(TASKS):
            raise ValueError(
                f"Participant {participant_id} does not have all tasks."
            )

        if len(unique_targets) != 1:
            raise ValueError(
                f"Participant {participant_id} has inconsistent targets."
            )

        participant_target = int(
            unique_targets[0]
        )

        if participant_target != label_targets[participant_id]:
            raise ValueError(
                f"Participant {participant_id} has conflicting saved "
                "and frozen targets."
            )

        participant_targets[participant_id] = (
            participant_target
        )

    target_counts = {
        target: list(
            participant_targets.values()
        ).count(target)
        for target in sorted(
            set(participant_targets.values())
        )
    }

    if target_counts != EXPECTED_TARGET_COUNTS:
        raise ValueError(
            "The retained target distribution changed: "
            f"{target_counts}."
        )

    expected_assignment_count = (
        EXPECTED_OUTER_FOLDS
        * EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
    )

    if len(nested_folds) != expected_assignment_count:
        raise ValueError(
            f"Expected {expected_assignment_count} nested-fold "
            f"assignments, found {len(nested_folds)}."
        )

    expected_record_values = {
        "experiment": (
            "duplicate_filtered_wavlm_embedding_extraction"
        ),
        "participant_level_fields": [],
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_sample_rate_hz": 8_000,
        "encoder_sample_rate_hz": 16_000,
        "hidden_size": 768,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "recordings": EXPECTED_RECORDINGS,
        "participants": EXPECTED_PARTICIPANTS,
        "tasks": list(TASKS),
        "target_counts": {"0": 52, "1": 48},
        "duplicate_policy": (
            "exclude_both_duplicate_connected_als_participants"
        ),
    }

    for field, expected_value in expected_record_values.items():
        if extraction_record.get(field) != expected_value:
            raise ValueError(
                f"Extraction record field {field!r} does not match "
                "the locked WavLM experiment."
            )

def build_embedding_row_lookup(
    artifact: WavLMEmbeddingArtifact,
) -> dict[tuple[str, str], int]:
    """Map each participant-task pair to its embedding row."""

    row_lookup = {
        (participant_id, task): row_index
        for row_index, (
            participant_id,
            task,
        ) in enumerate(
            zip(
                artifact.participant_ids,
                artifact.tasks,
                strict=True,
            )
        )
    }

    if len(row_lookup) != artifact.recording_count:
        raise ValueError(
            "Could not build a unique row lookup for every embedding."
        )

    return row_lookup

def build_target_lookup(
    labels: pd.DataFrame,
) -> dict[str, int]:
    """Map retained participant IDs to their frozen binary targets."""

    target_lookup = {
        normalize_participant_id(participant_id): int(target)
        for participant_id, target in zip(
            labels["participant_id"],
            labels["target"],
            strict=True,
        )
    }

    if len(target_lookup) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            "Could not build one target for every retained participant."
        )

    return target_lookup

def normalize_assignment_ids(
    assignment: dict[str, object],
    field: str,
) -> tuple[str, ...]:
    """Normalize one participant-ID list from a frozen fold assignment."""

    if field not in assignment:
        raise ValueError(
            f"Fold assignment is missing {field}."
        )

    values = assignment[field]

    if not isinstance(values, list):
        raise TypeError(
            f"Fold assignment field {field} must be a list."
        )

    normalized_ids = tuple(
        normalize_participant_id(value)
        for value in values
    )

    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(
            f"Fold assignment field {field} contains a duplicate ID."
        )

    return normalized_ids

def embedding_matrix_for_task(
    artifact: WavLMEmbeddingArtifact,
    row_lookup: dict[tuple[str, str], int],
    participant_ids: tuple[str, ...],
    task: str,
) -> np.ndarray:
    """Return one task's embeddings in the requested participant order."""

    missing_pairs = [
        (participant_id, task)
        for participant_id in participant_ids
        if (participant_id, task) not in row_lookup
    ]

    if missing_pairs:
        raise ValueError(
            "A frozen fold participant is missing a WavLM task "
            f"embedding: {missing_pairs[0]}."
        )

    row_indices = [
        row_lookup[(participant_id, task)]
        for participant_id in participant_ids
    ]

    return artifact.embeddings[
        row_indices
    ].copy()

def project_inner_fold_tasks(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    assignment: dict[str, object],
) -> ProjectedInnerFold:
    """Fit task-specific PCA only on one inner training partition."""

    outer_fold = int(
        assignment["outer_fold"]
    )
    inner_fold = int(
        assignment["inner_fold"]
    )

    training_ids = normalize_assignment_ids(
        assignment,
        "inner_train_ids",
    )
    validation_ids = normalize_assignment_ids(
        assignment,
        "inner_validation_ids",
    )
    outer_test_ids = normalize_assignment_ids(
        assignment,
        "outer_test_ids",
    )

    if set(training_ids).intersection(
        validation_ids
    ):
        raise ValueError(
            "Inner training and validation participants overlap."
        )

    if set(training_ids).intersection(
        outer_test_ids
    ) or set(validation_ids).intersection(
        outer_test_ids
    ):
        raise ValueError(
            "An outer-test participant entered inner preprocessing."
        )

    if len(training_ids) <= PCA_COMPONENTS_PER_TASK:
        raise ValueError(
            "The inner training partition is too small for the locked "
            "PCA dimension."
        )

    row_lookup = build_embedding_row_lookup(
        artifact
    )
    target_lookup = build_target_lookup(
        labels
    )

    training_targets = np.asarray(
        [
            target_lookup[participant_id]
            for participant_id in training_ids
        ],
        dtype=np.int64,
    )
    validation_targets = np.asarray(
        [
            target_lookup[participant_id]
            for participant_id in validation_ids
        ],
        dtype=np.int64,
    )

    training_features_by_task: dict[
        str,
        np.ndarray,
    ] = {}
    validation_features_by_task: dict[
        str,
        np.ndarray,
    ] = {}
    explained_variance_by_task: dict[
        str,
        float,
    ] = {}

    for task in TASKS:
        training_embeddings = embedding_matrix_for_task(
            artifact=artifact,
            row_lookup=row_lookup,
            participant_ids=training_ids,
            task=task,
        )
        validation_embeddings = embedding_matrix_for_task(
            artifact=artifact,
            row_lookup=row_lookup,
            participant_ids=validation_ids,
            task=task,
        )

        # Scaling and PCA are learned exclusively from inner training
        # participants. The validation partition is only transformed.
        scaler = StandardScaler()
        scaled_training = scaler.fit_transform(
            training_embeddings
        )
        scaled_validation = scaler.transform(
            validation_embeddings
        )

        reducer = PCA(
            n_components=PCA_COMPONENTS_PER_TASK,
            svd_solver="randomized",
            random_state=RANDOM_SEED,
        )
        projected_training = reducer.fit_transform(
            scaled_training
        )
        projected_validation = reducer.transform(
            scaled_validation
        )

        training_features_by_task[task] = np.asarray(
            projected_training,
            dtype=np.float32,
        )
        validation_features_by_task[task] = np.asarray(
            projected_validation,
            dtype=np.float32,
        )
        explained_variance_by_task[task] = float(
            reducer.explained_variance_ratio_.sum()
        )

    projected_fold = ProjectedInnerFold(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training_ids=training_ids,
        validation_ids=validation_ids,
        training_targets=training_targets,
        validation_targets=validation_targets,
        training_features_by_task=training_features_by_task,
        validation_features_by_task=validation_features_by_task,
        explained_variance_by_task=explained_variance_by_task,
    )

    validate_projected_inner_fold(
        projected_fold
    )

    return projected_fold

def validate_projected_inner_fold(
    projected_fold: ProjectedInnerFold,
) -> None:
    """Verify one projected split is finite, aligned, and usable."""

    training_count = len(
        projected_fold.training_ids
    )
    validation_count = len(
        projected_fold.validation_ids
    )

    if training_count != len(
        projected_fold.training_targets
    ):
        raise ValueError(
            "Projected training features and targets are misaligned."
        )

    if validation_count != len(
        projected_fold.validation_targets
    ):
        raise ValueError(
            "Projected validation features and targets are misaligned."
        )

    if set(projected_fold.training_targets) != {0, 1}:
        raise ValueError(
            "The projected training partition does not contain both "
            "target classes."
        )

    if set(projected_fold.validation_targets) != {0, 1}:
        raise ValueError(
            "The projected validation partition does not contain both "
            "target classes."
        )

    expected_training_shape = (
        training_count,
        PCA_COMPONENTS_PER_TASK,
    )
    expected_validation_shape = (
        validation_count,
        PCA_COMPONENTS_PER_TASK,
    )

    if set(
        projected_fold.training_features_by_task
    ) != set(TASKS):
        raise ValueError(
            "Projected training features do not contain all tasks."
        )

    if set(
        projected_fold.validation_features_by_task
    ) != set(TASKS):
        raise ValueError(
            "Projected validation features do not contain all tasks."
        )

    for task in TASKS:
        training_features = (
            projected_fold.training_features_by_task[
                task
            ]
        )
        validation_features = (
            projected_fold.validation_features_by_task[
                task
            ]
        )

        if training_features.shape != expected_training_shape:
            raise ValueError(
                f"Task {task} has unexpected projected training shape "
                f"{training_features.shape}."
            )

        if validation_features.shape != expected_validation_shape:
            raise ValueError(
                f"Task {task} has unexpected projected validation shape "
                f"{validation_features.shape}."
            )

        if not np.isfinite(
            training_features
        ).all() or not np.isfinite(
            validation_features
        ).all():
            raise ValueError(
                f"Task {task} contains a non-finite projected value."
            )

        explained_variance = (
            projected_fold.explained_variance_by_task[
                task
            ]
        )

        if not 0.0 < explained_variance <= 1.0:
            raise ValueError(
                f"Task {task} has invalid explained variance "
                f"{explained_variance}."
            )

def build_inner_projection_cache(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    nested_folds: list[dict[str, object]],
) -> dict[tuple[int, int], ProjectedInnerFold]:
    """Project every frozen inner fold once and reuse it across subsets."""

    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ] = {}

    ordered_assignments = sorted(
        nested_folds,
        key=lambda assignment: (
            int(assignment["outer_fold"]),
            int(assignment["inner_fold"]),
        ),
    )

    for assignment in ordered_assignments:
        outer_fold = int(
            assignment["outer_fold"]
        )
        inner_fold = int(
            assignment["inner_fold"]
        )
        cache_key = (
            outer_fold,
            inner_fold,
        )

        if cache_key in projection_cache:
            raise ValueError(
                "The nested-fold design contains a duplicate outer/inner "
                f"fold key: {cache_key}."
            )

        print(
            "Projecting WavLM features for outer fold "
            f"{outer_fold}, inner fold {inner_fold}..."
        )

        projection_cache[cache_key] = (
            project_inner_fold_tasks(
                artifact=artifact,
                labels=labels,
                assignment=assignment,
            )
        )

    validate_inner_projection_cache(
        projection_cache
    )

    return projection_cache

def validate_inner_projection_cache(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
) -> None:
    """Confirm the cache contains every expected inner fold exactly once."""

    expected_keys = {
        (outer_fold, inner_fold)
        for outer_fold in range(
            EXPECTED_OUTER_FOLDS
        )
        for inner_fold in range(
            EXPECTED_INNER_FOLDS_PER_OUTER_FOLD
        )
    }

    if set(projection_cache) != expected_keys:
        missing_keys = expected_keys.difference(
            projection_cache
        )
        unexpected_keys = set(
            projection_cache
        ).difference(expected_keys)

        raise ValueError(
            "The projection cache does not match the frozen nested "
            f"design. Missing: {sorted(missing_keys)}; unexpected: "
            f"{sorted(unexpected_keys)}."
        )

    for cache_key, projected_fold in (
        projection_cache.items()
    ):
        if cache_key != (
            projected_fold.outer_fold,
            projected_fold.inner_fold,
        ):
            raise ValueError(
                f"Projection cache key {cache_key} does not match its "
                "stored fold metadata."
            )

        validate_projected_inner_fold(
            projected_fold
        )

def print_projection_cache_summary(
    projection_cache: dict[
        tuple[int, int],
        ProjectedInnerFold,
    ],
) -> None:
    """Summarize retained variance across all leakage-safe projections."""

    print()
    print("WavLM inner-fold projection cache validated.")
    print(
        f"Projected inner folds: {len(projection_cache)}"
    )
    print(
        "PCA fits: "
        f"{len(projection_cache) * len(TASKS)}"
    )
    print(
        "PCA components per task: "
        f"{PCA_COMPONENTS_PER_TASK}"
    )
    print(
        "Explained variance across inner folds:"
    )
    print(
        "task  minimum      mean   maximum"
    )

    for task in TASKS:
        task_values = np.asarray(
            [
                projected_fold.explained_variance_by_task[
                    task
                ]
                for projected_fold in projection_cache.values()
            ],
            dtype=float,
        )

        print(
            f"{task:>4}  "
            f"{task_values.min():.6f}  "
            f"{task_values.mean():.6f}  "
            f"{task_values.max():.6f}"
        )

def concatenate_projected_tasks(
    projected_fold: ProjectedInnerFold,
    tasks: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate selected task projections in canonical task order."""

    if not tasks:
        raise ValueError(
            "At least one speech task is required."
        )

    if len(tasks) != len(set(tasks)):
        raise ValueError(
            "A speech task was requested more than once."
        )

    unknown_tasks = set(tasks).difference(
        TASKS
    )

    if unknown_tasks:
        raise ValueError(
            f"Unknown speech tasks requested: {sorted(unknown_tasks)}."
        )

    canonical_tasks = tuple(
        task
        for task in TASKS
        if task in tasks
    )

    if canonical_tasks != tasks:
        raise ValueError(
            "Speech tasks must follow the canonical dataset order."
        )

    training_features = np.concatenate(
        [
            projected_fold.training_features_by_task[
                task
            ]
            for task in tasks
        ],
        axis=1,
    )
    validation_features = np.concatenate(
        [
            projected_fold.validation_features_by_task[
                task
            ]
            for task in tasks
        ],
        axis=1,
    )

    expected_predictors = (
        len(tasks)
        * PCA_COMPONENTS_PER_TASK
    )

    if training_features.shape != (
        len(projected_fold.training_ids),
        expected_predictors,
    ):
        raise ValueError(
            "Concatenated training features have an unexpected shape."
        )

    if validation_features.shape != (
        len(projected_fold.validation_ids),
        expected_predictors,
    ):
        raise ValueError(
            "Concatenated validation features have an unexpected shape."
        )

    if not np.isfinite(
        training_features
    ).all() or not np.isfinite(
        validation_features
    ).all():
        raise ValueError(
            "Concatenated WavLM features contain a non-finite value."
        )

    return (
        training_features,
        validation_features,
    )

def build_wavlm_classifier(
    regularization_c: float,
) -> Pipeline:
    """Build the locked linear classifier for projected WavLM features."""

    if not np.isfinite(
        regularization_c
    ) or regularization_c <= 0.0:
        raise ValueError(
            "Logistic-regression C must be finite and positive."
        )

    # The second scaler gives every retained principal component a
    # comparable regularization scale after task projections are joined.
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(regularization_c),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=(
                        MAXIMUM_CLASSIFIER_ITERATIONS
                    ),
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

def project_outer_fold_tasks(
    artifact: WavLMEmbeddingArtifact,
    labels: pd.DataFrame,
    outer_fold: int,
) -> ProjectedOuterFold:
    """Fit each task's scaler and PCA on one complete outer train fold."""

    normalized_label_ids = labels[
        "participant_id"
    ].map(normalize_participant_id)
    test_mask = labels[
        "outer_fold"
    ].eq(outer_fold)

    training_ids = tuple(
        normalized_label_ids.loc[
            ~test_mask
        ]
    )
    test_ids = tuple(
        normalized_label_ids.loc[
            test_mask
        ]
    )

    if not test_ids:
        raise ValueError(
            f"Outer fold {outer_fold} has no test participants."
        )

    if set(training_ids).intersection(
        test_ids
    ):
        raise ValueError(
            "Outer training and test participants overlap."
        )

    artifact_ids = set(
        artifact.participant_ids
    )

    if set(training_ids).union(
        test_ids
    ) != artifact_ids:
        raise ValueError(
            "The outer train/test partition does not cover the WavLM "
            "cohort."
        )

    if len(training_ids) <= PCA_COMPONENTS_PER_TASK:
        raise ValueError(
            "The outer training partition is too small for the locked "
            "PCA dimension."
        )

    row_lookup = build_embedding_row_lookup(
        artifact
    )
    target_lookup = build_target_lookup(
        labels
    )
    training_targets = np.asarray(
        [
            target_lookup[participant_id]
            for participant_id in training_ids
        ],
        dtype=np.int64,
    )
    test_targets = np.asarray(
        [
            target_lookup[participant_id]
            for participant_id in test_ids
        ],
        dtype=np.int64,
    )

    training_features_by_task: dict[
        str,
        np.ndarray,
    ] = {}
    test_features_by_task: dict[
        str,
        np.ndarray,
    ] = {}
    explained_variance_by_task: dict[
        str,
        float,
    ] = {}

    for task in TASKS:
        training_embeddings = embedding_matrix_for_task(
            artifact=artifact,
            row_lookup=row_lookup,
            participant_ids=training_ids,
            task=task,
        )
        test_embeddings = embedding_matrix_for_task(
            artifact=artifact,
            row_lookup=row_lookup,
            participant_ids=test_ids,
            task=task,
        )

        # Outer-test embeddings are transformed only after preprocessing
        # has been fitted on every available outer-training participant.
        scaler = StandardScaler()
        scaled_training = scaler.fit_transform(
            training_embeddings
        )
        scaled_test = scaler.transform(
            test_embeddings
        )

        reducer = PCA(
            n_components=PCA_COMPONENTS_PER_TASK,
            svd_solver="randomized",
            random_state=RANDOM_SEED,
        )
        projected_training = reducer.fit_transform(
            scaled_training
        )
        projected_test = reducer.transform(
            scaled_test
        )

        training_features_by_task[task] = np.asarray(
            projected_training,
            dtype=np.float32,
        )
        test_features_by_task[task] = np.asarray(
            projected_test,
            dtype=np.float32,
        )
        explained_variance_by_task[task] = float(
            reducer.explained_variance_ratio_.sum()
        )

    projected_fold = ProjectedOuterFold(
        outer_fold=outer_fold,
        training_ids=training_ids,
        test_ids=test_ids,
        training_targets=training_targets,
        test_targets=test_targets,
        training_features_by_task=training_features_by_task,
        test_features_by_task=test_features_by_task,
        explained_variance_by_task=explained_variance_by_task,
    )

    validate_projected_outer_fold(
        projected_fold
    )

    return projected_fold

def validate_projected_outer_fold(
    projected_fold: ProjectedOuterFold,
) -> None:
    """Verify one untouched outer-test projection is aligned and finite."""

    training_count = len(
        projected_fold.training_ids
    )
    test_count = len(
        projected_fold.test_ids
    )

    if training_count + test_count != EXPECTED_PARTICIPANTS:
        raise ValueError(
            "The projected outer fold does not contain every retained "
            "participant."
        )

    if training_count != len(
        projected_fold.training_targets
    ) or test_count != len(
        projected_fold.test_targets
    ):
        raise ValueError(
            "Projected outer-fold features and targets are misaligned."
        )

    if set(projected_fold.training_targets) != {0, 1}:
        raise ValueError(
            "The outer-training partition does not contain both classes."
        )

    if set(projected_fold.test_targets) != {0, 1}:
        raise ValueError(
            "The outer-test partition does not contain both classes."
        )

    expected_training_shape = (
        training_count,
        PCA_COMPONENTS_PER_TASK,
    )
    expected_test_shape = (
        test_count,
        PCA_COMPONENTS_PER_TASK,
    )

    if set(
        projected_fold.training_features_by_task
    ) != set(TASKS) or set(
        projected_fold.test_features_by_task
    ) != set(TASKS):
        raise ValueError(
            "The projected outer fold does not contain every task."
        )

    for task in TASKS:
        training_features = (
            projected_fold.training_features_by_task[
                task
            ]
        )
        test_features = (
            projected_fold.test_features_by_task[
                task
            ]
        )

        if training_features.shape != expected_training_shape:
            raise ValueError(
                f"Task {task} has an unexpected outer-training shape."
            )

        if test_features.shape != expected_test_shape:
            raise ValueError(
                f"Task {task} has an unexpected outer-test shape."
            )

        if not np.isfinite(
            training_features
        ).all() or not np.isfinite(
            test_features
        ).all():
            raise ValueError(
                f"Task {task} has a non-finite outer projection."
            )

def concatenate_outer_tasks(
    projected_fold: ProjectedOuterFold,
    tasks: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate outer train/test task projections in canonical order."""

    if not tasks or len(tasks) != len(
        set(tasks)
    ):
        raise ValueError(
            "Outer evaluation requires unique selected tasks."
        )

    canonical_tasks = tuple(
        task
        for task in TASKS
        if task in tasks
    )

    if canonical_tasks != tasks:
        raise ValueError(
            "Outer-evaluation tasks are unknown or out of order."
        )

    training_features = np.concatenate(
        [
            projected_fold.training_features_by_task[
                task
            ]
            for task in tasks
        ],
        axis=1,
    )
    test_features = np.concatenate(
        [
            projected_fold.test_features_by_task[
                task
            ]
            for task in tasks
        ],
        axis=1,
    )

    expected_predictors = (
        len(tasks)
        * PCA_COMPONENTS_PER_TASK
    )

    if training_features.shape != (
        len(projected_fold.training_ids),
        expected_predictors,
    ) or test_features.shape != (
        len(projected_fold.test_ids),
        expected_predictors,
    ):
        raise ValueError(
            "Concatenated outer features have an unexpected shape."
        )

    return training_features, test_features

def calculate_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate the study's locked classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    if tp + fn == 0 or tn + fp == 0:
        raise ValueError(
            "An outer fold is missing a class required for sensitivity "
            "or specificity."
        )

    return {
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                true_labels,
                scores,
            )
        ),
        "sensitivity": float(
            tp / (tp + fn)
        ),
        "specificity": float(
            tn / (tn + fp)
        ),
        "positive_class_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
    }

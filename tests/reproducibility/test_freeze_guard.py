"""Freeze-, Held-out-Guard- und Zugriffsberechtigungstests (DEC-027, DEC-033)."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from provenance_repo import HELD_OUT, ProvenanceRepo, approve_and_verify
from vigiscoutx.artifacts.manifests import ArtifactEntry, ArtifactManifest
from vigiscoutx.domain.refs import DatasetSnapshotRef, SeedBundle
from vigiscoutx.errors import IntegrityError, PhaseViolationError
from vigiscoutx.experiment.code_provenance import EvaluationProvenance
from vigiscoutx.experiment.freeze import (
    FreezeManifest,
    build_freeze_manifest,
    verify_freeze_manifest,
)
from vigiscoutx.experiment.held_out_guard import (
    HeldOutEvaluationSession,
    HeldOutGuard,
    ObservedHashes,
    PartitionAccessPermit,
)
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition, Phase
from vigiscoutx.ports import AccessPurpose, DataAccessContext, DatasetPort

GIT_COMMIT = "a" * 40
ENV_LOCK = "e" * 64

SPLIT = SplitManifest(
    manifest_id="split-test",
    assignments=(
        SplitAssignment(
            sample_id="md1",
            group_id="l1",
            partition=Partition.MODEL_DEVELOPMENT,
            model_development_role=ModelDevelopmentRole.TRAIN,
        ),
        SplitAssignment(
            sample_id="cal1",
            group_id="l2",
            partition=Partition.CALIBRATION,
            model_development_role=None,
        ),
        SplitAssignment(
            sample_id="ho1",
            group_id="l3",
            partition=Partition.HELD_OUT,
            model_development_role=None,
        ),
    ),
)
SPLIT_SHA = split_manifest_sha256(SPLIT)


class InMemoryDataset:
    """Test-Adapter: lädt nur mit gültiger Zugriffsberechtigung."""

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return DatasetSnapshotRef(snapshot_id="ds", uri="memory://ds", sha256="1" * 64)

    def load_inputs(self, access: DataAccessContext, sample_ids: Sequence[str]) -> Sequence[str]:
        access.require_access(sample_ids, AccessPurpose.INPUTS)
        return [f"input:{s}" for s in sample_ids]


def frozen_guard(manifest: FreezeManifest) -> HeldOutGuard:
    guard = HeldOutGuard()
    guard.freeze(manifest)
    return guard


def _with_split_sha(data: dict[str, Any], sha: str) -> dict[str, Any]:
    scientific = {**data["scientific"], "split": {**data["scientific"]["split"], "sha256": sha}}
    return {**data, "scientific": scientific}


@pytest.fixture
def split_bound_session(
    provenance_repo: ProvenanceRepo,
    calibrated_config_data: dict[str, Any],
    seed_list: tuple[SeedBundle, ...],
    tmp_path: Path,
) -> tuple[HeldOutGuard, FreezeManifest]:
    config = RunConfig.model_validate(_with_split_sha(calibrated_config_data, SPLIT_SHA))
    manifest = build_freeze_manifest(
        config,
        seed_list=seed_list,
        git_commit=provenance_repo.calibration_commit,
        environment_lock_sha256=ENV_LOCK,
    )
    evaluation_provenance = approve_and_verify(
        provenance_repo, manifest.freeze_id, tmp_path / "split-approval", purpose=HELD_OUT
    )
    assert isinstance(evaluation_provenance, EvaluationProvenance)
    guard = frozen_guard(manifest)
    held_out = RunConfig.model_validate(
        {
            **_with_split_sha(calibrated_config_data, SPLIT_SHA),
            "run_id": "run-eval",
            "phase": "HELD_OUT_EVALUATION",
            "freeze_id": manifest.freeze_id,
        }
    )
    observed = ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=SPLIT_SHA,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=ENV_LOCK,
    )
    guard.open_held_out(held_out, observed, evaluation_provenance)
    return guard, manifest


# --- Freeze -----------------------------------------------------------------------------------


def test_freeze_manifest_is_deterministic(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...], freeze_manifest: FreezeManifest
) -> None:
    again = build_freeze_manifest(
        calibrated_config,
        seed_list=seed_list,
        git_commit=GIT_COMMIT,
        environment_lock_sha256=ENV_LOCK,
    )
    assert again == freeze_manifest
    verify_freeze_manifest(freeze_manifest)


def test_freeze_id_is_stable_across_frozen_mapping_implementation(
    freeze_manifest: FreezeManifest,
) -> None:
    """Referenzwert aus dem Stand vor der tupelbasierten FrozenMapping-Speicherung (9427ce0)."""
    assert freeze_manifest.freeze_id == (
        "47a63a5e5ebb5ab227d917c40253b7ad630122c2176533b86a4287c616076fa1"
    )


def test_freeze_manifest_signal_params_are_immutable(freeze_manifest: FreezeManifest) -> None:
    params = freeze_manifest.signals[1].params
    with pytest.raises(AttributeError):
        params._data["k"] = 999  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        params["k"] = 999  # type: ignore[index]
    verify_freeze_manifest(freeze_manifest)


def test_freeze_requires_all_calibration_values(
    calibration_config_data: dict[str, Any], seed_list: tuple[SeedBundle, ...]
) -> None:
    open_config = RunConfig.model_validate(calibration_config_data)
    with pytest.raises(IntegrityError, match="offene Calibration-Parameter"):
        build_freeze_manifest(
            open_config,
            seed_list=seed_list,
            git_commit=GIT_COMMIT,
            environment_lock_sha256=ENV_LOCK,
        )


def test_tampered_freeze_manifest_fails(freeze_manifest: FreezeManifest) -> None:
    tampered = freeze_manifest.model_copy(update={"window_size": freeze_manifest.window_size + 1})
    with pytest.raises(IntegrityError):
        verify_freeze_manifest(tampered)
    with pytest.raises(IntegrityError):
        HeldOutGuard().freeze(tampered)


def test_seed_list_must_match_repetitions(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...]
) -> None:
    with pytest.raises(ValidationError, match="repetitions"):
        build_freeze_manifest(
            calibrated_config,
            seed_list=seed_list[:1],
            git_commit=GIT_COMMIT,
            environment_lock_sha256=ENV_LOCK,
        )


@pytest.mark.parametrize("field", ["split_seed", "model_development_seed"])
def test_split_and_model_seeds_constant_across_repetitions(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...], field: str
) -> None:
    varied = (seed_list[0], seed_list[1].model_copy(update={field: 999}))
    with pytest.raises(ValidationError, match=field):
        build_freeze_manifest(
            calibrated_config,
            seed_list=varied,
            git_commit=GIT_COMMIT,
            environment_lock_sha256=ENV_LOCK,
        )


def test_repeat_ids_must_be_unique(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...]
) -> None:
    duplicate = (seed_list[0], seed_list[1].model_copy(update={"repeat_id": 0}))
    with pytest.raises(ValidationError, match="repeat_id"):
        build_freeze_manifest(
            calibrated_config,
            seed_list=duplicate,
            git_commit=GIT_COMMIT,
            environment_lock_sha256=ENV_LOCK,
        )


# --- Held-out Guard ---------------------------------------------------------------------------


def test_held_out_without_freeze_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    with pytest.raises(PhaseViolationError, match="gültigen Freeze"):
        HeldOutGuard().open_held_out(evaluation_held_out_config, observed, evaluation_provenance)


def test_valid_freeze_opens_held_out(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    guard = frozen_guard(evaluation_manifest)
    session = guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    assert guard.phase is Phase.HELD_OUT_EVALUATION
    assert session.freeze_id == evaluation_manifest.freeze_id
    guard.assert_partition_access(Partition.HELD_OUT)
    with pytest.raises(PhaseViolationError):
        guard.assert_partition_access(Partition.CALIBRATION)


def test_held_out_cannot_be_opened_twice(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    guard = frozen_guard(evaluation_manifest)
    guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    with pytest.raises(PhaseViolationError):
        guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)


def test_development_config_cannot_open_held_out(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    calibrated_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    with pytest.raises(PhaseViolationError, match="HELD_OUT_EVALUATION"):
        frozen_guard(evaluation_manifest).open_held_out(
            calibrated_config, observed, evaluation_provenance
        )


@pytest.mark.parametrize(
    "field",
    [
        "dataset_snapshot_sha256",
        "split_manifest_sha256",
        "model_weights_sha256",
        "environment_lock_sha256",
    ],
)
def test_observed_hash_mismatch_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    field: str,
) -> None:
    wrong = "b" * 64
    guard = frozen_guard(evaluation_manifest)
    with pytest.raises(IntegrityError, match="Hash-Abweichung"):
        guard.open_held_out(
            evaluation_held_out_config,
            observed.model_copy(update={field: wrong}),
            evaluation_provenance,
        )
    assert guard.phase is Phase.FROZEN


def test_config_changed_after_freeze_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    changed = RunConfig.model_validate(
        {
            **evaluation_held_out_config.model_dump(mode="json"),
            "repetitions": evaluation_held_out_config.repetitions + 1,
        }
    )
    with pytest.raises(IntegrityError, match=r"config\.repetitions"):
        frozen_guard(evaluation_manifest).open_held_out(changed, observed, evaluation_provenance)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("monitoring", "signals", 0, "threshold", "value"), 0.3, "config_sha256"),
        (("monitoring", "signals", 1, "params", "k"), 2, "config_sha256"),
        (("preprocessing", "input_height"), 16, "preprocessing_sha256"),
        (("relevance", "delta_rel"), 0.25, "config_sha256"),
        (("fusion", "rule"), "A_ANY_PRIMARY_GROUP", "config_sha256"),
    ],
)
def test_scientific_change_after_freeze_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    path: tuple[str | int, ...],
    value: object,
    expected: str,
) -> None:
    data = evaluation_held_out_config.model_dump(mode="json")
    parent: Any = data["scientific"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(IntegrityError, match=expected):
        frozen_guard(evaluation_manifest).open_held_out(
            RunConfig.model_validate(data), observed, evaluation_provenance
        )


def test_wrong_freeze_id_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    other = RunConfig.model_validate(
        {**evaluation_held_out_config.model_dump(mode="json"), "freeze_id": "c" * 64}
    )
    with pytest.raises(IntegrityError, match="freeze_id"):
        frozen_guard(evaluation_manifest).open_held_out(other, observed, evaluation_provenance)


def test_seed_outside_frozen_seed_list_fails(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    data = evaluation_held_out_config.model_dump(mode="json")
    data["seeds"] = {**data["seeds"], "repeat_id": 99}
    with pytest.raises(IntegrityError, match="Seed-Liste"):
        frozen_guard(evaluation_manifest).open_held_out(
            RunConfig.model_validate(data), observed, evaluation_provenance
        )


def test_session_config_cannot_be_mutated_after_open(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    session = frozen_guard(evaluation_manifest).open_held_out(
        evaluation_held_out_config, observed, evaluation_provenance
    )
    signal = session.config.scientific.monitoring.signals[1]
    with pytest.raises(TypeError):
        signal.params["k"] = 999  # type: ignore[index]
    with pytest.raises(ValidationError):
        signal.method = "other"  # type: ignore[misc]


def test_calibration_cannot_open_held_out_partition() -> None:
    guard = HeldOutGuard()
    guard.require_calibration_phase()
    guard.assert_partition_access(Partition.CALIBRATION)
    with pytest.raises(PhaseViolationError, match="HELD_OUT"):
        guard.assert_partition_access(Partition.HELD_OUT)


def test_frozen_guard_allows_no_data_access(freeze_manifest: FreezeManifest) -> None:
    guard = frozen_guard(freeze_manifest)
    for partition in Partition:
        with pytest.raises(PhaseViolationError, match="FROZEN"):
            guard.assert_partition_access(partition)


def test_no_calibration_or_update_after_freeze(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    guard = frozen_guard(evaluation_manifest)
    with pytest.raises(PhaseViolationError):
        guard.require_calibration_phase()
    with pytest.raises(PhaseViolationError):
        guard.freeze(evaluation_manifest)
    guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    with pytest.raises(PhaseViolationError):
        guard.require_calibration_phase()


def test_evaluation_session_offers_no_optimisation_operation() -> None:
    public = {name for name in dir(HeldOutEvaluationSession) if not name.startswith("_")}
    assert public == {"manifest", "config", "freeze_id"}
    assert dataclasses.is_dataclass(HeldOutEvaluationSession)
    assert HeldOutEvaluationSession.__dataclass_params__.frozen  # type: ignore[attr-defined]


# --- Zugriffsberechtigung (Permit) ------------------------------------------------------------


def test_calibration_cannot_obtain_held_out_permit() -> None:
    with pytest.raises(PhaseViolationError, match="HELD_OUT"):
        HeldOutGuard().issue_permit(Partition.HELD_OUT, SPLIT)


def test_calibration_permit_loads_only_its_partition() -> None:
    permit = HeldOutGuard().issue_permit(Partition.CALIBRATION, SPLIT)
    dataset = InMemoryDataset()
    assert dataset.load_inputs(permit, ["cal1"]) == ["input:cal1"]
    with pytest.raises(PhaseViolationError, match="außerhalb"):
        dataset.load_inputs(permit, ["ho1"])


def test_frozen_guard_issues_no_permit(freeze_manifest: FreezeManifest) -> None:
    guard = frozen_guard(freeze_manifest)
    with pytest.raises(PhaseViolationError):
        guard.issue_permit(Partition.HELD_OUT, SPLIT)


def test_valid_held_out_session_issues_held_out_permit(
    split_bound_session: tuple[HeldOutGuard, FreezeManifest],
) -> None:
    guard, manifest = split_bound_session
    permit = guard.issue_permit(Partition.HELD_OUT, SPLIT)
    assert permit.freeze_id == manifest.freeze_id
    assert InMemoryDataset().load_inputs(permit, ["ho1"]) == ["input:ho1"]
    with pytest.raises(PhaseViolationError):
        guard.issue_permit(Partition.CALIBRATION, SPLIT)


def test_held_out_permit_requires_frozen_split(
    split_bound_session: tuple[HeldOutGuard, FreezeManifest],
) -> None:
    guard, _ = split_bound_session
    other_split = SplitManifest(manifest_id="other", assignments=SPLIT.assignments)
    with pytest.raises(IntegrityError, match="Split-Manifest"):
        guard.issue_permit(Partition.HELD_OUT, other_split)


def test_calibration_permit_expires_after_phase_change(
    freeze_manifest: FreezeManifest,
) -> None:
    guard = HeldOutGuard()
    permit = guard.issue_permit(Partition.CALIBRATION, SPLIT)
    guard.freeze(freeze_manifest)
    with pytest.raises(PhaseViolationError, match="Phase"):
        permit.require_access(["cal1"], AccessPurpose.INPUTS)


def test_permit_cannot_be_forged() -> None:
    guard = HeldOutGuard()
    with pytest.raises(PhaseViolationError, match="nur vom HeldOutGuard"):
        PartitionAccessPermit(
            partition=Partition.HELD_OUT,
            model_development_role=None,
            phase=guard.phase,
            freeze_id=None,
            sample_ids=frozenset({"ho1"}),
            guard=guard,
            issue_token=object(),
        )


def test_dataset_port_contract_requires_permit() -> None:
    dataset: DatasetPort[str] = InMemoryDataset()
    with pytest.raises(TypeError):
        dataset.load_inputs(["cal1"])  # type: ignore[arg-type, call-arg]


# --- Artefakte --------------------------------------------------------------------------------


def test_artifact_manifest_never_overwrites_entries() -> None:
    entry = ArtifactEntry(path="results/signals.csv", sha256="d" * 64, role="signal-results")
    manifest = ArtifactManifest(run_id="run", entries=()).with_entry(entry)
    with pytest.raises(IntegrityError, match="kein Überschreiben"):
        manifest.with_entry(entry.model_copy(update={"sha256": "f" * 64}))


@pytest.mark.parametrize("path", ["/abs/file.json", "../escape.json", "a/../b.json", "a\\b.json"])
def test_artifact_paths_must_be_relative(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactEntry(path=path, sha256="d" * 64, role="x")


@pytest.mark.parametrize("alias", ["a/./b.csv", "./a/b.csv", "a//b.csv", "a/b.csv/"])
def test_artifact_paths_are_canonical(alias: str) -> None:
    assert ArtifactEntry(path=alias, sha256="d" * 64, role="x").path == "a/b.csv"
    manifest = ArtifactManifest(
        run_id="run", entries=(ArtifactEntry(path="a/b.csv", sha256="d" * 64, role="x"),)
    )
    with pytest.raises(IntegrityError):
        manifest.with_entry(ArtifactEntry(path=alias, sha256="e" * 64, role="x"))
    with pytest.raises(ValidationError):
        ArtifactManifest(
            run_id="run",
            entries=(
                ArtifactEntry(path="a/b.csv", sha256="d" * 64, role="x"),
                ArtifactEntry(path=alias, sha256="e" * 64, role="x"),
            ),
        )


@pytest.mark.parametrize("path", [".", "./"])
def test_empty_artifact_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactEntry(path=path, sha256="d" * 64, role="x")

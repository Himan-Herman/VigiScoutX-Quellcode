"""P4-03AE: Reference-Rekonstruktion vor HELD_OUT (DEC-084/085), ausschließlich synthetisch.

Ein einziger synthetischer Calibration-Lauf (Modul-Fixture) bis `FROZEN` mit dem echten
`Ham10000Dataset` auf synthetischen Bildern und dem deterministischen Texturmodell; sein
Calibration-Commit liegt in einem synthetischen Git-Repository, das als ausgeführtes Repository
gilt. Die Rekonstruktion wird gegen die im Calibration-Lauf berechneten Reference-Artefakte
geprüft. Keine realen Daten, kein ResNet, keine reale Freigabe, kein HELD_OUT.
"""

import dataclasses
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from calibration_e2e_world import (
    LAYOUT,
    PROTOCOL,
    RecordingDataset,
    TextureModel,
    World,
    preprocessing_spec,
    provenance_for,
    synthetic_execution,
    template_data,
    write_world,
)
from calibration_runner.orchestrator import CalibrationRunner, CalibrationRunResult
from provenance_repo import (
    GUARD_PATH,
    REFERENCE,
    ProvenanceRepo,
    approve_and_verify,
    create_provenance_repo,
)
from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.adapters.preprocessing import ImagePreprocessor, PreprocessedImage
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.domain.hashing import canonical_json_bytes, sha256_hex
from vigiscoutx.domain.refs import DatasetSnapshotRef, ModelSnapshotRef
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.errors import IntegrityError, PhaseViolationError
from vigiscoutx.evaluation import reference_reconstruction
from vigiscoutx.evaluation.reference_reconstruction import (
    EVIDENCE_MANIFEST_FILE,
    P4_03_REFERENCE_EXPECTATION,
    ReconstructedReference,
    ReferenceExecution,
    ReferenceExpectation,
    load_reconstructed_reference,
    load_reference_model,
    reconstruct_reference,
    resolve_expectation,
)
from vigiscoutx.evaluation.reference_sources import (
    FrozenCalibration,
    ReferenceReconstructionStop,
    derive_reference,
    load_frozen_calibration,
)
from vigiscoutx.experiment import code_provenance
from vigiscoutx.experiment.code_provenance import EvaluationProvenance
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split_integrity import partition_sample_ids
from vigiscoutx.experiment.types import Partition, Phase
from vigiscoutx.ports import AccessPurpose, DataAccessContext


@dataclasses.dataclass(frozen=True)
class Calibrated:
    repo: ProvenanceRepo
    world: World
    dataset: Ham10000Dataset
    preprocessor: ImagePreprocessor
    model: TextureModel
    runner: CalibrationRunner
    result: CalibrationRunResult

    @property
    def lock(self) -> Path:
        return self.repo.root / "requirements-lock.txt"


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> Calibrated:
    repo = create_provenance_repo(tmp_path_factory.mktemp("repo") / "provenance-repo")
    world = write_world(tmp_path_factory.mktemp("world"))
    dataset = Ham10000Dataset.from_directory(world.root, layout=LAYOUT)
    preprocessor = ImagePreprocessor(preprocessing_spec())
    model = TextureModel(preprocessor)
    template = RunConfig.model_validate(template_data(world, dataset, model))
    runner = CalibrationRunner(
        guard=HeldOutGuard(),
        split_manifest=world.split,
        provenance=provenance_for(
            template,
            dataset,
            model,
            git_commit=repo.calibration_commit,
            environment_lock_sha256=sha256_file(repo.root / "requirements-lock.txt"),
        ),
        protocol=PROTOCOL,
        dataset=dataset,
        audit_dataset=dataset,
        execution=synthetic_execution(dataset, model),
        model=model,
        preprocessor=preprocessor,
        template=template,
        output_dir=tmp_path_factory.mktemp("run") / "calibration",
    )
    result = runner.run()
    return Calibrated(repo, world, dataset, preprocessor, model, runner, result)


@dataclasses.dataclass
class Context:
    calibrated: Calibrated
    guard: HeldOutGuard
    provenance: EvaluationProvenance
    frozen: FrozenCalibration
    recorder: RecordingDataset
    expectation: ReferenceExpectation
    output: Path

    def run(self, **overrides: Any) -> ReconstructedReference:
        arguments: dict[str, Any] = {
            "guard": self.guard,
            "frozen": self.frozen,
            "split_manifest": self.calibrated.world.split,
            "dataset": self.recorder,
            "model": self.calibrated.model,
            "preprocessor": self.calibrated.preprocessor,
            "environment_lock": self.calibrated.lock,
            "provenance": self.provenance,
            "expectation": self.expectation,
            "execution": ReferenceExecution(device="cpu", batch_size=32),
            "output_dir": self.output,
            **overrides,
        }
        return reconstruct_reference(**arguments)

    def accesses(self) -> list[tuple[str, AccessPurpose]]:
        return [(s, p) for s, p, _ in self.recorder.accesses]

    def no_result(self) -> bool:
        staging = self.output.with_name(f".{self.output.name}.incomplete")
        return not self.output.exists() and not staging.exists()


@pytest.fixture
def ctx(calibrated: Calibrated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Context:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", calibrated.repo.root / GUARD_PATH)
    manifest = calibrated.result.manifest
    guard = HeldOutGuard()
    guard.freeze(manifest)
    provenance = approve_and_verify(
        calibrated.repo, manifest.freeze_id, tmp_path / "approval", purpose=REFERENCE
    )
    assert isinstance(provenance, EvaluationProvenance)
    vector = calibrated.runner.process.results.class_vector
    ba_ref = calibrated.runner.process.results.ba_ref
    assert vector is not None and ba_ref is not None
    return Context(
        calibrated=calibrated,
        guard=guard,
        provenance=provenance,
        frozen=load_frozen_calibration(calibrated.result.output_dir),
        recorder=RecordingDataset(calibrated.dataset),
        expectation=ReferenceExpectation(
            freeze_id=manifest.freeze_id,
            class_counts=tuple(vector.as_dict().items()),
            ba_ref_4dp=f"{ba_ref:.4f}",
            ba_ref_source="synthetischer Calibration-Lauf (Testfixture)",
        ),
        output=tmp_path / "reference",
    )


# --- Erfolgreicher synthetischer Lauf -----------------------------------------------------------


def test_reconstruction_reproduces_the_calibration_reference_artifacts(ctx: Context) -> None:
    reference = ctx.run()
    calibration = ctx.calibrated.runner._reference  # im Calibration-Lauf berechnete Artefakte
    assert calibration is not None
    assert reference.representation == calibration.representation
    assert reference.quality == calibration.quality
    assert reference.outputs == calibration.outputs
    assert reference.performance == calibration.performance
    assert reference.ba_ref == ctx.calibrated.runner.process.results.ba_ref
    record = ctx.frozen.record
    assert reference.derived.sample_ids == record.reference_sample_ids
    assert reference.derived.lesion_ids == record.reference_lesion_ids
    assert reference.window_id == record.reference_window_id
    assert reference.derived.context == "P4-01|OFFICIAL_REFERENCE|128"


def test_data_access_is_once_per_sample_and_purpose_via_the_permit_only(ctx: Context) -> None:
    ctx.run()
    reference = set(ctx.frozen.record.reference_sample_ids)
    accesses = ctx.accesses()
    for purpose in (AccessPurpose.INPUTS, AccessPurpose.LABELS):
        samples = [s for s, p in accesses if p is purpose]
        assert sorted(samples) == sorted(reference)  # jedes Sample genau einmal
    assert {p for _, p in accesses} == {AccessPurpose.INPUTS, AccessPurpose.LABELS}
    assert len(accesses) == 2 * len(reference)
    assert ctx.guard.phase is Phase.FROZEN
    with pytest.raises(PhaseViolationError, match="bereits einmal berechtigt"):
        ctx.run(output_dir=ctx.output.with_name("second"))


def test_execution_and_binding_are_documented(ctx: Context) -> None:
    reference = ctx.run()
    execution = json.loads((ctx.output / "provenance/execution.json").read_bytes())
    record = ctx.frozen.record
    assert execution["requested_device"] == execution["observed_default_device"] == "cpu"
    assert execution["input_tensor_devices"] == ["cpu"]
    assert execution["batch_size"] == 32 and execution["batches"] == [32, 32, 32, 32]
    assert execution["model_snapshot_sha256"] == record.model_snapshot_sha256
    assert execution["model_weights_sha256"] == record.model_weights_sha256
    assert execution["preprocessing_sha256"] == record.preprocessing_sha256
    assert execution["environment_lock_sha256"] == record.environment_lock_sha256
    assert execution["ba_ref"] == reference.ba_ref
    assert execution["ba_ref_4dp"] == ctx.expectation.ba_ref_4dp
    evidence = reference.evidence
    assert evidence.freeze_id == ctx.frozen.manifest.freeze_id
    assert evidence.calibration_commit == ctx.calibrated.repo.calibration_commit
    assert evidence.evaluation_commit == ctx.calibrated.repo.git("rev-parse", "HEAD")
    assert evidence.approval_sha256 == ctx.provenance.approval_sha256


# --- Write-once und Wiederverwendbarkeit --------------------------------------------------------


def test_artifacts_are_write_once_immutable_and_reusable(ctx: Context) -> None:
    reference = ctx.run()
    loaded = load_reconstructed_reference(
        ctx.output,
        freeze_id=reference.evidence.freeze_id,
        evaluation_commit=reference.evidence.evaluation_commit,
        expected_manifest_sha256=_manifest_sha256(reference),
    )
    assert loaded == reference
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.ba_ref = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        loaded.representation.embeddings = ()  # type: ignore[misc]
    assert all(not (p.stat().st_mode & 0o222) for p in ctx.output.rglob("*") if p.is_file())
    for freeze_id, commit in (
        ("0" * 64, reference.evidence.evaluation_commit),
        (
            reference.evidence.freeze_id,
            "0" * 40,
        ),
    ):
        with pytest.raises(ReferenceReconstructionStop, match="anderen Freeze"):
            load_reconstructed_reference(
                ctx.output,
                freeze_id=freeze_id,
                evaluation_commit=commit,
                expected_manifest_sha256=_manifest_sha256(reference),
            )


@pytest.mark.parametrize("tamper", ["modified", "removed", "extra", "manifest"])
def test_tampered_reference_artifacts_are_rejected_on_reuse(ctx: Context, tamper: str) -> None:
    reference = ctx.run()
    target = ctx.output / "reference/g1_representation.json"
    if tamper == "modified":
        target.chmod(0o644)
        target.write_bytes(target.read_bytes().replace(b"1", b"2", 1))
    elif tamper == "removed":
        target.chmod(0o644)
        target.unlink()
    elif tamper == "extra":
        (ctx.output / "reference/extra.json").write_text("{}", "utf-8")
    else:
        manifest = ctx.output / EVIDENCE_MANIFEST_FILE
        manifest.chmod(0o644)
        manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ReferenceReconstructionStop):
        load_reconstructed_reference(
            ctx.output,
            freeze_id=reference.evidence.freeze_id,
            evaluation_commit=reference.evidence.evaluation_commit,
            expected_manifest_sha256=_manifest_sha256(reference),
        )


def test_manifest_bytes_are_hash_checked_before_parsing(ctx: Context) -> None:
    """P4-03AO: genau ein Lesevorgang; Hashvergleich des Byte-Puffers vor jedem Parsing."""
    reference = ctx.run()
    expected = _manifest_sha256(reference)
    with pytest.raises(ReferenceReconstructionStop, match="freigegebenen Hash"):
        load_reconstructed_reference(
            ctx.output,
            freeze_id=reference.evidence.freeze_id,
            evaluation_commit=reference.evidence.evaluation_commit,
            expected_manifest_sha256="0" * 64,
        )
    manifest = ctx.output / EVIDENCE_MANIFEST_FILE
    manifest.chmod(0o644)
    manifest.write_bytes(b"not json at all")  # ohne Hash-Übereinstimmung nie geparst
    with pytest.raises(ReferenceReconstructionStop, match="freigegebenen Hash"):
        load_reconstructed_reference(
            ctx.output,
            freeze_id=reference.evidence.freeze_id,
            evaluation_commit=reference.evidence.evaluation_commit,
            expected_manifest_sha256=expected,
        )


def test_existing_output_is_never_overwritten(ctx: Context) -> None:
    ctx.output.mkdir()
    with pytest.raises(ReferenceReconstructionStop, match="existiert bereits"):
        ctx.run()
    assert ctx.accesses() == []  # STOP vor jedem Datenzugriff
    assert list(ctx.output.iterdir()) == []


# --- Falsche Pool-/Canonical-Auswahl, manipulierte IDs und Hashes -------------------------------


def _manifest_sha256(reference: ReconstructedReference) -> str:
    """Hash des write-once Manifests aus dem Ergebnisobjekt (nicht aus dem Dateizustand)."""
    return sha256_hex(canonical_json_bytes(reference.evidence))


def _edit_pool_manifest(frozen: FrozenCalibration, edit: str) -> bytes:
    data = json.loads(frozen.pool_manifest_bytes)
    blocks = [b for b in data["ref_tune_blocks"] if b["pool"] == "CAL_REF"]
    tune = [b for b in data["ref_tune_blocks"] if b["pool"] == "CAL_TUNE"]
    if edit == "reordered":
        blocks[1]["lesion_ids"] = list(reversed(blocks[1]["lesion_ids"]))
    elif edit == "swapped-with-tune":
        blocks[1]["lesion_ids"][0], tune[1]["lesion_ids"][0] = (
            tune[1]["lesion_ids"][0],
            blocks[1]["lesion_ids"][0],
        )
    else:
        blocks[0]["lesion_ids"] = blocks[0]["lesion_ids"][:-1]
    return canonical_json_bytes(data)


@pytest.mark.parametrize("edit", ["reordered", "swapped-with-tune", "removed-lesion"])
def test_wrong_pool_selection_stops_without_data_access(ctx: Context, edit: str) -> None:
    frozen = dataclasses.replace(
        ctx.frozen, pool_manifest_bytes=_edit_pool_manifest(ctx.frozen, edit)
    )
    with pytest.raises(ReferenceReconstructionStop, match="Pool"):
        ctx.run(frozen=frozen)
    assert ctx.accesses() == [] and ctx.no_result()


@pytest.mark.parametrize("edit", ["other-sample-of-lesion", "missing-lesion"])
def test_wrong_canonical_selection_stops_without_data_access(ctx: Context, edit: str) -> None:
    pairs = json.loads(ctx.frozen.canonical_samples_bytes)
    if edit == "missing-lesion":
        pairs = pairs[1:]
    else:
        lesion_samples: dict[str, list[str]] = {}
        for a in ctx.calibrated.world.split.assignments:
            lesion_samples.setdefault(a.group_id, []).append(a.sample_id)
        index = next(i for i, (lesion, _) in enumerate(pairs) if len(lesion_samples[lesion]) > 1)
        lesion, sample = pairs[index]
        pairs[index] = [lesion, next(s for s in lesion_samples[lesion] if s != sample)]
    frozen = dataclasses.replace(ctx.frozen, canonical_samples_bytes=canonical_json_bytes(pairs))
    with pytest.raises(ReferenceReconstructionStop, match="Canonical"):
        ctx.run(frozen=frozen)
    assert ctx.accesses() == [] and ctx.no_result()


def test_canonical_rule_is_rechecked_from_the_split(ctx: Context) -> None:
    """Auch ein (hypothetisch) hashkonsistentes Mapping muss der Canonical-Hash-Regel folgen."""
    pairs = json.loads(ctx.frozen.canonical_samples_bytes)
    lesion_samples: dict[str, list[str]] = {}
    for a in ctx.calibrated.world.split.assignments:
        lesion_samples.setdefault(a.group_id, []).append(a.sample_id)
    reference_lesions = set(ctx.frozen.record.reference_lesion_ids)
    index = next(
        i
        for i, (lesion, _) in enumerate(pairs)
        if lesion in reference_lesions and len(lesion_samples[lesion]) > 1
    )
    lesion, sample = pairs[index]
    pairs[index] = [lesion, next(s for s in lesion_samples[lesion] if s != sample)]
    data = canonical_json_bytes(pairs)
    record = ctx.frozen.record.model_copy(update={"canonical_samples_sha256": sha256_hex(data)})
    frozen = dataclasses.replace(ctx.frozen, canonical_samples_bytes=data, record=record)
    with pytest.raises(ReferenceReconstructionStop, match="Hash-Regel"):
        derive_reference(frozen, ctx.calibrated.world.split)


@pytest.mark.parametrize(
    "field", ["reference_sample_ids", "replaced-sample", "reference_window_id", "hash"]
)
def test_manipulated_frozen_reference_ids_or_hashes_stop(ctx: Context, field: str) -> None:
    record = ctx.frozen.record
    if field == "reference_sample_ids":
        ids = list(record.reference_sample_ids)
        ids[0], ids[1] = ids[1], ids[0]  # gleiche Menge, andere Reihenfolge
        update: dict[str, object] = {field: tuple(ids)}
    elif field == "replaced-sample":
        others = sorted(
            partition_sample_ids(ctx.calibrated.world.split, Partition.CALIBRATION)
            - set(record.reference_sample_ids)
        )
        update = {"reference_sample_ids": (*record.reference_sample_ids[:-1], others[0])}
    elif field == "reference_window_id":
        update = {field: "0" * 64}
    else:
        update = {"reference_sha256": "0" * 64}
    frozen = dataclasses.replace(ctx.frozen, record=record.model_copy(update=update))
    with pytest.raises(ReferenceReconstructionStop, match="weicht vom Freeze-Record ab"):
        derive_reference(frozen, ctx.calibrated.world.split)
    with pytest.raises(ReferenceReconstructionStop):  # STOP vor Ausstellung der Berechtigung
        ctx.run(frozen=frozen)
    assert ctx.accesses() == [] and ctx.no_result()


@pytest.mark.parametrize(
    "path",
    [
        "freeze/calibration_freeze_record.json",
        "freeze/artifact_manifest.json",
        "calibration/pool_manifest.json",
        "calibration/canonical_samples.json",
        "freeze/run_config.json",
    ],
)
def test_tampered_freeze_artifacts_are_rejected_when_loading(
    ctx: Context, tmp_path: Path, path: str
) -> None:
    copy = tmp_path / "calibration-copy"
    shutil.copytree(ctx.calibrated.result.output_dir, copy)
    target = copy / path
    value = json.loads(target.read_bytes())
    if path.endswith("calibration_freeze_record.json"):
        value["reference_window_id"] = "0" * 64
    elif path.endswith("artifact_manifest.json"):
        value["run_id"] = f"{value['run_id']}-x"
    elif path.endswith("pool_manifest.json"):
        value["split_hash"] = "0" * 64
    elif path.endswith("canonical_samples.json"):
        value = value[1:]
    else:
        value["scientific"]["relevance"]["delta_rel"] = 0.5
    target.chmod(0o644)
    target.write_bytes(canonical_json_bytes(value))
    with pytest.raises(IntegrityError):
        load_frozen_calibration(copy)


# --- Abweichende BA_ref und Ausführungsbindung --------------------------------------------------


@pytest.mark.parametrize("delta", [-0.0001, 0.0001])
def test_deviating_ba_ref_is_a_terminal_stop_without_result(ctx: Context, delta: float) -> None:
    expected = ctx.calibrated.runner.process.results.ba_ref
    assert expected is not None
    wrong = ctx.expectation.model_copy(update={"ba_ref_4dp": f"{expected + delta:.4f}"})
    with pytest.raises(ReferenceReconstructionStop, match="BA_ref"):
        ctx.run(expectation=wrong)
    assert ctx.no_result()
    with pytest.raises(PhaseViolationError, match="bereits einmal berechtigt"):
        ctx.run()  # kein automatischer zweiter Lauf auf demselben Guard


class _OtherSnapshotDataset(RecordingDataset):
    def snapshot_ref(self) -> DatasetSnapshotRef:
        return DatasetSnapshotRef(snapshot_id="other", uri="memory://other", sha256="0" * 64)


class _OtherModel:
    def __init__(self, model: TextureModel) -> None:
        self._model = model

    def snapshot_ref(self) -> ModelSnapshotRef:
        return self._model.snapshot_ref().model_copy(update={"sha256": "0" * 64})

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> ModelOutputBatch:
        return self._model.predict_proba(window_id, sample_ids, inputs)

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> RepresentationBatch:
        return self._model.embed(window_id, sample_ids, inputs)


@pytest.mark.parametrize(
    "binding",
    [
        "device",
        "batch",
        "observed-device",
        "lock",
        "dataset",
        "model",
        "preprocessing",
        "freeze-id",
        "class-vector",
    ],
)
def test_wrong_execution_binding_stops_before_any_data_access(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: str
) -> None:
    overrides: dict[str, Any] = {}
    if binding == "device":
        overrides["execution"] = ReferenceExecution(device="mps", batch_size=32)
    elif binding == "batch":
        overrides["execution"] = ReferenceExecution(device="cpu", batch_size=16)
    elif binding == "observed-device":
        monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    elif binding == "lock":
        lock = tmp_path / "requirements-lock.txt"
        lock.write_text("numpy==2.1\n", "utf-8")
        overrides["environment_lock"] = lock
    elif binding == "dataset":
        overrides["dataset"] = _OtherSnapshotDataset(ctx.calibrated.dataset)
    elif binding == "model":
        overrides["model"] = _OtherModel(ctx.calibrated.model)
    elif binding == "preprocessing":
        spec = ctx.calibrated.preprocessor.spec.model_copy(update={"antialias": False})
        overrides["preprocessor"] = ImagePreprocessor(spec)
    elif binding == "freeze-id":
        overrides["expectation"] = ctx.expectation.model_copy(update={"freeze_id": "0" * 64})
    else:
        counts = list(ctx.expectation.class_counts)
        counts[0], counts[1] = (counts[0][0], counts[0][1] + 1), (counts[1][0], counts[1][1] - 1)
        overrides["expectation"] = ctx.expectation.model_copy(
            update={"class_counts": tuple(counts)}
        )
    with pytest.raises(ReferenceReconstructionStop):
        ctx.run(**overrides)
    assert ctx.accesses() == [] and ctx.no_result()
    assert all(not d.accesses for d in overrides.values() if isinstance(d, RecordingDataset))
    if binding != "observed-device":
        assert ctx.run().ba_ref is not None  # Berechtigung wurde nicht verbraucht


def test_model_snapshot_is_checked_before_loading(
    ctx: Context, model_snapshot: tuple[Path, ModelSnapshotRef]
) -> None:
    path, _ = model_snapshot  # untrainierter Initialisierungs-Snapshot, nicht `resnet18-pilot-P1`
    with pytest.raises(ReferenceReconstructionStop, match="model_id"):
        load_reference_model(
            path, record=ctx.frozen.record, preprocessor=ctx.calibrated.preprocessor
        )


# --- Verbotene Zugriffe, Zweitnutzung, Abbruch --------------------------------------------------


class _GreedyDataset(RecordingDataset):
    """Versucht über die erhaltene Berechtigung mehr als vorgesehen zu lesen."""

    def __init__(self, dataset: Ham10000Dataset, attempt: str, foreign: str) -> None:
        super().__init__(dataset)
        self.attempt = attempt
        self.foreign = foreign
        self.permit: DataAccessContext | None = None

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> list[Image.Image]:
        self.permit = access
        images = super().load_inputs(access, sample_ids)
        if self.attempt == "second-read":
            super().load_inputs(access, sample_ids[:1])
        elif self.attempt == "foreign-sample":
            super().load_inputs(access, [self.foreign])
        elif self.attempt == "metadata":
            super().metadata_for(access, sample_ids[:1])
        return images


@pytest.mark.parametrize("attempt", ["second-read", "foreign-sample", "metadata"])
def test_forbidden_accesses_through_the_permit_are_rejected(ctx: Context, attempt: str) -> None:
    others = partition_sample_ids(ctx.calibrated.world.split, Partition.CALIBRATION) - set(
        ctx.frozen.record.reference_sample_ids
    )
    greedy = _GreedyDataset(ctx.calibrated.dataset, attempt, sorted(others)[0])
    ctx.recorder = greedy
    with pytest.raises(PhaseViolationError):
        ctx.run()
    assert ctx.no_result()
    assert greedy.permit is not None
    with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
        greedy.permit.require_access(
            ctx.frozen.record.reference_sample_ids[:1], AccessPurpose.LABELS
        )


def test_permit_cannot_be_used_after_reconstruction(ctx: Context) -> None:
    greedy = _GreedyDataset(ctx.calibrated.dataset, "none", "")
    ctx.recorder = greedy
    ctx.run()
    assert greedy.permit is not None
    held_out = sorted(partition_sample_ids(ctx.calibrated.world.split, Partition.HELD_OUT))
    for ids in (ctx.frozen.record.reference_sample_ids[:1], held_out[:1]):
        with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
            greedy.permit.require_access(ids, AccessPurpose.INPUTS)
    for partition in Partition:  # alle übrigen Zugriffspfade bleiben in FROZEN gesperrt
        with pytest.raises(PhaseViolationError, match="FROZEN"):
            ctx.guard.assert_partition_access(partition)


class _RevokingDataset(RecordingDataset):
    """Entfernt nach dem ersten Batch den Freigabenachweis (Abbruch mitten im Zugriff)."""

    def __init__(self, dataset: Ham10000Dataset, approval: Path) -> None:
        super().__init__(dataset)
        self._approval = approval

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> list[Image.Image]:
        images = super().load_inputs(access, sample_ids)
        if self._approval.exists():
            self._approval.unlink()
        return images


def test_abort_during_access_leaves_no_valid_result(ctx: Context) -> None:
    ctx.recorder = _RevokingDataset(ctx.calibrated.dataset, Path(ctx.provenance.approval_path))
    with pytest.raises(IntegrityError):
        ctx.run()
    assert ctx.no_result()
    assert len(ctx.accesses()) == 32  # nur der erste Batch, danach STOP
    with pytest.raises(PhaseViolationError, match="bereits einmal berechtigt"):
        ctx.run()


def test_write_failure_leaves_no_valid_result(
    ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(source: object, target: object) -> None:
        raise OSError("synthetischer Schreibfehler")

    monkeypatch.setattr(f"{reference_reconstruction.__name__}.os.rename", fail)
    with pytest.raises(OSError, match="synthetischer Schreibfehler"):
        ctx.run()
    assert ctx.no_result()


def test_labels_are_bound_to_the_frozen_reference(ctx: Context) -> None:
    """G4 nutzt exakt die 128 Labels der Reference in der Reihenfolge der sortierten IDs."""
    reference = ctx.run()
    labels: Mapping[str, int] = ctx.calibrated.runner._features._labels
    expected = [labels[s] for s in reference.sample_ids]
    matrix = reference.performance.confusion_matrix
    assert matrix is not None
    support = [sum(row) for row in matrix]
    assert support == [expected.count(c) for c in range(len(PROTOCOL.class_order))]


# --- P4-03AG: verbindliche Sollwerte realer Freezes (P4-03AF, HIGH) -----------------------

REAL_FREEZE_ID = "58b1e879f3adc03f9f84b10810f9ed727ab0eba41bbde891786c8f529dc52a93"


def test_real_freeze_values_are_registered_and_cannot_be_replaced() -> None:
    binding = resolve_expectation(REAL_FREEZE_ID, None)
    assert binding is P4_03_REFERENCE_EXPECTATION
    assert binding.ba_ref_4dp == "0.6586"
    assert dict(binding.class_counts) == {
        "MEL": 11,
        "NV": 92,
        "BCC": 6,
        "AKIEC": 4,
        "BKL": 12,
        "DF": 1,
        "VASC": 2,
    }
    assert resolve_expectation(REAL_FREEZE_ID, binding.model_copy()) is binding
    counts = (("MEL", 12), *binding.class_counts[1:])
    for update in ({"ba_ref_4dp": "0.6587"}, {"class_counts": counts}, {"ba_ref_source": "x"}):
        with pytest.raises(ReferenceReconstructionStop, match="widersprüchliche Erwartung"):
            resolve_expectation(REAL_FREEZE_ID, binding.model_copy(update=update))
    with pytest.raises(TypeError):
        reference_reconstruction._BINDING_EXPECTATIONS[REAL_FREEZE_ID] = binding  # type: ignore[index]


def _register(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, ba_ref_4dp: str
) -> ReferenceExpectation:
    """Simuliert den realen Pfad: die synthetische Freeze-ID wird als verbindlicher Freeze
    registriert (der reale Freeze `58b1e879…` ist synthetisch nicht erzeugbar)."""
    binding = ctx.expectation.model_copy(
        update={"ba_ref_4dp": ba_ref_4dp, "ba_ref_source": "simulierter verbindlicher Lauf-Log"}
    )
    registry = MappingProxyType({binding.freeze_id: binding})
    monkeypatch.setattr(reference_reconstruction, "_BINDING_EXPECTATIONS", registry, raising=False)
    return binding


def _next_4dp(value: str) -> str:
    return f"{float(value) + 0.0001:.4f}"


@pytest.mark.parametrize("field", ["ba_ref_4dp", "class_counts", "ba_ref_source"])
def test_contradicting_expectation_for_a_registered_freeze_stops_before_access(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """P4-03AF: der Aufrufer ersetzt die verbindliche BA_ref durch den passenden Wert."""
    reconstructed = ctx.expectation.ba_ref_4dp
    binding = _register(ctx, monkeypatch, _next_4dp(reconstructed))
    (first, n), *rest = binding.class_counts
    counts = ((first, n + 1), *rest)
    values: dict[str, object] = {
        "ba_ref_4dp": reconstructed,
        "class_counts": counts,
        "ba_ref_source": "Aufrufer",
    }
    manipulated = binding.model_copy(update={field: values[field]})
    with pytest.raises(ReferenceReconstructionStop, match="widersprüchliche Erwartung"):
        ctx.run(expectation=manipulated)
    assert ctx.accesses() == [] and ctx.no_result()


def test_registered_expectation_stays_binding_without_caller_value(
    ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register(ctx, monkeypatch, _next_4dp(ctx.expectation.ba_ref_4dp))
    with pytest.raises(ReferenceReconstructionStop, match="BA_ref"):
        ctx.run(expectation=None)
    assert ctx.no_result()


def test_registered_expectation_passes_when_the_reconstruction_matches(
    ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _register(ctx, monkeypatch, ctx.expectation.ba_ref_4dp)
    reference = ctx.run(expectation=None)
    execution = json.loads((ctx.output / "provenance/execution.json").read_bytes())
    assert execution["ba_ref_expected_4dp"] == binding.ba_ref_4dp == f"{reference.ba_ref:.4f}"
    assert execution["ba_ref_source"] == binding.ba_ref_source
    assert execution["ba_ref"] == reference.ba_ref  # ungerundet gespeichert


def test_synthetic_freeze_requires_its_own_distinct_expectation(ctx: Context) -> None:
    assert ctx.frozen.manifest.freeze_id != REAL_FREEZE_ID
    with pytest.raises(ReferenceReconstructionStop, match="fehlt die Erwartung"):
        ctx.run(expectation=None)
    with pytest.raises(ReferenceReconstructionStop, match="gehört nicht zu diesem Freeze"):
        ctx.run(expectation=P4_03_REFERENCE_EXPECTATION)
    assert ctx.accesses() == [] and ctx.no_result()
    assert ctx.run().ba_ref is not None  # eigene synthetische Erwartung → PASS

"""Einmalige Rekonstruktion der eingefrorenen Reference-Artefakte vor HELD_OUT (DEC-084/085).

Ablauf (Protokoll v1.1, Abschnitt 2); jede Abweichung führt zum terminalen STOP:

1. Bindungen vor jedem Datenzugriff: Freeze, Evaluation-Provenienz, Ausführung (`CPU`,
   `batch_size = 32`, ohne Fallback), Modell-Snapshot `resnet18-pilot-P1`, Preprocessing,
   Environment-Lock, Datensatz-Snapshot und die eingefrorenen G2-/G4-Methodenkonfigurationen.
2. Unabhängige strukturelle Ableitung der Reference (`reference_sources.derive_reference`).
3. Datenzugriff ausschließlich über die `ReferenceReconstructionPermit` des Guards: INPUTS und
   LABELS je Sample genau einmal in disjunkten Batches; danach wird die Berechtigung geschlossen
   und nur noch mit den eingelesenen Daten gerechnet.
4. G1-Embeddings, G2-Indikatoren, G3-Softmax und G4-Referenzleistung mit dem eingefrorenen Modell
   und den bestehenden Signalbausteinen (`compute_quality_indicators`, `evaluate_performance`) in
   derselben Zusammensetzung wie im Calibration-Runner (Bildzustand vor der ImageNet-
   Normalisierung, Merkmale je Sample, Vorhersage = erster Index der maximalen Wahrscheinlichkeit).
5. `BA_ref` muss mit vier Nachkommastellen dem historischen Wert entsprechen; der ungerundete
   Wert wird unverändert übernommen. Keine Behauptung bitidentischer historischer Merkmale.
6. Write-once-Evidence: alle Artefakte zunächst in ein Staging-Verzeichnis, Manifest zuletzt,
   danach atomare Umbenennung. Ein Fehler hinterlässt kein als gültig gekennzeichnetes Ergebnis.
"""

import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

import numpy as np
import torch
from PIL import Image
from pydantic import Field

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    ImagePreprocessor,
    PreprocessedImage,
)
from vigiscoutx.adapters.resnet18 import (
    ModelSnapshotManifest,
    ResNet18ModelAdapter,
    snapshot_ref_for_manifest,
)
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.artifacts.manifests import ArtifactEntry
from vigiscoutx.config.scientific_config import ScientificConfig
from vigiscoutx.domain.assessments import PerformanceEvaluation, SignalStatus
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import DatasetSnapshotRef, ModelSnapshotRef
from vigiscoutx.domain.types import (
    FrozenModel,
    GitCommitHash,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
    SignalGroup,
)
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.evaluation.reference_sources import (
    DerivedReference,
    FrozenCalibration,
    ReferenceReconstructionStop,
    derive_reference,
)
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    ExecutionPurpose,
    reverify_evaluation_provenance,
)
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.types import Phase
from vigiscoutx.ports import DataAccessContext
from vigiscoutx.signals.config import G2QualityConfig, method_config_from_signal_config
from vigiscoutx.signals.data_quality.quality import (
    QualityIndicatorBatch,
    QualityIndicators,
    QualityInputFailure,
    compute_quality_indicators,
)
from vigiscoutx.signals.performance.g4 import G4PerformanceConfig, evaluate_performance

REFERENCE_DEVICE: Final = "cpu"
REFERENCE_BATCH_SIZE: Final = 32
EVIDENCE_LIMIT: Final = (
    "BA_ref nur auf vier Nachkommastellen gegen den historischen Lauf-Log geprüft; der "
    "ungerundete Wert ist neu berechnet. Bitidentität historischer G1-Embeddings, "
    "G2-Indikatoren, G3-Softmaxwerte oder BA_ref wird nicht behauptet (DEC-085)."
)
FURTHER_HISTORICAL_METRICS: Final = (
    "keine: außer BA_ref (vier Nachkommastellen) ist keine historische Kennzahl genau der "
    "128 Reference-Samples mit belegter Präzision und Quelle vorhanden (σ stammt aus dem "
    "gesamten CAL_REF-Pool und liegt außerhalb dieser Berechtigung)"
)
EVIDENCE_MANIFEST_FILE: Final = "reference_manifest.json"
_G2_METHOD = "G2_QUALITY_KS_BONFERRONI"
_FILES: Final = {
    "reference.selection": "reference/selection.json",
    "reference.g1_representation": "reference/g1_representation.json",
    "reference.g2_quality": "reference/g2_quality.json",
    "reference.g3_model_outputs": "reference/g3_model_outputs.json",
    "reference.g4_performance": "reference/g4_performance.json",
    "provenance.execution": "provenance/execution.json",
}


def _stop(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceReconstructionStop(message)


# --- Erwartung, Ausführung und Ports ------------------------------------------------------------


class ReferenceExpectation(FrozenModel):
    """Vorab festgelegte Prüfwerte genau eines Freeze (Protokoll v1.1, Abschnitte 1 und 2)."""

    freeze_id: Sha256Hex
    class_counts: tuple[tuple[Identifier, NonNegativeInt], ...]
    ba_ref_4dp: str = Field(pattern=r"^[01]\.[0-9]{4}$")
    ba_ref_source: str = Field(min_length=1)


P4_03_REFERENCE_EXPECTATION: Final = ReferenceExpectation(
    freeze_id="58b1e879f3adc03f9f84b10810f9ed727ab0eba41bbde891786c8f529dc52a93",
    class_counts=(
        ("MEL", 11),
        ("NV", 92),
        ("BCC", 6),
        ("AKIEC", 4),
        ("BKL", 12),
        ("DF", 1),
        ("VASC", 2),
    ),
    ba_ref_4dp="0.6586",
    ba_ref_source="P4-03M-Lauf-Log (P4-03-calibration-real-2), HELD_OUT_EVALUATION_PROTOCOL v1.1",
)
"""Prüfwerte des gültigen Freeze `58b1e879…`; ausschließlich aus Protokoll v1.1 übernommen."""

_BINDING_EXPECTATIONS: Final[Mapping[str, ReferenceExpectation]] = MappingProxyType(
    {P4_03_REFERENCE_EXPECTATION.freeze_id: P4_03_REFERENCE_EXPECTATION}
)
"""Verbindliche Sollwerte realer Freezes; vom Aufrufer weder ersetzbar noch abschwächbar."""


def resolve_expectation(
    freeze_id: str, supplied: ReferenceExpectation | None
) -> ReferenceExpectation:
    """Verbindliche Erwartung für `freeze_id` (P4-03AG).

    Realer, registrierter Freeze: ausschließlich die eingetragenen Sollwerte; eine übergebene
    Erwartung muss exakt identisch sein, sonst STOP. Nicht registrierter (synthetischer) Freeze:
    nur eine ausdrücklich übergebene Erwartung genau dieses Freeze.
    """
    binding = _BINDING_EXPECTATIONS.get(freeze_id)
    if binding is not None:
        _stop(
            supplied is None or supplied == binding,
            "widersprüchliche Erwartung für einen realen Freeze; verbindlich sind ausschließlich "
            "die registrierten Sollwerte",
        )
        return binding
    _stop(supplied is not None, "für einen nicht registrierten Freeze fehlt die Erwartung")
    assert supplied is not None
    _stop(
        supplied.freeze_id == freeze_id and supplied.freeze_id not in _BINDING_EXPECTATIONS,
        "Erwartung gehört nicht zu diesem Freeze",
    )
    return supplied


class ReferenceExecution(FrozenModel):
    """Vorab fixierte technische Ausführung; jede andere Angabe ist ein STOP (kein Fallback)."""

    device: str
    batch_size: PositiveInt


class ReferenceDataPort(Protocol):
    """Zugriffsgebundener Datensatz; jeder Aufruf prüft die übergebene Berechtigung."""

    def snapshot_ref(self) -> DatasetSnapshotRef: ...

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Sequence[Image.Image]: ...

    def labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...


class ReferenceModelPort(Protocol):
    """Ausschließlich lesende Inferenz auf Bildzuständen vor der ImageNet-Normalisierung."""

    def snapshot_ref(self) -> ModelSnapshotRef: ...

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> ModelOutputBatch: ...

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> RepresentationBatch: ...


def load_reference_model(
    manifest_path: Path, *, record: CalibrationFreezeRecord, preprocessor: ImagePreprocessor
) -> ResNet18ModelAdapter:
    """Eingefrorenes P1-Modell über den unveränderten `ResNet18ModelAdapter.from_snapshot()`.

    Gewichte per `map_location="cpu"` (Adapter), `batch_size = 32`; Snapshot, Gewichte und
    Preprocessing werden vor dem Laden gegen den Freeze-Record geprüft.
    """
    snapshot = ModelSnapshotManifest.model_validate(json.loads(manifest_path.read_bytes()))
    ref = snapshot_ref_for_manifest(snapshot)
    checks = {
        "model_id": (ref.model_id, record.model_id),
        "model_snapshot_sha256": (ref.sha256, record.model_snapshot_sha256),
        "model_weights_sha256": (snapshot.weights_sha256, record.model_weights_sha256),
        "preprocessing": (snapshot.preprocessing_sha256, preprocessor.spec_sha256),
    }
    mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
    _stop(not mismatches, f"Modell-Snapshot passt nicht zum Freeze: {mismatches}")
    return ResNet18ModelAdapter.from_snapshot(
        manifest_path,
        expected=ref,
        class_order=DEMONSTRATOR_CLASS_ORDER,
        preprocessor=preprocessor,
        batch_size=REFERENCE_BATCH_SIZE,
        allow_untrained=False,
    )


# --- Ergebnis und Evidence ----------------------------------------------------------------------


class ReferenceEvidenceManifest(FrozenModel):
    """Write-once Hash-Manifest der Reference-Artefakte; bindet Freeze und Evaluation-Commit."""

    schema_version: Literal[1]
    kind: Literal["FROZEN_REFERENCE_RECONSTRUCTION"]
    freeze_id: Sha256Hex
    calibration_commit: GitCommitHash
    evaluation_commit: GitCommitHash
    approval_sha256: Sha256Hex
    policy_version: str
    reference_window_id: Sha256Hex
    reference_sha256: Sha256Hex
    entries: tuple[ArtifactEntry, ...]

    def artifacts_sha256(self) -> str:
        return canonical_sha256([e.model_dump(mode="json") for e in self.entries])


@dataclass(frozen=True, slots=True)
class ReconstructedReference:
    """Unveränderliche Reference-Artefakte; einmal erzeugt, danach nur wiederverwendet."""

    derived: DerivedReference
    representation: RepresentationBatch
    outputs: ModelOutputBatch
    quality: QualityIndicatorBatch
    performance: PerformanceEvaluation
    ba_ref: float
    evidence: ReferenceEvidenceManifest
    directory: Path

    @property
    def window_id(self) -> str:
        return self.derived.window_id

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.representation.sample_ids


# --- Prüfungen vor dem Datenzugriff -------------------------------------------------------------


def _verify_execution(execution: ReferenceExecution) -> dict[str, object]:
    _stop(
        execution.device == REFERENCE_DEVICE,
        f"Gerät {execution.device!r} unzulässig; vorab fixiert ist {REFERENCE_DEVICE!r}",
    )
    _stop(
        execution.batch_size == REFERENCE_BATCH_SIZE,
        f"batch_size {execution.batch_size} unzulässig; vorab fixiert ist {REFERENCE_BATCH_SIZE}",
    )
    default_device = torch.get_default_device().type
    _stop(
        default_device == REFERENCE_DEVICE,
        f"tatsächliches Standardgerät {default_device!r} ist nicht {REFERENCE_DEVICE!r}",
    )
    return {
        "requested_device": execution.device,
        "observed_default_device": default_device,
        "batch_size": execution.batch_size,
        "torch_version": torch.__version__,
    }


def _verify_components(
    frozen: FrozenCalibration,
    dataset: ReferenceDataPort,
    model: ReferenceModelPort,
    preprocessor: ImagePreprocessor,
    environment_lock: Path,
) -> dict[str, object]:
    record, manifest = frozen.record, frozen.manifest
    scientific = frozen.run_config.scientific
    model_ref = model.snapshot_ref()
    dataset_ref = dataset.snapshot_ref()
    _stop(environment_lock.is_file(), "Environment-Lock fehlt")
    lock_sha256 = sha256_file(environment_lock)
    preprocessing_sha256 = canonical_sha256(scientific.preprocessing)
    checks: dict[str, tuple[object, object]] = {
        "model_id": (model_ref.model_id, record.model_id),
        "model_snapshot_sha256": (model_ref.sha256, record.model_snapshot_sha256),
        "model_weights_sha256": (scientific.model.sha256, record.model_weights_sha256),
        "manifest.model_weights_sha256": (
            manifest.model_weights_sha256,
            record.model_weights_sha256,
        ),
        "preprocessing_sha256": (preprocessing_sha256, record.preprocessing_sha256),
        "preprocessing_spec": (
            preprocessor.spec,
            ImagePreprocessingSpec.from_config(scientific.preprocessing),
        ),
        "environment_lock_sha256": (lock_sha256, record.environment_lock_sha256),
        "dataset_snapshot": (
            (dataset_ref.snapshot_id, dataset_ref.sha256),
            (record.dataset_snapshot_id, record.dataset_snapshot_sha256),
        ),
    }
    mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
    _stop(not mismatches, f"Ausführungsbindung weicht vom Freeze ab: {mismatches}")
    return {
        "model_id": model_ref.model_id,
        "model_snapshot_sha256": model_ref.sha256,
        "model_weights_sha256": record.model_weights_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "preprocessing_spec_sha256": preprocessor.spec_sha256,
        "environment_lock_sha256": lock_sha256,
        "dataset_snapshot_id": dataset_ref.snapshot_id,
        "dataset_snapshot_sha256": dataset_ref.sha256,
    }


def _method_configs(
    scientific: ScientificConfig, record: CalibrationFreezeRecord
) -> tuple[G2QualityConfig, G4PerformanceConfig]:
    """Eingefrorene G2-Konfiguration aus der Run-Konfiguration, G4 mit dem einzig zulässigen Wert;
    beide müssen exakt in den Methodenidentitäten des Freeze-Records stehen."""
    signals = [s for s in scientific.monitoring.signals if s.method == _G2_METHOD]
    _stop(
        len(signals) == 1 and signals[0].group is SignalGroup.G2_DATA_QUALITY,
        "Run-Konfiguration benötigt genau ein G2-Signal",
    )
    g2 = method_config_from_signal_config(signals[0], G2QualityConfig)
    g4 = G4PerformanceConfig(
        performance_id="G4_PERFORMANCE",
        method_version="monitoring-methods-baseline:1.2",
        class_order=DEMONSTRATOR_CLASS_ORDER,
        metric="BALANCED_ACCURACY",
        missing_class_rule="NOT_EVALUABLE_NO_REAVERAGING",
        zero_division="NULL",
    )
    missing = [
        c.parameters_ref().config_id
        for c in (g2, g4)
        if c.parameters_ref() not in record.method_refs
    ]
    _stop(not missing, f"Methodenkonfiguration nicht im Freeze-Record gebunden: {missing}")
    return g2, g4


# --- Datenzugriff und Merkmale ------------------------------------------------------------------


def _chunks(sample_ids: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(sample_ids), size):
        yield list(sample_ids[start : start + size])


def _read_once(
    dataset: ReferenceDataPort,
    permit: DataAccessContext,
    sample_ids: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, Image.Image], dict[str, int]]:
    """INPUTS und LABELS je Sample genau einmal, in disjunkten Batches über die Berechtigung."""
    images: dict[str, Image.Image] = {}
    labels: dict[str, int] = {}
    for chunk in _chunks(sample_ids, batch_size):
        loaded = list(dataset.load_inputs(permit, chunk))
        _stop(len(loaded) == len(chunk), "Datensatz lieferte nicht alle angefragten Bilder")
        images.update(zip(chunk, loaded, strict=True))
        chunk_labels = dict(dataset.labels_for(permit, chunk))
        _stop(set(chunk_labels) == set(chunk), "Datensatz lieferte nicht alle Labels")
        labels.update(chunk_labels)
    return images, labels


def _argmax(row: Sequence[float]) -> int:
    """Erster Index der maximalen Wahrscheinlichkeit (wie im Calibration-Runner)."""
    best = 0
    for index, value in enumerate(row):
        if value > row[best]:
            best = index
    return best


@dataclass(frozen=True, slots=True)
class _Features:
    embeddings: tuple[tuple[float, ...], ...]
    probabilities: tuple[tuple[float, ...], ...]
    quality: tuple[QualityIndicators, ...]
    batches: tuple[int, ...]
    input_devices: tuple[str, ...]


def _features(
    window_id: str,
    sample_ids: Sequence[str],
    images: Mapping[str, Image.Image],
    model: ReferenceModelPort,
    preprocessor: ImagePreprocessor,
    g2: G2QualityConfig,
    batch_size: int,
) -> _Features:
    embeddings: list[tuple[float, ...]] = []
    probabilities: list[tuple[float, ...]] = []
    quality: list[QualityIndicators] = []
    batches: list[int] = []
    devices: set[str] = set()
    for chunk in _chunks(sample_ids, batch_size):
        states = [
            np.ascontiguousarray(preprocessor.unnormalized(images[s]).numpy(), dtype=np.float32)
            for s in chunk
        ]
        inputs = [
            PreprocessedImage(torch.from_numpy(np.asarray(state, dtype=np.float32)))
            for state in states
        ]
        devices |= {i.tensor.device.type for i in inputs}
        _stop(devices == {REFERENCE_DEVICE}, f"Eingaben nicht auf {REFERENCE_DEVICE}: {devices}")
        embedded = model.embed(window_id, chunk, inputs)
        predicted = model.predict_proba(window_id, chunk, inputs)
        _stop(
            embedded.sample_ids == tuple(chunk) == predicted.sample_ids,
            "Modell-Ausgaben gehören nicht exakt zum angefragten Batch",
        )
        embeddings += embedded.embeddings
        probabilities += predicted.probabilities
        for sample_id, state in zip(chunk, states, strict=True):
            result = compute_quality_indicators(sample_id, [sample_id], [state], g2)
            _stop(
                not isinstance(result, QualityInputFailure),
                f"G2-Eingabevertrag der Reference verletzt: {sample_id}",
            )
            assert isinstance(result, QualityIndicatorBatch)
            quality.append(result.indicators[0])
        batches.append(len(chunk))
    return _Features(
        embeddings=tuple(embeddings),
        probabilities=tuple(probabilities),
        quality=tuple(quality),
        batches=tuple(batches),
        input_devices=tuple(sorted(devices)),
    )


# --- Write-once-Evidence ------------------------------------------------------------------------


def _write_evidence(
    output_dir: Path, payloads: Mapping[str, object], binding: Mapping[str, str]
) -> ReferenceEvidenceManifest:
    """Staging → Manifest zuletzt → atomare Umbenennung; bei Fehler kein gültiges Ergebnis."""
    staging = output_dir.with_name(f".{output_dir.name}.incomplete")
    _stop(not output_dir.exists(), f"Ausgabeverzeichnis existiert bereits: {output_dir}")
    staging.mkdir(exist_ok=False)
    try:
        entries: list[ArtifactEntry] = []
        for role, relative in _FILES.items():
            data = canonical_json_bytes(payloads[role])
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
            target.chmod(0o444)
            entries.append(ArtifactEntry(path=relative, sha256=sha256_hex(data), role=role))
        evidence = ReferenceEvidenceManifest.model_validate(
            {
                "schema_version": 1,
                "kind": "FROZEN_REFERENCE_RECONSTRUCTION",
                **binding,
                "entries": entries,
            }
        )
        manifest_path = staging / EVIDENCE_MANIFEST_FILE
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_bytes(evidence))
        manifest_path.chmod(0o444)
        _stop(not output_dir.exists(), f"Ausgabeverzeichnis existiert bereits: {output_dir}")
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return evidence


# --- Rekonstruktion -----------------------------------------------------------------------------


def reconstruct_reference(
    *,
    guard: HeldOutGuard,
    frozen: FrozenCalibration,
    split_manifest: SplitManifest,
    dataset: ReferenceDataPort,
    model: ReferenceModelPort,
    preprocessor: ImagePreprocessor,
    environment_lock: Path,
    provenance: EvaluationProvenance,
    expectation: ReferenceExpectation | None,
    execution: ReferenceExecution,
    output_dir: Path,
) -> ReconstructedReference:
    """Genau eine Reference-Rekonstruktion je Guard; jeder Fehler ist ein terminaler STOP.

    `output_dir` darf noch nicht existieren und muss außerhalb des versionierten Arbeitsbaums
    liegen (z. B. git-ignoriertes `experiments/results/raw/`), damit die Provenienzprüfung des
    ausgeführten Repositorys unverändert gültig bleibt. Für einen registrierten realen Freeze gelten
    ausschließlich dessen Sollwerte (`resolve_expectation`); `expectation` darf dann nur fehlen oder
    exakt identisch sein.
    """
    record, manifest = frozen.record, frozen.manifest
    _stop(guard.phase is Phase.FROZEN, "Reference-Rekonstruktion nur im Zustand FROZEN")
    _stop(
        provenance.purpose is ExecutionPurpose.REFERENCE_RECONSTRUCTION,
        "Reference-Rekonstruktion erfordert eine Freigabe für REFERENCE_RECONSTRUCTION (DEC-087)",
    )
    expectation = resolve_expectation(manifest.freeze_id, expectation)
    _stop(
        expectation.freeze_id == manifest.freeze_id == provenance.freeze_id,
        "Freeze-ID von Erwartung, FreezeManifest und Provenienz stimmt nicht überein",
    )
    _stop(
        provenance.calibration_commit == manifest.git_commit,
        "Provenienz gehört nicht zum Calibration-Commit des Freeze",
    )
    observed = _verify_execution(execution)
    components = _verify_components(frozen, dataset, model, preprocessor, environment_lock)
    g2, g4 = _method_configs(frozen.run_config.scientific, record)
    derived = derive_reference(frozen, split_manifest)
    _stop(
        derived.class_counts == expectation.class_counts,
        "abgeleiteter Klassenvektor weicht vom eingefrorenen Klassenvektor ab",
    )
    _stop(not output_dir.exists(), f"Ausgabeverzeichnis existiert bereits: {output_dir}")

    ids = sorted(derived.sample_ids)
    permit = guard.issue_reference_reconstruction_permit(
        record=record,
        split_manifest=split_manifest,
        sample_ids=derived.sample_ids,
        provenance=provenance,
    )
    try:
        images, labels = _read_once(dataset, permit, ids, execution.batch_size)
    finally:
        guard.close_reference_reconstruction()

    window_id = derived.window_id
    features = _features(window_id, ids, images, model, preprocessor, g2, execution.batch_size)
    representation = RepresentationBatch(
        window_id=window_id, sample_ids=tuple(ids), embeddings=features.embeddings
    )
    outputs = ModelOutputBatch(
        window_id=window_id, sample_ids=tuple(ids), probabilities=features.probabilities
    )
    quality = QualityIndicatorBatch(
        window_id=window_id, sample_ids=tuple(ids), indicators=features.quality
    )
    performance = evaluate_performance(
        window_id, [labels[s] for s in ids], [_argmax(p) for p in outputs.probabilities], g4
    )
    ba_ref = performance.balanced_accuracy
    _stop(
        performance.status is SignalStatus.VALID and ba_ref is not None,
        f"G4-Referenzleistung nicht gültig: {performance.status.value}",
    )
    assert ba_ref is not None
    _stop(
        f"{ba_ref:.4f}" == expectation.ba_ref_4dp,
        f"BA_ref {ba_ref!r} ergibt {ba_ref:.4f}, erwartet {expectation.ba_ref_4dp}",
    )

    reverify_evaluation_provenance(
        provenance, calibration_commit=manifest.git_commit, freeze_id=manifest.freeze_id
    )
    execution_payload: dict[str, Any] = {
        **observed,
        "input_tensor_devices": list(features.input_devices),
        "batches": list(features.batches),
        **components,
        "g2_config": g2.parameters_ref().model_dump(mode="json"),
        "g4_config": g4.parameters_ref().model_dump(mode="json"),
        "freeze_id": manifest.freeze_id,
        "calibration_commit": provenance.calibration_commit,
        "evaluation_commit": provenance.evaluation_commit,
        "approval_sha256": provenance.approval_sha256,
        "ba_ref": ba_ref,
        "ba_ref_4dp": f"{ba_ref:.4f}",
        "ba_ref_expected_4dp": expectation.ba_ref_4dp,
        "ba_ref_source": expectation.ba_ref_source,
        "evidence_limit": EVIDENCE_LIMIT,
        "further_historical_metrics": FURTHER_HISTORICAL_METRICS,
    }
    payloads: dict[str, object] = {
        "reference.selection": derived,
        "reference.g1_representation": representation,
        "reference.g2_quality": quality,
        "reference.g3_model_outputs": outputs,
        "reference.g4_performance": performance,
        "provenance.execution": execution_payload,
    }
    binding = {
        "freeze_id": manifest.freeze_id,
        "calibration_commit": provenance.calibration_commit,
        "evaluation_commit": provenance.evaluation_commit,
        "approval_sha256": provenance.approval_sha256,
        "policy_version": provenance.policy_version,
        "reference_window_id": derived.window_id,
        "reference_sha256": derived.reference_sha256,
    }
    evidence = _write_evidence(output_dir, payloads, binding)
    return ReconstructedReference(
        derived=derived,
        representation=representation,
        outputs=outputs,
        quality=quality,
        performance=performance,
        ba_ref=ba_ref,
        evidence=evidence,
        directory=output_dir,
    )


# --- Wiederverwendung ---------------------------------------------------------------------------


def _read_file(directory: Path, relative: str) -> bytes:
    path = directory / relative
    _stop(not path.is_symlink() and path.is_file(), f"Reference-Artefakt fehlt: {relative}")
    return path.read_bytes()


def load_reconstructed_reference(
    directory: Path, *, freeze_id: str, evaluation_commit: str, expected_manifest_sha256: str
) -> ReconstructedReference:
    """Lädt die write-once Reference-Artefakte und prüft Manifest, Hashes und Bindung vollständig.

    Das Manifest wird genau einmal gelesen; der SHA-256 genau dieses Byte-Puffers muss vor jedem
    Parsing dem erwarteten (freigegebenen) Hash entsprechen (TOCTOU, P4-03AO). Jede fehlende,
    zusätzliche oder veränderte Datei sowie eine andere Freeze-ID oder ein anderer
    Evaluation-Commit ist ein STOP.
    """
    manifest_bytes = _read_file(directory, EVIDENCE_MANIFEST_FILE)
    _stop(
        sha256_hex(manifest_bytes) == expected_manifest_sha256,
        "Reference-Evidence-Manifest weicht vom freigegebenen Hash ab",
    )
    evidence = ReferenceEvidenceManifest.model_validate(json.loads(manifest_bytes))
    _stop(canonical_json_bytes(evidence) == manifest_bytes, "Reference-Manifest nicht kanonisch")
    _stop(
        (evidence.freeze_id, evidence.evaluation_commit) == (freeze_id, evaluation_commit),
        "Reference-Artefakte gehören zu einem anderen Freeze oder Evaluation-Commit",
    )
    _stop(
        {e.role: e.path for e in evidence.entries} == _FILES,
        "Reference-Manifest enthält nicht exakt die vorgesehenen Artefakte",
    )
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if not p.is_dir()}
    expected = {*_FILES.values(), EVIDENCE_MANIFEST_FILE}
    _stop(present == expected, f"unerwartete oder fehlende Dateien: {sorted(present ^ expected)}")
    values: dict[str, Any] = {}
    for entry in evidence.entries:
        data = _read_file(directory, entry.path)
        _stop(sha256_hex(data) == entry.sha256, f"{entry.role}: Inhalt verändert")
        values[entry.role] = json.loads(data)
    derived = DerivedReference.model_validate(values["reference.selection"])
    representation = RepresentationBatch.model_validate(values["reference.g1_representation"])
    quality = QualityIndicatorBatch.model_validate(values["reference.g2_quality"])
    outputs = ModelOutputBatch.model_validate(values["reference.g3_model_outputs"])
    performance = PerformanceEvaluation.model_validate(values["reference.g4_performance"])
    ba_ref = values["provenance.execution"]["ba_ref"]
    _stop(
        derived.window_id == evidence.reference_window_id
        and derived.reference_sha256 == evidence.reference_sha256
        and {representation.window_id, quality.window_id, outputs.window_id, performance.window_id}
        == {derived.window_id}
        and representation.sample_ids == quality.sample_ids == outputs.sample_ids
        and representation.sample_ids == tuple(sorted(derived.sample_ids))
        and performance.balanced_accuracy == ba_ref,
        "Reference-Artefakte sind untereinander inkonsistent",
    )
    return ReconstructedReference(
        derived=derived,
        representation=representation,
        outputs=outputs,
        quality=quality,
        performance=performance,
        ba_ref=ba_ref,
        evidence=evidence,
        directory=directory,
    )

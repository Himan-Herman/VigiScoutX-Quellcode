"""HELD_OUT-Evaluation gemäß Protokoll v1.1; jeder STOP ist terminal (→ manuelle Entscheidung).

Ablauf:

0. Zweckgebundene Freigabe (`DEC-087`): ausschließlich eine Freigabe `HELD_OUT_EVALUATION`
   (Schema 2) für denselben Evaluation-Commit wie die Reference-Phase, die den SHA-256 des
   geprüften Reference-Evidence-Manifests bindet; sonst STOP vor jedem Zugriff (fail-closed).
1. Bindungen vor dem Öffnen: Freeze, Provenienz, Ausführung (CPU, Batch 32), Modell, Datensatz,
   Preprocessing, Environment-Lock, eingefrorene Methoden-/Fusions-/Relevanzwerte und die
   hashgeprüfte Wiederverwendung der Reference (`load_reconstructed_reference`); keine zweite
   Rekonstruktion und keine neue Reference-Auswahl.
2. `HeldOutGuard.open_held_out()` (bestehendes Release-Gate) und HELD_OUT-Berechtigung.
3. Structural Gate und vollständiger Case-Plan; der Plan wird vor der ersten Inferenz write-once
   fixiert. Strukturlabels nur des kanonischen Bildes je Läsion.
4. Phase A (label-frei): G1–G3, Gruppen, `FUSION_A`, Primary-Risiko und -Empfehlung für jeden
   Case; danach unveränderlich fixiert. `FUSION_B` wird nicht berechnet.
5. Phase B: G4 getrennt (nicht auf Clean-FAR-Windows), label-aware Analyse, sekundäre
   Empfehlung, `MonitoringRunResult` je Window; Kennzahlen.
6. Write-once-Artefakte (Staging, `ArtifactManifest` zuletzt, atomare Umbenennung). Ein STOP
   hinterlässt nur einen Terminationsnachweis, kein als gültig gekennzeichnetes Ergebnis.
"""

import json
import os
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from calibration_runner.evaluation import (
    case_seed,
    model_outputs,
    performance,
    performance_status,
    quality_batch,
    representation,
)
from calibration_runner.features import SampleFeatures
from held_out_runner.features import HeldOutFeatures
from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.artifacts.manifests import ArtifactEntry, ArtifactManifest
from vigiscoutx.config.scientific_config import FusionRule, SignalConfig, scientific_config_sha256
from vigiscoutx.domain.assessments import (
    GroupAssessment,
    LabelAwareAssessment,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    Recommendation,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import ModelSnapshotRef, SeedBundle
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.evaluation.held_out_metrics import (
    ALL_METHODS,
    HeldOutMetrics,
    ScoredCase,
    Thresholds,
    held_out_metrics,
)
from vigiscoutx.evaluation.held_out_sampling import (
    CLASS_FAMILY,
    FAMILIES,
    LEVELS,
    EvaluationNamespace,
    HeldOutCase,
    HeldOutCasePlan,
    HeldOutEvaluationStop,
    build_case_plan,
    held_out_canonical_samples,
    held_out_samples_by_lesion,
    verify_case_plan,
    verify_held_out_case,
)
from vigiscoutx.evaluation.reference_reconstruction import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_DEVICE,
    ReconstructedReference,
    ReferenceDataPort,
    ReferenceExecution,
    ReferenceModelPort,
    load_reconstructed_reference,
)
from vigiscoutx.evaluation.reference_sources import FrozenCalibration
from vigiscoutx.experiment.calibration.config import (
    FusionCandidate,
    TechnicalShift,
    baseline_protocol_config,
)
from vigiscoutx.experiment.calibration.freeze_record import verify_record_against_manifest
from vigiscoutx.experiment.calibration.statistics import fusion_alarm
from vigiscoutx.experiment.calibration.structure import ClassVector
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    ExecutionPurpose,
    reverify_evaluation_provenance,
)
from vigiscoutx.experiment.freeze import verify_freeze_manifest
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256
from vigiscoutx.experiment.types import Partition, Phase
from vigiscoutx.grouping.aggregation import assess_group
from vigiscoutx.monitoring import MonitoringRunResult
from vigiscoutx.recommendations.engine import primary_recommendation, secondary_recommendation
from vigiscoutx.recommendations.policy import (
    RecommendationPolicyDefinition,
    resolve_recommendation_policy,
)
from vigiscoutx.risk.label_aware import assess_label_aware
from vigiscoutx.risk.mapper import assess_primary
from vigiscoutx.risk.policy import RiskPolicyDefinition, resolve_risk_policy
from vigiscoutx.signals.config import (
    G1MmdConfig,
    G2QualityConfig,
    G3PredictiveConfig,
    method_config_from_signal_config,
)
from vigiscoutx.signals.data_quality.quality import run_g2_quality
from vigiscoutx.signals.input.mmd import run_g1_mmd
from vigiscoutx.signals.performance.g4 import G4PerformanceConfig, assess_performance
from vigiscoutx.signals.predictive.g3 import run_g3

_SIGNALS: Final = {
    "G1": ("MMD_U2_RBF_PERMUTATION", SignalGroup.G1_INPUT),
    "G2": ("G2_QUALITY_KS_BONFERRONI", SignalGroup.G2_DATA_QUALITY),
    "G3": ("G3_MUKS_CONFIDENCE_KS", SignalGroup.G3_PREDICTIVE),
}
_FILES: Final = {
    "evaluation.case_plan": "plan/case_plan.json",
    "evaluation.label_free": "results/label_free.json",
    "evaluation.g4": "results/g4.json",
    "evaluation.monitoring_runs": "results/monitoring_runs.json",
    "evaluation.metrics": "results/metrics.json",
    "provenance.execution": "provenance/execution.json",
}
ARTIFACT_MANIFEST_FILE: Final = "artifact_manifest.json"


def _stop(condition: bool, message: str) -> None:
    if not condition:
        raise HeldOutEvaluationStop(message)


@dataclass(frozen=True, slots=True)
class MethodConfigs:
    g1: G1MmdConfig
    g2: G2QualityConfig
    g3: G3PredictiveConfig
    g4: G4PerformanceConfig


@dataclass(frozen=True, slots=True)
class LabelFree:
    """Unveränderliches label-freies Ergebnis genau eines Cases (Phase A)."""

    case: HeldOutCase
    seeds: SeedBundle
    g1: SignalResult
    g1_group: GroupAssessment
    g2_components: tuple[SignalResult, ...]
    g2_group: GroupAssessment
    g3_output: SignalResult
    g3_confidence: SignalResult
    g3_group: GroupAssessment
    fusion_a: bool
    primary: PrimaryRiskAssessment
    primary_recommendation: Recommendation
    probabilities: tuple[tuple[float, ...], ...]

    def alarms(self) -> dict[str, bool]:
        signals = [self.g1, *self.g2_components, self.g3_output, self.g3_confidence]
        alarms = {s.signal_id: s.alarm for s in signals}
        alarms |= {"G2_GROUP": self.g2_group.alarm, "G3_GROUP": self.g3_group.alarm}
        alarms["FUSION_A"] = self.fusion_a
        _stop(set(alarms) == set(ALL_METHODS), f"Verfahrensmenge weicht ab: {sorted(alarms)}")
        _stop(all(a is not None for a in alarms.values()), "Alarm eines Pflichtsignals fehlt")
        return {k: bool(v) for k, v in alarms.items()}

    def summary(self) -> dict[str, object]:
        return {
            "case_id": self.case.case_id,
            "alarms": dict(sorted(self.alarms().items())),
            "g1_p_value": self.g1.p_value,
            "g1_seed": self.seeds.statistical_seed,
            "primary_state": self.primary.state.value,
            "flagged_groups": [g.value for g in self.primary.flagged_groups],
            "primary_recommendation": self.primary_recommendation.code,
        }


@dataclass(frozen=True, slots=True)
class HeldOutRunResult:
    output_dir: Path
    plan: HeldOutCasePlan
    metrics: HeldOutMetrics
    manifest: ArtifactManifest
    n_monitoring_runs: int


def _mandatory_statuses(
    g1: SignalResult,
    g2_components: Sequence[SignalResult],
    g3_output: SignalResult,
    g3_confidence: SignalResult,
    groups: Sequence[GroupAssessment],
) -> list[SignalStatus]:
    """Wie im Calibration-Runner: G1, alle G2-Komponenten, die sieben MUKS-Komponenten, G3 und
    alle drei Gruppen."""
    return [
        g1.status,
        *(c.status for c in g2_components),
        *(c.status for c in g3_output.components),
        g3_output.status,
        g3_confidence.status,
        *(g.status for g in groups),
    ]


def require_valid_signals(statuses: Sequence[SignalStatus], label: str) -> None:
    """STOP bei `ERROR`/`NOT_EVALUABLE` eines verpflichtenden G1–G3-Ergebnisses in jedem Clean-,
    L0- oder Shift-Case (Protokoll §9); kein Entfernen von Cases oder Methoden."""
    invalid = sorted({s.value for s in statuses if s is not SignalStatus.VALID})
    _stop(not invalid, f"{label}: verpflichtendes G1–G3-Signal nicht VALID ({invalid})")


def require_valid_g4(assessment: PerformanceAssessment, label: str) -> None:
    """STOP, wenn G4 eines L0-/Shift-Cases nicht vollständig gültig und evaluierbar ist."""
    _stop(
        performance_status(assessment) is SignalStatus.VALID
        and assessment.delta_ba is not None
        and assessment.relevant_degradation is not None,
        f"{label}: G4 nicht vollständig gültig",
    )


class HeldOutRunner:
    """Genau ein HELD_OUT-Lauf je Guard; jeder STOP und jeder Fehler ist terminal."""

    def __init__(
        self,
        *,
        guard: HeldOutGuard,
        frozen: FrozenCalibration,
        split_manifest: SplitManifest,
        dataset: ReferenceDataPort,
        model: ReferenceModelPort,
        preprocessor: ImagePreprocessor,
        environment_lock: Path,
        provenance: EvaluationProvenance,
        config: RunConfig,
        observed: ObservedHashes,
        reference_dir: Path,
        execution: ReferenceExecution,
        output_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._guard = guard
        self._frozen = frozen
        self._split = split_manifest
        self._dataset = dataset
        self._model = model
        self._preprocessor = preprocessor
        self._lock = environment_lock
        self._provenance = provenance
        self._config = config
        self._observed = observed
        self._reference_dir = reference_dir
        self._execution = execution
        self._output = output_dir
        self._progress = progress or (lambda _message: None)
        self._phase = "BINDINGS"
        self._labels: dict[str, int] = {}

    # --- Ausführung -----------------------------------------------------------------------------

    def run(self) -> HeldOutRunResult:
        staging = self._output.with_name(f".{self._output.name}.incomplete")
        try:
            return self._run(staging)
        except BaseException as exc:
            shutil.rmtree(staging, ignore_errors=True)
            self._terminate(exc)
            raise

    def _terminate(self, exc: BaseException) -> None:
        """Unveränderter Terminationsnachweis ohne Ergebnisse (kein Überschreiben)."""
        target = self._output.with_name(f"{self._output.name}.STOPPED.json")
        if target.exists() or not target.parent.is_dir():
            return
        with target.open("xb") as handle:
            handle.write(
                canonical_json_bytes(
                    {
                        "status": "STOPPED",
                        "phase": self._phase,
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "freeze_id": self._frozen.manifest.freeze_id,
                        "valid_result": False,
                    }
                )
            )

    def _run(self, staging: Path) -> HeldOutRunResult:
        record = self._frozen.record
        bindings = self._verify_bindings()
        configs = self._method_configs()
        reference = self._reference()
        _stop(not self._output.exists() and not staging.exists(), "Ausgabeverzeichnis existiert")

        self._phase = "OPEN_HELD_OUT"
        self._guard.open_held_out(self._config, self._observed, self._provenance)
        _stop(self._guard.phase is Phase.HELD_OUT_EVALUATION, "HELD_OUT nicht geöffnet")
        permit = self._guard.issue_permit(Partition.HELD_OUT, self._split)

        self._phase = "STRUCTURAL_GATE"
        plan = self._plan(permit, reference)
        plan_sha256 = plan.plan_sha256()
        staging.mkdir(exist_ok=False)
        entries = [self._write(staging, "evaluation.case_plan", plan)]  # vor der ersten Inferenz
        self._progress(f"Case-Plan fixiert: {len(plan.cases)} Cases, {plan_sha256[:12]}")

        self._phase = "LABEL_FREE"
        features = HeldOutFeatures(
            dataset=self._dataset,
            permit=permit,
            model=self._model,
            preprocessor=self._preprocessor,
            g2_config=configs.g2,
            protocol=baseline_protocol_config(),
            batch_size=self._execution.batch_size,
        )
        verify_case_plan(plan, plan_sha256)
        label_free = self._label_free(plan, features, reference, configs)
        entries.append(
            self._write(staging, "evaluation.label_free", [r.summary() for r in label_free])
        )
        label_free_sha256 = entries[-1].sha256
        self._progress("Label-freie Ergebnisse fixiert")

        self._phase = "G4"
        labels = self._labels
        performances = {
            r.case.case_id: self._g4(r, labels, reference, configs)
            for r in label_free
            if r.case.namespace is EvaluationNamespace.HELD_OUT_SCENARIO
        }
        runs = [self._monitoring_run(r, performances.get(r.case.case_id)) for r in label_free]
        thresholds = Thresholds(delta_rel=record.delta_rel, q090=record.q090, q099=record.q099)
        scored = [
            ScoredCase(
                case_id=r.case.case_id,
                clean=r.case.namespace is EvaluationNamespace.HELD_OUT_CLEAN,
                family=r.case.family,
                level=r.case.level,
                repeat_id=r.case.repeat_id,
                alarms=tuple(sorted(r.alarms().items())),
                delta_ba=(
                    None
                    if r.case.case_id not in performances
                    else performances[r.case.case_id].delta_ba
                ),
            )
            for r in label_free
        ]
        metrics = held_out_metrics(scored, thresholds, families=FAMILIES, repeats=plan.repeats)

        self._phase = "ARTIFACTS"
        entries.append(
            self._write(
                staging,
                "evaluation.g4",
                [self._g4_summary(case_id, p, thresholds) for case_id, p in performances.items()],
            )
        )
        entries.append(
            self._write(
                staging, "evaluation.monitoring_runs", [r.model_dump(mode="json") for r in runs]
            )
        )
        entries.append(self._write(staging, "evaluation.metrics", metrics))
        execution = {
            **bindings,
            "plan_sha256": plan_sha256,
            "label_free_sha256": label_free_sha256,
            "reference_evidence_sha256": self._provenance.reference_evidence_sha256,
            "reference_artifacts_sha256": reference.evidence.artifacts_sha256(),
            "reference_approval_sha256": reference.evidence.approval_sha256,
            "ba_ref": reference.ba_ref,
            "method_refs": {
                "G1": configs.g1.parameters_ref().model_dump(mode="json"),
                "G2": configs.g2.parameters_ref().model_dump(mode="json"),
                "G3": configs.g3.parameters_ref().model_dump(mode="json"),
                "G4": configs.g4.parameters_ref().model_dump(mode="json"),
            },
            "thresholds": thresholds.model_dump(mode="json"),
            "fusion": FusionCandidate.FUSION_A.value,
            "fusion_b_computed": False,
            "input_tensor_devices": sorted(features.input_devices),
            "n_model_batches": len(features.batches),
            "max_model_batch": max(features.batches),
            "approval_purpose": self._provenance.purpose.value,
            "reference_evidence_sha256_bound": self._provenance.reference_evidence_sha256,
        }
        entries.append(self._write(staging, "provenance.execution", execution))
        artifact_manifest = ArtifactManifest(run_id=self._config.run_id, entries=tuple(entries))
        target = staging / ARTIFACT_MANIFEST_FILE
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(artifact_manifest))
        target.chmod(0o444)
        _stop(not self._output.exists(), "Ausgabeverzeichnis existiert")
        os.rename(staging, self._output)
        self._phase = "COMPLETED"
        return HeldOutRunResult(
            output_dir=self._output,
            plan=plan,
            metrics=metrics,
            manifest=artifact_manifest,
            n_monitoring_runs=len(runs),
        )

    # --- Bindungen ------------------------------------------------------------------------------

    def _verify_bindings(self) -> dict[str, object]:
        manifest, record = self._frozen.manifest, self._frozen.record
        # Vor jeder wissenschaftlichen Verwendung des Records und vor jedem Datenzugriff: die
        # bestehenden autoritativen Verifier (P4-03AK); jede Abweichung ist ein terminaler STOP.
        verify_freeze_manifest(manifest)
        verify_record_against_manifest(record, manifest)
        provenance, config, execution = self._provenance, self._config, self._execution
        scientific = config.scientific
        # Fail-closed Freigabeprüfung (DEC-087) vor jedem Zugriff: nur HELD_OUT_EVALUATION mit
        # gebundenem Reference-Evidence-Manifest; frische Neuprüfung von Repository und Freigabe.
        _stop(
            provenance.purpose is ExecutionPurpose.HELD_OUT_EVALUATION
            and provenance.reference_evidence_sha256 is not None,
            "HELD_OUT erfordert eine Freigabe für HELD_OUT_EVALUATION mit gebundenem "
            "Reference-Evidence-Manifest",
        )
        reverify_evaluation_provenance(
            provenance, calibration_commit=manifest.git_commit, freeze_id=manifest.freeze_id
        )
        _stop(self._guard.phase is Phase.FROZEN, "HELD_OUT-Evaluation nur aus FROZEN")
        _stop(
            (provenance.freeze_id, provenance.calibration_commit)
            == (manifest.freeze_id, manifest.git_commit),
            "Provenienz gehört nicht zu diesem Freeze",
        )
        _stop(
            execution.device == REFERENCE_DEVICE and execution.batch_size == REFERENCE_BATCH_SIZE,
            "Ausführung abweichend von CPU / batch_size 32",
        )
        default_device = torch.get_default_device().type
        _stop(default_device == REFERENCE_DEVICE, f"Standardgerät {default_device!r} ist nicht CPU")
        _stop(split_manifest_sha256(self._split) == record.split_manifest_sha256, "Split weicht ab")
        model_ref = self._model.snapshot_ref()
        dataset_ref = self._dataset.snapshot_ref()
        _stop(self._lock.is_file(), "Environment-Lock fehlt")
        lock_sha256 = sha256_file(self._lock)
        checks: dict[str, tuple[object, object]] = {
            "config_sha256": (scientific_config_sha256(scientific), manifest.config_sha256),
            "model_id": (model_ref.model_id, record.model_id),
            "model_snapshot_sha256": (model_ref.sha256, record.model_snapshot_sha256),
            "model_weights_sha256": (scientific.model.sha256, record.model_weights_sha256),
            "preprocessing_sha256": (
                canonical_sha256(scientific.preprocessing),
                record.preprocessing_sha256,
            ),
            "preprocessing_spec": (
                self._preprocessor.spec,
                ImagePreprocessingSpec.from_config(scientific.preprocessing),
            ),
            "environment_lock_sha256": (lock_sha256, record.environment_lock_sha256),
            "dataset_snapshot": (
                (dataset_ref.snapshot_id, dataset_ref.sha256),
                (record.dataset_snapshot_id, record.dataset_snapshot_sha256),
            ),
            "window_size": (scientific.monitoring.window_size, record.window_size),
            "delta_rel": (scientific.relevance.delta_rel, record.delta_rel),
            "fusion": (
                (scientific.fusion.rule, record.fusion_candidate),
                (FusionRule.A_ANY_PRIMARY_GROUP, FusionCandidate.FUSION_A),
            ),
            "repetitions": (manifest.repetitions, record.final_r),
        }
        mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
        _stop(not mismatches, f"Bindung weicht vom Freeze ab: {mismatches}")
        return {
            "freeze_id": manifest.freeze_id,
            "calibration_commit": provenance.calibration_commit,
            "evaluation_commit": provenance.evaluation_commit,
            "approval_sha256": provenance.approval_sha256,
            "policy_version": provenance.policy_version,
            "config_sha256": manifest.config_sha256,
            "model_id": model_ref.model_id,
            "model_snapshot_sha256": model_ref.sha256,
            "model_weights_sha256": record.model_weights_sha256,
            "dataset_snapshot_sha256": dataset_ref.sha256,
            "split_manifest_sha256": record.split_manifest_sha256,
            "preprocessing_sha256": record.preprocessing_sha256,
            "environment_lock_sha256": lock_sha256,
            "risk_policy": scientific.risk_policy.model_dump(mode="json"),
            "recommendation_policy": scientific.recommendation_policy.model_dump(mode="json"),
            "requested_device": execution.device,
            "observed_default_device": default_device,
            "batch_size": execution.batch_size,
            "torch_version": torch.__version__,
        }

    def _signal(self, key: str) -> SignalConfig:
        method, group = _SIGNALS[key]
        found = [s for s in self._config.scientific.monitoring.signals if s.method == method]
        _stop(len(found) == 1 and found[0].group is group, f"genau ein Signal {key} erforderlich")
        return found[0]

    def _method_configs(self) -> MethodConfigs:
        record = self._frozen.record
        g1 = method_config_from_signal_config(self._signal("G1"), G1MmdConfig)
        g2 = method_config_from_signal_config(self._signal("G2"), G2QualityConfig)
        g3 = method_config_from_signal_config(self._signal("G3"), G3PredictiveConfig)
        g4 = G4PerformanceConfig(
            performance_id="G4_PERFORMANCE",
            method_version="monitoring-methods-baseline:1.2",
            class_order=DEMONSTRATOR_CLASS_ORDER,
            metric="BALANCED_ACCURACY",
            missing_class_rule="NOT_EVALUABLE_NO_REAVERAGING",
            zero_division="NULL",
        )
        _stop(g1.sigma == record.mmd_sigma, "G1-σ weicht vom Freeze ab")
        missing = [
            c.parameters_ref().config_id
            for c in (g2, g3, g4)
            if c.parameters_ref() not in record.method_refs
        ]
        _stop(not missing, f"Methodenkonfiguration nicht im Freeze gebunden: {missing}")
        return MethodConfigs(g1=g1, g2=g2, g3=g3, g4=g4)

    def _reference(self) -> ReconstructedReference:
        """Ausschließlich die write-once rekonstruierte Reference; keine zweite Rekonstruktion."""
        manifest, record = self._frozen.manifest, self._frozen.record
        # Ausschließlich der in der geprüften Provenienz gebundene Hash (DEC-087); er wird nie aus
        # dem aktuellen Dateizustand abgeleitet. Der Loader prüft genau die gelesenen Bytes.
        bound = self._provenance.reference_evidence_sha256
        _stop(bound is not None, "HELD_OUT-Freigabe bindet kein Reference-Evidence-Manifest")
        assert bound is not None
        reference = load_reconstructed_reference(
            self._reference_dir,
            freeze_id=manifest.freeze_id,
            evaluation_commit=self._provenance.evaluation_commit,
            expected_manifest_sha256=bound,
        )
        _stop(
            reference.evidence.approval_sha256 != self._provenance.approval_sha256,
            "Reference- und HELD_OUT-Phase benötigen getrennte Freigabenachweise",
        )
        checks = {
            "calibration_commit": (reference.evidence.calibration_commit, manifest.git_commit),
            "reference_window_id": (reference.window_id, record.reference_window_id),
            "reference_sha256": (reference.derived.reference_sha256, record.reference_sha256),
            "sample_ids": (reference.sample_ids, tuple(sorted(record.reference_sample_ids))),
            "lesion_ids": (reference.derived.lesion_ids, record.reference_lesion_ids),
        }
        mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
        _stop(not mismatches, f"Reference-Artefakte passen nicht zum Freeze: {mismatches}")
        _stop(
            reference.performance.status is SignalStatus.VALID
            and reference.performance.balanced_accuracy == reference.ba_ref,
            "Reference-G4 ungültig",
        )
        return reference

    # --- Struktur -------------------------------------------------------------------------------

    def _plan(self, permit: Any, reference: ReconstructedReference) -> HeldOutCasePlan:
        record, manifest = self._frozen.record, self._frozen.manifest
        canonical = held_out_canonical_samples(self._split)
        samples = sorted(canonical.values())
        labels = dict(self._dataset.labels_for(permit, samples))
        _stop(set(labels) == set(samples), "Datensatz lieferte nicht alle Strukturlabels")
        order = baseline_protocol_config().class_order
        _stop(all(0 <= v < len(order) for v in labels.values()), "ungültiges Label")
        self._labels = labels
        lesion_classes = {lesion: order[labels[s]] for lesion, s in canonical.items()}
        _stop(
            set(lesion_classes) == set(held_out_samples_by_lesion(self._split)),
            "Strukturlabels unvollständig",
        )
        vector = ClassVector(
            window_size=record.window_size,
            counts=tuple(n for _, n in reference.derived.class_counts),
            class_order=tuple(c for c, _ in reference.derived.class_counts),
        )
        return build_case_plan(
            record=record,
            freeze_id=manifest.freeze_id,
            split=self._split,
            vector=vector,
            lesion_classes=lesion_classes,
            reference_lesion_ids=reference.derived.lesion_ids,
            reference_sample_ids=reference.derived.sample_ids,
            reference_window_id=reference.window_id,
        )

    # --- Phase A: label-frei --------------------------------------------------------------------

    def _blocks(
        self, plan: HeldOutCasePlan, features: HeldOutFeatures
    ) -> Iterator[list[tuple[HeldOutCase, list[SampleFeatures]]]]:
        """Merkmale blockweise in Plan-Reihenfolge: je Clean-Window einzeln, je Familie/Repeat
        L0–L3 gemeinsam (technische Familien: ein Bildzugriff je Sample für alle Levels)."""
        blocks: dict[tuple[str, int], list[HeldOutCase]] = {}
        for case in plan.cases:
            if case.namespace is EvaluationNamespace.HELD_OUT_CLEAN:
                yield [(case, features.clean(case.sample_ids))]
            else:
                _stop(case.family is not None, f"{case.case_id}: Familie fehlt")
                assert case.family is not None
                blocks.setdefault((case.family, case.repeat_id), []).append(case)
        for (family, repeat_id), cases in blocks.items():
            ordered = sorted(cases, key=lambda c: c.level or 0)
            _stop(
                [c.level for c in ordered] == list(LEVELS),
                f"{family}/{repeat_id}: Levels unvollständig",
            )
            if family == CLASS_FAMILY:
                yield [(c, features.clean(c.sample_ids)) for c in ordered]
                continue
            base, shifted = ordered[0], ordered[1:]
            values = [c.shift_value for c in shifted]
            _stop(all(v is not None for v in values), f"{family}/{repeat_id}: Severity fehlt")
            seeds = shifted[0].noise_seeds
            by_value = features.shifted(
                base.sample_ids,
                TechnicalShift(family),
                [float(v) for v in values if v is not None],
                noise_seeds=(
                    None if seeds is None else dict(zip(base.sample_ids, seeds, strict=True))
                ),
            )
            yield [
                (base, features.clean(base.sample_ids)),
                *((c, by_value[float(c.shift_value or 0.0)]) for c in shifted),
            ]

    def _label_free(
        self,
        plan: HeldOutCasePlan,
        features: HeldOutFeatures,
        reference: ReconstructedReference,
        configs: MethodConfigs,
    ) -> list[LabelFree]:
        scientific = self._config.scientific
        risk = resolve_risk_policy(scientific.risk_policy)
        recommendations = resolve_recommendation_policy(scientific.recommendation_policy)
        model_ref = self._model.snapshot_ref()
        results = [
            self._evaluate(case, rows, reference, configs, model_ref, risk, recommendations)
            for block in self._blocks(plan, features)
            for case, rows in block
        ]
        _stop(
            [r.case.case_id for r in results] == [c.case_id for c in plan.cases],
            "label-freie Ergebnisse nicht exakt für alle Cases des Plans",
        )
        return results

    def _evaluate(
        self,
        case: HeldOutCase,
        rows: Sequence[SampleFeatures],
        reference: ReconstructedReference,
        configs: MethodConfigs,
        model_ref: ModelSnapshotRef,
        risk: RiskPolicyDefinition,
        recommendations: RecommendationPolicyDefinition,
    ) -> LabelFree:
        """G1–G3 genau einmal je Case mit dem Case-Seed; jedes nicht valide Pflichtsignal → STOP."""
        verify_held_out_case(case, self._frozen.record.split_manifest_sha256, len(rows))
        ids = case.sample_ids
        seeds = case_seed(self._config.seeds, case.g1_seed, case.repeat_id)
        g1 = run_g1_mmd(
            reference.representation,
            representation(case.window_id, ids, rows),
            configs.g1,
            seeds,
            model_ref,
        )
        seed_ref = g1.seed_ref
        _stop(
            seed_ref is not None
            and (seed_ref.seed_field, seed_ref.seed, seed_ref.repeat_id)
            == (configs.g1.seed_field, seeds.statistical_seed, seeds.repeat_id),
            f"{case.case_id}: G1-SeedRef weicht vom Case-Seed ab",
        )
        g2_components, g2_group = run_g2_quality(
            reference.quality, quality_batch(case.window_id, ids, rows), configs.g2
        )
        g3_output, g3_confidence, g3_group = run_g3(
            reference.outputs, model_outputs(case.window_id, ids, rows), configs.g3, model_ref
        )
        g1_group = assess_group(SignalGroup.G1_INPUT, (g1,))
        groups = (g1_group, g2_group, g3_group)
        require_valid_signals(
            _mandatory_statuses(g1, g2_components, g3_output, g3_confidence, groups),
            f"{case.scenario_key}/{case.repeat_id}",
        )
        primary = assess_primary(case.window_id, groups, risk)
        _stop(isinstance(primary, PrimaryRiskAssessment), f"{case.case_id}: Primary ungültig")
        assert isinstance(primary, PrimaryRiskAssessment)
        return LabelFree(
            case=case,
            seeds=seeds,
            g1=g1,
            g1_group=g1_group,
            g2_components=g2_components,
            g2_group=g2_group,
            g3_output=g3_output,
            g3_confidence=g3_confidence,
            g3_group=g3_group,
            fusion_a=fusion_alarm(FusionCandidate.FUSION_A, groups),
            primary=primary,
            primary_recommendation=primary_recommendation(primary, recommendations),
            probabilities=tuple(r.probabilities for r in rows if r.probabilities is not None),
        )

    # --- Phase B: G4, Risiko, Ergebnisse --------------------------------------------------------

    def _g4(
        self,
        result: LabelFree,
        labels: Mapping[str, int],
        reference: ReconstructedReference,
        configs: MethodConfigs,
    ) -> PerformanceAssessment:
        case = result.case
        ids = case.sample_ids
        _stop(len(result.probabilities) == len(ids), f"{case.case_id}: Softmax fehlt")
        rows = [
            SampleFeatures(embedding=None, probabilities=p, quality=None)
            for p in result.probabilities
        ]
        monitoring = performance(case.window_id, ids, rows, [labels[s] for s in ids], configs.g4)
        assessment = assess_performance(
            reference.performance,
            monitoring,
            configs.g4,
            self._model.snapshot_ref(),
            self._frozen.record.delta_rel,
        )
        require_valid_g4(assessment, f"{case.scenario_key}/{case.repeat_id}")
        return assessment

    def _monitoring_run(
        self, result: LabelFree, assessment: PerformanceAssessment | None
    ) -> MonitoringRunResult:
        scientific = self._config.scientific
        risk = resolve_risk_policy(scientific.risk_policy)
        recommendations = resolve_recommendation_policy(scientific.recommendation_policy)
        secondary = (
            None
            if assessment is None
            else assess_label_aware(result.primary, assessment, scientific.relevance, risk)
        )
        _stop(
            secondary is None or isinstance(secondary, LabelAwareAssessment),
            f"{result.case.case_id}: label-aware Analyse ungültig",
        )
        secondary_rec = secondary_recommendation(secondary, recommendations)
        return MonitoringRunResult.model_validate(
            {
                "run_id": self._config.run_id,
                "config_sha256": self._frozen.manifest.config_sha256,
                "freeze_id": self._frozen.manifest.freeze_id,
                "risk_policy": scientific.risk_policy,
                "recommendation_policy": scientific.recommendation_policy,
                "relevance": scientific.relevance,
                "signal_results": (
                    result.g1,
                    *result.g2_components,
                    result.g3_output,
                    result.g3_confidence,
                ),
                "group_assessments": (result.g1_group, result.g2_group, result.g3_group),
                "primary_assessments": (result.primary,),
                "primary_failures": (),
                "label_aware_assessments": () if secondary is None else (secondary,),
                "label_aware_failures": (),
                "recommendations": (
                    result.primary_recommendation,
                    *(() if secondary_rec is None else (secondary_rec,)),
                ),
                "artifact_manifest_sha256": None,
            }
        )

    @staticmethod
    def _g4_summary(
        case_id: str, assessment: PerformanceAssessment, thresholds: Thresholds
    ) -> dict[str, object]:
        delta = assessment.delta_ba
        assert delta is not None
        return {
            "case_id": case_id,
            "ba_monitoring": assessment.monitoring.balanced_accuracy,
            "delta_ba": delta,
            "relevant_delta_rel": assessment.relevant_degradation,
            "relevant_q090": delta >= thresholds.q090,
            "relevant_q099": delta >= thresholds.q099,
        }

    # --- Artefakte ------------------------------------------------------------------------------

    @staticmethod
    def _write(staging: Path, role: str, value: object) -> ArtifactEntry:
        relative = _FILES[role]
        data = canonical_json_bytes(value)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        target.chmod(0o444)
        return ArtifactEntry(path=relative, sha256=sha256_hex(data), role=role)


def load_held_out_results(directory: Path) -> tuple[ArtifactManifest, dict[str, bytes]]:
    """Prüft das ArtifactManifest eines abgeschlossenen Laufs vollständig (Hashes, Dateimenge)."""
    data = (directory / ARTIFACT_MANIFEST_FILE).read_bytes()
    manifest = ArtifactManifest.model_validate(json.loads(data))
    _stop(canonical_json_bytes(manifest) == data, "ArtifactManifest nicht kanonisch")
    _stop({e.role: e.path for e in manifest.entries} == _FILES, "Artefaktmenge weicht ab")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    expected = {*_FILES.values(), ARTIFACT_MANIFEST_FILE}
    _stop(present == expected, f"unerwartete oder fehlende Dateien: {sorted(present ^ expected)}")
    contents: dict[str, bytes] = {}
    for entry in manifest.entries:
        content = (directory / entry.path).read_bytes()
        _stop(sha256_hex(content) == entry.sha256, f"{entry.role}: Inhalt verändert")
        contents[entry.role] = content
    return manifest, contents

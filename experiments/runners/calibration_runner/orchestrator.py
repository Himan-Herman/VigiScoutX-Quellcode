"""End-to-End-Orchestrierung der 28 eingefrorenen Calibration-Schritte (P4-01, P4-02D).

Der `CalibrationProcess` bleibt autoritativ: Er bestimmt Reihenfolge, Pools, Canonical Samples,
Reference, `δ_rel`, Severities, Kompositionen, `R`, Fusion, Audit-FAR und Freeze. Der Runner
erzeugt ausschließlich die dafür nötigen Messwerte mit den bestehenden Komponenten und übergibt
sie an die jeweiligen Prozessschritte.

STOP-/Fehlersemantik (Orchestrierungsbaseline Abschnitte 5, 6 und 10):

- Jeder `CalibrationStopError` ist terminal (STOP → manuelle Entscheidung), jede Protokoll- oder
  Integritätsverletzung ebenfalls (`CalibrationProcess.abort`). Es gibt keinen Ersatzdraw, keine
  Parameteränderung und keinen automatischen Wiederanlauf; insbesondere keine zweite
  AUDIT-Ziehung.
- `W_VALIDITY`: `NOT_EVALUABLE` → Kandidat nicht `ELIGIBLE`; `ERROR` → terminaler STOP.
- Severity: Kandidat gültig ⇔ G4 in allen 100 Base-Windows `VALID`; `ERROR` → STOP.
- R-/Fusion-/Audit-Cases: jedes nicht-valide Pflichtsignal → STOP; G4 jedes abrupten Shift-Cases
  muss vor der Relevanzklassifizierung `VALID` sein, sonst STOP.

HELD_OUT wird nie geöffnet.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibration_runner.authorization import (
    AuthorizationError,
    ExecutionMode,
    RealExecution,
    SyntheticExecution,
    require_real_components,
    require_synthetic_components,
)
from calibration_runner.cases import (
    CLASS_FAMILY,
    SEVERITY_BASE,
    SUBGROUP_FAMILY,
    CaseWindow,
    class_cases,
    class_compositions,
    clean_window,
    require_feasible,
    subgroup_cases,
    subgroup_compositions,
    subgroup_stratum,
    technical_cases,
    verify_case,
)
from calibration_runner.evaluation import (
    CaseEvaluation,
    MethodConfigs,
    ReferenceArtifacts,
    case_seed,
    evaluate_case,
    performance,
    performance_status,
    reference_artifacts,
    summary,
    worst_status,
)
from calibration_runner.features import (
    ALL_KINDS,
    PREDICTION_ONLY,
    FeatureKind,
    FeatureStore,
    SampleFeatures,
)
from calibration_runner.freeze_assembly import (
    ArtifactWriter,
    TemplateMethods,
    build_manifest,
    g4_config,
    seed_list_sha256,
    validate_template,
)
from calibration_runner.ports import CalibrationDataPort, CalibrationModelPort
from calibration_runner.real_template import validate_real_template
from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.domain.assessments import SignalStatus
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.errors import (
    CalibrationHelperError,
    CalibrationProtocolError,
    CalibrationStopError,
    IntegrityError,
)
from vigiscoutx.experiment.calibration.compositions import CLEAN, ClassShiftPlan, SubgroupPlan
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    FusionCandidate,
    SamplingNamespace,
    StopReason,
    TechnicalShift,
)
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.calibration.process import (
    CalibrationAction,
    CalibrationProcess,
    CalibrationRunProvenance,
    CalibrationStep,
    PublicPoolView,
)
from vigiscoutx.experiment.calibration.statistics import (
    MethodOutcomes,
    ProcedureOutcome,
    ShiftOutcome,
    d0,
    r_candidate_accepted,
    severity_median,
)
from vigiscoutx.experiment.calibration.structure import (
    ClassVector,
    repeat_seed,
    sampling_context,
)
from vigiscoutx.experiment.freeze import FreezeManifestV2
from vigiscoutx.experiment.held_out_guard import CalibrationAccessScope, HeldOutGuard
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.signals.input.mmd import median_positive_distance

Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CalibrationRunResult:
    """Ergebnis eines vollständig bis `FROZEN` geführten Laufs."""

    run_id: str
    record: CalibrationFreezeRecord
    manifest: FreezeManifestV2
    run_config: RunConfig
    output_dir: Path


def validity_decision(statuses: Sequence[SignalStatus]) -> bool:
    """`W_VALIDITY`-Run: `True` gültig, `False` nicht evaluierbar; `ERROR` → terminaler STOP."""
    status = worst_status(statuses)
    if status is SignalStatus.ERROR:
        raise CalibrationStopError(
            StopReason.MANDATORY_SIGNAL_NOT_VALID, "ERROR im W_VALIDITY-Run (kein W-Fallback)"
        )
    return status is SignalStatus.VALID


def require_valid(statuses: Sequence[SignalStatus], context: str) -> None:
    if worst_status(statuses) is not SignalStatus.VALID:
        raise CalibrationStopError(StopReason.MANDATORY_SIGNAL_NOT_VALID, context)


@dataclass(frozen=True, slots=True)
class _Repeat:
    clean: CaseEvaluation
    shifts: tuple[CaseEvaluation, ...]


class CalibrationRunner:
    """Führt genau einen Calibration-Lauf aus; jeder STOP und jeder Fehler ist terminal."""

    def __init__(
        self,
        *,
        guard: HeldOutGuard,
        split_manifest: SplitManifest,
        provenance: CalibrationRunProvenance,
        protocol: CalibrationProtocolConfig,
        dataset: CalibrationDataPort,
        audit_dataset: Ham10000Dataset,
        model: CalibrationModelPort,
        preprocessor: ImagePreprocessor,
        template: RunConfig,
        execution: ExecutionMode,
        output_dir: Path,
        batch_size: int = 32,
        progress: Progress | None = None,
    ) -> None:
        # Ausführungsgrenze zuerst: real nur autorisiert, synthetisch nur mit synthetischer
        # Provenienz. Danach alle Identitätsprüfungen; vor Prozessbeginn kein Datenzugriff.
        if isinstance(execution, RealExecution):
            # DEC-079: gepinnte Vorlagenidentität vor jedem Zugriff auf Modell oder Datensatz.
            validate_real_template(template)
            require_real_components(execution, model, dataset, audit_dataset)
        elif isinstance(execution, SyntheticExecution):
            require_synthetic_components(execution, model, dataset, audit_dataset)
        else:
            raise IntegrityError("unbekannter Ausführungsmodus")
        self._methods: TemplateMethods = validate_template(template, provenance)
        model_ref = model.snapshot_ref()
        if (model_ref.model_id, model_ref.sha256) != (
            provenance.model_id,
            provenance.model_snapshot_sha256,
        ):
            raise IntegrityError("Modell-Snapshot passt nicht zur Run-Provenienz")
        dataset_ref = dataset.snapshot_ref()
        if (dataset_ref.snapshot_id, dataset_ref.sha256) != (
            provenance.dataset.snapshot_id,
            provenance.dataset.sha256,
        ):
            raise IntegrityError("Datensatz passt nicht zur Run-Provenienz")
        if audit_dataset.snapshot_ref() != dataset_ref:
            raise IntegrityError("Audit-Datensatz ist nicht derselbe Snapshot")
        if isinstance(execution, RealExecution):
            observed = execution.observed
            bound = {
                "git_commit": (provenance.git_commit, observed.git_commit),
                "environment_lock_sha256": (
                    provenance.environment_lock_sha256,
                    observed.environment_lock_sha256,
                ),
                "template": (canonical_sha256(template), observed.run_config_template_sha256),
                "split": (provenance.split.sha256, observed.split_manifest_sha256),
                "model_weights_sha256": (
                    provenance.model_weights_sha256,
                    observed.model_weights_sha256,
                ),
                "authorization": (
                    provenance.config_snapshot_sha256s[2:3],
                    (execution.authorization.authorization_sha256(),),
                ),
            }
            mismatches = sorted(k for k, (a, b) in bound.items() if a != b)
            if mismatches:
                raise AuthorizationError(f"Provenienz weicht von der Freigabe ab: {mismatches}")
        expected_spec = ImagePreprocessingSpec.from_config(template.scientific.preprocessing)
        if preprocessor.spec != expected_spec:
            raise IntegrityError("Preprocessing passt nicht zur eingefrorenen Konfiguration")
        output_dir.mkdir(parents=True, exist_ok=False)
        self._writer = ArtifactWriter(output_dir, provenance.run_id)
        self._template = template
        self._execution = execution
        self._audit_dataset = audit_dataset
        self._protocol = protocol
        self._dataset = dataset
        self._model_ref: ModelSnapshotRef = model_ref
        self._progress = progress or (lambda _message: None)
        self._process = CalibrationProcess(guard, split_manifest, provenance, protocol)
        self._features = FeatureStore(
            process=self._process,
            dataset=dataset,
            model=model,
            preprocessor=preprocessor,
            g2_config=self._methods.g2,
            protocol=protocol,
            batch_size=batch_size,
        )
        self._configs: MethodConfigs | None = None
        self._pools: PublicPoolView | None = None
        self._canonical: dict[str, str] = {}
        self._reference: ReferenceArtifacts | None = None
        self._repeats: list[_Repeat] = []
        self._case_log: list[Mapping[str, object]] = []
        self._tune_sex: dict[str, str | None] = {}
        self._class_blocks: dict[CalibrationPool, dict[str, tuple[str, ...]]] = {}
        self._fixed_vector: ClassVector | None = None
        self._strata: dict[str, tuple[str, ...]] | None = None

    @property
    def process(self) -> CalibrationProcess:
        return self._process

    # --- Ausführung -----------------------------------------------------------------------------

    def run(self) -> CalibrationRunResult:
        started = time.monotonic()
        try:
            self._structure()
            self._reference_artifacts_and_sigma()
            self._window_size()
            self._official_reference()
            self._d0()
            self._severity()
            self._compositions()
            self._r_stability()
            self._fusion()
            self._audit()
            result = self._freeze()
        except CalibrationStopError as exc:
            if self._process.stopped is None:
                self._process.stop(StopReason(exc.reason))
            self._terminate("STOPPED", str(exc))
            raise
        except Exception as exc:
            if self._process.stopped is None and self._process.failure is None:
                self._process.abort(f"{type(exc).__name__}: {exc}")
            self._terminate("FAILED", f"{type(exc).__name__}: {exc}")
            raise
        self._progress(f"FROZEN nach {time.monotonic() - started:.1f} s")
        return result

    def _terminate(self, status: str, detail: str) -> None:
        """Terminalbericht ohne stichprobenbezogene CAL_AUDIT-Information."""
        self._writer.write_json(
            "termination.json",
            {
                "run_id": self._process.run_id,
                "status": status,
                "detail": detail,
                "stop_reason": None if self._process.stopped is None else self._process.stopped,
                "completed_steps": [s.name for s in self._process.completed],
            },
        )

    # --- Hilfen ---------------------------------------------------------------------------------

    def _execution_record(self) -> dict[str, object]:
        """Ausführungsmodus im ArtifactManifest; synthetische Läufe bleiben gekennzeichnet."""
        execution = self._execution
        if isinstance(execution, RealExecution):
            return {
                "mode": execution.mode,
                "authorization_sha256": execution.authorization.authorization_sha256(),
                "authorization_id": execution.authorization.authorization_id,
            }
        return {
            "mode": execution.mode,
            "authorization_sha256": None,
            "authorization_id": None,
            "synthetic_dataset_snapshot_sha256": execution.dataset_snapshot_sha256,
            "synthetic_model_snapshot_sha256": execution.model_snapshot_sha256,
        }

    def _cfg(self) -> MethodConfigs:
        if self._configs is None:
            raise CalibrationProtocolError("Methodenkonfiguration erst nach MMD-σ verfügbar")
        return self._configs

    def _pool_view(self) -> PublicPoolView:
        if self._pools is None:
            raise CalibrationProtocolError("Pools sind noch nicht abgeleitet")
        return self._pools

    def _ref(self) -> ReferenceArtifacts:
        if self._reference is None:
            raise CalibrationProtocolError("offizielle Reference fehlt")
        return self._reference

    def _by_class(self, pool: CalibrationPool) -> dict[str, tuple[str, ...]]:
        """Klassenblöcke von `CAL_REF`/`CAL_TUNE` (nach dem Pool-Split unveränderlich)."""
        if pool not in self._class_blocks:
            self._class_blocks[pool] = {
                b.class_id: b.lesion_ids
                for b in self._pool_view().ref_tune_blocks
                if b.pool is pool
            }
        return self._class_blocks[pool]

    def _samples_of(self, lesion_ids: Sequence[str]) -> list[str]:
        return [self._canonical[i] for i in lesion_ids]

    def _vector(self) -> ClassVector:
        """Klassenvektor `n_c(W)` (ab der offiziellen Reference unveränderlich)."""
        if self._fixed_vector is None:
            vector = self._process.results.class_vector
            if vector is None:
                raise CalibrationProtocolError("Klassenvektor fehlt")
            self._fixed_vector = vector
        return self._fixed_vector

    def _window_size_value(self) -> int:
        window_size = self._process.results.window_size
        if window_size is None:
            raise CalibrationProtocolError("Window Size fehlt")
        return window_size

    def _delta_rel(self) -> float:
        threshold = self._process.results.threshold
        if threshold is None or not self._process.results.delta_rel_frozen:
            raise CalibrationProtocolError("δ_rel ist nicht eingefroren")
        return threshold.delta_rel

    def _ba_ref(self) -> float:
        value = self._process.results.ba_ref
        if value is None:
            raise CalibrationProtocolError("BA_ref fehlt")
        return value

    def _allow(self, pool: CalibrationPool, actions: Sequence[CalibrationAction]) -> None:
        for action in actions:
            self._process.require_action(pool, action)

    def _seed(self, case: CaseWindow) -> int:
        return repeat_seed(
            self._process.split_hash,
            case.namespace,
            case.window_size,
            case.scenario_key,
            case.repeat_id,
        )

    def _evaluate(
        self,
        case: CaseWindow,
        rows: Sequence[SampleFeatures],
        reference: ReferenceArtifacts,
        *,
        with_g4: bool,
        delta_rel: float | None,
        verify: bool = True,
        audit_candidate: FusionCandidate | None = None,
    ) -> CaseEvaluation:
        """Genau eine Auswertung je Case; Window- und Seed-Provenienz werden vorher geprüft.

        `verify=False` nur für AUDIT_CLEAN: Dort bildet und prüft der Prozess die `window_id`
        intern; der Runner kennt die Sample-Identitäten nicht. `audit_candidate` ist genau für
        CAL_AUDIT Pflicht (Protokoll §13): nur der fixierte Kandidat wird berechnet.
        """
        if (audit_candidate is None) == (case.pool is CalibrationPool.CAL_AUDIT):
            raise CalibrationProtocolError("AUDIT_CLEAN erfordert genau den fixierten Kandidaten")
        if verify:
            verify_case(case, self._process.split_hash)
        signals = [
            CalibrationAction.SIGNAL_G1,
            CalibrationAction.SIGNAL_G2,
            CalibrationAction.SIGNAL_G3,
        ]
        if with_g4:
            signals.append(CalibrationAction.SIGNAL_G4)
        self._allow(case.pool, signals)
        seed = self._seed(case)
        evaluation = evaluate_case(
            case,
            reference,
            rows,
            configs=self._cfg(),
            seeds=case_seed(self._template.seeds, seed, case.repeat_id),
            model_ref=self._model_ref,
            labels=self._features.labels(case.sample_ids) if with_g4 else None,
            delta_rel=delta_rel,
        )
        self._case_log.append(
            {
                **case.provenance(reference.window_id, seed),
                "result": summary(evaluation, audit_candidate=audit_candidate),
            }
        )
        return evaluation

    def _reference_for(self, window_id: str, sample_ids: Sequence[str]) -> ReferenceArtifacts:
        ids = sorted(sample_ids)
        rows = self._features.clean(ids, ALL_KINDS)
        return reference_artifacts(window_id, ids, rows, self._features.labels(ids), g4_config())

    # --- Schritte 1–6 ---------------------------------------------------------------------------

    def _structure(self) -> None:
        process = self._process
        process.authorize()
        self._pools = process.derive_pool_assignment(self._dataset)
        self._canonical = process.derive_canonical_samples()
        process.restrict_audit()
        for pool, scope in (
            (CalibrationPool.CAL_REF, CalibrationAccessScope.CAL_REF_ACCESS),
            (CalibrationPool.CAL_TUNE, CalibrationAccessScope.CAL_TUNE_ACCESS),
        ):
            lesions = sorted(self._pool_view().lesion_ids(pool))
            self._features.bind_pool(pool, process.issue_permit(scope), self._samples_of(lesions))
        self._progress("Schritte 1–4: Pools, Canonical Samples, Audit-Sperre")

    def _reference_artifacts_and_sigma(self) -> None:
        process = self._process
        sources = list(process.sigma_source_sample_ids())
        self._features.clean(sources, ALL_KINDS)
        process.complete_reference_artifacts()
        embeddings = self._features.clean(sources, frozenset({FeatureKind.EMBEDDING}))
        matrix = np.array([row.embedding for row in embeddings], dtype=np.float64)
        try:
            sigma = median_positive_distance(matrix)
        except CalibrationHelperError as exc:
            raise CalibrationStopError(StopReason.SIGMA_INVALID, str(exc)) from exc
        process.record_mmd_sigma(sigma)
        self._configs = MethodConfigs(
            g1=self._methods.g1(sigma), g2=self._methods.g2, g3=self._methods.g3, g4=g4_config()
        )
        self._progress(f"Schritte 5–6: Reference-Artefakte, MMD-σ = {sigma:.6g}")

    # --- Schritte 7–14 --------------------------------------------------------------------------

    def _structurally_feasible(self, vector: ClassVector) -> bool:
        """Nur IDs, Klassen, Pools, Canonical-Mapping; ohne Pixelzugriff (CAL_AUDIT strukturell)."""
        if not vector.valid:
            return False
        counts = vector.as_dict()
        audit = dict(self._pool_view().audit_class_counts)
        for pool in (CalibrationPool.CAL_REF, CalibrationPool.CAL_TUNE):
            available = self._by_class(pool)
            if any(len(available.get(c, ())) < n for c, n in counts.items()):
                return False
        if any(audit.get(c, 0) < n for c, n in counts.items()):
            return False
        lesions = [i for b in self._pool_view().ref_tune_blocks for i in b.lesion_ids]
        return all(i in self._canonical for i in (*lesions, *self._pool_view().audit_lesion_ids))

    def _window_validity(self, window_size: int) -> bool:
        vector = self._process.class_vector_for(window_size)
        if not self._structurally_feasible(vector):
            return False
        candidate = self._process.candidate_reference(window_size)
        reference = self._reference_for(candidate.window_id, candidate.sample_ids)
        tune = self._by_class(CalibrationPool.CAL_TUNE)
        eligible = True
        for repeat_id in range(1, self._protocol.window_validity_runs + 1):
            case = clean_window(
                SamplingNamespace.W_VALIDITY,
                vector,
                repeat_id,
                pool=CalibrationPool.CAL_TUNE,
                split_hash=self._process.split_hash,
                available=tune,
                canonical=self._canonical,
            )
            rows = self._features.clean(case.sample_ids, ALL_KINDS)
            evaluation = self._evaluate(case, rows, reference, with_g4=True, delta_rel=None)
            assessment = evaluation.performance
            assert assessment is not None
            statuses = [*evaluation.mandatory_statuses(), performance_status(assessment)]
            eligible = validity_decision(statuses) and eligible
        return eligible

    def _window_size(self) -> None:
        for window_size in self._protocol.window_size_candidates:
            eligible = self._window_validity(window_size)
            self._process.record_window_validity(window_size, eligible)
            self._progress(f"Schritt W_VALIDITY: W = {window_size}, ELIGIBLE = {eligible}")
            if eligible:
                break
        self._process.select_window_size()

    def _official_reference(self) -> None:
        reference = self._process.derive_official_reference()
        self._reference = self._reference_for(reference.window_id, reference.sample_ids)
        evaluation = self._reference.performance
        if evaluation.status is not SignalStatus.VALID or evaluation.balanced_accuracy is None:
            raise CalibrationStopError(StopReason.MANDATORY_SIGNAL_NOT_VALID, "BA_ref undefiniert")
        self._process.record_ba_ref(evaluation.balanced_accuracy)
        self._progress(
            f"Schritte 10–11: offizielle Reference, BA_ref = {evaluation.balanced_accuracy:.4f}"
        )

    def _tune_clean(
        self, namespace: SamplingNamespace, repeat_id: int, scenario_key: str = CLEAN
    ) -> CaseWindow:
        return clean_window(
            namespace,
            self._vector(),
            repeat_id,
            pool=CalibrationPool.CAL_TUNE,
            split_hash=self._process.split_hash,
            available=self._by_class(CalibrationPool.CAL_TUNE),
            canonical=self._canonical,
            scenario_key=scenario_key,
        )

    def _d0(self) -> None:
        ba_ref = self._ba_ref()
        g4 = self._cfg().g4
        self._allow(CalibrationPool.CAL_TUNE, [CalibrationAction.D0])
        values: list[float] = []
        for repeat_id in range(1, self._protocol.d0_repeats + 1):
            case = self._tune_clean(SamplingNamespace.D0, repeat_id)
            verify_case(case, self._process.split_hash)
            rows = self._features.clean(case.sample_ids, PREDICTION_ONLY)
            labels = self._features.labels(case.sample_ids)
            evaluation = performance(case.window_id, case.sample_ids, rows, labels, g4)
            if evaluation.status is not SignalStatus.VALID or evaluation.balanced_accuracy is None:
                raise CalibrationStopError(StopReason.D0_REPEAT_NOT_VALID, f"Repeat {repeat_id}")
            values.append(d0(ba_ref, evaluation.balanced_accuracy))
        self._process.record_d0_values(values)
        threshold = self._process.derive_relevance_threshold()
        self._process.freeze_delta_rel()
        self._progress(f"Schritte 12–14: D0 (1.000 Windows), δ_rel = {threshold.delta_rel:.6g}")

    # --- Schritte 15–19 -------------------------------------------------------------------------

    def _severity(self) -> None:
        ba_ref = self._ba_ref()
        g4 = self._cfg().g4
        shifts = self._protocol.technical_shifts
        degradations: dict[TechnicalShift, list[list[float]]] = {
            s: [[] for _ in range(5)] for s in shifts
        }
        invalid: dict[TechnicalShift, set[int]] = {s: set() for s in shifts}
        for repeat_id in range(1, self._protocol.severity_base_windows + 1):
            window = self._tune_clean(SamplingNamespace.SEVERITY, repeat_id, SEVERITY_BASE)
            verify_case(window, self._process.split_hash)
            labels = self._features.labels(window.sample_ids)
            for shift in shifts:
                grid = self._protocol.severity_grid(shift)
                shifted = self._features.shifted(
                    window.sample_ids, shift, grid, PREDICTION_ONLY, repeat_id=repeat_id
                )
                for index, value in enumerate(grid):
                    evaluation = performance(
                        window.window_id, window.sample_ids, shifted[value], labels, g4
                    )
                    if evaluation.status is SignalStatus.ERROR:
                        raise CalibrationStopError(
                            StopReason.MANDATORY_SIGNAL_NOT_VALID, f"G4 ERROR {shift.value}"
                        )
                    if evaluation.balanced_accuracy is None:
                        invalid[shift].add(index)
                        continue
                    degradations[shift][index].append(d0(ba_ref, evaluation.balanced_accuracy))
        medians = {
            shift: tuple(
                None if index in invalid[shift] else severity_median(degradations[shift][index])
                for index in range(5)
            )
            for shift in shifts
        }
        self._process.record_severity_medians(medians)
        self._process.confirm_severity_selection()
        selected = {s.shift.value: s.values for s in self._process.results.severities or ()}
        self._progress(f"Schritte 15–16: Severities {selected}")

    def _compositions(self) -> None:
        process = self._process
        plan = process.derive_class_shift()
        reference = process.results.reference
        if reference is None:
            raise CalibrationProtocolError("offizielle Reference fehlt")
        ref_lesions = sorted(self._pool_view().lesion_ids(CalibrationPool.CAL_REF))
        tune_lesions = sorted(self._pool_view().lesion_ids(CalibrationPool.CAL_TUNE))
        ref_sex = self._sex(ref_lesions)
        self._tune_sex = self._sex(tune_lesions)
        subgroup = process.record_subgroup(
            {i: ref_sex[i] for i in reference.lesion_ids}, ref_sex, self._tune_sex
        )
        process.derive_schedules()
        self._progress(
            f"Schritte 17–19: Class-Shift {plan.levels}, Subgroup feasible = {subgroup.feasible}"
        )

    def _sex(self, lesion_ids: Sequence[str]) -> dict[str, str | None]:
        samples = self._samples_of(lesion_ids)
        metadata = self._features.metadata(samples)
        return {
            lesion: metadata[sample].sex for lesion, sample in zip(lesion_ids, samples, strict=True)
        }

    # --- Schritt 20: R-Stabilität ---------------------------------------------------------------

    def _class_plan(self) -> ClassShiftPlan:
        plan = self._process.results.class_shift
        if plan is None:
            raise CalibrationProtocolError("Class-Shift-Plan fehlt")
        return plan

    def _subgroup_plan(self) -> SubgroupPlan:
        plan = self._process.results.subgroup
        if plan is None:
            raise CalibrationProtocolError("Subgroup-Plan fehlt")
        return plan

    def _subgroup_strata(self) -> dict[str, tuple[str, ...]]:
        """(Klasse, `sex`)-Strata der `CAL_TUNE`-Läsionen (ab Schritt 18 unveränderlich)."""
        if self._strata is not None:
            return self._strata
        plan = self._subgroup_plan()
        tune = self._by_class(CalibrationPool.CAL_TUNE)
        strata: dict[str, tuple[str, ...]] = {}
        for sex in (plan.source, plan.target):
            if sex is None:
                continue
            for class_id in self._protocol.class_order:
                strata[subgroup_stratum(class_id, sex)] = tuple(
                    i for i in tune.get(class_id, ()) if self._tune_sex.get(i) == sex
                )
        self._strata = strata
        return strata

    def _repeat_cases(self, repeat_id: int) -> tuple[CaseWindow, ...]:
        """15 abrupte Shift-Cases je Repeat, 18 bei feasible `SUBGROUP_SEX` (DEC-061)."""
        split_hash = self._process.split_hash
        vector = self._vector()
        tune = self._by_class(CalibrationPool.CAL_TUNE)
        cases: list[CaseWindow] = []
        for severity in self._process.results.severities or ():
            cases += technical_cases(
                severity,
                vector,
                repeat_id,
                split_hash=split_hash,
                tune_by_class=tune,
                canonical=self._canonical,
            )
        cases += class_cases(
            self._class_plan(),
            vector,
            repeat_id,
            self._protocol,
            split_hash=split_hash,
            tune_by_class=tune,
            canonical=self._canonical,
        )
        if self._subgroup_plan().feasible:
            cases += subgroup_cases(
                self._subgroup_plan(),
                vector,
                repeat_id,
                self._protocol,
                split_hash=split_hash,
                strata=self._subgroup_strata(),
                canonical=self._canonical,
            )
        expected = 18 if self._subgroup_plan().feasible else 15
        if len(cases) != expected:
            raise CalibrationProtocolError(
                f"Repeat {repeat_id}: {len(cases)} statt {expected} Cases"
            )
        return tuple(cases)

    def _shift_rows(
        self, cases: Sequence[CaseWindow], repeat_id: int
    ) -> list[Sequence[SampleFeatures]]:
        rows: list[Sequence[SampleFeatures]] = []
        technical: dict[TechnicalShift, dict[float, list[SampleFeatures]]] = {}
        for case in cases:
            if case.shift is None:
                rows.append(self._features.clean(case.sample_ids, ALL_KINDS))
                continue
            if case.shift not in technical:
                family = [c for c in cases if c.shift is case.shift]
                values = [c.shift_value for c in family if c.shift_value is not None]
                technical[case.shift] = self._features.shifted(
                    case.sample_ids, case.shift, values, ALL_KINDS, repeat_id=repeat_id
                )
            assert case.shift_value is not None
            rows.append(technical[case.shift][case.shift_value])
        return rows

    def _run_repeat(self, repeat_id: int) -> _Repeat:
        reference = self._ref()
        delta_rel = self._delta_rel()
        clean_case = self._tune_clean(SamplingNamespace.R_STABILITY, repeat_id)
        clean = self._evaluate(
            clean_case,
            self._features.clean(clean_case.sample_ids, ALL_KINDS),
            reference,
            with_g4=False,
            delta_rel=None,
        )
        require_valid(clean.mandatory_statuses(), f"R_STABILITY CLEAN {repeat_id}")
        cases = self._repeat_cases(repeat_id)
        shifts: list[CaseEvaluation] = []
        for case, rows in zip(cases, self._shift_rows(cases, repeat_id), strict=True):
            evaluation = self._evaluate(case, rows, reference, with_g4=True, delta_rel=delta_rel)
            assessment = evaluation.performance
            assert assessment is not None
            # Festgelegte Vorgabe: G4 jedes vorgesehenen Cases VALID, sonst STOP (kein Filtern).
            require_valid([performance_status(assessment)], f"G4 {case.scenario_key}/{repeat_id}")
            require_valid(evaluation.mandatory_statuses(), f"{case.scenario_key}/{repeat_id}")
            shifts.append(evaluation)
        return _Repeat(clean=clean, shifts=tuple(shifts))

    def _method_outcomes(self) -> dict[str, MethodOutcomes]:
        clean = [r.clean.procedure_outcomes() for r in self._repeats]
        shifts = [[e.shift_outcomes() for e in r.shifts] for r in self._repeats]
        return {
            method: MethodOutcomes(
                method=method,
                clean=tuple(c[method] for c in clean),
                shifts=tuple(tuple(s[method] for s in repeat) for repeat in shifts),
            )
            for method in self._protocol.r_methods
        }

    def _r_stability(self) -> None:
        tune = self._by_class(CalibrationPool.CAL_TUNE)
        require_feasible(class_compositions(self._class_plan(), self._protocol), tune)
        if self._subgroup_plan().feasible:
            require_feasible(
                subgroup_compositions(self._subgroup_plan(), self._protocol),
                self._subgroup_strata(),
            )
        candidates = self._protocol.r_candidates
        for index, candidate in enumerate(candidates):
            while len(self._repeats) < candidate:
                self._repeats.append(self._run_repeat(len(self._repeats) + 1))
            outcomes = self._method_outcomes()
            last = index == len(candidates) - 1
            try:
                accepted = r_candidate_accepted(outcomes, candidate, self._protocol)
            except CalibrationStopError:
                accepted = True  # der Prozess reproduziert und protokolliert den STOP
            self._progress(f"Schritt 20: R = {candidate} akzeptiert = {accepted}")
            if accepted or last:
                final_r = self._process.record_r_outcomes(outcomes)
                self._progress(f"Schritt 20: finales R = {final_r}")
                return
        raise CalibrationProtocolError("R-Auswahl ohne Ergebnis beendet")

    # --- Schritte 21–24 -------------------------------------------------------------------------

    def _final_r(self) -> int:
        final_r = self._process.results.final_r
        if final_r is None:
            raise CalibrationProtocolError("finales R fehlt")
        return final_r

    def _fusion(self) -> None:
        final_r = self._final_r()
        reference = self._ref()
        self._allow(CalibrationPool.CAL_TUNE, [CalibrationAction.FUSION, CalibrationAction.FAR])
        clean: dict[FusionCandidate, list[ProcedureOutcome]] = {c: [] for c in FusionCandidate}
        for repeat_id in range(1, final_r + 1):
            case = self._tune_clean(SamplingNamespace.FUSION_CLEAN, repeat_id)
            evaluation = self._evaluate(
                case,
                self._features.clean(case.sample_ids, ALL_KINDS),
                reference,
                with_g4=False,
                delta_rel=None,
            )
            require_valid(evaluation.mandatory_statuses(), f"FUSION_CLEAN {repeat_id}")
            for candidate in FusionCandidate:
                clean[candidate].append(evaluation.fusion(candidate))
        # Fusion-Sensitivität: exakt dieselben abrupten R_STABILITY-Cases (DEC-064).
        shifts: dict[FusionCandidate, list[ShiftOutcome]] = {c: [] for c in FusionCandidate}
        for repeat in self._repeats[:final_r]:
            for evaluation in repeat.shifts:
                outcomes = evaluation.shift_outcomes()
                for candidate in FusionCandidate:
                    shifts[candidate].append(outcomes[candidate.value])
        selected = self._process.record_fusion_outcomes(clean, shifts)
        self._process.fix_fusion_candidate()
        self._progress(f"Schritte 21–22: Fusion {selected.value} fixiert")

    def _audit(self) -> None:
        """Genau ein AUDIT_CLEAN über R Windows; nur G1–G3 und der fixierte Kandidat (P4-02F).

        Die Window-Zugehörigkeit bleibt prozessintern: Der Prozess lädt die Bilder selbst und gibt
        sie ohne `sample_id`, `lesion_id` oder Pfad heraus; der Runner verwendet ausschließlich
        opake Positionskennungen und speichert keine Audit-Merkmale zwischen.
        """
        process = self._process
        process.release_audit()
        self._features.release_pool(CalibrationPool.CAL_TUNE)
        candidate = process.results.fusion_candidate
        if candidate is None or not process.results.fusion_fixed:
            raise CalibrationProtocolError("Fusion Candidate ist nicht fixiert")
        reference = self._ref()
        self._allow(CalibrationPool.CAL_AUDIT, [CalibrationAction.FUSION, CalibrationAction.FAR])
        outcomes: list[ProcedureOutcome] = []
        window_size = self._window_size_value()
        for repeat_id in range(1, self._final_r() + 1):
            inputs = process.audit_clean_inputs(repeat_id, self._audit_dataset)
            expected = sampling_context(
                SamplingNamespace.AUDIT_CLEAN, window_size, CLEAN, repeat_id
            )
            if inputs.context != expected or len(inputs.images) != window_size:
                raise IntegrityError(f"AUDIT_CLEAN-Window {repeat_id} weicht vom Protokoll ab")
            positions = tuple(f"audit-{repeat_id:03d}-{i:03d}" for i in range(window_size))
            case = CaseWindow(
                namespace=SamplingNamespace.AUDIT_CLEAN,
                window_size=window_size,
                scenario_key=CLEAN,
                repeat_id=repeat_id,
                pool=CalibrationPool.CAL_AUDIT,
                context=inputs.context,
                base_context=inputs.context,
                window_id=inputs.window_id,
                base_window_id=inputs.window_id,
                lesion_ids=(),
                sample_ids=positions,
                family=None,
                level=None,
                shift=None,
                shift_value=None,
            )
            rows = self._features.anonymous(CalibrationPool.CAL_AUDIT, positions, inputs.images)
            evaluation = self._evaluate(
                case,
                rows,
                reference,
                with_g4=False,
                delta_rel=None,
                verify=False,
                audit_candidate=candidate,
            )
            require_valid(evaluation.mandatory_statuses(), f"AUDIT_CLEAN {repeat_id}")
            outcomes.append(evaluation.audit_outcomes(candidate)[candidate.value])
        far = process.record_audit_outcomes(outcomes)
        self._progress(f"Schritte 23–24: Audit-FAR = {far.successes}/{far.trials}")

    # --- Schritte 25–27 -------------------------------------------------------------------------

    def _freeze(self) -> CalibrationRunResult:
        process = self._process
        process.validate()
        for role, data in sorted(process.artifact_payloads().items()):
            self._writer.write(f"calibration/{role.removeprefix('calibration.')}.json", data, role)
        self._writer.write_json("cases/case_provenance.json", self._case_log, "provenance.cases")
        self._writer.write_json(
            "provenance/execution.json", self._execution_record(), "provenance.execution"
        )
        process.record_artifact_manifest(self._writer.manifest())
        record = process.build_freeze_record()
        run_config, manifest = build_manifest(self._template, record)
        process.record_freeze_manifest(manifest)
        # Sämtliche Freeze-Dateien vor dem Zustandswechsel; nach FROZEN darf nichts mehr scheitern.
        self._writer.write_json("freeze/artifact_manifest.json", self._writer.manifest())
        self._writer.write_json("freeze/calibration_freeze_record.json", record)
        self._writer.write_json("freeze/freeze_manifest_v2.json", manifest)
        self._writer.write_json("freeze/run_config.json", run_config)
        self._writer.write_json(
            "freeze/summary.json",
            {
                "run_id": process.run_id,
                "freeze_id": manifest.freeze_id,
                "calibration_record_sha256": record.record_sha256(),
                "config_sha256": canonical_sha256(run_config.scientific),
                "seed_list_sha256": seed_list_sha256(manifest.seed_list),
            },
        )
        process.freeze()
        if not process.is_completed(CalibrationStep.FROZEN):
            raise CalibrationProtocolError("Freeze nicht abgeschlossen")
        return CalibrationRunResult(
            run_id=process.run_id,
            record=record,
            manifest=manifest,
            run_config=run_config,
            output_dir=self._writer.root,
        )


__all__ = [
    "CLASS_FAMILY",
    "SUBGROUP_FAMILY",
    "CalibrationRunResult",
    "CalibrationRunner",
    "require_valid",
    "validity_decision",
]

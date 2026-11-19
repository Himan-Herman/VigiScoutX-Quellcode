"""Autoritativer Calibration-Prozess: 28-Schritt-Sequenz, Ergebniszustand, Berechtigungen, Freeze.

Vertrauensmodell (P4-02B):

- Das Strukturinventar entsteht ausschließlich aus Split-Manifest (exakter Hash) und einer
  prozesseigenen strukturellen Berechtigung; Pools und Canonical Samples leitet der Prozess selbst
  mit dem eingefrorenen Algorithmus ab und verifiziert sie.
- Alle freeze-relevanten Ergebnisse werden beim Akzeptieren in einen unveränderlichen
  `CalibrationResults`-Zustand übernommen; wo möglich berechnet der Prozess sie selbst.
- Der `CalibrationFreezeRecord` wird ausschließlich aus diesem Zustand erzeugt; das
  FreezeManifest (schema 2) muss genau diesen Record binden.
- Berechtigungen sind prozess-, zustands- und pool-gebunden, einmalig je Scope und werden bei
  Zustandswechseln, STOP, Protokollfehler und FROZEN widerrufen.
- Jeder Protokollfehler und jeder STOP ist terminal.

Der Prozess führt selbst keine Bild- oder Modellberechnung aus.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Concatenate, Literal, Protocol, Self

from PIL import Image
from pydantic import Field, StrictBool, StrictFloat, model_validator

from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.artifacts.manifests import ArtifactManifest
from vigiscoutx.domain.assessments import ParametersRef
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import DatasetSnapshotRef, SplitManifestRef
from vigiscoutx.domain.types import FrozenModel, GitCommitHash, Identifier, Sha256Hex
from vigiscoutx.errors import (
    CalibrationProtocolError,
    CalibrationStopError,
    IntegrityError,
    PhaseViolationError,
)
from vigiscoutx.experiment.calibration.compositions import (
    CLEAN,
    ClassShiftPlan,
    SubgroupPlan,
    abrupt_schedule,
    class_shift_plan,
    gradual_schedule,
    subgroup_plan,
)
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    FusionCandidate,
    SamplingNamespace,
    StopReason,
    TechnicalShift,
)
from vigiscoutx.experiment.calibration.freeze_record import (
    CalibrationFreezeRecord,
    ClassShiftFreeze,
    PoolManifestRef,
    SubgroupFreeze,
    TechnicalSeverityFreeze,
    verify_record_against_manifest,
)
from vigiscoutx.experiment.calibration.statistics import (
    FusionMetrics,
    MethodOutcomes,
    ProcedureOutcome,
    Rate,
    RelevanceThreshold,
    ShiftOutcome,
    audit_passed,
    count_rate,
    fusion_metrics,
    relevance_threshold,
    require_valid_alarms,
    select_fusion,
    select_r,
    select_severity_triple,
)
from vigiscoutx.experiment.calibration.structure import (
    CalibrationInventory,
    CalibrationLesion,
    ClassVector,
    PoolAssignment,
    PoolClassBlock,
    WindowSelection,
    assign_pools,
    canonical_samples,
    class_vector,
    pool_manifest_sha256,
    reference_sha256,
    sample_window,
    sampling_context,
    select_reference,
    verify_canonical_mapping,
    verify_pool_assignment,
)
from vigiscoutx.experiment.freeze import AnyFreezeManifest, verify_freeze_manifest
from vigiscoutx.experiment.held_out_guard import (
    CalibrationAccessPermit,
    CalibrationAccessScope,
    HeldOutGuard,
)
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import verify_split_manifest_hash
from vigiscoutx.experiment.splitting import StratificationLabelSource
from vigiscoutx.experiment.types import Partition, Phase


class CalibrationStep(IntEnum):
    """Finale Calibration-Sequenz (CALIBRATION_PROTOCOL_BASELINE 14.1)."""

    AUTHORIZE = 1
    POOL_SPLIT = 2
    CANONICAL_SAMPLES = 3
    RESTRICT_AUDIT = 4
    REF_ARTIFACTS = 5
    MMD_SIGMA = 6
    W128_VALIDITY = 7
    W256_VALIDITY = 8
    SELECT_WINDOW_SIZE = 9
    OFFICIAL_REFERENCE = 10
    BA_REF = 11
    D0_WINDOWS = 12
    D0_QUANTILES = 13
    DELTA_REL = 14
    SEVERITY_CANDIDATES = 15
    SEVERITY_SELECTION = 16
    CLASS_SHIFT = 17
    SUBGROUP = 18
    SCHEDULES = 19
    R_SELECTION = 20
    FUSION_SELECTION = 21
    FUSION_FIXED = 22
    AUDIT_RELEASE = 23
    AUDIT_FAR = 24
    VALIDATE = 25
    FREEZE_MANIFEST = 26
    FROZEN = 27
    HELD_OUT = 28


class CalibrationAction(StrEnum):
    """Datenbezogene Aktionen, die je Pool freigegeben oder verweigert werden."""

    STRUCTURE = "STRUCTURE"
    IMAGE_LOAD = "IMAGE_LOAD"
    PIXELS = "PIXELS"
    PREPROCESSING = "PREPROCESSING"
    EMBEDDINGS = "EMBEDDINGS"
    INFERENCE = "INFERENCE"
    SIGNAL_G1 = "SIGNAL_G1"
    SIGNAL_G2 = "SIGNAL_G2"
    SIGNAL_G3 = "SIGNAL_G3"
    SIGNAL_G4 = "SIGNAL_G4"
    PERFORMANCE = "PERFORMANCE"
    D0 = "D0"
    FUSION = "FUSION"
    FAR = "FAR"
    SUBGROUP_METADATA = "SUBGROUP_METADATA"


AUDIT_ALLOWED_AFTER_RELEASE = frozenset(
    {
        CalibrationAction.STRUCTURE,
        CalibrationAction.IMAGE_LOAD,
        CalibrationAction.PIXELS,
        CalibrationAction.PREPROCESSING,
        CalibrationAction.EMBEDDINGS,
        CalibrationAction.INFERENCE,
        CalibrationAction.SIGNAL_G1,
        CalibrationAction.SIGNAL_G2,
        CalibrationAction.SIGNAL_G3,
        CalibrationAction.FUSION,
        CalibrationAction.FAR,
    }
)
"""`AUDIT_CLEAN`: nur G1–G3, gewählte Fusion und Clean FAR; nie G4, Performance, D0, Metadaten."""

_POOL_SCOPES: Mapping[CalibrationAccessScope, CalibrationPool] = {
    CalibrationAccessScope.CAL_REF_ACCESS: CalibrationPool.CAL_REF,
    CalibrationAccessScope.CAL_TUNE_ACCESS: CalibrationPool.CAL_TUNE,
    CalibrationAccessScope.CAL_AUDIT_STRUCTURAL_ACCESS: CalibrationPool.CAL_AUDIT,
    CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS: CalibrationPool.CAL_AUDIT,
}

_INTERNAL_SCOPES = frozenset(
    {
        CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS,
        CalibrationAccessScope.CAL_AUDIT_STRUCTURAL_ACCESS,
        CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS,
    }
)
"""Nie öffentlich ausstellbar. Strukturelle CAL_AUDIT-Information und die CAL_AUDIT-Bild-
berechtigung bleiben prozessintern (P4-02F): Die Window-Zugehörigkeit von `AUDIT_CLEAN` würde
zusammen mit dem öffentlichen Ranking die Audit-Diagnosen rekonstruierbar machen."""

_SCOPE_EPOCHS: Mapping[CalibrationAccessScope, frozenset[int]] = {
    CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS: frozenset({0}),
    CalibrationAccessScope.CAL_REF_ACCESS: frozenset({1, 2}),
    CalibrationAccessScope.CAL_TUNE_ACCESS: frozenset({1}),
    CalibrationAccessScope.CAL_AUDIT_STRUCTURAL_ACCESS: frozenset({1}),
    CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS: frozenset({2}),
}
"""Berechtigungsepochen: 0 vor der Audit-Sperre, 1 bis AUDIT_RELEASE, 2 danach, 3 FROZEN."""


class CalibrationRunProvenance(FrozenModel):
    """Aktive Run-Provenienz; wird bei Prozessbeginn festgelegt und in den Freeze übernommen."""

    run_id: Identifier
    dataset: DatasetSnapshotRef
    split: SplitManifestRef
    model_id: Literal["resnet18-pilot-P1"]
    model_weights_sha256: Sha256Hex
    model_snapshot_sha256: Sha256Hex
    preprocessing_sha256: Sha256Hex
    method_refs: tuple[ParametersRef, ...] = Field(min_length=1)
    config_snapshot_sha256s: tuple[Sha256Hex, ...] = Field(min_length=1)
    git_commit: GitCommitHash
    environment_lock_sha256: Sha256Hex


class SelectedSeverity(FrozenModel):
    shift: TechnicalShift
    medians: tuple[StrictFloat | None, ...]
    indices: tuple[int, ...]
    values: tuple[StrictFloat, ...]


class ScheduleSpec(FrozenModel):
    family: Identifier
    kind: Literal["ABRUPT", "GRADUAL"]
    timepoints: tuple[str, ...]


class _ResultFields(FrozenModel):
    """Gemeinsame Felder von internem Zustand und öffentlicher Sicht."""

    inventory_sha256: Sha256Hex | None
    pool_manifest_sha256: Sha256Hex | None
    canonical_samples: tuple[tuple[str, str], ...] | None
    canonical_samples_sha256: Sha256Hex | None
    mmd_sigma: StrictFloat | None
    mmd_sigma_source_sha256: Sha256Hex | None
    window_validity: tuple[tuple[int, StrictBool], ...] | None
    window_size: Literal[128, 256] | None
    class_vector: ClassVector | None
    reference: WindowSelection | None
    reference_sha256: Sha256Hex | None
    ba_ref: StrictFloat | None
    d0_values: tuple[StrictFloat, ...] | None
    d0_artifact_sha256: Sha256Hex | None
    threshold: RelevanceThreshold | None
    delta_rel_frozen: StrictBool | None
    severities: tuple[SelectedSeverity, ...] | None
    severity_medians_sha256: Sha256Hex | None
    class_shift: ClassShiftPlan | None
    subgroup: SubgroupPlan | None
    schedules: tuple[ScheduleSpec, ...] | None
    final_r: Literal[100, 200, 400] | None
    r_outcomes_sha256: Sha256Hex | None
    fusion_metrics: tuple[FusionMetrics, ...] | None
    fusion_candidate: FusionCandidate | None
    fusion_fixed: StrictBool | None
    audit_far: Rate | None
    artifact_manifest_sha256: Sha256Hex | None

    @model_validator(mode="after")
    def _sigma(self) -> Self:
        sigma = self.mmd_sigma
        if sigma is not None and not (math.isfinite(sigma) and sigma > 0):
            raise ValueError("MMD-σ muss endlich und > 0 sein")
        return self


class CalibrationResults(_ResultFields):
    """Interner, autoritativer Ergebniszustand (enthält die vollständige Pool-Zuordnung)."""

    pool_assignment: PoolAssignment | None

    @classmethod
    def empty(cls) -> Self:
        return cls.model_validate(dict.fromkeys(cls.model_fields))


class PublicPoolView(FrozenModel):
    """Pool-Sicht ohne stichprobenbezogene CAL_AUDIT-Diagnosen.

    `CAL_REF`/`CAL_TUNE` mit Klassenblöcken; `CAL_AUDIT` nur als sortierte `lesion_id`-Menge
    plus aggregierte Klassencounts.
    """

    ref_tune_blocks: tuple[PoolClassBlock, ...]
    audit_lesion_ids: tuple[Identifier, ...]
    audit_class_counts: tuple[tuple[Identifier, int], ...]

    def lesion_ids(self, pool: CalibrationPool) -> frozenset[str]:
        if pool is CalibrationPool.CAL_AUDIT:
            return frozenset(self.audit_lesion_ids)
        return frozenset(i for b in self.ref_tune_blocks if b.pool is pool for i in b.lesion_ids)


@dataclass(frozen=True, slots=True, eq=False)
class AuditWindowInputs:
    """Bildeingaben genau eines `AUDIT_CLEAN`-Windows ohne rückverknüpfbare Identität.

    Bilder in der Reihenfolge der sortierten `sample_id`s (dieselbe Signalreihenfolge wie bei
    allen Windows); weder `sample_id` noch `lesion_id`, Diagnose oder Dateipfad werden ausgegeben.
    `window_id` ist der nicht umkehrbare Hash `SHA256(split_hash ‖ context ‖ sorted(sample_ids))`.
    """

    repeat_id: int
    context: str
    window_id: str
    images: tuple[Image.Image, ...]


class CalibrationResultsView(_ResultFields):
    """Öffentliche, redigierte Sicht: keine Zuordnung CAL_AUDIT-Sample/Läsion → Diagnose."""

    pools: PublicPoolView | None


def _public_pools(assignment: PoolAssignment) -> PublicPoolView:
    audit = [b for b in assignment.blocks if b.pool is CalibrationPool.CAL_AUDIT]
    ref_tune = [b for b in assignment.blocks if b.pool is not CalibrationPool.CAL_AUDIT]
    return PublicPoolView(
        ref_tune_blocks=tuple(ref_tune),
        audit_lesion_ids=tuple(sorted(i for b in audit for i in b.lesion_ids)),
        audit_class_counts=tuple((b.class_id, len(b.lesion_ids)) for b in audit),
    )


ARTIFACT_ROLE_PREFIX = "calibration."
"""Reservierter Rollen-Namensraum wissenschaftlicher Calibration-Artefakte im ArtifactManifest."""


class CalibrationLabelSource(StratificationLabelSource, Protocol):
    """Strukturelle Labelquelle mit überprüfbarer Datensatzidentität."""

    def snapshot_ref(self) -> DatasetSnapshotRef: ...


CONSTRUCTION_TOKEN = object()
"""Modulinternes Token: nur der Konstruktor von `CalibrationProcess` bindet sich an den Guard."""


def _require[T](value: T | None) -> T:
    if value is None:
        raise CalibrationProtocolError("autoritativer Calibration-Wert fehlt")
    return value


def _models_sha256(models: Sequence[FrozenModel]) -> str:
    return canonical_sha256([m.model_dump(mode="json") for m in models])


def _guarded[**P, R](
    method: Callable[Concatenate["CalibrationProcess", P], R],
) -> Callable[Concatenate["CalibrationProcess", P], R]:
    """Jede Ausnahme in einem Prozessschritt ist terminal (STOP bzw. Protokollfehler)."""

    def wrapper(self: "CalibrationProcess", /, *args: P.args, **kwargs: P.kwargs) -> R:
        self._require_live()
        try:
            return method(self, *args, **kwargs)
        except CalibrationStopError as exc:
            self.stop(StopReason(exc.reason))
            raise
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

    wrapper.__name__ = method.__name__
    wrapper.__qualname__ = method.__qualname__
    wrapper.__doc__ = method.__doc__
    return wrapper


class CalibrationProcess:
    """Durchsetzung der Calibration-Sequenz mit autoritativem Ergebniszustand."""

    def __init__(
        self,
        guard: HeldOutGuard,
        split_manifest: SplitManifest,
        provenance: CalibrationRunProvenance,
        config: CalibrationProtocolConfig,
    ) -> None:
        verify_split_manifest_hash(split_manifest, provenance.split)
        self._guard = guard
        self._split = split_manifest
        self._provenance = provenance
        self._config = config
        self._completed: list[CalibrationStep] = []
        self._stopped: StopReason | None = None
        self._failure: str | None = None
        self._inventory: CalibrationInventory | None = None
        self._results = CalibrationResults.empty()
        self._permits: dict[CalibrationAccessScope, CalibrationAccessPermit] = {}
        self._issued_scopes: set[CalibrationAccessScope] = set()
        self._manifest: AnyFreezeManifest | None = None
        self._r_outcomes_payload: list[object] | None = None
        self._audit_drawn = 0
        self._capability = guard.bind_calibration_process(self, CONSTRUCTION_TOKEN)

    def __copy__(self) -> "CalibrationProcess":
        raise TypeError("CalibrationProcess ist nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "CalibrationProcess":
        raise TypeError("CalibrationProcess ist nicht kopierbar")

    # --- Zustand ---------------------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._provenance.run_id

    @property
    def split_hash(self) -> str:
        return self._provenance.split.sha256

    @property
    def results(self) -> CalibrationResultsView:
        """Öffentliche, redigierte Sicht ohne stichprobenbezogene CAL_AUDIT-Diagnosen."""
        internal = self._results
        data = internal.model_dump(exclude={"pool_assignment"})
        pools = internal.pool_assignment
        return CalibrationResultsView.model_validate(
            {**data, "pools": None if pools is None else _public_pools(pools)}
        )

    @property
    def completed(self) -> tuple[CalibrationStep, ...]:
        return tuple(self._completed)

    @property
    def stopped(self) -> StopReason | None:
        return self._stopped

    @property
    def failure(self) -> str | None:
        return self._failure

    def is_completed(self, step: CalibrationStep) -> bool:
        return step in self._completed

    def _require_live(self) -> None:
        if self._stopped is not None:
            raise CalibrationProtocolError(f"Calibration gestoppt ({self._stopped.value})")
        if self._failure is not None:
            raise CalibrationProtocolError(f"Protokollfehler, Calibration beendet: {self._failure}")

    def stop(self, reason: StopReason) -> None:
        """Terminaler STOP → manuelle Entscheidung; alle Berechtigungen werden widerrufen."""
        if self._stopped is None:
            self._stopped = reason
        self._permits.clear()

    def _fail(self, detail: str) -> None:
        if self._failure is None and self._stopped is None:
            self._failure = detail
        self._permits.clear()

    def abort(self, detail: str) -> None:
        """Terminaler Protokoll-/Integritätsfehler, den der Orchestrator außerhalb eines
        Prozessschritts erkannt hat; alle Berechtigungen werden widerrufen."""
        self._fail(detail)

    def _record(self, **values: Any) -> None:
        self._results = self._results.model_copy(update=values)

    def _next_step(self) -> CalibrationStep:
        if not self._completed:
            return CalibrationStep.AUTHORIZE
        following = CalibrationStep(self._completed[-1] + 1)
        validity = dict(self._results.window_validity or ())
        if following is CalibrationStep.W256_VALIDITY and validity.get(128) is True:
            return CalibrationStep.SELECT_WINDOW_SIZE
        return following

    def _expect(self, step: CalibrationStep) -> None:
        if step is CalibrationStep.HELD_OUT:
            raise CalibrationProtocolError("HELD_OUT öffnet ausschließlich ein späteres Paket")
        expected = self._next_step()
        if step is not expected:
            raise CalibrationProtocolError(
                f"Unzulässige Reihenfolge: erwartet {expected.name}, erhalten {step.name}"
            )

    def _done(self, step: CalibrationStep) -> None:
        self._expect(step)
        self._completed.append(step)

    def _inv(self) -> CalibrationInventory:
        return _require(self._inventory)

    def _pools(self) -> PoolAssignment:
        return _require(self._results.pool_assignment)

    def _canonical(self) -> dict[str, str]:
        return dict(_require(self._results.canonical_samples))

    def _pool_class_lesions(self, pool: CalibrationPool) -> dict[str, tuple[str, ...]]:
        return self._pools().by_class(pool)

    def _pool_sample_ids(self, pool: CalibrationPool) -> frozenset[str]:
        canonical = self._canonical()
        return frozenset(canonical[i] for i in self._pools().lesion_ids(pool))

    # --- Berechtigungen --------------------------------------------------------------------------

    def _epoch(self) -> int:
        if self.is_completed(CalibrationStep.FROZEN):
            return 3
        if self.is_completed(CalibrationStep.AUDIT_RELEASE):
            return 2
        if self.is_completed(CalibrationStep.RESTRICT_AUDIT):
            return 1
        return 0

    def _revoke(self, *scopes: CalibrationAccessScope) -> None:
        for scope in scopes:
            self._permits.pop(scope, None)

    def check_permit(self, permit: CalibrationAccessPermit) -> None:
        """Nur das registrierte, nicht widerrufene Objekt dieses Prozesses ist gültig."""
        if self._stopped is not None or self._failure is not None:
            raise PhaseViolationError("Calibration beendet: alle Berechtigungen sind widerrufen")
        if self._permits.get(permit.scope) is not permit or permit.run_id != self.run_id:
            raise PhaseViolationError("Berechtigung ist unbekannt, fremd oder widerrufen")
        expected_pool = (
            None
            if permit.scope is CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS
            else self._results.pool_manifest_sha256
        )
        if permit.pool_manifest_sha256 != expected_pool:
            raise PhaseViolationError("Berechtigung ist an eine andere Pool-Zuordnung gebunden")
        if self._epoch() not in _SCOPE_EPOCHS[permit.scope]:
            raise PhaseViolationError("Berechtigung ist im aktuellen Protokollzustand ungültig")

    @_guarded
    def issue_permit(self, scope: CalibrationAccessScope) -> CalibrationAccessPermit:
        """Öffentlich nur `CAL_REF`, `CAL_TUNE` und – nach AUDIT_RELEASE – `CAL_AUDIT_SIGNAL`.

        Strukturelle Berechtigungen (gesamte CALIBRATION bzw. CAL_AUDIT) sind prozessintern; sie
        werden nie an Aufrufer ausgegeben, damit stichprobenbezogene CAL_AUDIT-Diagnosen und
        spätere CAL_AUDIT-Bilder nicht über Projekt-APIs zusammenkommen.
        """
        if scope in _INTERNAL_SCOPES:
            raise CalibrationProtocolError(f"{scope.value} ist ausschließlich prozessintern")
        return self._issue(scope)

    def _issue(self, scope: CalibrationAccessScope) -> CalibrationAccessPermit:
        if not self.is_completed(CalibrationStep.AUTHORIZE):
            raise CalibrationProtocolError("Calibration ist noch nicht autorisiert")
        if scope in self._issued_scopes:
            raise CalibrationProtocolError(f"{scope.value} wurde bereits ausgestellt (einmalig)")
        epoch = self._epoch()
        if epoch not in _SCOPE_EPOCHS[scope]:
            raise CalibrationProtocolError(f"{scope.value} im Protokollzustand {epoch} unzulässig")
        if scope is CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS:
            ids = frozenset(
                a.sample_id for a in self._split.assignments if a.partition is Partition.CALIBRATION
            )
            pool_hash = None
        else:
            ids = self._pool_sample_ids(_POOL_SCOPES[scope])
            pool_hash = _require(self._results.pool_manifest_sha256)
        permit = self._guard.issue_calibration_permit(
            issuer=self,
            capability=self._capability,
            run_id=self.run_id,
            scope=scope,
            epoch=epoch,
            pool_manifest_sha256=pool_hash,
            split_manifest=self._split,
            sample_ids=ids,
        )
        self._issued_scopes.add(scope)
        self._permits[scope] = permit
        return permit

    def require_action(self, pool: CalibrationPool, action: CalibrationAction) -> None:
        """Verweigert jede vor dem jeweiligen Protokollschritt unzulässige Aktion."""
        self._require_live()
        if not self.is_completed(CalibrationStep.AUTHORIZE):
            raise CalibrationProtocolError("Calibration ist noch nicht autorisiert")
        if action is CalibrationAction.STRUCTURE:
            return
        if not self.is_completed(CalibrationStep.RESTRICT_AUDIT):
            raise CalibrationProtocolError("Vor Pool-Split und Audit-Sperre nur Strukturzugriff")
        if pool is not CalibrationPool.CAL_AUDIT:
            return
        if not self.is_completed(CalibrationStep.AUDIT_RELEASE):
            raise CalibrationProtocolError(
                f"CAL_AUDIT vor AUDIT_RELEASE nur strukturell, nicht {action.value}"
            )
        if action not in AUDIT_ALLOWED_AFTER_RELEASE:
            raise CalibrationProtocolError(f"CAL_AUDIT erlaubt niemals {action.value}")

    # --- Schritte 1–6 ----------------------------------------------------------------------------

    @_guarded
    def authorize(self) -> None:
        self._expect(CalibrationStep.AUTHORIZE)
        self._guard.require_calibration_phase()
        self._done(CalibrationStep.AUTHORIZE)

    @_guarded
    def derive_pool_assignment(self, label_source: CalibrationLabelSource) -> PublicPoolView:
        """Schritt 2: Inventar aus Split + Strukturberechtigung; Pools per P4-01-Algorithmus.

        Vor dem ersten Label-Zugriff muss die Datensatzidentität der Labelquelle exakt der aktiven
        Provenienz entsprechen. Die interne Strukturberechtigung endet unmittelbar nach dem Lesen.
        """
        self._expect(CalibrationStep.POOL_SPLIT)
        source_ref = label_source.snapshot_ref()
        expected = self._provenance.dataset
        if (source_ref.snapshot_id, source_ref.sha256) != (expected.snapshot_id, expected.sha256):
            raise CalibrationProtocolError(
                "Labelquelle gehört nicht zum Datensatz der aktiven Provenienz"
            )
        permit = self._issue(CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS)
        rows = [a for a in self._split.assignments if a.partition is Partition.CALIBRATION]
        try:
            labels = label_source.stratification_labels_for(
                permit, sorted(a.sample_id for a in rows)
            )
        finally:
            self._revoke(CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS)
        order = self._config.class_order
        samples: dict[str, list[str]] = {}
        classes: dict[str, set[str]] = {}
        for row in rows:
            samples.setdefault(row.group_id, []).append(row.sample_id)
            classes.setdefault(row.group_id, set()).add(order[labels[row.sample_id]])
        inconsistent = sorted(g for g, c in classes.items() if len(c) != 1)
        if inconsistent:
            raise IntegrityError(f"Läsionen mit uneinheitlicher Diagnoseklasse: {inconsistent}")
        inventory = CalibrationInventory(
            dataset_snapshot_id=self._provenance.dataset.snapshot_id,
            dataset_snapshot_sha256=self._provenance.dataset.sha256,
            split_id=self._provenance.split.manifest_id,
            split_manifest_sha256=self.split_hash,
            lesions=tuple(
                CalibrationLesion(
                    lesion_id=lesion_id,
                    class_id=next(iter(classes[lesion_id])),
                    sample_ids=tuple(sorted(samples[lesion_id])),
                )
                for lesion_id in sorted(samples)
            ),
        )
        assignment = assign_pools(list(inventory.lesions), self.split_hash, self._config)
        verify_pool_assignment(assignment, inventory, self._config)
        self._inventory = inventory
        self._record(
            inventory_sha256=inventory.inventory_sha256(),
            pool_assignment=assignment,
            pool_manifest_sha256=pool_manifest_sha256(assignment),
        )
        self._done(CalibrationStep.POOL_SPLIT)
        return _public_pools(assignment)

    @_guarded
    def derive_canonical_samples(self) -> dict[str, str]:
        self._expect(CalibrationStep.CANONICAL_SAMPLES)
        inventory = self._inv()
        mapping = canonical_samples(list(inventory.lesions), self.split_hash, self._config)
        verify_canonical_mapping(mapping, inventory, self._config)
        pairs = tuple(sorted(mapping.items()))
        self._record(canonical_samples=pairs, canonical_samples_sha256=canonical_sha256(pairs))
        self._done(CalibrationStep.CANONICAL_SAMPLES)
        return mapping

    @_guarded
    def restrict_audit(self) -> None:
        self._done(CalibrationStep.RESTRICT_AUDIT)
        self._revoke(CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS)

    @_guarded
    def complete_reference_artifacts(self) -> None:
        self._done(CalibrationStep.REF_ARTIFACTS)

    def sigma_source_sample_ids(self) -> tuple[str, ...]:
        """Alle kanonischen `CAL_REF`-Samples (einzige zulässige σ-Datengrundlage)."""
        return tuple(sorted(self._pool_sample_ids(CalibrationPool.CAL_REF)))

    @_guarded
    def record_mmd_sigma(self, sigma: float) -> None:
        self._expect(CalibrationStep.MMD_SIGMA)
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise CalibrationStopError(StopReason.SIGMA_INVALID, f"σ = {sigma}")
        self._record(
            mmd_sigma=float(sigma),
            mmd_sigma_source_sha256=canonical_sha256(self.sigma_source_sample_ids()),
        )
        self._done(CalibrationStep.MMD_SIGMA)

    # --- Schritte 7–14 ---------------------------------------------------------------------------

    def class_vector_for(self, window_size: int) -> ClassVector:
        ref = self._pool_class_lesions(CalibrationPool.CAL_REF)
        counts = {c: len(ref.get(c, ())) for c in self._config.class_order}
        return class_vector(window_size, counts, self._config)

    def candidate_reference(self, window_size: int) -> WindowSelection:
        """Kandidaten-Reference je W (OFFICIAL_REFERENCE-Ranking)."""
        self._require_live()
        return select_reference(
            self._pool_class_lesions(CalibrationPool.CAL_REF),
            self.class_vector_for(window_size),
            self._canonical(),
            self.split_hash,
            self._config,
        )

    @_guarded
    def record_window_validity(self, window_size: int, eligible: bool) -> None:
        steps = {128: CalibrationStep.W128_VALIDITY, 256: CalibrationStep.W256_VALIDITY}
        if window_size not in steps:
            raise CalibrationProtocolError("W ausschließlich 128 oder 256")
        self._expect(steps[window_size])
        self._record(
            window_validity=(*(self._results.window_validity or ()), (window_size, eligible))
        )
        self._done(steps[window_size])

    @_guarded
    def select_window_size(self) -> int:
        """a-priori smallest-valid-window priority/fallback rule: 128 → 256 → STOP."""
        self._expect(CalibrationStep.SELECT_WINDOW_SIZE)
        validity = dict(self._results.window_validity or ())
        for window_size in self._config.window_size_candidates:
            if validity.get(window_size) is True:
                self._record(window_size=window_size)
                self._done(CalibrationStep.SELECT_WINDOW_SIZE)
                return window_size
        raise CalibrationStopError(StopReason.NO_ELIGIBLE_WINDOW_SIZE)

    @_guarded
    def derive_official_reference(self) -> WindowSelection:
        self._expect(CalibrationStep.OFFICIAL_REFERENCE)
        window_size = _require(self._results.window_size)
        reference = self.candidate_reference(window_size)
        self._record(
            class_vector=self.class_vector_for(window_size),
            reference=reference,
            reference_sha256=reference_sha256(reference),
        )
        self._done(CalibrationStep.OFFICIAL_REFERENCE)
        return reference

    @_guarded
    def record_ba_ref(self, ba_ref: float) -> None:
        self._expect(CalibrationStep.BA_REF)
        if not (math.isfinite(ba_ref) and 0.0 <= ba_ref <= 1.0):
            raise CalibrationProtocolError("BA_ref muss in [0, 1] liegen")
        self._record(ba_ref=float(ba_ref))
        self._done(CalibrationStep.BA_REF)

    @_guarded
    def record_d0_values(self, values: Sequence[float | None]) -> None:
        """Schritt 12: exakt 1.000 D0-Werte; ein nicht-valider Repeat → STOP ohne Ersatzdraw."""
        self._expect(CalibrationStep.D0_WINDOWS)
        if len(values) != self._config.d0_repeats:
            raise CalibrationProtocolError(f"exakt {self._config.d0_repeats} D0-Werte erforderlich")
        if any(v is None or not math.isfinite(v) for v in values):
            raise CalibrationStopError(StopReason.D0_REPEAT_NOT_VALID)
        clean = tuple(float(v) for v in values if v is not None)
        self._record(d0_values=clean, d0_artifact_sha256=canonical_sha256(clean))
        self._done(CalibrationStep.D0_WINDOWS)

    @_guarded
    def derive_relevance_threshold(self) -> RelevanceThreshold:
        self._expect(CalibrationStep.D0_QUANTILES)
        threshold = relevance_threshold(_require(self._results.d0_values), self._config)
        self._record(threshold=threshold)
        self._done(CalibrationStep.D0_QUANTILES)
        return threshold

    @_guarded
    def freeze_delta_rel(self) -> None:
        self._done(CalibrationStep.DELTA_REL)
        self._record(delta_rel_frozen=True)

    # --- Schritte 15–19 --------------------------------------------------------------------------

    def _delta_rel(self) -> float:
        if not self._results.delta_rel_frozen:
            raise CalibrationProtocolError("δ_rel ist noch nicht eingefroren")
        return _require(self._results.threshold).delta_rel

    @_guarded
    def record_severity_medians(
        self, medians: Mapping[TechnicalShift, Sequence[float | None]]
    ) -> None:
        """Schritt 15: Mediane der fünf Kandidaten je Familie; Auswahl nach Zielfunktion E."""
        self._expect(CalibrationStep.SEVERITY_CANDIDATES)
        if set(medians) != set(self._config.technical_shifts):
            raise CalibrationProtocolError("Mediane für exakt die vier technischen Familien")
        if any(len(v) != 5 for v in medians.values()):
            raise CalibrationProtocolError("je Familie exakt fünf Kandidaten-Mediane")
        delta_rel = self._delta_rel()
        selected = []
        for shift in self._config.technical_shifts:
            values = tuple(None if m is None else float(m) for m in medians[shift])
            indices = select_severity_triple(values, delta_rel, self._config)
            grid = self._config.severity_grid(shift)
            selected.append(
                SelectedSeverity(
                    shift=shift,
                    medians=values,
                    indices=indices,
                    values=tuple(grid[i] for i in indices),
                )
            )
        self._record(
            severities=tuple(selected),
            severity_medians_sha256=canonical_sha256([s.medians for s in selected]),
        )
        self._done(CalibrationStep.SEVERITY_CANDIDATES)

    @_guarded
    def confirm_severity_selection(self) -> None:
        self._done(CalibrationStep.SEVERITY_SELECTION)

    @_guarded
    def derive_class_shift(self) -> ClassShiftPlan:
        self._expect(CalibrationStep.CLASS_SHIFT)
        tune = self._pool_class_lesions(CalibrationPool.CAL_TUNE)
        available = {c: len(tune.get(c, ())) for c in self._config.class_order}
        plan = class_shift_plan(_require(self._results.class_vector), available, self._config)
        self._record(class_shift=plan)
        self._done(CalibrationStep.CLASS_SHIFT)
        return plan

    @_guarded
    def record_subgroup(
        self,
        reference_sex: Mapping[str, str | None],
        cal_ref_sex: Mapping[str, str | None],
        cal_tune_sex: Mapping[str, str | None],
    ) -> SubgroupPlan:
        """Schritt 18: `sex` ausschließlich für Reference, `CAL_REF` und `CAL_TUNE` (kein Audit)."""
        self._expect(CalibrationStep.SUBGROUP)
        reference = _require(self._results.reference)
        pools = self._pools()
        if set(reference_sex) != set(reference.lesion_ids):
            raise CalibrationProtocolError("sex exakt für die Läsionen der offiziellen Reference")
        if set(cal_ref_sex) != pools.lesion_ids(CalibrationPool.CAL_REF):
            raise CalibrationProtocolError("sex exakt für alle CAL_REF-Läsionen")
        if set(cal_tune_sex) != pools.lesion_ids(CalibrationPool.CAL_TUNE):
            raise CalibrationProtocolError("sex exakt für alle CAL_TUNE-Läsionen")
        order = self._config.class_order
        classes = {lesion.lesion_id: lesion.class_id for lesion in self._inv().lesions}
        ref_by_class = {
            c: [reference_sex[i] for i in reference.lesion_ids if classes[i] == c] for c in order
        }
        tune_by_class = {
            c: [cal_tune_sex[i] for i in sorted(cal_tune_sex) if classes[i] == c] for c in order
        }
        plan = subgroup_plan(
            ref_by_class, [cal_ref_sex[i] for i in sorted(cal_ref_sex)], tune_by_class, self._config
        )
        self._record(subgroup=plan)
        self._done(CalibrationStep.SUBGROUP)
        return plan

    @_guarded
    def derive_schedules(self) -> tuple[ScheduleSpec, ...]:
        self._expect(CalibrationStep.SCHEDULES)
        severities = _require(self._results.severities)
        class_shift = _require(self._results.class_shift)
        subgroup = _require(self._results.subgroup)
        families: list[tuple[str, tuple[str, ...]]] = [
            (s.shift.value, tuple(f"{s.shift.value}:{v!r}" for v in s.values)) for s in severities
        ]
        families.append(("CLASS", tuple(f"CLASS:{d}" for d in class_shift.levels)))
        if subgroup.feasible:
            families.append(("SUBGROUP_SEX", tuple(f"SUBGROUP_SEX:{d}" for d in subgroup.levels)))
        specs: list[ScheduleSpec] = []
        for family, levels in families:
            specs += [
                ScheduleSpec(family=family, kind="ABRUPT", timepoints=abrupt_schedule(level))
                for level in levels
            ]
            specs.append(
                ScheduleSpec(family=family, kind="GRADUAL", timepoints=gradual_schedule(levels))
            )
        self._record(schedules=tuple(specs))
        self._done(CalibrationStep.SCHEDULES)
        return tuple(specs)

    # --- Schritte 20–24 --------------------------------------------------------------------------

    @_guarded
    def record_r_outcomes(self, outcomes: Mapping[str, MethodOutcomes]) -> int:
        """Schritt 20: R aus status-erhaltenden Ergebnissen der elf Verfahren."""
        self._expect(CalibrationStep.R_SELECTION)
        r = select_r(outcomes, self._config)
        payload: list[object] = [
            outcomes[m].model_dump(mode="json") for m in self._config.r_methods
        ]
        self._r_outcomes_payload = payload
        self._record(final_r=r, r_outcomes_sha256=canonical_sha256(payload))
        self._done(CalibrationStep.R_SELECTION)
        return r

    @_guarded
    def record_fusion_outcomes(
        self,
        clean: Mapping[FusionCandidate, Sequence[ProcedureOutcome]],
        shifts: Mapping[FusionCandidate, Sequence[ShiftOutcome]],
    ) -> FusionCandidate:
        """Schritt 21: Auswahl A/B ausschließlich auf `CAL_TUNE` mit R Clean-Windows."""
        self._expect(CalibrationStep.FUSION_SELECTION)
        r = _require(self._results.final_r)
        if set(clean) != set(FusionCandidate) or set(shifts) != set(FusionCandidate):
            raise CalibrationProtocolError("Ergebnisse für exakt Fusion A und B erforderlich")
        if any(len(clean[c]) != r for c in FusionCandidate):
            raise CalibrationProtocolError("Clean FAR über exakt R Windows je Kandidat")
        metrics = tuple(fusion_metrics(c, clean[c], shifts[c]) for c in FusionCandidate)
        selected = select_fusion(metrics, self._config)
        self._record(fusion_metrics=metrics, fusion_candidate=selected)
        self._done(CalibrationStep.FUSION_SELECTION)
        return selected

    @_guarded
    def fix_fusion_candidate(self) -> FusionCandidate:
        """Schritt 22: fixiert ausschließlich den regelkonform gewählten Kandidaten."""
        self._expect(CalibrationStep.FUSION_FIXED)
        candidate = _require(self._results.fusion_candidate)
        self._record(fusion_fixed=True)
        self._done(CalibrationStep.FUSION_FIXED)
        return candidate

    @_guarded
    def release_audit(self) -> None:
        """Schritt 23: Strukturzugriff auf CAL_AUDIT und TUNE-Zugriff enden; Signalzugriff frei."""
        self._done(CalibrationStep.AUDIT_RELEASE)
        self._revoke(
            CalibrationAccessScope.CAL_AUDIT_STRUCTURAL_ACCESS,
            CalibrationAccessScope.CAL_TUNE_ACCESS,
        )

    def _audit_selection(self, repeat_id: int) -> WindowSelection:
        """Prozessinterne, deterministische AUDIT_CLEAN-Window-Konstruktion (DEC-057, DEC-059)."""
        window_size = _require(self._results.window_size)
        context = sampling_context(SamplingNamespace.AUDIT_CLEAN, window_size, CLEAN, repeat_id)
        return sample_window(
            self.split_hash,
            context,
            self._pool_class_lesions(CalibrationPool.CAL_AUDIT),
            _require(self._results.class_vector).as_dict(),
            self._canonical(),
        )

    @_guarded
    def audit_clean_inputs(self, repeat_id: int, dataset: Ham10000Dataset) -> AuditWindowInputs:
        """Vertrauenswürdiger Audit-Executor: lädt die Bilder des `AUDIT_CLEAN`-Windows
        `repeat_id` prozessintern und gibt ausschließlich die Bilder aus (P4-02F).

        Nur nach AUDIT_RELEASE und vor dem Audit-FAR; jedes Window genau einmal und in der
        Reihenfolge `1…R` (keine zweite Ziehung). Geladen wird ausschließlich über den offiziellen
        `Ham10000Dataset` und eine prozessinterne Berechtigung mit dem Zweck `INPUTS` (keine
        Labels, keine Metadaten, kein G4).
        """
        if not self.is_completed(CalibrationStep.AUDIT_RELEASE) or self.is_completed(
            CalibrationStep.AUDIT_FAR
        ):
            raise CalibrationProtocolError("AUDIT_CLEAN-Eingaben nur nach AUDIT_RELEASE")
        final_r = _require(self._results.final_r)
        if repeat_id != self._audit_drawn + 1 or repeat_id > final_r:
            raise CalibrationProtocolError(
                f"AUDIT_CLEAN-Window {repeat_id}: jedes repeat_id 1…{final_r} genau einmal in Folge"
            )
        if type(dataset) is not Ham10000Dataset:
            raise CalibrationProtocolError(
                "AUDIT_CLEAN lädt nur über den offiziellen Ham10000Dataset"
            )
        source = dataset.snapshot_ref()
        expected = self._provenance.dataset
        if (source.snapshot_id, source.sha256) != (expected.snapshot_id, expected.sha256):
            raise CalibrationProtocolError("Datensatz gehört nicht zur aktiven Provenienz")
        permit = self._permits.get(CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS)
        if permit is None:
            permit = self._issue(CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS)
        selection = self._audit_selection(repeat_id)
        samples = sorted(selection.sample_ids)
        images = tuple(dataset.load_inputs(permit, samples))
        if len(images) != len(samples):
            raise IntegrityError("Datensatz lieferte nicht alle AUDIT_CLEAN-Bilder")
        self._audit_drawn = repeat_id
        return AuditWindowInputs(
            repeat_id=repeat_id,
            context=selection.context,
            window_id=selection.window_id,
            images=images,
        )

    @_guarded
    def record_audit_outcomes(self, outcomes: Sequence[ProcedureOutcome]) -> Rate:
        """Schritt 24: genau ein Audit über R Clean-Windows; FAIL → STOP ohne Tuning."""
        self._expect(CalibrationStep.AUDIT_FAR)
        if len(outcomes) != _require(self._results.final_r):
            raise CalibrationProtocolError("Audit-FAR erfordert exakt R Clean-Windows")
        far = count_rate(require_valid_alarms(outcomes))
        self._done(CalibrationStep.AUDIT_FAR)
        self._revoke(CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS)
        if not audit_passed(far, self._config):
            raise CalibrationStopError(StopReason.AUDIT_FAR_EXCEEDED, f"FAR {far.value}")
        self._record(audit_far=far)
        return far

    # --- Schritte 25–28 --------------------------------------------------------------------------

    @_guarded
    def validate(self) -> None:
        self._expect(CalibrationStep.VALIDATE)
        missing = sorted(
            k
            for k, v in self._results.model_dump().items()
            if v is None and k != "artifact_manifest_sha256"
        )
        if missing:
            raise CalibrationProtocolError(f"unvollständige Calibration-Ergebnisse: {missing}")
        self._done(CalibrationStep.VALIDATE)

    def _tune_far(self) -> Rate:
        candidate = _require(self._results.fusion_candidate)
        for metrics in _require(self._results.fusion_metrics):
            if metrics.candidate is candidate:
                return metrics.clean_far
        raise CalibrationProtocolError("TUNE-FAR des gewählten Kandidaten fehlt")

    def seed_derivation_sha256(self) -> str:
        """Hash der deterministischen Seed-Ableitung (Namespaces, Regeln, R, Split-Hash)."""
        return canonical_sha256(
            {
                "split_hash": self.split_hash,
                "sampling_namespaces": [n.value for n in self._config.sampling_namespaces],
                "repeat_seed_purpose": self._config.repeat_seed_purpose,
                "noise_seed_namespace": self._config.noise_seed_namespace,
                "final_r": self._results.final_r,
            }
        )

    def public_pool_manifest(self) -> dict[str, object]:
        """Redigiertes, öffentliches Pool-Manifest (Entscheidung zur Audit-Blindheit).

        Enthält `CAL_REF`/`CAL_TUNE` vollständig, für `CAL_AUDIT` aber ausschließlich die sortierten
        `lesion_id`s und aggregierte Klassenzahlen. Die vollständige Zuordnung (inklusive der
        klassensortierten CAL_AUDIT-Blöcke) bleibt prozessintern und ist nur über die nicht
        rückverknüpfbaren Hashes `pool_manifest_sha256` bzw. je Pool gebunden.
        """
        assignment = self._pools()
        view = _public_pools(assignment)
        return {
            "split_hash": assignment.split_hash,
            "pool_manifest_sha256": _require(self._results.pool_manifest_sha256),
            "pool_manifests": [
                {
                    "pool": pool.value,
                    "n_lesions": len(assignment.lesion_ids(pool)),
                    "sha256": assignment.manifest_sha256(pool),
                }
                for pool in CalibrationPool
            ],
            "ref_tune_blocks": [b.model_dump(mode="json") for b in view.ref_tune_blocks],
            "audit_lesion_ids": list(view.audit_lesion_ids),
            "audit_class_counts": [list(c) for c in view.audit_class_counts],
        }

    def required_artifact_hashes(self) -> dict[str, str]:
        """Rolle → SHA-256 der wissenschaftlichen Calibration-Artefakte aus dem Prozesszustand."""
        res = self._results
        return {
            f"{ARTIFACT_ROLE_PREFIX}pool_manifest": canonical_sha256(self.public_pool_manifest()),
            f"{ARTIFACT_ROLE_PREFIX}canonical_samples": _require(res.canonical_samples_sha256),
            f"{ARTIFACT_ROLE_PREFIX}reference": _require(res.reference_sha256),
            f"{ARTIFACT_ROLE_PREFIX}mmd_sigma_source": _require(res.mmd_sigma_source_sha256),
            f"{ARTIFACT_ROLE_PREFIX}d0": _require(res.d0_artifact_sha256),
            f"{ARTIFACT_ROLE_PREFIX}severity_medians": _require(res.severity_medians_sha256),
            f"{ARTIFACT_ROLE_PREFIX}class_shift": _models_sha256([_require(res.class_shift)]),
            f"{ARTIFACT_ROLE_PREFIX}subgroup": _models_sha256([_require(res.subgroup)]),
            f"{ARTIFACT_ROLE_PREFIX}schedules": _models_sha256(_require(res.schedules)),
            f"{ARTIFACT_ROLE_PREFIX}r_outcomes": _require(res.r_outcomes_sha256),
            f"{ARTIFACT_ROLE_PREFIX}fusion": _models_sha256(_require(res.fusion_metrics)),
            f"{ARTIFACT_ROLE_PREFIX}audit_far": _models_sha256([_require(res.audit_far)]),
        }

    def artifact_payloads(self) -> dict[str, bytes]:
        """Rolle → kanonischer JSON-Inhalt der wissenschaftlichen Artefakte (nach VALIDATE).

        Der Inhalt stammt ausschließlich aus dem Prozesszustand; sein SHA-256 entspricht exakt
        `required_artifact_hashes()`. Erst nach VALIDATE (nach Abschluss des Audits) verfügbar.
        Kein Inhalt verknüpft ein CAL_AUDIT-Sample oder eine CAL_AUDIT-Läsion mit einer Diagnose.
        """
        self._require_live()
        if not self.is_completed(CalibrationStep.VALIDATE):
            raise CalibrationProtocolError("Artefakt-Inhalte erst nach der Validierung")
        res = self._results
        reference = _require(res.reference)
        objects: dict[str, object] = {
            "pool_manifest": self.public_pool_manifest(),
            "canonical_samples": _require(res.canonical_samples),
            "reference": {
                "lesion_ids": sorted(reference.lesion_ids),
                "sample_ids": sorted(reference.sample_ids),
            },
            "mmd_sigma_source": self.sigma_source_sample_ids(),
            "d0": _require(res.d0_values),
            "severity_medians": [s.medians for s in _require(res.severities)],
            "class_shift": [_require(res.class_shift).model_dump(mode="json")],
            "subgroup": [_require(res.subgroup).model_dump(mode="json")],
            "schedules": [m.model_dump(mode="json") for m in _require(res.schedules)],
            "r_outcomes": _require(self._r_outcomes_payload),
            "fusion": [m.model_dump(mode="json") for m in _require(res.fusion_metrics)],
            "audit_far": [_require(res.audit_far).model_dump(mode="json")],
        }
        payloads = {
            f"{ARTIFACT_ROLE_PREFIX}{key}": canonical_json_bytes(value)
            for key, value in objects.items()
        }
        required = self.required_artifact_hashes()
        wrong = sorted(r for r, data in payloads.items() if sha256_hex(data) != required[r])
        if set(payloads) != set(required) or wrong:
            raise IntegrityError(f"Artefakt-Inhalt weicht vom Calibration-Zustand ab: {wrong}")
        return payloads

    @_guarded
    def record_artifact_manifest(self, manifest: ArtifactManifest) -> str:
        """Prüft das ArtifactManifest gegen den Prozesszustand und übernimmt seinen Hash.

        Nach VALIDATE, vor dem FreezeManifest, genau einmal. Jede wissenschaftliche Rolle muss
        genau einmal mit dem autoritativen Hash vorkommen; unbekannte wissenschaftliche Rollen
        und ein fremder `run_id` sind unzulässig. Pfade sind durch `ArtifactEntry` relativ.
        """
        if not self.is_completed(CalibrationStep.VALIDATE) or self.is_completed(
            CalibrationStep.FREEZE_MANIFEST
        ):
            raise CalibrationProtocolError("ArtifactManifest nur zwischen VALIDATE und Freeze")
        if self._results.artifact_manifest_sha256 is not None:
            raise CalibrationProtocolError("ArtifactManifest wurde bereits übernommen")
        if manifest.run_id != self.run_id:
            raise CalibrationProtocolError("ArtifactManifest gehört zu einem anderen Run")
        required = self.required_artifact_hashes()
        scientific = [e for e in manifest.entries if e.role.startswith(ARTIFACT_ROLE_PREFIX)]
        roles = [e.role for e in scientific]
        if sorted(roles) != sorted(required):
            raise IntegrityError(
                f"wissenschaftliche Artefaktrollen weichen ab: {sorted(set(roles) ^ set(required))}"
            )
        wrong = sorted(e.role for e in scientific if e.sha256 != required[e.role])
        if wrong:
            raise IntegrityError(f"Artefakt-Hash weicht vom Calibration-Zustand ab: {wrong}")
        digest = canonical_sha256(manifest)
        self._record(artifact_manifest_sha256=digest)
        return digest

    def build_freeze_record(self) -> CalibrationFreezeRecord:
        """Erzeugt den Freeze-Record ausschließlich aus dem autoritativen Prozesszustand."""
        self._require_live()
        if not self.is_completed(CalibrationStep.VALIDATE):
            raise CalibrationProtocolError("Freeze-Record erst nach der Validierung")
        artifact_manifest_sha256 = self._results.artifact_manifest_sha256
        if artifact_manifest_sha256 is None:
            raise CalibrationProtocolError("ArtifactManifest ist noch nicht übernommen")
        res, prov, pools = self._results, self._provenance, self._pools()
        reference = _require(res.reference)
        threshold = _require(res.threshold)
        class_shift = _require(res.class_shift)
        subgroup = _require(res.subgroup)
        schedules = _require(res.schedules)
        return CalibrationFreezeRecord(
            schema_version=1,
            protocol_version="calibration-protocol-baseline:1.0",
            protocol_config_sha256=self._config.config_sha256(),
            dataset_snapshot_id=prov.dataset.snapshot_id,
            dataset_snapshot_sha256=prov.dataset.sha256,
            split_id=prov.split.manifest_id,
            split_manifest_sha256=prov.split.sha256,
            pool_manifest_sha256=_require(res.pool_manifest_sha256),
            pool_manifests=tuple(
                PoolManifestRef(
                    pool=pool,
                    n_lesions=len(pools.lesion_ids(pool)),
                    sha256=pools.manifest_sha256(pool),
                )
                for pool in CalibrationPool
            ),
            canonical_rule="MIN_HASH_TIE_LEXICOGRAPHIC_SAMPLE_ID",
            canonical_samples_sha256=_require(res.canonical_samples_sha256),
            window_size=_require(res.window_size),
            reference_window_id=reference.window_id,
            reference_lesion_ids=reference.lesion_ids,
            reference_sample_ids=reference.sample_ids,
            reference_sha256=_require(res.reference_sha256),
            mmd_sigma_source_sha256=_require(res.mmd_sigma_source_sha256),
            mmd_sigma=_require(res.mmd_sigma),
            d0_artifact_sha256=_require(res.d0_artifact_sha256),
            q090=threshold.q090,
            q095=threshold.q095,
            q099=threshold.q099,
            delta_rel=threshold.delta_rel,
            technical_severities=tuple(
                TechnicalSeverityFreeze(
                    shift=s.shift,
                    candidates=self._config.severity_grid(s.shift),
                    selected_indices=s.indices,
                    selected_values=s.values,
                )
                for s in _require(res.severities)
            ),
            class_shift=ClassShiftFreeze(
                source=class_shift.source,
                target=class_shift.target,
                d_max=class_shift.d_max,
                levels=class_shift.levels,
                compositions_sha256=canonical_sha256(class_shift.compositions),
            ),
            subgroup=SubgroupFreeze(
                feasible=subgroup.feasible,
                exclusion=subgroup.exclusion,
                levels=subgroup.levels,
                compositions_sha256=(
                    _models_sha256(subgroup.compositions) if subgroup.feasible else None
                ),
            ),
            abrupt_schedules_sha256=_models_sha256([s for s in schedules if s.kind == "ABRUPT"]),
            gradual_schedules_sha256=_models_sha256([s for s in schedules if s.kind == "GRADUAL"]),
            final_r=_require(res.final_r),
            seed_derivation_sha256=self.seed_derivation_sha256(),
            fusion_candidate=_require(res.fusion_candidate),
            tune_clean_far=self._tune_far(),
            audit_clean_far=_require(res.audit_far),
            model_id=prov.model_id,
            model_weights_sha256=prov.model_weights_sha256,
            model_snapshot_sha256=prov.model_snapshot_sha256,
            preprocessing_sha256=prov.preprocessing_sha256,
            monitoring_methods_version=self._config.monitoring_methods_version,
            method_refs=prov.method_refs,
            config_snapshot_sha256s=prov.config_snapshot_sha256s,
            git_commit=prov.git_commit,
            environment_lock_sha256=prov.environment_lock_sha256,
            artifact_manifest_sha256=artifact_manifest_sha256,
        )

    def verify_freeze_record(self, record: CalibrationFreezeRecord) -> None:
        """Jede Abweichung vom autoritativen Prozesszustand ist ein Integritätsfehler."""
        expected = self.build_freeze_record().model_dump(mode="json")
        actual = record.model_dump(mode="json")
        mismatches = sorted(k for k in expected if expected[k] != actual[k])
        if mismatches:
            raise IntegrityError(f"Freeze-Record weicht vom Calibration-Zustand ab: {mismatches}")

    @_guarded
    def record_freeze_manifest(self, manifest: AnyFreezeManifest) -> CalibrationFreezeRecord:
        """Schritt 26: Manifest (schema 2) muss exakt den prozesserzeugten Record binden."""
        self._expect(CalibrationStep.FREEZE_MANIFEST)
        _require(self._results.audit_far)
        record = self.build_freeze_record()
        verify_freeze_manifest(manifest)
        verify_record_against_manifest(record, manifest)
        self._manifest = manifest
        self._done(CalibrationStep.FREEZE_MANIFEST)
        return record

    @_guarded
    def freeze(self) -> None:
        """Schritt 27: Guard wechselt nach FROZEN; alle Calibration-Berechtigungen enden."""
        self._expect(CalibrationStep.FROZEN)
        if self._manifest is None:
            raise CalibrationProtocolError("Freeze erfordert ein geprüftes FreezeManifest")
        self._guard.freeze(self._manifest)
        self._done(CalibrationStep.FROZEN)
        self._permits.clear()

    def require_frozen_for_held_out(self) -> None:
        if not self.is_completed(CalibrationStep.FROZEN) or self._guard.phase is not Phase.FROZEN:
            raise CalibrationProtocolError("HELD_OUT erst im Zustand FROZEN")

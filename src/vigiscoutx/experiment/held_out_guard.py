"""Technischer Held-out Guard (ARC-016, DEC-033).

Wissenschaftliche Prozesskontrolle, keine Security-Isolation gegen absichtliche Manipulation.
Jede Abweichung ist ein harter Fehler; es gibt keinen Warning-only-Modus. Eine Änderung nach dem
Freeze erfordert eine neue Guard-Instanz mit neuem Freeze.
"""

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NoReturn, Protocol, SupportsIndex

from vigiscoutx.config.scientific_config import scientific_config_sha256
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Sha256Hex
from vigiscoutx.errors import IntegrityError, PhaseViolationError
from vigiscoutx.experiment.calibration.freeze_record import (
    CalibrationFreezeRecord,
    verify_record_against_manifest,
)
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    ExecutionPurpose,
    reverify_evaluation_provenance,
)
from vigiscoutx.experiment.freeze import (
    AnyFreezeManifest,
    FreezeManifestV2,
    verify_freeze_manifest,
)
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import (
    partition_sample_ids,
    split_manifest_sha256,
    validate_split_manifest,
)
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition, Phase
from vigiscoutx.ports import AccessPurpose, DataAccessContext

_ISSUE_TOKEN = object()
_PARTITION_PURPOSES = frozenset(
    {AccessPurpose.INPUTS, AccessPurpose.LABELS, AccessPurpose.METADATA}
)
_REFERENCE_PURPOSES = frozenset({AccessPurpose.INPUTS, AccessPurpose.LABELS})


class ObservedHashes(FrozenModel):
    """Vom Aufrufer aus den tatsächlich vorliegenden Artefakten berechnete Hashes.

    Kein Git-Commit (P4-03Q, DEC-082): Die Code-Provenienz ermittelt der Guard nicht aus
    Aufruferangaben, sondern ausschließlich über `EvaluationProvenance`.
    """

    dataset_snapshot_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    model_weights_sha256: Sha256Hex
    environment_lock_sha256: Sha256Hex


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationSession:
    """Read-only Zugang zur Held-out-Evaluation.

    Bietet bewusst keine fit-, calibrate-, select-threshold- oder update-policy-Operation.
    """

    manifest: AnyFreezeManifest
    config: RunConfig

    @property
    def freeze_id(self) -> str:
        return self.manifest.freeze_id


@dataclass(frozen=True, slots=True, eq=False)
class PartitionAccessPermit(DataAccessContext):
    """Partition-gebundene Zugriffsberechtigung; nur über `HeldOutGuard.issue_permit` erzeugbar.

    Die Berechtigung erlischt, sobald der ausstellende Guard seine Phase wechselt.
    """

    partition: Partition
    model_development_role: ModelDevelopmentRole | None
    phase: Phase
    freeze_id: str | None
    sample_ids: frozenset[str]
    guard: "HeldOutGuard" = field(repr=False)
    issue_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.issue_token is not _ISSUE_TOKEN:
            raise PhaseViolationError("Zugriffsberechtigungen werden nur vom HeldOutGuard erzeugt")

    def require_access(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        if purpose not in _PARTITION_PURPOSES:
            raise PhaseViolationError(
                f"Partitionsberechtigung erlaubt keinen Zugriff zum Zweck {purpose.value}"
            )
        if self.guard.phase is not self.phase:
            raise PhaseViolationError("Zugriffsberechtigung ungültig: Guard-Phase hat gewechselt")
        outside = sorted(set(sample_ids) - self.sample_ids)
        if outside:
            scope = self.partition.value
            if self.model_development_role is not None:
                scope += f"/{self.model_development_role.value}"
            raise PhaseViolationError(f"Zugriff außerhalb der Partition {scope}: {outside}")


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceReconstructionPermit(DataAccessContext):
    """Einmalige FROZEN-Berechtigung für exakt die eingefrorenen Reference-Samples (DEC-084/085).

    Nur über `HeldOutGuard.issue_reference_reconstruction_permit` erzeugbar. Das Objekt trägt nur
    informative Felder; autoritativ sind ausschließlich guardinterne Zustände (P4-03AA): der
    ausstellende Guard (modulprivates Register), die eingefrorene Sample-Menge, die Zwecke INPUTS
    und LABELS, die verbrauchten `(sample_id, purpose)`-Paare und die bei jedem Zugriff vollständig
    neu geprüfte Evaluation-Provenienz. Nicht kopierbar, nicht serialisierbar.
    """

    freeze_id: str
    sample_ids: frozenset[str]
    sample_ids_sha256: str
    purposes: frozenset[AccessPurpose]
    issue_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.issue_token is not _ISSUE_TOKEN:
            raise PhaseViolationError("Zugriffsberechtigungen werden nur vom HeldOutGuard erzeugt")

    def require_access(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        issued = _REFERENCE_PERMITS.get(id(self))
        if issued is None or issued[0] is not self:
            raise PhaseViolationError("Reference-Berechtigung ungültig oder nicht mehr aktiv")
        issued[1].consume_reference_access(self, sample_ids, purpose)

    def __copy__(self) -> "ReferenceReconstructionPermit":
        raise TypeError("ReferenceReconstructionPermit ist nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "ReferenceReconstructionPermit":
        raise TypeError("ReferenceReconstructionPermit ist nicht kopierbar")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("ReferenceReconstructionPermit ist nicht serialisierbar")


_REFERENCE_PERMITS: dict[int, tuple[ReferenceReconstructionPermit, "HeldOutGuard"]] = {}
"""Ausgestellte Reference-Berechtigung → ausstellender Guard (Originalobjekt, nie herausgegeben)."""


class CalibrationAccessScope(StrEnum):
    """Getrennte Calibration-Berechtigungen (CALIBRATION_PROTOCOL_BASELINE 3, 13; DEC-059)."""

    STRUCTURAL_CALIBRATION_ACCESS = "STRUCTURAL_CALIBRATION_ACCESS"
    CAL_REF_ACCESS = "CAL_REF_ACCESS"
    CAL_TUNE_ACCESS = "CAL_TUNE_ACCESS"
    CAL_AUDIT_STRUCTURAL_ACCESS = "CAL_AUDIT_STRUCTURAL_ACCESS"
    CAL_AUDIT_SIGNAL_ACCESS = "CAL_AUDIT_SIGNAL_ACCESS"


CALIBRATION_SCOPE_PURPOSES: dict[CalibrationAccessScope, frozenset[AccessPurpose]] = {
    # Strukturell: nur IDs und Diagnoseklasse (Stratifizierungslabel), keine Bilder/Metadaten.
    CalibrationAccessScope.STRUCTURAL_CALIBRATION_ACCESS: frozenset(
        {AccessPurpose.STRATIFICATION_LABELS}
    ),
    CalibrationAccessScope.CAL_REF_ACCESS: _PARTITION_PURPOSES,
    CalibrationAccessScope.CAL_TUNE_ACCESS: _PARTITION_PURPOSES,
    CalibrationAccessScope.CAL_AUDIT_STRUCTURAL_ACCESS: frozenset(
        {AccessPurpose.STRATIFICATION_LABELS}
    ),
    # Nach Fusion-Freeze: nur Bilder für G1–G3; keine Labels (kein G4), keine Metadaten.
    CalibrationAccessScope.CAL_AUDIT_SIGNAL_ACCESS: frozenset({AccessPurpose.INPUTS}),
}


class CalibrationPermitIssuer(Protocol):
    """Besitzer des Lebenszyklus von Calibration-Berechtigungen (der Calibration-Prozess)."""

    def check_permit(self, permit: "CalibrationAccessPermit") -> None: ...


@dataclass(frozen=True, slots=True, eq=False)
class CalibrationAccessPermit(DataAccessContext):
    """Prozess-, zustands- und pool-gebundene CALIBRATION-Berechtigung (P4-01, P4-02B).

    Nur über `HeldOutGuard.issue_calibration_permit` im Auftrag eines Calibration-Prozesses
    erzeugbar. Gültig ist ausschließlich genau dieses beim Prozess registrierte Objekt, solange es
    nicht widerrufen wurde; Kopien und Serialisierung sind unzulässig.
    """

    run_id: str
    scope: CalibrationAccessScope
    epoch: int
    pool_manifest_sha256: str | None
    sample_ids: frozenset[str]
    sample_ids_sha256: str
    purposes: frozenset[AccessPurpose]
    phase: Phase
    issuer: CalibrationPermitIssuer = field(repr=False)
    guard: "HeldOutGuard" = field(repr=False)
    issue_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.issue_token is not _ISSUE_TOKEN:
            raise PhaseViolationError("Zugriffsberechtigungen werden nur vom HeldOutGuard erzeugt")

    def require_access(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        if purpose not in self.purposes:
            raise PhaseViolationError(
                f"{self.scope.value} erlaubt keinen Zugriff zum Zweck {purpose.value}"
            )
        if self.guard.phase is not self.phase:
            raise PhaseViolationError("Zugriffsberechtigung ungültig: Guard-Phase hat gewechselt")
        self.issuer.check_permit(self)
        outside = sorted(set(sample_ids) - self.sample_ids)
        if outside:
            raise PhaseViolationError(f"Zugriff außerhalb von {self.scope.value}: {outside}")

    def __copy__(self) -> "CalibrationAccessPermit":
        raise TypeError("CalibrationAccessPermit ist prozessgebunden und nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "CalibrationAccessPermit":
        raise TypeError("CalibrationAccessPermit ist prozessgebunden und nicht kopierbar")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("CalibrationAccessPermit ist prozessgebunden und nicht serialisierbar")


class CalibrationIssuerCapability:
    """Opake, vom Guard ausgegebene Fähigkeit des einzig gebundenen Calibration-Prozesses."""

    __slots__ = ("_guard",)

    def __init__(self, guard: "HeldOutGuard", issue_token: object) -> None:
        if issue_token is not _ISSUE_TOKEN:
            raise PhaseViolationError("Issuer-Fähigkeiten werden nur vom HeldOutGuard erzeugt")
        object.__setattr__(self, "_guard", guard)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("CalibrationIssuerCapability ist unveränderlich")

    def __copy__(self) -> "CalibrationIssuerCapability":
        raise TypeError("CalibrationIssuerCapability ist nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "CalibrationIssuerCapability":
        raise TypeError("CalibrationIssuerCapability ist nicht kopierbar")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("CalibrationIssuerCapability ist nicht serialisierbar")


class SplitPermitState(StrEnum):
    NEVER_ISSUED = "NEVER_ISSUED"
    ISSUED = "ISSUED"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True, slots=True, eq=False)
class SplitGenerationPermit(DataAccessContext):
    """Einmaliger Label-Zugriff ausschließlich zur stratifizierten Split-Erzeugung.

    Erlaubt nur `AccessPurpose.STRATIFICATION_LABELS` (keine Bilder, keine normalen Labels,
    keine Metadaten). Pro Guard wird genau ein Exemplar ausgestellt; gültig ist nur dieses
    registrierte Objekt (Kopien nicht), und nur bis `consume()` oder bis der Guard eine
    Partitionsberechtigung ausstellt. Danach ist der Split festgelegt.
    """

    sample_ids: frozenset[str]
    guard: "HeldOutGuard" = field(repr=False)
    issue_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self.issue_token is not _ISSUE_TOKEN:
            raise PhaseViolationError("Zugriffsberechtigungen werden nur vom HeldOutGuard erzeugt")

    def require_access(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        if purpose is not AccessPurpose.STRATIFICATION_LABELS:
            raise PhaseViolationError(
                f"Split-Erzeugungsberechtigung erlaubt nur Stratifizierungslabels, "
                f"nicht {purpose.value}"
            )
        if not self.guard._is_active_split_permit(self):
            raise PhaseViolationError("Split-Erzeugungsberechtigung ist erloschen")
        outside = sorted(set(sample_ids) - self.sample_ids)
        if outside:
            raise PhaseViolationError(f"Zugriff außerhalb der Split-Erzeugung: {outside}")

    def consume(self) -> None:
        """Verbraucht die Berechtigung; der Guard stellt danach keine weitere aus."""
        self.guard._consume_split_permit()

    def __copy__(self) -> "SplitGenerationPermit":
        raise TypeError("SplitGenerationPermit ist einmalig und nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "SplitGenerationPermit":
        raise TypeError("SplitGenerationPermit ist einmalig und nicht kopierbar")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("SplitGenerationPermit ist einmalig und nicht serialisierbar")


class HeldOutGuard:
    """Zustandsbehafteter Guard; Kopien würden den Lebenszyklus duplizieren und sind unzulässig."""

    def __copy__(self) -> "HeldOutGuard":
        raise TypeError("HeldOutGuard ist nicht kopierbar")

    def __deepcopy__(self, memo: dict[int, object]) -> "HeldOutGuard":
        raise TypeError("HeldOutGuard ist nicht kopierbar")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("HeldOutGuard ist nicht serialisierbar")

    def __init__(self) -> None:
        self._phase = Phase.DEVELOPMENT_CALIBRATION
        self._manifest: AnyFreezeManifest | None = None
        self._split_locked = False
        self._split_permit_state = SplitPermitState.NEVER_ISSUED
        self._active_split_permit: SplitGenerationPermit | None = None
        self._calibration_protocol_active = False
        self._calibration_issuer: CalibrationPermitIssuer | None = None
        self._calibration_capability: CalibrationIssuerCapability | None = None
        self._reference_permit: ReferenceReconstructionPermit | None = None
        self._reference_permit_issued = False
        self._reference_sample_ids: frozenset[str] = frozenset()
        self._reference_provenance: EvaluationProvenance | None = None
        # Evaluation-Commit der Reference-Phase; HELD_OUT muss denselben Commit verwenden.
        self._reference_evaluation_commit: str | None = None
        self._reference_consumed: set[tuple[str, AccessPurpose]] = set()
        # Serialisiert Ausstellung, Verbrauch, Widerruf und FROZEN→HELD_OUT (P4-03AC).
        self._reference_lock = threading.RLock()

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def split_locked(self) -> bool:
        """True, sobald der Split festgelegt ist (verbraucht oder Partitionszugriff erteilt)."""
        return self._split_locked

    @property
    def split_permit_state(self) -> SplitPermitState:
        return self._split_permit_state

    def _is_active_split_permit(self, permit: SplitGenerationPermit) -> bool:
        return (
            self._phase is Phase.DEVELOPMENT_CALIBRATION
            and not self._split_locked
            and self._split_permit_state is SplitPermitState.ISSUED
            and self._active_split_permit is permit
        )

    def _consume_split_permit(self) -> None:
        self._split_permit_state = SplitPermitState.CONSUMED
        self._active_split_permit = None
        self._split_locked = True

    def issue_split_generation_permit(self, sample_ids: Iterable[str]) -> SplitGenerationPermit:
        """Stellt pro Guard genau einmal eine Split-Erzeugungsberechtigung aus."""
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION or self._split_locked:
            raise PhaseViolationError(
                "Split-Erzeugung nur in DEVELOPMENT_CALIBRATION vor jeder Partitionsberechtigung"
            )
        if self._split_permit_state is not SplitPermitState.NEVER_ISSUED:
            raise PhaseViolationError(
                "Split-Erzeugungsberechtigung wurde bereits ausgestellt "
                f"({self._split_permit_state.value}); pro Guard nur einmal"
            )
        permit = SplitGenerationPermit(
            sample_ids=frozenset(sample_ids), guard=self, issue_token=_ISSUE_TOKEN
        )
        self._split_permit_state = SplitPermitState.ISSUED
        self._active_split_permit = permit
        return permit

    def require_calibration_phase(self) -> None:
        """Muss von jeder Calibration-/Fit-Operation vor ihrer Ausführung aufgerufen werden."""
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION:
            raise PhaseViolationError(
                f"Calibration-/Update-Operationen sind in {self._phase.value} unzulässig"
            )

    def assert_partition_access(self, partition: Partition) -> None:
        if self._phase is Phase.FROZEN:
            raise PhaseViolationError("Im Zustand FROZEN ist kein Datenzugriff vorgesehen")
        if self._phase is Phase.DEVELOPMENT_CALIBRATION and partition is Partition.HELD_OUT:
            raise PhaseViolationError("Calibration darf die HELD_OUT-Partition nicht öffnen")
        if self._phase is Phase.HELD_OUT_EVALUATION and partition is not Partition.HELD_OUT:
            raise PhaseViolationError("Die Held-out-Evaluation greift nur auf HELD_OUT zu")

    def issue_permit(
        self, partition: Partition, split_manifest: SplitManifest
    ) -> PartitionAccessPermit:
        """Stellt eine Zugriffsberechtigung für genau eine Partition aus."""
        self.assert_partition_access(partition)
        if partition is Partition.CALIBRATION and self._calibration_protocol_active:
            raise PhaseViolationError(
                "Während des Calibration-Protokolls nur scope-gebundene Calibration-Berechtigungen"
            )
        validate_split_manifest(split_manifest)
        if self._phase is Phase.HELD_OUT_EVALUATION:
            if self._manifest is None:
                raise PhaseViolationError("Held-out-Zugriff erfordert einen gültigen Freeze")
            verify_freeze_manifest(self._manifest)
            if split_manifest_sha256(split_manifest) != self._manifest.split_manifest_sha256:
                raise IntegrityError("Split-Manifest weicht vom eingefrorenen Stand ab")
        self._split_locked = True
        return PartitionAccessPermit(
            partition=partition,
            model_development_role=None,
            phase=self._phase,
            freeze_id=None if self._manifest is None else self._manifest.freeze_id,
            sample_ids=partition_sample_ids(split_manifest, partition),
            guard=self,
            issue_token=_ISSUE_TOKEN,
        )

    def issue_model_development_permit(
        self, role: ModelDevelopmentRole, split_manifest: SplitManifest
    ) -> PartitionAccessPermit:
        """Berechtigung nur für TRAIN oder nur für MODEL_VALIDATION innerhalb MODEL_DEVELOPMENT."""
        self.assert_partition_access(Partition.MODEL_DEVELOPMENT)
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION:
            raise PhaseViolationError("Model Development nur in DEVELOPMENT_CALIBRATION")
        validate_split_manifest(split_manifest)
        self._split_locked = True
        return PartitionAccessPermit(
            partition=Partition.MODEL_DEVELOPMENT,
            model_development_role=role,
            phase=self._phase,
            freeze_id=None,
            sample_ids=frozenset(
                a.sample_id
                for a in split_manifest.assignments
                if a.partition is Partition.MODEL_DEVELOPMENT and a.model_development_role is role
            ),
            guard=self,
            issue_token=_ISSUE_TOKEN,
        )

    def bind_calibration_process(
        self, process: CalibrationPermitIssuer, construction_token: object
    ) -> CalibrationIssuerCapability:
        """Bindet genau einen echten `CalibrationProcess` (exakter Typ + Konstruktions-Token).

        Kein Duck-Typing und kein „erster Aufrufer gewinnt“: Akzeptiert wird ausschließlich ein
        Objekt exakt der Klasse `CalibrationProcess`, das sich aus dessen Konstruktor heraus mit
        dem modulinternen Token bindet. Ein zweiter Prozess wird abgelehnt.
        """
        from vigiscoutx.experiment.calibration import process as process_module

        if (
            type(process) is not process_module.CalibrationProcess
            or construction_token is not process_module.CONSTRUCTION_TOKEN
        ):
            raise PhaseViolationError("Nur ein echter CalibrationProcess darf gebunden werden")
        if self._calibration_issuer is not None:
            raise PhaseViolationError("Guard ist bereits an einen Calibration-Prozess gebunden")
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION:
            raise PhaseViolationError("Calibration-Prozess nur in DEVELOPMENT_CALIBRATION")
        capability = CalibrationIssuerCapability(self, _ISSUE_TOKEN)
        self._calibration_issuer = process
        self._calibration_capability = capability
        return capability

    def issue_calibration_permit(
        self,
        *,
        issuer: CalibrationPermitIssuer,
        capability: CalibrationIssuerCapability,
        run_id: str,
        scope: CalibrationAccessScope,
        epoch: int,
        pool_manifest_sha256: str | None,
        split_manifest: SplitManifest,
        sample_ids: Iterable[str],
    ) -> CalibrationAccessPermit:
        """Scope-gebundene Berechtigung ausschließlich für CALIBRATION-Samples (P4-01).

        Die Protokollzustandsregeln prüft der ausstellende Calibration-Prozess; der Guard prüft
        Phase und Partition. Danach sind breite `issue_permit(CALIBRATION)` gesperrt.
        """
        self.assert_partition_access(Partition.CALIBRATION)
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION:
            raise PhaseViolationError("Calibration-Zugriff nur in DEVELOPMENT_CALIBRATION")
        validate_split_manifest(split_manifest)
        if (
            self._calibration_issuer is None
            or issuer is not self._calibration_issuer
            or capability is not self._calibration_capability
        ):
            raise PhaseViolationError("Berechtigung nur für den gebundenen Calibration-Prozess")
        requested = frozenset(sample_ids)
        outside = sorted(requested - partition_sample_ids(split_manifest, Partition.CALIBRATION))
        if not requested or outside:
            raise PhaseViolationError(f"Calibration-Berechtigung außerhalb CALIBRATION: {outside}")
        self._split_locked = True
        self._calibration_protocol_active = True
        return CalibrationAccessPermit(
            run_id=run_id,
            scope=scope,
            epoch=epoch,
            pool_manifest_sha256=pool_manifest_sha256,
            sample_ids=requested,
            sample_ids_sha256=canonical_sha256(sorted(requested)),
            purposes=CALIBRATION_SCOPE_PURPOSES[scope],
            phase=self._phase,
            issuer=issuer,
            guard=self,
            issue_token=_ISSUE_TOKEN,
        )

    def freeze(self, manifest: AnyFreezeManifest) -> None:
        if self._phase is not Phase.DEVELOPMENT_CALIBRATION:
            raise PhaseViolationError("Freeze ist nur aus DEVELOPMENT_CALIBRATION möglich")
        verify_freeze_manifest(manifest)
        self._manifest = manifest
        self._phase = Phase.FROZEN

    def issue_reference_reconstruction_permit(
        self,
        *,
        record: CalibrationFreezeRecord,
        split_manifest: SplitManifest,
        sample_ids: Sequence[str],
        provenance: EvaluationProvenance,
    ) -> ReferenceReconstructionPermit:
        """Einmalige Reference-Berechtigung im Zustand `FROZEN` (DEC-084/085).

        Gebunden an das FreezeManifestV2 dieses Guards, den dazu verifizierten
        Calibration-Freeze-Record, das eingefrorene Split-Manifest, exakt dessen
        `reference_sample_ids` (CALIBRATION-Partition, Anzahl = W) und eine erneut geprüfte
        Evaluation-Provenienz mit separatem Release-Nachweis. Andere Zugriffspfade bleiben in
        `FROZEN` gesperrt.
        """
        with self._reference_lock:
            if self._phase is not Phase.FROZEN or self._manifest is None:
                raise PhaseViolationError("Reference-Rekonstruktion nur im Zustand FROZEN")
            if self._reference_permit_issued:
                raise PhaseViolationError(
                    "Reference-Rekonstruktion wurde bereits einmal berechtigt"
                )
            manifest = self._manifest
            verify_freeze_manifest(manifest)
            if type(manifest) is not FreezeManifestV2:
                raise IntegrityError(
                    "Reference-Rekonstruktion erfordert FreezeManifest schema_version 2"
                )
            verify_record_against_manifest(record, manifest)
            validate_split_manifest(split_manifest)
            if split_manifest_sha256(split_manifest) != manifest.split_manifest_sha256:
                raise IntegrityError("Split-Manifest weicht vom eingefrorenen Stand ab")
            requested = tuple(sample_ids)
            frozen = frozenset(record.reference_sample_ids)
            if (
                len(requested) != len(set(requested))
                or frozenset(requested) != frozen
                or len(frozen) != manifest.window_size
            ):
                raise IntegrityError(
                    "Berechtigung nur für exakt die eingefrorenen Reference-Samples"
                )
            if not frozen <= partition_sample_ids(split_manifest, Partition.CALIBRATION):
                raise IntegrityError("Reference-Samples liegen nicht vollständig in CALIBRATION")
            if type(provenance) is not EvaluationProvenance:
                raise IntegrityError("Reference-Rekonstruktion erfordert eine geprüfte Provenienz")
            if provenance.purpose is not ExecutionPurpose.REFERENCE_RECONSTRUCTION:
                raise IntegrityError(
                    "Reference-Rekonstruktion erfordert eine Freigabe für REFERENCE_RECONSTRUCTION"
                )
            if (provenance.calibration_commit, provenance.freeze_id) != (
                manifest.git_commit,
                manifest.freeze_id,
            ):
                raise IntegrityError("Provenienz bezieht sich nicht auf diesen Freeze")
            reverify_evaluation_provenance(
                provenance, calibration_commit=manifest.git_commit, freeze_id=manifest.freeze_id
            )
            permit = ReferenceReconstructionPermit(
                freeze_id=manifest.freeze_id,
                sample_ids=frozen,
                sample_ids_sha256=canonical_sha256(sorted(frozen)),
                purposes=_REFERENCE_PURPOSES,
                issue_token=_ISSUE_TOKEN,
            )
            self._reference_permit_issued = True
            self._reference_permit = permit
            self._reference_sample_ids = frozen
            self._reference_provenance = provenance
            self._reference_evaluation_commit = provenance.evaluation_commit
            self._reference_consumed = set()
            _REFERENCE_PERMITS[id(permit)] = (permit, self)
            return permit

    def consume_reference_access(
        self,
        permit: ReferenceReconstructionPermit,
        sample_ids: Sequence[str],
        purpose: AccessPurpose,
    ) -> None:
        """Prüft vollständig und registriert erst danach den Verbrauch je Sample/Zweck.

        Die gesamte Operation läuft unter der guardinternen Sperre (P4-03AC): parallele Zugriffe
        auf dasselbe `(sample_id, purpose)` werden serialisiert, genau einer wird verbraucht.

        Autoritativ sind nur guardinterne Zustände; bei jedem Zugriff wird die beim Ausstellen
        hinterlegte Provenienz vollständig neu geprüft (Freeze-Commit, Freeze-ID, ausgeführtes
        Repository, HEAD, Working Tree, separater Release-Nachweis).
        """
        with self._reference_lock:
            if (
                self._phase is not Phase.FROZEN
                or self._reference_permit is not permit
                or self._manifest is None
                or self._reference_provenance is None
            ):
                raise PhaseViolationError("Reference-Berechtigung ungültig oder nicht mehr aktiv")
            if purpose not in _REFERENCE_PURPOSES:
                raise PhaseViolationError(
                    f"Reference-Rekonstruktion erlaubt keinen Zugriff zum Zweck {purpose.value}"
                )
            requested = list(sample_ids)
            if len(requested) != len(set(requested)):
                raise PhaseViolationError("Reference-Rekonstruktion: doppelte sample_ids")
            outside = sorted(set(requested) - self._reference_sample_ids)
            if outside:
                raise PhaseViolationError(
                    f"Zugriff außerhalb der eingefrorenen Reference: {outside}"
                )
            keys = {(sample, purpose) for sample in requested}
            repeated = sorted(sample for sample, _ in keys & self._reference_consumed)
            if repeated:
                raise PhaseViolationError(
                    f"Reference-Rekonstruktion: zweiter Zugriff ({purpose.value}) auf {repeated}"
                )
            manifest = self._manifest
            reverify_evaluation_provenance(
                self._reference_provenance,
                calibration_commit=manifest.git_commit,
                freeze_id=manifest.freeze_id,
            )
            self._reference_consumed |= keys

    def close_reference_reconstruction(self) -> None:
        """Beendet die Reference-Berechtigung endgültig (keine Neuausstellung)."""
        self._revoke_reference_permit()

    def _revoke_reference_permit(self) -> None:
        # Wartet auf einen laufenden Verbrauch; danach scheitert jeder weitere Zugriff (P4-03AC).
        with self._reference_lock:
            if self._reference_permit is not None:
                _REFERENCE_PERMITS.pop(id(self._reference_permit), None)
            self._reference_permit = None
            self._reference_provenance = None

    def open_held_out(
        self, config: RunConfig, observed: ObservedHashes, provenance: EvaluationProvenance
    ) -> HeldOutEvaluationSession:
        """Öffnet HELD_OUT nur bei gültigem Freeze, unveränderter Konfiguration, beobachteten
        Hashes und selbst ermittelter Code-Provenienz (P4-03Q, DEC-082).

        Der Freeze-Commit (`manifest.git_commit`) bleibt Calibration-Provenienz; der tatsächliche
        Evaluation-Commit stammt aus `provenance` (über Git ermittelt, Policy-geprüft, separat
        freigegeben) und wird hier erneut gegen das ausgeführte Repository geprüft.
        """
        if self._phase is not Phase.FROZEN or self._manifest is None:
            raise PhaseViolationError("Held-out-Evaluation erfordert einen gültigen Freeze")
        manifest = self._manifest
        verify_freeze_manifest(manifest)
        if type(provenance) is not EvaluationProvenance:
            raise IntegrityError("Held-out-Evaluation erfordert eine geprüfte EvaluationProvenance")
        if (
            provenance.purpose is not ExecutionPurpose.HELD_OUT_EVALUATION
            or provenance.reference_evidence_sha256 is None
        ):
            raise IntegrityError(
                "Held-out-Evaluation erfordert eine Freigabe für HELD_OUT_EVALUATION mit "
                "gebundenem Reference-Evidence-Manifest (DEC-087)"
            )
        if self._reference_evaluation_commit not in (None, provenance.evaluation_commit):
            raise IntegrityError(
                "Held-out-Evaluation muss denselben Evaluation-Commit wie die Reference-Phase "
                "verwenden"
            )
        if (provenance.calibration_commit, provenance.freeze_id) != (
            manifest.git_commit,
            manifest.freeze_id,
        ):
            raise IntegrityError(
                "Code-Provenienz bezieht sich nicht auf den Calibration-Commit des Freeze"
            )
        # Neuprüfung unmittelbar vor dem Öffnen (TOCTOU, P4-03S): ausgeführtes Repository, HEAD,
        # Working Tree, Policy, Dateitypen und separater Freigabenachweis.
        reverify_evaluation_provenance(
            provenance, calibration_commit=manifest.git_commit, freeze_id=manifest.freeze_id
        )
        if config.phase is not Phase.HELD_OUT_EVALUATION:
            raise PhaseViolationError("Konfiguration ist keine HELD_OUT_EVALUATION-Konfiguration")
        if config.freeze_id != manifest.freeze_id:
            raise IntegrityError("freeze_id der Konfiguration passt nicht zum FreezeManifest")

        scientific = config.scientific
        expected: dict[str, object] = {
            "config_sha256": manifest.config_sha256,
            "preprocessing_sha256": manifest.preprocessing_sha256,
            "config.dataset.sha256": manifest.dataset_snapshot_sha256,
            "config.split.sha256": manifest.split_manifest_sha256,
            "config.model.sha256": manifest.model_weights_sha256,
            "config.repetitions": manifest.repetitions,
            "config.evaluation_metrics": manifest.evaluation_metrics,
            "observed.dataset_snapshot_sha256": manifest.dataset_snapshot_sha256,
            "observed.split_manifest_sha256": manifest.split_manifest_sha256,
            "observed.model_weights_sha256": manifest.model_weights_sha256,
            "observed.environment_lock_sha256": manifest.environment_lock_sha256,
        }
        actual: dict[str, object] = {
            "config_sha256": scientific_config_sha256(scientific),
            "preprocessing_sha256": canonical_sha256(scientific.preprocessing),
            "config.dataset.sha256": scientific.dataset.sha256,
            "config.split.sha256": scientific.split.sha256,
            "config.model.sha256": scientific.model.sha256,
            "config.repetitions": config.repetitions,
            "config.evaluation_metrics": config.evaluation_metrics,
            "observed.dataset_snapshot_sha256": observed.dataset_snapshot_sha256,
            "observed.split_manifest_sha256": observed.split_manifest_sha256,
            "observed.model_weights_sha256": observed.model_weights_sha256,
            "observed.environment_lock_sha256": observed.environment_lock_sha256,
        }
        mismatches = sorted(k for k in expected if expected[k] != actual[k])
        if mismatches:
            raise IntegrityError(f"Hash-Abweichung gegenüber dem Freeze: {mismatches}")
        if config.seeds not in manifest.seed_list:
            raise IntegrityError("SeedBundle ist nicht Teil der eingefrorenen Seed-Liste")

        with self._reference_lock:  # Widerruf und Zustandswechsel atomar (P4-03AC)
            if self._phase is not Phase.FROZEN:
                raise PhaseViolationError("Held-out-Evaluation erfordert einen gültigen Freeze")
            self._revoke_reference_permit()  # Reference-Berechtigung erlischt mit dem Wechsel
            self._phase = Phase.HELD_OUT_EVALUATION
        return HeldOutEvaluationSession(manifest=manifest, config=config)

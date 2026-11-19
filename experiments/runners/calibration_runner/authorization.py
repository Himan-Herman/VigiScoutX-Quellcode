"""Pre-Run-Authorization-Contract und Trennung realer/synthetischer Ausführung (P4-02F, DEC-072).

Ein realer Calibration-Lauf ist nur mit einem separat freigegebenen
Autorisierungsartefakt (`PreRunAuthorization`, JSON-Datei) zulässig. Es bindet den Execution-Commit,
den vollständigen SHA-256 der realen RunConfig-Vorlage, Dataset-, Split-, Model-Snapshot- und
Gewichts-Hash, den Environment-Lock und die Baseline-Versionen. Die CLI prüft die tatsächlich
beobachteten Werte gegen dieses Artefakt, bevor eine Calibration-Berechtigung ausgestellt wird.
Ein vom Aufrufer gesetztes Flag ersetzt die Freigabe nicht.

Prozesskontrolle, keine kryptografische Signatur: Die Freigabe wird durch ein versioniertes,
separat bereitgestelltes Artefakt nachgewiesen; der Runner erzeugt nie selbst eine Freigabe.

Ausführungsmodi:
- `RealExecution` entsteht ausschließlich über `authorize_real_execution(...)` und verlangt den
  verifizierten `ResNet18ModelAdapter` (Architektur `resnet18`, fine-tuned) und den offiziellen
  `Ham10000Dataset`.
- `SyntheticExecution` ist ausschließlich für Tests und bindet die synthetische Dataset- und
  Modellprovenienz explizit (Snapshot-Hashes). Der Runner prüft sie gegen die tatsächlichen
  Komponenten (`require_synthetic_components`): kein `ResNet18ModelAdapter`, keine
  `resnet18`-Architektur, nie der eingefrorene reale HAM10000-Snapshot, und der geparste
  Datensatz muss inhaltlich als synthetische Fixture markiert sein (Metadatenspalte
  `SYNTHETIC_FIXTURE_COLUMN`, die der reale Datensatz nicht besitzt). Ein synthetischer Lauf
  trägt die Kennzeichnung `SYNTHETIC` samt Bindung im ArtifactManifest und kann keine reale
  Freigabe erfüllen.
"""

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import StrictBool

from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.adapters.resnet18 import ResNet18ModelAdapter, TrainingStatus
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, GitCommitHash, Identifier, Sha256Hex
from vigiscoutx.errors import IntegrityError, PhaseViolationError

PROTOCOL_BASELINE_VERSION: Final = "calibration-protocol-baseline:1.0"
ORCHESTRATION_BASELINE_VERSION: Final = "calibration-orchestration-baseline:1.0"
MONITORING_METHODS_VERSION: Final = "monitoring-methods-baseline:1.2"
REAL_MODEL_ID: Final = "resnet18-pilot-P1"
REAL_HAM10000_SNAPSHOT_SHA256: Final = (
    "e4e91b1e651734fd007efa469fcaf026533de189c37ce9a0b22ae88483737f6f"
)
"""Eingefrorener realer Dataset-Snapshot (`REAL_DATA_PREFLIGHT.md`); nie im synthetischen Modus."""
SYNTHETIC_FIXTURE_COLUMN: Final = "vigiscoutx_synthetic_fixture"
"""Inhaltliche Markierung synthetischer HAM10000-Fixtures (Metadatenspalte, Teil des Snapshot-
Hashes); der reale HAM10000-Datensatz besitzt diese Spalte nicht."""
_TOKEN = object()


class AuthorizationError(IntegrityError):
    """Fehlende, nicht freigegebene oder abweichende Pre-Run-Autorisierung."""


class PreRunAuthorization(FrozenModel):
    """Separat freizugebendes Autorisierungsartefakt für genau einen realen Lauf."""

    schema_version: Literal[1]
    authorization_id: Identifier
    author_release: Literal["AUTHOR_APPROVED"]
    run_id: Identifier
    execution_git_commit: GitCommitHash
    run_config_template_sha256: Sha256Hex
    dataset_snapshot_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    model_id: Literal["resnet18-pilot-P1"]
    model_snapshot_sha256: Sha256Hex
    model_weights_sha256: Sha256Hex
    environment_lock_sha256: Sha256Hex
    calibration_protocol_version: Literal["calibration-protocol-baseline:1.0"]
    orchestration_baseline_version: Literal["calibration-orchestration-baseline:1.0"]
    monitoring_methods_version: Literal["monitoring-methods-baseline:1.2"]

    def authorization_sha256(self) -> str:
        return canonical_sha256(self)


class ObservedExecution(FrozenModel):
    """Vor dem Lauf tatsächlich beobachtete Identitäten (nicht vom Artefakt übernommen)."""

    run_id: Identifier
    git_commit: GitCommitHash
    working_tree_clean: StrictBool
    run_config_template_sha256: Sha256Hex
    dataset_snapshot_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    model_id: Identifier
    model_architecture: Identifier
    model_snapshot_sha256: Sha256Hex
    model_weights_sha256: Sha256Hex
    environment_lock_sha256: Sha256Hex
    calibration_protocol_version: Identifier
    orchestration_baseline_version: Identifier
    monitoring_methods_version: Identifier


@dataclass(frozen=True, slots=True)
class SyntheticExecution:
    """Ausschließlich synthetische Tests; bindet die erwartete synthetische Provenienz."""

    dataset_snapshot_sha256: str
    model_snapshot_sha256: str
    mode: Literal["SYNTHETIC"] = "SYNTHETIC"


@dataclass(frozen=True, slots=True, eq=False)
class RealExecution:
    """Nur über `authorize_real_execution` erzeugbar."""

    authorization: PreRunAuthorization
    observed: ObservedExecution
    token: object

    def __post_init__(self) -> None:
        if self.token is not _TOKEN:
            raise PhaseViolationError("RealExecution nur über authorize_real_execution")

    @property
    def mode(self) -> Literal["REAL"]:
        return "REAL"


type ExecutionMode = SyntheticExecution | RealExecution


def _pairs(
    authorization: PreRunAuthorization, observed: ObservedExecution
) -> dict[str, tuple[object, object]]:
    return {
        "run_id": (authorization.run_id, observed.run_id),
        "execution_git_commit": (authorization.execution_git_commit, observed.git_commit),
        "run_config_template_sha256": (
            authorization.run_config_template_sha256,
            observed.run_config_template_sha256,
        ),
        "dataset_snapshot_sha256": (
            authorization.dataset_snapshot_sha256,
            observed.dataset_snapshot_sha256,
        ),
        "split_manifest_sha256": (
            authorization.split_manifest_sha256,
            observed.split_manifest_sha256,
        ),
        "model_id": (authorization.model_id, observed.model_id),
        "model_architecture": ("resnet18", observed.model_architecture),
        "model_snapshot_sha256": (
            authorization.model_snapshot_sha256,
            observed.model_snapshot_sha256,
        ),
        "model_weights_sha256": (authorization.model_weights_sha256, observed.model_weights_sha256),
        "environment_lock_sha256": (
            authorization.environment_lock_sha256,
            observed.environment_lock_sha256,
        ),
        "calibration_protocol_version": (
            authorization.calibration_protocol_version,
            observed.calibration_protocol_version,
        ),
        "orchestration_baseline_version": (
            authorization.orchestration_baseline_version,
            observed.orchestration_baseline_version,
        ),
        "monitoring_methods_version": (
            authorization.monitoring_methods_version,
            observed.monitoring_methods_version,
        ),
    }


def require_matching(
    authorization: PreRunAuthorization, observed: ObservedExecution, keys: frozenset[str] | None
) -> None:
    """Vergleicht die (ausgewählten) Freigabewerte mit den beobachteten Werten."""
    if not observed.working_tree_clean:
        raise AuthorizationError("Working Tree nicht sauber")
    pairs = _pairs(authorization, observed)
    selected = pairs if keys is None else {k: v for k, v in pairs.items() if k in keys}
    mismatches = sorted(k for k, (want, have) in selected.items() if want != have)
    if mismatches:
        raise AuthorizationError(f"Pre-Run-Autorisierung weicht ab: {mismatches}")


DATASET_INDEPENDENT_KEYS: Final = frozenset(
    {
        "run_id",
        "execution_git_commit",
        "run_config_template_sha256",
        "split_manifest_sha256",
        "model_id",
        "model_architecture",
        "model_snapshot_sha256",
        "model_weights_sha256",
        "environment_lock_sha256",
        "calibration_protocol_version",
        "orchestration_baseline_version",
        "monitoring_methods_version",
    }
)
"""Vor dem Laden des Datensatzes prüfbare Freigabewerte (CLI-Vorprüfung)."""


def authorize_real_execution(
    authorization: PreRunAuthorization, observed: ObservedExecution
) -> RealExecution:
    """Jede Abweichung zwischen Freigabe und Beobachtung ist ein harter Fehler."""
    require_matching(authorization, observed, None)
    return RealExecution(authorization=authorization, observed=observed, token=_TOKEN)


def require_real_components(
    execution: RealExecution, model: object, dataset: object, audit_dataset: object
) -> None:
    """Realer Modus: ausschließlich der verifizierte Adapter und der offizielle Datensatz."""
    if type(model) is not ResNet18ModelAdapter:
        raise AuthorizationError("Realer Modus erfordert ResNet18ModelAdapter.from_snapshot(...)")
    if type(dataset) is not Ham10000Dataset or audit_dataset is not dataset:
        raise AuthorizationError("Realer Modus erfordert genau einen offiziellen Ham10000Dataset")
    manifest = model.manifest
    ref = model.snapshot_ref()
    observed = execution.observed
    checks = {
        "model_id": (manifest.model_id, execution.authorization.model_id),
        "architecture": (manifest.architecture, "resnet18"),
        "training_status": (manifest.training_status, TrainingStatus.FINE_TUNED),
        "model_snapshot_sha256": (ref.sha256, observed.model_snapshot_sha256),
        "model_weights_sha256": (manifest.weights_sha256, observed.model_weights_sha256),
        "dataset_snapshot_sha256": (
            dataset.snapshot_ref().sha256,
            observed.dataset_snapshot_sha256,
        ),
    }
    mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
    if mismatches:
        raise AuthorizationError(f"Reale Komponenten weichen von der Freigabe ab: {mismatches}")


def require_synthetic_components(
    execution: SyntheticExecution, model: object, dataset: object, audit_dataset: object
) -> None:
    """Synthetischer Modus nur mit nachweisbar synthetischer Dataset- und Modellprovenienz."""
    if type(model) is ResNet18ModelAdapter:
        raise AuthorizationError("Synthetischer Modus akzeptiert keinen ResNet18ModelAdapter")
    snapshot_ref = getattr(model, "snapshot_ref", None)
    if snapshot_ref is None:
        raise AuthorizationError("Synthetisches Modell ohne Snapshot-Identität")
    model_ref = snapshot_ref()
    if model_ref.architecture == "resnet18":
        raise AuthorizationError("Synthetischer Modus akzeptiert keine resnet18-Architektur")
    if type(audit_dataset) is not Ham10000Dataset:
        raise AuthorizationError("Synthetischer Modus erfordert einen geprüften Fixture-Datensatz")
    refs = [audit_dataset.snapshot_ref(), getattr(dataset, "snapshot_ref", lambda: None)()]
    if any(ref is None or ref.sha256 == REAL_HAM10000_SNAPSHOT_SHA256 for ref in refs):
        raise AuthorizationError("Synthetischer Modus akzeptiert nie den realen HAM10000-Snapshot")
    if SYNTHETIC_FIXTURE_COLUMN not in audit_dataset.snapshot_info().metadata_columns:
        raise AuthorizationError("Datensatz ist nicht inhaltlich als synthetische Fixture markiert")
    declared = (execution.dataset_snapshot_sha256, execution.model_snapshot_sha256)
    actual = (audit_dataset.snapshot_ref().sha256, model_ref.sha256)
    if declared != actual or refs[1] != refs[0]:
        raise AuthorizationError("Synthetische Fixture-Provenienz ist inkonsistent")

"""Referenzen auf eingefrorene externe Artefakte sowie Seeds."""

from typing import Literal

from pydantic import Field

from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, Sha256Hex


class DatasetSnapshotRef(FrozenModel):
    snapshot_id: Identifier
    uri: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex


class SplitManifestRef(FrozenModel):
    manifest_id: Identifier
    uri: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex


class ModelSnapshotRef(FrozenModel):
    model_id: Identifier
    architecture: Identifier
    uri: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex


class SampleRef(FrozenModel):
    """Beobachtung mit Leakage-Gruppe; im Demonstrator ist `group_id` die `lesion_id`."""

    sample_id: Identifier
    group_id: Identifier


PERFORMANCE_EVIDENCE = "performance_assessment"

EvidenceKind = Literal[
    "signal_result",
    "group_assessment",
    "performance_assessment",
    "artifact",
    "primary_risk_assessment",
    "primary_risk_failure",
    "label_aware_assessment",
    "label_aware_failure",
    "relevance_config",
]

LABEL_AWARE_EVIDENCE: frozenset[str] = frozenset(
    {"performance_assessment", "label_aware_assessment", "label_aware_failure", "relevance_config"}
)
"""Evidenzarten des label-aware Pfads; im Primary-Pfad unzulässig (DEC-030, DEC-076)."""


class EvidenceRef(FrozenModel):
    kind: EvidenceKind
    ref_id: Identifier


def reject_performance_evidence(refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    """Primary-Ergebnisse dürfen keine G4-/Performance-Evidenz referenzieren (DEC-030)."""
    if any(ref.kind in LABEL_AWARE_EVIDENCE for ref in refs):
        raise ValueError("Performance-Evidenz (G4) ist im Primary-Pfad unzulässig")
    return refs


# --- Auflösbare Ergebnisreferenzen (P4-03B, DEC-076) ------------------------------------------
#
# Schema `"<kind>:<window_id>:<local_id>"`: `kind` bestimmt den Ergebnistyp, `window_id` das
# Monitoring-Window, `local_id` das Ergebnis innerhalb des Windows (Gruppe, Signal-ID oder
# Analyseebene). Die Relevanzkonfiguration ist windowunabhängig: `"relevance_config:<sha256>"`.
# Damit ist jede Referenz deterministisch und innerhalb eines `MonitoringRunResult` eindeutig.

PRIMARY_LOCAL_ID = "primary"
SECONDARY_LOCAL_ID = "secondary"
PERFORMANCE_LOCAL_ID = "G4"


def window_evidence(kind: EvidenceKind, window_id: str, local_id: str) -> EvidenceRef:
    if kind == "relevance_config" or kind == "artifact":
        raise ValueError(f"{kind} ist keine windowgebundene Ergebnisreferenz")
    return EvidenceRef(kind=kind, ref_id=f"{kind}:{window_id}:{local_id}")


def relevance_evidence(relevance_config_sha256: str) -> EvidenceRef:
    return EvidenceRef(
        kind="relevance_config", ref_id=f"relevance_config:{relevance_config_sha256}"
    )


def parse_window_evidence(ref: EvidenceRef) -> tuple[str, str]:
    """`(window_id, local_id)` einer windowgebundenen Referenz; Abweichung vom Schema = Fehler."""
    prefix = f"{ref.kind}:"
    if not ref.ref_id.startswith(prefix) or ref.kind in ("relevance_config", "artifact"):
        raise ValueError(f"Referenz folgt nicht dem Ergebnisschema: {ref.ref_id}")
    window_id, separator, local_id = ref.ref_id[len(prefix) :].rpartition(":")
    if not separator or not window_id or not local_id:
        raise ValueError(f"Referenz folgt nicht dem Ergebnisschema: {ref.ref_id}")
    return window_id, local_id


class SeedBundle(FrozenModel):
    split_seed: NonNegativeInt
    model_development_seed: NonNegativeInt
    sampling_seed: NonNegativeInt
    shift_seed: NonNegativeInt
    statistical_seed: NonNegativeInt
    repeat_id: NonNegativeInt

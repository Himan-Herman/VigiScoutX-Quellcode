"""Calibration-Freeze-Record: alle Pflichtinhalte des P4-01-Freeze-Gates (DEC-060).

Der Record wird über `FreezeManifest.calibration_record_sha256` an das FreezeManifest gebunden
und vor dem Zustand `FROZEN` gegen dieses geprüft. Er enthält ausschließlich IDs, Hashes und
Parameter – keine Pfade.
"""

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictFloat, model_validator

from vigiscoutx.domain.assessments import ParametersRef
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import (
    FrozenModel,
    GitCommitHash,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
)
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.calibration.compositions import SubgroupExclusion
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    FusionCandidate,
    TechnicalShift,
    baseline_protocol_config,
)
from vigiscoutx.experiment.calibration.statistics import Rate
from vigiscoutx.experiment.freeze import AnyFreezeManifest, FreezeManifestV2

_FUSION_RULES = {
    FusionCandidate.FUSION_A: "A_ANY_PRIMARY_GROUP",
    FusionCandidate.FUSION_B: "B_AT_LEAST_TWO_PRIMARY_GROUPS",
}


class PoolManifestRef(FrozenModel):
    pool: CalibrationPool
    n_lesions: PositiveInt
    sha256: Sha256Hex


class TechnicalSeverityFreeze(FrozenModel):
    shift: TechnicalShift
    candidates: tuple[StrictFloat, ...] = Field(min_length=5, max_length=5)
    selected_indices: tuple[NonNegativeInt, ...] = Field(min_length=3, max_length=3)
    selected_values: tuple[StrictFloat, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _frozen_grid(self) -> Self:
        if self.candidates != baseline_protocol_config().severity_grid(self.shift):
            raise ValueError(f"{self.shift.value}: Kandidatengitter weicht vom P4-01-Gitter ab")
        i1, i2, i3 = self.selected_indices
        if not i1 < i2 < i3 < len(self.candidates):
            raise ValueError("selected_indices müssen i1 < i2 < i3 innerhalb der Kandidaten sein")
        if self.selected_values != tuple(self.candidates[i] for i in self.selected_indices):
            raise ValueError("gewählte Severities müssen Werte des eingefrorenen Gitters sein")
        return self


class ClassShiftFreeze(FrozenModel):
    source: Identifier
    target: Identifier
    d_max: PositiveInt
    levels: tuple[PositiveInt, ...] = Field(min_length=3, max_length=3)
    compositions_sha256: Sha256Hex


class SubgroupFreeze(FrozenModel):
    feasible: StrictBool
    exclusion: SubgroupExclusion | None
    levels: tuple[PositiveInt, ...]
    compositions_sha256: Sha256Hex | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.feasible != (self.exclusion is None):
            raise ValueError("feasible ⇔ keine Ausschlussbegründung")
        if self.feasible != (len(self.levels) == 3 and self.compositions_sha256 is not None):
            raise ValueError("feasible erfordert drei Levels und Kompositions-Hash")
        return self


class CalibrationFreezeRecord(FrozenModel):
    schema_version: Literal[1]
    protocol_version: Literal["calibration-protocol-baseline:1.0"]
    protocol_config_sha256: Sha256Hex
    dataset_snapshot_id: Identifier
    dataset_snapshot_sha256: Sha256Hex
    split_id: Identifier
    split_manifest_sha256: Sha256Hex
    pool_manifest_sha256: Sha256Hex
    pool_manifests: tuple[PoolManifestRef, ...] = Field(min_length=3, max_length=3)
    canonical_rule: Literal["MIN_HASH_TIE_LEXICOGRAPHIC_SAMPLE_ID"]
    canonical_samples_sha256: Sha256Hex
    window_size: Literal[128, 256]
    reference_window_id: Sha256Hex
    reference_lesion_ids: tuple[Identifier, ...]
    reference_sample_ids: tuple[Identifier, ...]
    reference_sha256: Sha256Hex
    mmd_sigma_source_sha256: Sha256Hex
    mmd_sigma: StrictFloat = Field(gt=0.0)
    d0_artifact_sha256: Sha256Hex
    q090: StrictFloat
    q095: StrictFloat
    q099: StrictFloat
    delta_rel: StrictFloat = Field(gt=0.0)
    technical_severities: tuple[TechnicalSeverityFreeze, ...] = Field(min_length=4, max_length=4)
    class_shift: ClassShiftFreeze
    subgroup: SubgroupFreeze
    abrupt_schedules_sha256: Sha256Hex
    gradual_schedules_sha256: Sha256Hex
    final_r: Literal[100, 200, 400]
    seed_derivation_sha256: Sha256Hex
    fusion_candidate: FusionCandidate
    tune_clean_far: Rate
    audit_clean_far: Rate
    model_id: Literal["resnet18-pilot-P1"]
    model_weights_sha256: Sha256Hex
    model_snapshot_sha256: Sha256Hex
    preprocessing_sha256: Sha256Hex
    monitoring_methods_version: Literal["monitoring-methods-baseline:1.2"]
    method_refs: tuple[ParametersRef, ...] = Field(min_length=1)
    config_snapshot_sha256s: tuple[Sha256Hex, ...] = Field(min_length=1)
    git_commit: GitCommitHash
    environment_lock_sha256: Sha256Hex
    artifact_manifest_sha256: Sha256Hex

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.protocol_config_sha256 != baseline_protocol_config().config_sha256():
            raise ValueError("protocol_config_sha256 weicht von der P4-01-Konfiguration ab")
        if {p.pool for p in self.pool_manifests} != set(CalibrationPool):
            raise ValueError("je genau ein Manifest für CAL_REF, CAL_TUNE und CAL_AUDIT")
        if {s.shift for s in self.technical_severities} != set(TechnicalShift):
            raise ValueError("finale Severities für alle vier technischen Familien erforderlich")
        if not (
            len(self.reference_lesion_ids) == len(self.reference_sample_ids) == self.window_size
        ):
            raise ValueError("Reference muss exakt W Läsionen/Samples enthalten")
        if not self.q090 <= self.q095 <= self.q099 or self.delta_rel != self.q095:
            raise ValueError("Quantile geordnet und δ_rel = Q0.95 erforderlich")
        for far in (self.tune_clean_far, self.audit_clean_far):
            if far.trials != self.final_r or far.successes * 20 > far.trials:
                raise ValueError("Clean FAR über R Windows und ≤ 0.05 erforderlich")
        return self

    def record_sha256(self) -> str:
        return canonical_sha256(self)


def verify_record_against_manifest(
    record: CalibrationFreezeRecord, manifest: AnyFreezeManifest
) -> None:
    """Record und FreezeManifest v2 müssen dieselbe Calibration beschreiben; Abweichung = FAIL."""
    if not isinstance(manifest, FreezeManifestV2):
        raise IntegrityError("Der P4-Calibration-Freeze erfordert FreezeManifest schema_version 2")
    expected: dict[str, object] = {
        "calibration_record_sha256": record.record_sha256(),
        "window_size": record.window_size,
        "delta_rel": record.delta_rel,
        "repetitions": record.final_r,
        "fusion_rule": _FUSION_RULES[record.fusion_candidate],
        "dataset_snapshot_sha256": record.dataset_snapshot_sha256,
        "split_manifest_sha256": record.split_manifest_sha256,
        "model_weights_sha256": record.model_weights_sha256,
        "preprocessing_sha256": record.preprocessing_sha256,
        "git_commit": record.git_commit,
        "environment_lock_sha256": record.environment_lock_sha256,
    }
    actual: dict[str, object] = {
        "calibration_record_sha256": manifest.calibration_record_sha256,
        "window_size": manifest.window_size,
        "delta_rel": manifest.relevance.delta_rel,
        "repetitions": manifest.repetitions,
        "fusion_rule": None if manifest.fusion.rule is None else manifest.fusion.rule.value,
        "dataset_snapshot_sha256": manifest.dataset_snapshot_sha256,
        "split_manifest_sha256": manifest.split_manifest_sha256,
        "model_weights_sha256": manifest.model_weights_sha256,
        "preprocessing_sha256": manifest.preprocessing_sha256,
        "git_commit": manifest.git_commit,
        "environment_lock_sha256": manifest.environment_lock_sha256,
    }
    mismatches = sorted(k for k in expected if expected[k] != actual[k])
    if mismatches:
        raise IntegrityError(f"Calibration-Record weicht vom FreezeManifest ab: {mismatches}")

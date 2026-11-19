"""FreezeManifest (ARC-044; DEC-027, DEC-032, DEC-033).

Versionierung: Schema 1 ist das historische Manifest und bleibt unverändert parse- und
hash-verifizierbar. Schema 2 (P4, `DEC-060`) bindet zusätzlich den Calibration-Freeze-Record
über `calibration_record_sha256` und ist für den Calibration-Freeze verpflichtend. Bestehende
v1-Artefakte werden nie migriert oder umgeschrieben.
"""

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from vigiscoutx.config.scientific_config import (
    FusionConfig,
    PolicyRef,
    RelevanceConfig,
    SignalConfig,
    calibration_gaps,
    require_frozen_policy_refs,
    scientific_config_sha256,
    signal_gaps,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.domain.types import FrozenModel, GitCommitHash, PositiveInt, Sha256Hex
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import EvaluationMetric, Phase


class FreezeManifestFields(FrozenModel):
    """Gemeinsamer Inhalt aller Schema-Versionen; `freeze_id` ist der Hash aller übrigen Felder."""

    freeze_id: Sha256Hex
    dataset_snapshot_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    model_weights_sha256: Sha256Hex
    preprocessing_sha256: Sha256Hex
    config_sha256: Sha256Hex
    window_size: PositiveInt
    signals: tuple[SignalConfig, ...] = Field(min_length=1)
    relevance: RelevanceConfig
    fusion: FusionConfig
    risk_policy: PolicyRef
    recommendation_policy: PolicyRef
    repetitions: PositiveInt
    seed_list: tuple[SeedBundle, ...] = Field(min_length=1)
    evaluation_metrics: tuple[EvaluationMetric, ...] = Field(min_length=1)
    git_commit: GitCommitHash
    environment_lock_sha256: Sha256Hex

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.fusion.rule is None:
            raise ValueError("die Fusionsregel muss eingefroren sein")
        if self.relevance.delta_rel is None:
            raise ValueError("delta_rel muss eingefroren sein")
        if len(self.seed_list) != self.repetitions:
            raise ValueError("Anzahl der SeedBundles muss repetitions entsprechen")
        if len({s.repeat_id for s in self.seed_list}) != len(self.seed_list):
            raise ValueError("repeat_id in seed_list muss eindeutig sein")
        if len({s.split_seed for s in self.seed_list}) != 1:
            raise ValueError("split_seed muss in allen Wiederholungen identisch sein")
        if len({s.model_development_seed for s in self.seed_list}) != 1:
            raise ValueError("model_development_seed muss in allen Wiederholungen identisch sein")
        return self


class FreezeManifest(FreezeManifestFields):
    """Historisches Schema 1 (unverändert): jede Signalschwelle muss gesetzt sein."""

    schema_version: Literal[1]

    @model_validator(mode="after")
    def _thresholds(self) -> Self:
        if any(s.threshold is None for s in self.signals):
            raise ValueError("alle Signalschwellen müssen eingefroren sein")
        return self


class FreezeManifestV2(FreezeManifestFields):
    """Schema 2 für den P4-Calibration-Freeze: bindet den Calibration-Freeze-Record.

    Bindet ausschließlich die eingefrorenen Policies `vsx-risk-r0-r3`/`vsx-recommendation-review`
    1.0.0 (DEC-074, DEC-075); unbekannte `PolicyRef`s sind in jedem V2-Pfad ein harter Fehler
    (`PolicyResolutionError`), auch bei formal korrekter `freeze_id`. Schema 1 bleibt unverändert.
    """

    schema_version: Literal[2]
    calibration_record_sha256: Sha256Hex

    @model_validator(mode="after")
    def _frozen_policies(self) -> Self:
        require_frozen_policy_refs(self.risk_policy, self.recommendation_policy)
        return self

    @model_validator(mode="after")
    def _signals_complete(self) -> Self:
        """Dieselbe Semantik wie `calibration_gaps`: methodenentschiedene G1–G3-Signale ohne
        `ThresholdConfig`, alle übrigen Signale mit eingefrorener Schwelle."""
        gaps = signal_gaps(self.signals)
        if gaps:
            raise ValueError(f"Signalkonfiguration nicht eingefroren: {gaps}")
        return self


AnyFreezeManifest = FreezeManifest | FreezeManifestV2


def parse_freeze_manifest(payload: Mapping[str, Any]) -> AnyFreezeManifest:
    """Parst ein Manifest strikt nach seiner `schema_version`; keine stille Migration."""
    version = payload.get("schema_version")
    if version == 1:
        return FreezeManifest.model_validate(payload)
    if version == 2:
        return FreezeManifestV2.model_validate(payload)
    raise ValueError(f"unbekannte FreezeManifest-schema_version: {version!r}")


def compute_freeze_id(manifest: AnyFreezeManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"freeze_id"})
    return canonical_sha256(payload)


def _manifest_fields(
    config: RunConfig,
    *,
    seed_list: tuple[SeedBundle, ...],
    git_commit: str,
    environment_lock_sha256: str,
) -> dict[str, Any]:
    if config.phase is not Phase.DEVELOPMENT_CALIBRATION:
        raise IntegrityError("Freeze nur aus einer DEVELOPMENT_CALIBRATION-Konfiguration")
    scientific = config.scientific
    gaps = calibration_gaps(scientific)
    if gaps:
        raise IntegrityError(f"Freeze unmöglich, offene Calibration-Parameter: {gaps}")
    fields: dict[str, Any] = {
        "schema_version": 1,
        "dataset_snapshot_sha256": scientific.dataset.sha256,
        "split_manifest_sha256": scientific.split.sha256,
        "model_weights_sha256": scientific.model.sha256,
        "preprocessing_sha256": canonical_sha256(scientific.preprocessing),
        "config_sha256": scientific_config_sha256(scientific),
        "window_size": scientific.monitoring.window_size,
        "signals": scientific.monitoring.signals,
        "relevance": scientific.relevance,
        "fusion": scientific.fusion,
        "risk_policy": scientific.risk_policy,
        "recommendation_policy": scientific.recommendation_policy,
        "repetitions": config.repetitions,
        "seed_list": seed_list,
        "evaluation_metrics": config.evaluation_metrics,
        "git_commit": git_commit,
        "environment_lock_sha256": environment_lock_sha256,
    }
    return fields


def build_freeze_manifest(
    config: RunConfig,
    *,
    seed_list: tuple[SeedBundle, ...],
    git_commit: str,
    environment_lock_sha256: str,
) -> FreezeManifest:
    """Erzeugt das (historische) Schema-1-Manifest aus einer kalibrierten Konfiguration."""
    fields = _manifest_fields(
        config,
        seed_list=seed_list,
        git_commit=git_commit,
        environment_lock_sha256=environment_lock_sha256,
    )
    draft = FreezeManifest.model_validate({**fields, "freeze_id": "0" * 64})
    return draft.model_copy(update={"freeze_id": compute_freeze_id(draft)})


def build_freeze_manifest_v2(
    config: RunConfig,
    *,
    seed_list: tuple[SeedBundle, ...],
    git_commit: str,
    environment_lock_sha256: str,
    calibration_record_sha256: str,
) -> FreezeManifestV2:
    """Schema-2-Manifest für den P4-Calibration-Freeze (bindet den Calibration-Freeze-Record)."""
    require_frozen_policy_refs(
        config.scientific.risk_policy, config.scientific.recommendation_policy
    )
    fields = _manifest_fields(
        config,
        seed_list=seed_list,
        git_commit=git_commit,
        environment_lock_sha256=environment_lock_sha256,
    )
    draft = FreezeManifestV2.model_validate(
        {
            **fields,
            "schema_version": 2,
            "calibration_record_sha256": calibration_record_sha256,
            "freeze_id": "0" * 64,
        }
    )
    return draft.model_copy(update={"freeze_id": compute_freeze_id(draft)})


def verify_freeze_manifest(manifest: AnyFreezeManifest) -> None:
    """Prüft, dass das Manifest seit dem Freeze unverändert ist. Abweichung = FAIL.

    Schema 2 prüft zusätzlich die Policy-Bindung, damit auch ein am Validator vorbei erzeugtes
    Manifest mit passender `freeze_id` keine unbekannte Policy legitimiert.
    """
    if isinstance(manifest, FreezeManifestV2):
        require_frozen_policy_refs(manifest.risk_policy, manifest.recommendation_policy)
    expected = compute_freeze_id(manifest)
    if manifest.freeze_id != expected:
        raise IntegrityError(
            f"FreezeManifest verändert: freeze_id {manifest.freeze_id} ≠ berechnet {expected}"
        )

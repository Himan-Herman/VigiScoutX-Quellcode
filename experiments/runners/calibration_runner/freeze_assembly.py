"""Run-Konfiguration, write-once Artefakte und FreezeManifestV2 (P4-02E, Protokollvorgabe).

Der Runner erhält eine separat freigegebene `DEVELOPMENT_CALIBRATION`-RunConfig als Vorlage.
Offen sind darin ausschließlich `relevance.delta_rel`, `fusion.rule` und `sigma` in den
G1-Parametern (`null`). G1–G3 sind methodenentschiedene Signale (`METHOD_DECIDED_SIGNALS`) und
tragen keine `ThresholdConfig`. Der Runner setzt nach der Calibration nur
`monitoring.window_size`, `relevance.delta_rel`, `fusion.rule`, `repetitions = R` und `sigma`.

`seed_list`-Register (Protokollregel P4-02E): für `r = 1…R` ist `seed_list[r] = template.seeds`
mit ausschließlich `repeat_id = r`. Es ist das Run-Zulassungsregister für `open_held_out()`, nicht
das Register der case-spezifischen G1-Permutationsseeds (diese bleiben je Namespace, W,
`scenario_key` und `repeat_id` abgeleitet und in der Case-Provenienz nachgewiesen).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vigiscoutx.artifacts.manifests import ArtifactEntry, ArtifactManifest, canonical_artifact_path
from vigiscoutx.config.scientific_config import FusionRule, SignalConfig, calibration_gaps
from vigiscoutx.domain.assessments import ParametersRef
from vigiscoutx.domain.frozen_json import thaw_json
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.calibration.config import CalibrationProtocolConfig, FusionCandidate
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.calibration.process import CalibrationRunProvenance
from vigiscoutx.experiment.freeze import FreezeManifestV2, build_freeze_manifest_v2
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import Phase
from vigiscoutx.signals.config import (
    G1MmdConfig,
    G2QualityConfig,
    G3PredictiveConfig,
    method_config_from_signal_config,
)
from vigiscoutx.signals.performance.g4 import G4PerformanceConfig

FUSION_RULES = {
    FusionCandidate.FUSION_A: FusionRule.A_ANY_PRIMARY_GROUP,
    FusionCandidate.FUSION_B: FusionRule.B_AT_LEAST_TWO_PRIMARY_GROUPS,
}
EXPECTED_GAPS = frozenset(
    {"relevance.delta_rel", "fusion.rule", "monitoring.signals[G1_MMD_EMBEDDING].method_config"}
)
_G1_METHOD = "MMD_U2_RBF_PERMUTATION"
_G2_METHOD = "G2_QUALITY_KS_BONFERRONI"
_G3_METHOD = "G3_MUKS_CONFIDENCE_KS"


def g4_config() -> G4PerformanceConfig:
    """G4 ist vollständig eingefroren (jedes Feld hat genau einen zulässigen Wert)."""
    return G4PerformanceConfig(
        performance_id="G4_PERFORMANCE",
        method_version="monitoring-methods-baseline:1.2",
        class_order=("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"),
        metric="BALANCED_ACCURACY",
        missing_class_rule="NOT_EVALUABLE_NO_REAVERAGING",
        zero_division="NULL",
    )


@dataclass(frozen=True, slots=True)
class TemplateMethods:
    """Aus der Vorlage validierte Methodenkonfigurationen (G1 ohne σ)."""

    g1_params: Mapping[str, Any]
    g2: G2QualityConfig
    g3: G3PredictiveConfig

    def g1(self, sigma: float) -> G1MmdConfig:
        return G1MmdConfig.model_validate({**self.g1_params, "sigma": sigma})


def _signal(template: RunConfig, method: str, group: SignalGroup) -> SignalConfig:
    matches = [s for s in template.scientific.monitoring.signals if s.method == method]
    if len(matches) != 1 or matches[0].group is not group:
        raise IntegrityError(f"Vorlage benötigt genau ein Signal {method} ({group.value})")
    return matches[0]


def template_methods(template: RunConfig) -> TemplateMethods:
    """Validiert die Signal-/Fusionsstruktur der Vorlage gegen die eingefrorenen Methoden."""
    if template.phase is not Phase.DEVELOPMENT_CALIBRATION or template.freeze_id is not None:
        raise IntegrityError("Vorlage muss eine DEVELOPMENT_CALIBRATION-Konfiguration sein")
    gaps = set(calibration_gaps(template.scientific))
    if gaps != EXPECTED_GAPS:
        raise IntegrityError(f"Vorlage: offen dürfen nur {sorted(EXPECTED_GAPS)} sein: {gaps}")
    scientific = template.scientific
    if len(scientific.monitoring.signals) != 3:
        raise IntegrityError("Vorlage enthält exakt die Signale G1, G2 und G3")
    if any(s.threshold is not None for s in scientific.monitoring.signals):
        raise IntegrityError("Vorlage: methodenentschiedene G1–G3-Signale tragen keine Schwelle")
    if set(scientific.fusion.groups) != {
        SignalGroup.G1_INPUT,
        SignalGroup.G2_DATA_QUALITY,
        SignalGroup.G3_PREDICTIVE,
    }:
        raise IntegrityError("Fusion umfasst exakt G1, G2 und G3")
    g1_signal = _signal(template, _G1_METHOD, SignalGroup.G1_INPUT)
    g1_params = thaw_json(g1_signal.params)
    if (
        not isinstance(g1_params, dict)
        or "sigma" not in g1_params
        or g1_params["sigma"] is not None
    ):
        raise IntegrityError("G1-Parameter der Vorlage müssen sigma = null enthalten")
    methods = TemplateMethods(
        g1_params=g1_params,
        g2=method_config_from_signal_config(
            _signal(template, _G2_METHOD, SignalGroup.G2_DATA_QUALITY), G2QualityConfig
        ),
        g3=method_config_from_signal_config(
            _signal(template, _G3_METHOD, SignalGroup.G3_PREDICTIVE), G3PredictiveConfig
        ),
    )
    methods.g1(1.0)  # strukturelle Prüfung aller übrigen eingefrorenen G1-Felder
    return methods


def validate_template(template: RunConfig, provenance: CalibrationRunProvenance) -> TemplateMethods:
    """Prüft die Vorlage vor jedem Datenzugriff gegen die aktive Run-Provenienz."""
    methods = template_methods(template)
    scientific = template.scientific
    expected: dict[str, tuple[object, object]] = {
        "run_id": (template.run_id, provenance.run_id),
        "config_snapshot": (canonical_sha256(template), provenance.config_snapshot_sha256s[0]),
        "dataset.sha256": (scientific.dataset.sha256, provenance.dataset.sha256),
        "split.sha256": (scientific.split.sha256, provenance.split.sha256),
        "model.sha256": (scientific.model.sha256, provenance.model_weights_sha256),
        "preprocessing_sha256": (
            canonical_sha256(scientific.preprocessing),
            provenance.preprocessing_sha256,
        ),
        "method_refs": (method_refs(methods), provenance.method_refs),
    }
    mismatches = sorted(k for k, (have, want) in expected.items() if have != want)
    if mismatches:
        raise IntegrityError(f"Vorlage passt nicht zur Run-Provenienz: {mismatches}")
    return methods


def method_refs(methods: TemplateMethods) -> tuple[ParametersRef, ...]:
    """Methodenidentitäten der Provenienz (G1 ohne σ ist über den Vorlagen-Hash gebunden; das
    kalibrierte σ steht im Freeze-Record)."""
    return (methods.g2.parameters_ref(), methods.g3.parameters_ref(), g4_config().parameters_ref())


def config_snapshot_hashes(
    template: RunConfig, protocol: CalibrationProtocolConfig
) -> tuple[str, str]:
    """(Vorlage, Protokollkonfiguration) – die Vorlage steht immer an erster Stelle."""
    return (canonical_sha256(template), protocol.config_sha256())


def final_run_config(template: RunConfig, record: CalibrationFreezeRecord) -> RunConfig:
    """Vorlage plus ausschließlich die kalibrierten Werte aus dem Freeze-Record."""
    data = template.model_dump(mode="json")
    scientific = data["scientific"]
    scientific["monitoring"]["window_size"] = record.window_size
    scientific["relevance"]["delta_rel"] = record.delta_rel
    scientific["fusion"]["rule"] = FUSION_RULES[record.fusion_candidate].value
    for signal in scientific["monitoring"]["signals"]:
        if signal["method"] == _G1_METHOD:
            signal["params"]["sigma"] = record.mmd_sigma
    data["repetitions"] = record.final_r
    return RunConfig.model_validate(data)


def template_seed_list(template: RunConfig, final_r: int) -> tuple[SeedBundle, ...]:
    """`seed_list[r] = template.seeds` mit ausschließlich `repeat_id = r`, `r = 1…R`."""
    if final_r < 1:
        raise IntegrityError("R muss positiv sein")
    seeds = tuple(template.seeds.model_copy(update={"repeat_id": r}) for r in range(1, final_r + 1))
    verify_seed_list(template, seeds, final_r)
    return seeds


def verify_seed_list(template: RunConfig, seeds: tuple[SeedBundle, ...], final_r: int) -> None:
    """Exakt R Einträge, aufsteigend `1…R`, alle übrigen Felder identisch zur Vorlage."""
    if [s.repeat_id for s in seeds] != list(range(1, final_r + 1)):
        raise IntegrityError("seed_list muss exakt repeat_id 1…R aufsteigend enthalten")
    fixed = template.seeds.model_dump(exclude={"repeat_id"})
    if any(s.model_dump(exclude={"repeat_id"}) != fixed for s in seeds):
        raise IntegrityError("seed_list weicht außer in repeat_id von der Vorlage ab")


def seed_list_sha256(seeds: tuple[SeedBundle, ...]) -> str:
    """Reproduzierbarer Hash des Zulassungsregisters (kanonisches JSON, Reihenfolge 1…R)."""
    return canonical_sha256([s.model_dump(mode="json") for s in seeds])


def build_manifest(
    template: RunConfig, record: CalibrationFreezeRecord
) -> tuple[RunConfig, FreezeManifestV2]:
    run_config = final_run_config(template, record)
    seed_list = template_seed_list(template, record.final_r)
    manifest = build_freeze_manifest_v2(
        run_config,
        seed_list=seed_list,
        git_commit=record.git_commit,
        environment_lock_sha256=record.environment_lock_sha256,
        calibration_record_sha256=record.record_sha256(),
    )
    return run_config, manifest


class ArtifactWriter:
    """Write-once Ablage unter dem Run-Verzeichnis; vorhandene Dateien werden nie überschrieben."""

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root
        self._run_id = run_id
        self._entries: list[ArtifactEntry] = []

    @property
    def root(self) -> Path:
        return self._root

    def write(self, relative: str, data: bytes, role: str | None = None) -> str:
        path = canonical_artifact_path(relative)
        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        digest = sha256_hex(data)
        if role is not None:
            self._entries.append(ArtifactEntry(path=path, sha256=digest, role=role))
        return digest

    def write_json(self, relative: str, value: object, role: str | None = None) -> str:
        return self.write(relative, canonical_json_bytes(value), role)

    def manifest(self) -> ArtifactManifest:
        return ArtifactManifest(run_id=self._run_id, entries=tuple(self._entries))

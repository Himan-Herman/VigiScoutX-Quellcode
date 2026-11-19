"""Synthetische Fixtures für den Calibration-Runner (keine echten CALIBRATION-Daten)."""

from collections.abc import Mapping, Sequence
from typing import Any

from policy_fixtures import with_frozen_policies
from vigiscoutx.artifacts.manifests import ArtifactEntry, ArtifactManifest
from vigiscoutx.config.scientific_config import PreprocessingConfig
from vigiscoutx.domain.assessments import ParametersRef, SignalStatus
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import DatasetSnapshotRef, SampleRef, SeedBundle
from vigiscoutx.experiment.calibration.config import (
    R_METHODS,
    CalibrationPool,
    FusionCandidate,
    TechnicalShift,
    baseline_protocol_config,
)
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.calibration.process import (
    CalibrationProcess,
    CalibrationRunProvenance,
    CalibrationStep,
)
from vigiscoutx.experiment.calibration.statistics import (
    MethodOutcomes,
    ProcedureOutcome,
    Rate,
    ShiftOutcome,
)
from vigiscoutx.experiment.calibration.structure import CalibrationLesion
from vigiscoutx.experiment.freeze import FreezeManifestV2, build_freeze_manifest_v2
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.splitting import split_manifest_ref
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition
from vigiscoutx.ports import AccessPurpose, DataAccessContext

CONFIG = baseline_protocol_config()
CLASSES = CONFIG.class_order
SPLIT_HASH = "5" * 64
# Synthetische, HAM10000-ähnlich unausgewogene Läsionszahlen; CAL_REF erlaubt W = 128.
LESIONS_PER_CLASS = {"MEL": 70, "NV": 300, "BCC": 30, "AKIEC": 25, "BKL": 70, "DF": 12, "VASC": 13}
# Entspricht der Preprocessing-Konfiguration der conftest-Fixture `calibration_config_data`.
PREPROCESSING_SHA256 = canonical_sha256(
    PreprocessingConfig(
        pipeline_id="prep-test",
        input_height=8,
        input_width=8,
        color_mode="RGB",
        params=FrozenMapping({}),
    )
)


def hex64(char: str) -> str:
    return char * 64


def synthetic_lesions(images_per_lesion: int = 2) -> list[CalibrationLesion]:
    return [
        CalibrationLesion(
            lesion_id=f"L-{cls}-{i:03d}",
            class_id=cls,
            sample_ids=tuple(f"S-{cls}-{i:03d}-{k}" for k in range(images_per_lesion)),
        )
        for cls, n in LESIONS_PER_CLASS.items()
        for i in range(n)
    ]


def synthetic_split(lesions: Sequence[CalibrationLesion]) -> SplitManifest:
    calibration = [
        SplitAssignment(
            sample_id=s,
            group_id=lesion.lesion_id,
            partition=Partition.CALIBRATION,
            model_development_role=None,
        )
        for lesion in lesions
        for s in lesion.sample_ids
    ]
    others = [
        SplitAssignment(
            sample_id="S-HO-1",
            group_id="L-HO-1",
            partition=Partition.HELD_OUT,
            model_development_role=None,
        ),
        SplitAssignment(
            sample_id="S-MD-1",
            group_id="L-MD-1",
            partition=Partition.MODEL_DEVELOPMENT,
            model_development_role=ModelDevelopmentRole.TRAIN,
        ),
    ]
    return SplitManifest(manifest_id="synthetic-split", assignments=(*calibration, *others))


class SyntheticLabelSource:
    """Strukturelle Labelquelle; prüft wie ein echter Adapter die Berechtigung."""

    def __init__(
        self,
        lesions: Sequence[CalibrationLesion],
        dataset: DatasetSnapshotRef | None = None,
        relabel: Mapping[str, str] | None = None,
    ) -> None:
        self._dataset = dataset or DATASET
        relabel = relabel or {}
        self._labels = {
            s: CLASSES.index(lesion.class_id) for lesion in lesions for s in lesion.sample_ids
        }
        self._refs = tuple(
            SampleRef(sample_id=s, group_id=lesion.lesion_id)
            for lesion in lesions
            for s in lesion.sample_ids
        )
        for sample_id, cls in relabel.items():
            self._labels[sample_id] = CLASSES.index(cls)
        self.accessed: list[tuple[str, AccessPurpose]] = []
        self.permits: list[DataAccessContext] = []

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return self._dataset

    def sample_refs(self) -> tuple[SampleRef, ...]:
        return self._refs

    def stratification_labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]:
        self.permits.append(access)
        access.require_access(sample_ids, AccessPurpose.STRATIFICATION_LABELS)
        self.accessed += [(s, AccessPurpose.STRATIFICATION_LABELS) for s in sample_ids]
        return {s: self._labels[s] for s in sample_ids}


DATASET = DatasetSnapshotRef(snapshot_id="ds-test", uri="memory://ds", sha256="1" * 64)
LESIONS = synthetic_lesions()
SPLIT = synthetic_split(LESIONS)
SPLIT_REF = split_manifest_ref(SPLIT)


def provenance(**overrides: Any) -> CalibrationRunProvenance:
    data: dict[str, Any] = {
        "run_id": "cal-run-1",
        "dataset": DATASET,
        "split": SPLIT_REF,
        "model_id": "resnet18-pilot-P1",
        "model_weights_sha256": hex64("3"),
        "model_snapshot_sha256": hex64("e"),
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "method_refs": (ParametersRef(config_id="G1", method_version="v", sha256=hex64("1")),),
        "config_snapshot_sha256s": (hex64("2"),),
        "git_commit": "a" * 40,
        "environment_lock_sha256": "e" * 64,
    }
    data.update(overrides)
    return CalibrationRunProvenance.model_validate(data)


def new_process(run_id: str = "cal-run-1") -> tuple[CalibrationProcess, HeldOutGuard]:
    guard = HeldOutGuard()
    return CalibrationProcess(guard, SPLIT, provenance(run_id=run_id), CONFIG), guard


# --- Synthetische, status-erhaltende Ergebnisse -------------------------------------------------


def clean(alarm: bool) -> ProcedureOutcome:
    return ProcedureOutcome(status=SignalStatus.VALID, alarm=alarm)


def shift(alarm: bool, relevant: bool = True) -> ShiftOutcome:
    return ShiftOutcome(status=SignalStatus.VALID, alarm=alarm, relevant=relevant)


def method_outcomes(
    clean_alarms: Sequence[bool], shifts_per_repeat: int = 12
) -> dict[str, MethodOutcomes]:
    repeats = tuple(clean(a) for a in clean_alarms)
    shifts = tuple(tuple(shift(True) for _ in range(shifts_per_repeat)) for _ in clean_alarms)
    return {m: MethodOutcomes(method=m, clean=repeats, shifts=shifts) for m in R_METHODS}


SEVERITY_MEDIANS = {
    TechnicalShift.NOISE: (0.01, 0.025, 0.05, 0.1, 0.2),
    TechnicalShift.CONTRAST: (0.0, 0.02, 0.05, 0.1, 0.3),
    TechnicalShift.BLUR: (0.02, 0.03, 0.05, 0.09, 0.2),
    TechnicalShift.RESOLUTION: (0.0, 0.01, 0.025, 0.05, 0.1),
}


def sex_maps(process: CalibrationProcess) -> tuple[dict[str, str | None], ...]:
    """Alternierendes male/female je Läsion (synthetisch)."""
    pools = process.results.pools
    reference = process.results.reference
    assert pools is not None and reference is not None

    def sex(lesion_id: str) -> str:
        return "male" if int(lesion_id[-3:]) % 2 else "female"

    return (
        {i: sex(i) for i in reference.lesion_ids},
        {i: sex(i) for i in pools.lesion_ids(CalibrationPool.CAL_REF)},
        {i: sex(i) for i in pools.lesion_ids(CalibrationPool.CAL_TUNE)},
    )


def advance_to(process: CalibrationProcess, target: CalibrationStep) -> None:
    """Treibt die Sequenz (ab dem aktuellen Stand) bis einschließlich `target` mit synthetischen,
    gültigen Werten."""
    source = SyntheticLabelSource(LESIONS)
    d0 = [(i % 100) / 1000 - 0.03 for i in range(1000)]
    r_clean = [i % 20 == 0 for i in range(400)]
    fusion_clean = [clean(i % 25 == 0) for i in range(100)]
    fusion_shifts = [shift(i % 3 != 0, relevant=i % 4 != 0) for i in range(60)]
    S = CalibrationStep
    actions: dict[CalibrationStep, Any] = {
        S.AUTHORIZE: process.authorize,
        S.POOL_SPLIT: lambda: process.derive_pool_assignment(source),
        S.CANONICAL_SAMPLES: process.derive_canonical_samples,
        S.RESTRICT_AUDIT: process.restrict_audit,
        S.REF_ARTIFACTS: process.complete_reference_artifacts,
        S.MMD_SIGMA: lambda: process.record_mmd_sigma(12.5),
        S.W128_VALIDITY: lambda: process.record_window_validity(128, True),
        S.SELECT_WINDOW_SIZE: process.select_window_size,
        S.OFFICIAL_REFERENCE: process.derive_official_reference,
        S.BA_REF: lambda: process.record_ba_ref(0.7),
        S.D0_WINDOWS: lambda: process.record_d0_values(d0),
        S.D0_QUANTILES: process.derive_relevance_threshold,
        S.DELTA_REL: process.freeze_delta_rel,
        S.SEVERITY_CANDIDATES: lambda: process.record_severity_medians(SEVERITY_MEDIANS),
        S.SEVERITY_SELECTION: process.confirm_severity_selection,
        S.CLASS_SHIFT: process.derive_class_shift,
        S.SUBGROUP: lambda: process.record_subgroup(*sex_maps(process)),
        S.SCHEDULES: process.derive_schedules,
        S.R_SELECTION: lambda: process.record_r_outcomes(method_outcomes(r_clean)),
        S.FUSION_SELECTION: lambda: process.record_fusion_outcomes(
            dict.fromkeys(FusionCandidate, fusion_clean),
            dict.fromkeys(FusionCandidate, fusion_shifts),
        ),
        S.FUSION_FIXED: process.fix_fusion_candidate,
        S.AUDIT_RELEASE: process.release_audit,
        S.AUDIT_FAR: lambda: process.record_audit_outcomes(
            [clean(i % 50 == 0) for i in range(100)]
        ),
        S.VALIDATE: process.validate,
    }
    for step in CalibrationStep:
        if step > target:
            return
        if step in actions and step not in process.completed:
            actions[step]()


def artifact_manifest(process: CalibrationProcess, **role_overrides: str) -> ArtifactManifest:
    """ArtifactManifest mit den autoritativen Hashes plus einem nicht-wissenschaftlichen Log."""
    hashes = {**process.required_artifact_hashes(), **role_overrides}
    entries = [
        ArtifactEntry(
            path=f"calibration/{role.removeprefix('calibration.')}.json", sha256=h, role=role
        )
        for role, h in sorted(hashes.items())
    ]
    entries.append(ArtifactEntry(path="logs/run.log", sha256=hex64("4"), role="log"))
    return ArtifactManifest(run_id=process.run_id, entries=tuple(entries))


def record_artifacts(process: CalibrationProcess) -> str:
    if process.results.artifact_manifest_sha256 is not None:
        return process.results.artifact_manifest_sha256
    return process.record_artifact_manifest(artifact_manifest(process))


def manifest_for(
    process: CalibrationProcess,
    calibrated_config_data: dict[str, Any],
    record_sha256: str | None = None,
) -> FreezeManifestV2:
    """Schema-2-Manifest, das (standardmäßig) den prozesserzeugten Record bindet."""
    record_artifacts(process)
    record = process.build_freeze_record()
    data = with_frozen_policies(calibrated_config_data)
    scientific = data["scientific"]
    scientific["dataset"]["sha256"] = record.dataset_snapshot_sha256
    scientific["split"]["sha256"] = record.split_manifest_sha256
    scientific["model"]["sha256"] = record.model_weights_sha256
    scientific["monitoring"]["window_size"] = record.window_size
    scientific["relevance"]["delta_rel"] = record.delta_rel
    scientific["fusion"]["rule"] = (
        "B_AT_LEAST_TWO_PRIMARY_GROUPS"
        if record.fusion_candidate is FusionCandidate.FUSION_B
        else "A_ANY_PRIMARY_GROUP"
    )
    data["repetitions"] = record.final_r
    config = RunConfig.model_validate(data)
    seeds: tuple[SeedBundle, ...] = tuple(
        config.seeds.model_copy(update={"repeat_id": i}) for i in range(record.final_r)
    )
    return build_freeze_manifest_v2(
        config,
        seed_list=seeds,
        git_commit=record.git_commit,
        environment_lock_sha256=record.environment_lock_sha256,
        calibration_record_sha256=record_sha256 or record.record_sha256(),
    )


def standalone_record(process: CalibrationProcess, **overrides: Any) -> CalibrationFreezeRecord:
    record_artifacts(process)
    record = process.build_freeze_record()
    data = {**record.model_dump(mode="python"), **overrides}
    return CalibrationFreezeRecord.model_validate(data)


def rate(successes: int, trials: int) -> Rate:
    return Rate(successes=successes, trials=trials)

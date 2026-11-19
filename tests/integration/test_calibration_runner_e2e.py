"""Vollständiger synthetischer End-to-End-Lauf des Calibration-Runners bis `FROZEN` (P4-02E).

Ein einziger Lauf (Modul-Fixture) durch alle 28 Protokollschritte mit dem echten
`Ham10000Dataset`, den echten Signalmodulen G1–G4, den Shift-Operatoren, dem HeldOutGuard, dem
ArtifactManifest, dem Calibration-Freeze-Record und FreezeManifestV2. Keine echten Daten.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from audit_leakage import leaking_paths
from calibration_e2e_world import (
    LAYOUT,
    PROTOCOL,
    RecordingDataset,
    TextureModel,
    World,
    preprocessing_spec,
    provenance_for,
    synthetic_execution,
    template_data,
    write_world,
)
from calibration_runner.freeze_assembly import verify_seed_list
from calibration_runner.orchestrator import CalibrationRunner, CalibrationRunResult
from provenance_repo import HELD_OUT, ProvenanceRepo, approve_and_verify
from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.adapters.preprocessing import ImagePreprocessor
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.artifacts.manifests import ArtifactManifest
from vigiscoutx.config.scientific_config import (
    RECOMMENDATION_POLICY_REF,
    RISK_POLICY_REF,
    calibration_gaps,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.errors import IntegrityError, PhaseViolationError
from vigiscoutx.experiment.calibration.config import (
    R_METHODS,
    CalibrationPool,
    FusionCandidate,
    SamplingNamespace,
)
from vigiscoutx.experiment.calibration.freeze_record import verify_record_against_manifest
from vigiscoutx.experiment.calibration.process import CalibrationStep
from vigiscoutx.experiment.calibration.structure import repeat_seed
from vigiscoutx.experiment.code_provenance import EvaluationProvenance
from vigiscoutx.experiment.freeze import (
    FreezeManifest,
    compute_freeze_id,
    verify_freeze_manifest,
)
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import Partition, Phase
from vigiscoutx.ports import AccessPurpose


class StepRecordingDataset(RecordingDataset):
    """Protokolliert zusätzlich den zuletzt abgeschlossenen Protokollschritt je Zugriff."""

    runner: CalibrationRunner | None = None

    def _log(self, sample_ids: Any, purpose: AccessPurpose) -> None:
        completed = () if self.runner is None else self.runner.process.completed
        self.step = int(completed[-1]) if completed else 0
        super()._log(sample_ids, purpose)


@dataclass
class E2E:
    world: World
    dataset: Ham10000Dataset
    runner: CalibrationRunner
    guard: HeldOutGuard
    recorder: StepRecordingDataset
    template: RunConfig
    result: CalibrationRunResult


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> E2E:
    world = write_world(tmp_path_factory.mktemp("world"))
    dataset = Ham10000Dataset.from_directory(world.root, layout=LAYOUT)
    recorder = StepRecordingDataset(dataset)
    preprocessor = ImagePreprocessor(preprocessing_spec())
    model = TextureModel(preprocessor)
    template = RunConfig.model_validate(template_data(world, dataset, model))
    guard = HeldOutGuard()
    runner = CalibrationRunner(
        guard=guard,
        split_manifest=world.split,
        provenance=provenance_for(template, dataset, model),
        protocol=PROTOCOL,
        dataset=recorder,
        audit_dataset=dataset,
        execution=synthetic_execution(dataset, model),
        model=model,
        preprocessor=preprocessor,
        template=template,
        output_dir=tmp_path_factory.mktemp("run") / "calibration",
    )
    recorder.runner = runner
    result = runner.run()
    return E2E(world, dataset, runner, guard, recorder, template, result)


def _cases(e2e: E2E) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(
        (e2e.result.output_dir / "cases/case_provenance.json").read_bytes()
    )
    return data


def test_all_protocol_steps_complete_until_frozen(e2e: E2E) -> None:
    process = e2e.runner.process
    expected = [s for s in CalibrationStep if s is not CalibrationStep.HELD_OUT]
    if process.results.window_validity == ((128, True),):
        expected.remove(CalibrationStep.W256_VALIDITY)
    assert list(process.completed) == expected
    assert process.stopped is None and process.failure is None
    assert e2e.guard.phase is Phase.FROZEN
    assert not process.is_completed(CalibrationStep.HELD_OUT)


def test_freeze_record_and_manifest_bind_the_process_state(e2e: E2E) -> None:
    record, manifest = e2e.result.record, e2e.result.manifest
    process = e2e.runner.process
    process.verify_freeze_record(record)
    verify_freeze_manifest(manifest)
    verify_record_against_manifest(record, manifest)
    assert manifest.schema_version == 2
    assert manifest.calibration_record_sha256 == record.record_sha256()
    # P4-03C: der synthetische Freeze bindet die eingefrorenen v1.0.0-Policies.
    assert (manifest.risk_policy, manifest.recommendation_policy) == (
        RISK_POLICY_REF,
        RECOMMENDATION_POLICY_REF,
    )
    results = process.results
    assert record.window_size == results.window_size
    assert record.delta_rel == results.threshold.delta_rel  # type: ignore[union-attr]
    assert record.final_r == results.final_r == manifest.repetitions == len(manifest.seed_list)
    assert record.fusion_candidate is results.fusion_candidate
    stored = json.loads((e2e.result.output_dir / "freeze/freeze_manifest_v2.json").read_bytes())
    assert stored["freeze_id"] == manifest.freeze_id


def test_run_config_changes_only_calibrated_values(e2e: E2E) -> None:
    record = e2e.result.record
    before = e2e.template.model_dump(mode="json")
    after = e2e.result.run_config.model_dump(mode="json")
    assert after["scientific"]["monitoring"]["window_size"] == record.window_size
    assert after["scientific"]["relevance"]["delta_rel"] == record.delta_rel
    assert after["repetitions"] == record.final_r
    g1 = next(s for s in after["scientific"]["monitoring"]["signals"] if s["group"] == "G1")
    assert g1["params"]["sigma"] == record.mmd_sigma
    for data in (before, after):
        data["scientific"]["monitoring"]["window_size"] = None
        data["scientific"]["relevance"]["delta_rel"] = None
        data["scientific"]["fusion"]["rule"] = None
        data["repetitions"] = None
        for signal in data["scientific"]["monitoring"]["signals"]:
            signal["params"].pop("sigma", None)
    assert before == after


def test_artifact_manifest_is_authoritative_and_write_once(e2e: E2E) -> None:
    root = e2e.result.output_dir
    manifest = ArtifactManifest.model_validate_json(
        (root / "freeze/artifact_manifest.json").read_bytes()
    )
    assert canonical_sha256(manifest) == e2e.result.record.artifact_manifest_sha256
    for entry in manifest.entries:
        assert sha256_file(root / entry.path) == entry.sha256
    roles = {e.role for e in manifest.entries}
    assert roles == {
        *e2e.runner.process.required_artifact_hashes(),
        "provenance.cases",
        "provenance.execution",
    }
    execution = json.loads((root / "provenance/execution.json").read_bytes())
    assert execution == {
        "mode": "SYNTHETIC",
        "authorization_sha256": None,
        "authorization_id": None,
        "synthetic_dataset_snapshot_sha256": e2e.dataset.snapshot_ref().sha256,
        "synthetic_model_snapshot_sha256": e2e.runner.process._provenance.model_snapshot_sha256,
    }
    assert not (root / "termination.json").exists()


def test_data_access_respects_partitions_pools_and_audit_blindness(e2e: E2E) -> None:
    calibration = e2e.world.partition_samples(Partition.CALIBRATION)
    accessed = {s for s, _, _ in e2e.recorder.accesses}
    assert accessed <= calibration
    assert not accessed & e2e.world.partition_samples(Partition.HELD_OUT)
    pools = e2e.runner.process.results.pools
    canonical = dict(e2e.runner.process.results.canonical_samples or ())
    assert pools is not None
    audit = {canonical[i] for i in pools.lesion_ids(CalibrationPool.CAL_AUDIT)}
    purposes: dict[str, set[AccessPurpose]] = defaultdict(set)
    for sample_id, purpose, _ in e2e.recorder.accesses:
        purposes[sample_id].add(purpose)
    # Über den (öffentlichen) Datenport sieht der Runner CAL_AUDIT nur beim strukturellen
    # Pool-Split; die Audit-Bilder lädt ausschließlich der prozessinterne Audit-Executor.
    for sample_id in audit:
        assert purposes[sample_id] <= {AccessPurpose.STRATIFICATION_LABELS}
    assert e2e.runner.process._audit_drawn == e2e.runner.process.results.final_r
    store = e2e.runner._features
    cached = {
        key[0]
        for cache in (store._embeddings, store._probabilities, store._quality_rows)
        for key in cache
    }
    assert cached and not cached & audit
    assert not set(store._labels) & audit
    labelled = {s for s, p, _ in e2e.recorder.accesses if p is AccessPurpose.LABELS}
    assert labelled and not labelled & audit
    images = {s for s, p, _ in e2e.recorder.accesses if p is AccessPurpose.INPUTS}
    assert images <= set(canonical.values())


def test_case_matrix_seeds_and_paired_construction(e2e: E2E) -> None:
    cases = _cases(e2e)
    results = e2e.runner.process.results
    final_r = results.final_r
    window_size = results.window_size
    subgroup = results.subgroup
    assert final_r is not None and window_size is not None and subgroup is not None
    per_repeat = 18 if subgroup.feasible else 15
    by_namespace = Counter(c["namespace"] for c in cases)
    validity_runs = sum(1 for _, ok in results.window_validity or () if ok is not None)
    assert by_namespace[SamplingNamespace.W_VALIDITY.value] == 100 * validity_runs
    assert by_namespace[SamplingNamespace.FUSION_CLEAN.value] == final_r
    assert by_namespace[SamplingNamespace.AUDIT_CLEAN.value] == final_r
    stability = [c for c in cases if c["namespace"] == SamplingNamespace.R_STABILITY.value]
    repeats = Counter(c["repeat_id"] for c in stability)
    assert sorted(repeats) == list(range(1, len(repeats) + 1))
    assert set(repeats.values()) == {per_repeat + 1}
    split_hash = e2e.runner.process.split_hash
    for case in cases:
        namespace = SamplingNamespace(case["namespace"])
        seed = repeat_seed(
            split_hash, namespace, case["window_size"], case["scenario_key"], case["repeat_id"]
        )
        assert case["statistical_seed"] == seed
        assert "sample_ids" not in case and "lesion_ids" not in case
    for repeat_id in repeats:
        rows = [c for c in stability if c["repeat_id"] == repeat_id]
        assert len({c["statistical_seed"] for c in rows}) == len(rows)
        for family in ("NOISE", "CONTRAST", "BLUR", "RESOLUTION", "CLASS", "SUBGROUP_SEX"):
            levels = [c for c in rows if c["family"] == family]
            if family == "SUBGROUP_SEX" and not subgroup.feasible:
                assert not levels
                continue
            assert [c["scenario_key"] for c in levels] == [
                f"ABRUPT_{family}_L{k}" for k in (1, 2, 3)
            ]
            if family not in ("CLASS", "SUBGROUP_SEX"):
                assert len({c["base_window_id"] for c in levels}) == 1


def test_fusion_reuses_exactly_the_r_stability_cases(e2e: E2E) -> None:
    results = e2e.runner.process.results
    final_r = results.final_r
    assert final_r is not None and results.fusion_metrics is not None
    shifts = [
        c
        for c in _cases(e2e)
        if c["namespace"] == SamplingNamespace.R_STABILITY.value
        and c["level"] is not None
        and c["repeat_id"] <= final_r
    ]
    relevant = [c for c in shifts if c["result"]["relevant"]]
    nonrelevant = [c for c in shifts if c["result"]["relevant"] is False]
    assert len(relevant) + len(nonrelevant) == len(shifts)
    for metrics in results.fusion_metrics:
        assert metrics.sensitivity.trials == len(relevant)
        assert metrics.nonrelevant_shift_alarm_rate.trials == len(nonrelevant)
        assert metrics.clean_far.trials == final_r
        alarms = sum(
            c["result"]["procedures"][metrics.candidate.value]["alarm"] is True for c in relevant
        )
        assert metrics.sensitivity.successes == alarms
    assert results.fusion_candidate in set(FusionCandidate)


def test_audit_far_from_r_fresh_clean_windows(e2e: E2E) -> None:
    results = e2e.runner.process.results
    assert results.audit_far is not None and results.final_r is not None
    assert results.audit_far.trials == results.final_r
    audit = [c for c in _cases(e2e) if c["namespace"] == SamplingNamespace.AUDIT_CLEAN.value]
    candidate = results.fusion_candidate
    assert candidate is not None
    alarms = sum(c["result"]["procedures"][candidate.value]["alarm"] is True for c in audit)
    assert results.audit_far.successes == alarms
    assert all(c["result"]["delta_ba"] is None for c in audit)
    # Protokoll §13 (P4-03H): im Audit nur G1–G3 und der fixierte Kandidat, genau R Windows.
    primary = set(R_METHODS) - {c.value for c in FusionCandidate}
    assert all(set(c["result"]["procedures"]) == primary | {candidate.value} for c in audit)
    assert sorted(c["repeat_id"] for c in audit) == list(range(1, results.final_r + 1))
    assert len({c["monitoring_window_id"] for c in audit}) == results.final_r
    for namespace in (SamplingNamespace.R_STABILITY, SamplingNamespace.FUSION_CLEAN):
        cases = [c for c in _cases(e2e) if c["namespace"] == namespace.value]
        assert cases and all(set(c["result"]["procedures"]) == set(R_METHODS) for c in cases)


def test_held_out_stays_closed_after_frozen(e2e: E2E) -> None:
    assert e2e.guard.phase is Phase.FROZEN
    with pytest.raises(PhaseViolationError):
        e2e.guard.issue_permit(Partition.HELD_OUT, e2e.world.split)
    with pytest.raises(PhaseViolationError):
        e2e.guard.issue_permit(Partition.CALIBRATION, e2e.world.split)


def test_output_directory_is_never_reused(e2e: E2E, tmp_path: Path) -> None:
    model = TextureModel(ImagePreprocessor(preprocessing_spec()))
    with pytest.raises(FileExistsError):
        CalibrationRunner(
            guard=HeldOutGuard(),
            split_manifest=e2e.world.split,
            provenance=e2e.runner.process._provenance,
            protocol=PROTOCOL,
            dataset=e2e.recorder,
            audit_dataset=e2e.dataset,
            execution=synthetic_execution(e2e.dataset, model),
            model=model,
            preprocessor=ImagePreprocessor(preprocessing_spec()),
            template=e2e.template,
            output_dir=e2e.result.output_dir,
        )


# --- Projektentscheidungen P4-02E ----------------------------------------------------------------


def test_threshold_contract_method_decided_signals_without_threshold(e2e: E2E) -> None:
    run_config, manifest = e2e.result.run_config, e2e.result.manifest
    assert all(s.threshold is None for s in e2e.template.scientific.monitoring.signals)
    assert all(s.threshold is None for s in manifest.signals)
    assert calibration_gaps(run_config.scientific) == []
    stored = json.loads((e2e.result.output_dir / "freeze/run_config.json").read_bytes())
    assert all(s["threshold"] is None for s in stored["scientific"]["monitoring"]["signals"])
    # Das historische Schema v1 behält seine Regel (jede Schwelle gesetzt) und lehnt ab.
    data = manifest.model_dump(mode="json", exclude={"calibration_record_sha256"})
    with pytest.raises(ValueError, match="Signalschwellen"):
        FreezeManifest.model_validate({**data, "schema_version": 1})


def test_seed_list_register_rule(e2e: E2E) -> None:
    manifest, template = e2e.result.manifest, e2e.template
    final_r = e2e.result.record.final_r
    seeds = manifest.seed_list
    assert [s.repeat_id for s in seeds] == list(range(1, final_r + 1))
    fixed = template.seeds.model_dump(exclude={"repeat_id"})
    assert all(s.model_dump(exclude={"repeat_id"}) == fixed for s in seeds)
    verify_seed_list(template, seeds, final_r)
    assert compute_freeze_id(manifest) == manifest.freeze_id
    summary = json.loads((e2e.result.output_dir / "freeze/summary.json").read_bytes())
    assert summary["seed_list_sha256"] == canonical_sha256(
        [s.model_dump(mode="json") for s in seeds]
    )
    with pytest.raises(IntegrityError):
        verify_seed_list(template, seeds[:-1], final_r)
    with pytest.raises(IntegrityError):
        verify_seed_list(template, tuple(reversed(seeds)), final_r)


def _held_out_config(e2e: E2E, seeds: Any, freeze_id: str) -> RunConfig:
    data = e2e.result.run_config.model_dump(mode="json")
    data.update(
        {
            "run_id": "synthetic-held-out-authorization",
            "phase": "HELD_OUT_EVALUATION",
            "freeze_id": freeze_id,
            "seeds": seeds.model_dump(mode="json"),
        }
    )
    return RunConfig.model_validate(data)


def _observed(e2e: E2E) -> ObservedHashes:
    manifest = e2e.result.manifest
    return ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=manifest.split_manifest_sha256,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=manifest.environment_lock_sha256,
    )


def test_register_authorizes_only_registered_held_out_seed_bundles(
    e2e: E2E, provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    """Synthetisch auf einem separaten Guard: nur die Zulassungsprüfung, keine Partition.

    Der synthetische Freeze wird für diesen Test an den Commit eines synthetischen
    Git-Repositories gebunden (DEC-082: Code-Provenienz wird selbst ermittelt).
    """
    rebound = e2e.result.manifest.model_copy(
        update={"git_commit": provenance_repo.calibration_commit}
    )
    manifest = rebound.model_copy(update={"freeze_id": compute_freeze_id(rebound)})
    evaluation_provenance = approve_and_verify(
        provenance_repo, manifest.freeze_id, tmp_path / "approval", purpose=HELD_OUT
    )
    assert isinstance(evaluation_provenance, EvaluationProvenance)
    registered = manifest.seed_list[len(manifest.seed_list) // 2]
    guard = HeldOutGuard()
    guard.freeze(manifest)
    session = guard.open_held_out(
        _held_out_config(e2e, registered, manifest.freeze_id), _observed(e2e), evaluation_provenance
    )
    assert session.freeze_id == manifest.freeze_id
    foreign = registered.model_copy(update={"statistical_seed": registered.statistical_seed + 1})
    other = HeldOutGuard()
    other.freeze(manifest)
    with pytest.raises(IntegrityError, match="SeedBundle"):
        other.open_held_out(
            _held_out_config(e2e, foreign, manifest.freeze_id),
            _observed(e2e),
            evaluation_provenance,
        )
    outside = registered.model_copy(update={"repeat_id": len(manifest.seed_list) + 1})
    with pytest.raises(IntegrityError, match="SeedBundle"):
        other.open_held_out(
            _held_out_config(e2e, outside, manifest.freeze_id),
            _observed(e2e),
            evaluation_provenance,
        )
    held_out = e2e.world.partition_samples(Partition.HELD_OUT)
    assert not {s for s, _, _ in e2e.recorder.accesses} & held_out
    assert e2e.guard.phase is Phase.FROZEN


def test_case_specific_g1_seed_refs_are_independent_of_the_register(e2e: E2E) -> None:
    register = {s.statistical_seed for s in e2e.result.manifest.seed_list}
    seeds = [c["statistical_seed"] for c in _cases(e2e)]
    assert len(register) == 1 and not register & set(seeds)
    assert len(set(seeds)) == len(seeds)


def test_no_public_artifact_links_audit_ids_to_diagnoses(e2e: E2E) -> None:
    process = e2e.runner.process
    results = process.results
    assert results.pools is not None
    lesions = set(results.pools.lesion_ids(CalibrationPool.CAL_AUDIT))
    canonical = dict(results.canonical_samples or ())
    ids = lesions | {canonical[i] for i in lesions}
    root = e2e.result.output_dir
    public: dict[str, Any] = {}
    for path in sorted(root.rglob("*.json")):
        name = path.relative_to(root).as_posix()
        role = {
            "calibration/pool_manifest.json": "calibration.pool_manifest",
            "calibration/canonical_samples.json": "calibration.canonical_samples",
        }.get(name, name)
        public[role] = json.loads(path.read_bytes())
    public["view"] = results.model_dump(mode="json")
    assert not leaking_paths(public, ids)
    pool_manifest = public["calibration.pool_manifest"]
    assert sorted(lesions) == pool_manifest["audit_lesion_ids"]
    assert all(
        b["pool"] != CalibrationPool.CAL_AUDIT.value for b in pool_manifest["ref_tune_blocks"]
    )


# --- P4-03J: Reproduktion – die run_id beeinflusst keine wissenschaftlichen Ergebnisse ---------


@pytest.fixture(scope="module")
def e2e_other_run_id(e2e: E2E, tmp_path_factory: pytest.TempPathFactory) -> CalibrationRunResult:
    """Zweiter synthetischer Lauf auf derselben Welt; nur die Ausführungsidentität ändert sich."""
    preprocessor = ImagePreprocessor(preprocessing_spec())
    model = TextureModel(preprocessor)
    data = template_data(e2e.world, e2e.dataset, model)
    data["run_id"] = "synthetic-calibration-e2e-reproduction"
    template = RunConfig.model_validate(data)
    runner = CalibrationRunner(
        guard=HeldOutGuard(),
        split_manifest=e2e.world.split,
        provenance=provenance_for(template, e2e.dataset, model),
        protocol=PROTOCOL,
        dataset=e2e.dataset,
        audit_dataset=e2e.dataset,
        execution=synthetic_execution(e2e.dataset, model),
        model=model,
        preprocessor=preprocessor,
        template=template,
        output_dir=tmp_path_factory.mktemp("reproduction") / "calibration",
    )
    return runner.run()


def test_run_id_changes_no_pool_window_seed_or_method_result(
    e2e: E2E, e2e_other_run_id: CalibrationRunResult
) -> None:
    first, second = e2e.result, e2e_other_run_id
    assert first.run_id != second.run_id
    calibration = sorted(p.name for p in (first.output_dir / "calibration").iterdir())
    assert calibration == sorted(p.name for p in (second.output_dir / "calibration").iterdir())
    for name in (*(f"calibration/{n}" for n in calibration), "cases/case_provenance.json"):
        # Pools, Canonical Samples, Reference, σ-Quelle, D0, Severity, R, Fusion, Audit,
        # sämtliche Window-IDs, Seeds und Methodenergebnisse: byteidentisch.
        assert (first.output_dir / name).read_bytes() == (second.output_dir / name).read_bytes()
    a = first.record.model_dump(mode="json")
    b = second.record.model_dump(mode="json")
    differing = sorted(k for k in a if a[k] != b[k])
    # Nur die Ausführungsidentität: Vorlagen-Hash (enthält run_id) und ArtifactManifest (run_id).
    assert differing == ["artifact_manifest_sha256", "config_snapshot_sha256s"]
    assert a["config_snapshot_sha256s"][1:] == b["config_snapshot_sha256s"][1:]
    assert first.manifest.config_sha256 == second.manifest.config_sha256
    assert first.manifest.seed_list == second.manifest.seed_list
    assert first.run_config.scientific == second.run_config.scientific

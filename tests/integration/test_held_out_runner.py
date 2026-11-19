"""P4-03AI: synthetischer End-to-End-Lauf des HELD_OUT-Evaluationsrunners (Protokoll v1.1).

Modul-Fixture: synthetische Welt mit Texturbildern auch für HELD_OUT, vollständiger synthetischer
Calibration-Lauf bis `FROZEN` (Calibration-Commit in einem synthetischen Git-Repository),
einmalige Reference-Rekonstruktion und genau ein vollständiger HELD_OUT-Lauf. Weitere Tests
brechen vor bzw. kurz nach dem Öffnen ab. Keine realen Daten, kein ResNet, keine reale Freigabe.
"""

import dataclasses
import json
import shutil
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

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
from calibration_runner.orchestrator import CalibrationRunner, CalibrationRunResult
from held_out_runner import entrypoints, orchestrator
from held_out_runner.entrypoints import (
    EntryPointStop,
    ExecutionBinding,
    ExecutionLayout,
    LoadedModel,
    derive_held_out_config,
    run_held_out_phase,
    run_reference_phase,
)
from held_out_runner.orchestrator import HeldOutRunner, HeldOutRunResult, load_held_out_results
from provenance_repo import (
    GUARD_PATH,
    HELD_OUT,
    REFERENCE,
    ProvenanceRepo,
    approve_and_verify,
    create_provenance_repo,
    write_approval,
)
from vigiscoutx.adapters.ham10000 import Ham10000Dataset
from vigiscoutx.adapters.preprocessing import ImagePreprocessor
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.domain.assessments import SignalStatus
from vigiscoutx.domain.hashing import canonical_json_bytes, sha256_hex
from vigiscoutx.errors import IntegrityError
from vigiscoutx.evaluation import reference_reconstruction
from vigiscoutx.evaluation.held_out_metrics import (
    ALL_METHODS,
    FUSION,
    HeldOutMetrics,
    ScoredCase,
    Thresholds,
    held_out_metrics,
)
from vigiscoutx.evaluation.held_out_sampling import (
    FAMILIES,
    EvaluationNamespace,
    HeldOutCasePlan,
    HeldOutEvaluationStop,
    held_out_canonical_samples,
)
from vigiscoutx.evaluation.reference_reconstruction import (
    EVIDENCE_MANIFEST_FILE,
    ReconstructedReference,
    ReferenceExecution,
    ReferenceExpectation,
    reconstruct_reference,
)
from vigiscoutx.evaluation.reference_sources import (
    FrozenCalibration,
    ReferenceReconstructionStop,
    load_frozen_calibration,
)
from vigiscoutx.experiment import code_provenance
from vigiscoutx.experiment.calibration import process as calibration_process
from vigiscoutx.experiment.calibration.config import CalibrationProtocolConfig, FusionCandidate
from vigiscoutx.experiment.calibration.freeze_record import (
    CalibrationFreezeRecord,
    verify_record_against_manifest,
)
from vigiscoutx.experiment.calibration.statistics import FusionMetrics, select_fusion
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    ExecutionPurpose,
    verify_evaluation_provenance,
)
from vigiscoutx.experiment.freeze import FreezeManifestV2, compute_freeze_id, verify_freeze_manifest
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split_integrity import partition_sample_ids
from vigiscoutx.experiment.splitting import write_split_manifest
from vigiscoutx.experiment.types import Partition, Phase
from vigiscoutx.monitoring import MonitoringRunResult
from vigiscoutx.ports import AccessPurpose


@dataclasses.dataclass(frozen=True)
class Setup:
    repo: ProvenanceRepo
    world: World
    dataset: Ham10000Dataset
    preprocessor: ImagePreprocessor
    model: TextureModel
    calibration: CalibrationRunResult
    frozen: FrozenCalibration
    reference_dir: Path
    config: RunConfig
    observed: ObservedHashes
    root: Path
    expectation: ReferenceExpectation

    @property
    def lock(self) -> Path:
        return self.repo.root / "requirements-lock.txt"


def _prefer_fusion_a(
    original: Any,
) -> Any:
    """Synthetische Fixture: Der reale Freeze fixiert `FUSION_A`; in dieser Testwelt führt der
    eingefrorene Gleichstand zu B. Die Originalregel wird weiter geprüft (FAR-STOP), danach wird
    ausschließlich für diesen synthetischen Lauf A fixiert, sofern A die FAR-Grenze einhält."""

    def select(metrics: Sequence[FusionMetrics], config: CalibrationProtocolConfig) -> Any:
        original(metrics, config)
        a = next(m for m in metrics if m.candidate is FusionCandidate.FUSION_A)
        assert a.clean_far.value <= Fraction(str(config.far_max))
        return FusionCandidate.FUSION_A

    return select


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Setup]:
    root = tmp_path_factory.mktemp("held-out")
    repo = create_provenance_repo(root / "provenance-repo")
    world = write_world(root / "world", held_out_textures=True)
    dataset = Ham10000Dataset.from_directory(world.root, layout=LAYOUT)
    preprocessor = ImagePreprocessor(preprocessing_spec())
    model = TextureModel(preprocessor)
    template = RunConfig.model_validate(template_data(world, dataset, model))
    lock = repo.root / "requirements-lock.txt"
    runner = CalibrationRunner(
        guard=HeldOutGuard(),
        split_manifest=world.split,
        provenance=provenance_for(
            template,
            dataset,
            model,
            git_commit=repo.calibration_commit,
            environment_lock_sha256=sha256_file(lock),
        ),
        protocol=PROTOCOL,
        dataset=dataset,
        audit_dataset=dataset,
        execution=synthetic_execution(dataset, model),
        model=model,
        preprocessor=preprocessor,
        template=template,
        output_dir=root / "calibration",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            calibration_process,
            "select_fusion",
            _prefer_fusion_a(select_fusion),
        )
        calibration = runner.run()
    assert calibration.record.fusion_candidate is FusionCandidate.FUSION_A
    frozen = load_frozen_calibration(calibration.output_dir)
    manifest = calibration.manifest
    vector = runner.process.results.class_vector
    ba_ref = runner.process.results.ba_ref
    assert vector is not None and ba_ref is not None
    expectation = ReferenceExpectation(
        freeze_id=manifest.freeze_id,
        class_counts=tuple(vector.as_dict().items()),
        ba_ref_4dp=f"{ba_ref:.4f}",
        ba_ref_source="synthetischer Calibration-Lauf (Testfixture)",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", repo.root / GUARD_PATH)
        guard = HeldOutGuard()
        guard.freeze(manifest)
        provenance = approve_and_verify(
            repo, manifest.freeze_id, root / "approval-reference", purpose=REFERENCE
        )
        assert isinstance(provenance, EvaluationProvenance)
        reconstruct_reference(
            guard=guard,
            frozen=frozen,
            split_manifest=world.split,
            dataset=dataset,
            model=model,
            preprocessor=preprocessor,
            environment_lock=lock,
            provenance=provenance,
            expectation=expectation,
            execution=ReferenceExecution(device="cpu", batch_size=32),
            output_dir=root / "reference",
        )
    data = calibration.run_config.model_dump(mode="json")
    data.update(
        {
            "run_id": "synthetic-held-out-evaluation",
            "phase": "HELD_OUT_EVALUATION",
            "freeze_id": manifest.freeze_id,
            "seeds": manifest.seed_list[0].model_dump(mode="json"),
        }
    )
    yield Setup(
        repo=repo,
        world=world,
        dataset=dataset,
        preprocessor=preprocessor,
        model=model,
        calibration=calibration,
        frozen=frozen,
        reference_dir=root / "reference",
        config=RunConfig.model_validate(data),
        observed=ObservedHashes(
            dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
            split_manifest_sha256=manifest.split_manifest_sha256,
            model_weights_sha256=manifest.model_weights_sha256,
            environment_lock_sha256=manifest.environment_lock_sha256,
        ),
        root=root,
        expectation=expectation,
    )


@dataclasses.dataclass
class Run:
    setup: Setup
    guard: HeldOutGuard
    provenance: EvaluationProvenance
    recorder: RecordingDataset
    output: Path

    def runner(self, **overrides: Any) -> HeldOutRunner:
        arguments: dict[str, Any] = {
            "guard": self.guard,
            "frozen": self.setup.frozen,
            "split_manifest": self.setup.world.split,
            "dataset": self.recorder,
            "model": self.setup.model,
            "preprocessor": self.setup.preprocessor,
            "environment_lock": self.setup.lock,
            "provenance": self.provenance,
            "config": self.setup.config,
            "observed": self.setup.observed,
            "reference_dir": self.setup.reference_dir,
            "execution": ReferenceExecution(device="cpu", batch_size=32),
            "output_dir": self.output,
            **overrides,
        }
        return HeldOutRunner(**arguments)

    @property
    def stopped(self) -> dict[str, Any] | None:
        path = self.output.with_name(f"{self.output.name}.STOPPED.json")
        return json.loads(path.read_bytes()) if path.exists() else None

    def no_result(self) -> bool:
        staging = self.output.with_name(f".{self.output.name}.incomplete")
        return not self.output.exists() and not staging.exists()


def _evidence_sha256(setup: Setup) -> str:
    return sha256_file(setup.reference_dir / EVIDENCE_MANIFEST_FILE)


def _new_run(
    setup: Setup,
    directory: Path,
    *,
    purpose: str = HELD_OUT,
    reference_evidence_sha256: str | None = None,
) -> Run:
    """Phase 2 mit eigenem HELD_OUT-Nachweis, der das geprüfte Reference-Manifest bindet."""
    guard = HeldOutGuard()
    guard.freeze(setup.calibration.manifest)
    provenance = approve_and_verify(
        setup.repo,
        setup.calibration.manifest.freeze_id,
        directory / "approval",
        purpose=purpose,
        reference_evidence_sha256=reference_evidence_sha256 or _evidence_sha256(setup),
    )
    assert isinstance(provenance, EvaluationProvenance)
    return Run(setup, guard, provenance, RecordingDataset(setup.dataset), directory / "held-out")


@dataclasses.dataclass(frozen=True)
class Completed:
    run: Run
    result: HeldOutRunResult
    contents: dict[str, bytes]


@pytest.fixture(scope="module")
def completed(setup: Setup) -> Iterator[Completed]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
        run = _new_run(setup, setup.root / "completed")
        result = run.runner().run()
        _, contents = load_held_out_results(result.output_dir)
        yield Completed(run, result, contents)


@pytest.fixture
def fresh(setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
    return _new_run(setup, tmp_path)


def _json(completed: Completed, role: str) -> Any:
    return json.loads(completed.contents[role])


# --- Vollständiger synthetischer Lauf -----------------------------------------------------------


def test_complete_run_has_the_exact_case_matrix(completed: Completed) -> None:
    result = completed.result
    repeats = completed.run.setup.frozen.record.final_r
    assert repeats == 100
    assert len(result.plan.cases) == 100 + 5 * 4 * 100
    metrics = result.metrics
    assert (metrics.n_clean, metrics.n_baselines, metrics.n_abrupt, metrics.n_trajectories) == (
        100,
        500,
        1500,
        500,
    )
    assert result.n_monitoring_runs == 2100
    assert completed.run.guard.phase is Phase.HELD_OUT_EVALUATION


def test_only_fusion_a_and_the_seven_isolated_methods_are_evaluated(completed: Completed) -> None:
    for role, content in completed.contents.items():
        assert b"FUSION_B" not in content, role
    label_free = _json(completed, "evaluation.label_free")
    assert all(set(r["alarms"]) == set(ALL_METHODS) for r in label_free)
    assert all(
        {s.method for s in table.pooled} == set(ALL_METHODS)
        for table in completed.result.metrics.confusion
    )


def test_label_free_results_are_fixed_before_g4(completed: Completed) -> None:
    roles = [e.role for e in completed.result.manifest.entries]
    assert roles.index("evaluation.case_plan") < roles.index("evaluation.label_free")
    assert roles.index("evaluation.label_free") < roles.index("evaluation.g4")
    for row in _json(completed, "evaluation.label_free"):
        assert "delta_ba" not in row and "relevant" not in json.dumps(row)
    g4 = _json(completed, "evaluation.g4")
    clean_ids = {
        c.case_id
        for c in completed.result.plan.cases
        if c.namespace is EvaluationNamespace.HELD_OUT_CLEAN
    }
    assert len(g4) == 2000 and not {row["case_id"] for row in g4} & clean_ids


def test_data_access_is_held_out_only_via_the_permit(completed: Completed) -> None:
    setup = completed.run.setup
    held_out = partition_sample_ids(setup.world.split, Partition.HELD_OUT)
    accesses = completed.run.recorder.accesses
    assert accesses and {s for s, _, _ in accesses} <= held_out
    assert {p for _, p, _ in accesses} == {AccessPurpose.INPUTS, AccessPurpose.LABELS}
    canonical = set(held_out_canonical_samples(setup.world.split).values())
    labels = [s for s, p, _ in accesses if p is AccessPurpose.LABELS]
    assert sorted(labels) == sorted(canonical)  # Strukturlabels genau einmal je Läsion
    reference = set(setup.frozen.record.reference_sample_ids)
    assert not {s for s, _, _ in accesses} & reference


def test_metrics_follow_exactly_from_the_written_case_results(completed: Completed) -> None:
    plan = HeldOutCasePlan.model_validate(_json(completed, "evaluation.case_plan"))
    assert plan == completed.result.plan
    label_free = {r["case_id"]: r for r in _json(completed, "evaluation.label_free")}
    g4 = {r["case_id"]: r for r in _json(completed, "evaluation.g4")}
    record = completed.run.setup.frozen.record
    thresholds = Thresholds(delta_rel=record.delta_rel, q090=record.q090, q099=record.q099)
    scored = [
        ScoredCase(
            case_id=c.case_id,
            clean=c.namespace is EvaluationNamespace.HELD_OUT_CLEAN,
            family=c.family,
            level=c.level,
            repeat_id=c.repeat_id,
            alarms=tuple(sorted(label_free[c.case_id]["alarms"].items())),
            delta_ba=g4[c.case_id]["delta_ba"] if c.case_id in g4 else None,
        )
        for c in plan.cases
    ]
    recomputed = held_out_metrics(scored, thresholds, families=FAMILIES, repeats=plan.repeats)
    assert recomputed == HeldOutMetrics.model_validate(_json(completed, "evaluation.metrics"))
    assert recomputed == completed.result.metrics
    for row in g4.values():
        assert row["relevant_delta_rel"] == (row["delta_ba"] >= record.delta_rel)
        assert row["relevant_q090"] == (row["delta_ba"] >= record.q090)
        assert row["relevant_q099"] == (row["delta_ba"] >= record.q099)


def test_result_is_informative_on_the_synthetic_benchmark(completed: Completed) -> None:
    metrics = completed.result.metrics
    table = next(t for t in metrics.confusion if t.threshold == "DELTA_REL")
    assert 0 < table.n_relevant < table.n_cases  # beide Klassen der Ground Truth vorhanden
    pooled = {s.method: s.confusion for s in table.pooled}
    assert all(c.tp + c.fp + c.tn + c.fn == 1500 for c in pooled.values())
    assert pooled[FUSION].tp > 0
    assert all(r.denominator == 100 for _, r in metrics.clean_far)
    blocks = [b for b in metrics.repeat_blocks if b.method == FUSION]
    assert len(blocks) == 100 and all(b.tp_minus_fusion == 0 for b in blocks)


def test_monitoring_runs_bind_signals_risk_recommendations_and_g4(completed: Completed) -> None:
    runs = _json(completed, "evaluation.monitoring_runs")
    assert len(runs) == 2100
    manifest = completed.run.setup.calibration.manifest
    plan = completed.result.plan.by_id()
    for data in runs[::97]:  # vollständige Neuvalidierung einer Stichprobe aus der Datei
        run = MonitoringRunResult.model_validate(data)
        assert (run.freeze_id, run.config_sha256) == (manifest.freeze_id, manifest.config_sha256)
        (primary,) = run.primary_assessments
        case = plan[primary.window_id]
        assert {s.reference_window_id for s in run.signal_results} == {
            completed.run.setup.frozen.record.reference_window_id
        }
        primary_recs = [r for r in run.recommendations if r.level.value == "PRIMARY"]
        assert len(primary_recs) == 1
        if case.namespace is EvaluationNamespace.HELD_OUT_CLEAN:
            assert run.label_aware_assessments == ()
        else:
            (secondary,) = run.label_aware_assessments
            assert secondary.performance.delta_rel == completed.run.setup.frozen.record.delta_rel
            secondary_recs = [r for r in run.recommendations if r.level.value == "SECONDARY"]
            assert len(secondary_recs) == (1 if secondary.state.value == "R3" else 0)
        assert "scenario_key" not in json.dumps(data)  # Evaluationsumgebung bleibt außen vor


def test_case_provenance_contains_seeds_and_is_bound(completed: Completed) -> None:
    execution = _json(completed, "provenance.execution")
    run = completed.run
    assert execution["plan_sha256"] == completed.result.plan.plan_sha256()
    assert execution["evaluation_commit"] == run.provenance.evaluation_commit
    assert execution["freeze_id"] == run.setup.calibration.manifest.freeze_id
    assert execution["fusion_b_computed"] is False
    assert execution["approval_purpose"] == "HELD_OUT_EVALUATION"
    assert execution["reference_evidence_sha256_bound"] == _evidence_sha256(run.setup)
    assert execution["reference_approval_sha256"] != execution["approval_sha256"]
    assert execution["requested_device"] == execution["observed_default_device"] == "cpu"
    assert execution["input_tensor_devices"] == ["cpu"] and execution["max_model_batch"] == 32
    label_free = {r["case_id"]: r for r in _json(completed, "evaluation.label_free")}
    for case in completed.result.plan.cases:
        assert label_free[case.case_id]["g1_seed"] == case.g1_seed
    noise = [c for c in completed.result.plan.cases if c.noise_seeds is not None]
    assert len(noise) == 300 and all(len(c.noise_seeds or ()) == 128 for c in noise)


def test_results_are_write_once_and_tampering_is_detected(
    completed: Completed, tmp_path: Path
) -> None:
    directory = completed.result.output_dir
    assert all(not (p.stat().st_mode & 0o222) for p in directory.rglob("*") if p.is_file())
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    target = copy / "results/metrics.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes().replace(b"0", b"1", 1))
    with pytest.raises(HeldOutEvaluationStop, match="verändert"):
        load_held_out_results(copy)
    (copy / "results/extra.json").write_text("{}", "utf-8")
    with pytest.raises(HeldOutEvaluationStop):
        load_held_out_results(copy)


def test_held_out_cannot_be_evaluated_twice(completed: Completed, tmp_path: Path) -> None:
    with pytest.raises(HeldOutEvaluationStop, match="nur aus FROZEN"):
        completed.run.runner(output_dir=tmp_path / "again").run()


# --- STOP vor dem Öffnen ------------------------------------------------------------------------


# --- P4-03AM: zweistufige, zweckgebundene Freigabe (DEC-087) -----------------------------------


def _assert_stopped_before_access(run: Run) -> None:
    assert run.guard.phase is Phase.FROZEN and run.recorder.accesses == []
    assert run.no_result() and run.stopped is not None
    assert run.stopped["valid_result"] is False


def test_reference_only_approval_never_opens_held_out(setup: Setup, tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
        run = _new_run(setup, tmp_path, purpose=REFERENCE)
        assert run.provenance.purpose is ExecutionPurpose.REFERENCE_RECONSTRUCTION
        with pytest.raises(HeldOutEvaluationStop, match="Freigabe für HELD_OUT_EVALUATION"):
            run.runner().run()
        _assert_stopped_before_access(run)
        with pytest.raises(IntegrityError, match="HELD_OUT_EVALUATION"):
            run.guard.open_held_out(setup.config, setup.observed, run.provenance)
        assert run.guard.phase is Phase.FROZEN


def test_held_out_approval_never_permits_reference_reconstruction(
    fresh: Run, tmp_path: Path
) -> None:
    setup = fresh.setup
    with pytest.raises(ReferenceReconstructionStop, match="REFERENCE_RECONSTRUCTION"):
        reconstruct_reference(
            guard=fresh.guard,
            frozen=setup.frozen,
            split_manifest=setup.world.split,
            dataset=fresh.recorder,
            model=setup.model,
            preprocessor=setup.preprocessor,
            environment_lock=setup.lock,
            provenance=fresh.provenance,
            expectation=None,
            execution=ReferenceExecution(device="cpu", batch_size=32),
            output_dir=tmp_path / "second-reference",
        )
    with pytest.raises(IntegrityError, match="REFERENCE_RECONSTRUCTION"):
        fresh.guard.issue_reference_reconstruction_permit(
            record=setup.frozen.record,
            split_manifest=setup.world.split,
            sample_ids=setup.frozen.record.reference_sample_ids,
            provenance=fresh.provenance,
        )
    assert fresh.recorder.accesses == [] and not (tmp_path / "second-reference").exists()


def test_wrong_or_missing_reference_evidence_blocks_phase_two(setup: Setup, tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
        wrong = _new_run(setup, tmp_path / "wrong", reference_evidence_sha256="0" * 64)
        with pytest.raises(IntegrityError, match="weicht vom freigegebenen Hash ab"):
            wrong.runner().run()
        _assert_stopped_before_access(wrong)
        missing = _new_run(setup, tmp_path / "missing")
        with pytest.raises(IntegrityError, match="Reference-Artefakt fehlt"):
            missing.runner(reference_dir=tmp_path / "no-reference").run()
        _assert_stopped_before_access(missing)


@dataclasses.dataclass
class _Swap:
    """Deterministischer A → B → A-Austausch genau um den Lesezugriff des Loaders (P4-03AN)."""

    manifest: Path
    approved: bytes  # A: freigegeben, anderer Evaluation-Commit
    genuine: bytes  # B: echter Evaluation-Commit
    loaded: list[bytes]
    outer: list[Path]


def _swap_attack(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Swap, Path]:
    copy = tmp_path / "reference-swap"
    shutil.copytree(setup.reference_dir, copy)
    manifest = copy / EVIDENCE_MANIFEST_FILE
    genuine = manifest.read_bytes()
    data = json.loads(genuine)
    data["evaluation_commit"] = "0" * 40
    approved = canonical_json_bytes(data)
    manifest.chmod(0o644)
    manifest.write_bytes(approved)
    swap = _Swap(manifest, approved, genuine, [], [])
    original_read = reference_reconstruction._read_file

    def read(directory: Path, relative: str) -> bytes:
        if relative != EVIDENCE_MANIFEST_FILE:
            return original_read(directory, relative)
        manifest.write_bytes(genuine)  # A → B unmittelbar vor dem Lesen
        content = original_read(directory, relative)
        manifest.write_bytes(approved)  # B → A unmittelbar danach
        swap.loaded.append(content)
        return content

    def recording_sha(path: Path) -> str:
        swap.outer.append(path)
        return sha256_file(path)

    monkeypatch.setattr(reference_reconstruction, "_read_file", read)
    monkeypatch.setattr(f"{orchestrator.__name__}.sha256_file", recording_sha)
    return swap, copy


def test_a_b_a_manifest_swap_stops_at_the_buffer_hash(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P4-03AN: äußere Datei-Hashes A, Loader verarbeitet B → jetzt STOP vor dem Parsing."""
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
    swap, copy = _swap_attack(setup, tmp_path, monkeypatch)
    run = _new_run(setup, tmp_path / "run", reference_evidence_sha256=sha256_hex(swap.approved))
    with pytest.raises(ReferenceReconstructionStop, match="weicht vom freigegebenen Hash ab"):
        run.runner(reference_dir=copy)._reference()
    assert swap.loaded == [swap.genuine]  # genau ein Lesevorgang, Bytes B erkannt
    with pytest.raises(ReferenceReconstructionStop, match="weicht vom freigegebenen Hash ab"):
        run.runner(reference_dir=copy).run()
    _assert_stopped_before_access(run)
    assert swap.manifest.name not in {p.name for p in swap.outer}  # kein Hash aus Dateizustand


def test_unswapped_approved_manifest_passes_the_buffer_check(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
    run = _new_run(setup, tmp_path)
    reference = run.runner()._reference()
    assert (
        sha256_hex(canonical_json_bytes(reference.evidence))
        == run.provenance.reference_evidence_sha256
    )
    assert reference.evidence.evaluation_commit == run.provenance.evaluation_commit
    assert run.guard.phase is Phase.FROZEN and run.recorder.accesses == []


def test_tampered_artifact_with_correct_manifest_stops(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
    copy = tmp_path / "reference-artifact"
    shutil.copytree(setup.reference_dir, copy)
    target = copy / "reference/g3_model_outputs.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes().replace(b"0", b"1", 1))
    run = _new_run(setup, tmp_path)  # Freigabe bindet das unveränderte, korrekte Manifest
    with pytest.raises(ReferenceReconstructionStop, match="Inhalt verändert"):
        run.runner(reference_dir=copy).run()
    _assert_stopped_before_access(run)


def test_reference_evidence_of_another_commit_is_rejected(setup: Setup, tmp_path: Path) -> None:
    """Auch ein hashgebundenes, aber zu einem anderen Evaluation-Commit gehörendes Manifest."""
    copy = tmp_path / "reference-other-commit"
    shutil.copytree(setup.reference_dir, copy)
    manifest = copy / EVIDENCE_MANIFEST_FILE
    data = json.loads(manifest.read_bytes())
    data["evaluation_commit"] = "0" * 40
    manifest.chmod(0o644)
    manifest.write_bytes(canonical_json_bytes(data))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", setup.repo.root / GUARD_PATH)
        run = _new_run(setup, tmp_path, reference_evidence_sha256=sha256_file(manifest))
        with pytest.raises(IntegrityError, match="anderen Freeze oder Evaluation-Commit"):
            run.runner(reference_dir=copy).run()
        _assert_stopped_before_access(run)


@pytest.mark.parametrize("tamper", ["modified", "removed", "purpose-swapped"])
def test_manipulated_or_removed_held_out_approval_stops(fresh: Run, tamper: str) -> None:
    approval = Path(fresh.provenance.approval_path)
    if tamper == "removed":
        approval.unlink()
    elif tamper == "modified":
        data = json.loads(approval.read_bytes())
        approval.write_text(json.dumps({**data, "approval_id": "forged"}), "utf-8")
    else:  # derselbe Nachweis nachträglich auf die Reference-Phase umgeschrieben
        data = json.loads(approval.read_bytes())
        approval.write_text(
            json.dumps({**data, "purpose": REFERENCE, "reference_evidence_sha256": None}), "utf-8"
        )
    with pytest.raises(IntegrityError):
        fresh.runner().run()
    _assert_stopped_before_access(fresh)


def test_legacy_single_phase_approval_authorizes_nothing(fresh: Run, tmp_path: Path) -> None:
    setup = fresh.setup
    legacy = write_approval(
        tmp_path / "legacy",
        evaluation_commit=setup.repo.git("rev-parse", "HEAD"),
        calibration_commit=setup.repo.calibration_commit,
        freeze_id=setup.calibration.manifest.freeze_id,
        purpose=HELD_OUT,
        overrides={"schema_version": 1, "purpose": None, "reference_evidence_sha256": None},
    )
    for purpose in ExecutionPurpose:
        with pytest.raises(IntegrityError, match="veraltet"):
            verify_evaluation_provenance(
                calibration_commit=setup.repo.calibration_commit,
                freeze_id=setup.calibration.manifest.freeze_id,
                approval_path=legacy,
                purpose=purpose,
            )


@pytest.mark.parametrize("defect", ["reference", "batch", "config", "approval", "output"])
def test_binding_defects_stop_before_held_out_is_opened(
    fresh: Run, tmp_path: Path, defect: str
) -> None:
    overrides: dict[str, Any] = {}
    if defect == "reference":
        copy = tmp_path / "reference-copy"
        shutil.copytree(fresh.setup.reference_dir, copy)
        target = copy / "reference/g2_quality.json"
        target.chmod(0o644)
        target.write_bytes(target.read_bytes().replace(b"0", b"1", 1))
        overrides["reference_dir"] = copy
    elif defect == "batch":
        overrides["execution"] = ReferenceExecution(device="cpu", batch_size=16)
    elif defect == "config":
        data = fresh.setup.config.model_dump(mode="json")
        data["scientific"]["relevance"]["delta_rel"] = 0.5
        overrides["config"] = RunConfig.model_validate(data)
    elif defect == "approval":
        Path(fresh.provenance.approval_path).unlink()
    else:
        fresh.output.mkdir()
    with pytest.raises(IntegrityError):
        fresh.runner(**overrides).run()
    assert fresh.guard.phase is Phase.FROZEN and fresh.recorder.accesses == []
    assert fresh.stopped is not None and fresh.stopped["valid_result"] is False


# --- STOP nach dem Öffnen: nicht evaluierbare Pflichtsignale ------------------------------------


@pytest.mark.parametrize(("call", "label"), [(1, "CLEAN/1"), (101, "NOISE_L0/1")])
def test_non_evaluable_mandatory_signal_is_a_terminal_stop(
    fresh: Run, monkeypatch: pytest.MonkeyPatch, call: int, label: str
) -> None:
    original = orchestrator._mandatory_statuses
    calls = {"n": 0}

    def statuses(*args: Any) -> list[SignalStatus]:
        calls["n"] += 1
        result = original(*args)
        return [SignalStatus.NOT_EVALUABLE, *result[1:]] if calls["n"] == call else result

    monkeypatch.setattr(orchestrator, "_mandatory_statuses", statuses)
    with pytest.raises(HeldOutEvaluationStop, match=label):
        fresh.runner().run()
    assert fresh.no_result()
    assert fresh.stopped is not None and fresh.stopped["phase"] == "LABEL_FREE"
    assert fresh.guard.phase is Phase.HELD_OUT_EVALUATION  # geöffnet, kein Ersatz, kein Neulauf


# --- P4-03AK: Freeze-Record-Integrität (P4-03AJ, HIGH) ------------------------------------


def _manipulated_record(frozen: FrozenCalibration, field: str) -> FrozenCalibration:
    record = frozen.record
    update: dict[str, Any]
    if field == "q090":
        update = {"q090": 0.0}  # Manipulation: Sensitivitätsschwelle Q0.90 auf 0.0
    elif field == "q099":
        update = {"q099": record.q099 + 0.05}
    elif field == "severity":
        severity = record.technical_severities[0]
        indices = (0, 1, 2) if severity.selected_indices != (0, 1, 2) else (1, 2, 3)
        changed = severity.model_copy(
            update={
                "selected_indices": indices,
                "selected_values": tuple(severity.candidates[i] for i in indices),
            }
        )
        update = {"technical_severities": (changed, *record.technical_severities[1:])}
    elif field == "class_shift":
        levels = tuple(level + 1 for level in record.class_shift.levels)
        update = {"class_shift": record.class_shift.model_copy(update={"levels": levels})}
    elif field == "mmd_sigma":
        update = {"mmd_sigma": record.mmd_sigma * 2}
    else:
        update = {"reference_window_id": "0" * 64}
    return dataclasses.replace(frozen, record=record.model_copy(update=update))


@pytest.mark.parametrize(
    "field", ["q090", "q099", "severity", "class_shift", "mmd_sigma", "reference_window_id"]
)
def test_manipulated_freeze_record_stops_before_any_data_access(fresh: Run, field: str) -> None:
    frozen = _manipulated_record(fresh.setup.frozen, field)
    with pytest.raises(IntegrityError, match="Calibration-Record weicht vom FreezeManifest ab"):
        fresh.runner(frozen=frozen).run()
    assert fresh.guard.phase is Phase.FROZEN and fresh.recorder.accesses == []
    assert fresh.no_result()
    assert fresh.stopped is not None
    assert (fresh.stopped["phase"], fresh.stopped["valid_result"]) == ("BINDINGS", False)


@pytest.mark.parametrize("consistent_freeze_id", [True, False])
def test_wrong_calibration_record_sha256_stops_before_any_data_access(
    fresh: Run, consistent_freeze_id: bool
) -> None:
    manifest = fresh.setup.frozen.manifest
    fields = {**manifest.__dict__, "calibration_record_sha256": "0" * 64}
    forged = FreezeManifestV2.model_construct(**fields)  # am Validator vorbei erzeugt
    if consistent_freeze_id:  # in sich konsistentes Manifest mit falschem Record-Hash
        forged = FreezeManifestV2.model_construct(
            **{**fields, "freeze_id": compute_freeze_id(forged)}
        )
        match = r"weicht vom FreezeManifest ab: \['calibration_record_sha256'\]"
    else:
        match = "FreezeManifest verändert"
    frozen = dataclasses.replace(fresh.setup.frozen, manifest=forged)
    with pytest.raises(IntegrityError, match=match):
        fresh.runner(frozen=frozen).run()
    assert fresh.guard.phase is Phase.FROZEN and fresh.recorder.accesses == []
    assert fresh.no_result() and fresh.stopped is not None
    assert fresh.stopped["valid_result"] is False


def test_unchanged_record_and_manifest_pair_passes(completed: Completed) -> None:
    frozen = completed.run.setup.frozen
    verify_freeze_manifest(frozen.manifest)
    verify_record_against_manifest(frozen.record, frozen.manifest)
    execution = _json(completed, "provenance.execution")
    assert execution["freeze_id"] == frozen.manifest.freeze_id
    assert completed.result.metrics.thresholds.q090 == frozen.record.q090
    assert completed.result.metrics.thresholds.q099 == frozen.record.q099


# --- P4-03AR: produktive Einstiegspunkte (nur synthetisch) --------------------------------------


@dataclasses.dataclass(frozen=True)
class Entry:
    setup: Setup
    layout: ExecutionLayout
    binding: ExecutionBinding

    def loader(
        self, _snapshot: Path, record: CalibrationFreezeRecord, _preprocessor: ImagePreprocessor
    ) -> LoadedModel:
        """Synthetisches Testmodell statt `resnet18-pilot-P1` (keine realen Gewichte)."""
        return LoadedModel(model=self.setup.model, weights_sha256=record.model_weights_sha256)

    def approval(
        self,
        directory: Path,
        purpose: str,
        *,
        evidence: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> Path:
        return write_approval(
            directory,
            evaluation_commit=self.setup.repo.git("rev-parse", "HEAD"),
            calibration_commit=self.binding.calibration_commit,
            freeze_id=self.binding.freeze_id,
            purpose=purpose,
            reference_evidence_sha256=evidence,
            name=f"{purpose.lower()}.json",
            overrides=overrides,
        )

    def reference(self, **overrides: Any) -> ReconstructedReference:
        arguments: dict[str, Any] = {
            "binding": self.binding,
            "layout": self.layout,
            "model_loader": self.loader,
            "expectation": self.setup.expectation,
            **overrides,
        }
        return run_reference_phase(**arguments)

    def held_out(self, **overrides: Any) -> HeldOutRunResult:
        arguments: dict[str, Any] = {
            "binding": self.binding,
            "layout": self.layout,
            "model_loader": self.loader,
            **overrides,
        }
        return run_held_out_phase(**arguments)


def _integrity_manifest(setup: Setup, target: Path) -> None:
    """Synthetisches Gegenstück zum versionierten Integritätsmanifest (nur Hashes)."""
    run = setup.calibration.output_dir
    files = [
        {
            "path": str(p.relative_to(setup.root)),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(run.rglob("*"))
        if p.is_file()
    ]
    manifest = setup.calibration.manifest
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "freeze_id": manifest.freeze_id,
                "execution_git_commit": manifest.git_commit,
                "files": files,
            }
        ),
        "utf-8",
    )


@pytest.fixture(scope="module")
def entry(setup: Setup) -> Entry:
    base = setup.root / "entry"
    base.mkdir()
    split = base / "split.json"
    write_split_manifest(setup.world.split, split)
    _integrity_manifest(setup, base / "integrity.json")
    (base / "results").mkdir()
    layout = ExecutionLayout(
        root=setup.root,
        calibration_run=setup.calibration.output_dir,
        integrity_manifest=base / "integrity.json",
        split_manifest=split,
        dataset_root=setup.world.root,
        dataset_layout=LAYOUT,
        model_snapshot=base / "model_snapshot.json",  # vom synthetischen Lader nie gelesen
        environment_lock=setup.lock,
        results_root=base / "results",
    )
    manifest = setup.calibration.manifest
    return Entry(setup, layout, ExecutionBinding(manifest.freeze_id, manifest.git_commit))


@dataclasses.dataclass(frozen=True)
class TwoPhase:
    reference: ReconstructedReference
    held_out: HeldOutRunResult
    reference_approval: Path
    held_out_approval: Path


@pytest.fixture(scope="module")
def two_phase(entry: Entry) -> Iterator[TwoPhase]:
    base = entry.setup.root / "two-phase"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code_provenance, "_EXECUTING_FILE", entry.setup.repo.root / GUARD_PATH)
        reference_approval = entry.approval(base, REFERENCE)
        reference = entry.reference(
            approval=reference_approval,
            output_dir=entry.layout.results_root / "reference",
            execute=True,
        )
        evidence = sha256_file(reference.directory / EVIDENCE_MANIFEST_FILE)
        held_out_approval = entry.approval(base, HELD_OUT, evidence=evidence)
        held_out = entry.held_out(
            approval=held_out_approval,
            reference_dir=reference.directory,
            output_dir=entry.layout.results_root / "held-out",
            execute=True,
        )
        yield TwoPhase(reference, held_out, reference_approval, held_out_approval)


def test_entrypoints_run_the_complete_two_phase_sequence(two_phase: TwoPhase) -> None:
    reference, held_out = two_phase.reference, two_phase.held_out
    assert len(reference.sample_ids) == 128
    assert held_out.n_monitoring_runs == 2100 and held_out.metrics.n_abrupt == 1500
    _, contents = load_held_out_results(held_out.output_dir)
    execution = json.loads(contents["provenance.execution"])
    assert execution["evaluation_commit"] == reference.evidence.evaluation_commit
    assert execution["approval_purpose"] == "HELD_OUT_EVALUATION"
    assert execution["reference_approval_sha256"] == reference.evidence.approval_sha256
    assert execution["reference_approval_sha256"] != execution["approval_sha256"]
    assert execution["reference_evidence_sha256"] == sha256_file(
        reference.directory / EVIDENCE_MANIFEST_FILE
    )
    assert two_phase.reference_approval != two_phase.held_out_approval


class _Sentinels:
    """Schlägt fehl und protokolliert, sobald Pools, Split, Dataset oder Gewichte berührt werden."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, pools: bool = True) -> None:
        self.calls: list[str] = []

        def record(name: str) -> Any:
            def fail(*_args: Any, **_kwargs: Any) -> Any:
                self.calls.append(name)
                raise AssertionError(f"unzulässiger Zugriff vor gültiger Freigabe: {name}")

            return fail

        if pools:
            monkeypatch.setattr(entrypoints, "load_frozen_calibration", record("pools"))
            monkeypatch.setattr(entrypoints, "read_split_manifest", record("split"))
        dataset = type("Dataset", (), {"from_directory": staticmethod(record("dataset"))})
        monkeypatch.setattr(entrypoints, "Ham10000Dataset", dataset)
        self.loader = record("weights")


@pytest.fixture
def guarded(entry: Entry, monkeypatch: pytest.MonkeyPatch) -> tuple[Entry, _Sentinels]:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", entry.setup.repo.root / GUARD_PATH)
    return entry, _Sentinels(monkeypatch)


@pytest.mark.parametrize(
    "defect",
    [
        "no-execute",
        "missing-approval",
        "held-out-approval",
        "other-commit",
        "other-freeze",
        "legacy-schema",
        "existing-output",
        "output-outside-results",
    ],
)
def test_reference_entrypoint_stops_before_any_sensitive_access(
    guarded: tuple[Entry, _Sentinels], tmp_path: Path, defect: str
) -> None:
    entry, sentinels = guarded
    output = entry.layout.results_root / f"reference-{defect}"
    arguments: dict[str, Any] = {
        "approval": entry.approval(tmp_path, REFERENCE),
        "output_dir": output,
        "execute": True,
        "model_loader": sentinels.loader,
    }
    if defect == "no-execute":
        arguments["execute"] = False
    elif defect == "missing-approval":
        arguments["approval"] = tmp_path / "missing.json"
    elif defect == "held-out-approval":
        arguments["approval"] = entry.approval(tmp_path / "h", HELD_OUT, evidence="e" * 64)
    elif defect == "other-commit":
        arguments["approval"] = entry.approval(
            tmp_path / "c", REFERENCE, overrides={"evaluation_commit": "0" * 40}
        )
    elif defect == "other-freeze":
        arguments["approval"] = entry.approval(
            tmp_path / "f", REFERENCE, overrides={"freeze_id": "1" * 64}
        )
    elif defect == "legacy-schema":
        arguments["approval"] = entry.approval(
            tmp_path / "l", REFERENCE, overrides={"schema_version": 1, "purpose": None}
        )
    elif defect == "existing-output":
        output.mkdir()
    else:
        arguments["output_dir"] = tmp_path / "outside"
    with pytest.raises(IntegrityError):
        entry.reference(**arguments)
    assert sentinels.calls == []
    assert defect == "existing-output" or not output.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "no-execute",
        "missing-approval",
        "reference-approval",
        "wrong-evidence",
        "missing-reference",
        "other-commit",
        "existing-output",
    ],
)
def test_held_out_entrypoint_stops_before_any_sensitive_access(
    guarded: tuple[Entry, _Sentinels], two_phase: TwoPhase, tmp_path: Path, defect: str
) -> None:
    entry, sentinels = guarded
    reference_dir = two_phase.reference.directory
    evidence = sha256_file(reference_dir / EVIDENCE_MANIFEST_FILE)
    output = entry.layout.results_root / f"held-out-{defect}"
    arguments: dict[str, Any] = {
        "approval": entry.approval(tmp_path, HELD_OUT, evidence=evidence),
        "reference_dir": reference_dir,
        "output_dir": output,
        "execute": True,
        "model_loader": sentinels.loader,
    }
    if defect == "no-execute":
        arguments["execute"] = False
    elif defect == "missing-approval":
        arguments["approval"] = tmp_path / "missing.json"
    elif defect == "reference-approval":
        arguments["approval"] = entry.approval(tmp_path / "r", REFERENCE)
    elif defect == "wrong-evidence":
        arguments["approval"] = entry.approval(tmp_path / "w", HELD_OUT, evidence="0" * 64)
    elif defect == "missing-reference":
        arguments["reference_dir"] = entry.layout.results_root / "no-reference"
    elif defect == "other-commit":
        arguments["approval"] = entry.approval(
            tmp_path / "c", HELD_OUT, evidence=evidence, overrides={"evaluation_commit": "0" * 40}
        )
    else:
        output.mkdir()
    with pytest.raises(IntegrityError):
        entry.held_out(**arguments)
    assert sentinels.calls == []
    assert defect == "existing-output" or not output.exists()


def test_deviating_freeze_identity_stops_before_pools_dataset_or_weights(
    guarded: tuple[Entry, _Sentinels], tmp_path: Path
) -> None:
    entry, sentinels = guarded
    other = ExecutionBinding("1" * 64, entry.binding.calibration_commit)
    approval = write_approval(
        tmp_path,
        evaluation_commit=entry.setup.repo.git("rev-parse", "HEAD"),
        calibration_commit=other.calibration_commit,
        freeze_id=other.freeze_id,
        purpose=REFERENCE,
    )
    with pytest.raises(EntryPointStop, match="gebundenen Freeze-Identität"):
        entry.reference(
            approval=approval,
            output_dir=entry.layout.results_root / "other-freeze",
            execute=True,
            binding=other,
            model_loader=sentinels.loader,
            expectation=None,
        )
    assert sentinels.calls == []


@pytest.mark.parametrize("missing", ["integrity", "freeze-file", "split"])
def test_missing_files_stop_without_dataset_or_weights(
    entry: Entry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", entry.setup.repo.root / GUARD_PATH)
    sentinels = _Sentinels(monkeypatch, pools=False)
    layout = entry.layout
    if missing == "integrity":
        layout = dataclasses.replace(layout, integrity_manifest=tmp_path / "none.json")
    elif missing == "freeze-file":
        copy = tmp_path / "calibration"
        shutil.copytree(layout.calibration_run, copy)
        (copy / "freeze/summary.json").unlink()
        layout = dataclasses.replace(layout, root=tmp_path, calibration_run=copy)
        integrity = json.loads(layout.integrity_manifest.read_bytes())
        target = tmp_path / "integrity.json"
        target.write_text(json.dumps(integrity), "utf-8")
        layout = dataclasses.replace(layout, integrity_manifest=target)
    else:
        layout = dataclasses.replace(layout, split_manifest=tmp_path / "none.json")
    with pytest.raises(IntegrityError):
        entry.reference(
            approval=entry.approval(tmp_path / "a", REFERENCE),
            output_dir=entry.layout.results_root / f"missing-{missing}",
            execute=True,
            layout=layout,
            model_loader=sentinels.loader,
        )
    assert sentinels.calls == []


@pytest.mark.parametrize("defect", ["metrics", "phase", "seeds", "scientific"])
def test_faulty_held_out_run_config_is_rejected(entry: Entry, defect: str) -> None:
    frozen = entry.setup.frozen
    assert derive_held_out_config(frozen).freeze_id == frozen.manifest.freeze_id
    if defect == "metrics":
        changed = frozen.run_config.model_copy(update={"evaluation_metrics": ("SENSITIVITY",)})
        frozen = dataclasses.replace(frozen, run_config=changed)
    elif defect == "phase":
        data = frozen.run_config.model_dump(mode="json")
        data.update({"phase": "HELD_OUT_EVALUATION", "freeze_id": frozen.manifest.freeze_id})
        frozen = dataclasses.replace(frozen, run_config=RunConfig.model_validate(data))
    elif defect == "seeds":
        seeds = list(frozen.manifest.seed_list)
        seeds[1] = seeds[1].model_copy(update={"sampling_seed": seeds[1].sampling_seed + 1})
        forged = FreezeManifestV2.model_construct(
            **{**frozen.manifest.__dict__, "seed_list": tuple(seeds)}
        )
        frozen = dataclasses.replace(frozen, manifest=forged)
    else:
        data = frozen.run_config.model_dump(mode="json")
        data["scientific"]["relevance"]["delta_rel"] = 0.5
        frozen = dataclasses.replace(frozen, run_config=RunConfig.model_validate(data))
    with pytest.raises(EntryPointStop):
        derive_held_out_config(frozen)


@pytest.mark.parametrize("phase", ["reference", "held-out"])
def test_cli_is_fail_closed_and_binds_the_real_freeze_in_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str
) -> None:
    sentinels = _Sentinels(monkeypatch)
    output = "experiments/results/raw/P4-03AR-cli-test-never-created"
    common = [phase, "--approval", str(tmp_path / "missing.json"), "--output-dir", output]
    if phase == "held-out":
        common += ["--reference-dir", "experiments/results/raw/P4-03AR-cli-test-no-reference"]
    for argv in (common, [*common, "--execute"]):
        assert entrypoints.main(argv) == entrypoints.STOP_EXIT_CODE
        assert json.loads(capsys.readouterr().out.strip())["status"] == "STOPPED"
    assert sentinels.calls == [] and not Path(output).exists()
    assert (
        ExecutionBinding(
            "58b1e879f3adc03f9f84b10810f9ed727ab0eba41bbde891786c8f529dc52a93",
            "5ad433a22cf176aa1846d5030b393837b4454c7b",
        )
        == entrypoints.REAL_BINDING
    )
    with pytest.raises(SystemExit):
        entrypoints.main([*common, "--execute", "--freeze-id", "0" * 64])

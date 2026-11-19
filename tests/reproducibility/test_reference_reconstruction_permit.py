"""P4-03Z: eng begrenzte REFERENCE_RECONSTRUCTION-Berechtigung (DEC-084/085), nur synthetisch.

Ein synthetischer Calibration-Prozess erzeugt Record und FreezeManifestV2; der Calibration-Commit
liegt in einem synthetischen Git-Repository, das als ausgeführtes Repository gilt. Keine realen
Daten, keine Bilder, keine Inferenz.
"""

import copy
import dataclasses
import pickle
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from calibration_fixtures import (
    CONFIG,
    SPLIT,
    advance_to,
    manifest_for,
    provenance,
    record_artifacts,
)
from policy_fixtures import with_frozen_policies
from provenance_repo import HELD_OUT, REFERENCE, ProvenanceRepo, approve_and_verify
from vigiscoutx.errors import IntegrityError, PhaseViolationError
from vigiscoutx.experiment import held_out_guard
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.calibration.process import CalibrationProcess, CalibrationStep
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    reverify_evaluation_provenance,
)
from vigiscoutx.experiment.freeze import FreezeManifestV2, compute_freeze_id
from vigiscoutx.experiment.held_out_guard import (
    HeldOutGuard,
    ObservedHashes,
    ReferenceReconstructionPermit,
)
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import partition_sample_ids
from vigiscoutx.experiment.types import Partition, Phase
from vigiscoutx.ports import AccessPurpose


@dataclasses.dataclass
class Frozen:
    guard: HeldOutGuard
    record: CalibrationFreezeRecord
    manifest: FreezeManifestV2
    provenance: EvaluationProvenance  # REFERENCE_RECONSTRUCTION
    held_out: EvaluationProvenance  # HELD_OUT_EVALUATION (getrennter Nachweis, DEC-087)

    @property
    def reference(self) -> tuple[str, ...]:
        return tuple(self.record.reference_sample_ids)

    def issue(self, **overrides: Any) -> ReferenceReconstructionPermit:
        arguments: dict[str, Any] = {
            "record": self.record,
            "split_manifest": SPLIT,
            "sample_ids": self.reference,
            "provenance": self.provenance,
            **overrides,
        }
        return self.guard.issue_reference_reconstruction_permit(**arguments)


@pytest.fixture
def frozen(
    provenance_repo: ProvenanceRepo, calibrated_config_data: dict[str, Any], tmp_path: Path
) -> Frozen:
    process = CalibrationProcess(
        HeldOutGuard(), SPLIT, provenance(git_commit=provenance_repo.calibration_commit), CONFIG
    )
    advance_to(process, CalibrationStep.VALIDATE)
    record_artifacts(process)
    manifest = manifest_for(process, calibrated_config_data)
    record = process.build_freeze_record()
    guard = HeldOutGuard()
    guard.freeze(manifest)
    evaluation = approve_and_verify(
        provenance_repo, manifest.freeze_id, tmp_path / "approval", purpose=REFERENCE
    )
    held_out = approve_and_verify(
        provenance_repo, manifest.freeze_id, tmp_path / "approval", purpose=HELD_OUT
    )
    assert isinstance(evaluation, EvaluationProvenance)
    assert isinstance(held_out, EvaluationProvenance)
    return Frozen(
        guard=guard, record=record, manifest=manifest, provenance=evaluation, held_out=held_out
    )


# --- Positivfall ---------------------------------------------------------------------------------


def test_permit_allows_exactly_inputs_and_labels_of_the_frozen_reference(frozen: Frozen) -> None:
    permit = frozen.issue()
    assert permit.sample_ids == frozenset(frozen.reference)
    assert len(permit.sample_ids) == frozen.manifest.window_size
    assert permit.purposes == {AccessPurpose.INPUTS, AccessPurpose.LABELS}
    half = len(frozen.reference) // 2
    permit.require_access(frozen.reference[:half], AccessPurpose.INPUTS)  # Batches zulässig
    permit.require_access(frozen.reference[half:], AccessPurpose.INPUTS)
    permit.require_access(frozen.reference, AccessPurpose.LABELS)
    assert frozen.guard.phase is Phase.FROZEN


# --- Negativfälle: Umfang und Zwecke -------------------------------------------------------------


@pytest.mark.parametrize(
    "purpose",
    [AccessPurpose.METADATA, AccessPurpose.STRATIFICATION_LABELS],
)
def test_other_purposes_are_rejected(frozen: Frozen, purpose: AccessPurpose) -> None:
    permit = frozen.issue()
    with pytest.raises(PhaseViolationError, match="keinen Zugriff zum Zweck"):
        permit.require_access(frozen.reference[:1], purpose)


def test_samples_outside_the_frozen_reference_are_rejected(frozen: Frozen) -> None:
    permit = frozen.issue()
    others = sorted(partition_sample_ids(SPLIT, Partition.CALIBRATION) - set(frozen.reference))
    held_out = sorted(partition_sample_ids(SPLIT, Partition.HELD_OUT))
    for foreign in (others[:1], held_out[:1], ["not-a-sample"]):
        with pytest.raises(PhaseViolationError, match="außerhalb der eingefrorenen Reference"):
            permit.require_access(foreign, AccessPurpose.INPUTS)


@pytest.mark.parametrize("mutation", ["missing", "extra", "replaced", "duplicate"])
def test_wrong_reference_ids_are_rejected_at_issuance(frozen: Frozen, mutation: str) -> None:
    ids = list(frozen.reference)
    others = sorted(partition_sample_ids(SPLIT, Partition.CALIBRATION) - set(ids))
    if mutation == "missing":
        ids = ids[:-1]
    elif mutation == "extra":
        ids = [*ids, others[0]]
    elif mutation == "replaced":
        ids = [*ids[:-1], others[0]]
    else:
        ids = [*ids[:-1], ids[0]]
    with pytest.raises(IntegrityError, match="exakt die eingefrorenen Reference-Samples"):
        frozen.issue(sample_ids=tuple(ids))
    assert frozen.issue()  # eine fehlgeschlagene Anfrage verbraucht die Berechtigung nicht


# --- Negativfälle: Bindungen ---------------------------------------------------------------------


def test_missing_or_manipulated_provenance_is_rejected(frozen: Frozen) -> None:
    with pytest.raises(IntegrityError, match="geprüfte Provenienz"):
        frozen.issue(provenance=object())
    with pytest.raises(IntegrityError, match="nicht von verify_evaluation_provenance"):
        frozen.issue(provenance=dataclasses.replace(frozen.provenance))
    object.__setattr__(frozen.provenance, "approval_sha256", "0" * 64)
    with pytest.raises(IntegrityError, match="nach der Ausstellung verändert"):
        frozen.issue()


def test_missing_release_approval_blocks_the_permit(frozen: Frozen) -> None:
    Path(frozen.provenance.approval_path).unlink()
    with pytest.raises(IntegrityError, match="Kein Evaluation-Freigabenachweis"):
        frozen.issue()


def test_wrong_freeze_record_or_split_is_rejected(frozen: Frozen) -> None:
    tampered = frozen.record.model_copy(update={"mmd_sigma": frozen.record.mmd_sigma + 1.0})
    with pytest.raises(IntegrityError):
        frozen.issue(record=tampered)
    other_split = SplitManifest(manifest_id="other-split", assignments=SPLIT.assignments)
    with pytest.raises(IntegrityError, match="Split-Manifest"):
        frozen.issue(split_manifest=other_split)


def test_wrong_freeze_is_rejected(frozen: Frozen) -> None:
    other = frozen.manifest.model_copy(update={"environment_lock_sha256": "9" * 64})
    other = other.model_copy(update={"freeze_id": compute_freeze_id(other)})
    guard = HeldOutGuard()
    guard.freeze(other)
    with pytest.raises(IntegrityError):
        guard.issue_reference_reconstruction_permit(
            record=frozen.record,
            split_manifest=SPLIT,
            sample_ids=frozen.reference,
            provenance=frozen.provenance,
        )


def test_permit_requires_frozen_state(frozen: Frozen) -> None:
    with pytest.raises(PhaseViolationError, match="nur im Zustand FROZEN"):
        HeldOutGuard().issue_reference_reconstruction_permit(
            record=frozen.record,
            split_manifest=SPLIT,
            sample_ids=frozen.reference,
            provenance=frozen.provenance,
        )


# --- Negativfälle: Einmaligkeit, Zweitzugriff, Zustandswechsel, Kopien -------------------------


def test_permit_is_issued_once_and_samples_are_read_once_per_purpose(frozen: Frozen) -> None:
    permit = frozen.issue()
    with pytest.raises(PhaseViolationError, match="bereits einmal berechtigt"):
        frozen.issue()
    permit.require_access(frozen.reference[:4], AccessPurpose.INPUTS)
    with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
        permit.require_access(frozen.reference[2:6], AccessPurpose.INPUTS)
    frozen.guard.close_reference_reconstruction()
    with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
        permit.require_access(frozen.reference[10:12], AccessPurpose.LABELS)
    with pytest.raises(PhaseViolationError, match="bereits einmal berechtigt"):
        frozen.issue()


def _held_out_config(frozen: Frozen, calibrated_config_data: dict[str, Any]) -> RunConfig:
    """HELD_OUT-Konfiguration exakt zum synthetischen Freeze (wie `manifest_for`)."""
    record = frozen.record
    data = with_frozen_policies(calibrated_config_data)
    scientific = data["scientific"]
    scientific["dataset"]["sha256"] = record.dataset_snapshot_sha256
    scientific["split"]["sha256"] = record.split_manifest_sha256
    scientific["model"]["sha256"] = record.model_weights_sha256
    scientific["monitoring"]["window_size"] = record.window_size
    scientific["relevance"]["delta_rel"] = record.delta_rel
    scientific["fusion"]["rule"] = frozen.manifest.fusion.rule
    data.update(
        repetitions=record.final_r, phase="HELD_OUT_EVALUATION", freeze_id=frozen.manifest.freeze_id
    )
    return RunConfig.model_validate(data)


def test_permit_expires_with_state_change_and_frozen_lock_remains(
    frozen: Frozen, calibrated_config_data: dict[str, Any]
) -> None:
    permit = frozen.issue()
    for partition in Partition:  # alle übrigen Zugriffspfade bleiben in FROZEN gesperrt
        with pytest.raises(PhaseViolationError, match="FROZEN"):
            frozen.guard.assert_partition_access(partition)
    manifest = frozen.manifest
    observed = ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=manifest.split_manifest_sha256,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=manifest.environment_lock_sha256,
    )
    frozen.guard.open_held_out(
        _held_out_config(frozen, calibrated_config_data), observed, frozen.held_out
    )
    assert frozen.guard.phase is Phase.HELD_OUT_EVALUATION
    with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
        permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS)
    with pytest.raises(PhaseViolationError, match="nur im Zustand FROZEN"):
        frozen.issue()


def test_permit_cannot_be_copied_serialised_or_constructed(frozen: Frozen) -> None:
    permit = frozen.issue()
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(permit)
    with pytest.raises(PhaseViolationError, match="nur vom HeldOutGuard"):
        ReferenceReconstructionPermit(
            freeze_id=permit.freeze_id,
            sample_ids=permit.sample_ids,
            sample_ids_sha256=permit.sample_ids_sha256,
            purposes=permit.purposes,
            issue_token=object(),
        )
    forged = dataclasses.replace(permit)
    with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
        forged.require_access(frozen.reference[:1], AccessPurpose.INPUTS)


# --- P4-03AA: HIGH – Verbrauchszustand ausschließlich guardintern ------------------------------


def test_consumption_state_is_not_exposed_and_former_attacks_are_ineffective(
    frozen: Frozen,
) -> None:
    """P4-03Z: `accessed.clear()`, Leeren eines Sets, Ersetzen eines Dict-Eintrags."""
    permit = frozen.issue()
    permit.require_access(frozen.reference[:3], AccessPurpose.INPUTS)
    assert not hasattr(permit, "accessed")
    attacks: list[dict[AccessPurpose, set[str]]] = [
        {},
        {AccessPurpose.INPUTS: set()},
        {AccessPurpose.INPUTS: {"x"}},
    ]
    for attack in attacks:
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(permit, "accessed", attack)
    with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
        permit.require_access(frozen.reference[:3], AccessPurpose.INPUTS)


def test_mutating_informative_permit_fields_grants_nothing(frozen: Frozen) -> None:
    permit = frozen.issue()
    others = sorted(partition_sample_ids(SPLIT, Partition.CALIBRATION) - set(frozen.reference))
    object.__setattr__(permit, "sample_ids", frozenset([*frozen.reference, *others]))
    object.__setattr__(permit, "purposes", frozenset(AccessPurpose))
    with pytest.raises(PhaseViolationError, match="außerhalb der eingefrorenen Reference"):
        permit.require_access(others[:1], AccessPurpose.INPUTS)
    with pytest.raises(PhaseViolationError, match="keinen Zugriff zum Zweck"):
        permit.require_access(frozen.reference[:1], AccessPurpose.METADATA)


def test_failed_access_consumes_and_releases_nothing(frozen: Frozen) -> None:
    permit = frozen.issue()
    permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS)
    others = sorted(partition_sample_ids(SPLIT, Partition.CALIBRATION) - set(frozen.reference))
    for batch in (
        [frozen.reference[0], frozen.reference[1]],  # bereits verbraucht + neu
        [frozen.reference[1], others[0]],  # in der Reference + fremd
    ):
        with pytest.raises(PhaseViolationError):
            permit.require_access(batch, AccessPurpose.INPUTS)
    permit.require_access(frozen.reference[1:2], AccessPurpose.INPUTS)  # nicht verbraucht worden
    permit.require_access(frozen.reference[:2], AccessPurpose.LABELS)  # Zwecke getrennt


def test_disjoint_batches_cover_the_whole_reference_once_per_purpose(frozen: Frozen) -> None:
    permit = frozen.issue()
    for purpose in (AccessPurpose.INPUTS, AccessPurpose.LABELS):
        for start in range(0, len(frozen.reference), 32):
            permit.require_access(frozen.reference[start : start + 32], purpose)
        with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
            permit.require_access(frozen.reference[-1:], purpose)


def test_replaced_permit_object_is_never_accepted(frozen: Frozen) -> None:
    permit = frozen.issue()
    clone = dataclasses.replace(permit)
    with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
        clone.require_access(frozen.reference[:1], AccessPurpose.INPUTS)
    permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS)


# --- P4-03AA: MEDIUM – Neuprüfung der Provenienz bei jedem Zugriff -----------------------------


@pytest.mark.parametrize(
    "tamper", ["approval-removed", "approval-modified", "dirty-file", "new-commit", "provenance"]
)
def test_stale_release_or_repository_change_blocks_access_immediately(
    frozen: Frozen, provenance_repo: ProvenanceRepo, tamper: str
) -> None:
    permit = frozen.issue()
    permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS)
    approval = Path(frozen.provenance.approval_path)
    if tamper == "approval-removed":
        approval.unlink()
    elif tamper == "approval-modified":
        approval.write_text(
            approval.read_text(encoding="utf-8").replace("synthetic", "forged"), "utf-8"
        )
    elif tamper == "dirty-file":
        (provenance_repo.root / "src/vigiscoutx/signals/method.py").write_text(
            "ALPHA = 0.1\n", "utf-8"
        )
    elif tamper == "new-commit":
        provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# after issuance\n"})
    else:
        object.__setattr__(frozen.provenance, "approval_sha256", "0" * 64)
    with pytest.raises(IntegrityError):
        permit.require_access(frozen.reference[1:2], AccessPurpose.INPUTS)
    with pytest.raises(IntegrityError):
        permit.require_access(frozen.reference[1:2], AccessPurpose.LABELS)


# --- P4-03AC: HIGH – parallele Zugriffe auf dasselbe (sample_id, purpose) ------------------------

_WAIT = 10.0  # Sicherheitsobergrenze gegen Hänger; das Ergebnis hängt nicht vom Timing ab


class _Interleaving:
    """Hält den ersten Zugriff innerhalb des kritischen Abschnitts (Provenienz-Neuprüfung) an.

    Solange der erste Zugriff angehalten ist, versuchen weitere Threads parallel zuzugreifen;
    ohne guardinterne Sperre liefen sie ungehindert durch, mit Sperre müssen sie warten.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._original = reverify_evaluation_provenance
        self._mutex = threading.Lock()
        self.log: list[str] = []
        self.arm()
        monkeypatch.setattr(held_out_guard, "reverify_evaluation_provenance", self._reverify)

    def arm(self, *, fail_first: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_first = fail_first
        self.calls = 0

    def _reverify(self, *args: Any, **kwargs: Any) -> None:
        with self._mutex:
            self.calls += 1
            first = self.calls == 1
            self.log.append(f"reverify:{threading.current_thread().name}")
        if first:
            self.entered.set()
            assert self.release.wait(_WAIT)
            if self.fail_first:
                raise IntegrityError("synthetischer Provenienzfehler während des Zugriffs")
        self._original(*args, **kwargs)


class _Race:
    def __init__(self) -> None:
        self.results: dict[str, object] = {}
        self.threads: list[threading.Thread] = []

    def start(self, name: str, call: Callable[[], object]) -> threading.Thread:
        def run() -> None:
            try:
                call()
                self.results[name] = "ok"
            except Exception as error:  # Ergebnis wird im Test geprüft
                self.results[name] = error

        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def blocked(self, thread: threading.Thread) -> bool:
        thread.join(timeout=0.2)
        return thread.is_alive()

    def finish(self, interleaving: _Interleaving) -> None:
        interleaving.release.set()
        for thread in self.threads:
            thread.join(_WAIT)
            assert not thread.is_alive()


def _held_race(
    frozen: Frozen,
    interleaving: _Interleaving,
    first: tuple[str, AccessPurpose],
    second: tuple[str, AccessPurpose],
    permit: ReferenceReconstructionPermit,
) -> dict[str, object]:
    """A wird im kritischen Abschnitt angehalten, B greift währenddessen parallel zu."""
    race = _Race()
    race.start("A", lambda: permit.require_access([first[0]], first[1]))
    assert interleaving.entered.wait(_WAIT)
    b = race.start("B", lambda: permit.require_access([second[0]], second[1]))
    try:
        assert race.blocked(b), "B lief während des angehaltenen Zugriffs A durch"
    finally:
        race.finish(interleaving)
    return race.results


def _access(
    permit: ReferenceReconstructionPermit,
    sample: str,
    purpose: AccessPurpose,
    barrier: threading.Barrier | None = None,
) -> None:
    if barrier is not None:
        barrier.wait(_WAIT)
    permit.require_access([sample], purpose)


def _is_second_access(result: object) -> bool:
    return isinstance(result, PhaseViolationError) and "zweiter Zugriff" in str(result)


def test_parallel_access_to_the_same_sample_and_purpose_succeeds_exactly_once(
    frozen: Frozen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P4-03AB: wiederholt unter kontrollierter Verzahnung, je Sample ein neues Rennen."""
    permit = frozen.issue()  # Ausstellung vor der Instrumentierung
    interleaving = _Interleaving(monkeypatch)
    for sample in frozen.reference[:10]:
        interleaving.arm()
        key = (sample, AccessPurpose.INPUTS)
        results = _held_race(frozen, interleaving, key, key, permit)
        assert results["A"] == "ok"
        assert _is_second_access(results["B"])
        with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
            permit.require_access([sample], AccessPurpose.INPUTS)


def test_many_simultaneous_attempts_on_one_key_yield_one_success(frozen: Frozen) -> None:
    permit = frozen.issue()
    for sample in frozen.reference[:5]:
        barrier = threading.Barrier(8)
        race = _Race()
        for index in range(8):
            race.start(
                f"T{index}",
                partial(_access, permit, sample, AccessPurpose.LABELS, barrier),
            )
        for thread in race.threads:
            thread.join(_WAIT)
        outcomes = list(race.results.values())
        assert outcomes.count("ok") == 1
        assert sum(_is_second_access(outcome) for outcome in outcomes) == 7


def test_parallel_access_to_different_samples_is_both_allowed(
    frozen: Frozen, monkeypatch: pytest.MonkeyPatch
) -> None:
    permit = frozen.issue()  # Ausstellung vor der Instrumentierung
    interleaving = _Interleaving(monkeypatch)
    first, second = frozen.reference[:2]
    results = _held_race(
        frozen, interleaving, (first, AccessPurpose.INPUTS), (second, AccessPurpose.INPUTS), permit
    )
    assert results == {"A": "ok", "B": "ok"}


def test_parallel_inputs_and_labels_for_one_sample_stay_separately_allowed(
    frozen: Frozen, monkeypatch: pytest.MonkeyPatch
) -> None:
    permit = frozen.issue()  # Ausstellung vor der Instrumentierung
    interleaving = _Interleaving(monkeypatch)
    sample = frozen.reference[0]
    results = _held_race(
        frozen, interleaving, (sample, AccessPurpose.INPUTS), (sample, AccessPurpose.LABELS), permit
    )
    assert results == {"A": "ok", "B": "ok"}
    for purpose in (AccessPurpose.INPUTS, AccessPurpose.LABELS):
        with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
            permit.require_access([sample], purpose)


def test_provenance_failure_during_access_stops_without_consumption(
    frozen: Frozen, monkeypatch: pytest.MonkeyPatch
) -> None:
    permit = frozen.issue()
    interleaving = _Interleaving(monkeypatch)
    interleaving.arm(fail_first=True)
    key = (frozen.reference[0], AccessPurpose.INPUTS)
    results = _held_race(frozen, interleaving, key, key, permit)
    assert isinstance(results["A"], IntegrityError)  # STOP ohne Verbrauch …
    assert results["B"] == "ok"  # … daher ist das Paar für den serialisierten Zugriff B noch frei
    with pytest.raises(PhaseViolationError, match="zweiter Zugriff"):
        permit.require_access([key[0]], key[1])


@pytest.mark.parametrize("change", ["close", "open_held_out"])
def test_revoke_or_state_change_during_parallel_attempts_grants_nothing_late(
    frozen: Frozen,
    monkeypatch: pytest.MonkeyPatch,
    calibrated_config_data: dict[str, Any],
    change: str,
) -> None:
    permit = frozen.issue()  # Ausstellung vor der Instrumentierung
    interleaving = _Interleaving(monkeypatch)
    manifest = frozen.manifest
    config = _held_out_config(frozen, calibrated_config_data)
    observed = ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=manifest.split_manifest_sha256,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=manifest.environment_lock_sha256,
    )

    def revoke() -> None:
        if change == "close":
            frozen.guard.close_reference_reconstruction()
        else:
            frozen.guard.open_held_out(config, observed, frozen.held_out)
        interleaving.log.append("revoked")

    race = _Race()
    race.start("A", lambda: permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS))
    assert interleaving.entered.wait(_WAIT)
    revoker = race.start("R", revoke)
    try:
        assert race.blocked(revoker), "Widerruf überholte einen laufenden Zugriff"
        for index, sample in enumerate(frozen.reference[1:4]):
            race.start(f"B{index}", partial(_access, permit, sample, AccessPurpose.INPUTS))
    finally:
        race.finish(interleaving)
    assert race.results["A"] == "ok"  # vor dem Widerruf begonnen und vollständig geprüft
    assert race.results["R"] == "ok"
    revoked_at = interleaving.log.index("revoked")
    for name, outcome in race.results.items():
        if name.startswith("B"):
            if outcome == "ok":  # nur zulässig, wenn vollständig vor dem Widerruf geprüft
                assert interleaving.log.index(f"reverify:{name}") < revoked_at
            else:
                assert isinstance(outcome, PhaseViolationError)
                assert "nicht mehr aktiv" in str(outcome)
                assert f"reverify:{name}" not in interleaving.log
    for sample in frozen.reference[4:6]:
        with pytest.raises(PhaseViolationError, match="nicht mehr aktiv"):
            permit.require_access([sample], AccessPurpose.LABELS)


# --- P4-03AM: Phasengrenzen im Guard (DEC-087) --------------------------------------------------


def test_reference_approval_never_opens_held_out_and_vice_versa(
    frozen: Frozen, calibrated_config_data: dict[str, Any]
) -> None:
    manifest = frozen.manifest
    observed = ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=manifest.split_manifest_sha256,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=manifest.environment_lock_sha256,
    )
    config = _held_out_config(frozen, calibrated_config_data)
    with pytest.raises(IntegrityError, match="HELD_OUT_EVALUATION"):
        frozen.guard.open_held_out(config, observed, frozen.provenance)
    with pytest.raises(IntegrityError, match="REFERENCE_RECONSTRUCTION"):
        frozen.issue(provenance=frozen.held_out)
    assert frozen.guard.phase is Phase.FROZEN
    frozen.issue()  # weder Fehlversuch hat die Reference-Berechtigung verbraucht


def test_held_out_must_use_the_evaluation_commit_of_the_reference_phase(
    frozen: Frozen,
    provenance_repo: ProvenanceRepo,
    calibrated_config_data: dict[str, Any],
    tmp_path: Path,
) -> None:
    permit = frozen.issue()
    permit.require_access(frozen.reference[:1], AccessPurpose.INPUTS)
    frozen.guard.close_reference_reconstruction()
    provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# after reference phase\n"})
    later = approve_and_verify(
        provenance_repo, frozen.manifest.freeze_id, tmp_path / "later", purpose=HELD_OUT
    )
    assert isinstance(later, EvaluationProvenance)
    assert later.evaluation_commit != frozen.provenance.evaluation_commit
    manifest = frozen.manifest
    observed = ObservedHashes(
        dataset_snapshot_sha256=manifest.dataset_snapshot_sha256,
        split_manifest_sha256=manifest.split_manifest_sha256,
        model_weights_sha256=manifest.model_weights_sha256,
        environment_lock_sha256=manifest.environment_lock_sha256,
    )
    with pytest.raises(IntegrityError, match="denselben Evaluation-Commit"):
        frozen.guard.open_held_out(
            _held_out_config(frozen, calibrated_config_data), observed, later
        )
    assert frozen.guard.phase is Phase.FROZEN

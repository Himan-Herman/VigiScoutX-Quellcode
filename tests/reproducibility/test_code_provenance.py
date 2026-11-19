"""P4-03Q/S: Post-Freeze-Code-Provenienz (DEC-082), ausschließlich synthetische Git-Repositories.

Calibration-Commit (Freeze-Provenienz) und tatsächlicher Evaluation-Commit sind getrennt. HEAD,
Working Tree, Repository-Identität, Dateitypen und der separate Freigabenachweis werden
bei Ausstellung und unmittelbar vor `open_held_out` erneut geprüft. Die vier P4-03R-Befunde
(manipulierte Provenienz, Repository-Bindung/TOCTOU, Symlinks/Dateitypen, Selbstfreigabe) sind
als gezielte Negativtests abgebildet.
"""

import dataclasses
import os
from pathlib import Path

import pytest

from provenance_repo import (
    GUARD_PATH,
    HELD_OUT,
    ProvenanceRepo,
    create_provenance_repo,
    git,
    write_approval,
)
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment import code_provenance
from vigiscoutx.experiment.code_provenance import (
    POST_FREEZE_CHANGE_POLICY,
    EvaluationProvenance,
    ExecutionPurpose,
    change_violation,
    verify_evaluation_provenance,
)
from vigiscoutx.experiment.freeze import FreezeManifest
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import Phase

FREEZE = "f" * 64


def _approve(
    repo: ProvenanceRepo,
    directory: Path,
    freeze_id: str = FREEZE,
    overrides: dict[str, object] | None = None,
    purpose: str = HELD_OUT,
) -> Path:
    changes = dict(overrides or {})
    commit = str(changes.pop("evaluation_commit", repo.git("rev-parse", "HEAD")))
    return write_approval(
        directory,
        evaluation_commit=commit,
        calibration_commit=repo.calibration_commit,
        freeze_id=freeze_id,
        purpose=purpose,
        overrides=changes,
    )


def _verify(
    repo: ProvenanceRepo,
    approval: Path,
    freeze_id: str = FREEZE,
    purpose: ExecutionPurpose = ExecutionPurpose.HELD_OUT_EVALUATION,
) -> EvaluationProvenance:
    return verify_evaluation_provenance(
        calibration_commit=repo.calibration_commit,
        freeze_id=freeze_id,
        approval_path=approval,
        purpose=purpose,
    )


def _frozen(manifest: FreezeManifest) -> HeldOutGuard:
    guard = HeldOutGuard()
    guard.freeze(manifest)
    return guard


# --- Positivfälle (synthetisch) ------------------------------------------------------------------


def test_documentation_head_with_independent_approval_opens_held_out(
    provenance_repo: ProvenanceRepo,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    tmp_path: Path,
) -> None:
    head = provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# docs only\n"})
    approval = _approve(provenance_repo, tmp_path / "a", evaluation_manifest.freeze_id)
    provenance = _verify(provenance_repo, approval, evaluation_manifest.freeze_id)
    assert provenance.evaluation_commit == head != provenance.calibration_commit
    assert provenance.policy_version == POST_FREEZE_CHANGE_POLICY
    guard = _frozen(evaluation_manifest)
    session = guard.open_held_out(evaluation_held_out_config, observed, provenance)
    assert session.freeze_id == evaluation_manifest.freeze_id


def test_new_evaluation_commit_within_policy_is_accepted(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    provenance_repo.commit(
        {
            "src/vigiscoutx/experiment/held_out_guard.py": "GUARD = 2\n",
            "src/vigiscoutx/experiment/code_provenance.py": "POLICY = 1\n",
            "src/vigiscoutx/evaluation/metrics.py": "def f(): ...\n",
            "experiments/runners/held_out_runner/run.py": "RUN = 1\n",
            "experiments/configs/evaluation/held_out.yaml": "run_id: eval\n",
            "tests/test_new.py": "def test(): ...\n",
        }
    )
    provenance_repo.commit({"src/vigiscoutx/evaluation/metrics.py": "def g(): ...\n"})
    provenance = _verify(provenance_repo, _approve(provenance_repo, tmp_path / "a"))
    assert (
        dict((p, s) for s, p in provenance.changed_paths)["src/vigiscoutx/evaluation/metrics.py"]
        == "A"
    )


# --- Befund 1: manipulierte Provenienz -----------------------------------------------------------


def test_replaced_or_mutated_provenance_never_opens_held_out(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    guard = _frozen(evaluation_manifest)
    for forged in (
        dataclasses.replace(evaluation_provenance),
        dataclasses.replace(evaluation_provenance, evaluation_commit="b" * 40),
    ):
        with pytest.raises(IntegrityError, match="nicht von verify_evaluation_provenance"):
            guard.open_held_out(evaluation_held_out_config, observed, forged)
    object.__setattr__(evaluation_provenance, "approval_sha256", "0" * 64)
    with pytest.raises(IntegrityError, match="nach der Ausstellung verändert"):
        guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    assert guard.phase is Phase.FROZEN


def test_provenance_cannot_be_constructed_directly() -> None:
    with pytest.raises(IntegrityError, match="verify_evaluation_provenance"):
        EvaluationProvenance(
            repository="/tmp/x",
            calibration_commit="a" * 40,
            evaluation_commit="a" * 40,
            freeze_id=FREEZE,
            policy_version=POST_FREEZE_CHANGE_POLICY,
            purpose=ExecutionPurpose.HELD_OUT_EVALUATION,
            reference_evidence_sha256="e" * 64,
            approval_path="/tmp/a.json",
            approval_sha256="0" * 64,
            changed_paths=(),
            token=object(),
        )


# --- Befund 2: Repository-Bindung und TOCTOU -----------------------------------------------------


GIT_REDIRECTS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
)


def _foreign_clean_clone(provenance_repo: ProvenanceRepo, tmp_path: Path) -> Path:
    """Saubere, policy-konforme Fremdkopie auf dem Calibration-Commit."""
    clone = tmp_path / "foreign-clone"
    git(tmp_path, "clone", "-q", str(provenance_repo.root), str(clone))
    return clone


def _redirect(monkeypatch: pytest.MonkeyPatch, clone: Path, names: tuple[str, ...]) -> None:
    targets = {
        "GIT_DIR": clone / ".git",
        "GIT_WORK_TREE": clone,
        "GIT_INDEX_FILE": clone / ".git/index",
        "GIT_COMMON_DIR": clone / ".git",
        "GIT_OBJECT_DIRECTORY": clone / ".git/objects",
    }
    for name in names:
        monkeypatch.setenv(name, str(targets[name]))


@pytest.mark.parametrize(
    "names",
    [*((name,) for name in GIT_REDIRECTS), GIT_REDIRECTS],
    ids=[*GIT_REDIRECTS, "combined"],
)
def test_git_environment_cannot_redirect_to_a_foreign_repository(
    provenance_repo: ProvenanceRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
) -> None:
    """Das ausgeführte Repository verletzt die Policy; eine saubere Fremdkopie darf das nicht
    über `GIT_*`-Umleitungen verdecken – vor Ausstellung einer Provenienz."""
    clone = _foreign_clean_clone(provenance_repo, tmp_path)
    provenance_repo.commit({"src/vigiscoutx/signals/method.py": "ALPHA = 0.10\n"})
    foreign_head = git(clone, "rev-parse", "HEAD")
    approval = _approve(
        provenance_repo, tmp_path / "a", overrides={"evaluation_commit": foreign_head}
    )
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.worktree'='" + str(clone) + "'")
    _redirect(monkeypatch, clone, names)
    with pytest.raises(IntegrityError, match="Unzulässige Änderungen"):
        _verify(provenance_repo, approval)


@pytest.mark.parametrize("names", [("GIT_DIR", "GIT_WORK_TREE"), GIT_REDIRECTS], ids=["dir", "all"])
def test_git_environment_does_not_change_the_verified_repository(
    provenance_repo: ProvenanceRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
) -> None:
    clone = _foreign_clean_clone(provenance_repo, tmp_path)
    head = provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# executing repo\n"})
    approval = _approve(provenance_repo, tmp_path / "a")
    _redirect(monkeypatch, clone, names)
    provenance = _verify(provenance_repo, approval)
    assert provenance.evaluation_commit == head != git(clone, "rev-parse", "HEAD")
    assert provenance.repository == str(provenance_repo.root.resolve())


def test_redirect_between_issuance_and_use_cannot_hide_changes(
    evaluation_provenance: EvaluationProvenance,
    provenance_repo: ProvenanceRepo,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _foreign_clean_clone(provenance_repo, tmp_path)
    (provenance_repo.root / "src/vigiscoutx/signals/method.py").write_text(
        "ALPHA = 0.10\n", "utf-8"
    )
    _redirect(monkeypatch, clone, GIT_REDIRECTS)
    guard = _frozen(evaluation_manifest)
    with pytest.raises(IntegrityError, match="Working Tree nicht sauber"):
        guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    assert guard.phase is Phase.FROZEN


@pytest.mark.parametrize("location", ["outside-repo", "untracked-in-repo"])
def test_foreign_or_untracked_module_location_is_rejected_before_issuance(
    provenance_repo: ProvenanceRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    approval = _approve(provenance_repo, tmp_path / "a")
    if location == "outside-repo":
        outside = tmp_path / "elsewhere" / "held_out_guard.py"
        outside.parent.mkdir(parents=True)
        outside.write_text("GUARD = 1\n", encoding="utf-8")
        monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", outside)
        match = "Repository"
    else:
        ignored = provenance_repo.root / ".git" / "info" / "exclude"
        ignored.write_text("shadow.py\n", encoding="utf-8")
        shadow = provenance_repo.root / "shadow.py"
        shadow.write_text("X = 1\n", encoding="utf-8")
        monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", shadow)
        match = "nicht versioniert"
    with pytest.raises(IntegrityError, match=match):
        _verify(provenance_repo, approval)


def test_foreign_repository_copy_is_rejected_as_executing_repository(
    provenance_repo: ProvenanceRepo,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenienz einer Fremdkopie gilt nicht für das tatsächlich ausgeführte Repository."""
    clone = _foreign_clean_clone(provenance_repo, tmp_path)
    approval = _approve(provenance_repo, tmp_path / "a", evaluation_manifest.freeze_id)
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", clone / GUARD_PATH)
    foreign = _verify(provenance_repo, approval, evaluation_manifest.freeze_id)
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", provenance_repo.root / GUARD_PATH)
    with pytest.raises(IntegrityError, match="nicht zum ausgeführten Repository"):
        _frozen(evaluation_manifest).open_held_out(evaluation_held_out_config, observed, foreign)


@pytest.mark.parametrize(
    "tamper",
    ["dirty-file", "untracked-file", "new-commit", "approval-replaced", "assume-unchanged"],
)
def test_changes_after_provenance_check_stop_open_held_out(
    evaluation_provenance: EvaluationProvenance,
    provenance_repo: ProvenanceRepo,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
    tamper: str,
) -> None:
    root = provenance_repo.root
    science = root / "src/vigiscoutx/signals/method.py"
    if tamper == "dirty-file":
        science.write_text("ALPHA = 0.10\n", encoding="utf-8")
    elif tamper == "untracked-file":
        (root / "src/vigiscoutx/signals/extra.py").write_text("X = 1\n", encoding="utf-8")
    elif tamper == "new-commit":
        provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# after check\n"})
    elif tamper == "approval-replaced":
        Path(evaluation_provenance.approval_path).write_text("{}", encoding="utf-8")
    else:
        provenance_repo.git(
            "update-index", "--assume-unchanged", "src/vigiscoutx/signals/method.py"
        )
        science.write_text("ALPHA = 0.10\n", encoding="utf-8")
    guard = _frozen(evaluation_manifest)
    with pytest.raises(IntegrityError):
        guard.open_held_out(evaluation_held_out_config, observed, evaluation_provenance)
    assert guard.phase is Phase.FROZEN


def test_unclean_working_tree_stops_at_issuance(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    approval = _approve(provenance_repo, tmp_path / "a")
    (provenance_repo.root / "untracked.txt").write_text("x", encoding="utf-8")
    with pytest.raises(IntegrityError, match="Working Tree nicht sauber"):
        _verify(provenance_repo, approval)


def test_head_must_descend_from_the_calibration_commit(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    later = provenance_repo.commit({"docs/00_scope/PROJECT_STATUS.md": "# later\n"})
    provenance_repo.git("checkout", "-q", provenance_repo.calibration_commit)
    repo = ProvenanceRepo(root=provenance_repo.root, calibration_commit=later)
    with pytest.raises(IntegrityError, match="stammt nicht vom Calibration-Commit ab"):
        _verify(repo, _approve(repo, tmp_path / "a"))


# --- Befund 3: Symlinks, Gitlinks, ausführbare Dateien, Typwechsel ------------------------------


def _symlink_commit(repo: ProvenanceRepo, path: str, target: str) -> None:
    link = repo.root / path
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "symlink")


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("docs/external.md", "/etc/hosts"),
        ("src/vigiscoutx/evaluation/runner.py", "../signals/method.py"),
        ("docs/00_scope/PROJECT_STATUS.md", "../../src/vigiscoutx/signals/method.py"),
    ],
    ids=["external-target", "evaluation-namespace", "type-change"],
)
def test_symlinks_and_type_changes_are_rejected(
    provenance_repo: ProvenanceRepo, tmp_path: Path, path: str, target: str
) -> None:
    _symlink_commit(provenance_repo, path, target)
    with pytest.raises(IntegrityError):
        _verify(provenance_repo, _approve(provenance_repo, tmp_path / "a"))


def test_gitlink_is_rejected(provenance_repo: ProvenanceRepo, tmp_path: Path) -> None:
    provenance_repo.git(
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{provenance_repo.calibration_commit},docs/sub",
    )
    provenance_repo.git("commit", "-q", "-m", "gitlink")
    with pytest.raises(IntegrityError, match="Symlinks/Gitlinks"):
        _verify(provenance_repo, _approve(provenance_repo, tmp_path / "a"))


def test_executable_file_in_allowed_path_is_rejected(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    script = provenance_repo.root / "tests/run.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(script, 0o755)
    provenance_repo.git("add", "-A")
    provenance_repo.git("commit", "-q", "-m", "executable")
    with pytest.raises(IntegrityError, match="100755"):
        _verify(provenance_repo, _approve(provenance_repo, tmp_path / "a"))


# --- Befund 4: unabhängiger Freigabenachweis statt übergebenem HEAD ----------------------------


def test_missing_approval_keeps_access_locked(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    with pytest.raises(IntegrityError, match="Kein Evaluation-Freigabenachweis"):
        _verify(provenance_repo, tmp_path / "missing.json")


@pytest.mark.parametrize(
    "overrides",
    [
        {"evaluation_commit": "b" * 40},
        {"freeze_id": "e" * 64},
        {"policy_version": "post-freeze-change-policy:1.0"},
        {"author_release": "PENDING"},
    ],
    ids=["other-commit", "other-freeze", "other-policy", "not-released"],
)
def test_approval_must_bind_exactly_commit_freeze_and_policy(
    provenance_repo: ProvenanceRepo, tmp_path: Path, overrides: dict[str, object]
) -> None:
    freeze = str(overrides.pop("freeze_id", FREEZE))
    approval = _approve(provenance_repo, tmp_path / "a", freeze, overrides)
    with pytest.raises(IntegrityError, match="Freigabenachweis"):
        _verify(provenance_repo, approval)


def test_tracked_approval_inside_repository_is_rejected(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    head = provenance_repo.git("rev-parse", "HEAD")
    inside = _approve(
        provenance_repo, provenance_repo.root / "docs", overrides={"evaluation_commit": head}
    )
    provenance_repo.git("add", "-A")
    provenance_repo.git("commit", "-q", "-m", "self-approval")
    with pytest.raises(IntegrityError):
        _verify(provenance_repo, inside)


def test_real_repository_has_no_approval_and_stays_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne unabhängigen Nachweis ist für das Projekt-Repository keine Provenienz erzeugbar."""
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", Path(code_provenance.__file__))
    with pytest.raises(IntegrityError):
        verify_evaluation_provenance(
            calibration_commit="5ad433a22cf176aa1846d5030b393837b4454c7b",
            freeze_id="58b1e879f3adc03f9f84b10810f9ed727ab0eba41bbde891786c8f529dc52a93",
            approval_path=Path("/nonexistent/evaluation-approval.json"),
            purpose=ExecutionPurpose.HELD_OUT_EVALUATION,
        )


# --- Policy (deny by default) --------------------------------------------------------------------


@pytest.mark.parametrize(
    "files",
    [
        {"src/vigiscoutx/signals/method.py": "ALPHA = 0.10\n"},
        {"experiments/configs/calibration/template.yaml": "run_id: other\n"},
        {"experiments/runners/calibration_runner/orchestrator.py": "STEPS = 27\n"},
        {"requirements-lock.txt": "numpy==2.1\n"},
        {"docs/06_experiments/MONITORING_METHODS_BASELINE.md": "# changed\n"},
        {"src/vigiscoutx/new_helper.py": "X = 1\n"},
        {"src/vigiscoutx/experiment/held_out_guard.py": None},
    ],
    ids=["science", "config", "runner", "lock", "baseline", "unscoped", "guard-deleted"],
)
def test_scientific_or_unscoped_changes_stop(
    provenance_repo: ProvenanceRepo, tmp_path: Path, files: dict[str, str | None]
) -> None:
    provenance_repo.commit(files)
    # Die gelöschte Guard-Datei ist im Test zugleich das ausgeführte Modul (Fundort-Bindung).
    match = "nicht auffindbar" if GUARD_PATH in files else "Unzulässige Änderungen"
    with pytest.raises(IntegrityError, match=match):
        _verify(provenance_repo, _approve(provenance_repo, tmp_path / "a"))


def test_evaluation_namespace_never_modifies_frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = create_provenance_repo(tmp_path / "repo")
    frozen = base.commit({"src/vigiscoutx/evaluation/existing.py": "X = 1\n"}, "freeze")
    repo = ProvenanceRepo(root=base.root, calibration_commit=frozen)
    repo.commit({"src/vigiscoutx/evaluation/existing.py": "X = 2\n"})
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", repo.root / GUARD_PATH)
    with pytest.raises(IntegrityError, match="nur neue Dateien"):
        _verify(repo, _approve(repo, tmp_path / "a"))


@pytest.mark.parametrize(
    ("status", "path", "allowed"),
    [
        ("M", "docs/00_scope/DECISION_LOG.md", True),
        ("M", "docs/06_experiments/CALIBRATION_PROTOCOL_BASELINE.md", False),
        ("A", "src/vigiscoutx/evaluation/runner.py", True),
        ("M", "src/vigiscoutx/evaluation/runner.py", False),
        ("T", "docs/00_scope/DECISION_LOG.md", False),
        ("M", "src/vigiscoutx/experiment/held_out_guard.py", True),
        ("M", "src/vigiscoutx/risk/mapper.py", False),
        ("M", "pyproject.toml", False),
    ],
)
def test_policy_is_deny_by_default(status: str, path: str, allowed: bool) -> None:
    assert (change_violation(status, path) is None) is allowed


def test_guard_rejects_missing_provenance_and_wrong_freeze_id(
    evaluation_provenance: EvaluationProvenance,
    evaluation_manifest: FreezeManifest,
    evaluation_held_out_config: RunConfig,
    observed: ObservedHashes,
) -> None:
    guard = _frozen(evaluation_manifest)
    with pytest.raises(IntegrityError, match="geprüfte EvaluationProvenance"):
        guard.open_held_out(evaluation_held_out_config, observed, object())  # type: ignore[arg-type]
    wrong = RunConfig.model_validate(
        {**evaluation_held_out_config.model_dump(mode="json"), "freeze_id": "c" * 64}
    )
    with pytest.raises(IntegrityError, match="freeze_id"):
        guard.open_held_out(wrong, observed, evaluation_provenance)
    assert guard.phase is Phase.FROZEN


def test_observed_hashes_no_longer_accept_a_caller_commit() -> None:
    assert "git_commit" not in ObservedHashes.model_fields


# --- P4-03AM: zweckgebundene Freigabe, Schema 2 (DEC-087) ---------------------------------------


@pytest.mark.parametrize(
    ("issued", "requested"),
    [
        ("REFERENCE_RECONSTRUCTION", ExecutionPurpose.HELD_OUT_EVALUATION),
        ("HELD_OUT_EVALUATION", ExecutionPurpose.REFERENCE_RECONSTRUCTION),
    ],
)
def test_approval_for_the_other_phase_authorizes_nothing(
    provenance_repo: ProvenanceRepo, tmp_path: Path, issued: str, requested: ExecutionPurpose
) -> None:
    approval = _approve(provenance_repo, tmp_path, purpose=issued)
    with pytest.raises(IntegrityError, match="purpose"):
        _verify(provenance_repo, approval, purpose=requested)
    provenance = _verify(provenance_repo, approval, purpose=ExecutionPurpose(issued))
    assert provenance.purpose is ExecutionPurpose(issued)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 1, "purpose": None, "reference_evidence_sha256": None},
        {"schema_version": 3},
        {"purpose": "BOTH"},
        {"reference_evidence_sha256": None},  # HELD_OUT ohne Reference-Evidence
    ],
)
def test_legacy_unknown_or_unbound_approvals_are_rejected(
    provenance_repo: ProvenanceRepo, tmp_path: Path, overrides: dict[str, object]
) -> None:
    approval = _approve(provenance_repo, tmp_path, overrides=overrides)
    with pytest.raises(IntegrityError):
        _verify(provenance_repo, approval)


def test_reference_approval_must_not_bind_reference_evidence(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    approval = _approve(
        provenance_repo,
        tmp_path,
        purpose="REFERENCE_RECONSTRUCTION",
        overrides={"reference_evidence_sha256": "e" * 64},
    )
    with pytest.raises(IntegrityError, match="reference_evidence_sha256"):
        _verify(provenance_repo, approval, purpose=ExecutionPurpose.REFERENCE_RECONSTRUCTION)


def test_provenance_binds_purpose_and_reference_evidence(
    provenance_repo: ProvenanceRepo, tmp_path: Path
) -> None:
    approval = _approve(
        provenance_repo, tmp_path, overrides={"reference_evidence_sha256": "d" * 64}
    )
    provenance = _verify(provenance_repo, approval)
    assert provenance.purpose is ExecutionPurpose.HELD_OUT_EVALUATION
    assert provenance.reference_evidence_sha256 == "d" * 64

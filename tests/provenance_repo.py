"""Synthetische Git-Repositories für die Post-Freeze-Code-Provenienz (P4-03Q, DEC-082).

Ausschließlich lokale Test-Repositories in temporären Verzeichnissen; kein Zugriff auf das
Projekt-Repository und keine realen Daten.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

INITIAL_FILES = {
    "src/vigiscoutx/signals/method.py": "ALPHA = 0.05\n",
    "src/vigiscoutx/experiment/held_out_guard.py": "GUARD = 1\n",
    "experiments/configs/calibration/template.yaml": "run_id: synthetic\n",
    "experiments/runners/calibration_runner/orchestrator.py": "STEPS = 28\n",
    "docs/06_experiments/MONITORING_METHODS_BASELINE.md": "# frozen\n",
    "docs/00_scope/PROJECT_STATUS.md": "# status\n",
    "requirements-lock.txt": "numpy==2.0\n",
}


@dataclass(frozen=True, slots=True)
class ProvenanceRepo:
    root: Path
    calibration_commit: str

    def git(self, *args: str) -> str:
        return git(self.root, *args)

    def commit(self, files: dict[str, str | None], message: str = "change") -> str:
        """Schreibt (bzw. löscht bei `None`) Dateien und committet; liefert den neuen HEAD."""
        for relative, content in files.items():
            path = self.root / relative
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=VigiScoutX Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    ).stdout.strip()


def create_provenance_repo(root: Path) -> ProvenanceRepo:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    repo = ProvenanceRepo(root=root, calibration_commit="")
    commit = repo.commit(dict(INITIAL_FILES), "calibration freeze")
    return ProvenanceRepo(root=root, calibration_commit=commit)


GUARD_PATH = "src/vigiscoutx/experiment/held_out_guard.py"
REFERENCE = "REFERENCE_RECONSTRUCTION"
HELD_OUT = "HELD_OUT_EVALUATION"
SYNTHETIC_REFERENCE_EVIDENCE_SHA256 = "e" * 64
"""Platzhalter für Guard-Tests ohne Reference-Lauf; der Runner prüft den echten Manifest-Hash."""


def write_approval(
    directory: Path,
    *,
    evaluation_commit: str,
    calibration_commit: str,
    freeze_id: str,
    purpose: str,
    reference_evidence_sha256: str | None = None,
    name: str = "approval.json",
    overrides: dict[str, object] | None = None,
) -> Path:
    """Synthetischer, zweckgebundener Freigabenachweis (Schema 2, DEC-087) außerhalb
    des Repositorys (nur Tests). HELD_OUT ohne angegebenes Evidence-Manifest erhält den
    synthetischen Platzhalter."""
    import json

    from vigiscoutx.experiment.code_provenance import POST_FREEZE_CHANGE_POLICY

    data: dict[str, object] = {
        "schema_version": 2,
        "approval_id": f"synthetic-{purpose.lower()}-approval",
        "author_release": "AUTHOR_APPROVED",
        "review_record_id": "synthetic-review",
        "purpose": purpose,
        "evaluation_commit": evaluation_commit,
        "calibration_commit": calibration_commit,
        "freeze_id": freeze_id,
        "policy_version": POST_FREEZE_CHANGE_POLICY,
        "reference_evidence_sha256": (
            reference_evidence_sha256 or SYNTHETIC_REFERENCE_EVIDENCE_SHA256
            if purpose == HELD_OUT
            else None
        ),
        **(overrides or {}),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def approve_and_verify(
    repo: ProvenanceRepo,
    freeze_id: str,
    directory: Path,
    *,
    purpose: str,
    reference_evidence_sha256: str | None = None,
) -> object:
    """Zweckgebundene Freigabe für den aktuellen HEAD schreiben und Provenienz prüfen."""
    from vigiscoutx.experiment.code_provenance import (
        ExecutionPurpose,
        verify_evaluation_provenance,
    )

    approval = write_approval(
        directory,
        evaluation_commit=repo.git("rev-parse", "HEAD"),
        calibration_commit=repo.calibration_commit,
        freeze_id=freeze_id,
        purpose=purpose,
        reference_evidence_sha256=reference_evidence_sha256,
        name=f"{purpose.lower()}.json",
    )
    return verify_evaluation_provenance(
        calibration_commit=repo.calibration_commit,
        freeze_id=freeze_id,
        approval_path=approval,
        purpose=ExecutionPurpose(purpose),
    )

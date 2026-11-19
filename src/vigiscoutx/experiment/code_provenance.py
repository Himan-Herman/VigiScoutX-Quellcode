"""Post-Freeze-Code-Provenienz für die Held-out-Evaluation (P4-03Q/S, DEC-082).

Trennt den Calibration-Execution-Commit (Freeze-Provenienz, `FreezeManifest.git_commit`) vom
tatsächlichen Evaluation-Commit. Alle Zustände werden ausschließlich selbst über Git und das
Dateisystem des **ausgeführten** Repositorys ermittelt (Fundort dieses Moduls); Angaben des
Aufrufers können eine Freigabe nur verhindern, nie ermöglichen.

Freigabe erfordert zusätzlich eine separate, nicht versionierte `EvaluationReleaseApproval`
(separater Nachweis für exakt Evaluation-Commit, Freeze-ID, Calibration-Commit und
Policy-Version). Dieses Modul erzeugt nie selbst eine Freigabe; ohne Nachweis bleibt der Zugriff
gesperrt. `HeldOutGuard.open_held_out` prüft unmittelbar vor dem Öffnen alles erneut.

Zweistufige, zweckgebundene Freigabe (`DEC-087`, Schema 2): `REFERENCE_RECONSTRUCTION`
berechtigt ausschließlich zur einmaligen Reference-Rekonstruktion, `HELD_OUT_EVALUATION`
ausschließlich zur HELD_OUT-Evaluation und bindet zusätzlich den SHA-256 des geprüften
Reference-Evidence-Manifests. Beide Phasen gelten für genau denselben Evaluation-Commit (HEAD).
Einphasen-Freigaben (Schema 1) sind veraltet und autorisieren keine Ausführung.

Zulässige Änderungen gegenüber dem Calibration-Commit (`POST_FREEZE_CHANGE_POLICY`, deny by
default): genau benannte Guard-Dateien; Neuanlagen in den Evaluations-Namespaces; nicht
ausführbare Dokumentation (ohne `*_BASELINE.md`), Tests und Nachweis-Manifeste. Jede geänderte
oder neue Datei muss eine reguläre, nicht ausführbare Datei sein (Git-Modus `100644`, kein
Symlink, kein Gitlink, kein Typwechsel). Wissenschaftliche Prozesskontrolle ohne kryptografische
Signatur und ohne Isolation gegen einen Angreifer mit Schreibzugriff auf Prozess oder Repository.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import model_validator

from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, GitCommitHash, Identifier, Sha256Hex
from vigiscoutx.errors import IntegrityError

POST_FREEZE_CHANGE_POLICY: Final = "post-freeze-change-policy:1.1"
GUARD_FILES: Final = frozenset(
    {
        "src/vigiscoutx/experiment/code_provenance.py",
        "src/vigiscoutx/experiment/held_out_guard.py",
    }
)
EVALUATION_ADD_ONLY_PREFIXES: Final = (
    "src/vigiscoutx/evaluation/",
    "experiments/runners/held_out_runner/",
    "experiments/configs/evaluation/",
)
NON_EXECUTABLE_PREFIXES: Final = ("docs/", "tests/", "experiments/manifests/calibration/")
NON_EXECUTABLE_FILES: Final = frozenset({"README.md"})
PROTECTED_DOC_SUFFIX: Final = "_BASELINE.md"
REGULAR_FILE_MODE: Final = "100644"
_FORBIDDEN_TREE_MODES: Final = frozenset({"120000", "160000"})  # Symlink, Gitlink
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXECUTING_FILE: Path = Path(__file__)
"""Fundort des ausgeführten Codes; bestimmt das zu prüfende Repository."""
_TOKEN = object()
_ISSUED: dict[int, tuple[object, tuple[object, ...]]] = {}


class ExecutionPurpose(StrEnum):
    """Zweck einer Ausführungsfreigabe (`DEC-087`); jede Phase erfordert ihren eigenen Nachweis."""

    REFERENCE_RECONSTRUCTION = "REFERENCE_RECONSTRUCTION"
    HELD_OUT_EVALUATION = "HELD_OUT_EVALUATION"


APPROVAL_SCHEMA_VERSION: Final = 2


class EvaluationReleaseApproval(FrozenModel):
    """Separater, zweckgebundener Freigabenachweis (Schema 2, nie versioniert).

    `HELD_OUT_EVALUATION` bindet zwingend den SHA-256 des Reference-Evidence-Manifests;
    `REFERENCE_RECONSTRUCTION` darf keinen tragen.
    """

    schema_version: Literal[2]
    approval_id: Identifier
    author_release: Literal["AUTHOR_APPROVED"]
    review_record_id: Identifier
    purpose: ExecutionPurpose
    evaluation_commit: GitCommitHash
    calibration_commit: GitCommitHash
    freeze_id: Sha256Hex
    policy_version: Literal["post-freeze-change-policy:1.1"]
    reference_evidence_sha256: Sha256Hex | None

    @model_validator(mode="after")
    def _phase_binding(self) -> Self:
        held_out = self.purpose is ExecutionPurpose.HELD_OUT_EVALUATION
        if held_out != (self.reference_evidence_sha256 is not None):
            raise ValueError("reference_evidence_sha256 genau für HELD_OUT_EVALUATION erforderlich")
        return self

    def approval_sha256(self) -> str:
        return canonical_sha256(self)


def change_violation(status: str, path: str) -> str | None:
    """Grund, warum eine Änderung `status` (A/M/D/T) an `path` unzulässig ist; sonst `None`."""
    if status == "T":
        return "Typwechsel unzulässig"
    if path in GUARD_FILES:
        return None if status in {"A", "M"} else "Guard-Datei darf nicht entfernt werden"
    if path.startswith(EVALUATION_ADD_ONLY_PREFIXES):
        return None if status == "A" else "Evaluations-Namespace erlaubt nur neue Dateien"
    if path.startswith("docs/") and path.endswith(PROTECTED_DOC_SUFFIX):
        return "eingefrorene wissenschaftliche Baseline"
    if path.startswith(NON_EXECUTABLE_PREFIXES) or path in NON_EXECUTABLE_FILES:
        return None
    return "nicht freigegebener Pfad (deny by default)"


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationProvenance:
    """Selbst ermittelte Provenienz eines Evaluation-Commits; nur über
    `verify_evaluation_provenance` ausgestellt und nur als ausgestelltes Originalobjekt gültig."""

    repository: str
    calibration_commit: str
    evaluation_commit: str
    freeze_id: str
    policy_version: str
    purpose: ExecutionPurpose
    reference_evidence_sha256: str | None
    approval_path: str
    approval_sha256: str
    changed_paths: tuple[tuple[str, str], ...]
    token: object

    def __post_init__(self) -> None:
        if self.token is not _TOKEN:
            raise IntegrityError("EvaluationProvenance nur über verify_evaluation_provenance")


def _state(provenance: EvaluationProvenance) -> tuple[object, ...]:
    return tuple(getattr(provenance, f.name) for f in fields(provenance))


def _isolated_env() -> dict[str, str]:
    """Prozessumgebung ohne jede `GIT_*`-Variable (u. a. `GIT_DIR`, `GIT_WORK_TREE`,
    `GIT_INDEX_FILE`, `GIT_COMMON_DIR`, `GIT_OBJECT_DIRECTORY`, `GIT_CONFIG_*`); System- und
    Nutzerkonfiguration sind abgeschaltet (P4-03U)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return env


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), "-c", "core.fsmonitor=false", *args],
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_env(),
    )


def _require_ok(result: subprocess.CompletedProcess[str], what: str) -> str:
    if result.returncode != 0:
        raise IntegrityError(f"{what} nicht ermittelbar: {result.stderr.strip()}")
    return result.stdout


def _require_commit(value: str, name: str) -> None:
    if not _COMMIT.fullmatch(value):
        raise IntegrityError(f"{name} muss ein vollständiger 40-stelliger Commit-Hash sein")


def executing_repository() -> str:
    """Repository, aus dem dieses Modul tatsächlich geladen wurde (P4-03U).

    Unabhängig an den kanonisch aufgelösten Modulfundort gebunden: Das Modul liegt im
    Arbeitsbaum, dessen Git-Verzeichnis exakt `<root>/.git` ist, und ist dort versioniert.
    """
    try:
        module = _EXECUTING_FILE.resolve(strict=True)
    except OSError as exc:
        raise IntegrityError(f"Ausgeführtes Modul nicht auffindbar: {exc}") from exc
    top = _require_ok(_git(module.parent, "rev-parse", "--show-toplevel"), "Repository").strip()
    root = Path(top).resolve(strict=True)
    if not module.is_relative_to(root):
        raise IntegrityError("Ausgeführtes Modul liegt nicht im geprüften Repository")
    git_dir = _require_ok(_git(root, "rev-parse", "--absolute-git-dir"), "Git-Verzeichnis").strip()
    if Path(git_dir).resolve(strict=True) != (root / ".git").resolve(strict=True):
        raise IntegrityError("Git-Verzeichnis gehört nicht zum Arbeitsbaum des ausgeführten Moduls")
    relative = str(module.relative_to(root))
    if _git(root, "ls-files", "--error-unmatch", "--", relative).returncode != 0:
        raise IntegrityError("Ausgeführtes Modul ist im geprüften Repository nicht versioniert")
    return str(root)


def _require_clean(root: Path) -> None:
    status = _require_ok(_git(root, "status", "--porcelain", "--untracked-files=all"), "Status")
    if status.strip():
        raise IntegrityError("Working Tree nicht sauber (Änderungen oder unversionierte Dateien)")
    flags = _require_ok(_git(root, "ls-files", "-v"), "Index").splitlines()
    hidden = sorted(line[2:] for line in flags if line[:1].islower() or line[:1] == "S")
    if hidden:
        raise IntegrityError(f"assume-unchanged/skip-worktree verbergen Änderungen: {hidden}")


def _require_tree_types(root: Path, head: str) -> None:
    listing = _require_ok(_git(root, "ls-tree", "-r", "-z", head), "Git-Baum")
    forbidden = []
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        if meta.split()[0] in _FORBIDDEN_TREE_MODES:
            forbidden.append(path)
    if forbidden:
        raise IntegrityError(f"Symlinks/Gitlinks im Evaluation-Commit unzulässig: {forbidden}")


def _changes(root: Path, base: str, head: str) -> tuple[tuple[str, str, str], ...]:
    """`(status, neuer Git-Modus, Pfad)` aller Änderungen zwischen `base` und `head`."""
    raw = _require_ok(_git(root, "diff", "--raw", "--no-renames", "-z", base, head), "Git-Diff")
    parts = [p for p in raw.split("\0") if p]
    if len(parts) % 2:
        raise IntegrityError("Git-Diff unvollständig")
    changes = []
    for i in range(0, len(parts), 2):
        meta = parts[i].lstrip(":").split()
        changes.append((meta[4][0], meta[1], parts[i + 1]))
    return tuple(changes)


def _type_violation(root: Path, status: str, mode: str, path: str) -> str | None:
    if status == "D":
        return None
    if mode != REGULAR_FILE_MODE:
        return f"Git-Modus {mode} unzulässig (nur reguläre, nicht ausführbare Dateien)"
    target = root / path
    if target.is_symlink() or not target.is_file():
        return "Arbeitsbaum enthält keine reguläre Datei"
    if os.access(target, os.X_OK):
        return "ausführbare Datei unzulässig"
    return None


def _load_approval(root: Path, approval_path: Path) -> EvaluationReleaseApproval:
    path = approval_path.resolve(strict=False)
    if not path.is_file() or approval_path.is_symlink():
        raise IntegrityError("Kein Evaluation-Freigabenachweis vorhanden; Zugriff gesperrt")
    if path.is_relative_to(root):
        tracked = _git(root, "ls-files", "--error-unmatch", str(path.relative_to(root)))
        if tracked.returncode == 0:
            raise IntegrityError("Freigabenachweis darf nicht versioniert sein")
    data = path.read_bytes()
    try:
        version = json.loads(data).get("schema_version")
    except (ValueError, AttributeError) as exc:
        raise IntegrityError(f"Freigabenachweis ungültig: {exc}") from exc
    if version != APPROVAL_SCHEMA_VERSION:
        raise IntegrityError(
            f"Freigabenachweis schema_version {version!r} veraltet oder unbekannt; nur "
            f"zweckgebundene Freigaben (Schema {APPROVAL_SCHEMA_VERSION}) autorisieren"
        )
    try:
        return EvaluationReleaseApproval.model_validate_json(data)
    except ValueError as exc:
        raise IntegrityError(f"Freigabenachweis ungültig: {exc}") from exc


def verify_evaluation_provenance(
    *,
    calibration_commit: str,
    freeze_id: str,
    approval_path: Path,
    purpose: ExecutionPurpose,
) -> EvaluationProvenance:
    """Prüft ausschließlich das ausgeführte Repository gegen Freeze, Policy und Freigabe.

    Ohne passenden `EvaluationReleaseApproval` (Schema 2, exakt Zweck,
    Evaluation-Commit, Freeze-ID, Calibration-Commit und Policy-Version) ist keine Provenienz
    erzeugbar. Eine Freigabe für einen anderen Zweck autorisiert nichts.
    """
    _require_commit(calibration_commit, "Calibration-Commit")
    root = Path(executing_repository())
    head = _require_ok(_git(root, "rev-parse", "--verify", "HEAD^{commit}"), "Git-HEAD").strip()
    _require_tree_types(root, head)
    _require_clean(root)
    if _git(root, "cat-file", "-e", f"{calibration_commit}^{{commit}}").returncode != 0:
        raise IntegrityError(f"Calibration-Commit {calibration_commit} existiert nicht")
    if _git(root, "merge-base", "--is-ancestor", calibration_commit, head).returncode != 0:
        raise IntegrityError("Evaluation-Commit stammt nicht vom Calibration-Commit ab")
    changes = _changes(root, calibration_commit, head)
    violations = sorted(
        f"{s} {p}: {reason}"
        for s, mode, p in changes
        if (reason := change_violation(s, p) or _type_violation(root, s, mode, p)) is not None
    )
    if violations:
        raise IntegrityError(f"Unzulässige Änderungen seit dem Freeze-Commit: {violations}")
    approval = _load_approval(root, approval_path)
    expected = {
        "purpose": ExecutionPurpose(purpose),
        "evaluation_commit": head,
        "calibration_commit": calibration_commit,
        "freeze_id": freeze_id,
        "policy_version": POST_FREEZE_CHANGE_POLICY,
    }
    mismatches = sorted(k for k, v in expected.items() if getattr(approval, k) != v)
    if mismatches:
        raise IntegrityError(f"Freigabenachweis passt nicht zum Repository-Zustand: {mismatches}")
    provenance = EvaluationProvenance(
        repository=str(root),
        calibration_commit=calibration_commit,
        evaluation_commit=head,
        freeze_id=freeze_id,
        policy_version=POST_FREEZE_CHANGE_POLICY,
        purpose=approval.purpose,
        reference_evidence_sha256=approval.reference_evidence_sha256,
        approval_path=str(approval_path.resolve(strict=True)),
        approval_sha256=approval.approval_sha256(),
        changed_paths=tuple((s, p) for s, _, p in changes),
        token=_TOKEN,
    )
    _ISSUED[id(provenance)] = (provenance, _state(provenance))
    return provenance


def reverify_evaluation_provenance(
    provenance: EvaluationProvenance, *, calibration_commit: str, freeze_id: str
) -> None:
    """Unmittelbar vor dem Zugriff: Originalobjekt, unveränderter Zustand, ausgeführtes Repository
    und vollständige Neuprüfung von HEAD, Working Tree, Policy und Freigabe."""
    issued = _ISSUED.get(id(provenance))
    if (
        type(provenance) is not EvaluationProvenance
        or issued is None
        or issued[0] is not provenance
    ):
        raise IntegrityError("Provenienz wurde nicht von verify_evaluation_provenance ausgestellt")
    if issued[1] != _state(provenance):
        raise IntegrityError("Provenienz wurde nach der Ausstellung verändert")
    if provenance.repository != executing_repository():
        raise IntegrityError("Provenienz gehört nicht zum ausgeführten Repository")
    fresh = verify_evaluation_provenance(
        calibration_commit=calibration_commit,
        freeze_id=freeze_id,
        approval_path=Path(provenance.approval_path),
        purpose=provenance.purpose,
    )
    del _ISSUED[id(fresh)]
    if _state(fresh)[:-1] != _state(provenance)[:-1]:
        raise IntegrityError(
            "Repository-Zustand oder Freigabe seit der Provenienzprüfung verändert"
        )

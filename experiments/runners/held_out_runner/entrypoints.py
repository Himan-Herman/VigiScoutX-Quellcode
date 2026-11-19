"""Produktive, fail-closed Einstiegspunkte der zweistufigen Ausführung (DEC-087, Protokoll v1.1).

Schmale Orchestrierung bereits geprüfter Komponenten, kein neuer wissenschaftlicher Algorithmus:

- `reference`: `verify_evaluation_provenance(REFERENCE_RECONSTRUCTION)` → versioniertes
  Integritätsmanifest (nur Hashes) → `load_frozen_calibration()` → Split-Loader/-Verifier →
  `HeldOutGuard.freeze()` → `Ham10000Dataset.from_directory()` → eingefrorener
  `ImagePreprocessor` → `load_reference_model()` (`resnet18-pilot-P1`) → `reconstruct_reference()`
  mit der registrierten historischen Erwartung, CPU und Batch 32.
- `held-out`: `verify_evaluation_provenance(HELD_OUT_EVALUATION)` → Vorprüfung des in der Freigabe
  gebundenen Reference-Evidence-Manifests → Integritätsmanifest → Freeze, Split, aus dem Freeze
  abgeleitete und validierte HELD_OUT-RunConfig → Guard, Dataset, Preprocessor, Modell →
  `HeldOutRunner` (keine zweite Reference-Rekonstruktion).

Ohne `--execute` und ohne phasengerechte externe Freigabe wird nichts geöffnet. Die Freigabe und die
Code-Provenienz (HEAD, Working Tree, Policy 1.1, Freeze-ID, Calibration- und Evaluation-Commit)
werden geprüft, bevor CAL_REF-Pools oder Dataset-Metadaten verarbeitet, Bilder geöffnet oder
Gewichte geladen werden. Freeze-ID und Calibration-Commit sind im Code gebunden, nicht frei als
CLI-Argument wählbar. Dieses Modul erzeugt nie eine Freigabe. Ergebnisse sind write-once.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from held_out_runner.orchestrator import HeldOutRunner, HeldOutRunResult
from vigiscoutx.adapters.ham10000 import Ham10000Dataset, Ham10000Layout
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.config.scientific_config import scientific_config_sha256
from vigiscoutx.domain.hashing import sha256_hex
from vigiscoutx.errors import IntegrityError
from vigiscoutx.evaluation.reference_reconstruction import (
    EVIDENCE_MANIFEST_FILE,
    P4_03_REFERENCE_EXPECTATION,
    REFERENCE_BATCH_SIZE,
    REFERENCE_DEVICE,
    ReconstructedReference,
    ReferenceExecution,
    ReferenceExpectation,
    ReferenceModelPort,
    load_reference_model,
    reconstruct_reference,
)
from vigiscoutx.evaluation.reference_sources import FrozenCalibration, load_frozen_calibration
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.code_provenance import (
    EvaluationProvenance,
    ExecutionPurpose,
    executing_repository,
    verify_evaluation_provenance,
)
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256, validate_split_manifest
from vigiscoutx.experiment.splitting import read_split_manifest
from vigiscoutx.experiment.types import Phase

REAL_FREEZE_ID: Final = P4_03_REFERENCE_EXPECTATION.freeze_id
REAL_CALIBRATION_COMMIT: Final = "5ad433a22cf176aa1846d5030b393837b4454c7b"
STOP_EXIT_CODE: Final = 2


class EntryPointStop(IntegrityError):
    """Terminaler STOP eines Einstiegspunkts vor oder während der Ausführung."""


def _stop(condition: bool, message: str) -> None:
    if not condition:
        raise EntryPointStop(message)


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    """Verbindliche Freeze-Identität; real ausschließlich `REAL_BINDING` (nicht per CLI wählbar)."""

    freeze_id: str
    calibration_commit: str


REAL_BINDING: Final = ExecutionBinding(REAL_FREEZE_ID, REAL_CALIBRATION_COMMIT)


@dataclass(frozen=True, slots=True)
class ExecutionLayout:
    """Feste Ablageorte; `root` löst die Pfade des versionierten Integritätsmanifests auf."""

    root: Path
    calibration_run: Path
    integrity_manifest: Path
    split_manifest: Path
    dataset_root: Path
    dataset_layout: Ham10000Layout
    model_snapshot: Path
    environment_lock: Path
    results_root: Path


REAL_DATASET_LAYOUT: Final = Ham10000Layout(
    metadata_file="HAM10000_metadata.csv",
    image_dirs=("HAM10000_images_part_1", "HAM10000_images_part_2"),
    image_suffix=".jpg",
)


def real_layout() -> ExecutionLayout:
    """Reale Ablage relativ zum ausgeführten Repository (kein frei gewählter Pfad)."""
    root = Path(executing_repository())
    return ExecutionLayout(
        root=root,
        calibration_run=root / "experiments/results/raw/P4-03-calibration-real-2",
        integrity_manifest=root
        / "experiments/manifests/calibration/P4-03-calibration-real-2_evidence.json",
        split_manifest=root / "experiments/manifests/splits/HAM10000-lesion-split-v1.json",
        dataset_root=root / "data/raw/HAM10000",
        dataset_layout=REAL_DATASET_LAYOUT,
        model_snapshot=root / "experiments/results/raw/P3-02G/P1/snapshot/model_snapshot.json",
        environment_lock=root / "requirements-lock.txt",
        results_root=root / "experiments/results/raw",
    )


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: ReferenceModelPort
    weights_sha256: str


ModelLoader = Callable[[Path, CalibrationFreezeRecord, ImagePreprocessor], LoadedModel]


def load_frozen_model(
    snapshot: Path, record: CalibrationFreezeRecord, preprocessor: ImagePreprocessor
) -> LoadedModel:
    """`resnet18-pilot-P1` über den unveränderten Adapter (CPU, Batch 32, Hashprüfung)."""
    adapter = load_reference_model(snapshot, record=record, preprocessor=preprocessor)
    return LoadedModel(model=adapter, weights_sha256=adapter.manifest.weights_sha256)


# --- Prüfungen vor sensiblen Zugriffen ----------------------------------------------------------


def _authorize(
    binding: ExecutionBinding, approval: Path | None, purpose: ExecutionPurpose
) -> EvaluationProvenance:
    """Phasengerechte Freigabe und Code-Provenienz vor jedem Daten-, Pool- oder Modellzugriff."""
    _stop(approval is not None and approval.is_file(), "Freigabedatei fehlt; Zugriff gesperrt")
    assert approval is not None
    return verify_evaluation_provenance(
        calibration_commit=binding.calibration_commit,
        freeze_id=binding.freeze_id,
        approval_path=approval,
        purpose=purpose,
    )


def _within_results(layout: ExecutionLayout, path: Path, what: str) -> Path:
    resolved = path.resolve()
    _stop(
        resolved.is_relative_to(layout.results_root.resolve()),
        f"{what} muss unter {layout.results_root} (nicht versioniert) liegen",
    )
    return resolved


def _new_output(layout: ExecutionLayout, path: Path) -> Path:
    output = _within_results(layout, path, "Ergebnisverzeichnis")
    staging = output.with_name(f".{output.name}.incomplete")
    _stop(
        not output.exists() and not staging.exists(),
        f"Ergebnisverzeichnis existiert bereits (write-once): {output}",
    )
    _stop(output.parent.is_dir(), f"Elternverzeichnis fehlt: {output.parent}")
    return output


def verify_integrity_manifest(layout: ExecutionLayout, binding: ExecutionBinding) -> None:
    """Freeze-Artefakte gegen das versionierte Integritätsmanifest, ausschließlich über Hashes."""
    _stop(layout.integrity_manifest.is_file(), "Integritätsmanifest fehlt")
    evidence = json.loads(layout.integrity_manifest.read_bytes())
    _stop(
        (evidence.get("freeze_id"), evidence.get("execution_git_commit"))
        == (binding.freeze_id, binding.calibration_commit),
        "Integritätsmanifest gehört nicht zur gebundenen Freeze-Identität",
    )
    files = evidence.get("files")
    _stop(isinstance(files, list) and bool(files), "Integritätsmanifest ohne Dateiliste")
    listed: set[Path] = set()
    for entry in files:
        path = (layout.root / entry["path"]).resolve()
        _stop(
            path.is_file() and not (layout.root / entry["path"]).is_symlink(),
            f"Freeze-Datei fehlt: {entry['path']}",
        )
        _stop(
            path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"],
            f"Freeze-Datei weicht vom Integritätsmanifest ab: {entry['path']}",
        )
        listed.add(path)
    run = layout.calibration_run.resolve()
    present = {p.resolve() for p in run.rglob("*") if p.is_file() or p.is_symlink()}
    _stop(
        present == {p for p in listed if p.is_relative_to(run)},
        "Calibration-Laufverzeichnis enthält nicht exakt die gehashten Dateien",
    )


def _frozen(layout: ExecutionLayout, binding: ExecutionBinding) -> FrozenCalibration:
    verify_integrity_manifest(layout, binding)
    frozen = load_frozen_calibration(layout.calibration_run)
    _stop(
        (frozen.manifest.freeze_id, frozen.manifest.git_commit)
        == (binding.freeze_id, binding.calibration_commit),
        "Freeze weicht von der gebundenen Freeze-Identität ab",
    )
    return frozen


def _split(layout: ExecutionLayout, frozen: FrozenCalibration) -> SplitManifest:
    _stop(layout.split_manifest.is_file(), "Split-Manifest fehlt")
    split = read_split_manifest(layout.split_manifest, frozen.run_config.scientific.split)
    validate_split_manifest(split)
    _stop(
        split_manifest_sha256(split) == frozen.record.split_manifest_sha256,
        "Split-Manifest weicht vom Freeze ab",
    )
    return split


def _components(
    layout: ExecutionLayout, frozen: FrozenCalibration, model_loader: ModelLoader
) -> tuple[HeldOutGuard, Ham10000Dataset, ImagePreprocessor, LoadedModel]:
    """Erst nach Freigabe-, Provenienz-, Integritäts-, Freeze- und Split-Prüfung."""
    guard = HeldOutGuard()
    guard.freeze(frozen.manifest)
    dataset = Ham10000Dataset.from_directory(layout.dataset_root, layout=layout.dataset_layout)
    preprocessor = ImagePreprocessor(
        ImagePreprocessingSpec.from_config(frozen.run_config.scientific.preprocessing)
    )
    loaded = model_loader(layout.model_snapshot, frozen.record, preprocessor)
    return guard, dataset, preprocessor, loaded


def _execution() -> ReferenceExecution:
    return ReferenceExecution(device=REFERENCE_DEVICE, batch_size=REFERENCE_BATCH_SIZE)


# --- Phase 1: REFERENCE_RECONSTRUCTION ----------------------------------------------------------


def run_reference_phase(
    *,
    approval: Path | None,
    output_dir: Path,
    execute: bool,
    binding: ExecutionBinding = REAL_BINDING,
    layout: ExecutionLayout | None = None,
    model_loader: ModelLoader = load_frozen_model,
    expectation: ReferenceExpectation | None = None,
) -> ReconstructedReference:
    """Genau eine Reference-Rekonstruktion des eingefrorenen 128er-Windows."""
    _stop(execute, "kein Ausführungsmodus: --execute und eine REFERENCE-Freigabe erforderlich")
    _stop(
        binding != REAL_BINDING or expectation is None,
        "realer Freeze: ausschließlich die registrierte historische Erwartung",
    )
    layout = layout or real_layout()
    output = _new_output(layout, output_dir)
    provenance = _authorize(binding, approval, ExecutionPurpose.REFERENCE_RECONSTRUCTION)
    frozen = _frozen(layout, binding)
    split = _split(layout, frozen)
    guard, dataset, preprocessor, loaded = _components(layout, frozen, model_loader)
    return reconstruct_reference(
        guard=guard,
        frozen=frozen,
        split_manifest=split,
        dataset=dataset,
        model=loaded.model,
        preprocessor=preprocessor,
        environment_lock=layout.environment_lock,
        provenance=provenance,
        expectation=expectation,
        execution=_execution(),
        output_dir=output,
    )


# --- Phase 2: HELD_OUT_EVALUATION ---------------------------------------------------------------


def derive_held_out_config(frozen: FrozenCalibration) -> RunConfig:
    """HELD_OUT-RunConfig ausschließlich aus dem Freeze: unveränderte wissenschaftliche
    Konfiguration, eingefrorene Repetitions/Metriken und das erste registrierte SeedBundle.

    Alle Einträge der Seed-Liste unterscheiden sich ausschließlich in `repeat_id`, das je Case
    überschrieben wird; die Wahl des ersten Eintrags ist daher keine wissenschaftliche Einstellung.
    """
    manifest, base = frozen.manifest, frozen.run_config
    seeds = manifest.seed_list
    _stop(len(seeds) == manifest.repetitions, "Seed-Liste passt nicht zu R")
    fixed = seeds[0].model_dump(exclude={"repeat_id"})
    _stop(
        all(s.model_dump(exclude={"repeat_id"}) == fixed for s in seeds)
        and len({s.repeat_id for s in seeds}) == len(seeds),
        "Seed-Liste weicht außer in repeat_id ab",
    )
    _stop(
        base.phase is Phase.DEVELOPMENT_CALIBRATION and base.freeze_id is None,
        "eingefrorene Run-Konfiguration ist keine Calibration-Konfiguration",
    )
    data = base.model_dump(mode="json")
    data.update(
        {
            "run_id": f"{base.run_id}-held-out",
            "phase": Phase.HELD_OUT_EVALUATION.value,
            "freeze_id": manifest.freeze_id,
            "seeds": seeds[0].model_dump(mode="json"),
        }
    )
    config = RunConfig.model_validate(data)
    checks = {
        "scientific": (config.scientific, base.scientific),
        "config_sha256": (scientific_config_sha256(config.scientific), manifest.config_sha256),
        "repetitions": (config.repetitions, manifest.repetitions),
        "evaluation_metrics": (config.evaluation_metrics, manifest.evaluation_metrics),
        "seeds": (config.seeds in manifest.seed_list, True),
    }
    mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
    _stop(not mismatches, f"HELD_OUT-RunConfig weicht vom Freeze ab: {mismatches}")
    return config


def _require_bound_evidence(reference_dir: Path, provenance: EvaluationProvenance) -> None:
    """Frühe Vorprüfung vor Dataset/Modell; die verbindliche Byte-Prüfung erfolgt im Loader."""
    bound = provenance.reference_evidence_sha256
    manifest = reference_dir / EVIDENCE_MANIFEST_FILE
    _stop(bound is not None and manifest.is_file(), "Reference-Evidence-Manifest fehlt")
    _stop(
        sha256_hex(manifest.read_bytes()) == bound,
        "Reference-Evidence-Manifest weicht von der HELD_OUT-Freigabe ab",
    )


def run_held_out_phase(
    *,
    approval: Path | None,
    reference_dir: Path,
    output_dir: Path,
    execute: bool,
    binding: ExecutionBinding = REAL_BINDING,
    layout: ExecutionLayout | None = None,
    model_loader: ModelLoader = load_frozen_model,
) -> HeldOutRunResult:
    """Genau eine HELD_OUT-Evaluation mit der bereits rekonstruierten, hashgeprüften Reference."""
    _stop(execute, "kein Ausführungsmodus: --execute und eine HELD_OUT-Freigabe erforderlich")
    layout = layout or real_layout()
    output = _new_output(layout, output_dir)
    reference = _within_results(layout, reference_dir, "Reference-Verzeichnis")
    provenance = _authorize(binding, approval, ExecutionPurpose.HELD_OUT_EVALUATION)
    _require_bound_evidence(reference, provenance)
    frozen = _frozen(layout, binding)
    split = _split(layout, frozen)
    config = derive_held_out_config(frozen)
    guard, dataset, preprocessor, loaded = _components(layout, frozen, model_loader)
    observed = ObservedHashes(
        dataset_snapshot_sha256=dataset.snapshot_ref().sha256,
        split_manifest_sha256=split_manifest_sha256(split),
        model_weights_sha256=loaded.weights_sha256,
        environment_lock_sha256=sha256_file(layout.environment_lock),
    )
    return HeldOutRunner(
        guard=guard,
        frozen=frozen,
        split_manifest=split,
        dataset=dataset,
        model=loaded.model,
        preprocessor=preprocessor,
        environment_lock=layout.environment_lock,
        provenance=provenance,
        config=config,
        observed=observed,
        reference_dir=reference,
        execution=_execution(),
        output_dir=output,
    ).run()


# --- CLI ----------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m held_out_runner.entrypoints",
        description="VigiScoutX: zweistufige reale Ausführung (DEC-087); fail-closed",
    )
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("reference", "held-out"):
        phase = phases.add_parser(name)
        phase.add_argument("--approval", type=Path, required=True)
        phase.add_argument("--output-dir", type=Path, required=True)
        phase.add_argument("--execute", action="store_true", help="ausdrücklicher Ausführungsmodus")
        if name == "held-out":
            phase.add_argument("--reference-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.phase == "reference":
            reference = run_reference_phase(
                approval=args.approval, output_dir=args.output_dir, execute=args.execute
            )
            summary: dict[str, object] = {
                "status": "COMPLETED",
                "phase": "REFERENCE_RECONSTRUCTION",
                "output_dir": str(reference.directory),
                "reference_evidence_sha256": sha256_file(
                    reference.directory / EVIDENCE_MANIFEST_FILE
                ),
                "ba_ref_4dp": f"{reference.ba_ref:.4f}",
            }
        else:
            result = run_held_out_phase(
                approval=args.approval,
                reference_dir=args.reference_dir,
                output_dir=args.output_dir,
                execute=args.execute,
            )
            summary = {
                "status": "COMPLETED",
                "phase": "HELD_OUT_EVALUATION",
                "output_dir": str(result.output_dir),
                "monitoring_runs": result.n_monitoring_runs,
            }
    except Exception as exc:  # jeder Fehler ist ein terminaler STOP ohne gültiges Ergebnis
        print(
            json.dumps(
                {"status": "STOPPED", "error": type(exc).__name__, "detail": str(exc)},
                ensure_ascii=False,
            )
        )
        return STOP_EXIT_CODE
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

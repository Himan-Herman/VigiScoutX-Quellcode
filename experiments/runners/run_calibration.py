"""Realer Calibration-Lauf gemäß P4-01/P4-02D (P4-02E/F-Runner; Ausführung NICHT freigegeben).

Ein realer Lauf ist ausschließlich mit einem separat freigegebenen
Autorisierungsartefakt (`PreRunAuthorization`, `--authorization <datei.json>`) zulässig
(`DEC-066`, `DEC-072`). Vor jeder Calibration-Berechtigung werden die beobachteten Werte geprüft:
Execution-Commit (= HEAD, sauberer Working Tree), SHA-256 der realen RunConfig-Vorlage, Split,
Model Snapshot und Gewichte (`ResNet18ModelAdapter.from_snapshot`, `resnet18-pilot-P1`),
Environment-Lock, Baseline-Versionen und – nach dem Parsen des Snapshots – der Dataset-Hash.
Jede fehlende, nicht freigegebene oder abweichende Autorisierung beendet das Skript, bevor ein
`CalibrationProcess` entsteht. HELD_OUT wird nie geöffnet.

Aufruf (aus dem Repository-Wurzelverzeichnis, erst nach Freigabe):
    python experiments/runners/run_calibration.py --authorization <freigabe.json> \
        --template <vorlage.yaml> --model-snapshot <model_snapshot.json> \
        --output-dir <neues Verzeichnis>
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_runner.authorization import (
    DATASET_INDEPENDENT_KEYS,
    MONITORING_METHODS_VERSION,
    ORCHESTRATION_BASELINE_VERSION,
    AuthorizationError,
    ObservedExecution,
    PreRunAuthorization,
    authorize_real_execution,
    require_matching,
)
from calibration_runner.freeze_assembly import (
    config_snapshot_hashes,
    method_refs,
    template_methods,
)
from calibration_runner.orchestrator import CalibrationRunner
from calibration_runner.real_template import validate_real_template
from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.ham10000 import Ham10000Dataset, Ham10000Layout
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.adapters.resnet18 import (
    ModelSnapshotManifest,
    ResNet18ModelAdapter,
    snapshot_ref_for_manifest,
)
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.config.loader import load_model
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.errors import IntegrityError, PolicyResolutionError
from vigiscoutx.experiment.calibration.config import baseline_protocol_config
from vigiscoutx.experiment.calibration.process import CalibrationRunProvenance
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.model_development.snapshot import current_git_commit
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.splitting import read_split_manifest

DATASET_ROOT = Path("data/raw/HAM10000")
DATASET_LAYOUT = Ham10000Layout(
    metadata_file="HAM10000_metadata.csv",
    image_dirs=("HAM10000_images_part_1", "HAM10000_images_part_2"),
    image_suffix=".jpg",
)
SPLIT_MANIFEST = Path("experiments/manifests/splits/HAM10000-lesion-split-v1.json")
ENVIRONMENT_LOCK = Path("requirements-lock.txt")


def working_tree_clean() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return not status.strip()


def load_authorization(path: Path) -> PreRunAuthorization:
    if not path.is_file():
        raise SystemExit("Keine Pre-Run-Autorisierung vorhanden; realer Lauf nicht zulässig")
    try:
        return PreRunAuthorization.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise SystemExit(f"Pre-Run-Autorisierung ungültig oder nicht freigegeben: {exc}") from exc


def run(args: argparse.Namespace) -> dict[str, object]:
    authorization = load_authorization(args.authorization)
    if args.output_dir.exists():
        raise SystemExit(f"Ausgabeverzeichnis existiert bereits: {args.output_dir}")
    template = load_model(args.template, RunConfig)
    methods = template_methods(template)
    protocol = baseline_protocol_config()
    split = read_split_manifest(SPLIT_MANIFEST, template.scientific.split)
    snapshot = ModelSnapshotManifest.model_validate(json.loads(args.model_snapshot.read_bytes()))
    snapshot_ref = snapshot_ref_for_manifest(snapshot)
    observed: dict[str, object] = {
        "run_id": template.run_id,
        "git_commit": current_git_commit(Path(".")),
        "working_tree_clean": working_tree_clean(),
        "run_config_template_sha256": canonical_sha256(template),
        "split_manifest_sha256": template.scientific.split.sha256,
        "model_id": snapshot.model_id,
        "model_architecture": snapshot.architecture,
        "model_snapshot_sha256": snapshot_ref.sha256,
        "model_weights_sha256": snapshot.weights_sha256,
        "environment_lock_sha256": sha256_file(ENVIRONMENT_LOCK),
        "calibration_protocol_version": protocol.protocol_version,
        "orchestration_baseline_version": ORCHESTRATION_BASELINE_VERSION,
        "monitoring_methods_version": MONITORING_METHODS_VERSION,
    }
    try:
        # Vorprüfung aller datensatzunabhängigen Werte vor dem Laden des Datensatzes.
        provisional = ObservedExecution.model_validate(
            {**observed, "dataset_snapshot_sha256": authorization.dataset_snapshot_sha256}
        )
        require_matching(authorization, provisional, DATASET_INDEPENDENT_KEYS)
        if template.scientific.model.sha256 != snapshot.weights_sha256:
            raise AuthorizationError("Vorlage bindet andere Modellgewichte")
        # Real-Template-Guard (DEC-079) vor jedem Modell- und Datensatzzugriff.
        validate_real_template(template)
    except (IntegrityError, PolicyResolutionError) as exc:
        raise SystemExit(str(exc)) from exc
    preprocessor = ImagePreprocessor(
        ImagePreprocessingSpec.from_config(template.scientific.preprocessing)
    )
    model = ResNet18ModelAdapter.from_snapshot(
        args.model_snapshot,
        expected=snapshot_ref,
        class_order=DEMONSTRATOR_CLASS_ORDER,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        allow_untrained=False,
    )
    dataset = Ham10000Dataset.from_directory(DATASET_ROOT, layout=DATASET_LAYOUT)
    try:
        execution = authorize_real_execution(
            authorization,
            ObservedExecution.model_validate(
                {**observed, "dataset_snapshot_sha256": dataset.snapshot_ref().sha256}
            ),
        )
    except AuthorizationError as exc:
        raise SystemExit(str(exc)) from exc
    provenance = CalibrationRunProvenance(
        run_id=template.run_id,
        dataset=dataset.snapshot_ref(),
        split=template.scientific.split,
        model_id="resnet18-pilot-P1",
        model_weights_sha256=model.manifest.weights_sha256,
        model_snapshot_sha256=model.snapshot_ref().sha256,
        preprocessing_sha256=canonical_sha256(template.scientific.preprocessing),
        method_refs=method_refs(methods),
        config_snapshot_sha256s=(
            *config_snapshot_hashes(template, protocol),
            authorization.authorization_sha256(),
        ),
        git_commit=execution.observed.git_commit,
        environment_lock_sha256=execution.observed.environment_lock_sha256,
    )
    runner = CalibrationRunner(
        guard=HeldOutGuard(),
        split_manifest=split,
        provenance=provenance,
        protocol=protocol,
        dataset=dataset,
        audit_dataset=dataset,
        model=model,
        preprocessor=preprocessor,
        template=template,
        execution=execution,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    result = runner.run()
    return {
        "run_id": result.run_id,
        "freeze_id": result.manifest.freeze_id,
        "calibration_record_sha256": result.record.record_sha256(),
        "authorization_sha256": authorization.authorization_sha256(),
        "output_dir": str(args.output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VigiScoutX Calibration-Lauf (P4-02)")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

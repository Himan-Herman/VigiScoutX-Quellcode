"""Fine-tuned ModelSnapshot aus einem Model-Development-Lauf (P3-02C)."""

import subprocess
from pathlib import Path

import torch
import torchvision

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.resnet18 import (
    MANIFEST_FILE,
    TRAINING_RESULT_FILE,
    WEIGHTS_FILE,
    FineTunedProvenance,
    ModelSnapshotManifest,
    TrainingStatus,
    snapshot_ref_for_manifest,
)
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.experiment.model_development.config import (
    ModelDevelopmentConfig,
    training_config_sha256,
)
from vigiscoutx.experiment.model_development.trainer import TrainedModel


def current_git_commit(repository: Path) -> str:
    """Commit-Hash des Arbeitsstands; der Aufrufer sorgt für einen sauberen Working Tree."""
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_fine_tuned_snapshot(
    directory: Path,
    trained: TrainedModel,
    config: ModelDevelopmentConfig,
    *,
    git_commit: str,
) -> ModelSnapshotRef:
    """Schreibt Gewichte, Historie und Manifest; vorhandene Dateien werden nie überschrieben."""
    if trained.result.training_config_sha256 != training_config_sha256(config):
        raise ValueError("Trainingsergebnis gehört nicht zu dieser ModelDevelopmentConfig")
    directory.mkdir(parents=True, exist_ok=True)
    result_bytes = canonical_json_bytes(trained.result)
    with (directory / TRAINING_RESULT_FILE).open("xb") as handle:
        handle.write(result_bytes)
    weights_path = directory / WEIGHTS_FILE
    with weights_path.open("xb") as handle:
        torch.save({k: v.detach().cpu() for k, v in trained.network.state_dict().items()}, handle)
    manifest = ModelSnapshotManifest(
        schema_version=2,
        model_id=config.model_id,
        architecture="resnet18",
        num_classes=7,
        class_order=DEMONSTRATOR_CLASS_ORDER,
        weights_file=WEIGHTS_FILE,
        weights_sha256=sha256_file(weights_path),
        preprocessing_sha256=canonical_sha256(trained.preprocessing),
        training_status=TrainingStatus.FINE_TUNED,
        initialization_seed=None,
        provenance=FineTunedProvenance(
            training_config_sha256=trained.result.training_config_sha256,
            training_result_sha256=sha256_hex(result_bytes),
            split_manifest_sha256=config.split.sha256,
            dataset_snapshot_sha256=trained.result.dataset_snapshot_sha256,
            pretrained_weights_sha256=config.pretrained_weights.sha256,
            resize_strategy=config.resize.strategy,
            model_seed=config.model_seed,
            git_commit=git_commit,
        ),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
    )
    with (directory / MANIFEST_FILE).open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
    return snapshot_ref_for_manifest(manifest)

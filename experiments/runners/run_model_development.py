"""Führt genau einen präregistrierten Model-Development-Lauf aus (P3-02G).

Nutzt ausschließlich die bestehende Pipeline: offizieller HAM10000-Adapter, persistiertes
Split-Manifest, HeldOutGuard-Berechtigungen nur für TRAIN und MODEL_VALIDATION, Trainer und
Fine-tuned-Snapshot. CALIBRATION und HELD_OUT werden nicht geöffnet.

Aufruf (aus dem Repository-Wurzelverzeichnis):
    python experiments/runners/run_model_development.py \
        --config experiments/configs/development/P1.yaml
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image

from vigiscoutx.adapters.ham10000 import Ham10000Dataset, Ham10000Layout
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.config.loader import load_model
from vigiscoutx.domain.hashing import canonical_json_bytes
from vigiscoutx.domain.refs import DatasetSnapshotRef
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, PartitionAccessPermit
from vigiscoutx.experiment.model_development.config import (
    ModelDevelopmentConfig,
    training_config_sha256,
)
from vigiscoutx.experiment.model_development.snapshot import (
    current_git_commit,
    write_fine_tuned_snapshot,
)
from vigiscoutx.experiment.model_development.trainer import ModelDevelopmentTrainer
from vigiscoutx.experiment.splitting import read_split_manifest
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition

DATASET_ROOT = Path("data/raw/HAM10000")
DATASET_LAYOUT = Ham10000Layout(
    metadata_file="HAM10000_metadata.csv",
    image_dirs=("HAM10000_images_part_1", "HAM10000_images_part_2"),
    image_suffix=".jpg",
)
SPLIT_MANIFEST = Path("experiments/manifests/splits/HAM10000-lesion-split-v1.json")


class RecordingDataset:
    """Leitet an den offiziellen Adapter weiter und protokolliert jede angefragte sample_id."""

    def __init__(self, dataset: Ham10000Dataset) -> None:
        self._dataset = dataset
        self.accessed: set[str] = set()

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return self._dataset.snapshot_ref()

    def load_inputs(
        self, access: PartitionAccessPermit, sample_ids: Sequence[str]
    ) -> Sequence[Image.Image]:
        self.accessed.update(sample_ids)
        return self._dataset.load_inputs(access, sample_ids)

    def labels_for(
        self, access: PartitionAccessPermit, sample_ids: Sequence[str]
    ) -> Mapping[str, int]:
        self.accessed.update(sample_ids)
        return self._dataset.labels_for(access, sample_ids)


def _require_clean_working_tree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise SystemExit("Working Tree nicht sauber; Lauf wird nicht gestartet (Provenance)")


def run(config_path: Path) -> dict[str, object]:
    _require_clean_working_tree()
    config = load_model(config_path, ModelDevelopmentConfig)
    output = Path(config.output_dir)
    if output.exists():
        raise SystemExit(f"Ausgabeverzeichnis existiert bereits: {config.output_dir}")

    dataset = Ham10000Dataset.from_directory(DATASET_ROOT, layout=DATASET_LAYOUT)
    manifest = read_split_manifest(SPLIT_MANIFEST, config.split)
    source = RecordingDataset(dataset)
    guard = HeldOutGuard()
    train_permit = guard.issue_model_development_permit(ModelDevelopmentRole.TRAIN, manifest)
    validation_permit = guard.issue_model_development_permit(
        ModelDevelopmentRole.MODEL_VALIDATION, manifest
    )
    trained = ModelDevelopmentTrainer(config, source, manifest).train(
        train_permit, validation_permit
    )
    snapshot_ref = write_fine_tuned_snapshot(
        output / "snapshot", trained, config, git_commit=current_git_commit(Path("."))
    )

    model_development = train_permit.sample_ids | validation_permit.sample_ids
    other = {
        a.sample_id for a in manifest.assignments if a.partition is not Partition.MODEL_DEVELOPMENT
    }
    summary: dict[str, object] = {
        "run_id": config.run_id,
        "training_config_sha256": training_config_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "snapshot": snapshot_ref.model_dump(mode="json"),
        "weights_sha256": sha256_file(output / "snapshot" / "weights.pt"),
        "training_result_sha256": sha256_file(output / "snapshot" / "training_result.json"),
        "access": {
            "accessed_sample_ids": len(source.accessed),
            "model_development_sample_ids": len(model_development),
            "accessed_subset_of_train_and_validation": source.accessed <= model_development,
            "calibration_or_held_out_accessed": bool(source.accessed & other),
        },
    }
    (output / "run_summary.json").write_bytes(canonical_json_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VigiScoutX Model-Development-Lauf")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

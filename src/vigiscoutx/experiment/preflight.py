"""Real-Data-Preflight: lokaler HAM10000-Pfad → Adaptervalidierung → Snapshot → Split → Checks.

Es wird nichts heruntergeladen. Ohne lokalen Datensatz liefert der Preflight den Status
`REAL_DATASET_VALIDATION_PENDING` statt eines Fehlers.

Aufruf:
    python -m vigiscoutx.experiment.preflight --dataset-root PFAD --metadata-file DATEI \
        --image-dir VERZ1 --image-dir VERZ2 --image-suffix .jpg \
        --manifest-id ID --split-seed N --train-fraction F [--write-manifest ZIEL.json]
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.ham10000 import Ham10000Dataset, Ham10000Layout
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import DatasetSnapshotRef, SplitManifestRef
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.split_integrity import (
    validate_split_manifest,
    validate_split_matches_samples,
)
from vigiscoutx.experiment.splitting import (
    SplitGenerationConfig,
    generate_split_from_source,
    held_out_reference,
    split_manifest_ref,
    write_split_manifest,
)


class PreflightStatus(StrEnum):
    REAL_DATASET_VALIDATION_PENDING = "REAL_DATASET_VALIDATION_PENDING"
    PASSED = "PASSED"


class RoleCount(FrozenModel):
    role: Identifier
    samples: NonNegativeInt
    lesions: NonNegativeInt


class PreflightReport(FrozenModel):
    status: PreflightStatus
    message: str
    dataset: DatasetSnapshotRef | None
    split: SplitManifestRef | None
    held_out_ids_sha256: str | None
    role_counts: tuple[RoleCount, ...]


def run_preflight(
    dataset_root: Path | None,
    *,
    layout: Ham10000Layout,
    split_config: SplitGenerationConfig,
    manifest_output: Path | None,
) -> PreflightReport:
    if dataset_root is None or not dataset_root.is_dir():
        return PreflightReport(
            status=PreflightStatus.REAL_DATASET_VALIDATION_PENDING,
            message="real dataset validation pending: kein lokaler HAM10000-Datensatz gefunden",
            dataset=None,
            split=None,
            held_out_ids_sha256=None,
            role_counts=(),
        )
    dataset = Ham10000Dataset.from_directory(dataset_root, layout=layout)
    refs = dataset.sample_refs()
    manifest = generate_split_from_source(
        dataset, HeldOutGuard(), split_config, DEMONSTRATOR_CLASS_ORDER
    )
    validate_split_manifest(manifest)
    validate_split_matches_samples(manifest, refs)
    ref = (
        write_split_manifest(manifest, manifest_output)
        if manifest_output is not None
        else split_manifest_ref(manifest)
    )
    role_samples: Counter[str] = Counter()
    role_lesions: dict[str, set[str]] = {}
    for a in manifest.assignments:
        role = a.partition.value
        if a.model_development_role is not None:
            role += f"/{a.model_development_role.value}"
        role_samples[role] += 1
        role_lesions.setdefault(role, set()).add(a.group_id)
    return PreflightReport(
        status=PreflightStatus.PASSED,
        message="Adapter, Snapshot, Split und Integritätschecks bestanden",
        dataset=dataset.snapshot_ref(),
        split=ref,
        held_out_ids_sha256=canonical_sha256(held_out_reference(manifest)),
        role_counts=tuple(
            RoleCount(role=role, samples=role_samples[role], lesions=len(role_lesions[role]))
            for role in sorted(role_samples)
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VigiScoutX Real-Data-Preflight")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--image-dir", action="append", required=True)
    parser.add_argument("--image-suffix", required=True, choices=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args(argv)
    report = run_preflight(
        args.dataset_root,
        layout=Ham10000Layout(
            metadata_file=args.metadata_file,
            image_dirs=tuple(args.image_dir),
            image_suffix=args.image_suffix,
        ),
        split_config=SplitGenerationConfig(
            manifest_id=args.manifest_id,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
        ),
        manifest_output=args.write_manifest,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

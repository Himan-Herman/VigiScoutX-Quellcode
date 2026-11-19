"""Offizieller Dataset-Adapter → Model-Adapter → Output-/Representation-Batches.

Nutzt einen synthetischen Datensatz in voller Baseline-Größe, damit der offizielle,
nicht aufweichbare Produktionspfad `Ham10000Dataset.from_directory` verwendet wird.
"""

import math
from pathlib import Path

from vigiscoutx.adapters.ham10000 import Ham10000Dataset, Ham10000Layout
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.adapters.resnet18 import EMBEDDING_DIM, ResNet18ModelAdapter
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.split_integrity import validate_split_matches_samples
from vigiscoutx.experiment.types import Partition


def test_official_dataset_to_model_batches(
    canonical_dataset_root: Path,
    canonical_layout: Ham10000Layout,
    model_snapshot: tuple[Path, ModelSnapshotRef],
    preprocessing_spec: ImagePreprocessingSpec,
) -> None:
    dataset = Ham10000Dataset.from_directory(canonical_dataset_root, layout=canonical_layout)
    split = SplitManifest(
        manifest_id="split-integration",
        assignments=tuple(
            SplitAssignment(
                sample_id=ref.sample_id,
                group_id=ref.group_id,
                partition=Partition.CALIBRATION,
                model_development_role=None,
            )
            for ref in dataset.sample_refs()
        ),
    )
    validate_split_matches_samples(split, dataset.sample_refs())
    permit = HeldOutGuard().issue_permit(Partition.CALIBRATION, split)

    manifest_path, ref = model_snapshot
    model = ResNet18ModelAdapter.from_snapshot(
        manifest_path,
        expected=ref,
        class_order=dataset.class_order(),
        preprocessor=ImagePreprocessor(preprocessing_spec),
        batch_size=4,
        allow_untrained=True,
    )

    sample_ids = [f"ISIC_S{i:07d}" for i in range(7)]
    inputs = dataset.load_inputs(permit, sample_ids)
    outputs = model.predict_proba("cal-window-0", sample_ids, inputs)
    representations = model.embed("cal-window-0", sample_ids, inputs)

    assert isinstance(outputs, ModelOutputBatch)
    assert isinstance(representations, RepresentationBatch)
    assert outputs.sample_ids == representations.sample_ids == tuple(sample_ids)
    assert all(len(row) == 7 and math.isclose(sum(row), 1.0) for row in outputs.probabilities)
    assert {len(row) for row in representations.embeddings} == {EMBEDDING_DIM}
    assert set(dataset.labels_for(permit, sample_ids).values()) == set(range(7))
    assert model.manifest.class_order == dataset.class_order()

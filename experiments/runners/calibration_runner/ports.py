"""Ports des Calibration-Runners.

Der reale Lauf verwendet ausschließlich `Ham10000Dataset` und `ResNet18ModelAdapter`
(`from_snapshot`, Snapshot `resnet18-pilot-P1`); beide erfüllen diese Protokolle strukturell.
Synthetische End-to-End-Tests setzen kleine Test-Implementierungen derselben Protokolle ein.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from PIL import Image

from vigiscoutx.adapters.ham10000 import SampleMetadata
from vigiscoutx.adapters.preprocessing import PreprocessedImage
from vigiscoutx.domain.refs import DatasetSnapshotRef, ModelSnapshotRef, SampleRef
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.ports import DataAccessContext


class CalibrationDataPort(Protocol):
    """Zugriffsgebundener Datensatz (DatasetPort, LabelPort, Metadaten, Strukturlabels)."""

    def snapshot_ref(self) -> DatasetSnapshotRef: ...

    def sample_refs(self) -> tuple[SampleRef, ...]: ...

    def stratification_labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Sequence[Image.Image]: ...

    def labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...

    def metadata_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, SampleMetadata]: ...


class CalibrationModelPort(Protocol):
    """Ausschließlich lesende Inferenz auf Bildzuständen vor der ImageNet-Normalisierung."""

    def snapshot_ref(self) -> ModelSnapshotRef: ...

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> ModelOutputBatch: ...

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> RepresentationBatch: ...

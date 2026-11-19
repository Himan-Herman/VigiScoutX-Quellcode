"""Unveränderliche Reference-Artefakte und Cache-Key (DEC-046, MONITORING_METHODS_BASELINE 6).

Reference-Artefakte sind die bestehenden unveränderlichen Contracts: G1 `RepresentationBatch`,
G2 `QualityIndicatorBatch`, G3 `ModelOutputBatch`; G4 ausschließlich label-aware
(`PerformanceEvaluation`). Ein persistenter Cache-Service ist nicht Teil von P3-03A; hier wird nur
der Key-Vertrag festgelegt. Ändert sich eine Key-Komponente, entsteht ein neuer Key.
"""

from collections.abc import Iterable
from typing import Self

from pydantic import model_validator

from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Sha256Hex, SignalGroup


def reference_sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Hash der sortierten Reference-Sample-IDs (reihenfolgeunabhängig, Duplikate unzulässig)."""
    ids = sorted(sample_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Reference-Sample-IDs müssen nicht leer und eindeutig sein")
    return canonical_sha256(ids)


class ReferenceCacheKey(FrozenModel):
    group: SignalGroup
    dataset_snapshot_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    reference_sample_ids_sha256: Sha256Hex
    preprocessing_sha256: Sha256Hex
    model_weights_sha256: Sha256Hex | None
    signal_config_sha256: Sha256Hex

    @model_validator(mode="after")
    def _model_relevance(self) -> Self:
        needs_model = self.group is not SignalGroup.G2_DATA_QUALITY
        if needs_model != (self.model_weights_sha256 is not None):
            raise ValueError("Model-Weight-Hash: für G1/G3/G4 erforderlich, für G2 unzulässig")
        return self

    def key_sha256(self) -> str:
        return canonical_sha256(self)

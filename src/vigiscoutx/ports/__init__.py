"""Ports des Cores (ARC-018 – ARC-021). Adapter implementieren diese Protokolle."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol, TypeVar

from vigiscoutx.domain.refs import DatasetSnapshotRef, ModelSnapshotRef
from vigiscoutx.domain.windows import ModelOutputBatch, ReferenceWindow, RepresentationBatch

InputT_co = TypeVar("InputT_co", covariant=True)
InputT_contra = TypeVar("InputT_contra", contravariant=True)


class AccessPurpose(StrEnum):
    """Zweck eines Datenzugriffs; eine Berechtigung erlaubt nur bestimmte Zwecke."""

    INPUTS = "INPUTS"
    LABELS = "LABELS"
    METADATA = "METADATA"
    STRATIFICATION_LABELS = "STRATIFICATION_LABELS"


class DataAccessContext(ABC):
    """Zugriffsberechtigung für Daten- und Labelzugriffe.

    Der Core kennt keine Partitionen; die konkrete Berechtigung wird ausschließlich von der
    Experimentumgebung (Held-out Guard) ausgestellt. Adapter müssen `require_access` vor jedem
    Laden aufrufen. Prozesskontrolle, keine Security-Isolation.
    """

    __slots__ = ()

    @abstractmethod
    def require_access(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        """Wirft einen Fehler, wenn Zweck oder eine der sample_ids nicht erlaubt ist."""


class DatasetPort(Protocol[InputT_co]):
    def snapshot_ref(self) -> DatasetSnapshotRef: ...

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Sequence[InputT_co]: ...


class ReferencePort(Protocol):
    def reference_window(self, window_id: str) -> ReferenceWindow: ...


class ModelPort(Protocol[InputT_contra]):
    """Ausschließlich lesende Inferenz; keine Trainings-, Fit- oder Update-Operation (NFR-006)."""

    def snapshot_ref(self) -> ModelSnapshotRef: ...

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[InputT_contra]
    ) -> ModelOutputBatch: ...

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[InputT_contra]
    ) -> RepresentationBatch: ...


class LabelPort(Protocol):
    """Optionale Referenzlabels; nicht an den Primary-Pfad angebunden."""

    def labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...

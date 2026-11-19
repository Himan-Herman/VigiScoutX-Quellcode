"""ResNet-18-ModelPort-Adapter (ARC-019; DEC-014, DEC-018, DEC-029).

Ausschließlich lesende Inferenz im eval-mode: kein Training, Fit oder Update. Gewichte werden nur
aus einem lokalen Snapshot geladen; vortrainierte Gewichte werden nie heruntergeladen.
Ein Initialisierungs-Snapshot (`UNTRAINED_INITIALIZATION`) dient nur der technischen Validierung
und wird nur mit ausdrücklicher Erlaubnis geladen.
"""

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self, cast

import torch
import torchvision
from pydantic import Field, ValidationError, field_validator, model_validator
from torch import nn
from torchvision.models import resnet18

from vigiscoutx.adapters.demonstrator import (
    DEMONSTRATOR_CLASS_ORDER,
    require_canonical_class_order,
)
from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    ImagePreprocessor,
    ModelInput,
    ResizeStrategy,
)
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.artifacts.manifests import canonical_artifact_path
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.domain.types import (
    FrozenModel,
    GitCommitHash,
    Identifier,
    NonNegativeInt,
    Sha256Hex,
)
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.errors import ModelSnapshotError, PhaseViolationError

ARCHITECTURE = "resnet18"
NUM_CLASSES = len(DEMONSTRATOR_CLASS_ORDER)
EMBEDDING_DIM = 512
MANIFEST_FILE = "model_snapshot.json"
WEIGHTS_FILE = "weights.pt"
TRAINING_RESULT_FILE = "training_result.json"


class TrainingStatus(StrEnum):
    UNTRAINED_INITIALIZATION = "UNTRAINED_INITIALIZATION"
    FINE_TUNED = "FINE_TUNED"


class FineTunedProvenance(FrozenModel):
    """Herkunft eines fine-tuned Snapshots (Model Development, P3-02C)."""

    training_config_sha256: Sha256Hex
    training_result_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    dataset_snapshot_sha256: Sha256Hex
    pretrained_weights_sha256: Sha256Hex
    resize_strategy: ResizeStrategy
    model_seed: NonNegativeInt
    git_commit: GitCommitHash


class ModelSnapshotManifest(FrozenModel):
    schema_version: Literal[2]
    model_id: Identifier
    architecture: Literal["resnet18"]
    num_classes: Literal[7]
    class_order: tuple[Identifier, ...]
    weights_file: str = Field(min_length=1, strict=True)
    weights_sha256: Sha256Hex
    preprocessing_sha256: Sha256Hex
    training_status: TrainingStatus
    initialization_seed: NonNegativeInt | None
    provenance: FineTunedProvenance | None
    torch_version: str = Field(min_length=1, strict=True)
    torchvision_version: str = Field(min_length=1, strict=True)

    @field_validator("weights_file")
    @classmethod
    def _relative(cls, value: str) -> str:
        return canonical_artifact_path(value)

    @field_validator("class_order")
    @classmethod
    def _canonical_order(cls, order: tuple[str, ...]) -> tuple[str, ...]:
        return require_canonical_class_order(order)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        untrained = self.training_status is TrainingStatus.UNTRAINED_INITIALIZATION
        if untrained != (self.initialization_seed is not None):
            raise ValueError("initialization_seed genau für UNTRAINED_INITIALIZATION angeben")
        if untrained == (self.provenance is not None):
            raise ValueError("provenance genau für FINE_TUNED angeben")
        return self


def _build_network() -> nn.Module:
    """ResNet-18 ohne vortrainierte Gewichte (`weights=None` → kein Download)."""
    return cast(nn.Module, resnet18(weights=None, num_classes=NUM_CLASSES))


def snapshot_ref_for_manifest(manifest: ModelSnapshotManifest) -> ModelSnapshotRef:
    """Referenz auf einen Snapshot; `sha256` ist der kanonische Manifest-Hash."""
    return ModelSnapshotRef(
        model_id=manifest.model_id,
        architecture=ARCHITECTURE,
        uri=f"local:{manifest.model_id}",
        sha256=canonical_sha256(manifest),
    )


def write_initialization_snapshot(
    directory: Path,
    *,
    model_id: str,
    preprocessing: ImagePreprocessingSpec,
    seed: int,
) -> ModelSnapshotRef:
    """Schreibt einen untrainierten, reproduzierbaren Initialisierungs-Snapshot.

    Nur für technische Validierung; ersetzt kein fine-tuned Modell. Dateien werden nie
    überschrieben.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = _build_network()
    weights_path = directory / WEIGHTS_FILE
    with weights_path.open("xb") as handle:
        torch.save(network.state_dict(), handle)
    manifest = ModelSnapshotManifest(
        schema_version=2,
        model_id=model_id,
        architecture="resnet18",
        num_classes=7,
        class_order=DEMONSTRATOR_CLASS_ORDER,
        weights_file=WEIGHTS_FILE,
        weights_sha256=sha256_file(weights_path),
        preprocessing_sha256=canonical_sha256(preprocessing),
        training_status=TrainingStatus.UNTRAINED_INITIALIZATION,
        initialization_seed=seed,
        provenance=None,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
    )
    with (directory / MANIFEST_FILE).open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
    return snapshot_ref_for_manifest(manifest)


_FACTORY_TOKEN = object()


class ResNet18ModelAdapter:
    """ModelPort für PIL-RGB-Bilder bzw. `PreprocessedImage`-Bildzustände (Shift-Pfad);
    Preprocessing ist fest an den Snapshot gebunden.

    Einziger Konstruktionsweg ist `from_snapshot`; ein direkter Konstruktoraufruf schlägt fehl.
    """

    def __init__(
        self,
        manifest: ModelSnapshotManifest,
        network: nn.Module,
        preprocessor: ImagePreprocessor,
        batch_size: int,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise PhaseViolationError(
                "ResNet18ModelAdapter nur über from_snapshot erzeugbar (validierter Snapshot)"
            )
        self._manifest = manifest
        self._preprocessor = preprocessor
        self._batch_size = batch_size
        children = list(network.children())
        self._backbone = nn.Sequential(*children[:-1])
        self._head = children[-1]
        self._network = network

    @classmethod
    def from_snapshot(
        cls,
        manifest_path: Path,
        *,
        expected: ModelSnapshotRef,
        class_order: tuple[str, ...],
        preprocessor: ImagePreprocessor,
        batch_size: int,
        allow_untrained: bool,
    ) -> "ResNet18ModelAdapter":
        if batch_size < 1:
            raise ValueError("batch_size muss positiv sein")
        try:
            require_canonical_class_order(class_order)
        except ValueError as exc:
            raise ModelSnapshotError(f"class_order: {exc}") from exc
        raw = manifest_path.read_bytes()
        try:
            manifest = ModelSnapshotManifest.model_validate(json.loads(raw))
        except ValidationError as exc:
            raise ModelSnapshotError(f"Ungültiges Modell-Manifest: {exc}") from exc
        if sha256_hex(raw) != canonical_sha256(manifest):
            raise ModelSnapshotError("Manifestdatei ist nicht kanonisch serialisiert")
        ref = snapshot_ref_for_manifest(manifest)
        checks = {
            "sha256": (expected.sha256, ref.sha256),
            "model_id": (expected.model_id, ref.model_id),
            "architecture": (expected.architecture, ref.architecture),
            "class_order": (class_order, manifest.class_order),
            "preprocessing_sha256": (preprocessor.spec_sha256, manifest.preprocessing_sha256),
            "torch_version": (torch.__version__, manifest.torch_version),
            "torchvision_version": (torchvision.__version__, manifest.torchvision_version),
        }
        mismatches = sorted(k for k, (want, have) in checks.items() if want != have)
        if mismatches:
            raise ModelSnapshotError(f"Modell-Snapshot passt nicht: {mismatches}")
        if manifest.training_status is TrainingStatus.UNTRAINED_INITIALIZATION and not (
            allow_untrained
        ):
            raise ModelSnapshotError(
                "Untrainierter Initialisierungs-Snapshot ist kein finales Modell; "
                "Laden nur mit allow_untrained=True"
            )
        if manifest.provenance is not None:
            result_path = manifest_path.parent / TRAINING_RESULT_FILE
            if not result_path.is_file():
                raise ModelSnapshotError("Trainingshistorie des fine-tuned Snapshots fehlt")
            if sha256_file(result_path) != manifest.provenance.training_result_sha256:
                raise ModelSnapshotError("Trainingshistorie weicht vom Manifest-Hash ab")
        weights_path = manifest_path.parent / manifest.weights_file
        if sha256_file(weights_path) != manifest.weights_sha256:
            raise ModelSnapshotError("Gewichtsdatei weicht vom Manifest-Hash ab")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        network = _build_network()
        network.load_state_dict(state, strict=True)
        network.eval()
        network.requires_grad_(False)
        return cls(manifest, network, preprocessor, batch_size, _factory_token=_FACTORY_TOKEN)

    @property
    def manifest(self) -> ModelSnapshotManifest:
        return self._manifest

    @property
    def in_eval_mode(self) -> bool:
        return not any(module.training for module in self._network.modules())

    def snapshot_ref(self) -> ModelSnapshotRef:
        return snapshot_ref_for_manifest(self._manifest)

    def _forward(self, inputs: Sequence[ModelInput], *, embed: bool) -> torch.Tensor:
        if not self.in_eval_mode:
            raise ModelSnapshotError("Modell ist nicht im eval-mode")
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self._batch_size):
                batch = self._preprocessor.batch(inputs[start : start + self._batch_size])
                features = torch.flatten(self._backbone(batch), 1)
                if embed:
                    outputs.append(features.to(torch.float64))
                else:
                    logits = self._head(features).to(torch.float64)
                    outputs.append(torch.softmax(logits, dim=1))
        return torch.cat(outputs)

    @staticmethod
    def _require_alignment(sample_ids: Sequence[str], inputs: Sequence[ModelInput]) -> None:
        if len(sample_ids) != len(inputs) or not sample_ids:
            raise ValueError("sample_ids und inputs müssen gleich lang und nicht leer sein")

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[ModelInput]
    ) -> ModelOutputBatch:
        self._require_alignment(sample_ids, inputs)
        probabilities = self._forward(inputs, embed=False)
        rows = tuple(tuple(float(p) for p in row) for row in probabilities.tolist())
        return ModelOutputBatch(
            window_id=window_id, sample_ids=tuple(sample_ids), probabilities=rows
        )

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[ModelInput]
    ) -> RepresentationBatch:
        self._require_alignment(sample_ids, inputs)
        embeddings = self._forward(inputs, embed=True)
        rows = tuple(tuple(float(v) for v in row) for row in embeddings.tolist())
        return RepresentationBatch(
            window_id=window_id, sample_ids=tuple(sample_ids), embeddings=rows
        )

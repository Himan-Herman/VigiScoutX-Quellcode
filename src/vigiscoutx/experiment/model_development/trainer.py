"""ResNet-18-Fine-Tuning (DEC-018): liest ausschließlich TRAIN, wählt ausschließlich auf
MODEL_VALIDATION aus. CALIBRATION und HELD_OUT sind über die Zugriffsberechtigungen technisch
unzugänglich.
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol, Self

import torch
from PIL import Image
from pydantic import Field, StrictBool, StrictFloat, model_validator
from torch import nn
from torch.nn import functional

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import DatasetSnapshotRef
from vigiscoutx.domain.types import (
    FrozenModel,
    Identifier,
    NonNegativeFloat,
    PositiveInt,
    Sha256Hex,
    UnitInterval,
)
from vigiscoutx.errors import IntegrityError, PhaseViolationError, TrainingError
from vigiscoutx.experiment.classification_metrics import (
    balanced_accuracy,
    macro_f1,
    per_class_recall,
)
from vigiscoutx.experiment.held_out_guard import PartitionAccessPermit
from vigiscoutx.experiment.model_development.augmentation import GeometricAugmenter
from vigiscoutx.experiment.model_development.config import (
    Device,
    EarlyStoppingMonitor,
    LearningRateSchedulerConfig,
    ModelDevelopmentConfig,
    OptimizerKind,
    SchedulerKind,
    preprocessing_spec,
    training_config_sha256,
)
from vigiscoutx.experiment.model_development.pretrained import load_pretrained_resnet18
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256, validate_split_manifest
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition

NUM_CLASSES = len(DEMONSTRATOR_CLASS_ORDER)


class TrainingDataSource(Protocol):
    def snapshot_ref(self) -> DatasetSnapshotRef: ...

    def load_inputs(
        self, access: PartitionAccessPermit, sample_ids: Sequence[str]
    ) -> Sequence[Image.Image]: ...

    def labels_for(
        self, access: PartitionAccessPermit, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...


class EpochRecord(FrozenModel):
    epoch: PositiveInt
    learning_rate: Annotated[StrictFloat, Field(gt=0.0)]
    """Tatsächlich in dieser Trainingsepoche verwendete Lernrate."""
    train_loss: NonNegativeFloat
    validation_loss: NonNegativeFloat
    validation_balanced_accuracy: UnitInterval
    validation_per_class_recall: tuple[UnitInterval, ...] = Field(min_length=7, max_length=7)
    validation_macro_f1: UnitInterval


class TrainingResult(FrozenModel):
    """Reproduzierbare Trainingshistorie; enthält keine Calibration- oder Held-out-Metriken."""

    run_id: Identifier
    model_id: Identifier
    training_config_sha256: Sha256Hex
    split_manifest_sha256: Sha256Hex
    dataset_snapshot_sha256: Sha256Hex
    train_sample_ids_sha256: Sha256Hex
    validation_sample_ids_sha256: Sha256Hex
    class_weights: tuple[StrictFloat, ...] = Field(min_length=7, max_length=7)
    selection_monitor: EarlyStoppingMonitor
    best_epoch: PositiveInt
    stopped_early: StrictBool
    epochs: tuple[EpochRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        numbers = [e.epoch for e in self.epochs]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("Epochen müssen fortlaufend ab 1 nummeriert sein")
        if self.best_epoch not in numbers:
            raise ValueError("best_epoch muss eine protokollierte Epoche sein")
        return self


@dataclass(frozen=True, slots=True)
class TrainedModel:
    network: nn.Module
    result: TrainingResult
    preprocessing: ImagePreprocessingSpec


def compute_class_weights(labels: Sequence[int], num_classes: int) -> tuple[float, ...]:
    """w_c = N / (K · n_c), ausschließlich aus den übergebenen TRAIN-Labels."""
    counts = Counter(labels)
    missing = [c for c in range(num_classes) if counts[c] == 0]
    if missing:
        raise TrainingError(f"Klassen ohne TRAIN-Beispiele: {missing}")
    total = len(labels)
    return tuple(total / (num_classes * counts[c]) for c in range(num_classes))


def _role_ids(manifest: SplitManifest, role: ModelDevelopmentRole) -> frozenset[str]:
    return frozenset(
        a.sample_id
        for a in manifest.assignments
        if a.partition is Partition.MODEL_DEVELOPMENT and a.model_development_role is role
    )


def _require_role_permit(
    permit: object, role: ModelDevelopmentRole, manifest: SplitManifest
) -> list[str]:
    if not isinstance(permit, PartitionAccessPermit):
        raise PhaseViolationError(
            f"Trainer akzeptiert nur Partitionsberechtigungen, nicht {type(permit).__name__}"
        )
    if permit.partition is not Partition.MODEL_DEVELOPMENT or permit.model_development_role is not (
        role
    ):
        scope = permit.partition.value
        if permit.model_development_role is not None:
            scope += f"/{permit.model_development_role.value}"
        raise PhaseViolationError(
            f"Trainer benötigt eine Berechtigung für MODEL_DEVELOPMENT/{role.value}, "
            f"erhalten: {scope}"
        )
    if permit.sample_ids != _role_ids(manifest, role):
        raise IntegrityError(f"Berechtigung für {role.value} passt nicht zum Split-Manifest")
    return sorted(permit.sample_ids)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: LearningRateSchedulerConfig
) -> torch.optim.lr_scheduler.StepLR:
    if config.kind is not SchedulerKind.STEP_LR:
        raise TrainingError(f"Nicht unterstützter Lernratenplan: {config.kind}")
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.gamma
    )


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(rates) != 1:
        raise TrainingError("Alle Parametergruppen müssen dieselbe Lernrate haben")
    return rates.pop()


def _device(device: Device) -> torch.device:
    if device is Device.MPS and not torch.backends.mps.is_available():
        raise TrainingError("Gerät mps ist nicht verfügbar")
    if device is Device.CUDA and not torch.cuda.is_available():
        raise TrainingError("Gerät cuda ist nicht verfügbar")
    return torch.device(device.value)


def _batches(ids: Sequence[str], size: int) -> list[list[str]]:
    """Batches ohne Einzelbatch am Ende (BatchNorm benötigt ≥ 2 Beispiele im Training)."""
    batches = [list(ids[i : i + size]) for i in range(0, len(ids), size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def _improved(
    monitor: EarlyStoppingMonitor, value: float, best: float | None, min_delta: float
) -> bool:
    if best is None:
        return True
    if monitor is EarlyStoppingMonitor.VALIDATION_BALANCED_ACCURACY:
        return value > best + min_delta
    return value < best - min_delta


class ModelDevelopmentTrainer:
    def __init__(
        self,
        config: ModelDevelopmentConfig,
        data: TrainingDataSource,
        split_manifest: SplitManifest,
    ) -> None:
        validate_split_manifest(split_manifest)
        if split_manifest_sha256(split_manifest) != config.split.sha256:
            raise IntegrityError("Split-Manifest passt nicht zur ModelDevelopmentConfig")
        actual = data.snapshot_ref()
        if (actual.snapshot_id, actual.sha256) != (
            config.dataset.snapshot_id,
            config.dataset.sha256,
        ):
            raise IntegrityError(
                "Dataset-Snapshot passt nicht zur ModelDevelopmentConfig "
                f"(erwartet {config.dataset.sha256}, tatsächlich {actual.sha256})"
            )
        self._dataset = actual
        self._config = config
        self._data = data
        self._manifest = split_manifest

    def train(
        self, train_permit: PartitionAccessPermit, validation_permit: PartitionAccessPermit
    ) -> TrainedModel:
        config = self._config
        train_ids = _require_role_permit(train_permit, ModelDevelopmentRole.TRAIN, self._manifest)
        validation_ids = _require_role_permit(
            validation_permit, ModelDevelopmentRole.MODEL_VALIDATION, self._manifest
        )
        if len(train_ids) < 2:
            raise TrainingError("TRAIN benötigt mindestens zwei Beispiele")
        device = _device(config.device)
        spec = preprocessing_spec(config)
        preprocessor = ImagePreprocessor(spec)
        train_labels = self._data.labels_for(train_permit, train_ids)
        validation_labels = self._data.labels_for(validation_permit, validation_ids)
        weights = compute_class_weights([train_labels[s] for s in train_ids], NUM_CLASSES)

        shuffle_rng = random.Random(f"shuffle-{config.model_seed}")
        augmenter = GeometricAugmenter(
            config.augmentation, random.Random(f"aug-{config.model_seed}")
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.model_seed)
            network = load_pretrained_resnet18(
                config.pretrained_weights, model_seed=config.model_seed
            ).to(device)
            optimizer = self._optimizer(network)
            scheduler = build_lr_scheduler(optimizer, config.lr_scheduler)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

            records: list[EpochRecord] = []
            best_value: float | None = None
            best_epoch = 1
            best_state: dict[str, torch.Tensor] = {}
            stale = 0
            stopped_early = False
            for epoch in range(1, config.max_epochs + 1):
                order = list(train_ids)
                shuffle_rng.shuffle(order)
                network.train()
                total_loss = 0.0
                epoch_lr = current_learning_rate(optimizer)
                for batch in _batches(order, config.batch_size):
                    images = [
                        augmenter(image) for image in self._data.load_inputs(train_permit, batch)
                    ]
                    inputs = preprocessor.batch(images).to(device)
                    targets = torch.tensor([train_labels[s] for s in batch], device=device)
                    loss = criterion(network(inputs), targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch)
                # Genau ein Schritt je abgeschlossener Trainingsepoche, vor und unabhängig von
                # der Validierung.
                scheduler.step()

                record = self._validate(
                    network,
                    epoch,
                    epoch_lr,
                    total_loss / len(order),
                    validation_permit,
                    validation_ids,
                    validation_labels,
                    preprocessor,
                    device,
                )
                records.append(record)
                value = (
                    record.validation_balanced_accuracy
                    if config.early_stopping.monitor
                    is EarlyStoppingMonitor.VALIDATION_BALANCED_ACCURACY
                    else record.validation_loss
                )
                if _improved(
                    config.early_stopping.monitor,
                    value,
                    best_value,
                    config.early_stopping.min_delta,
                ):
                    best_value, best_epoch, stale = value, epoch, 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in network.state_dict().items()
                    }
                else:
                    stale += 1
                    if stale >= config.early_stopping.patience:
                        stopped_early = True
                        break

        network.load_state_dict(best_state)
        network.to(torch.device("cpu"))
        network.eval()
        result = TrainingResult(
            run_id=config.run_id,
            model_id=config.model_id,
            training_config_sha256=training_config_sha256(config),
            split_manifest_sha256=config.split.sha256,
            dataset_snapshot_sha256=self._dataset.sha256,
            train_sample_ids_sha256=canonical_sha256(train_ids),
            validation_sample_ids_sha256=canonical_sha256(validation_ids),
            class_weights=weights,
            selection_monitor=config.early_stopping.monitor,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            epochs=tuple(records),
        )
        return TrainedModel(network=network, result=result, preprocessing=spec)

    def _optimizer(self, network: nn.Module) -> torch.optim.Optimizer:
        config = self._config
        options = config.optimizer
        params = network.parameters()
        if options.kind is OptimizerKind.SGD:
            return torch.optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=float(options.momentum or 0.0),
                weight_decay=options.weight_decay,
            )
        betas = options.betas or (0.0, 0.0)
        return torch.optim.AdamW(
            params, lr=config.learning_rate, betas=betas, weight_decay=options.weight_decay
        )

    def _validate(
        self,
        network: nn.Module,
        epoch: int,
        learning_rate: float,
        train_loss: float,
        permit: PartitionAccessPermit,
        ids: Sequence[str],
        labels: Mapping[str, int],
        preprocessor: ImagePreprocessor,
        device: torch.device,
    ) -> EpochRecord:
        """Unverändertes (nicht balanciertes) MODEL_VALIDATION-Set, ohne Augmentation."""
        network.eval()
        total_loss = 0.0
        predictions: list[int] = []
        with torch.inference_mode():
            for batch in _batches(ids, self._config.batch_size):
                inputs = preprocessor.batch(self._data.load_inputs(permit, batch)).to(device)
                targets = torch.tensor([labels[s] for s in batch], device=device)
                logits = network(inputs)
                total_loss += float(functional.cross_entropy(logits, targets, reduction="sum"))
                predictions.extend(int(p) for p in logits.argmax(dim=1).tolist())
        truth = [labels[s] for s in ids]
        return EpochRecord(
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            validation_loss=total_loss / len(ids),
            validation_balanced_accuracy=balanced_accuracy(truth, predictions, NUM_CLASSES),
            validation_per_class_recall=tuple(per_class_recall(truth, predictions, NUM_CLASSES)),
            validation_macro_f1=macro_f1(truth, predictions, NUM_CLASSES),
        )

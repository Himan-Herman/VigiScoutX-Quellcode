"""ModelDevelopmentConfig (DEC-018): alle wirksamen Trainingsparameter explizit, ohne Defaults."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, model_validator

from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    Interpolation,
    ResizeStrategy,
    spec_for_strategy,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import DatasetSnapshotRef, SplitManifestRef
from vigiscoutx.domain.types import (
    FrozenModel,
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
    UnitInterval,
)
from vigiscoutx.experiment.splitting import SplitGenerationConfig

PositiveFloat = Annotated[StrictFloat, Field(gt=0.0)]


class ResizeChoice(FrozenModel):
    """Einer der beiden Resize-Kandidaten; welcher gewinnt, ist bewusst offen."""

    strategy: ResizeStrategy
    interpolation: Interpolation
    antialias: StrictBool


class OptimizerKind(StrEnum):
    SGD = "SGD"
    ADAMW = "ADAMW"


class OptimizerConfig(FrozenModel):
    kind: OptimizerKind
    weight_decay: NonNegativeFloat
    momentum: NonNegativeFloat | None
    betas: tuple[UnitInterval, UnitInterval] | None

    @model_validator(mode="after")
    def _kind_specific(self) -> Self:
        if self.kind is OptimizerKind.SGD and (self.momentum is None or self.betas is not None):
            raise ValueError("SGD erfordert momentum und keine betas")
        if self.kind is OptimizerKind.ADAMW and (self.betas is None or self.momentum is not None):
            raise ValueError("ADAMW erfordert betas und kein momentum")
        return self


class SchedulerKind(StrEnum):
    """Einziger unterstützter Lernratenplan (MODEL_DEVELOPMENT_PILOT_BASELINE, DEC-037)."""

    STEP_LR = "STEP_LR"


class LearningRateSchedulerConfig(FrozenModel):
    """StepLR: Lernrate × gamma nach jeweils `step_size` abgeschlossenen Trainingsepochen."""

    kind: SchedulerKind
    step_size: PositiveInt
    gamma: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]


class EarlyStoppingMonitor(StrEnum):
    VALIDATION_BALANCED_ACCURACY = "VALIDATION_BALANCED_ACCURACY"
    VALIDATION_LOSS = "VALIDATION_LOSS"


class EarlyStoppingConfig(FrozenModel):
    monitor: EarlyStoppingMonitor
    patience: PositiveInt
    min_delta: NonNegativeFloat


class GeometricAugmentationConfig(FrozenModel):
    """Ausschließlich verlustfreie geometrische Operationen (Spiegeln, 90°-Rotation).

    Rauschen, Unschärfe, Kontrast- oder Auflösungsänderungen sind nicht darstellbar (DEC-018).
    """

    horizontal_flip_probability: UnitInterval
    vertical_flip_probability: UnitInterval
    random_rotate_90: StrictBool


class Device(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class PretrainedWeightsRef(FrozenModel):
    """Lokale Datei mit den offiziellen TorchVision-ImageNet-Gewichten (kein Download)."""

    path: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex


class ModelDevelopmentConfig(FrozenModel):
    schema_version: Literal[1]
    run_id: Identifier
    model_id: Identifier
    dataset: DatasetSnapshotRef
    split: SplitManifestRef
    split_generation: SplitGenerationConfig
    model_seed: NonNegativeInt
    resize: ResizeChoice
    batch_size: Annotated[int, Field(ge=2, strict=True)]
    learning_rate: PositiveFloat
    optimizer: OptimizerConfig
    lr_scheduler: LearningRateSchedulerConfig
    max_epochs: PositiveInt
    early_stopping: EarlyStoppingConfig
    augmentation: GeometricAugmentationConfig
    device: Device
    pretrained_weights: PretrainedWeightsRef
    output_dir: str = Field(min_length=1, strict=True)


def preprocessing_spec(config: ModelDevelopmentConfig) -> ImagePreprocessingSpec:
    return spec_for_strategy(
        config.resize.strategy,
        pipeline_id=f"{config.model_id}-preprocessing",
        interpolation=config.resize.interpolation,
        antialias=config.resize.antialias,
    )


def training_config_sha256(config: ModelDevelopmentConfig) -> str:
    """Hash der wirksamen Trainingskonfiguration; lokale Pfade und Run-ID ausgenommen."""
    return canonical_sha256(
        config.model_dump(
            mode="json",
            exclude={"run_id": True, "output_dir": True, "pretrained_weights": {"path"}},
        )
    )

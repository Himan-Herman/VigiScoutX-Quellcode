"""Eingefrorener Modell-Input-Pfad (DEC-018): RGB, 224 × 224, ImageNet-Normalisierung.

Enthält ausschließlich das normale, deterministische Preprocessing. Experimentelle
Shift-Transformationen gehören nicht hierher; sie werden in einem eigenen Modul vor diesem
Preprocessing angewendet (Auflösungsmanipulation vor dem finalen Resize).
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal, Self, cast

import torch
from PIL import Image
from pydantic import StrictBool, model_validator
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from vigiscoutx.config.scientific_config import PreprocessingConfig
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Identifier, PositiveInt

MODEL_INPUT_SIZE: Final = 224


class ResizeMode(StrEnum):
    EXACT = "EXACT"
    SHORTER_SIDE_THEN_CENTER_CROP = "SHORTER_SIDE_THEN_CENTER_CROP"


SHORTER_SIDE_RESIZE: Final = 256
"""Einzige zulässige Zwischengröße für SHORTER_SIDE_THEN_CENTER_CROP (Kandidat B)."""


class ResizeStrategy(StrEnum):
    """Genau zwei Model-Development-Kandidaten; die Wahl ist offen und wird nicht vorgegeben."""

    DIRECT_RESIZE_224 = "DIRECT_RESIZE_224"
    RESIZE_SHORTER_256_CENTER_CROP_224 = "RESIZE_SHORTER_256_CENTER_CROP_224"


class Interpolation(StrEnum):
    BILINEAR = "BILINEAR"
    BICUBIC = "BICUBIC"


class Normalization(StrEnum):
    TORCHVISION_IMAGENET1K_V1 = "TORCHVISION_IMAGENET1K_V1"


_INTERPOLATION = {
    Interpolation.BILINEAR: InterpolationMode.BILINEAR,
    Interpolation.BICUBIC: InterpolationMode.BICUBIC,
}
_PARAM_KEYS = frozenset(
    {"resize_mode", "resize_shorter_side", "interpolation", "antialias", "normalization"}
)


class ImagePreprocessingSpec(FrozenModel):
    """Vollständig explizite Preprocessing-Spezifikation ohne versteckte Defaults."""

    pipeline_id: Identifier
    output_height: Literal[224]
    output_width: Literal[224]
    color_mode: Literal["RGB"]
    resize_mode: ResizeMode
    resize_shorter_side: PositiveInt | None
    interpolation: Interpolation
    antialias: StrictBool
    normalization: Normalization

    @model_validator(mode="after")
    def _resize_consistent(self) -> Self:
        needs_side = self.resize_mode is ResizeMode.SHORTER_SIDE_THEN_CENTER_CROP
        if needs_side and self.resize_shorter_side is None:
            raise ValueError("SHORTER_SIDE_THEN_CENTER_CROP erfordert resize_shorter_side")
        if not needs_side and self.resize_shorter_side is not None:
            raise ValueError(
                "resize_shorter_side ist nur für SHORTER_SIDE_THEN_CENTER_CROP zulässig"
            )
        side = self.resize_shorter_side
        if needs_side and side is not None and side != SHORTER_SIDE_RESIZE:
            raise ValueError("resize_shorter_side muss exakt 256 sein (Kandidat B)")
        return self

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> "ImagePreprocessingSpec":
        keys = set(config.params)
        if keys != _PARAM_KEYS:
            raise ValueError(
                f"Preprocessing-Parameter müssen exakt {sorted(_PARAM_KEYS)} sein, "
                f"erhalten: {sorted(keys)}"
            )
        return cls.model_validate(
            {
                "pipeline_id": config.pipeline_id,
                "output_height": config.input_height,
                "output_width": config.input_width,
                "color_mode": config.color_mode,
                **{key: config.params[key] for key in sorted(_PARAM_KEYS)},
            }
        )


def spec_for_strategy(
    strategy: ResizeStrategy,
    *,
    pipeline_id: str,
    interpolation: Interpolation,
    antialias: bool,
) -> ImagePreprocessingSpec:
    """Übersetzt einen der beiden Resize-Kandidaten in die eingefrorene Spezifikation."""
    direct = strategy is ResizeStrategy.DIRECT_RESIZE_224
    return ImagePreprocessingSpec(
        pipeline_id=pipeline_id,
        output_height=MODEL_INPUT_SIZE,
        output_width=MODEL_INPUT_SIZE,
        color_mode="RGB",
        resize_mode=ResizeMode.EXACT if direct else ResizeMode.SHORTER_SIDE_THEN_CENTER_CROP,
        resize_shorter_side=None if direct else SHORTER_SIDE_RESIZE,
        interpolation=interpolation,
        antialias=antialias,
        normalization=Normalization.TORCHVISION_IMAGENET1K_V1,
    )


def strategy_of(spec: ImagePreprocessingSpec) -> ResizeStrategy:
    if spec.resize_mode is ResizeMode.EXACT:
        return ResizeStrategy.DIRECT_RESIZE_224
    return ResizeStrategy.RESIZE_SHORTER_256_CENTER_CROP_224


def imagenet_normalization() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mittelwert/Standardabweichung aus den TorchVision-Metadaten (ohne Download)."""
    transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
    mean = tuple(float(v) for v in cast(Sequence[float], transforms.mean))
    std = tuple(float(v) for v in cast(Sequence[float], transforms.std))
    return mean, std


class PreprocessedImage:
    """Bildzustand nach Resize, vor der ImageNet-Normalisierung: (3, 224, 224), float32.

    Eingang für Bilder, auf die ein experimenteller Shift im eingefrorenen Datenzustand angewendet
    wurde (CALIBRATION_PROTOCOL_BASELINE 7.1). Das Modell wendet darauf ausschließlich noch die
    Normalisierung an; ein unverschobenes Bild liefert so exakt denselben Modell-Input wie der
    PIL-Pfad (`normalize(unnormalized(image))`). Der Wertebereich [0, 1] ist Vertrag der
    Shift-Operatoren bzw. von `unnormalized` und wird hier nicht verändert (keine Klammerung).
    Der Tensor wird kopiert; die Instanz ist unveränderlich.
    """

    __slots__ = ("_tensor",)
    _tensor: torch.Tensor

    def __init__(self, tensor: torch.Tensor) -> None:
        shape = (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError("PreprocessedImage erwartet einen float32-Tensor (3, 224, 224)")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("PreprocessedImage erwartet endliche Werte")
        object.__setattr__(self, "_tensor", tensor.detach().clone())

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PreprocessedImage ist unveränderlich")

    @property
    def tensor(self) -> torch.Tensor:
        """Kopie des Bildzustands (der interne Tensor bleibt unverändert)."""
        return self._tensor.clone()


type ModelInput = Image.Image | PreprocessedImage
"""Zulässige Modell-Eingaben: Originalbild (vollständiges Preprocessing) oder Bildzustand."""


class ImagePreprocessor:
    """Deterministisches Preprocessing PIL-Bild → Tensor (3, 224, 224), float32."""

    def __init__(self, spec: ImagePreprocessingSpec) -> None:
        self._spec = spec
        self._spec_sha256 = canonical_sha256(spec)
        mean, std = imagenet_normalization()
        self._mean = list(mean)
        self._std = list(std)

    @property
    def spec(self) -> ImagePreprocessingSpec:
        return self._spec

    @property
    def spec_sha256(self) -> str:
        return self._spec_sha256

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return cast(
            torch.Tensor, tvf.normalize(self.unnormalized(image), mean=self._mean, std=self._std)
        )

    def unnormalized(self, image: Image.Image) -> torch.Tensor:
        """RGB 224 × 224 als Float in [0, 1], direkt vor der ImageNet-Normalisierung (G2-Input)."""
        spec = self._spec
        rgb = image.convert("RGB")
        tensor = cast(torch.Tensor, tvf.pil_to_tensor(rgb)).to(torch.float32) / 255.0
        interpolation = _INTERPOLATION[spec.interpolation]
        size = [spec.output_height, spec.output_width]
        if spec.resize_mode is ResizeMode.EXACT:
            tensor = cast(
                torch.Tensor,
                tvf.resize(tensor, size, interpolation=interpolation, antialias=spec.antialias),
            )
        else:
            tensor = cast(
                torch.Tensor,
                tvf.resize(
                    tensor,
                    [cast(int, spec.resize_shorter_side)],
                    interpolation=interpolation,
                    antialias=spec.antialias,
                ),
            )
            tensor = cast(torch.Tensor, tvf.center_crop(tensor, size))
        return tensor.clamp(0.0, 1.0)

    def normalize(self, image: PreprocessedImage) -> torch.Tensor:
        """Ausschließlich die eingefrorene ImageNet-Normalisierung eines Bildzustands."""
        return cast(torch.Tensor, tvf.normalize(image.tensor, mean=self._mean, std=self._std))

    def model_input(self, item: ModelInput) -> torch.Tensor:
        if isinstance(item, PreprocessedImage):
            return self.normalize(item)
        return self(item)

    def batch(self, images: Sequence[ModelInput]) -> torch.Tensor:
        return torch.stack([self.model_input(image) for image in images])

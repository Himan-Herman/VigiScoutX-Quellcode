"""Geometrische Trainingsaugmentation: nur verlustfreies Spiegeln und 90°-Rotation."""

import random

from PIL import Image

from vigiscoutx.experiment.model_development.config import GeometricAugmentationConfig

_ROTATIONS = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


class GeometricAugmenter:
    def __init__(self, config: GeometricAugmentationConfig, rng: random.Random) -> None:
        self._config = config
        self._rng = rng

    def __call__(self, image: Image.Image) -> Image.Image:
        config = self._config
        result = image
        if self._rng.random() < config.horizontal_flip_probability:
            result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self._rng.random() < config.vertical_flip_probability:
            result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if config.random_rotate_90:
            rotation = _ROTATIONS[self._rng.randrange(len(_ROTATIONS))]
            if rotation is not None:
                result = result.transpose(rotation)
        return result

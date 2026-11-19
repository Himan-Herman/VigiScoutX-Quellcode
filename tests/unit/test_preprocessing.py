"""Eingefrorener Modell-Input-Pfad (P3-02B)."""

import inspect
import re

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from vigiscoutx.adapters import preprocessing
from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    ImagePreprocessor,
    ResizeMode,
    imagenet_normalization,
)
from vigiscoutx.config.scientific_config import PreprocessingConfig


def test_output_is_rgb_224_float_tensor(preprocessing_spec: ImagePreprocessingSpec) -> None:
    grayscale = Image.new("L", (600, 450), color=128)
    tensor = ImagePreprocessor(preprocessing_spec)(grayscale)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocessing_is_deterministic(preprocessing_spec: ImagePreprocessingSpec) -> None:
    image = Image.effect_mandelbrot((600, 450), (-2.0, -1.2, 1.0, 1.2), 64).convert("RGB")
    first = ImagePreprocessor(preprocessing_spec)(image)
    second = ImagePreprocessor(preprocessing_spec)(image)
    assert torch.equal(first, second)


def test_normalization_matches_torchvision_imagenet(
    preprocessing_spec: ImagePreprocessingSpec,
) -> None:
    mean, std = imagenet_normalization()
    tensor = ImagePreprocessor(preprocessing_spec)(Image.new("RGB", (50, 40), color=(255, 0, 51)))
    expected = [(v / 255.0 - m) / s for v, m, s in zip((255, 0, 51), mean, std, strict=True)]
    for channel, value in enumerate(expected):
        assert torch.allclose(tensor[channel], torch.full((224, 224), value), atol=1e-5)


def test_unnormalized_is_g2_input_before_normalization(
    preprocessing_spec: ImagePreprocessingSpec,
) -> None:
    image = Image.effect_mandelbrot((600, 450), (-2.0, -1.2, 1.0, 1.2), 64).convert("RGB")
    preprocessor = ImagePreprocessor(preprocessing_spec)
    raw = preprocessor.unnormalized(image)
    assert raw.shape == (3, 224, 224)
    assert float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0
    mean, std = imagenet_normalization()
    expected = (raw - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    assert torch.allclose(preprocessor(image), expected, atol=1e-6)


def test_shorter_side_mode_center_crops(preprocessing_spec: ImagePreprocessingSpec) -> None:
    spec = preprocessing_spec.model_copy(
        update={
            "resize_mode": ResizeMode.SHORTER_SIDE_THEN_CENTER_CROP,
            "resize_shorter_side": 256,
        }
    )
    assert ImagePreprocessor(spec)(Image.new("RGB", (600, 450))).shape == (3, 224, 224)
    with pytest.raises(ValidationError):
        preprocessing_spec.model_copy(
            update={"resize_mode": ResizeMode.SHORTER_SIDE_THEN_CENTER_CROP}
        )
    with pytest.raises(ValidationError):
        spec.model_copy(update={"resize_shorter_side": 200})


@pytest.mark.parametrize(
    ("field", "value"),
    [("output_height", 256), ("output_width", 192), ("color_mode", "L"), ("antialias", "yes")],
)
def test_frozen_input_contract_cannot_be_changed(
    preprocessing_spec: ImagePreprocessingSpec, field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        preprocessing_spec.model_copy(update={field: value})


def test_spec_from_scientific_config_requires_all_params(
    preprocessing_spec: ImagePreprocessingSpec,
) -> None:
    params = {
        "resize_mode": "EXACT",
        "resize_shorter_side": None,
        "interpolation": "BILINEAR",
        "antialias": True,
        "normalization": "TORCHVISION_IMAGENET1K_V1",
    }
    config = PreprocessingConfig.model_validate(
        {
            "pipeline_id": "prep-test",
            "input_height": 224,
            "input_width": 224,
            "color_mode": "RGB",
            "params": params,
        }
    )
    assert ImagePreprocessingSpec.from_config(config) == preprocessing_spec
    incomplete = config.model_copy(
        update={"params": {k: v for k, v in params.items() if k != "antialias"}}
    )
    with pytest.raises(ValueError, match="exakt"):
        ImagePreprocessingSpec.from_config(incomplete)


def test_no_shift_transformation_in_preprocessing() -> None:
    assert set(ImagePreprocessingSpec.model_fields) == {
        "pipeline_id",
        "output_height",
        "output_width",
        "color_mode",
        "resize_mode",
        "resize_shorter_side",
        "interpolation",
        "antialias",
        "normalization",
    }
    source = inspect.getsource(preprocessing).lower()
    assert not re.search(r"noise|blur|contrast|jitter|gauss|sharp|random|augment", source)

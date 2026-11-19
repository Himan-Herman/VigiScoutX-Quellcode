"""Die vier technischen Shift-Operatoren (DEC-054, CALIBRATION_PROTOCOL_BASELINE 7).

Eingabe und Ausgabe: 224×224 RGB als Float in [0, 1], Layout CHW (3, 224, 224) wie der G2-Input
`ImagePreprocessor.unnormalized` – nach `DIRECT_RESIZE_224`, vor der ImageNet-Normalisierung.
Alle Operatoren sind reine Funktionen; die Eingabe wird nie verändert.

Technische Umsetzungsdetails (nicht wissenschaftlich wirksam, über Code-Version und Environment-
Lock eingefroren): Die ε-Realisierung wird mit `numpy.random.default_rng(seed_ε)` (PCG64) in der
Indexreihenfolge `ε_{h,w,c}` erzeugt und nach CHW umgeordnet.

Numerische Postcondition für Blur und Resolution (P4-02E): Beide Operatoren bilden
mathematisch konvexe Kombinationen der Eingabe und bleiben damit in [0, 1]. Float64-Rundung kann
dieses Intervall um wenige ULP verlassen (beobachtet: `1 + 4.4e-16` bei gesättigten Pixeln).
Ausschließlich Werte innerhalb `ROUNDING_TOLERANCE` außerhalb von [0, 1] werden auf den Rand
gesetzt. Das ist ein Schutz gegen Gleitkomma-Rundung, keine Shift-Transformation: Kernel,
Padding, Interpolation, Antialiasing, Severity und Sampling bleiben unverändert. Nichtfinite
Werte und größere Bereichsverletzungen schlagen unmittelbar im Operator hart fehl
(`ShiftRangeError`), in jedem Pfad – auch dort, wo zunächst nur G4 berechnet wird.
"""

import math
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from vigiscoutx.experiment.calibration.hashing import first64bits, sha256_concat

FloatArray = npt.NDArray[np.float64]
IMAGE_SHAPE = (3, 224, 224)
ROUNDING_TOLERANCE: Final = 1e-12
"""Maximale Float64-Rundungsabweichung von [0, 1], die die Postcondition auf den Rand setzt."""


class ShiftRangeError(ValueError):
    """Shift-Ergebnis nichtfinit oder um mehr als die Rundungstoleranz außerhalb von [0, 1]."""


def rounding_postcondition(result: FloatArray) -> FloatArray:
    """Setzt nur rundungsbedingte Überschreitungen `≤ ROUNDING_TOLERANCE` auf 0 bzw. 1;
    nichtfinite Werte und größere Verletzungen sind ein harter Fehler."""
    out = np.array(result, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ShiftRangeError("Shift-Ergebnis enthält nichtfinite Werte")
    if out.min() < -ROUNDING_TOLERANCE or out.max() > 1.0 + ROUNDING_TOLERANCE:
        raise ShiftRangeError(
            f"Shift-Ergebnis verlässt [0, 1] über die Rundungstoleranz: "
            f"[{out.min()!r}, {out.max()!r}]"
        )
    return np.clip(out, 0.0, 1.0)


def _require_image(image: npt.ArrayLike) -> FloatArray:
    array = np.asarray(image)
    if array.shape != IMAGE_SHAPE or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Shift-Operatoren erwarten Float-Bilder der Form (3, 224, 224)")
    data = np.array(array, dtype=np.float64)
    if not np.isfinite(data).all() or data.min() < 0.0 or data.max() > 1.0:
        raise ValueError("Shift-Operatoren erwarten endliche Werte in [0, 1]")
    return data


def noise_seed(split_hash: str, noise_namespace: str, repeat_id: int, sample_id: str) -> int:
    """`first64bits(SHA256(split_hash ‖ "P4-01|SHIFT_NOISE_EPS|<repeat_id>|<sample_id>"))`."""
    return first64bits(sha256_concat(split_hash, f"{noise_namespace}{repeat_id}|{sample_id}"))


def noise_epsilon(seed: int) -> FloatArray:
    """iid `N(0, 1)` je Pixel und Kanal; unabhängig von Severity, Methode und Ergebnissen."""
    height, width = IMAGE_SHAPE[1:]
    hwc = np.random.default_rng(seed).standard_normal((height, width, IMAGE_SHAPE[0]))
    return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)), dtype=np.float64)


def apply_noise(image: npt.ArrayLike, epsilon: FloatArray, sigma_n: float) -> FloatArray:
    """`clip(x + σ_n · ε, 0, 1)` mit derselben ε-Matrix für alle Severities."""
    data = _require_image(image)
    if epsilon.shape != IMAGE_SHAPE or not (math.isfinite(sigma_n) and sigma_n > 0):
        raise ValueError("ε muss (3, 224, 224) sein und σ_n > 0")
    return np.clip(data + sigma_n * epsilon, 0.0, 1.0)


def apply_contrast(image: npt.ArrayLike, gamma: float) -> FloatArray:
    """Kanalweise mean-preserving: `clip(μ_c + γ (x_c − μ_c), 0, 1)`, `μ_c` über alle Pixel."""
    data = _require_image(image)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("γ muss in [0, 1] liegen")
    mean = data.mean(axis=(1, 2), keepdims=True)
    return np.clip(mean + gamma * (data - mean), 0.0, 1.0)


def gaussian_kernel_1d(sigma_b: float, radius_sigmas: int) -> FloatArray:
    """`g(i) = exp(−i²/(2σ²)) / Σ_j exp(−j²/(2σ²))`, `i = −r…r`, `r = ⌈3σ⌉`."""
    if not (math.isfinite(sigma_b) and sigma_b > 0):
        raise ValueError("σ_b muss endlich und > 0 sein")
    radius = math.ceil(radius_sigmas * sigma_b)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets**2) / (2.0 * sigma_b**2))
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def _convolve_axis(data: FloatArray, kernel: FloatArray, axis: int) -> FloatArray:
    radius = (len(kernel) - 1) // 2
    pad = [(0, 0)] * data.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(data, pad, mode="reflect")
    out = np.zeros_like(data)
    length = data.shape[axis]
    for offset, weight in enumerate(kernel):
        out += weight * np.take(padded, np.arange(offset, offset + length), axis=axis)
    return out


def apply_blur(image: npt.ArrayLike, sigma_b: float, radius_sigmas: int) -> FloatArray:
    """Separabler Gauß-Blur je RGB-Kanal; Rand `reflect` ohne Wiederholung des Randpixels."""
    data = _require_image(image)
    kernel = gaussian_kernel_1d(sigma_b, radius_sigmas)
    return rounding_postcondition(
        _convolve_axis(_convolve_axis(data, kernel, axis=1), kernel, axis=2)
    )


def apply_resolution(image: npt.ArrayLike, size: int) -> FloatArray:
    """`224 → s` bilinear mit Antialiasing, dann `s → 224` bilinear ohne Antialiasing."""
    data = _require_image(image)
    if not 0 < size < IMAGE_SHAPE[1]:
        raise ValueError("Zwischenauflösung muss zwischen 0 und 224 liegen")
    tensor = torch.from_numpy(data)
    low = tvf.resize(tensor, [size, size], InterpolationMode.BILINEAR, antialias=True)
    high = tvf.resize(low, list(IMAGE_SHAPE[1:]), InterpolationMode.BILINEAR, antialias=False)
    return rounding_postcondition(np.asarray(high.numpy(), dtype=np.float64))

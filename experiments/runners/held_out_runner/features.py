"""Permitgebundene HELD_OUT-Merkmale (Bildzustand, Shifts, Embedding, Softmax, G2-Indikatoren).

Identische Zusammensetzung wie im Calibration-Runner (`calibration_runner.features`): RGB 224 × 224
in [0, 1] vor der ImageNet-Normalisierung, darauf ausschließlich die eingefrorenen Operatoren
(`apply_shift`), Modell auf dem (verschobenen) Zustand, G2 je Sample. Inferenz in Blöcken von 32 in
Anfragereihenfolge. Zwischengespeichert werden nur der unverschobene Zustand und deterministische
Shifts (Contrast, Blur, Resolution) je `(sample_id, Familie, Wert)`; Noise hängt von
`(repeat_id, sample_id)` ab und teilt innerhalb eines Repeats dieselbe ε-Realisierung über L1–L3.
Jeder Datenzugriff läuft über die übergebene HELD_OUT-Berechtigung.
"""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from calibration_runner.features import SampleFeatures, SampleInputFailure, apply_shift
from vigiscoutx.adapters.preprocessing import ImagePreprocessor, PreprocessedImage
from vigiscoutx.evaluation.held_out_sampling import HeldOutEvaluationStop
from vigiscoutx.evaluation.reference_reconstruction import (
    REFERENCE_DEVICE,
    ReferenceDataPort,
    ReferenceModelPort,
)
from vigiscoutx.experiment.calibration.config import CalibrationProtocolConfig, TechnicalShift
from vigiscoutx.experiment.calibration.shifts import noise_epsilon
from vigiscoutx.ports import DataAccessContext
from vigiscoutx.signals.config import G2QualityConfig
from vigiscoutx.signals.data_quality.quality import QualityInputFailure, compute_quality_indicators

CacheKey = tuple[str, str, float]
FEATURE_WINDOW_ID = "held-out-feature-batch"
"""Technische Window-ID der Merkmalsberechnung; Signale erhalten stets die echte Window-ID."""


class HeldOutFeatures:
    """Zugriffsgebundene Merkmalsquelle des HELD_OUT-Runners (keine eigene Wissenschaftslogik)."""

    def __init__(
        self,
        *,
        dataset: ReferenceDataPort,
        permit: DataAccessContext,
        model: ReferenceModelPort,
        preprocessor: ImagePreprocessor,
        g2_config: G2QualityConfig,
        protocol: CalibrationProtocolConfig,
        batch_size: int,
    ) -> None:
        self._dataset = dataset
        self._permit = permit
        self._model = model
        self._preprocessor = preprocessor
        self._g2 = g2_config
        self._protocol = protocol
        self._batch_size = batch_size
        self._cache: dict[CacheKey, SampleFeatures] = {}
        self.batches: list[int] = []
        self.input_devices: set[str] = set()

    def _chunks(self, sample_ids: Sequence[str]) -> Iterator[list[str]]:
        for start in range(0, len(sample_ids), self._batch_size):
            yield list(sample_ids[start : start + self._batch_size])

    def _states(self, sample_ids: Sequence[str]) -> list[npt.NDArray[np.float32]]:
        images = list(self._dataset.load_inputs(self._permit, list(sample_ids)))
        if len(images) != len(sample_ids):
            raise HeldOutEvaluationStop("Datensatz lieferte nicht alle angefragten Bilder")
        return [
            np.ascontiguousarray(self._preprocessor.unnormalized(image).numpy(), dtype=np.float32)
            for image in images
        ]

    def _compute(
        self, sample_ids: Sequence[str], states: Sequence[npt.NDArray[np.floating]]
    ) -> list[SampleFeatures]:
        inputs = [
            PreprocessedImage(torch.from_numpy(np.asarray(state, dtype=np.float32)))
            for state in states
        ]
        self.input_devices |= {i.tensor.device.type for i in inputs}
        if self.input_devices != {REFERENCE_DEVICE}:
            raise HeldOutEvaluationStop(f"Eingaben nicht auf {REFERENCE_DEVICE}")
        embedded = self._model.embed(FEATURE_WINDOW_ID, sample_ids, inputs)
        predicted = self._model.predict_proba(FEATURE_WINDOW_ID, sample_ids, inputs)
        if not embedded.sample_ids == tuple(sample_ids) == predicted.sample_ids:
            raise HeldOutEvaluationStop("Modell-Ausgaben gehören nicht zum angefragten Batch")
        self.batches.append(len(sample_ids))
        rows: list[SampleFeatures] = []
        for index, (sample_id, state) in enumerate(zip(sample_ids, states, strict=True)):
            result = compute_quality_indicators(sample_id, [sample_id], [state], self._g2)
            quality = (
                SampleInputFailure(status=result.status, reason_code=result.reason_code)
                if isinstance(result, QualityInputFailure)
                else result.indicators[0]
            )
            rows.append(
                SampleFeatures(
                    embedding=embedded.embeddings[index],
                    probabilities=predicted.probabilities[index],
                    quality=quality,
                )
            )
        return rows

    def _cached(
        self,
        sample_ids: Sequence[str],
        transforms: Sequence[tuple[str, float]],
        shift: TechnicalShift | None,
    ) -> dict[tuple[str, float], list[SampleFeatures]]:
        """Deterministische Zustände; je fehlendem Sample genau ein Bildzugriff für alle Werte."""
        missing = [s for s in sample_ids if any((s, *t) not in self._cache for t in transforms)]
        for chunk in self._chunks(missing):
            clean = self._states(chunk)
            for transform in transforms:
                if shift is None:
                    states: Sequence[npt.NDArray[np.floating]] = clean
                else:
                    states = [
                        apply_shift(shift, state, transform[1], None, self._protocol)
                        for state in clean
                    ]
                for sample_id, row in zip(chunk, self._compute(chunk, states), strict=True):
                    self._cache[(sample_id, *transform)] = row
        return {t: [self._cache[(s, *t)] for s in sample_ids] for t in transforms}

    def clean(self, sample_ids: Sequence[str]) -> list[SampleFeatures]:
        return self._cached(sample_ids, [("CLEAN", 0.0)], None)[("CLEAN", 0.0)]

    def shifted(
        self,
        sample_ids: Sequence[str],
        shift: TechnicalShift,
        values: Sequence[float],
        *,
        noise_seeds: Mapping[str, int] | None,
    ) -> dict[float, list[SampleFeatures]]:
        """Merkmale je Shift-Wert; Noise mit dem Case-Seed je `(repeat_id, sample_id)` (DEC-054)."""
        if shift is not TechnicalShift.NOISE:
            transforms = [(shift.value, float(v)) for v in values]
            result = self._cached(sample_ids, transforms, shift)
            return {t[1]: rows for t, rows in result.items()}
        if noise_seeds is None or set(noise_seeds) != set(sample_ids):
            raise HeldOutEvaluationStop("Noise erfordert je Sample den eingefrorenen Seed")
        output: dict[float, list[SampleFeatures]] = {float(v): [] for v in values}
        for chunk in self._chunks(sample_ids):
            clean = self._states(chunk)
            epsilons = [noise_epsilon(noise_seeds[s]) for s in chunk]
            for value in values:
                states = [
                    apply_shift(shift, state, value, eps, self._protocol)
                    for state, eps in zip(clean, epsilons, strict=True)
                ]
                output[float(value)].extend(self._compute(chunk, states))
        return output

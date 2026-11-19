"""Bildzustände, Shift-Anwendung und zwischengespeicherte Per-Sample-Merkmale.

Datenzustand (CALIBRATION_PROTOCOL_BASELINE 7.1): `ImagePreprocessor.unnormalized` liefert das
RGB-Bild 224 × 224 als Float in [0, 1] nach dem eingefrorenen Resize und vor der
ImageNet-Normalisierung. Darauf wirken ausschließlich die eingefrorenen Shift-Operatoren; das
Modell erhält den (verschobenen) Zustand als `PreprocessedImage` und normalisiert nur noch.
G2 rechnet auf genau diesem Zustand.

Technische, wissenschaftlich nicht wirksame Festlegungen (Orchestrierungsbaseline Abschnitt 9):

- Merkmale (Embedding, Softmax, G2-Indikatoren) werden je Sample berechnet und für den
  unverschobenen Zustand sowie für deterministische Shifts (Contrast, Blur, Resolution) je
  `(sample_id, Familie, Wert)` zwischengespeichert. Noise hängt von `(repeat_id, sample_id)` ab
  und wird nie über Cases hinweg gespeichert; innerhalb eines Aufrufs teilen alle Severities
  dieselbe `ε`-Realisierung.
- Inferenz erfolgt in Blöcken fester Größe in Anfragereihenfolge (Batch-Orchestrierung).
- Jeder Daten-/Modellzugriff prüft vorher die `CalibrationAction`-Grenze des Prozesses und
  verwendet ausschließlich die scope-gebundene Berechtigung des jeweiligen Pools.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from calibration_runner.ports import CalibrationDataPort, CalibrationModelPort
from vigiscoutx.adapters.ham10000 import SampleMetadata
from vigiscoutx.adapters.preprocessing import ImagePreprocessor, PreprocessedImage
from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.errors import CalibrationProtocolError
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    TechnicalShift,
)
from vigiscoutx.experiment.calibration.process import CalibrationAction, CalibrationProcess
from vigiscoutx.experiment.calibration.shifts import (
    apply_blur,
    apply_contrast,
    apply_noise,
    apply_resolution,
    noise_epsilon,
    noise_seed,
)
from vigiscoutx.experiment.held_out_guard import CalibrationAccessPermit
from vigiscoutx.signals.config import G2QualityConfig
from vigiscoutx.signals.data_quality.quality import (
    QualityIndicators,
    QualityInputFailure,
    compute_quality_indicators,
)

FloatImage = npt.NDArray[np.floating]
CacheKey = tuple[str, str, float]
BATCH_ID = "feature-batch"
"""Technische Window-ID der Merkmalsberechnung; Signale erhalten stets die echte Window-ID."""


class FeatureKind(StrEnum):
    EMBEDDING = "EMBEDDING"
    PROBABILITIES = "PROBABILITIES"
    QUALITY = "QUALITY"


ALL_KINDS = frozenset(FeatureKind)
PREDICTION_ONLY = frozenset({FeatureKind.PROBABILITIES})


@dataclass(frozen=True, slots=True)
class SampleInputFailure:
    """G2-Eingabevertrag eines einzelnen Samples verletzt (Status und Reason-Code)."""

    status: SignalStatus
    reason_code: ReasonCode


type QualityRow = QualityIndicators | SampleInputFailure


@dataclass(frozen=True, slots=True)
class SampleFeatures:
    embedding: tuple[float, ...] | None
    probabilities: tuple[float, ...] | None
    quality: QualityRow | None


def require_features(
    features: Sequence[SampleFeatures], kind: FeatureKind
) -> list[tuple[float, ...]] | list[QualityRow]:
    """Spalte einer Merkmalsart; fehlende Werte sind ein Orchestrierungsfehler."""
    if kind is FeatureKind.EMBEDDING:
        rows = [f.embedding for f in features]
    elif kind is FeatureKind.PROBABILITIES:
        rows = [f.probabilities for f in features]
    else:
        quality = [f.quality for f in features]
        if any(q is None for q in quality):
            raise CalibrationProtocolError("G2-Indikatoren fehlen")
        return [q for q in quality if q is not None]
    if any(r is None for r in rows):
        raise CalibrationProtocolError(f"Merkmal {kind.value} fehlt")
    return [r for r in rows if r is not None]


def apply_shift(
    shift: TechnicalShift,
    image: FloatImage,
    value: float,
    epsilon: npt.NDArray[np.float64] | None,
    config: CalibrationProtocolConfig,
) -> npt.NDArray[np.float64]:
    """Ausschließlich die eingefrorenen Operatoren (DEC-054) mit dem eingefrorenen Wert."""
    if shift is TechnicalShift.NOISE:
        if epsilon is None:
            raise CalibrationProtocolError("Noise erfordert die ε-Realisierung des Samples")
        return apply_noise(image, epsilon, value)
    if shift is TechnicalShift.CONTRAST:
        return apply_contrast(image, value)
    if shift is TechnicalShift.BLUR:
        return apply_blur(image, value, config.blur_radius_sigmas)
    if not float(value).is_integer():
        raise CalibrationProtocolError("Resolution erfordert eine ganzzahlige Zwischenauflösung")
    return apply_resolution(image, int(value))


_LOAD_ACTIONS = (
    CalibrationAction.IMAGE_LOAD,
    CalibrationAction.PIXELS,
    CalibrationAction.PREPROCESSING,
)
_KIND_ACTIONS = {
    FeatureKind.EMBEDDING: CalibrationAction.EMBEDDINGS,
    FeatureKind.PROBABILITIES: CalibrationAction.INFERENCE,
    FeatureKind.QUALITY: CalibrationAction.SIGNAL_G2,
}


class FeatureStore:
    """Zugriffsgebundene Merkmalsquelle des Runners (keine eigene Wissenschaftslogik)."""

    def __init__(
        self,
        *,
        process: CalibrationProcess,
        dataset: CalibrationDataPort,
        model: CalibrationModelPort,
        preprocessor: ImagePreprocessor,
        g2_config: G2QualityConfig,
        protocol: CalibrationProtocolConfig,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size muss positiv sein")
        self._process = process
        self._dataset = dataset
        self._model = model
        self._preprocessor = preprocessor
        self._g2 = g2_config
        self._protocol = protocol
        self._batch_size = batch_size
        self._permits: dict[CalibrationPool, CalibrationAccessPermit] = {}
        self._pool_of: dict[str, CalibrationPool] = {}
        self._embeddings: dict[CacheKey, tuple[float, ...]] = {}
        self._probabilities: dict[CacheKey, tuple[float, ...]] = {}
        self._quality_rows: dict[CacheKey, QualityRow] = {}
        self._labels: dict[str, int] = {}

    # --- Pools und Berechtigungen ---------------------------------------------------------------

    def bind_pool(
        self, pool: CalibrationPool, permit: CalibrationAccessPermit, sample_ids: Iterable[str]
    ) -> None:
        if pool in self._permits:
            raise CalibrationProtocolError(f"{pool.value} ist bereits gebunden")
        for sample_id in sample_ids:
            known = self._pool_of.setdefault(sample_id, pool)
            if known is not pool:
                raise CalibrationProtocolError(f"{sample_id} gehört zu {known.value}")
        self._permits[pool] = permit

    def release_pool(self, pool: CalibrationPool) -> None:
        self._permits.pop(pool, None)

    def _pool(self, sample_ids: Sequence[str]) -> CalibrationPool:
        pools = {self._pool_of.get(s) for s in sample_ids}
        if len(pools) != 1 or None in pools:
            raise CalibrationProtocolError("Merkmale nur für Samples genau eines gebundenen Pools")
        pool = next(iter(pools))
        assert pool is not None
        return pool

    def _permit(self, pool: CalibrationPool) -> CalibrationAccessPermit:
        permit = self._permits.get(pool)
        if permit is None:
            raise CalibrationProtocolError(f"keine Berechtigung für {pool.value}")
        return permit

    def _allow(self, pool: CalibrationPool, actions: Iterable[CalibrationAction]) -> None:
        for action in actions:
            self._process.require_action(pool, action)

    # --- Bildzustände ---------------------------------------------------------------------------

    def _clean_states(self, sample_ids: Sequence[str]) -> list[npt.NDArray[np.float32]]:
        pool = self._pool(sample_ids)
        self._allow(pool, _LOAD_ACTIONS)
        images = self._dataset.load_inputs(self._permit(pool), list(sample_ids))
        if len(images) != len(sample_ids):
            raise CalibrationProtocolError("Datensatz lieferte nicht alle angefragten Bilder")
        states: list[npt.NDArray[np.float32]] = []
        for image in images:
            tensor = self._preprocessor.unnormalized(image)
            states.append(np.ascontiguousarray(tensor.numpy(), dtype=np.float32))
        return states

    def _compute(
        self,
        pool: CalibrationPool,
        sample_ids: Sequence[str],
        states: Sequence[FloatImage],
        kinds: frozenset[FeatureKind],
    ) -> list[SampleFeatures]:
        self._allow(pool, (_KIND_ACTIONS[k] for k in sorted(kinds)))
        embeddings: list[tuple[float, ...]] | None = None
        probabilities: list[tuple[float, ...]] | None = None
        if kinds & {FeatureKind.EMBEDDING, FeatureKind.PROBABILITIES}:
            inputs = [
                PreprocessedImage(torch.from_numpy(np.asarray(state, dtype=np.float32)))
                for state in states
            ]
            if FeatureKind.EMBEDDING in kinds:
                embeddings = list(self._model.embed(BATCH_ID, sample_ids, inputs).embeddings)
            if FeatureKind.PROBABILITIES in kinds:
                probabilities = list(
                    self._model.predict_proba(BATCH_ID, sample_ids, inputs).probabilities
                )
        quality: list[QualityRow] | None = None
        if FeatureKind.QUALITY in kinds:
            quality = [self._quality(s, state) for s, state in zip(sample_ids, states, strict=True)]
        return [
            SampleFeatures(
                embedding=None if embeddings is None else embeddings[i],
                probabilities=None if probabilities is None else probabilities[i],
                quality=None if quality is None else quality[i],
            )
            for i in range(len(sample_ids))
        ]

    def _quality(self, sample_id: str, state: FloatImage) -> QualityRow:
        result = compute_quality_indicators(sample_id, [sample_id], [state], self._g2)
        if isinstance(result, QualityInputFailure):
            return SampleInputFailure(status=result.status, reason_code=result.reason_code)
        return result.indicators[0]

    def _chunks(self, sample_ids: Sequence[str]) -> Iterable[list[str]]:
        for start in range(0, len(sample_ids), self._batch_size):
            yield list(sample_ids[start : start + self._batch_size])

    # --- Öffentliche Merkmalsabfragen -------------------------------------------------------------

    def _complete(self, key: CacheKey, kinds: frozenset[FeatureKind]) -> bool:
        stores: dict[FeatureKind, Mapping[CacheKey, object]] = {
            FeatureKind.EMBEDDING: self._embeddings,
            FeatureKind.PROBABILITIES: self._probabilities,
            FeatureKind.QUALITY: self._quality_rows,
        }
        return all(key in stores[k] for k in kinds)

    def _cached(
        self,
        sample_ids: Sequence[str],
        transform: tuple[str, float],
        kinds: frozenset[FeatureKind],
        compute: Callable[[list[str]], list[SampleFeatures]],
    ) -> list[SampleFeatures]:
        missing = [
            s for s in dict.fromkeys(sample_ids) if not self._complete((s, *transform), kinds)
        ]
        for chunk in self._chunks(missing):
            for sample_id, features in zip(chunk, compute(chunk), strict=True):
                key = (sample_id, *transform)
                if features.embedding is not None:
                    self._embeddings[key] = features.embedding
                if features.probabilities is not None:
                    self._probabilities[key] = features.probabilities
                if features.quality is not None:
                    self._quality_rows[key] = features.quality
        result: list[SampleFeatures] = []
        for sample_id in sample_ids:
            key = (sample_id, *transform)
            result.append(
                SampleFeatures(
                    embedding=self._embeddings[key] if FeatureKind.EMBEDDING in kinds else None,
                    probabilities=(
                        self._probabilities[key] if FeatureKind.PROBABILITIES in kinds else None
                    ),
                    quality=self._quality_rows[key] if FeatureKind.QUALITY in kinds else None,
                )
            )
        return result

    def clean(
        self, sample_ids: Sequence[str], kinds: frozenset[FeatureKind] = ALL_KINDS
    ) -> list[SampleFeatures]:
        pool = self._pool(sample_ids)

        def compute(chunk: list[str]) -> list[SampleFeatures]:
            return self._compute(pool, chunk, self._clean_states(chunk), kinds)

        return self._cached(sample_ids, ("CLEAN", 0.0), kinds, compute)

    def shifted(
        self,
        sample_ids: Sequence[str],
        shift: TechnicalShift,
        values: Sequence[float],
        kinds: frozenset[FeatureKind],
        *,
        repeat_id: int,
    ) -> dict[float, list[SampleFeatures]]:
        """Merkmale je Shift-Wert; Noise mit `seed_ε(repeat_id, sample_id)` (DEC-054)."""
        pool = self._pool(sample_ids)
        if shift is not TechnicalShift.NOISE:
            result: dict[float, list[SampleFeatures]] = {}
            for value in values:

                def compute(chunk: list[str], value: float = value) -> list[SampleFeatures]:
                    states = [
                        apply_shift(shift, state, value, None, self._protocol)
                        for state in self._clean_states(chunk)
                    ]
                    return self._compute(pool, chunk, states, kinds)

                result[value] = self._cached(sample_ids, (shift.value, value), kinds, compute)
            return result
        output: dict[float, list[SampleFeatures]] = {value: [] for value in values}
        split_hash = self._process.split_hash
        for chunk in self._chunks(sample_ids):
            clean_states = self._clean_states(chunk)
            epsilons = [
                noise_epsilon(
                    noise_seed(split_hash, self._protocol.noise_seed_namespace, repeat_id, s)
                )
                for s in chunk
            ]
            for value in values:
                states = [
                    apply_shift(shift, state, value, eps, self._protocol)
                    for state, eps in zip(clean_states, epsilons, strict=True)
                ]
                output[value].extend(self._compute(pool, chunk, states, kinds))
        return output

    def anonymous(
        self,
        pool: CalibrationPool,
        positions: Sequence[str],
        images: Sequence[Image.Image],
        kinds: frozenset[FeatureKind] = ALL_KINDS,
    ) -> list[SampleFeatures]:
        """Merkmale ID-freier Bildeingaben (AUDIT_CLEAN, P4-02F): `positions` sind opake
        Positionskennungen innerhalb des Windows; es wird nichts zwischengespeichert."""
        if len(positions) != len(images) or len(set(positions)) != len(positions):
            raise CalibrationProtocolError("Positionen und Bilder müssen eindeutig zugeordnet sein")
        self._allow(pool, _LOAD_ACTIONS)
        features: list[SampleFeatures] = []
        for start in range(0, len(images), self._batch_size):
            chunk = list(positions[start : start + self._batch_size])
            states = [
                np.ascontiguousarray(
                    self._preprocessor.unnormalized(image).numpy(), dtype=np.float32
                )
                for image in images[start : start + self._batch_size]
            ]
            features.extend(self._compute(pool, chunk, states, kinds))
        return features

    def labels(self, sample_ids: Sequence[str]) -> list[int]:
        """Ground-Truth-Klassenindex (nur `CAL_REF`/`CAL_TUNE`; G4 auf CAL_AUDIT ist gesperrt)."""
        pool = self._pool(sample_ids)
        self._allow(pool, (CalibrationAction.SIGNAL_G4, CalibrationAction.PERFORMANCE))
        missing = [s for s in dict.fromkeys(sample_ids) if s not in self._labels]
        if missing:
            labels = self._dataset.labels_for(self._permit(pool), missing)
            if set(labels) != set(missing):
                raise CalibrationProtocolError("Datensatz lieferte nicht alle Labels")
            self._labels.update(labels)
        return [self._labels[s] for s in sample_ids]

    def metadata(self, sample_ids: Sequence[str]) -> Mapping[str, SampleMetadata]:
        """Metadaten ausschließlich für den `sex`-Audit auf `CAL_REF`/`CAL_TUNE`."""
        pool = self._pool(sample_ids)
        self._allow(pool, (CalibrationAction.SUBGROUP_METADATA,))
        metadata = self._dataset.metadata_for(self._permit(pool), list(sample_ids))
        if set(metadata) != set(sample_ids):
            raise CalibrationProtocolError("Datensatz lieferte nicht alle Metadaten")
        return metadata

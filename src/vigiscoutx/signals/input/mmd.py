"""G1 – MMD auf dem 512-d Penultimate-Embedding (DEC-042, MONITORING_METHODS_BASELINE 2).

- Schätzer: unverzerrtes quadratisches `MMD_u²` (Gretton et al., Lemma 6, Eq. (3)), nicht auf 0
  begrenzt; negative Werte sind zulässig.
- Kernel: `k(x, y) = exp(−‖x − y‖² / (2σ²))` mit eingefrorenem, übergebenem `σ`.
- Inferenz: `B` unabhängig gezogene Permutationen des gepoolten Samples mit festen Gruppengrößen,
  rechter Tail, `p = (1 + #{T_b ≥ T_obs}) / (B + 1)`: gültiger, konservativer
  Monte-Carlo-Permutations-p-Wert (kein exakter Permutations-p-Wert).

Die gepoolte Kernelmatrix wird einmal berechnet und für alle Permutationen wiederverwendet;
`T` hängt nur von der Partition ab, daher liefert jede Permutation mit derselben Partition wie
die Beobachtung exakt denselben Wert wie `T_obs`.
"""

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, pdist

from vigiscoutx.domain.assessments import (
    ReasonCode,
    SeedRef,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.refs import ModelSnapshotRef, SeedBundle
from vigiscoutx.domain.types import FrozenModel, NonNegativeInt, SignalGroup
from vigiscoutx.domain.windows import RepresentationBatch
from vigiscoutx.errors import CalibrationHelperError
from vigiscoutx.signals.config import G1MmdConfig
from vigiscoutx.signals.statistics import is_significant

FloatArray = npt.NDArray[np.float64]
MMD_METHOD = "MMD_U2_RBF_PERMUTATION"


def rbf_kernel(x: FloatArray, y: FloatArray, sigma: float) -> FloatArray:
    """`exp(−‖x − y‖² / (2σ²))` für alle Paare (Zeilen von x gegen Zeilen von y)."""
    _require_sigma(sigma)
    squared = cdist(x, y, metric="sqeuclidean")
    return np.asarray(np.exp(-squared / (2.0 * sigma**2)), dtype=np.float64)


def mmd2_unbiased(x: FloatArray, y: FloatArray, sigma: float) -> float:
    """Direkte Auswertung von Eq. (3); Referenzimplementierung für Tests."""
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("MMD_u² erfordert m ≥ 2 und n ≥ 2")
    kxx, kyy, kxy = rbf_kernel(x, x, sigma), rbf_kernel(y, y, sigma), rbf_kernel(x, y, sigma)
    return float(
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        - 2.0 * kxy.sum() / (m * n)
    )


def _mmd2_for_mask(kernel: FloatArray, diagonal: FloatArray, mask: FloatArray) -> float:
    """`MMD_u²` einer Partition; `mask` ist der Indikator der Gruppe der Größe m.

    Beobachtung und alle Permutationen nutzen exakt diesen Rechenpfad, damit gleiche Partitionen
    bitgleiche Werte liefern.
    """
    other = 1.0 - mask
    m, n = float(mask.sum()), float(other.sum())
    kx, ky = mask @ kernel, other @ kernel
    sum_xx = float(kx @ mask - mask @ diagonal)
    sum_yy = float(ky @ other - other @ diagonal)
    sum_xy = float(kx @ other)
    return sum_xx / (m * (m - 1.0)) + sum_yy / (n * (n - 1.0)) - 2.0 * sum_xy / (m * n)


class PermutationOutcome(FrozenModel):
    mmd2_unbiased: float
    exceedances: NonNegativeInt
    n_permutations: NonNegativeInt
    p_value: float


def permutation_p_value(exceedances: int, n_permutations: int) -> float:
    """`p = (1 + #{T_b ≥ T_obs}) / (B + 1)`."""
    if not 0 <= exceedances <= n_permutations:
        raise ValueError("Anzahl Überschreitungen muss in [0, B] liegen")
    return (1 + exceedances) / (n_permutations + 1)


def mmd_permutation_test(
    reference: FloatArray, monitoring: FloatArray, sigma: float, n_permutations: int, seed: int
) -> PermutationOutcome:
    """Beobachtetes `MMD_u²` und Monte-Carlo-Permutations-p-Wert (Reference bleibt unverändert)."""
    _require_sigma(sigma)
    m, n = reference.shape[0], monitoring.shape[0]
    if m < 2 or n < 2:
        raise ValueError("MMD_u² erfordert m ≥ 2 und n ≥ 2")
    pooled = np.concatenate([reference, monitoring], axis=0)
    kernel = rbf_kernel(pooled, pooled, sigma)
    diagonal = np.ascontiguousarray(np.diag(kernel))
    mask = np.zeros(m + n, dtype=np.float64)
    mask[:m] = 1.0
    observed = _mmd2_for_mask(kernel, diagonal, mask)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        mask = np.zeros(m + n, dtype=np.float64)
        mask[rng.permutation(m + n)[:m]] = 1.0
        if _mmd2_for_mask(kernel, diagonal, mask) >= observed:
            exceedances += 1
    return PermutationOutcome(
        mmd2_unbiased=observed,
        exceedances=exceedances,
        n_permutations=n_permutations,
        p_value=permutation_p_value(exceedances, n_permutations),
    )


def median_positive_distance(embeddings: FloatArray) -> float:
    """Bandbreitenregel (nur für den späteren Calibration-Schritt).

    Median der positiven paarweisen euklidischen Distanzen (ungeordnete Paare, Distanz 0
    ausgeschlossen); bei gerader Anzahl arithmetisches Mittel der beiden mittleren Werte.
    """
    data = np.asarray(embeddings, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        raise CalibrationHelperError("mindestens zwei Embeddings erforderlich")
    if not np.isfinite(data).all():
        raise CalibrationHelperError("nichtfinite Embeddings")
    distances = np.sort(pdist(data, metric="euclidean"))
    positive = distances[distances > 0.0]
    if positive.size == 0:
        raise CalibrationHelperError("keine positive paarweise Distanz")
    middle = positive.size // 2
    sigma = (
        float(positive[middle])
        if positive.size % 2
        else float((positive[middle - 1] + positive[middle]) / 2.0)
    )
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise CalibrationHelperError("ungültige Bandbreite")
    return sigma


def _require_sigma(sigma: float) -> None:
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma muss endlich und > 0 sein")


def _as_matrix(batch: RepresentationBatch) -> FloatArray:
    matrix = np.array(batch.embeddings, dtype=np.float64)
    matrix.setflags(write=False)
    return matrix


def run_g1_mmd(
    reference: RepresentationBatch,
    monitoring: RepresentationBatch,
    config: G1MmdConfig,
    seeds: SeedBundle,
    model_snapshot_ref: ModelSnapshotRef,
) -> SignalResult:
    """G1_MMD_EMBEDDING für ein Reference-/Monitoring-Paar."""
    seed_ref = SeedRef(
        seed_field=config.seed_field, seed=seeds.statistical_seed, repeat_id=seeds.repeat_id
    )
    ref, mon = _as_matrix(reference), _as_matrix(monitoring)
    base = {
        "signal_id": config.signal_id,
        "group": SignalGroup.G1_INPUT,
        "method": MMD_METHOD,
        "method_version": config.method_version,
        "components": (),
        "parameters_ref": config.parameters_ref(),
        "reference_window_id": reference.window_id,
        "monitoring_window_id": monitoring.window_id,
        "reference_n": ref.shape[0],
        "monitoring_n": mon.shape[0],
        "seed_ref": seed_ref,
        "model_snapshot_ref": model_snapshot_ref,
        "artifact_refs": (),
        "evidence_refs": (),
    }
    fixed = {
        "sigma": config.sigma,
        "n_permutations": config.n_permutations,
        "alpha": config.alpha,
    }

    def failure(status: SignalStatus, reason: ReasonCode) -> SignalResult:
        return SignalResult.model_validate(
            {
                **base,
                "status": status,
                "statistics": FrozenMapping(fixed),
                "p_value": None,
                "adjusted_p_value": None,
                "alpha_or_threshold": config.alpha,
                "alarm": None,
                "reason_code": reason,
            }
        )

    if ref.shape[1] != config.embedding_dim or mon.shape[1] != config.embedding_dim:
        return failure(SignalStatus.ERROR, ReasonCode.EMBEDDING_DIMENSION_MISMATCH)
    if not (np.isfinite(ref).all() and np.isfinite(mon).all()):
        return failure(SignalStatus.ERROR, ReasonCode.NONFINITE_INPUT)
    if ref.shape[0] < 2 or mon.shape[0] < 2:
        return failure(SignalStatus.NOT_EVALUABLE, ReasonCode.INSUFFICIENT_SAMPLE)
    try:
        outcome = mmd_permutation_test(
            ref, mon, config.sigma, config.n_permutations, seeds.statistical_seed
        )
    except (ValueError, FloatingPointError, MemoryError):
        return failure(SignalStatus.ERROR, ReasonCode.MMD_COMPUTATION_FAILED)
    if not math.isfinite(outcome.mmd2_unbiased):
        return failure(SignalStatus.ERROR, ReasonCode.MMD_COMPUTATION_FAILED)
    return SignalResult.model_validate(
        {
            **base,
            "status": SignalStatus.VALID,
            "statistics": FrozenMapping(
                {
                    **fixed,
                    "mmd2_unbiased": outcome.mmd2_unbiased,
                    "exceedances": outcome.exceedances,
                }
            ),
            "p_value": outcome.p_value,
            "adjusted_p_value": None,
            "alpha_or_threshold": config.alpha,
            "alarm": is_significant(outcome.p_value, config.alpha),
            "reason_code": None,
        }
    )

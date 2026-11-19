"""G1 – MMD_u² mit RBF-Kernel und Permutationsinferenz (DEC-042)."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from signal_factories import MODEL_REF, g1_config
from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.domain.windows import RepresentationBatch
from vigiscoutx.errors import CalibrationHelperError
from vigiscoutx.signals.config import BASELINE_PERMUTATIONS
from vigiscoutx.signals.input.mmd import (
    median_positive_distance,
    mmd2_unbiased,
    mmd_permutation_test,
    permutation_p_value,
    rbf_kernel,
    run_g1_mmd,
)


def _seeds(statistical_seed: int = 7) -> SeedBundle:
    return SeedBundle(
        split_seed=0,
        model_development_seed=0,
        sampling_seed=0,
        shift_seed=0,
        statistical_seed=statistical_seed,
        repeat_id=3,
    )


def _batch(window_id: str, data: np.ndarray) -> RepresentationBatch:
    return RepresentationBatch(
        window_id=window_id,
        sample_ids=tuple(f"{window_id}-{i}" for i in range(len(data))),
        embeddings=tuple(tuple(float(v) for v in row) for row in data),
    )


def _manual_mmd2(x: list[list[float]], y: list[list[float]], sigma: float) -> float:
    def k(a: list[float], b: list[float]) -> float:
        return math.exp(-sum((p - q) ** 2 for p, q in zip(a, b, strict=True)) / (2 * sigma**2))

    m, n = len(x), len(y)
    xx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    yy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    xy = sum(k(a, b) for a in x for b in y) / (m * n)
    return xx + yy - 2 * xy


def test_rbf_kernel_matches_manual_value() -> None:
    kernel = rbf_kernel(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0], [0.0, 0.0]]), sigma=5.0)
    assert kernel[0, 0] == pytest.approx(math.exp(-25 / 50))
    assert kernel[0, 1] == 1.0


def test_unbiased_mmd2_matches_manual_reference() -> None:
    x = [[0.0, 1.0], [1.0, 0.5], [2.0, -1.0]]
    y = [[3.0, 0.0], [4.0, 1.0]]
    expected = _manual_mmd2(x, y, sigma=1.5)
    assert mmd2_unbiased(np.array(x), np.array(y), 1.5) == pytest.approx(expected, abs=1e-14)
    pooled = mmd_permutation_test(np.array(x), np.array(y), 1.5, 9, seed=0)
    assert pooled.mmd2_unbiased == pytest.approx(expected, abs=1e-14)


def test_negative_mmd2_is_not_clipped() -> None:
    x = np.array([[0.0], [1.0]])
    k01 = math.exp(-0.5)
    value = mmd2_unbiased(x, x.copy(), 1.0)
    assert value == pytest.approx(k01 - 1.0)
    assert value < 0
    assert mmd_permutation_test(x, x.copy(), 1.0, 9, seed=0).mmd2_unbiased < 0


@pytest.mark.parametrize("sigma", [0.0, -1.0, math.inf, math.nan])
def test_invalid_sigma_is_rejected(sigma: float) -> None:
    with pytest.raises(ValidationError):
        g1_config(sigma=sigma)
    with pytest.raises(ValueError):
        rbf_kernel(np.zeros((1, 1)), np.zeros((1, 1)), sigma)


def test_median_rule_odd_count() -> None:
    assert median_positive_distance(np.array([[0.0], [1.0], [3.0]])) == 2.0


def test_median_rule_even_count_averages_middle_values() -> None:
    assert median_positive_distance(np.array([[0.0], [1.0], [3.0], [7.0]])) == 3.5


def test_median_rule_ignores_zero_distances() -> None:
    assert median_positive_distance(np.array([[0.0], [0.0], [1.0]])) == 1.0


@pytest.mark.parametrize(
    "data",
    [np.array([[1.0], [1.0]]), np.array([[0.0]]), np.array([[0.0], [math.nan]])],
)
def test_median_rule_undefined_raises(data: np.ndarray) -> None:
    with pytest.raises(CalibrationHelperError):
        median_positive_distance(data)


def test_permutation_p_value_formula() -> None:
    assert permutation_p_value(0, 999) == 1 / 1000
    assert permutation_p_value(49, 999) == 0.05
    assert permutation_p_value(999, 999) == 1.0
    with pytest.raises(ValueError):
        permutation_p_value(1000, 999)


def test_baseline_permutation_count_is_exactly_999() -> None:
    assert BASELINE_PERMUTATIONS == 999
    assert g1_config().n_permutations == 999
    with pytest.raises(ValidationError):
        g1_config().model_copy(update={"n_permutations": 998})


def test_permutations_are_deterministic_per_seed() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(12, 3)), rng.normal(size=(10, 3))
    first = mmd_permutation_test(x, y, 1.0, 999, seed=11)
    assert mmd_permutation_test(x, y, 1.0, 999, seed=11) == first
    others = {mmd_permutation_test(x, y, 1.0, 999, seed=s).exceedances for s in range(12, 17)}
    assert others - {first.exceedances}


def test_identical_partition_counts_as_exceedance() -> None:
    x, y = np.array([[0.0], [1.0]]), np.array([[5.0], [6.0]])
    outcome = mmd_permutation_test(x, y, 1.0, 999, seed=0)
    # C(4, 2) = 6 Partitionen; die beobachtete (und ihr Spiegelbild) ist die extremste.
    assert outcome.exceedances > 0
    assert outcome.p_value == (1 + outcome.exceedances) / 1000


def test_reference_is_not_mutated() -> None:
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(6, 512)), rng.normal(size=(5, 512))
    before = x.copy()
    mmd_permutation_test(x, y, 30.0, 99, seed=0)
    assert np.array_equal(x, before)
    reference = _batch("ref", x)
    snapshot = reference.model_dump()
    run_g1_mmd(reference, _batch("mon", y), g1_config(30.0), _seeds(), MODEL_REF)
    assert reference.model_dump() == snapshot


def test_run_g1_detects_shift_and_stores_provenance() -> None:
    rng = np.random.default_rng(3)
    reference = _batch("ref", rng.normal(size=(20, 512)))
    shifted = _batch("mon", rng.normal(loc=0.5, size=(20, 512)))
    result = run_g1_mmd(reference, shifted, g1_config(32.0), _seeds(), MODEL_REF)
    assert result.status is SignalStatus.VALID
    assert result.signal_id == "G1_MMD_EMBEDDING"
    assert result.alarm is True
    assert result.p_value == 1 / 1000
    stats = result.statistics
    assert {"mmd2_unbiased", "sigma", "n_permutations", "alpha", "exceedances"} <= set(stats)
    assert (stats["sigma"], stats["n_permutations"], stats["alpha"]) == (32.0, 999, 0.05)
    assert (result.reference_n, result.monitoring_n) == (20, 20)
    assert result.seed_ref is not None
    assert (result.seed_ref.seed_field, result.seed_ref.seed) == ("statistical_seed", 7)
    assert result.model_snapshot_ref == MODEL_REF
    assert result.parameters_ref == g1_config(32.0).parameters_ref()


def test_run_g1_same_distribution_alarm_follows_p_rule() -> None:
    rng = np.random.default_rng(4)
    reference = _batch("ref", rng.normal(size=(15, 512)))
    monitoring = _batch("mon", rng.normal(size=(15, 512)))
    result = run_g1_mmd(reference, monitoring, g1_config(32.0), _seeds(), MODEL_REF)
    assert result.status is SignalStatus.VALID
    assert result.p_value is not None
    assert result.alarm is (result.p_value <= 0.05)


def test_run_g1_small_sample_is_not_evaluable() -> None:
    rng = np.random.default_rng(5)
    result = run_g1_mmd(
        _batch("ref", rng.normal(size=(1, 512))),
        _batch("mon", rng.normal(size=(3, 512))),
        g1_config(),
        _seeds(),
        MODEL_REF,
    )
    assert result.status is SignalStatus.NOT_EVALUABLE
    assert result.alarm is None
    assert result.reason_code is ReasonCode.INSUFFICIENT_SAMPLE


def test_run_g1_wrong_dimension_is_error() -> None:
    rng = np.random.default_rng(6)
    result = run_g1_mmd(
        _batch("ref", rng.normal(size=(3, 4))),
        _batch("mon", rng.normal(size=(3, 4))),
        g1_config(),
        _seeds(),
        MODEL_REF,
    )
    assert result.status is SignalStatus.ERROR
    assert result.alarm is None
    assert result.reason_code is ReasonCode.EMBEDDING_DIMENSION_MISMATCH

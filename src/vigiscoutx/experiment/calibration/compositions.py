"""Class-Distribution- und `sex`-Subgroup-Kompositionen sowie Schedules (DEC-055, DEC-056).

Delay/Lead werden erst in der späteren Evaluation berechnet und sind hier nicht implementiert.

Ausschließlich die eingefrorenen deterministischen Konstruktionsregeln; keine Performance-
optimierung, keine weitere Metadatendimension.
"""

import itertools
from collections.abc import Mapping, Sequence
from enum import StrEnum
from fractions import Fraction

from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt
from vigiscoutx.errors import CalibrationStopError
from vigiscoutx.experiment.calibration.config import CalibrationProtocolConfig, StopReason
from vigiscoutx.experiment.calibration.structure import ClassVector, round_half_up

# --- Class-Distribution-Shift -------------------------------------------------------------------


class ClassShiftPlan(FrozenModel):
    source: Identifier
    target: Identifier
    d_max: NonNegativeInt
    levels: tuple[NonNegativeInt, ...]
    compositions: tuple[tuple[NonNegativeInt, ...], ...]


def _levels(maximum: int, config: CalibrationProtocolConfig) -> tuple[int, ...]:
    return tuple(round_half_up(Fraction(str(f)) * maximum) for f in config.shift_level_fractions)


def _strictly_increasing_positive(levels: Sequence[int]) -> bool:
    return levels[0] > 0 and all(a < b for a, b in itertools.pairwise(levels))


def class_shift_plan(
    vector: ClassVector, tune_available: Mapping[str, int], config: CalibrationProtocolConfig
) -> ClassShiftPlan:
    """source = größtes, target = zweitgrößtes `n_c(W)` (Gleichstand → kanonische Reihenfolge);
    `d_max = min(n_source − 1, A_target − n_target)`; Levels 25/50/75 % half-up."""
    order = config.class_order
    ranked = sorted(order, key=lambda c: (-vector.as_dict()[c], order.index(c)))
    source, target = ranked[0], ranked[1]
    counts = vector.as_dict()
    d_max = min(counts[source] - 1, tune_available[target] - counts[target])
    levels = _levels(max(d_max, 0), config)
    if d_max <= 0 or not _strictly_increasing_positive(levels):
        raise CalibrationStopError(StopReason.CLASS_SHIFT_LEVELS_INVALID, f"Levels {levels}")
    compositions = []
    for d in levels:
        shifted = dict(counts)
        shifted[source] -= d
        shifted[target] += d
        compositions.append(tuple(shifted[c] for c in order))
    return ClassShiftPlan(
        source=source,
        target=target,
        d_max=d_max,
        levels=levels,
        compositions=tuple(compositions),
    )


# --- sex-Subgroup-Shift ----------------------------------------------------------------------


class SubgroupExclusion(StrEnum):
    """Gründe für `NOT_FEASIBLE_BY_PROTOCOL` (vollständiger Ausschluss, keine Ersatzdimension)."""

    FURTHER_NONEMPTY_CATEGORY = "FURTHER_NONEMPTY_CATEGORY"
    REFERENCE_SEX_NOT_USABLE = "REFERENCE_SEX_NOT_USABLE"
    CLEAN_COMPOSITION_NOT_REPRODUCIBLE = "CLEAN_COMPOSITION_NOT_REPRODUCIBLE"
    NO_FEASIBLE_REPLACEMENT = "NO_FEASIBLE_REPLACEMENT"
    LEVELS_NOT_STRICT = "LEVELS_NOT_STRICT"
    LEVEL_COMPOSITION_INFEASIBLE = "LEVEL_COMPOSITION_INFEASIBLE"


class SubgroupComposition(FrozenModel):
    """Je Klasse (kanonische Reihenfolge) Anzahl source- bzw. target-Läsionen."""

    source_counts: tuple[NonNegativeInt, ...]
    target_counts: tuple[NonNegativeInt, ...]


class SubgroupPlan(FrozenModel):
    feasible: bool
    exclusion: SubgroupExclusion | None
    source: Identifier | None
    target: Identifier | None
    d_max: NonNegativeInt | None
    levels: tuple[NonNegativeInt, ...]
    compositions: tuple[SubgroupComposition, ...]


def _excluded(reason: SubgroupExclusion) -> SubgroupPlan:
    return SubgroupPlan(
        feasible=False,
        exclusion=reason,
        source=None,
        target=None,
        d_max=None,
        levels=(),
        compositions=(),
    )


def _usable(value: str | None, config: CalibrationProtocolConfig) -> str | None:
    """`male`/`female` verwertbar; `None`, leer, `null`, `missing`, `unknown` nicht verwertbar."""
    if value is None or value in config.subgroup_unusable_values:
        return None
    return value


def _distribute(d: int, q_max: Sequence[int], config: CalibrationProtocolConfig) -> list[int]:
    total = sum(q_max)
    raw = [Fraction(d * q, total) for q in q_max]
    base = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for index in order[: d - sum(base)]:
        base[index] += 1
    return base


def subgroup_plan(
    reference_sex_by_class: Mapping[str, Sequence[str | None]],
    cal_ref_sex: Sequence[str | None],
    cal_tune_sex_by_class: Mapping[str, Sequence[str | None]],
    config: CalibrationProtocolConfig,
) -> SubgroupPlan:
    """`sex`-Subgroup-Shift gemäß CALIBRATION_PROTOCOL_BASELINE 8.2.

    `reference_sex_by_class`: `sex` der Läsionen des offiziellen Reference Windows je Klasse;
    `cal_ref_sex`: `sex` aller `CAL_REF`-Läsionen; `cal_tune_sex_by_class`: `sex` der
    verfügbaren `CAL_TUNE`-Läsionen je Klasse. Nur `CAL_REF`/`CAL_TUNE`, nie `CAL_AUDIT`.
    """
    order = config.class_order
    categories = set(config.subgroup_categories)
    observed = [
        *cal_ref_sex,
        *(v for c in order for v in cal_tune_sex_by_class.get(c, ())),
        *(v for c in order for v in reference_sex_by_class.get(c, ())),
    ]
    if any((u := _usable(v, config)) is not None and u not in categories for v in observed):
        return _excluded(SubgroupExclusion.FURTHER_NONEMPTY_CATEGORY)
    reference = {c: [_usable(v, config) for v in reference_sex_by_class[c]] for c in order}
    if any(v is None for values in reference.values() for v in values):
        return _excluded(SubgroupExclusion.REFERENCE_SEX_NOT_USABLE)
    usable_ref = [u for v in cal_ref_sex if (u := _usable(v, config)) is not None]
    male, female = usable_ref.count("male"), usable_ref.count("female")
    rarer = "male" if male < female else "female"
    target = config.subgroup_tie_target if male == female else rarer
    source = "female" if target == "male" else "male"
    s0 = [reference[c].count(source) for c in order]
    t0 = [reference[c].count(target) for c in order]
    tune = {c: [_usable(v, config) for v in cal_tune_sex_by_class.get(c, ())] for c in order}
    a_s = [tune[c].count(source) for c in order]
    a_t = [tune[c].count(target) for c in order]
    if any(a < s for a, s in zip(a_s, s0, strict=True)) or any(
        a < t for a, t in zip(a_t, t0, strict=True)
    ):
        return _excluded(SubgroupExclusion.CLEAN_COMPOSITION_NOT_REPRODUCIBLE)
    q_max = [min(s, a - t) for s, a, t in zip(s0, a_t, t0, strict=True)]
    d_max = sum(q_max)
    if d_max == 0:
        return _excluded(SubgroupExclusion.NO_FEASIBLE_REPLACEMENT)
    levels = _levels(d_max, config)
    if not _strictly_increasing_positive(levels) or levels[-1] > d_max:
        return _excluded(SubgroupExclusion.LEVELS_NOT_STRICT)
    compositions = []
    for d in levels:
        q = _distribute(d, q_max, config)
        if any(qc > qm for qc, qm in zip(q, q_max, strict=True)):
            return _excluded(SubgroupExclusion.LEVEL_COMPOSITION_INFEASIBLE)
        sources = tuple(s - qc for s, qc in zip(s0, q, strict=True))
        targets = tuple(t + qc for t, qc in zip(t0, q, strict=True))
        if any(
            src > a or tgt > b for src, tgt, a, b in zip(sources, targets, a_s, a_t, strict=True)
        ):
            return _excluded(SubgroupExclusion.LEVEL_COMPOSITION_INFEASIBLE)
        compositions.append(SubgroupComposition(source_counts=sources, target_counts=targets))
    return SubgroupPlan(
        feasible=True,
        exclusion=None,
        source=source,
        target=target,
        d_max=d_max,
        levels=levels,
        compositions=tuple(compositions),
    )


# --- Schedules (DEC-056) --------------------------------------------------------

CLEAN = "CLEAN"


def abrupt_schedule(level: str) -> tuple[str, ...]:
    """`t0` Clean, `t1` `L_j`."""
    return (CLEAN, level)


def gradual_schedule(levels: Sequence[str]) -> tuple[str, ...]:
    """`t0` Clean, `t1` `L1`, `t2` `L2`, `t3` `L3`; keine Zwischenstufen."""
    if len(levels) != 3:
        raise ValueError("graduelle Schedules haben exakt drei Levels")
    return (CLEAN, *levels)

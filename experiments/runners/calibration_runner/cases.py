"""Case-Matrix und Window-Konstruktion (CALIBRATION_ORCHESTRATION_BASELINE 3–4, DEC-061/063).

- `repeat_id` ist in allen Namespaces 1-basiert.
- Clean-Cases und Base-Windows: Ranking-Kontext = Case-Kontext.
- Abrupte Shift-Cases: Ranking über `ABRUPT_<FAMILY>_BASE` (Anchor, kein Evaluationsfall);
  `window_id` und G1-Seed aus dem Level-`scenario_key` `ABRUPT_<FAMILY>_L1/L2/L3`.
- Technische Familien: L1–L3 mit identischen `sample_id`s des Base-Windows.
- `CLASS`/`SUBGROUP_SEX`: eingefrorene Zielanzahlen je Stratum als Prefix der einmaligen
  Anchor-Rangfolge; kein level-spezifisches Um-Sampling.

Signalreihenfolge (technisch): Samples eines Windows werden für alle Signale in sortierter
`sample_id`-Reihenfolge verarbeitet – derselben Ordnung, über die `window_id` definiert ist.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vigiscoutx.errors import CalibrationProtocolError, IntegrityError
from vigiscoutx.experiment.calibration.compositions import CLEAN, ClassShiftPlan, SubgroupPlan
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    SamplingNamespace,
    TechnicalShift,
)
from vigiscoutx.experiment.calibration.process import SelectedSeverity
from vigiscoutx.experiment.calibration.structure import (
    ClassVector,
    sample_window,
    sampling_context,
    window_id,
)

SEVERITY_BASE = "SEVERITY_BASE"
CLASS_FAMILY = "CLASS"
SUBGROUP_FAMILY = "SUBGROUP_SEX"


def base_key(family: str) -> str:
    return f"ABRUPT_{family}_BASE"


def level_key(family: str, level: int) -> str:
    return f"ABRUPT_{family}_L{level}"


@dataclass(frozen=True, slots=True)
class CaseWindow:
    """Eindeutig zugeordnetes Monitoring-Window eines Cases inklusive Provenienz."""

    namespace: SamplingNamespace
    window_size: int
    scenario_key: str
    repeat_id: int
    pool: CalibrationPool
    context: str
    base_context: str
    window_id: str
    base_window_id: str
    lesion_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    family: str | None
    level: int | None
    shift: TechnicalShift | None
    shift_value: float | None

    @property
    def is_shift(self) -> bool:
        return self.level is not None

    def provenance(self, reference_window_id: str, seed: int | None) -> dict[str, object]:
        return {
            "namespace": self.namespace.value,
            "window_size": self.window_size,
            "scenario_key": self.scenario_key,
            "repeat_id": self.repeat_id,
            "family": self.family,
            "level": self.level,
            "shift_value": self.shift_value,
            "reference_window_id": reference_window_id,
            "monitoring_window_id": self.window_id,
            "base_window_id": self.base_window_id,
            "statistical_seed": seed,
        }


def verify_case(case: CaseWindow, split_hash: str) -> None:
    """Window muss eindeutig aus `namespace + W + scenario_key + repeat_id + sample_ids`
    hervorgehen und die erwartete deterministische `window_id` tragen (vor G1)."""
    expected_context = sampling_context(
        case.namespace, case.window_size, case.scenario_key, case.repeat_id
    )
    if case.context != expected_context:
        raise IntegrityError(f"Case-Kontext weicht ab: {case.context}")
    if tuple(sorted(case.sample_ids)) != case.sample_ids or len(set(case.sample_ids)) != len(
        case.sample_ids
    ):
        raise IntegrityError("Case-Samples müssen eindeutig und sortiert sein")
    if window_id(split_hash, case.context, case.sample_ids) != case.window_id:
        raise IntegrityError(f"window_id des Cases {case.scenario_key}/{case.repeat_id} weicht ab")
    if case.window_size != len(case.sample_ids):
        raise IntegrityError("Case-Window hat nicht die Größe W")


def _window(
    *,
    namespace: SamplingNamespace,
    window_size: int,
    scenario_key: str,
    repeat_id: int,
    pool: CalibrationPool,
    split_hash: str,
    available: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    canonical: Mapping[str, str],
    anchor_key: str | None = None,
    family: str | None = None,
    level: int | None = None,
    shift: TechnicalShift | None = None,
    shift_value: float | None = None,
) -> CaseWindow:
    if repeat_id < 1:
        raise CalibrationProtocolError("repeat_id ist 1-basiert")
    context = sampling_context(namespace, window_size, scenario_key, repeat_id)
    base_context = (
        context
        if anchor_key is None
        else sampling_context(namespace, window_size, anchor_key, repeat_id)
    )
    selection = sample_window(split_hash, base_context, available, counts, canonical)
    samples = tuple(sorted(selection.sample_ids))
    return CaseWindow(
        namespace=namespace,
        window_size=window_size,
        scenario_key=scenario_key,
        repeat_id=repeat_id,
        pool=pool,
        context=context,
        base_context=base_context,
        window_id=window_id(split_hash, context, samples),
        base_window_id=selection.window_id,
        lesion_ids=selection.lesion_ids,
        sample_ids=samples,
        family=family,
        level=level,
        shift=shift,
        shift_value=shift_value,
    )


def clean_window(
    namespace: SamplingNamespace,
    vector: ClassVector,
    repeat_id: int,
    *,
    pool: CalibrationPool,
    split_hash: str,
    available: Mapping[str, Sequence[str]],
    canonical: Mapping[str, str],
    scenario_key: str = CLEAN,
) -> CaseWindow:
    """Clean-Window mit dem festen Klassenvektor `n_c(W)` (ohne Replacement)."""
    return _window(
        namespace=namespace,
        window_size=vector.window_size,
        scenario_key=scenario_key,
        repeat_id=repeat_id,
        pool=pool,
        split_hash=split_hash,
        available=available,
        counts=vector.as_dict(),
        canonical=canonical,
    )


def technical_cases(
    severity: SelectedSeverity,
    vector: ClassVector,
    repeat_id: int,
    *,
    split_hash: str,
    tune_by_class: Mapping[str, Sequence[str]],
    canonical: Mapping[str, str],
) -> tuple[CaseWindow, ...]:
    """Ein Base-Window je Repeat/Familie; L1–L3 mit identischen Samples (DEC-063)."""
    family = severity.shift.value
    cases = tuple(
        _window(
            namespace=SamplingNamespace.R_STABILITY,
            window_size=vector.window_size,
            scenario_key=level_key(family, level),
            repeat_id=repeat_id,
            pool=CalibrationPool.CAL_TUNE,
            split_hash=split_hash,
            available=tune_by_class,
            counts=vector.as_dict(),
            canonical=canonical,
            anchor_key=base_key(family),
            family=family,
            level=level,
            shift=severity.shift,
            shift_value=value,
        )
        for level, value in enumerate(severity.values, start=1)
    )
    if len({c.sample_ids for c in cases}) != 1 or len({c.base_window_id for c in cases}) != 1:
        raise IntegrityError(f"{family}: L1–L3 müssen dasselbe Base-Window verwenden")
    return cases


def _composition_cases(
    family: str,
    compositions: Sequence[Mapping[str, int]],
    vector: ClassVector,
    repeat_id: int,
    *,
    split_hash: str,
    available: Mapping[str, Sequence[str]],
    canonical: Mapping[str, str],
) -> tuple[CaseWindow, ...]:
    return tuple(
        _window(
            namespace=SamplingNamespace.R_STABILITY,
            window_size=vector.window_size,
            scenario_key=level_key(family, level),
            repeat_id=repeat_id,
            pool=CalibrationPool.CAL_TUNE,
            split_hash=split_hash,
            available=available,
            counts=counts,
            canonical=canonical,
            anchor_key=base_key(family),
            family=family,
            level=level,
        )
        for level, counts in enumerate(compositions, start=1)
    )


def class_compositions(
    plan: ClassShiftPlan, config: CalibrationProtocolConfig
) -> tuple[dict[str, int], ...]:
    return tuple(dict(zip(config.class_order, c, strict=True)) for c in plan.compositions)


def class_cases(
    plan: ClassShiftPlan,
    vector: ClassVector,
    repeat_id: int,
    config: CalibrationProtocolConfig,
    *,
    split_hash: str,
    tune_by_class: Mapping[str, Sequence[str]],
    canonical: Mapping[str, str],
) -> tuple[CaseWindow, ...]:
    """Strata = Diagnoseklasse; Zielanzahlen der eingefrorenen Kompositionen (Prefix)."""
    return _composition_cases(
        CLASS_FAMILY,
        class_compositions(plan, config),
        vector,
        repeat_id,
        split_hash=split_hash,
        available=tune_by_class,
        canonical=canonical,
    )


def subgroup_stratum(class_id: str, sex: str) -> str:
    return f"{class_id}|{sex}"


def subgroup_compositions(
    plan: SubgroupPlan, config: CalibrationProtocolConfig
) -> tuple[dict[str, int], ...]:
    if not plan.feasible or plan.source is None or plan.target is None:
        raise CalibrationProtocolError("ausgeschlossene Subgroup erzeugt keine Cases")
    source, target = plan.source, plan.target
    return tuple(
        {
            **{
                subgroup_stratum(c, source): n
                for c, n in zip(config.class_order, comp.source_counts, strict=True)
            },
            **{
                subgroup_stratum(c, target): n
                for c, n in zip(config.class_order, comp.target_counts, strict=True)
            },
        }
        for comp in plan.compositions
    )


def subgroup_cases(
    plan: SubgroupPlan,
    vector: ClassVector,
    repeat_id: int,
    config: CalibrationProtocolConfig,
    *,
    split_hash: str,
    strata: Mapping[str, Sequence[str]],
    canonical: Mapping[str, str],
) -> tuple[CaseWindow, ...]:
    """Strata = (Diagnoseklasse, `sex`); Zielanzahlen der eingefrorenen Kompositionen."""
    return _composition_cases(
        SUBGROUP_FAMILY,
        subgroup_compositions(plan, config),
        vector,
        repeat_id,
        split_hash=split_hash,
        available=strata,
        canonical=canonical,
    )


def require_feasible(
    compositions: Sequence[Mapping[str, int]], available: Mapping[str, Sequence[str]]
) -> None:
    """Feasibility vor jeder Case-Ausführung: Verfügbarkeit je Stratum, kein Replacement."""
    for counts in compositions:
        for stratum, needed in counts.items():
            if needed < 0 or needed > len(available.get(stratum, ())):
                raise CalibrationProtocolError(
                    f"Komposition nicht ausführbar: {stratum} benötigt {needed}"
                )

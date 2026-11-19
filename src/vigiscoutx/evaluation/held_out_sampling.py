"""Deterministisches HELD_OUT-Sampling, Structural Gate und Case-Plan (Protokoll v1.1, §3–4).

Es werden ausschließlich die bestehenden kanonischen Implementierungen verwendet:
`canonical_sample`, `sampling_context`, `sample_window`, `window_id`, `repeat_seed` und
`noise_seed`. Die zwei Evaluations-Namespaces sind ein evaluationslokaler `StrEnum` mit exakt den
Protokoll-Strings; `sampling_context()`/`repeat_seed()` lesen davon ausschließlich `.value`
(isolierte statische Typanpassung `_as_sampling_namespace`, `DEC-085`). Keine Änderung unter
`experiment/calibration/`, kein neues Hash- oder Seed-Verfahren.

Die Selektion verwendet nur IDs, Diagnoseklassen und Counts – keine Pixel, keine Modelloutputs,
keine Ergebnisse.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final, Literal, cast

from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, PositiveInt, Sha256Hex
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.calibration.config import (
    CalibrationProtocolConfig,
    SamplingNamespace,
    TechnicalShift,
    baseline_protocol_config,
)
from vigiscoutx.experiment.calibration.freeze_record import CalibrationFreezeRecord
from vigiscoutx.experiment.calibration.shifts import noise_seed
from vigiscoutx.experiment.calibration.structure import (
    CalibrationLesion,
    ClassVector,
    WindowNotConstructibleError,
    canonical_sample,
    repeat_seed,
    sample_window,
    sampling_context,
    window_id,
)
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256
from vigiscoutx.experiment.types import Partition


class EvaluationNamespace(StrEnum):
    """Evaluationslokale HELD_OUT-Namespaces (Protokoll §3.1); Calibration-Enum unverändert."""

    HELD_OUT_CLEAN = "P4-03W|HELD_OUT_CLEAN"
    HELD_OUT_SCENARIO = "P4-03W|HELD_OUT_SCENARIO"


CLEAN_KEY: Final = "CLEAN"
CLASS_FAMILY: Final = "CLASS"
FAMILIES: Final = (*(s.value for s in TechnicalShift), CLASS_FAMILY)
LEVELS: Final = (0, 1, 2, 3)


class HeldOutEvaluationStop(IntegrityError):
    """Terminaler STOP der HELD_OUT-Evaluation; kein Ersatz und keine Wiederholung."""


def _stop(condition: bool, message: str) -> None:
    if not condition:
        raise HeldOutEvaluationStop(message)


def _as_sampling_namespace(namespace: EvaluationNamespace) -> SamplingNamespace:
    """Isolierte statische Typanpassung: die bestehenden Funktionen verwenden nur `.value`."""
    return cast(SamplingNamespace, namespace)


def evaluation_context(
    namespace: EvaluationNamespace, window_size: int, scenario_key: str, repeat_id: int
) -> str:
    """`<namespace>|<W>|<scenario_key>|<repeat_id>` über das unveränderte `sampling_context`."""
    return sampling_context(_as_sampling_namespace(namespace), window_size, scenario_key, repeat_id)


def evaluation_seed(
    split_hash: str,
    namespace: EvaluationNamespace,
    window_size: int,
    scenario_key: str,
    repeat_id: int,
) -> int:
    """G1-Permutationsseed über das unveränderte `repeat_seed`."""
    return repeat_seed(
        split_hash, _as_sampling_namespace(namespace), window_size, scenario_key, repeat_id
    )


def base_key(family: str) -> str:
    return f"{family}_BASE"


def level_key(family: str, level: int) -> str:
    return f"{family}_L{level}"


# --- HELD_OUT-Strukturinventar ------------------------------------------------------------------


def held_out_samples_by_lesion(split: SplitManifest) -> dict[str, tuple[str, ...]]:
    """Sample-IDs je HELD_OUT-Läsion (sortiert), ausschließlich aus dem Split-Manifest."""
    samples: dict[str, list[str]] = {}
    for a in split.assignments:
        if a.partition is Partition.HELD_OUT:
            samples.setdefault(a.group_id, []).append(a.sample_id)
    return {lesion: tuple(sorted(ids)) for lesion, ids in sorted(samples.items())}


def held_out_canonical_samples(split: SplitManifest) -> dict[str, str]:
    """Kanonisches Bild je HELD_OUT-Läsion nach der bestehenden Hash-Regel (ohne Labels)."""
    config = baseline_protocol_config()
    split_hash = split_manifest_sha256(split)
    return {
        lesion: rank_canonical(lesion, ids, split_hash, config)
        for lesion, ids in held_out_samples_by_lesion(split).items()
    }


def rank_canonical(
    lesion_id: str, sample_ids: Sequence[str], split_hash: str, config: CalibrationProtocolConfig
) -> str:
    # `canonical_sample` benötigt nur die Sample-IDs; die Klasse ist für das Ranking irrelevant.
    lesion = CalibrationLesion(lesion_id=lesion_id, class_id="NV", sample_ids=tuple(sample_ids))
    return canonical_sample(lesion, split_hash, config)


# --- Case-Plan ----------------------------------------------------------------------------------


class HeldOutCase(FrozenModel):
    """Genau ein eindeutiger HELD_OUT-Case mit vollständiger Sampling-Provenienz."""

    case_id: Sha256Hex
    namespace: EvaluationNamespace
    scenario_key: Identifier
    repeat_id: PositiveInt
    family: Identifier | None
    level: NonNegativeInt | None
    shift: TechnicalShift | None
    shift_value: float | None
    context: str
    anchor_context: str
    window_id: Sha256Hex
    anchor_window_id: Sha256Hex
    lesion_ids: tuple[Identifier, ...]
    sample_ids: tuple[Identifier, ...]
    g1_seed: NonNegativeInt
    noise_seeds: tuple[NonNegativeInt, ...] | None


class HeldOutCasePlan(FrozenModel):
    """Vollständiger deterministischer Case-Plan; nach dem Structural Gate gehasht und fixiert."""

    schema_version: Literal[1]
    freeze_id: Sha256Hex
    split_manifest_sha256: Sha256Hex
    reference_window_id: Sha256Hex
    window_size: PositiveInt
    repeats: PositiveInt
    class_order: tuple[Identifier, ...]
    class_counts: tuple[NonNegativeInt, ...]
    class_compositions: tuple[tuple[NonNegativeInt, ...], ...]
    severities: tuple[tuple[TechnicalShift, tuple[float, ...]], ...]
    canonical_samples_sha256: Sha256Hex
    lesion_classes_sha256: Sha256Hex
    cases: tuple[HeldOutCase, ...]

    def plan_sha256(self) -> str:
        return canonical_sha256(self)

    def by_id(self) -> dict[str, HeldOutCase]:
        return {c.case_id: c for c in self.cases}


def frozen_class_compositions(
    record: CalibrationFreezeRecord, vector: ClassVector
) -> tuple[tuple[int, ...], ...]:
    """L1–L3 der Class-Familie aus `n_c(W)` und den eingefrorenen Transfers; Hash wie im Freeze."""
    order = vector.class_order
    shift = record.class_shift
    compositions = []
    for d in shift.levels:
        counts = vector.as_dict()
        counts[shift.source] -= d
        counts[shift.target] += d
        compositions.append(tuple(counts[c] for c in order))
    result = tuple(compositions)
    _stop(
        canonical_sha256(result) == shift.compositions_sha256,
        "Class-Kompositionen weichen vom Freeze ab",
    )
    return result


def _severities(record: CalibrationFreezeRecord) -> dict[TechnicalShift, tuple[float, ...]]:
    values = {s.shift: tuple(s.selected_values) for s in record.technical_severities}
    _stop(set(values) == set(TechnicalShift), "eingefrorene Severities unvollständig")
    return values


def _counts_for(
    family: str, level: int, vector: ClassVector, compositions: Sequence[Sequence[int]]
) -> dict[str, int]:
    if family == CLASS_FAMILY and level > 0:
        return dict(zip(vector.class_order, compositions[level - 1], strict=True))
    return vector.as_dict()


def _require_window(
    lesion_ids: Sequence[str],
    sample_ids: Sequence[str],
    lesion_class: Mapping[str, str],
    held_out_samples: frozenset[str],
    window_size: int,
    class_order: Sequence[str],
    context: str,
) -> None:
    _stop(
        len(lesion_ids) == len(set(lesion_ids)) == window_size,
        f"{context}: nicht exakt {window_size} unterschiedliche Läsionen",
    )
    _stop(len(set(sample_ids)) == window_size, f"{context}: Samples nicht eindeutig")
    _stop(set(sample_ids) <= held_out_samples, f"{context}: Sample außerhalb HELD_OUT (Leakage)")
    _stop(
        {lesion_class[i] for i in lesion_ids} == set(class_order),
        f"{context}: nicht alle sieben Klassen vorhanden",
    )


def build_case_plan(
    *,
    record: CalibrationFreezeRecord,
    freeze_id: str,
    split: SplitManifest,
    vector: ClassVector,
    lesion_classes: Mapping[str, str],
    reference_lesion_ids: Sequence[str],
    reference_sample_ids: Sequence[str],
    reference_window_id: str,
) -> HeldOutCasePlan:
    """Structural Gate (§3.4) und vollständiger Case-Plan (§4); jede Abweichung ist ein STOP.

    `lesion_classes` ordnet jeder HELD_OUT-Läsion die Diagnoseklasse ihres kanonischen Bildes zu.
    Es wird ausschließlich mit den eingefrorenen Zielanzahlen gesampelt; fehlende Kapazität
    führt zum STOP (keine kleineren Windows, keine geänderten Anteile, kein Ersatzsampling).
    """
    config = baseline_protocol_config()
    split_hash = split_manifest_sha256(split)
    _stop(split_hash == record.split_manifest_sha256, "Split-Manifest weicht vom Freeze ab")
    _stop(vector.window_size == record.window_size, "Window Size weicht vom Freeze ab")
    _stop(vector.valid and vector.class_order == config.class_order, "Klassenvektor ungültig")
    by_lesion = held_out_samples_by_lesion(split)
    _stop(set(lesion_classes) == set(by_lesion), "Klassen für exakt alle HELD_OUT-Läsionen nötig")
    _stop(set(lesion_classes.values()) <= set(config.class_order), "unbekannte Diagnoseklasse")
    held_out_samples = frozenset(s for ids in by_lesion.values() for s in ids)
    _stop(
        not set(reference_lesion_ids) & set(by_lesion)
        and not set(reference_sample_ids) & held_out_samples,
        "Reference und HELD_OUT sind nicht läsionsdisjunkt (Leakage)",
    )
    compositions = frozen_class_compositions(record, vector)
    _stop(
        all(sum(c) == vector.window_size and min(c) >= 1 for c in compositions),
        "Class-Komposition verletzt W oder enthält nicht alle sieben Klassen",
    )
    severities = _severities(record)
    canonical = held_out_canonical_samples(split)
    by_class: dict[str, list[str]] = {c: [] for c in config.class_order}
    for lesion_id in sorted(lesion_classes):
        by_class[lesion_classes[lesion_id]].append(lesion_id)
    needed = {
        c: max([vector.as_dict()[c], *(comp[i] for comp in compositions)])
        for i, c in enumerate(config.class_order)
    }
    short = sorted(c for c in config.class_order if needed[c] > len(by_class[c]))
    _stop(not short, f"Structural Gate: unzureichende HELD_OUT-Kapazität für {short}")

    size, repeats = record.window_size, record.final_r
    cases: list[HeldOutCase] = []

    def add(
        namespace: EvaluationNamespace,
        scenario_key: str,
        anchor_key: str,
        repeat_id: int,
        counts: Mapping[str, int],
        family: str | None,
        level: int | None,
    ) -> None:
        context = evaluation_context(namespace, size, scenario_key, repeat_id)
        anchor = evaluation_context(namespace, size, anchor_key, repeat_id)
        try:
            selection = sample_window(split_hash, anchor, by_class, counts, canonical)
        except WindowNotConstructibleError as exc:
            raise HeldOutEvaluationStop(f"Structural Gate: {context}: {exc}") from exc
        samples = tuple(sorted(selection.sample_ids))
        _require_window(
            selection.lesion_ids,
            samples,
            lesion_classes,
            held_out_samples,
            size,
            config.class_order,
            context,
        )
        technical = family is not None and family != CLASS_FAMILY and bool(level)
        shift = TechnicalShift(family) if technical and family is not None else None
        identifier = window_id(split_hash, context, samples)
        cases.append(
            HeldOutCase(
                case_id=identifier,
                namespace=namespace,
                scenario_key=scenario_key,
                repeat_id=repeat_id,
                family=family,
                level=level,
                shift=shift,
                shift_value=None
                if shift is None or level is None
                else severities[shift][level - 1],
                context=context,
                anchor_context=anchor,
                window_id=identifier,
                anchor_window_id=selection.window_id,
                lesion_ids=selection.lesion_ids,
                sample_ids=samples,
                g1_seed=evaluation_seed(split_hash, namespace, size, scenario_key, repeat_id),
                noise_seeds=(
                    tuple(
                        noise_seed(split_hash, config.noise_seed_namespace, repeat_id, s)
                        for s in samples
                    )
                    if shift is TechnicalShift.NOISE
                    else None
                ),
            )
        )

    for repeat_id in range(1, repeats + 1):
        add(
            EvaluationNamespace.HELD_OUT_CLEAN,
            CLEAN_KEY,
            CLEAN_KEY,
            repeat_id,
            vector.as_dict(),
            None,
            None,
        )
    for family in FAMILIES:
        for repeat_id in range(1, repeats + 1):
            for level in LEVELS:
                add(
                    EvaluationNamespace.HELD_OUT_SCENARIO,
                    level_key(family, level),
                    base_key(family),
                    repeat_id,
                    _counts_for(family, level, vector, compositions),
                    family,
                    level,
                )
    _verify_plan_structure(cases, repeats)
    lesion_class_items = sorted(lesion_classes.items())
    return HeldOutCasePlan(
        schema_version=1,
        freeze_id=freeze_id,
        split_manifest_sha256=split_hash,
        reference_window_id=reference_window_id,
        window_size=size,
        repeats=repeats,
        class_order=config.class_order,
        class_counts=vector.counts,
        class_compositions=compositions,
        severities=tuple((s, severities[s]) for s in TechnicalShift),
        canonical_samples_sha256=canonical_sha256(sorted(canonical.items())),
        lesion_classes_sha256=canonical_sha256(lesion_class_items),
        cases=tuple(cases),
    )


def _verify_plan_structure(cases: Sequence[HeldOutCase], repeats: int) -> None:
    """Exakte Matrix: R Clean-Windows, 5 Familien × (L0–L3) × R; technische Familien mit
    identischen Base-Samples über L0–L3; eindeutige Case-IDs."""
    ids = [c.case_id for c in cases]
    _stop(len(ids) == len(set(ids)), "Case-IDs sind nicht eindeutig")
    clean = [c for c in cases if c.namespace is EvaluationNamespace.HELD_OUT_CLEAN]
    _stop(len(clean) == repeats, "Anzahl Clean-Windows weicht ab")
    scenario = [c for c in cases if c.namespace is EvaluationNamespace.HELD_OUT_SCENARIO]
    _stop(len(scenario) == len(FAMILIES) * len(LEVELS) * repeats, "Szenario-Cases fehlen")
    for family in FAMILIES:
        for repeat_id in range(1, repeats + 1):
            block = [c for c in scenario if (c.family, c.repeat_id) == (family, repeat_id)]
            _stop(
                sorted(c.level for c in block if c.level is not None) == list(LEVELS),
                f"{family}/{repeat_id}: L0–L3 unvollständig",
            )
            _stop(
                len({c.anchor_context for c in block}) == 1,
                f"{family}/{repeat_id}: Levels teilen nicht denselben Anchor",
            )
            if family != CLASS_FAMILY:
                _stop(
                    len({c.sample_ids for c in block}) == 1,
                    f"{family}/{repeat_id}: L0–L3 nicht mit identischen Base-Samples",
                )


def verify_case_plan(plan: HeldOutCasePlan, expected_sha256: str) -> None:
    """Fixierter Plan unverändert (Hash) und strukturell vollständig."""
    _stop(plan.plan_sha256() == expected_sha256, "Case-Plan weicht vom fixierten Hash ab")
    _verify_plan_structure(plan.cases, plan.repeats)


def verify_held_out_case(case: HeldOutCase, split_hash: str, window_size: int) -> None:
    """Vor G1: Kontext, Sortierung, Größe, `window_id` und G1-Seed folgen exakt aus dem Case."""
    context = evaluation_context(case.namespace, window_size, case.scenario_key, case.repeat_id)
    _stop(case.context == context, f"Case-Kontext weicht ab: {case.context}")
    _stop(
        case.sample_ids == tuple(sorted(set(case.sample_ids)))
        and len(case.sample_ids) == window_size,
        f"{case.case_id}: Samples nicht eindeutig, sortiert und vollständig",
    )
    _stop(
        window_id(split_hash, context, case.sample_ids) == case.window_id == case.case_id,
        f"{case.case_id}: window_id weicht ab",
    )
    _stop(
        evaluation_seed(split_hash, case.namespace, window_size, case.scenario_key, case.repeat_id)
        == case.g1_seed,
        f"{case.case_id}: G1-Seed weicht ab",
    )

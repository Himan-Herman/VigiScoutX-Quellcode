"""Real-Template-Guard für den realen Calibration-Lauf (P4-03D, Protokollvorgabe `DEC-079`).

Ergänzt `template_methods` (Phase, exakt drei Calibration-Gaps, Signal-/Fusionsstruktur, G1 ohne σ)
um die nur für die reale Vorlage verbindlichen Werte: eingefrorene PolicyRefs, exakte
Evaluationsmetriken, freigegebene Template-Seeds, Template-Platzhalter `W = 128` und `R = 100`
sowie die Relevanzkennzahl. Wird ausschließlich im realen Pfad aufgerufen (CLI vor jedem
Modell- und Datensatzzugriff, `CalibrationRunner` nur bei `RealExecution`); synthetische
Testläufe mit technischen Testwerten bleiben unberührt.

Identität: Die reale Vorlage ist über ihren kanonischen SHA-256 (`canonical_sha256(RunConfig)`)
fest gepinnt (`REAL_TEMPLATE_SHA256`). Die Sollkonstante ist unabhängig von Vorlage und
Autorisierung; eine abweichende RunConfig scheitert daher auch dann, wenn eine dazu passende
`PreRunAuthorization` existiert. Eine neue reale Vorlage erfordert eine neue separate Freigabe
und eine Codeänderung dieser Konstante.

W und R sind hier Platzhalter, keine kalibrierten Werte; nach der Calibration ersetzt
`final_run_config()` ausschließlich W, δ_rel, Fusionsregel, R und σ aus dem Freeze-Record.
"""

from typing import Final

from calibration_runner.freeze_assembly import TemplateMethods, template_methods
from vigiscoutx.config.scientific_config import require_frozen_policy_refs
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.domain.types import PerformanceMetric
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import EvaluationMetric

REAL_TEMPLATE_EVALUATION_METRICS: Final = (
    EvaluationMetric.SENSITIVITY,
    EvaluationMetric.SPECIFICITY,
    EvaluationMetric.FALSE_ALARM_RATE,
    EvaluationMetric.PRECISION,
    EvaluationMetric.F1,
    EvaluationMetric.DETECTION_DELAY,
)
REAL_TEMPLATE_SEEDS: Final = SeedBundle(
    split_seed=0,
    model_development_seed=0,
    sampling_seed=0,
    shift_seed=0,
    statistical_seed=0,
    repeat_id=0,
)
TEMPLATE_WINDOW_SIZE: Final = 128
"""Platzhalter; W wird in der Calibration aus {128, 256} bestimmt (DEC-051)."""
TEMPLATE_REPETITIONS: Final = 100
"""Platzhalter; R wird in der Calibration aus {100, 200, 400} bestimmt (DEC-058)."""
REAL_TEMPLATE_SHA256: Final = "9fada8875e58c3bf0c495682c7a3136927d0d171ef2e3e1338e7ef85a53aff35"
"""Gepinnter kanonischer Hash von `P4-03_calibration_reproduction_template.yaml` (DEC-081;
`run_id = P4-03-calibration-real-2`, wissenschaftlich identisch zu DEC-079). Die historische
Vorlage `P4-03_calibration_template.yaml` (`966adc4c…`, verworfener Lauf) wird abgelehnt."""
REAL_RELEVANCE_METRIC: Final = PerformanceMetric.BALANCED_ACCURACY
REAL_CLEAN_QUANTILE: Final = 0.95


def validate_real_template(template: RunConfig) -> TemplateMethods:
    """Harte Prüfung der realen Vorlage vor jedem realen Datenzugriff; Abweichung = FAIL."""
    methods = template_methods(template)
    scientific = template.scientific
    require_frozen_policy_refs(scientific.risk_policy, scientific.recommendation_policy)
    checks: dict[str, tuple[object, object]] = {
        "evaluation_metrics": (template.evaluation_metrics, REAL_TEMPLATE_EVALUATION_METRICS),
        "seeds": (template.seeds, REAL_TEMPLATE_SEEDS),
        "monitoring.window_size": (scientific.monitoring.window_size, TEMPLATE_WINDOW_SIZE),
        "repetitions": (template.repetitions, TEMPLATE_REPETITIONS),
        "relevance.performance_metric": (
            scientific.relevance.performance_metric,
            REAL_RELEVANCE_METRIC,
        ),
        "relevance.clean_quantile": (scientific.relevance.clean_quantile, REAL_CLEAN_QUANTILE),
    }
    mismatches = sorted(name for name, (have, want) in checks.items() if have != want)
    if mismatches:
        raise IntegrityError(f"Reale Calibration-Vorlage weicht von der Freigabe ab: {mismatches}")
    actual = canonical_sha256(template)
    if actual != REAL_TEMPLATE_SHA256:
        raise IntegrityError(
            f"Reale Calibration-Vorlage ist nicht die freigegebene Vorlage: {actual} "
            f"≠ {REAL_TEMPLATE_SHA256}"
        )
    return methods

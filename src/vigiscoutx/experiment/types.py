"""Experimentbezogene Aufzählungen."""

from enum import StrEnum


class Partition(StrEnum):
    """Hauptpartitionen (DEC-017)."""

    MODEL_DEVELOPMENT = "MODEL_DEVELOPMENT"
    CALIBRATION = "CALIBRATION"
    HELD_OUT = "HELD_OUT"


class ModelDevelopmentRole(StrEnum):
    """Unterrollen ausschließlich innerhalb von MODEL_DEVELOPMENT."""

    TRAIN = "TRAIN"
    MODEL_VALIDATION = "MODEL_VALIDATION"


class Phase(StrEnum):
    DEVELOPMENT_CALIBRATION = "DEVELOPMENT_CALIBRATION"
    FROZEN = "FROZEN"
    HELD_OUT_EVALUATION = "HELD_OUT_EVALUATION"


class EvaluationMetric(StrEnum):
    SENSITIVITY = "SENSITIVITY"
    SPECIFICITY = "SPECIFICITY"
    FALSE_ALARM_RATE = "FALSE_ALARM_RATE"
    PRECISION = "PRECISION"
    F1 = "F1"
    AUROC = "AUROC"
    DETECTION_DELAY = "DETECTION_DELAY"

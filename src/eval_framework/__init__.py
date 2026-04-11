"""Main package initialization."""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import Settings, get_settings, configure_logging
from .types import (
    EvaluationMetric,
    QAPair,
    SystemOutput,
    EvaluationResult,
    EvaluationReport,
)

__all__ = [
    "Settings",
    "get_settings",
    "configure_logging",
    "EvaluationMetric",
    "QAPair",
    "SystemOutput",
    "EvaluationResult",
    "EvaluationReport",
]

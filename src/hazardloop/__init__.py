"""hazardloop — censoring-aware competing-risk survival analysis for LLM-agent trajectories.

Estimator + fail-closed policy interface + offline-replay decision-quality evaluation.
CPU-only, no GPU. See ``docs/CLAIMS.md`` for the CLAIM / NON-CLAIM boundary.
"""

from __future__ import annotations

from hazardloop.controller import Controller
from hazardloop.core.aalen_johansen import aalen_johansen, cif_at
from hazardloop.core.bootstrap import BootstrapCI, cluster_bootstrap_ci
from hazardloop.core.kaplan_meier import kaplan_meier, survival_at
from hazardloop.core.nelson_aalen import cumulative_hazard_at, hazard_at, nelson_aalen
from hazardloop.core.weibull_aft import weibull_aft
from hazardloop.estimate import fit_survival, has_observed_events
from hazardloop.policy import HazardThresholdPolicy
from hazardloop.replay import ReplayEvaluator, ReplayMetrics, ReplayReport, evaluate_policy
from hazardloop.types import (
    CIFResult,
    ControlDecision,
    EventModel,
    KMResult,
    NAResult,
    StepRecord,
    SurvivalRecord,
    SurvivalReport,
    TerminationMode,
    WeibullAFTResult,
)

__version__ = "0.1.0a3"

__all__ = [
    "BootstrapCI",
    "CIFResult",
    "ControlDecision",
    "Controller",
    "EventModel",
    "HazardThresholdPolicy",
    "KMResult",
    "NAResult",
    "ReplayEvaluator",
    "ReplayMetrics",
    "ReplayReport",
    "StepRecord",
    "SurvivalRecord",
    "SurvivalReport",
    "TerminationMode",
    "WeibullAFTResult",
    "__version__",
    "aalen_johansen",
    "cif_at",
    "cluster_bootstrap_ci",
    "cumulative_hazard_at",
    "evaluate_policy",
    "fit_survival",
    "has_observed_events",
    "hazard_at",
    "kaplan_meier",
    "nelson_aalen",
    "survival_at",
    "weibull_aft",
]

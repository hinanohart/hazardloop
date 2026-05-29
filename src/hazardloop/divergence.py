"""Divergence between a censoring-blind static logistic horizon and censoring-aware KM.

The METR-style time-horizon methodology fits a static logistic curve to *binary*
success/failure against task length, discarding censoring. hazardloop's Kaplan-Meier
treats unresolved-but-cut-off runs as right-censored. On the same data the two estimators
of "probability the run has failed by step t" can diverge; this module measures that
divergence.

This is a statement about two estimators on one dataset — it is **not** a claim that the
METR methodology is incorrect (R5). The logistic fit here is a deliberately simple stand-in
for the censoring-blind approach, not a reproduction of METR's full pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from hazardloop.core.kaplan_meier import kaplan_meier, survival_at
from hazardloop.types import EventModel, FloatArray, SurvivalRecord


@dataclass(frozen=True)
class DivergenceResult:
    times: FloatArray
    logistic_failure_prob: FloatArray  # censoring-blind: 1 - sigmoid(a + b·t)
    km_failure_prob: FloatArray  # censoring-aware: 1 - S(t)
    mean_abs_divergence: float
    max_abs_divergence: float
    logistic_a: float
    logistic_b: float


def _neg_loglik(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    z = params[0] + params[1] * x
    # numerically stable Bernoulli log-likelihood with p = sigmoid(z)
    log_p = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0, z)
    ll = float(np.sum(y * log_p + (1.0 - y) * log_1mp))
    return -ll if np.isfinite(ll) else 1e300


def fit_logistic_success(durations: np.ndarray, success: np.ndarray) -> tuple[float, float]:
    """MLE of P(success | t) = sigmoid(a + b·t). Returns (a, b)."""
    x = np.asarray(durations, dtype=np.float64)
    y = np.asarray(success, dtype=np.float64)
    if x.size == 0:
        raise ValueError("fit_logistic_success requires at least one observation")
    # centre/scale x for conditioning, then map coefficients back
    mu, sd = float(np.mean(x)), float(np.std(x)) or 1.0
    xs = (x - mu) / sd
    res = minimize(_neg_loglik, np.array([0.0, 0.0]), args=(xs, y), method="BFGS")
    a_s, b_s = res.x
    b = b_s / sd
    a = a_s - b * mu
    return float(a), float(b)


def logistic_vs_km_divergence(
    records: Sequence[SurvivalRecord], event_model: EventModel
) -> DivergenceResult:
    """Compare the censoring-blind logistic failure CDF with the KM failure CDF (1 − S)."""
    if len(records) == 0:
        raise ValueError("logistic_vs_km_divergence requires at least one record")
    durations = np.asarray([r.duration for r in records], dtype=np.float64)
    # success = NOT an event under the model (failure-as-event mode-A -> success = censored)
    success = np.asarray(
        [0.0 if event_model.is_event(r.terminal_mode) else 1.0 for r in records], dtype=np.float64
    )
    a, b = fit_logistic_success(durations, success)

    km = kaplan_meier(records, event_model)
    times = np.unique(durations)
    z = a + b * times
    logistic_fail = 1.0 - 1.0 / (1.0 + np.exp(-z))  # 1 - sigmoid
    km_fail = np.array([1.0 - survival_at(km, float(t)) for t in times], dtype=np.float64)
    diff = np.abs(logistic_fail - km_fail)

    return DivergenceResult(
        times=times,
        logistic_failure_prob=logistic_fail.astype(np.float64),
        km_failure_prob=km_fail,
        mean_abs_divergence=float(np.mean(diff)),
        max_abs_divergence=float(np.max(diff)),
        logistic_a=a,
        logistic_b=b,
    )

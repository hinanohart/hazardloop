"""Weibull fit by maximum likelihood with right-censoring.

Parametrisation: ``S(t) = exp(-(t/λ)^β)`` with shape ``β`` and scale ``λ``.
The shape carries the operational reading: ``β < 1`` early-mortality (risk falls with
elapsed steps), ``β > 1`` wear-out (risk rises). This is the intercept-only accelerated
failure-time (AFT) baseline; covariate AFT (``λ = exp(x'γ)``) is deferred to v0.2.

Log-likelihood (events E, censored C):
    ℓ = Σ_E [log β - β·logλ + (β-1)·log T - (T/λ)^β]  +  Σ_C [-(T/λ)^β]

Optimised over (logβ, logλ) so the search is unconstrained and positivity is automatic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from hazardloop.types import EventModel, SurvivalRecord, WeibullAFTResult


def _neg_loglik(params: np.ndarray, t: np.ndarray, event: np.ndarray) -> float:
    log_beta, log_lambda = params
    beta = np.exp(log_beta)
    lam = np.exp(log_lambda)
    z = (t / lam) ** beta
    # event contribution: log β - β logλ + (β-1) log t - (t/λ)^β
    ll_event = np.log(beta) - beta * log_lambda + (beta - 1.0) * np.log(t) - z
    ll = np.where(event, ll_event, -z)
    total = float(np.sum(ll))
    if not np.isfinite(total):
        return 1e300
    return -total


def weibull_aft(records: Sequence[SurvivalRecord], event_model: EventModel) -> WeibullAFTResult:
    """Fit a baseline Weibull by censored MLE.

    Requires strictly positive durations (the Weibull density is undefined at t=0 for
    β<1); callers working on a step axis should ensure runs have at least one step.
    """
    durations = np.asarray([r.duration for r in records], dtype=np.float64)
    event = np.asarray([event_model.is_event(r.terminal_mode) for r in records], dtype=bool)

    if durations.size == 0:
        raise ValueError("weibull_aft requires at least one record")
    if np.any(durations <= 0.0):
        raise ValueError("weibull_aft requires strictly positive durations (t > 0)")
    n_events = int(event.sum())
    if n_events == 0:
        raise ValueError("weibull_aft requires at least one observed event (all censored)")
    # With zero spread in event times the MLE shape diverges (β→∞); refuse rather than
    # return a meaningless ~1e15 silently.
    if np.unique(durations[event]).size < 2:
        raise ValueError(
            "weibull_aft requires >= 2 distinct event times; the shape parameter is "
            "undefined (diverges) when all events share one duration"
        )

    # Moment-ish initialisation: β≈1, λ≈mean event time.
    lam0 = float(np.mean(durations[event])) if n_events > 0 else float(np.mean(durations))
    x0 = np.array([0.0, np.log(max(lam0, 1e-6))], dtype=np.float64)

    res = minimize(
        _neg_loglik,
        x0,
        args=(durations, event),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10_000},
    )
    log_beta, log_lambda = res.x
    beta = float(np.exp(log_beta))
    lam = float(np.exp(log_lambda))

    return WeibullAFTResult(
        shape=beta,
        scale=lam,
        loglik=float(-res.fun),
        n_events=n_events,
        n_censored=int(durations.size - n_events),
    )


def weibull_survival_at(result: WeibullAFTResult, t: float) -> float:
    """S(t) = exp(-(t/λ)^β) for the fitted Weibull."""
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    return float(np.exp(-((t / result.scale) ** result.shape)))

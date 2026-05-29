"""Kaplan-Meier product-limit estimator with Greenwood variance and a complementary
log-log (cloglog) confidence band.

The cloglog band is used rather than the naive ``S ± z·se`` band because the latter can
leave [0, 1] near the tails; cloglog is transform-respecting and asymmetric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from hazardloop.core._risksets import RiskTable, build_risk_table, step_lookup
from hazardloop.types import EventModel, KMResult, SurvivalRecord


def kaplan_meier_from_table(table: RiskTable, alpha: float = 0.05) -> KMResult:
    """Kaplan-Meier estimate from a prebuilt risk table.

    ``alpha`` is the two-sided significance level (0.05 → 95% band).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    times = table.times
    if times.size == 0:
        z = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int64)
        return KMResult(
            times=times,
            survival=z,
            var_greenwood=z,
            ci_lower=z,
            ci_upper=z,
            n_at_risk=zi,
            n_events=zi,
            alpha=alpha,
        )

    y = table.n_at_risk.astype(np.float64)
    d = table.n_events.astype(np.float64)

    # product-limit survival
    factors = 1.0 - d / y
    survival = np.cumprod(factors).astype(np.float64)

    # Greenwood variance: Var(S(t)) = S(t)^2 * sum_{k<=j} d_k / (Y_k (Y_k - d_k))
    denom = y * (y - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        increments = np.where(denom > 0.0, d / denom, 0.0)
    cumulative = np.cumsum(increments)
    var_greenwood = survival**2 * cumulative

    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower, ci_upper = _cloglog_band(survival, var_greenwood, z_crit)

    return KMResult(
        times=times,
        survival=survival,
        var_greenwood=var_greenwood,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_at_risk=table.n_at_risk,
        n_events=table.n_events,
        alpha=alpha,
    )


def _cloglog_band(
    survival: np.ndarray, var: np.ndarray, z_crit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Complementary log-log confidence band for the survival function.

    With phi = log(-log S), se(phi) = sqrt(Var S) / (S |log S|), the band on phi maps back
    to  S_lower = S**exp(+z·se_phi),  S_upper = S**exp(-z·se_phi).  S in {0, 1} (where the
    transform is undefined) collapses to a degenerate point interval.
    """
    s = survival
    log_s = np.log(np.clip(s, a_min=1e-300, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        se_phi = np.sqrt(var) / np.abs(s * log_s)
    interior = (s > 0.0) & (s < 1.0) & np.isfinite(se_phi)
    lower = np.where(interior, s ** np.exp(+z_crit * se_phi), s)
    upper = np.where(interior, s ** np.exp(-z_crit * se_phi), s)
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def kaplan_meier(
    records: Sequence[SurvivalRecord], event_model: EventModel, alpha: float = 0.05
) -> KMResult:
    """Kaplan-Meier estimate directly from survival records under an event model."""
    return kaplan_meier_from_table(build_risk_table(records, event_model), alpha)


def survival_at(result: KMResult, t: float) -> float:
    """Right-continuous survival probability S(t). Returns 1.0 before the first event."""
    return step_lookup(result.times, result.survival, t, baseline=1.0)

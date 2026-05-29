"""Aalen-Johansen competing-risk cumulative incidence function (the core differentiator).

For competing causes the naive ``1 - cause-specific KM`` systematically *over*-estimates a
cause's incidence, because it treats the other causes as censoring rather than as events
that preclude the cause of interest (Putter, Fiocco & Geskus, Stat. Med. 2007). The
Aalen-Johansen estimator weights each cause-specific increment by the overall survival
just before the event time:

    CIF_c(t_j) = Σ_{k≤j}  S(t_{k-1}) · d_{c,k} / Y_k

where ``S`` is the all-cause Kaplan-Meier survival and ``S(t_0^-) = 1``. This yields the
additivity identity   Σ_c CIF_c(t) = 1 - S(t)   which is asserted as a numeric invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hazardloop.core._risksets import RiskTable, build_risk_table, step_lookup
from hazardloop.types import CIFResult, EventModel, FloatArray, SurvivalRecord


def aalen_johansen_from_table(table: RiskTable) -> CIFResult:
    times = table.times
    causes = table.causes
    if times.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return CIFResult(
            times=times, cif_by_cause={c: z.copy() for c in causes}, overall_survival=z
        )

    y = table.n_at_risk.astype(np.float64)
    d = table.n_events.astype(np.float64)

    # All-cause KM survival and its left-limit S(t_{k-1}); S(t_0^-) = 1.
    overall_survival = np.cumprod(1.0 - d / y).astype(np.float64)
    s_left = np.empty_like(overall_survival)
    s_left[0] = 1.0
    s_left[1:] = overall_survival[:-1]

    cif_by_cause: dict[str, FloatArray] = {}
    for c in causes:
        d_c = table.events_by_cause[c].astype(np.float64)
        cif_by_cause[c] = np.cumsum(s_left * d_c / y)

    return CIFResult(times=times, cif_by_cause=cif_by_cause, overall_survival=overall_survival)


def aalen_johansen(records: Sequence[SurvivalRecord], event_model: EventModel) -> CIFResult:
    return aalen_johansen_from_table(build_risk_table(records, event_model))


def cif_at(result: CIFResult, cause: str, t: float) -> float:
    """Right-continuous CIF for ``cause`` at time t. Returns 0.0 before the first event."""
    if cause not in result.cif_by_cause:
        raise KeyError(f"unknown cause {cause!r}; known causes: {tuple(result.cif_by_cause)}")
    return step_lookup(result.times, result.cif_by_cause[cause], t, baseline=0.0)

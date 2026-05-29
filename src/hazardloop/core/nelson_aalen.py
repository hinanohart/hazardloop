"""Nelson-Aalen cumulative hazard estimator with tie-corrected (Aalen) variance.

The per-time hazard increment ``ĥ_j = d_j / Y_j`` is the primary signal a control policy
reads (instantaneous risk at a step); the cumulative hazard ``H(t) = Σ_{k≤j} d_k/Y_k`` is
its running integral.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hazardloop.core._risksets import RiskTable, build_risk_table, step_lookup
from hazardloop.types import EventModel, NAResult, SurvivalRecord


def nelson_aalen_from_table(table: RiskTable) -> NAResult:
    times = table.times
    if times.size == 0:
        z = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int64)
        return NAResult(
            times=times,
            cumulative_hazard=z,
            variance=z,
            hazard_increment=z,
            n_at_risk=zi,
            n_events=zi,
        )

    y = table.n_at_risk.astype(np.float64)
    d = table.n_events.astype(np.float64)

    hazard_increment = d / y
    cumulative_hazard = np.cumsum(hazard_increment)

    # Tie-corrected Aalen variance:  Var(H(t)) = Σ_{k≤j} (Y_k - d_k) d_k / Y_k^3
    var_increment = (y - d) * d / y**3
    variance = np.cumsum(var_increment)

    return NAResult(
        times=times,
        cumulative_hazard=cumulative_hazard,
        variance=variance,
        hazard_increment=hazard_increment,
        n_at_risk=table.n_at_risk,
        n_events=table.n_events,
    )


def nelson_aalen(records: Sequence[SurvivalRecord], event_model: EventModel) -> NAResult:
    return nelson_aalen_from_table(build_risk_table(records, event_model))


def cumulative_hazard_at(result: NAResult, t: float) -> float:
    """Right-continuous cumulative hazard H(t). Returns 0.0 before the first event."""
    return step_lookup(result.times, result.cumulative_hazard, t, baseline=0.0)


def hazard_at(result: NAResult, t: float) -> float:
    """Instantaneous hazard increment at the most recent event time <= t (0 before first)."""
    return step_lookup(result.times, result.hazard_increment, t, baseline=0.0)

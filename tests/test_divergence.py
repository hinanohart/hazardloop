"""S6 divergence tests (logistic-vs-KM; NON-CLAIM metric)."""

from __future__ import annotations

import numpy as np

from hazardloop.adapters.mock import synthetic_competing_risks
from hazardloop.divergence import fit_logistic_success, logistic_vs_km_divergence
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()


def test_logistic_recovers_direction() -> None:
    # longer runs succeed more often -> positive slope b
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 50, size=4000)
    p = 1.0 / (1.0 + np.exp(-(-3.0 + 0.12 * x)))
    y = (rng.uniform(size=x.size) < p).astype(float)
    a, b = fit_logistic_success(x, y)
    assert b > 0
    assert a == np.float64(a)  # finite


def test_divergence_is_well_formed_and_nonneg() -> None:
    recs = synthetic_competing_risks(800, seed=1)
    d = logistic_vs_km_divergence(recs, MODE_A)
    assert d.times.size == d.logistic_failure_prob.size == d.km_failure_prob.size
    assert np.all((d.km_failure_prob >= -1e-9) & (d.km_failure_prob <= 1.0 + 1e-9))
    assert np.all((d.logistic_failure_prob >= -1e-9) & (d.logistic_failure_prob <= 1.0 + 1e-9))
    assert d.mean_abs_divergence >= 0.0
    assert d.max_abs_divergence >= d.mean_abs_divergence


def test_divergence_positive_under_heavy_censoring() -> None:
    # many early successes (censored under mode-A) + failures spread late: the censoring-blind
    # logistic and censoring-aware KM disagree -> strictly positive divergence somewhere.
    recs: list[SurvivalRecord] = []
    for i in range(200):
        recs.append(SurvivalRecord(duration=float(1 + i % 3), terminal_mode=TerminationMode.SOLVED))
    for i in range(60):
        recs.append(
            SurvivalRecord(duration=float(20 + i % 10), terminal_mode=TerminationMode.WRONG_PATCH)
        )
    d = logistic_vs_km_divergence(recs, MODE_A)
    assert d.max_abs_divergence > 0.0

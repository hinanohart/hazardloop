"""S2 Weibull AFT tests: recover known parameters by MLE, censoring, degenerate guards."""

from __future__ import annotations

import numpy as np
import pytest

from hazardloop.core.weibull_aft import weibull_aft, weibull_survival_at
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()


def _weibull_records(
    n: int, shape: float, scale: float, seed: int, censor_above: float | None = None
) -> list[SurvivalRecord]:
    rng = np.random.default_rng(seed)
    raw = rng.weibull(shape, size=n) * scale  # shape a, scale 1 -> scaled
    recs: list[SurvivalRecord] = []
    for t in raw:
        if censor_above is not None and t > censor_above:
            recs.append(SurvivalRecord(duration=censor_above, terminal_mode=TerminationMode.SOLVED))
        else:
            recs.append(
                SurvivalRecord(duration=float(t), terminal_mode=TerminationMode.WRONG_PATCH)
            )
    return recs


def test_recovers_shape_below_one_early_mortality() -> None:
    recs = _weibull_records(8000, shape=0.7, scale=10.0, seed=1)
    fit = weibull_aft(recs, MODE_A)
    assert fit.shape == pytest.approx(0.7, rel=0.08)
    assert fit.scale == pytest.approx(10.0, rel=0.08)
    assert fit.shape < 1.0  # early-mortality regime correctly identified


def test_recovers_shape_above_one_wear_out() -> None:
    recs = _weibull_records(8000, shape=1.8, scale=5.0, seed=2)
    fit = weibull_aft(recs, MODE_A)
    assert fit.shape == pytest.approx(1.8, rel=0.08)
    assert fit.scale == pytest.approx(5.0, rel=0.08)
    assert fit.shape > 1.0  # wear-out regime


def test_censoring_is_handled() -> None:
    recs = _weibull_records(8000, shape=1.3, scale=8.0, seed=3, censor_above=10.0)
    fit = weibull_aft(recs, MODE_A)
    assert fit.n_censored > 0
    assert fit.shape == pytest.approx(1.3, rel=0.12)


def test_survival_function_monotone() -> None:
    recs = _weibull_records(2000, shape=1.5, scale=6.0, seed=4)
    fit = weibull_aft(recs, MODE_A)
    ts = np.linspace(0.0, 20.0, 50)
    s = np.array([weibull_survival_at(fit, float(t)) for t in ts])
    assert s[0] == pytest.approx(1.0)
    assert np.all(np.diff(s) <= 1e-12)


def test_degenerate_guards() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        weibull_aft([], MODE_A)
    with pytest.raises(ValueError, match="positive durations"):
        weibull_aft(
            [SurvivalRecord(duration=0.0, terminal_mode=TerminationMode.WRONG_PATCH)], MODE_A
        )
    with pytest.raises(ValueError, match="all censored"):
        weibull_aft([SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.SOLVED)], MODE_A)

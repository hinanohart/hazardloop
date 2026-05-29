"""S2 Kaplan-Meier tests: hand-computed golden, monotonicity, Greenwood, lifelines."""

from __future__ import annotations

import numpy as np
import pytest

from hazardloop.core.kaplan_meier import kaplan_meier, survival_at
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

from ._helpers import GOLDEN, GOLDEN_EVENT_TIMES, GOLDEN_KM_SURVIVAL

MODE_A = EventModel.failure_as_event()


def test_golden_survival() -> None:
    km = kaplan_meier(GOLDEN, MODE_A)
    np.testing.assert_allclose(km.times, GOLDEN_EVENT_TIMES)
    np.testing.assert_allclose(km.survival, GOLDEN_KM_SURVIVAL, atol=1e-12)


def test_golden_greenwood_variance() -> None:
    km = kaplan_meier(GOLDEN, MODE_A)
    # Var at t=2: 0.6^2 * (1/(5*4) + 1/(4*3)) = 0.36 * 0.133333... = 0.048
    assert km.var_greenwood[1] == pytest.approx(0.048, abs=1e-9)
    # At t=4, S=0 so Greenwood variance is 0 (S^2 factor).
    assert km.var_greenwood[2] == pytest.approx(0.0, abs=1e-12)


def test_survival_is_monotone_non_increasing() -> None:
    rng = np.random.default_rng(7)
    recs = [
        SurvivalRecord(
            duration=float(rng.integers(1, 30)),
            terminal_mode=rng.choice(
                [TerminationMode.WRONG_PATCH, TerminationMode.SOLVED, TerminationMode.TOOL_ERROR]
            ),
        )
        for _ in range(200)
    ]
    km = kaplan_meier(recs, MODE_A)
    assert np.all(np.diff(km.survival) <= 1e-12)
    assert np.all((km.survival >= -1e-12) & (km.survival <= 1.0 + 1e-12))


def test_cloglog_band_within_unit_interval() -> None:
    km = kaplan_meier(GOLDEN, MODE_A)
    assert np.all(km.ci_lower >= 0.0) and np.all(km.ci_lower <= 1.0)
    assert np.all(km.ci_upper >= 0.0) and np.all(km.ci_upper <= 1.0)
    # band brackets the point estimate where the transform is defined (interior S)
    interior = (km.survival > 0.0) & (km.survival < 1.0)
    assert np.all(km.ci_lower[interior] <= km.survival[interior] + 1e-12)
    assert np.all(km.ci_upper[interior] >= km.survival[interior] - 1e-12)


def test_survival_at_step_lookup() -> None:
    km = kaplan_meier(GOLDEN, MODE_A)
    assert survival_at(km, 0.5) == 1.0  # before first event
    assert survival_at(km, 1.0) == pytest.approx(0.8)
    assert survival_at(km, 1.9) == pytest.approx(0.8)  # right-continuous
    assert survival_at(km, 2.0) == pytest.approx(0.6)
    assert survival_at(km, 100.0) == pytest.approx(0.0)


def test_lifelines_cross_check() -> None:
    lifelines = pytest.importorskip("lifelines")
    durations = [r.duration for r in GOLDEN]
    observed = [MODE_A.is_event(r.terminal_mode) for r in GOLDEN]
    kmf = lifelines.KaplanMeierFitter().fit(durations, event_observed=observed)
    km = kaplan_meier(GOLDEN, MODE_A)
    ours = km.survival
    theirs = kmf.survival_function_at_times(km.times).to_numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-9)

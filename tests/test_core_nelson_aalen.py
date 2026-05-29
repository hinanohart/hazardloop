"""S2 Nelson-Aalen tests: golden cumulative hazard, hazard increments, lifelines."""

from __future__ import annotations

import numpy as np
import pytest

from hazardloop.core.nelson_aalen import cumulative_hazard_at, hazard_at, nelson_aalen
from hazardloop.types import EventModel

from ._helpers import GOLDEN, GOLDEN_NA_CUMHAZ

MODE_A = EventModel.failure_as_event()


def test_golden_cumulative_hazard() -> None:
    na = nelson_aalen(GOLDEN, MODE_A)
    np.testing.assert_allclose(na.cumulative_hazard, GOLDEN_NA_CUMHAZ, atol=1e-12)


def test_golden_hazard_increments() -> None:
    na = nelson_aalen(GOLDEN, MODE_A)
    np.testing.assert_allclose(na.hazard_increment, [0.2, 0.25, 1.0], atol=1e-12)


def test_tie_corrected_variance() -> None:
    na = nelson_aalen(GOLDEN, MODE_A)
    # (Y-d)d/Y^3: t1 4/125=0.032, t2 3/64=0.046875, t4 0 -> cumsum
    np.testing.assert_allclose(na.variance, [0.032, 0.078875, 0.078875], atol=1e-12)


def test_cumulative_hazard_is_non_decreasing() -> None:
    na = nelson_aalen(GOLDEN, MODE_A)
    assert np.all(np.diff(na.cumulative_hazard) >= -1e-12)


def test_hazard_step_lookup() -> None:
    na = nelson_aalen(GOLDEN, MODE_A)
    assert cumulative_hazard_at(na, 0.0) == 0.0
    assert cumulative_hazard_at(na, 1.0) == pytest.approx(0.2)
    assert cumulative_hazard_at(na, 3.0) == pytest.approx(0.45)
    assert hazard_at(na, 2.5) == pytest.approx(0.25)


def test_lifelines_cross_check() -> None:
    lifelines = pytest.importorskip("lifelines")
    durations = [r.duration for r in GOLDEN]
    observed = [MODE_A.is_event(r.terminal_mode) for r in GOLDEN]
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False).fit(
        durations, event_observed=observed
    )
    na = nelson_aalen(GOLDEN, MODE_A)
    theirs = naf.cumulative_hazard_at_times(na.times).to_numpy()
    np.testing.assert_allclose(na.cumulative_hazard, theirs, atol=1e-9)

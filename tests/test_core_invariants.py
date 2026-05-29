"""S2 invariants: degenerate value-correctness (BP5) and hypothesis property tests.

BP5 requires degenerate cases to assert *values* (S≡1, CIF≡0), not merely non-crash, and
empty input to raise explicitly rather than silently return a degenerate object.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hazardloop.core._risksets import build_risk_table
from hazardloop.core.aalen_johansen import aalen_johansen, cif_at
from hazardloop.core.kaplan_meier import kaplan_meier, survival_at
from hazardloop.core.nelson_aalen import cumulative_hazard_at, nelson_aalen
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()
_MODES = [
    TerminationMode.WRONG_PATCH,
    TerminationMode.TOOL_ERROR,
    TerminationMode.INFINITE_LOOP,
    TerminationMode.BUDGET_EXHAUSTED,
    TerminationMode.SOLVED,
    TerminationMode.TIMEOUT,
]


# --- BP5 degenerate value-correctness -----------------------------------------------------
def test_all_censored_gives_unit_survival_and_zero_cif() -> None:
    recs = [
        SurvivalRecord(duration=float(t), terminal_mode=TerminationMode.SOLVED) for t in range(1, 6)
    ]
    km = kaplan_meier(recs, MODE_A)
    na = nelson_aalen(recs, MODE_A)
    cif = aalen_johansen(recs, MODE_A)
    assert km.times.size == 0  # no event steps
    for q in [0.0, 1.0, 3.5, 99.0]:
        assert survival_at(km, q) == 1.0
        assert cumulative_hazard_at(na, q) == 0.0
        for cause in MODE_A.causes:
            assert cif_at(cif, cause, q) == 0.0


def test_single_event_single_cause_cif_equals_one_minus_km() -> None:
    recs = [
        SurvivalRecord(duration=1.0, terminal_mode=TerminationMode.WRONG_PATCH),
        SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.SOLVED),
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.WRONG_PATCH),
    ]
    km = kaplan_meier(recs, MODE_A)
    cif = aalen_johansen(recs, MODE_A)
    np.testing.assert_allclose(cif.cif_by_cause["wrong_patch"], 1.0 - km.survival, atol=1e-12)
    # every other cause is identically zero
    for cause in MODE_A.causes:
        if cause != "wrong_patch":
            np.testing.assert_allclose(
                cif.cif_by_cause[cause], np.zeros_like(km.survival), atol=1e-15
            )


def test_empty_input_raises_not_silently_degenerate() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        build_risk_table([], MODE_A)
    with pytest.raises(ValueError):
        kaplan_meier([], MODE_A)
    with pytest.raises(ValueError):
        aalen_johansen([], MODE_A)


# --- hypothesis property tests (non-vacuous: real random data, real assertions) -----------
_records_strategy = st.lists(
    st.tuples(st.integers(min_value=1, max_value=50), st.sampled_from(_MODES)),
    min_size=1,
    max_size=120,
).map(lambda pairs: [SurvivalRecord(duration=float(d), terminal_mode=m) for d, m in pairs])


@settings(max_examples=200, deadline=None)
@given(records=_records_strategy)
def test_property_km_monotone_and_bounded(records: list[SurvivalRecord]) -> None:
    km = kaplan_meier(records, MODE_A)
    assert np.all(np.diff(km.survival) <= 1e-12)
    assert np.all((km.survival >= -1e-12) & (km.survival <= 1.0 + 1e-12))


@settings(max_examples=200, deadline=None)
@given(records=_records_strategy)
def test_property_cif_additivity(records: list[SurvivalRecord]) -> None:
    cif = aalen_johansen(records, MODE_A)
    np.testing.assert_allclose(cif.total_cif(), 1.0 - cif.overall_survival, atol=1e-9)
    for curve in cif.cif_by_cause.values():
        assert np.all(np.diff(curve) >= -1e-12)
    assert np.all(cif.total_cif() <= 1.0 + 1e-9)


@settings(max_examples=200, deadline=None)
@given(records=_records_strategy)
def test_property_na_non_decreasing(records: list[SurvivalRecord]) -> None:
    na = nelson_aalen(records, MODE_A)
    assert np.all(np.diff(na.cumulative_hazard) >= -1e-12)
    assert np.all(na.hazard_increment >= -1e-12)


@settings(max_examples=100, deadline=None)
@given(records=_records_strategy)
def test_property_determinism(records: list[SurvivalRecord]) -> None:
    a = kaplan_meier(records, MODE_A).survival
    b = kaplan_meier(records, MODE_A).survival
    np.testing.assert_array_equal(a, b)

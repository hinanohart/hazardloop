"""S2 Aalen-Johansen CIF tests: golden, additivity invariant, KM reduction, lifelines.

These are the moat's correctness tests — they assert *value* correctness (not merely
non-crash), including the additivity identity and the two-state reduction to KM.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazardloop.core.aalen_johansen import aalen_johansen, cif_at
from hazardloop.core.kaplan_meier import kaplan_meier
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

from ._helpers import GOLDEN, GOLDEN_CIF_TOOL, GOLDEN_CIF_WRONG

MODE_A = EventModel.failure_as_event()


def test_golden_cif_by_cause() -> None:
    cif = aalen_johansen(GOLDEN, MODE_A)
    np.testing.assert_allclose(cif.cif_by_cause["wrong_patch"], GOLDEN_CIF_WRONG, atol=1e-12)
    np.testing.assert_allclose(cif.cif_by_cause["tool_error"], GOLDEN_CIF_TOOL, atol=1e-12)
    # causes with no events stay identically zero
    np.testing.assert_allclose(cif.cif_by_cause["infinite_loop"], [0.0, 0.0, 0.0], atol=1e-15)


def test_additivity_invariant_sum_cif_equals_one_minus_survival() -> None:
    cif = aalen_johansen(GOLDEN, MODE_A)
    total = cif.total_cif()
    np.testing.assert_allclose(total, 1.0 - cif.overall_survival, atol=1e-12)
    # and the overall survival equals the standalone KM
    km = kaplan_meier(GOLDEN, MODE_A)
    np.testing.assert_allclose(cif.overall_survival, km.survival, atol=1e-12)


def test_each_cif_is_non_decreasing_and_bounded() -> None:
    rng = np.random.default_rng(11)
    modes = [
        TerminationMode.WRONG_PATCH,
        TerminationMode.TOOL_ERROR,
        TerminationMode.INFINITE_LOOP,
        TerminationMode.SOLVED,
    ]
    recs = [
        SurvivalRecord(duration=float(rng.integers(1, 40)), terminal_mode=rng.choice(modes))
        for _ in range(300)
    ]
    cif = aalen_johansen(recs, MODE_A)
    for curve in cif.cif_by_cause.values():
        assert np.all(np.diff(curve) >= -1e-12)
        assert np.all((curve >= -1e-12) & (curve <= 1.0 + 1e-12))
    assert np.all(cif.total_cif() <= 1.0 + 1e-12)


def test_two_state_reduces_to_kaplan_meier() -> None:
    # single event cause -> AJ CIF for that cause equals 1 - KM survival exactly.
    recs = [
        SurvivalRecord(duration=1.0, terminal_mode=TerminationMode.UNLABELED),
        SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.SOLVED),
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.UNLABELED),
        SurvivalRecord(duration=4.0, terminal_mode=TerminationMode.UNLABELED),
    ]
    cif = aalen_johansen(recs, MODE_A)
    km = kaplan_meier(recs, MODE_A)
    np.testing.assert_allclose(cif.cif_by_cause["unlabeled"], 1.0 - km.survival, atol=1e-12)


def test_naive_one_minus_km_overestimates_vs_aalen_johansen() -> None:
    # Demonstrate the over-estimation the AJ estimator corrects: under competing risks the
    # cause-specific 1 - KM_c is >= the AJ CIF for that cause, strictly so somewhere.
    recs = [
        SurvivalRecord(duration=float(i % 5 + 1), terminal_mode=TerminationMode.WRONG_PATCH)
        for i in range(40)
    ] + [
        SurvivalRecord(duration=float(i % 5 + 1), terminal_mode=TerminationMode.TOOL_ERROR)
        for i in range(40)
    ]
    cif = aalen_johansen(recs, MODE_A)
    # cause-specific KM treating tool_error as censoring
    only_wrong = EventModel(
        event_modes=frozenset({TerminationMode.WRONG_PATCH}),
        censoring_modes=frozenset(
            set(EventModel.failure_as_event().censoring_modes)
            | {
                TerminationMode.TOOL_ERROR,
                TerminationMode.INFINITE_LOOP,
                TerminationMode.BUDGET_EXHAUSTED,
                TerminationMode.UNLABELED,
            }
        ),
        name="only-wrong",
    )
    km_c = kaplan_meier(recs, only_wrong)
    naive = 1.0 - km_c.survival
    aj = cif.cif_by_cause["wrong_patch"]
    assert np.all(naive >= aj - 1e-12)
    assert np.any(naive > aj + 1e-9)


def test_cif_at_step_lookup() -> None:
    cif = aalen_johansen(GOLDEN, MODE_A)
    assert cif_at(cif, "wrong_patch", 0.5) == 0.0  # before first event
    assert cif_at(cif, "wrong_patch", 1.0) == pytest.approx(0.2)
    assert cif_at(cif, "wrong_patch", 3.9) == pytest.approx(0.2)  # right-continuous
    assert cif_at(cif, "wrong_patch", 4.0) == pytest.approx(0.8)
    with pytest.raises(KeyError):
        cif_at(cif, "no_such_cause", 1.0)


def test_lifelines_cross_check() -> None:
    lifelines = pytest.importorskip("lifelines")
    ajf_cls = lifelines.AalenJohansenFitter

    # Tie-free data so lifelines does not jitter ties (keeps the comparison exact).
    notie = [
        SurvivalRecord(duration=1.0, terminal_mode=TerminationMode.WRONG_PATCH),
        SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.TOOL_ERROR),
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED),  # censored
        SurvivalRecord(duration=4.0, terminal_mode=TerminationMode.WRONG_PATCH),
        SurvivalRecord(duration=5.0, terminal_mode=TerminationMode.TOOL_ERROR),
    ]
    durations = [r.duration for r in notie]
    code = {TerminationMode.WRONG_PATCH: 1, TerminationMode.TOOL_ERROR: 2}
    event = [code.get(r.terminal_mode, 0) for r in notie]
    cif = aalen_johansen(notie, MODE_A)
    for cause_code, cause_name in [(1, "wrong_patch"), (2, "tool_error")]:
        ajf = ajf_cls(seed=0)
        ajf.fit(durations, event, event_of_interest=cause_code)
        theirs = ajf.predict(cif.times).to_numpy().ravel()
        np.testing.assert_allclose(cif.cif_by_cause[cause_name], theirs, atol=1e-9)

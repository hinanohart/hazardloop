"""S1 IR contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from hazardloop.types import (
    ALL_MODES,
    ControlDecision,
    EventModel,
    StepRecord,
    SurvivalRecord,
    TerminationMode,
)


def test_step_record_rejects_negative_step() -> None:
    StepRecord(run_id="r", step=0)  # ok
    with pytest.raises(ValueError, match="step must be"):
        StepRecord(run_id="r", step=-1)


def test_step_record_rejects_negative_wall_time() -> None:
    StepRecord(run_id="r", step=3, wall_time=1.5)  # ok
    with pytest.raises(ValueError, match="wall_time"):
        StepRecord(run_id="r", step=3, wall_time=-0.1)


def test_survival_record_rejects_bad_duration() -> None:
    SurvivalRecord(duration=0.0, terminal_mode=TerminationMode.SOLVED)  # ok (tie at 0)
    with pytest.raises(ValueError, match="finite"):
        SurvivalRecord(duration=float("inf"), terminal_mode=TerminationMode.SOLVED)
    with pytest.raises(ValueError, match=">= 0"):
        SurvivalRecord(duration=-2.0, terminal_mode=TerminationMode.WRONG_PATCH)


def test_event_model_partition_must_be_total_and_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        EventModel(
            event_modes=frozenset({TerminationMode.SOLVED}),
            censoring_modes=frozenset({TerminationMode.SOLVED}),
        )
    with pytest.raises(ValueError, match="cover every"):
        EventModel(
            event_modes=frozenset({TerminationMode.WRONG_PATCH}),
            censoring_modes=frozenset({TerminationMode.SOLVED}),
        )


def test_mode_a_classification() -> None:
    m = EventModel.failure_as_event()
    assert m.name == "mode-A"
    assert m.is_event(TerminationMode.WRONG_PATCH)
    assert m.is_event(TerminationMode.BUDGET_EXHAUSTED)
    assert m.is_event(TerminationMode.UNLABELED)
    assert not m.is_event(TerminationMode.SOLVED)
    assert not m.is_event(TerminationMode.TIMEOUT)
    assert m.cause_of(TerminationMode.SOLVED) is None
    assert m.cause_of(TerminationMode.TOOL_ERROR) == "tool_error"


def test_mode_b_classification() -> None:
    m = EventModel.completion_as_event()
    assert m.name == "mode-B"
    assert m.is_event(TerminationMode.SOLVED)
    assert not m.is_event(TerminationMode.WRONG_PATCH)
    assert m.causes == ("solved",)


def test_causes_are_sorted_and_stable() -> None:
    m = EventModel.failure_as_event()
    assert m.causes == tuple(sorted(m.causes))
    # binary-only (real-narrow) usage: a single UNLABELED cause is well-defined.
    binary = EventModel(
        event_modes=frozenset({TerminationMode.UNLABELED}),
        censoring_modes=frozenset(set(ALL_MODES) - {TerminationMode.UNLABELED}),
        name="binary",
    )
    assert binary.causes == ("unlabeled",)


def test_enums_round_trip_as_strings() -> None:
    assert TerminationMode("solved") is TerminationMode.SOLVED
    assert ControlDecision("abort") is ControlDecision.ABORT
    assert len(ALL_MODES) == len(set(ALL_MODES)) == 8


def test_all_modes_covered_by_default_models() -> None:
    for m in (EventModel.failure_as_event(), EventModel.completion_as_event()):
        union = m.event_modes | m.censoring_modes
        assert union == set(ALL_MODES)
        assert not (m.event_modes & m.censoring_modes)


def test_np_import_sanity() -> None:
    # the IR module exposes numpy-typed array aliases; ensure numpy is wired correctly.
    assert np.array([1.0, 2.0]).dtype == np.float64

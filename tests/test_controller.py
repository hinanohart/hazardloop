"""S4 controller / policy tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hazardloop.controller import Controller
from hazardloop.core.nelson_aalen import nelson_aalen
from hazardloop.policy import HazardThresholdPolicy
from hazardloop.types import ControlDecision, EventModel, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()


def test_policy_decision_thresholds() -> None:
    p = HazardThresholdPolicy(abort_threshold=1.0, fork_threshold=0.6, checkpoint_threshold=0.3)
    assert p.decide(None) is ControlDecision.ABSTAIN  # fail-closed
    assert p.decide(math.nan) is ControlDecision.ABSTAIN
    assert p.decide(0.1) is ControlDecision.CONTINUE
    assert p.decide(0.3) is ControlDecision.CHECKPOINT
    assert p.decide(0.6) is ControlDecision.FORK
    assert p.decide(1.0) is ControlDecision.ABORT
    assert p.decide(5.0) is ControlDecision.ABORT


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="abort_threshold must be > 0"):
        HazardThresholdPolicy(abort_threshold=0.0)
    with pytest.raises(ValueError, match="checkpoint_threshold"):
        HazardThresholdPolicy(abort_threshold=1.0, checkpoint_threshold=2.0)


def test_controller_continue_before_events() -> None:
    recs = [
        SurvivalRecord(duration=5.0, terminal_mode=TerminationMode.WRONG_PATCH),
        SurvivalRecord(duration=8.0, terminal_mode=TerminationMode.SOLVED),
    ]
    na = nelson_aalen(recs, MODE_A)
    ctrl = Controller(na, HazardThresholdPolicy(abort_threshold=0.5))
    # before any event the cumulative hazard estimate is 0.0 -> CONTINUE (a real estimate)
    assert ctrl.decide_at(0.0) is ControlDecision.CONTINUE


def test_controller_first_abort_step() -> None:
    # cumulative hazard reaches threshold at the single event time t=5
    recs = [SurvivalRecord(duration=5.0, terminal_mode=TerminationMode.WRONG_PATCH)]
    na = nelson_aalen(recs, MODE_A)  # one event at t=5, cumhaz=1.0
    ctrl = Controller(na, HazardThresholdPolicy(abort_threshold=0.5))
    assert ctrl.first_abort_step(max_step=10.0) == pytest.approx(5.0)
    assert ctrl.first_abort_step(max_step=4.0) is None  # run never reached the crossing


def test_controller_no_events_never_aborts() -> None:
    recs = [SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED)]
    na = nelson_aalen(recs, MODE_A)  # all censored -> empty table
    ctrl = Controller(na, HazardThresholdPolicy(abort_threshold=0.5))
    assert ctrl.first_abort_step(max_step=100.0) is None
    assert ctrl.decide_at(2.0) is ControlDecision.CONTINUE  # hazard 0.0
    assert na.times.size == 0
    assert np.array_equal(na.cumulative_hazard, np.zeros(0))

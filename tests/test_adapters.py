"""S3 adapter tests: normalisation rules, synthetic generator, SWE-smith pure parsing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hazardloop.adapters.base import TrajectoryBackend
from hazardloop.adapters.mock import MockBackend, synthetic_competing_risks
from hazardloop.adapters.normalize import (
    classify_binary_outcome,
    reduce_steprecords,
    survival_record_from_steps,
)
from hazardloop.adapters.swebench import (
    SweSmithTrajectories,
    count_agent_steps,
    heuristic_failure_mode,
    parse_smith_row,
)
from hazardloop.core.aalen_johansen import aalen_johansen
from hazardloop.types import EventModel, StepRecord, TerminationMode


# --- normalize ---------------------------------------------------------------------------
def test_survival_record_from_steps() -> None:
    r = survival_record_from_steps("run-1", 7, TerminationMode.WRONG_PATCH, cluster="m0")
    assert r.duration == 7.0
    assert r.run_id == "run-1"
    assert r.cluster == "m0"
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        survival_record_from_steps("r", 0, TerminationMode.SOLVED)


def test_reduce_steprecords_terminal_event() -> None:
    steps = [
        StepRecord(run_id="r", step=0),
        StepRecord(run_id="r", step=1),
        StepRecord(run_id="r", step=2, event_type=TerminationMode.TOOL_ERROR),
    ]
    rec = reduce_steprecords(steps, cluster="m1")
    assert rec.duration == 3.0  # steps 0..2 -> 3 steps
    assert rec.terminal_mode is TerminationMode.TOOL_ERROR
    assert rec.run_id == "r"


def test_reduce_steprecords_censored_terminal() -> None:
    steps = [StepRecord(run_id="r", step=0), StepRecord(run_id="r", step=1, censored=True)]
    rec = reduce_steprecords(steps)
    assert rec.terminal_mode is TerminationMode.CENSORED
    with pytest.raises(ValueError, match="at least one step"):
        reduce_steprecords([])


def test_classify_binary_outcome() -> None:
    assert classify_binary_outcome(True) is TerminationMode.SOLVED
    assert classify_binary_outcome(False) is TerminationMode.UNLABELED


# --- mock generator ----------------------------------------------------------------------
def test_synthetic_is_deterministic() -> None:
    a = synthetic_competing_risks(200, seed=3)
    b = synthetic_competing_risks(200, seed=3)
    assert [(r.duration, r.terminal_mode, r.cluster) for r in a] == [
        (r.duration, r.terminal_mode, r.cluster) for r in b
    ]


def test_synthetic_has_multiple_causes_and_valid_durations() -> None:
    recs = synthetic_competing_risks(1000, seed=1)
    modes = {r.terminal_mode for r in recs}
    # at least two distinct failure causes appear (so the typed CIF is non-trivial)
    failure_causes = modes & {
        TerminationMode.WRONG_PATCH,
        TerminationMode.TOOL_ERROR,
        TerminationMode.INFINITE_LOOP,
        TerminationMode.BUDGET_EXHAUSTED,
    }
    assert len(failure_causes) >= 2
    assert all(r.duration >= 1.0 for r in recs)
    # the synthetic CIF is well-formed (additivity)
    cif = aalen_johansen(recs, EventModel.failure_as_event())
    np.testing.assert_allclose(cif.total_cif(), 1.0 - cif.overall_survival, atol=1e-9)


def test_mock_backend_is_synthetic_and_protocol_conformant() -> None:
    be = MockBackend(n=50, seed=0)
    assert isinstance(be, TrajectoryBackend)
    status = be.doctor()
    assert status.is_synthetic is True
    assert status.available is True
    assert len(list(be.load(limit=10))) == 10


# --- SWE-smith pure parsing (no network) -------------------------------------------------
def _row(resolved: bool, n_assistant: int = 2, extra_msgs: list[dict] | None = None) -> dict:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n_assistant):
        msgs.append({"role": "assistant", "content": f"act-{i}"})
        msgs.append({"role": "tool", "content": "output"})
    msgs.extend(extra_msgs or [])
    return {
        "messages": json.dumps(msgs),
        "resolved": resolved,
        "model": "claude-3-7-sonnet-20250219",
        "traj_id": "django__django.abc.func__xyz",
        "instance_id": "django__django.abc",
    }


def test_count_agent_steps() -> None:
    msgs = json.loads(_row(True, n_assistant=5)["messages"])
    assert count_agent_steps(msgs) == 5
    assert count_agent_steps([]) == 1  # floor at 1


def test_parse_smith_row_resolved_and_unresolved() -> None:
    solved = parse_smith_row(_row(True, n_assistant=3))
    assert solved.terminal_mode is TerminationMode.SOLVED
    assert solved.duration == 3.0
    assert solved.cluster == "claude-3-7-sonnet-20250219"
    assert solved.run_id == "django__django.abc.func__xyz"

    failed = parse_smith_row(_row(False, n_assistant=4))
    assert failed.terminal_mode is TerminationMode.UNLABELED  # never an invented typed cause
    assert failed.duration == 4.0


def test_heuristic_failure_mode_branches() -> None:
    # resolved -> SOLVED
    msgs = json.loads(_row(True, n_assistant=2)["messages"])
    assert heuristic_failure_mode(msgs, resolved=True) is TerminationMode.SOLVED
    # budget exhausted (steps >= budget)
    big = json.loads(_row(False, n_assistant=10)["messages"])
    assert (
        heuristic_failure_mode(big, resolved=False, budget_steps=5)
        is TerminationMode.BUDGET_EXHAUSTED
    )
    # infinite loop (identical assistant tail)
    loop = [{"role": "assistant", "content": "same"} for _ in range(4)]
    assert (
        heuristic_failure_mode(loop, resolved=False, budget_steps=99)
        is TerminationMode.INFINITE_LOOP
    )
    # tool error marker
    terr = [
        {"role": "assistant", "content": "run"},
        {"role": "tool", "content": "Traceback (most recent call last): boom"},
    ]
    assert (
        heuristic_failure_mode(terr, resolved=False, budget_steps=99) is TerminationMode.TOOL_ERROR
    )
    # nothing fires -> UNLABELED (no invented cause)
    plain = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert (
        heuristic_failure_mode(plain, resolved=False, budget_steps=99) is TerminationMode.UNLABELED
    )


def test_swebench_doctor_reports_real_not_synthetic() -> None:
    be = SweSmithTrajectories()
    assert isinstance(be, TrajectoryBackend)
    status = be.doctor()
    assert status.is_synthetic is False  # honest: never claims synthetic data is real or vice-versa


@pytest.mark.network
def test_swebench_real_load_smoke() -> None:
    pytest.importorskip("datasets")
    be = SweSmithTrajectories()
    recs = list(be.load(limit=20))
    assert len(recs) == 20
    assert all(r.duration >= 1.0 for r in recs)
    assert all(r.terminal_mode in {TerminationMode.SOLVED, TerminationMode.UNLABELED} for r in recs)

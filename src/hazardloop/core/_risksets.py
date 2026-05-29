"""Shared counting-process risk table.

Every estimator (KM, Nelson-Aalen, Aalen-Johansen) is a deterministic function of the
same object: the ordered distinct *event* times together with, at each one, the number
at risk ``Y_j`` and the number of events ``d_j`` (overall and per competing cause).
Building this once keeps the estimators consistent by construction.

Conventions:
- The process only *steps* at times where at least one event occurs; right-censored
  observations affect the estimators solely through the at-risk counts ``Y_j``.
- ``Y_j = #{i : T_i >= t_j}`` — an observation censored exactly at ``t_j`` is counted as
  at risk at ``t_j`` (standard "events before censoring at ties" convention).
- An all-censored sample yields an empty table; estimators then return the trivial
  ``S(t) ≡ 1`` / ``H(t) ≡ 0`` / ``CIF ≡ 0`` (see the step-evaluation helpers).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from hazardloop.types import EventModel, FloatArray, IntArray, SurvivalRecord


@dataclass(frozen=True)
class RiskTable:
    times: FloatArray
    n_at_risk: IntArray
    n_events: IntArray
    n_censored_at_time: IntArray
    events_by_cause: Mapping[str, IntArray]
    causes: tuple[str, ...]
    n_total: int
    n_events_total: int
    n_censored_total: int


def build_risk_table(records: Sequence[SurvivalRecord], event_model: EventModel) -> RiskTable:
    """Construct the risk table from one-row-per-run survival records.

    Raises ``ValueError`` on an empty record set (an empty input is a caller bug, never a
    silently-returned degenerate result — see bootstrap protocol BP5).
    """
    if len(records) == 0:
        raise ValueError("build_risk_table requires at least one record")

    durations = np.asarray([r.duration for r in records], dtype=np.float64)
    is_event = np.asarray([event_model.is_event(r.terminal_mode) for r in records], dtype=bool)
    cause_of = np.asarray(
        [event_model.cause_of(r.terminal_mode) or "" for r in records], dtype=object
    )

    n_total = len(records)
    n_events_total = int(is_event.sum())
    n_censored_total = n_total - n_events_total

    event_durations = durations[is_event]
    times = np.unique(event_durations)  # sorted ascending, distinct event times

    if times.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return RiskTable(
            times=np.zeros(0, dtype=np.float64),
            n_at_risk=empty_i,
            n_events=empty_i,
            n_censored_at_time=empty_i,
            events_by_cause={c: empty_i for c in event_model.causes},
            causes=event_model.causes,
            n_total=n_total,
            n_events_total=0,
            n_censored_total=n_censored_total,
        )

    n_at_risk = np.empty(times.size, dtype=np.int64)
    n_events = np.empty(times.size, dtype=np.int64)
    n_censored_at_time = np.empty(times.size, dtype=np.int64)
    causes = event_model.causes
    events_by_cause: dict[str, IntArray] = {c: np.zeros(times.size, dtype=np.int64) for c in causes}

    for j, t in enumerate(times):
        at_t = durations == t
        n_at_risk[j] = int(np.count_nonzero(durations >= t))
        n_events[j] = int(np.count_nonzero(at_t & is_event))
        n_censored_at_time[j] = int(np.count_nonzero(at_t & ~is_event))
        for c in causes:
            events_by_cause[c][j] = int(np.count_nonzero(at_t & is_event & (cause_of == c)))

    return RiskTable(
        times=times,
        n_at_risk=n_at_risk,
        n_events=n_events,
        n_censored_at_time=n_censored_at_time,
        events_by_cause=events_by_cause,
        causes=causes,
        n_total=n_total,
        n_events_total=n_events_total,
        n_censored_total=n_censored_total,
    )


def step_lookup(times: FloatArray, values: FloatArray, query: float, baseline: float) -> float:
    """Right-continuous step-function evaluation: value of the last step at time <= query.

    Returns ``baseline`` when ``query`` precedes the first step (or the table is empty).
    Used to read S(t), H(t), CIF(t) at an arbitrary time (e.g. a control step).
    """
    if times.size == 0 or query < times[0]:
        return baseline
    idx = int(np.searchsorted(times, query, side="right")) - 1
    return float(values[idx])

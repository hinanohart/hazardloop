"""Normalisation rules: raw run shapes -> SurvivalRecord.

These functions are the *only* place the censoring and failure-mode conventions live, so
they are 1:1 with the :class:`~hazardloop.types.EventModel` partition used by the core.
Each rule has a unit test pinning its behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hazardloop.types import StepRecord, SurvivalRecord, TerminationMode


def survival_record_from_steps(
    run_id: str,
    n_steps: int,
    terminal_mode: TerminationMode,
    *,
    cluster: str | None = None,
    covariates: Mapping[str, float] | None = None,
) -> SurvivalRecord:
    """Build a record on the step axis. ``duration = n_steps`` (must be >= 1)."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return SurvivalRecord(
        duration=float(n_steps),
        terminal_mode=terminal_mode,
        cluster=cluster,
        run_id=run_id,
        covariates=dict(covariates or {}),
    )


def reduce_steprecords(
    steps: Sequence[StepRecord], *, cluster: str | None = None
) -> SurvivalRecord:
    """Collapse a per-step trajectory into one survival record.

    Duration is the maximum step index seen (steps are 0-based, so duration = max+1 steps
    taken). The terminal mode is the terminal step's ``event_type``; if the terminal step
    is flagged ``censored`` (or carries no event type), the record is right-censored
    (``CENSORED``).
    """
    if len(steps) == 0:
        raise ValueError("reduce_steprecords requires at least one step")
    ordered = sorted(steps, key=lambda s: s.step)
    last = ordered[-1]
    duration = float(last.step + 1)
    if last.censored or last.event_type is None:
        terminal = TerminationMode.CENSORED
    else:
        terminal = last.event_type
    cov = dict(last.covariates)
    return SurvivalRecord(
        duration=duration,
        terminal_mode=terminal,
        cluster=cluster,
        run_id=last.run_id,
        covariates=cov,
    )


def classify_binary_outcome(resolved: bool) -> TerminationMode:
    """The real-narrow [B] rule for binary-only data (e.g. SWE-smith-trajectories).

    A resolved run is a success (censored under failure-as-event); an unresolved run is an
    observed non-success whose cause is *not* labelled in the data, hence ``UNLABELED`` —
    never a silently-invented typed cause (bootstrap protocol BP4).
    """
    return TerminationMode.SOLVED if resolved else TerminationMode.UNLABELED

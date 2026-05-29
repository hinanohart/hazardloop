"""Intermediate representation (IR) and result types for hazardloop.

This module is the single contract between trajectory adapters and the survival core.
It is dependency-light (numpy only) so it sits at the root of the import DAG:

    types  ->  (numpy)
    core   ->  types
    adapters, replay, controller, policy, cli  ->  core + types

The key design decision is :class:`EventModel`: the mapping from a terminal
:class:`TerminationMode` to "observed event (with a cause)" vs "right-censored" is made
**explicit, total, and machine-checkable** here, rather than hard-coded inside an
estimator. That is what lets the same records be analysed under failure-as-event
(``mode-A``) or completion-as-event (``mode-B``) without re-labelling the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class TerminationMode(StrEnum):
    """How an agent run ended.

    The failure causes are competing risks; ``SOLVED`` is success; ``TIMEOUT`` and
    ``CENSORED`` are administrative right-censoring by default; ``UNLABELED`` is an
    observed non-success whose cause could not be reliably extracted (see bootstrap
    protocol BP4 — never silently invent a typed cause).
    """

    SOLVED = "solved"
    WRONG_PATCH = "wrong_patch"
    TOOL_ERROR = "tool_error"
    INFINITE_LOOP = "infinite_loop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    UNLABELED = "unlabeled"
    CENSORED = "censored"


ALL_MODES: tuple[TerminationMode, ...] = tuple(TerminationMode)


@dataclass(frozen=True)
class StepRecord:
    """One step of one agent run — the sole join point between adapters and core.

    ``event_type`` is set only on the terminal step (``None`` elsewhere); ``censored``
    marks a run whose terminal step is right-censored. ``covariates`` carries optional
    per-step features (e.g. tool-error count) used by richer policies.
    """

    run_id: str
    step: int
    event_type: TerminationMode | None = None
    censored: bool = False
    wall_time: float | None = None
    covariates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type is not None and not isinstance(self.event_type, TerminationMode):
            object.__setattr__(self, "event_type", TerminationMode(self.event_type))
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if self.wall_time is not None and self.wall_time < 0:
            raise ValueError(f"wall_time must be >= 0, got {self.wall_time}")


@dataclass(frozen=True)
class SurvivalRecord:
    """Reduced one-row-per-run form consumed by the estimators.

    ``duration`` is the time-to-event on the chosen axis (default: step count).
    ``terminal_mode`` is the rich terminal label; the split into event/censored is the
    job of :class:`EventModel`, not of this record.
    """

    duration: float
    terminal_mode: TerminationMode
    cluster: str | None = None
    covariates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_mode, TerminationMode):
            # accept a plain str / numpy str; reject unknown values (fail fast).
            object.__setattr__(self, "terminal_mode", TerminationMode(self.terminal_mode))
        if not np.isfinite(self.duration):
            raise ValueError(f"duration must be finite, got {self.duration}")
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0, got {self.duration}")


class ControlDecision(StrEnum):
    """A fail-closed control policy's decision at a step. ``ABSTAIN`` is the safe default
    when the estimator is too uncertain to recommend anything."""

    CONTINUE = "continue"
    CHECKPOINT = "checkpoint"
    FORK = "fork"
    ABORT = "abort"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class EventModel:
    """Total, disjoint partition of :class:`TerminationMode` into events vs censoring.

    Invariant (validated): ``event_modes`` and ``censoring_modes`` are disjoint and their
    union is exactly :data:`ALL_MODES`. Each event mode is its own competing-risk cause.
    """

    event_modes: frozenset[TerminationMode]
    censoring_modes: frozenset[TerminationMode]
    name: str = "custom"

    def __post_init__(self) -> None:
        overlap = self.event_modes & self.censoring_modes
        if overlap:
            raise ValueError(f"event/censoring modes overlap: {sorted(m.value for m in overlap)}")
        union = self.event_modes | self.censoring_modes
        missing = set(ALL_MODES) - union
        if missing:
            raise ValueError(
                "EventModel must cover every TerminationMode; missing: "
                f"{sorted(m.value for m in missing)}"
            )

    def is_event(self, mode: TerminationMode) -> bool:
        return mode in self.event_modes

    def cause_of(self, mode: TerminationMode) -> str | None:
        """Competing-risk cause label for an event mode, else ``None`` (censored)."""
        return mode.value if mode in self.event_modes else None

    @property
    def causes(self) -> tuple[str, ...]:
        """Sorted, stable cause labels (the CIF has one curve per cause)."""
        return tuple(sorted(m.value for m in self.event_modes))

    @classmethod
    def failure_as_event(cls) -> EventModel:
        """mode-A (default): failures (incl. budget exhaustion and unlabeled non-success)
        are events; success and administrative timeout are censored."""
        return cls(
            event_modes=frozenset(
                {
                    TerminationMode.WRONG_PATCH,
                    TerminationMode.TOOL_ERROR,
                    TerminationMode.INFINITE_LOOP,
                    TerminationMode.BUDGET_EXHAUSTED,
                    TerminationMode.UNLABELED,
                }
            ),
            censoring_modes=frozenset(
                {TerminationMode.SOLVED, TerminationMode.TIMEOUT, TerminationMode.CENSORED}
            ),
            name="mode-A",
        )

    @classmethod
    def completion_as_event(cls) -> EventModel:
        """mode-B: success is the event; everything else is censored."""
        return cls(
            event_modes=frozenset({TerminationMode.SOLVED}),
            censoring_modes=frozenset(set(ALL_MODES) - {TerminationMode.SOLVED}),
            name="mode-B",
        )


# --- Result containers (populated by core; defined here to keep the import DAG acyclic) --
@dataclass(frozen=True)
class KMResult:
    """Kaplan-Meier survival estimate with Greenwood variance and a cloglog band."""

    times: FloatArray
    survival: FloatArray
    var_greenwood: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    n_at_risk: IntArray
    n_events: IntArray
    alpha: float


@dataclass(frozen=True)
class NAResult:
    """Nelson-Aalen cumulative hazard with per-time hazard increments."""

    times: FloatArray
    cumulative_hazard: FloatArray
    variance: FloatArray
    hazard_increment: FloatArray
    n_at_risk: IntArray
    n_events: IntArray


@dataclass(frozen=True)
class CIFResult:
    """Aalen-Johansen competing-risk cumulative incidence, one curve per cause."""

    times: FloatArray
    cif_by_cause: Mapping[str, FloatArray]
    overall_survival: FloatArray

    def total_cif(self) -> FloatArray:
        """Sum of per-cause CIFs; the additivity invariant is ``total_cif + S == 1``."""
        if not self.cif_by_cause:
            return np.zeros_like(self.times, dtype=np.float64)
        stacked = np.vstack(list(self.cif_by_cause.values()))
        return cast(FloatArray, np.sum(stacked, axis=0))


@dataclass(frozen=True)
class WeibullAFTResult:
    """Weibull fit. ``shape`` (β) < 1 = early-mortality, > 1 = wear-out."""

    shape: float
    scale: float
    loglik: float
    n_events: int
    n_censored: int


@dataclass(frozen=True)
class SurvivalReport:
    """Top-level fit bundle returned by the high-level API."""

    n_runs: int
    event_model_name: str
    km: KMResult
    na: NAResult
    cif: CIFResult | None
    weibull: WeibullAFTResult | None
    cif_mode: str  # "live" (causes from data) or "synthetic" (causes from synthetic only)


__all__ = [
    "ALL_MODES",
    "CIFResult",
    "ControlDecision",
    "EventModel",
    "FloatArray",
    "IntArray",
    "KMResult",
    "NAResult",
    "StepRecord",
    "SurvivalRecord",
    "SurvivalReport",
    "TerminationMode",
    "WeibullAFTResult",
]

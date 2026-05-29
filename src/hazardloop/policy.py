"""Fail-closed control policies (rule-based; nothing is learned).

A policy maps an estimated risk at a step to a :class:`~hazardloop.types.ControlDecision`.
The reference policy thresholds on the Nelson-Aalen cumulative hazard. "Fail-closed" means
that when there is no usable estimate the policy returns ``ABSTAIN`` rather than guessing —
the caller decides what to do with an abstention, and the default is to keep running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hazardloop.types import ControlDecision


@dataclass(frozen=True)
class HazardThresholdPolicy:
    """Abort when the cumulative hazard reaches ``abort_threshold``; optionally checkpoint
    at an earlier ``checkpoint_threshold``. Forking is offered between the two when
    ``fork_threshold`` is set (its rescue benefit is intentionally never measured —
    NON-CLAIM)."""

    abort_threshold: float
    checkpoint_threshold: float | None = None
    fork_threshold: float | None = None
    name: str = "hazard-threshold"

    def __post_init__(self) -> None:
        if self.abort_threshold <= 0:
            raise ValueError("abort_threshold must be > 0")
        for attr in ("checkpoint_threshold", "fork_threshold"):
            v = getattr(self, attr)
            if v is not None and not (0.0 < v <= self.abort_threshold):
                raise ValueError(f"{attr} must be in (0, abort_threshold]")

    def decide(self, cumulative_hazard: float | None) -> ControlDecision:
        if cumulative_hazard is None or not math.isfinite(cumulative_hazard):
            return ControlDecision.ABSTAIN  # fail-closed: no estimate -> do not guess
        if cumulative_hazard >= self.abort_threshold:
            return ControlDecision.ABORT
        if self.fork_threshold is not None and cumulative_hazard >= self.fork_threshold:
            return ControlDecision.FORK
        if self.checkpoint_threshold is not None and cumulative_hazard >= self.checkpoint_threshold:
            return ControlDecision.CHECKPOINT
        return ControlDecision.CONTINUE

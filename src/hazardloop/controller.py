"""Controller: bind a fitted hazard estimate to a fail-closed policy.

The controller is the live-facing interface — it answers "given the cumulative hazard at
this step, what should we do?" — but it never executes, retrains, or stops anything itself;
it only returns a decision. The actual middleware that would call it on a running agent is
deferred (v0.2); in this release the controller is exercised through offline replay.
"""

from __future__ import annotations

import numpy as np

from hazardloop.core.nelson_aalen import cumulative_hazard_at
from hazardloop.policy import HazardThresholdPolicy
from hazardloop.types import ControlDecision, NAResult


class Controller:
    def __init__(self, na: NAResult, policy: HazardThresholdPolicy) -> None:
        self._na = na
        self._policy = policy

    @property
    def policy(self) -> HazardThresholdPolicy:
        return self._policy

    def decide_at(self, step: float) -> ControlDecision:
        """Decision at ``step`` from the fitted cumulative hazard. Before any event the
        hazard estimate is 0.0 (a genuine estimate, so this is CONTINUE, not ABSTAIN)."""
        return self._policy.decide(cumulative_hazard_at(self._na, step))

    def first_abort_step(self, max_step: float) -> float | None:
        """Earliest event time at which cumulative hazard reaches the abort threshold and
        which the run actually reached (``<= max_step``); ``None`` if it never aborts."""
        na = self._na
        if na.times.size == 0:
            return None
        # cumulative hazard is monotone non-decreasing -> first crossing by searchsorted
        idx = int(np.searchsorted(na.cumulative_hazard, self._policy.abort_threshold, side="left"))
        if idx >= na.times.size:
            return None
        t = float(na.times[idx])
        return t if t <= max_step else None

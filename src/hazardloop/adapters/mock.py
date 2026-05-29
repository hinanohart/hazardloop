"""Deterministic synthetic competing-risks generator.

Each run draws a latent time per failure cause from a constant-hazard (exponential) model
plus an independent success time; the earliest wins, and runs exceeding a step cap are
administratively censored (``TIMEOUT``). Because the cause-specific hazards are known and
constant, the resulting Aalen-Johansen CIFs are meaningful and reproducible — this is what
the synthetic ("algorithm validation only") typed-CIF path uses (``cif_mode='synthetic'``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from hazardloop.adapters.base import BackendStatus
from hazardloop.adapters.normalize import survival_record_from_steps
from hazardloop.types import SurvivalRecord, TerminationMode

DEFAULT_CAUSE_RATES: Mapping[str, float] = {
    "wrong_patch": 0.05,
    "tool_error": 0.03,
    "infinite_loop": 0.02,
    "budget_exhausted": 0.015,
}


def synthetic_competing_risks(
    n: int = 500,
    *,
    seed: int = 0,
    cause_rates: Mapping[str, float] | None = None,
    success_rate: float = 0.04,
    max_steps: int = 60,
    n_clusters: int = 4,
) -> list[SurvivalRecord]:
    """Generate ``n`` synthetic runs with known competing-risk structure (deterministic)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rates = dict(cause_rates or DEFAULT_CAUSE_RATES)
    if any(r <= 0 for r in rates.values()) or success_rate <= 0:
        raise ValueError("all rates must be > 0")
    rng = np.random.default_rng(seed)

    records: list[SurvivalRecord] = []
    for i in range(n):
        cause_times = {c: float(rng.exponential(1.0 / r)) for c, r in rates.items()}
        t_success = float(rng.exponential(1.0 / success_rate))
        best_cause = min(cause_times, key=lambda c: cause_times[c])
        t_event = cause_times[best_cause]

        if t_success < t_event:
            duration, mode = t_success, TerminationMode.SOLVED
        else:
            duration, mode = t_event, TerminationMode(best_cause)

        steps = max(1, math.ceil(duration))
        if steps >= max_steps:
            steps, mode = max_steps, TerminationMode.TIMEOUT  # administrative censoring at cap

        records.append(
            survival_record_from_steps(
                run_id=f"synthetic-{i}",
                n_steps=steps,
                terminal_mode=mode,
                cluster=f"synthetic-model-{i % n_clusters}",
            )
        )
    return records


class MockBackend:
    """A :class:`~hazardloop.adapters.base.TrajectoryBackend` over the synthetic generator."""

    def __init__(self, n: int = 500, seed: int = 0) -> None:
        self._n = n
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock"

    def doctor(self) -> BackendStatus:
        return BackendStatus(
            name="mock",
            available=True,
            is_synthetic=True,
            detail=f"deterministic synthetic competing-risks generator (n={self._n}, seed={self._seed})",
        )

    def load(self, limit: int | None = None) -> Sequence[SurvivalRecord]:
        n = self._n if limit is None else min(self._n, limit)
        return synthetic_competing_risks(n, seed=self._seed)

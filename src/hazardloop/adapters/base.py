"""Trajectory backend protocol and an honest status report.

A backend turns some source of logged agent runs into one-row-per-run
:class:`~hazardloop.types.SurvivalRecord` objects. ``doctor()`` reports — honestly —
whether the backend is a synthetic generator or a real data source, and whether the real
source is actually reachable. The CLI surfaces this so a synthetic run is never mistaken
for a real measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hazardloop.types import SurvivalRecord


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    is_synthetic: bool
    detail: str


@runtime_checkable
class TrajectoryBackend(Protocol):
    """A source of survival records. Implementations must be explicit about whether their
    output is synthetic (``doctor().is_synthetic``)."""

    @property
    def name(self) -> str: ...

    def doctor(self) -> BackendStatus: ...

    def load(self, limit: int | None = None) -> Sequence[SurvivalRecord]: ...

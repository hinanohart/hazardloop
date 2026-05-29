"""Placeholder backends for live RL harnesses (``verifiers``, ``OpenEnv``).

These are declared but **not implemented** in this release: hazardloop 0.1.0a1 reads only
saved trajectory logs and never runs an agent live (the live middleware that would consume
these is deferred to v0.2). The classes exist so the backend surface is discoverable and so
callers get an explicit, honest ``NotImplementedError`` rather than a missing attribute.
"""

from __future__ import annotations

from collections.abc import Sequence

from hazardloop.adapters.base import BackendStatus
from hazardloop.types import SurvivalRecord

_DEFERRED = (
    "live RL-harness backends are deferred to v0.2; 0.1.0a1 consumes saved trajectory logs "
    "only (use the mock or swe-smith backends)."
)


class _DeferredBackend:
    _name = "deferred"

    @property
    def name(self) -> str:
        return self._name

    def doctor(self) -> BackendStatus:
        return BackendStatus(name=self._name, available=False, is_synthetic=False, detail=_DEFERRED)

    def load(self, limit: int | None = None) -> Sequence[SurvivalRecord]:
        raise NotImplementedError(f"{self._name}: {_DEFERRED}")


class VerifiersBackend(_DeferredBackend):
    """Stub for the ``verifiers`` RL-environment harness (v0.2)."""

    _name = "verifiers"


class OpenEnvBackend(_DeferredBackend):
    """Stub for the ``OpenEnv`` harness (v0.2)."""

    _name = "openenv"

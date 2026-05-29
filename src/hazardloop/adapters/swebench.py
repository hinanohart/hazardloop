"""Adapter for the ``SWE-bench/SWE-smith-trajectories`` Hugging Face dataset (MIT).

The dataset gives, per trajectory, a JSON ``messages`` conversation, a binary ``resolved``
flag, and ids — but **no** typed failure-mode label and **no** explicit censoring flag
(verified by live inspection at S0; GATE-1 verdict = real-narrow [B]).

Consequently:
- The default :meth:`SweSmithTrajectories.load` produces real records on the step axis
  with binary outcomes only (resolved -> SOLVED, unresolved -> UNLABELED). This supports
  real Kaplan-Meier / Nelson-Aalen (the real gate-real numbers).
- A typed competing-risk CIF cannot be computed from the real labels. An *opt-in*
  heuristic cause extractor is provided, but its accuracy is **UNVALIDATED** in this
  release: it is reported as an extractor self-description, never as a CLAIM, and its
  output is not used for headline metrics (bootstrap protocol BP4 / BP1).

The pure parsing functions take plain dicts so they are unit-tested without any network;
only :meth:`load` touches the network (lazy ``datasets`` import, marked in CI).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hazardloop.adapters.base import BackendStatus
from hazardloop.adapters.normalize import classify_binary_outcome, survival_record_from_steps
from hazardloop.types import SurvivalRecord, TerminationMode

DATASET_ID = "SWE-bench/SWE-smith-trajectories"


def _as_messages(raw: Any) -> list[dict[str, Any]]:
    """Coerce the ``messages`` field (a JSON string in the parquet) into a list of dicts."""
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, list):
        raise ValueError("messages must decode to a list")
    return [m for m in parsed if isinstance(m, dict)]


def count_agent_steps(messages: Sequence[Mapping[str, Any]]) -> int:
    """Number of agent steps = number of assistant turns (>= 1)."""
    n = sum(1 for m in messages if m.get("role") == "assistant")
    return max(1, n)


def parse_smith_row(row: Mapping[str, Any], *, cluster_by_model: bool = True) -> SurvivalRecord:
    """Pure conversion of one dataset row into a real survival record (binary outcome)."""
    messages = _as_messages(row["messages"])
    n_steps = count_agent_steps(messages)
    resolved = bool(row["resolved"])
    mode = classify_binary_outcome(resolved)
    model = str(row.get("model", "")) or None
    run_id = str(row.get("traj_id") or row.get("instance_id") or "")
    return survival_record_from_steps(
        run_id=run_id or "unknown",
        n_steps=n_steps,
        terminal_mode=mode,
        cluster=model if cluster_by_model else None,
    )


# --- BP4: opt-in heuristic cause extractor (UNVALIDATED) ----------------------------------
_TOOL_ERROR_MARKERS = ("traceback (most recent call last)", "command not found", "no such file")
_LOOP_WINDOW = 4


def heuristic_failure_mode(
    messages: Sequence[Mapping[str, Any]], resolved: bool, *, budget_steps: int = 75
) -> TerminationMode:
    """Best-effort typed failure mode from the conversation. UNVALIDATED — not a CLAIM.

    Returns ``UNLABELED`` whenever no rule fires, so a typed cause is never invented.
    """
    if resolved:
        return TerminationMode.SOLVED
    n_steps = count_agent_steps(messages)
    if n_steps >= budget_steps:
        return TerminationMode.BUDGET_EXHAUSTED
    assistant_contents = [
        str(m.get("content", "")) for m in messages if m.get("role") == "assistant"
    ]
    if len(assistant_contents) >= _LOOP_WINDOW:
        tail = assistant_contents[-_LOOP_WINDOW:]
        if len(set(tail)) == 1:
            return TerminationMode.INFINITE_LOOP
    tool_text = " ".join(
        str(m.get("content", "")).lower() for m in messages if m.get("role") in {"tool", "user"}
    )
    if any(marker in tool_text for marker in _TOOL_ERROR_MARKERS):
        return TerminationMode.TOOL_ERROR
    return TerminationMode.UNLABELED


@dataclass(frozen=True)
class HeuristicReport:
    """Self-description of the heuristic extractor's output. ``validated`` is False in this
    release: the per-cause agreement rate against ground truth has not been measured, so
    these counts must not be presented as a calibrated CLAIM (BP4)."""

    n_runs: int
    cause_counts: Mapping[str, int]
    validated: bool
    note: str


class SweSmithTrajectories:
    """Real backend over SWE-smith-trajectories (binary outcomes; lazy ``datasets`` import)."""

    def __init__(self, split: str = "tool", dataset_id: str = DATASET_ID) -> None:
        self._split = split
        self._dataset_id = dataset_id

    @property
    def name(self) -> str:
        return f"swe-smith-trajectories:{self._split}"

    def doctor(self) -> BackendStatus:
        try:
            import datasets  # noqa: F401
        except ImportError:
            return BackendStatus(
                name=self.name,
                available=False,
                is_synthetic=False,
                detail="install the [data] extra (datasets) to load real trajectories",
            )
        return BackendStatus(
            name=self.name,
            available=True,
            is_synthetic=False,
            detail=f"real dataset {self._dataset_id} split={self._split} (binary outcome only; "
            "typed failure mode not in source — see BP4)",
        )

    def _iter_rows(self, limit: int | None) -> list[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(self._dataset_id, split=self._split, streaming=True)
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            rows.append(dict(row))
        return rows

    def load(self, limit: int | None = None) -> Sequence[SurvivalRecord]:
        """Load real records with binary outcomes (network; lazy import)."""
        return [parse_smith_row(row) for row in self._iter_rows(limit)]

    def load_with_heuristic_causes(
        self, limit: int | None = None, *, budget_steps: int = 75
    ) -> tuple[list[SurvivalRecord], HeuristicReport]:
        """Opt-in: attach heuristic typed causes. Output is UNVALIDATED (BP4)."""
        records: list[SurvivalRecord] = []
        counts: dict[str, int] = {}
        for row in self._iter_rows(limit):
            messages = _as_messages(row["messages"])
            resolved = bool(row["resolved"])
            mode = heuristic_failure_mode(messages, resolved, budget_steps=budget_steps)
            counts[mode.value] = counts.get(mode.value, 0) + 1
            records.append(
                survival_record_from_steps(
                    run_id=str(row.get("traj_id") or "unknown"),
                    n_steps=count_agent_steps(messages),
                    terminal_mode=mode,
                    cluster=str(row.get("model", "")) or None,
                )
            )
        report = HeuristicReport(
            n_runs=len(records),
            cause_counts=counts,
            validated=False,
            note="heuristic cause labels — accuracy UNVALIDATED against ground truth; "
            "not a calibrated CLAIM (BP4). Use for exploration only.",
        )
        return records, report

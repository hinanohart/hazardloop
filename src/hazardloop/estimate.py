"""High-level fit that assembles the survival core into a single :class:`SurvivalReport`.

``cif_mode`` records the provenance of the typed competing-risk CIF:
- ``"live"``  — the causes come from real per-cause labels in the data;
- ``"synthetic"`` — the multi-cause typed CIF is validated on synthetic data only
  (the real source lacked typed failure-mode labels). This flag drives the BP1 honest
  disclaimer downstream.

When the chosen event model observes no events, the report is built honestly: the survival
curve is the trivial ``S ≡ 1`` (an empty step table) and the Weibull fit is skipped — the
situation is censoring-dominated and is reported as such rather than fabricated.
"""

from __future__ import annotations

from collections.abc import Sequence

from hazardloop.core._risksets import build_risk_table
from hazardloop.core.aalen_johansen import aalen_johansen_from_table
from hazardloop.core.kaplan_meier import kaplan_meier_from_table
from hazardloop.core.nelson_aalen import nelson_aalen_from_table
from hazardloop.core.weibull_aft import weibull_aft
from hazardloop.types import EventModel, SurvivalRecord, SurvivalReport, WeibullAFTResult

VALID_CIF_MODES = ("live", "synthetic")


def fit_survival(
    records: Sequence[SurvivalRecord],
    event_model: EventModel | None = None,
    *,
    cif_mode: str = "live",
    alpha: float = 0.05,
    fit_weibull: bool = True,
) -> SurvivalReport:
    if cif_mode not in VALID_CIF_MODES:
        raise ValueError(f"cif_mode must be one of {VALID_CIF_MODES}, got {cif_mode!r}")
    model = event_model or EventModel.failure_as_event()
    table = build_risk_table(records, model)

    km = kaplan_meier_from_table(table, alpha)
    na = nelson_aalen_from_table(table)
    cif = aalen_johansen_from_table(table) if table.n_events_total > 0 else None

    weibull: WeibullAFTResult | None = None
    if fit_weibull and table.n_events_total > 0:
        durations_positive = all(r.duration > 0 for r in records)
        if durations_positive:
            weibull = weibull_aft(records, model)

    return SurvivalReport(
        n_runs=len(records),
        event_model_name=model.name,
        km=km,
        na=na,
        cif=cif,
        weibull=weibull,
        cif_mode=cif_mode,
    )


def has_observed_events(report: SurvivalReport) -> bool:
    """True if any event was observed (the survival curve is informative, not just S≡1)."""
    return report.km.times.size > 0

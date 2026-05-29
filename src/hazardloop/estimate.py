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
UNLABELED_CAUSE = "unlabeled"


def can_claim_live_cif(records: Sequence[SurvivalRecord], event_model: EventModel) -> bool:
    """True only if the data carries >= 2 distinct *real* typed causes (not the placeholder
    ``unlabeled``). This is a necessary condition for a genuine multi-cause typed CIF; it is
    *not sufficient* — synthetic data can also have many causes — so ``cif_mode='live'``
    additionally requires the caller to vouch that the data is real (see ``fit_survival``).
    """
    observed = {
        c
        for r in records
        if (c := event_model.cause_of(r.terminal_mode)) is not None and c != UNLABELED_CAUSE
    }
    return len(observed) >= 2


def fit_survival(
    records: Sequence[SurvivalRecord],
    event_model: EventModel | None = None,
    *,
    cif_mode: str | None = None,
    alpha: float = 0.05,
    fit_weibull: bool = True,
) -> SurvivalReport:
    """Fit the full survival report.

    ``cif_mode`` defaults to ``"synthetic"`` — the conservative, disclaimer-forcing label.
    ``"live"`` (the typed competing-risk CIF is backed by *real* per-cause labels) is never
    inferred automatically: it must be passed explicitly by a caller that vouches the data
    is real, and it is rejected unless the data actually has >= 2 real typed causes. This is
    what prevents a synthetic or single-cause source from masquerading as the typed moat.
    """
    model = event_model or EventModel.failure_as_event()
    if cif_mode is None:
        cif_mode = "synthetic"
    if cif_mode not in VALID_CIF_MODES:
        raise ValueError(f"cif_mode must be one of {VALID_CIF_MODES}, got {cif_mode!r}")
    if cif_mode == "live" and not can_claim_live_cif(records, model):
        raise ValueError(
            "cif_mode='live' requires >= 2 distinct real typed causes in the data; "
            "this source cannot substantiate a live typed competing-risk CIF "
            "(use the default 'synthetic')"
        )
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

"""Typer command-line interface: ``hazardloop fit | replay | control | doctor``.

``compare`` (anytime-valid sequential comparison) is intentionally a stub in this release
(deferred; see arXiv:2512.03109). The CLI is honest about data provenance: synthetic
backends are labelled, and a run with no observed events is reported as censoring-dominated
rather than presented as an informative survival curve.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated

import typer

from hazardloop.adapters.base import TrajectoryBackend
from hazardloop.adapters.mock import MockBackend
from hazardloop.adapters.swebench import SweSmithTrajectories
from hazardloop.controller import Controller
from hazardloop.core.nelson_aalen import nelson_aalen
from hazardloop.estimate import fit_survival, has_observed_events
from hazardloop.policy import HazardThresholdPolicy
from hazardloop.replay import ReplayEvaluator
from hazardloop.types import EventModel, SurvivalRecord

app = typer.Typer(
    name="hazardloop",
    help="Censoring-aware competing-risk survival analysis for LLM-agent trajectories.",
    no_args_is_help=True,
    add_completion=False,
)

BackendOpt = Annotated[str, typer.Option("--backend", "-b", help="mock | swe-smith")]
LimitOpt = Annotated[int, typer.Option("--limit", "-n", help="max runs to load")]
SeedOpt = Annotated[int, typer.Option("--seed", help="random seed")]
JsonOpt = Annotated[bool, typer.Option("--json", help="emit JSON instead of text")]


def _make_backend(name: str, limit: int, seed: int) -> TrajectoryBackend:
    if name == "mock":
        return MockBackend(n=limit, seed=seed)
    if name in {"swe-smith", "swe_smith", "swebench"}:
        return SweSmithTrajectories()
    raise typer.BadParameter(f"unknown backend {name!r}; choose 'mock' or 'swe-smith'")


def _load(backend: TrajectoryBackend, limit: int) -> list[SurvivalRecord]:
    return list(backend.load(limit=limit))


@app.command()
def doctor(backend: BackendOpt = "mock", limit: LimitOpt = 200, seed: SeedOpt = 0) -> None:
    """Report whether a backend is reachable and whether its data is synthetic."""
    status = _make_backend(backend, limit, seed).doctor()
    typer.echo(
        f"backend={status.name} available={status.available} "
        f"synthetic={status.is_synthetic}\n  {status.detail}"
    )


@app.command()
def fit(
    backend: BackendOpt = "mock",
    limit: LimitOpt = 500,
    seed: SeedOpt = 0,
    cif_mode: Annotated[
        str, typer.Option(help="live | synthetic (default: derived from the data)")
    ] = "",
    json_out: JsonOpt = False,
) -> None:
    """Fit KM / Nelson-Aalen / Aalen-Johansen CIF / Weibull from a backend."""
    records = _load(_make_backend(backend, limit, seed), limit)
    report = fit_survival(records, cif_mode=(cif_mode or None))
    mode = report.cif_mode
    observed = has_observed_events(report)
    km_final = float(report.km.survival[-1]) if report.km.survival.size else 1.0
    shape: float | None = report.weibull.shape if report.weibull else None
    causes: list[str] = sorted(report.cif.cif_by_cause) if report.cif else []
    if json_out:
        payload = {
            "backend": backend,
            "n_runs": report.n_runs,
            "event_model": report.event_model_name,
            "cif_mode": report.cif_mode,
            "observed_events": observed,
            "km_final_survival": km_final,
            "weibull_shape": shape,
            "cif_causes": causes,
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"runs={report.n_runs} event_model={report.event_model_name} cif_mode={mode}")
    if not observed:
        typer.echo(
            "  no events observed under this model: censoring-dominated, S(t) ≡ 1 (descriptive only)"
        )
        return
    typer.echo(f"  final KM survival = {km_final:.4f}")
    if shape is not None:
        regime = "early-mortality" if shape < 1 else "wear-out"
        typer.echo(f"  Weibull shape β = {shape:.3f} ({regime})")
    typer.echo(f"  CIF causes: {', '.join(causes)}")


@app.command()
def replay(
    backend: BackendOpt = "mock",
    limit: LimitOpt = 600,
    seed: SeedOpt = 0,
    json_out: JsonOpt = False,
) -> None:
    """Offline-replay decision quality (train/test split + cluster-bootstrap CIs)."""
    records = _load(_make_backend(backend, limit, seed), limit)
    report = ReplayEvaluator(EventModel.failure_as_event(), seed=seed).evaluate(records)
    m = report.test_metrics
    payload = {
        "backend": backend,
        "threshold": report.threshold,
        "threshold_selected_on": report.threshold_selected_on,
        "evaluated_on": report.evaluated_on,
        "n_train": report.n_train,
        "n_test": report.n_test,
        "premature_abort_rate": m.premature_abort_rate,
        "premature_abort_rate_ci": [
            report.premature_abort_rate_ci.lower,
            report.premature_abort_rate_ci.upper,
        ],
        "recall": m.recall,
        "saved_compute_fraction_all": m.saved_compute_fraction_all,
        "lead_time_median": m.lead_time_median,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"threshold={report.threshold:.3f} (selected on {report.threshold_selected_on}, "
        f"evaluated on {report.evaluated_on}; n_train={report.n_train} n_test={report.n_test})"
    )
    typer.echo(f"  premature-abort rate = {m.premature_abort_rate:.3f}")
    typer.echo(f"  recall               = {m.recall:.3f}")
    typer.echo(f"  saved-compute (all)  = {m.saved_compute_fraction_all:.3f}")
    typer.echo(f"  lead-time median     = {m.lead_time_median:.2f} steps")


@app.command()
def control(
    backend: BackendOpt = "mock",
    limit: LimitOpt = 500,
    seed: SeedOpt = 0,
    abort_threshold: Annotated[float, typer.Option(help="cumulative-hazard abort threshold")] = 1.0,
) -> None:
    """Show what the fail-closed controller would decide along the fitted hazard."""
    records = _load(_make_backend(backend, limit, seed), limit)
    na = nelson_aalen(records, EventModel.failure_as_event())
    ctrl = Controller(na, HazardThresholdPolicy(abort_threshold=abort_threshold))
    abort_step = ctrl.first_abort_step(max_step=float("inf"))
    typer.echo(
        f"abort_threshold={abort_threshold} (rule-based, not learned; fail-closed ABSTAIN on no estimate)"
    )
    if abort_step is None:
        typer.echo("  cumulative hazard never reaches the threshold within observed runs")
    else:
        typer.echo(f"  controller would first recommend ABORT at step {abort_step:.0f}")


@app.command()
def compare() -> None:
    """[v0.2 stub] Anytime-valid sequential comparison — not implemented in this release."""
    typer.echo(
        "compare (anytime-valid sequential comparison) is deferred to v0.2.\n"
        "It is intentionally not implemented here; the binary-correctness stopping problem\n"
        "is occupied by arXiv:2512.03109 (E-valuator), which this project cites rather than\n"
        "re-implements. Use `hazardloop replay` for offline decision-quality evaluation.",
        err=True,
    )
    raise typer.Exit(code=2)


def main(argv: Sequence[str] | None = None) -> None:
    app(args=list(argv) if argv is not None else None)


if __name__ == "__main__":
    app()

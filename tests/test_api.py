"""S5 public API tests."""

from __future__ import annotations

import pytest

import hazardloop
from hazardloop import EventModel, TerminationMode, fit_survival, has_observed_events
from hazardloop.adapters.mock import synthetic_competing_risks
from hazardloop.types import SurvivalRecord


def test_version_and_public_symbols() -> None:
    assert hazardloop.__version__ == "0.1.0a1"
    for name in hazardloop.__all__:
        assert hasattr(hazardloop, name), name


def test_fit_survival_full_report() -> None:
    recs = synthetic_competing_risks(500, seed=0)
    report = fit_survival(recs, cif_mode="synthetic")
    assert report.n_runs == 500
    assert report.cif_mode == "synthetic"
    assert report.km.times.size > 0
    assert report.cif is not None
    assert report.weibull is not None
    assert has_observed_events(report)
    # additivity still holds end to end
    import numpy as np

    np.testing.assert_allclose(report.cif.total_cif(), 1.0 - report.cif.overall_survival, atol=1e-9)


def test_fit_survival_all_censored_is_honest() -> None:
    recs = [
        SurvivalRecord(duration=float(t), terminal_mode=TerminationMode.SOLVED) for t in range(1, 6)
    ]
    report = fit_survival(recs, EventModel.failure_as_event())
    assert not has_observed_events(report)
    assert report.weibull is None  # no events -> no Weibull fabricated
    assert report.cif is None


def test_fit_survival_rejects_bad_cif_mode() -> None:
    recs = synthetic_competing_risks(20, seed=1)
    with pytest.raises(ValueError, match="cif_mode"):
        fit_survival(recs, cif_mode="real-ish")

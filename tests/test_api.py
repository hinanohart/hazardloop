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


def test_cif_mode_defaults_to_synthetic() -> None:
    # the conservative, disclaimer-forcing default — "live" is never inferred automatically
    recs = synthetic_competing_risks(50, seed=2)
    assert fit_survival(recs).cif_mode == "synthetic"


def test_live_cif_requires_real_typed_causes() -> None:
    from hazardloop.estimate import can_claim_live_cif

    # binary-only (real-narrow) data: a single 'unlabeled' cause cannot back a live CIF
    binary = [
        SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.UNLABELED),
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED),
    ]
    assert can_claim_live_cif(binary, EventModel.failure_as_event()) is False
    with pytest.raises(ValueError, match="live"):
        fit_survival(binary, cif_mode="live")
    # multi-cause data satisfies the necessary condition
    multi = synthetic_competing_risks(200, seed=3)
    assert can_claim_live_cif(multi, EventModel.failure_as_event()) is True


def test_deferred_backends_are_honest_stubs() -> None:
    import pytest as _pytest

    from hazardloop.adapters.stubs import OpenEnvBackend, VerifiersBackend

    for be in (VerifiersBackend(), OpenEnvBackend()):
        status = be.doctor()
        assert status.available is False
        assert status.is_synthetic is False
        with _pytest.raises(NotImplementedError):
            be.load()

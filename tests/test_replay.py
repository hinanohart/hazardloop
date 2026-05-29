"""S4 ReplayEvaluator tests: hand-computed metrics, split discipline, fork NON-CLAIM."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from hazardloop.adapters.mock import synthetic_competing_risks
from hazardloop.replay import (
    ReplayEvaluator,
    ReplayMetrics,
    ReplayReport,
    decision_curve,
    evaluate_policy,
    train_test_split,
)
from hazardloop.types import EventModel, NAResult, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()


def _manual_na() -> NAResult:
    """Cumulative hazard 0.3 at t=2 and 0.8 at t=5 (crosses 0.5 at t=5)."""
    times = np.array([2.0, 5.0])
    return NAResult(
        times=times,
        cumulative_hazard=np.array([0.3, 0.8]),
        variance=np.array([0.01, 0.04]),
        hazard_increment=np.array([0.3, 0.5]),
        n_at_risk=np.array([4, 2], dtype=np.int64),
        n_events=np.array([1, 1], dtype=np.int64),
    )


def test_evaluate_policy_hand_computed() -> None:
    na = _manual_na()
    records = [
        SurvivalRecord(
            duration=8.0, terminal_mode=TerminationMode.WRONG_PATCH
        ),  # fail, abort@5 -> TP, lead 3
        SurvivalRecord(duration=4.0, terminal_mode=TerminationMode.WRONG_PATCH),  # fail, 5>=4 -> FN
        SurvivalRecord(
            duration=10.0, terminal_mode=TerminationMode.SOLVED
        ),  # success, abort@5 -> FP
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED),  # success, 5>=3 -> TN
    ]
    m = evaluate_policy(records, na, threshold=0.5, event_model=MODE_A)
    assert (m.true_pos, m.false_pos, m.false_neg, m.true_neg) == (1, 1, 1, 1)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.premature_abort_rate == pytest.approx(0.5)
    assert m.lead_time_median == pytest.approx(3.0)
    assert m.saved_compute_fraction_all == pytest.approx((3.0 + 5.0) / 25.0)
    assert m.saved_compute_fraction_failures == pytest.approx(3.0 / 12.0)


def test_evaluate_policy_undefined_metrics_are_nan() -> None:
    na = _manual_na()
    # all successes -> recall undefined (no failures), precision undefined (no positives if none abort)
    recs = [SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED)]
    m = evaluate_policy(recs, na, threshold=0.5, event_model=MODE_A)
    assert math.isnan(m.recall)


def test_train_test_split_clusters_do_not_cross() -> None:
    recs = [
        SurvivalRecord(
            duration=float(i % 7 + 1),
            terminal_mode=TerminationMode.WRONG_PATCH,
            cluster=f"c{i % 6}",
        )
        for i in range(120)
    ]
    train, test = train_test_split(recs, test_frac=0.5, seed=1)
    assert train and test
    train_clusters = {r.cluster for r in train}
    test_clusters = {r.cluster for r in test}
    assert train_clusters.isdisjoint(test_clusters)


def test_train_test_split_deterministic() -> None:
    recs = synthetic_competing_risks(200, seed=2)
    a = train_test_split(recs, test_frac=0.4, seed=5)
    b = train_test_split(recs, test_frac=0.4, seed=5)
    assert [r.run_id for r in a[0]] == [r.run_id for r in b[0]]


def test_decision_curve_is_well_formed() -> None:
    na = _manual_na()
    recs = [
        SurvivalRecord(duration=8.0, terminal_mode=TerminationMode.WRONG_PATCH),
        SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED),
    ]
    curve = decision_curve(recs, na, MODE_A, thresholds=[0.2, 0.5, 1.0])
    assert len(curve) == 3
    for pt in curve:
        assert 0.0 <= pt.threshold_probability < 1.0
        assert math.isfinite(pt.net_benefit)


def test_replay_evaluator_split_discipline() -> None:
    recs = synthetic_competing_risks(600, seed=4)
    report = ReplayEvaluator(MODE_A, seed=0, n_boot=300).evaluate(recs)
    assert isinstance(report, ReplayReport)
    assert report.threshold_selected_on == "train"
    assert report.evaluated_on == "test"
    assert report.n_train > 0 and report.n_test > 0
    # CIs bracket their point estimates where defined
    for ci in (report.premature_abort_rate_ci, report.recall_ci, report.saved_compute_fraction_ci):
        if math.isfinite(ci.point):
            assert ci.lower <= ci.point <= ci.upper


def test_fork_rescue_rate_is_never_reported() -> None:
    # NON-CLAIM: no field of any replay output mentions fork or rescue.
    for cls in (ReplayMetrics, ReplayReport):
        names = {f.name.lower() for f in dataclasses.fields(cls)}
        assert not any("fork" in n or "rescue" in n for n in names), names

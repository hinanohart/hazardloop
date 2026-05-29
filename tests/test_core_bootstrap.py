"""S2 cluster-bootstrap tests: determinism, point coverage, degenerate fallback, clustering."""

from __future__ import annotations

import numpy as np

from hazardloop.core.bootstrap import cluster_bootstrap_ci
from hazardloop.core.kaplan_meier import kaplan_meier
from hazardloop.types import EventModel, SurvivalRecord, TerminationMode

MODE_A = EventModel.failure_as_event()


def _median_duration(records: list[SurvivalRecord]) -> float:
    return float(np.median([r.duration for r in records]))


def _km_at_5(records: list[SurvivalRecord]) -> float:
    from hazardloop.core.kaplan_meier import survival_at

    return survival_at(kaplan_meier(records, MODE_A), 5.0)


def _sample(n: int, seed: int, with_cluster: bool = False) -> list[SurvivalRecord]:
    rng = np.random.default_rng(seed)
    out: list[SurvivalRecord] = []
    for i in range(n):
        out.append(
            SurvivalRecord(
                duration=float(rng.integers(1, 20)),
                terminal_mode=rng.choice([TerminationMode.WRONG_PATCH, TerminationMode.SOLVED]),
                cluster=(f"m{i % 4}" if with_cluster else None),
            )
        )
    return out


def test_determinism_same_seed() -> None:
    recs = _sample(120, seed=5)
    a = cluster_bootstrap_ci(recs, _median_duration, n_boot=500, seed=42)
    b = cluster_bootstrap_ci(recs, _median_duration, n_boot=500, seed=42)
    assert (a.lower, a.upper, a.point) == (b.lower, b.upper, b.point)


def test_point_within_interval() -> None:
    recs = _sample(200, seed=6)
    ci = cluster_bootstrap_ci(recs, _median_duration, n_boot=800, seed=1, method="percentile")
    assert ci.lower <= ci.point <= ci.upper


def test_bca_runs_and_brackets_point() -> None:
    recs = _sample(200, seed=8, with_cluster=True)
    ci = cluster_bootstrap_ci(recs, _km_at_5, n_boot=800, seed=2, method="bca")
    assert ci.method in {"bca", "percentile"}  # may fall back if degenerate
    assert ci.lower <= ci.point <= ci.upper
    assert 0.0 <= ci.lower <= 1.0 and 0.0 <= ci.upper <= 1.0


def test_degenerate_constant_statistic_falls_back() -> None:
    # a statistic that is constant on every resample -> z0 undefined -> percentile fallback,
    # and the interval collapses to the point.
    recs = _sample(50, seed=9)
    ci = cluster_bootstrap_ci(recs, lambda r: 3.0, n_boot=200, seed=3, method="bca")
    assert ci.point == 3.0
    assert ci.lower == 3.0 and ci.upper == 3.0
    assert ci.method == "percentile"


def test_clustered_interval_is_wider_than_naive() -> None:
    # Inject strong within-cluster correlation: each cluster is internally identical.
    rng = np.random.default_rng(13)
    clustered: list[SurvivalRecord] = []
    flat: list[SurvivalRecord] = []
    for c in range(20):
        d = float(rng.integers(1, 20))
        mode = rng.choice([TerminationMode.WRONG_PATCH, TerminationMode.SOLVED])
        for _ in range(10):
            clustered.append(SurvivalRecord(duration=d, terminal_mode=mode, cluster=f"c{c}"))
            flat.append(SurvivalRecord(duration=d, terminal_mode=mode, cluster=None))
    ci_cluster = cluster_bootstrap_ci(
        clustered, _median_duration, n_boot=600, seed=7, method="percentile"
    )
    ci_flat = cluster_bootstrap_ci(flat, _median_duration, n_boot=600, seed=7, method="percentile")
    # resampling whole clusters injects more variability than resampling 200 "independent" rows
    assert (ci_cluster.upper - ci_cluster.lower) >= (ci_flat.upper - ci_flat.lower)

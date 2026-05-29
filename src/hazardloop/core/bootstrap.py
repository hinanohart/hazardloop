"""Cluster bootstrap confidence intervals.

Sequential agent runs are not independent: runs from the same model / harness share
failure tendencies, so the Greenwood / Aalen variances (which assume independence) can
under-cover. Resampling whole *clusters* (model or harness id) with replacement keeps the
within-cluster dependence intact and restores honest coverage.

Two interval methods: ``percentile`` and bias-corrected-and-accelerated (``bca``). BCa
uses a leave-one-cluster-out jackknife for the acceleration term and falls back to the
percentile interval whenever the correction is numerically degenerate (e.g. a single
cluster, or all replicates equal).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from hazardloop.types import SurvivalRecord

Statistic = Callable[[Sequence[SurvivalRecord]], float]


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    alpha: float
    method: str
    n_boot: int
    n_effective: int  # finite replicates actually used


def _group_clusters(records: Sequence[SurvivalRecord]) -> list[list[SurvivalRecord]]:
    """Group by ``cluster``; records with ``cluster is None`` each form their own cluster
    (i.e. ordinary record-level bootstrap when no clustering is supplied)."""
    named: dict[str, list[SurvivalRecord]] = {}
    singletons: list[list[SurvivalRecord]] = []
    for r in records:
        if r.cluster is None:
            singletons.append([r])
        else:
            named.setdefault(r.cluster, []).append(r)
    return list(named.values()) + singletons


def cluster_bootstrap_ci(
    records: Sequence[SurvivalRecord],
    statistic: Statistic,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: int = 0,
) -> BootstrapCI:
    if len(records) == 0:
        raise ValueError("cluster_bootstrap_ci requires at least one record")
    if method not in {"percentile", "bca"}:
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    point = float(statistic(records))
    clusters = _group_clusters(records)
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)

    replicates: list[float] = []
    idx = np.arange(n_clusters)
    for _ in range(n_boot):
        chosen = rng.choice(idx, size=n_clusters, replace=True)
        resampled: list[SurvivalRecord] = []
        for ci in chosen:
            resampled.extend(clusters[ci])
        try:
            val = float(statistic(resampled))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(val):
            replicates.append(val)

    boot = np.asarray(replicates, dtype=np.float64)
    if boot.size == 0:
        return BootstrapCI(point, point, point, alpha, method, n_boot, 0)

    if method == "percentile":
        lo, hi = _percentile_interval(boot, alpha)
        return BootstrapCI(point, lo, hi, alpha, "percentile", n_boot, boot.size)

    lo, hi, used = _bca_interval(boot, point, clusters, statistic, alpha)
    return BootstrapCI(point, lo, hi, alpha, used, n_boot, boot.size)


def _percentile_interval(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def _bca_interval(
    boot: np.ndarray,
    point: float,
    clusters: list[list[SurvivalRecord]],
    statistic: Statistic,
    alpha: float,
) -> tuple[float, float, str]:
    n_clusters = len(clusters)
    prop_less = float(np.mean(boot < point))
    # z0 undefined at proportions 0 or 1 -> fall back to percentile.
    if prop_less <= 0.0 or prop_less >= 1.0 or n_clusters < 2:
        lo, hi = _percentile_interval(boot, alpha)
        return lo, hi, "percentile"
    z0 = float(norm.ppf(prop_less))

    # leave-one-cluster-out jackknife for acceleration
    jack: list[float] = []
    for i in range(n_clusters):
        subset: list[SurvivalRecord] = []
        for j, cl in enumerate(clusters):
            if j != i:
                subset.extend(cl)
        try:
            jack.append(float(statistic(subset)))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            jack.append(np.nan)
    jack_arr = np.asarray(jack, dtype=np.float64)
    if not np.all(np.isfinite(jack_arr)):
        lo, hi = _percentile_interval(boot, alpha)
        return lo, hi, "percentile"

    jack_mean = float(np.mean(jack_arr))
    diff = jack_mean - jack_arr
    denom = 6.0 * (float(np.sum(diff**2)) ** 1.5)
    if denom == 0.0:
        lo, hi = _percentile_interval(boot, alpha)
        return lo, hi, "percentile"
    accel = float(np.sum(diff**3)) / denom

    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))
    a1 = _bca_alpha(z0, z_lo, accel)
    a2 = _bca_alpha(z0, z_hi, accel)
    if not (np.isfinite(a1) and np.isfinite(a2)) or not (0.0 < a1 < a2 < 1.0):
        lo, hi = _percentile_interval(boot, alpha)
        return lo, hi, "percentile"
    return float(np.quantile(boot, a1)), float(np.quantile(boot, a2)), "bca"


def _bca_alpha(z0: float, z: float, accel: float) -> float:
    adj = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
    return float(norm.cdf(adj))

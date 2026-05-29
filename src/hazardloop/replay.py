"""Offline-replay decision-quality evaluation (the headline use case).

A fail-closed abort policy is applied *counterfactually* to logged runs — no agent is
re-executed. For each run we ask: would the policy, reading the fitted cumulative hazard,
have aborted before the run ended? From that we derive:

- lead time to failure  : steps between a correct early abort and the run's actual end;
- premature-abort rate   : fraction of successful runs the policy would have killed (FP);
- saved-compute fraction : steps not spent because doomed runs were aborted early;
- precision / recall      : of "abort" as a predictor of "this run fails";
- a decision curve        : net benefit across operating thresholds (Vickers-Elkin).

To avoid optimistic, in-sample thresholds (a hindsight R5 violation), the abort threshold
is selected on a TRAIN split and every reported number is computed on a DISJOINT TEST
split, with clusters (model / harness) never crossing the split boundary (bootstrap
protocol BP2). Confidence intervals come from a cluster bootstrap of the test runs with
the train-derived threshold held fixed.

The ``fork`` decision's rescue outcome is unobserved in logged data and is therefore
**never computed or reported** here (NON-CLAIM).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hazardloop.core.bootstrap import BootstrapCI, cluster_bootstrap_ci
from hazardloop.core.nelson_aalen import nelson_aalen
from hazardloop.types import EventModel, NAResult, SurvivalRecord


@dataclass(frozen=True)
class ReplayMetrics:
    threshold: float
    n_runs: int
    n_failures: int
    n_success: int
    n_aborted: int
    true_pos: int
    false_pos: int
    false_neg: int
    true_neg: int
    precision: float
    recall: float
    premature_abort_rate: float
    lead_time_median: float
    lead_time_mean: float
    saved_compute_fraction_all: float
    saved_compute_fraction_failures: float


@dataclass(frozen=True)
class DecisionCurvePoint:
    threshold: float
    threshold_probability: float
    net_benefit: float


@dataclass(frozen=True)
class ReplayReport:
    threshold: float
    threshold_selected_on: str
    evaluated_on: str
    n_train: int
    n_test: int
    test_metrics: ReplayMetrics
    premature_abort_rate_ci: BootstrapCI
    recall_ci: BootstrapCI
    saved_compute_fraction_ci: BootstrapCI


def train_test_split(
    records: Sequence[SurvivalRecord], *, test_frac: float = 0.5, seed: int = 0
) -> tuple[list[SurvivalRecord], list[SurvivalRecord]]:
    """Split records so that whole clusters land on a single side (no cluster crosses the
    boundary). Records without a cluster are split individually."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    groups: dict[str, list[SurvivalRecord]] = {}
    for i, r in enumerate(records):
        key = r.cluster if r.cluster is not None else f"__row_{i}"
        groups.setdefault(key, []).append(r)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_total = len(records)
    target_test = test_frac * n_total
    test: list[SurvivalRecord] = []
    train: list[SurvivalRecord] = []
    for k in keys:
        if len(test) < target_test:
            test.extend(groups[k])
        else:
            train.extend(groups[k])
    if not train or not test:  # degenerate (e.g. one giant cluster): fall back to row split
        flat = list(records)
        cut = max(1, round((1.0 - test_frac) * n_total))
        train, test = flat[:cut], flat[cut:]
    return train, test


def _abort_time(na: NAResult, threshold: float) -> float | None:
    """First event time at which cumulative hazard reaches ``threshold`` (monotone)."""
    if na.times.size == 0:
        return None
    idx = int(np.searchsorted(na.cumulative_hazard, threshold, side="left"))
    if idx >= na.times.size:
        return None
    return float(na.times[idx])


def evaluate_policy(
    records: Sequence[SurvivalRecord],
    na: NAResult,
    threshold: float,
    event_model: EventModel,
) -> ReplayMetrics:
    """Counterfactually apply the abort threshold to ``records`` using ``na``'s hazard."""
    if len(records) == 0:
        raise ValueError("evaluate_policy requires at least one record")
    abort_t = _abort_time(na, threshold)

    tp = fp = fn = tn = 0
    lead_times: list[float] = []
    saved_all = 0.0
    saved_fail = 0.0
    total_dur = 0.0
    total_dur_fail = 0.0

    for r in records:
        fail = event_model.is_event(r.terminal_mode)
        total_dur += r.duration
        if fail:
            total_dur_fail += r.duration
        aborted = abort_t is not None and abort_t < r.duration
        if aborted:
            assert abort_t is not None
            saved = r.duration - abort_t
            saved_all += saved
            if fail:
                tp += 1
                lead_times.append(saved)
                saved_fail += saved
            else:
                fp += 1
        else:
            if fail:
                fn += 1
            else:
                tn += 1

    n_fail = tp + fn
    n_success = fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / n_fail if n_fail > 0 else math.nan
    premature = fp / n_success if n_success > 0 else math.nan
    lead_arr = np.asarray(lead_times, dtype=np.float64)

    return ReplayMetrics(
        threshold=threshold,
        n_runs=len(records),
        n_failures=n_fail,
        n_success=n_success,
        n_aborted=tp + fp,
        true_pos=tp,
        false_pos=fp,
        false_neg=fn,
        true_neg=tn,
        precision=precision,
        recall=recall,
        premature_abort_rate=premature,
        lead_time_median=float(np.median(lead_arr)) if lead_arr.size else math.nan,
        lead_time_mean=float(np.mean(lead_arr)) if lead_arr.size else math.nan,
        saved_compute_fraction_all=saved_all / total_dur if total_dur > 0 else 0.0,
        saved_compute_fraction_failures=saved_fail / total_dur_fail if total_dur_fail > 0 else 0.0,
    )


def decision_curve(
    records: Sequence[SurvivalRecord],
    na: NAResult,
    event_model: EventModel,
    thresholds: Sequence[float],
) -> list[DecisionCurvePoint]:
    """Net benefit across operating thresholds.

    A cumulative-hazard threshold τ implies a failure probability ``p = 1 - exp(-τ)`` and a
    cost-ratio weight ``w = p / (1 - p)``. Net benefit = (TP - w·FP) / n (Vickers-Elkin).
    """
    n = len(records)
    if n == 0:
        raise ValueError("decision_curve requires at least one record")
    out: list[DecisionCurvePoint] = []
    for tau in thresholds:
        m = evaluate_policy(records, na, tau, event_model)
        p = 1.0 - math.exp(-tau)
        w = p / (1.0 - p) if p < 1.0 else math.inf
        nb = (m.true_pos - w * m.false_pos) / n if math.isfinite(w) else -math.inf
        out.append(DecisionCurvePoint(threshold=tau, threshold_probability=p, net_benefit=nb))
    return out


def select_abort_threshold(
    train: Sequence[SurvivalRecord],
    na_train: NAResult,
    event_model: EventModel,
    candidate_thresholds: Sequence[float],
) -> float:
    """Pick the threshold maximising net benefit on the TRAIN split (ties -> larger
    threshold, i.e. the more conservative / fewer-abort choice)."""
    if len(candidate_thresholds) == 0:
        raise ValueError("need at least one candidate threshold")
    curve = decision_curve(train, na_train, event_model, candidate_thresholds)
    best = max(curve, key=lambda pt: (pt.net_benefit, pt.threshold))
    return best.threshold


class ReplayEvaluator:
    """Train/test-split offline-replay evaluation with cluster-bootstrap confidence intervals."""

    def __init__(
        self,
        event_model: EventModel,
        *,
        candidate_thresholds: Sequence[float] | None = None,
        test_frac: float = 0.5,
        seed: int = 0,
        n_boot: int = 2000,
    ) -> None:
        self._event_model = event_model
        self._candidates = list(
            candidate_thresholds if candidate_thresholds is not None else np.linspace(0.1, 3.0, 30)
        )
        self._test_frac = test_frac
        self._seed = seed
        self._n_boot = n_boot

    def evaluate(self, records: Sequence[SurvivalRecord]) -> ReplayReport:
        if len(records) < 2:
            raise ValueError("ReplayEvaluator.evaluate needs at least two records to split")
        train, test = train_test_split(records, test_frac=self._test_frac, seed=self._seed)
        na_train = nelson_aalen(train, self._event_model)
        threshold = select_abort_threshold(train, na_train, self._event_model, self._candidates)
        metrics = evaluate_policy(test, na_train, threshold, self._event_model)

        def _premature(rs: Sequence[SurvivalRecord]) -> float:
            return evaluate_policy(rs, na_train, threshold, self._event_model).premature_abort_rate

        def _recall(rs: Sequence[SurvivalRecord]) -> float:
            return evaluate_policy(rs, na_train, threshold, self._event_model).recall

        def _saved(rs: Sequence[SurvivalRecord]) -> float:
            return evaluate_policy(
                rs, na_train, threshold, self._event_model
            ).saved_compute_fraction_all

        return ReplayReport(
            threshold=threshold,
            threshold_selected_on="train",
            evaluated_on="test",
            n_train=len(train),
            n_test=len(test),
            test_metrics=metrics,
            premature_abort_rate_ci=cluster_bootstrap_ci(
                test, _premature, n_boot=self._n_boot, seed=self._seed, method="percentile"
            ),
            recall_ci=cluster_bootstrap_ci(
                test, _recall, n_boot=self._n_boot, seed=self._seed, method="percentile"
            ),
            saved_compute_fraction_ci=cluster_bootstrap_ci(
                test, _saved, n_boot=self._n_boot, seed=self._seed, method="percentile"
            ),
        )

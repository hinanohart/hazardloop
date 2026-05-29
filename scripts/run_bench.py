#!/usr/bin/env python3
"""Generate env-stamped benchmark results (the single source of every README number).

Writes two files under bench_results/:
- ``real_survival.json``  : REAL gate-real. Kaplan-Meier / Nelson-Aalen on real
  SWE-smith-trajectories (binary outcomes, mode-A), the logistic-vs-KM divergence
  (NON-CLAIM), and offline-replay decision quality. ``dataset.mode == "live"``.
- ``synthetic_cif.json``  : the typed multi-cause Aalen-Johansen CIF demonstration on the
  deterministic synthetic generator. ``cif_mode == "synthetic"`` -> forces the BP1
  disclaimer in the README.

IMPORTANT: the ``datasets`` streaming loader segfaults at *interpreter exit* (a known
pyarrow/GIL teardown issue) AFTER all work completes. Both files are written before return,
so verify success by FILE EXISTENCE + CONTENT, never by this process's exit code.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hazardloop import __version__  # noqa: E402
from hazardloop.adapters.mock import synthetic_competing_risks  # noqa: E402
from hazardloop.adapters.swebench import SweSmithTrajectories  # noqa: E402
from hazardloop.divergence import logistic_vs_km_divergence  # noqa: E402
from hazardloop.estimate import fit_survival  # noqa: E402
from hazardloop.replay import ReplayEvaluator  # noqa: E402
from hazardloop.types import EventModel  # noqa: E402

MODE_A = EventModel.failure_as_event()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _env_stamp(n: int, mode: str, source: str, seed: int) -> dict[str, object]:
    return {
        "n": n,
        "mode": mode,
        "source": source,
        "seed": seed,
        "version": __version__,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "date": datetime.now(UTC).date().isoformat(),
    }


def synthetic_bench(n: int = 1500, seed: int = 0) -> dict[str, object]:
    recs = synthetic_competing_risks(n, seed=seed)
    report = fit_survival(recs, MODE_A, cif_mode="synthetic")
    assert report.cif is not None and report.weibull is not None
    cif_final = {c: float(curve[-1]) for c, curve in report.cif.cif_by_cause.items()}
    return {
        "dataset": _env_stamp(n, "synthetic", "synthetic_competing_risks", seed),
        "cif_mode": "synthetic",
        "km_final_survival": float(report.km.survival[-1]) if report.km.survival.size else 1.0,
        "n_event_times": int(report.km.times.size),
        "weibull_shape": report.weibull.shape,
        "weibull_scale": report.weibull.scale,
        "cif_final_by_cause": cif_final,
        "cif_additivity_max_error": float(
            max(abs(report.cif.total_cif() - (1.0 - report.cif.overall_survival)))
        ),
        "note": "typed multi-cause Aalen-Johansen CIF validated on synthetic data only",
    }


def real_bench(limit: int = 400, seed: int = 0) -> dict[str, object]:
    import dataclasses

    backend = SweSmithTrajectories()
    raw = list(backend.load(limit=limit))
    # SWE-smith trajectories all share one model -> a single cluster collapses the bootstrap.
    # Re-cluster by repository (the instance/traj id prefix), the natural correlated unit,
    # so the cluster bootstrap CI is informative. This is honest: difficulty correlates
    # within a repo. cluster = run_id up to the first '.'
    recs = [dataclasses.replace(r, cluster=(r.run_id or "unknown").split(".")[0]) for r in raw]
    from hazardloop.replay import train_test_split

    n_clusters = len({r.cluster for r in recs})
    _train, _test = train_test_split(recs, test_frac=0.5, seed=seed)
    n_test_clusters = len({r.cluster for r in _test})
    n = len(recs)
    n_solved = sum(1 for r in recs if r.terminal_mode.value == "solved")
    report = fit_survival(recs, MODE_A, cif_mode="synthetic")  # binary -> no real typed CIF
    km = report.km
    div = logistic_vs_km_divergence(recs, MODE_A)
    replay = ReplayEvaluator(MODE_A, seed=seed, n_boot=1000).evaluate(recs)
    m = replay.test_metrics
    return {
        "dataset": _env_stamp(n, "live", backend.name, seed),
        "n_solved": n_solved,
        "n_unresolved": n - n_solved,
        "cif_mode": "synthetic",
        "km_final_survival": float(km.survival[-1]) if km.survival.size else 1.0,
        "km_n_event_times": int(km.times.size),
        "median_survival_step": _median_survival_step(km),
        "logistic_vs_km": {
            "mean_abs_divergence": div.mean_abs_divergence,
            "max_abs_divergence": div.max_abs_divergence,
            "note": "two estimators on one dataset can diverge; NOT a claim METR is wrong",
        },
        "replay": {
            "threshold": replay.threshold,
            "threshold_selected_on": replay.threshold_selected_on,
            "evaluated_on": replay.evaluated_on,
            "n_clusters": n_clusters,
            "n_test_clusters": n_test_clusters,
            "n_train": replay.n_train,
            "n_test": replay.n_test,
            "premature_abort_rate": m.premature_abort_rate,
            "premature_abort_rate_ci": [
                replay.premature_abort_rate_ci.lower,
                replay.premature_abort_rate_ci.upper,
            ],
            "recall": m.recall,
            "saved_compute_fraction_all": m.saved_compute_fraction_all,
            "lead_time_median": m.lead_time_median,
        },
    }


def _median_survival_step(km: object) -> float | None:
    import numpy as np

    times = km.times  # type: ignore[attr-defined]
    surv = km.survival  # type: ignore[attr-defined]
    below = np.where(surv <= 0.5)[0]
    return float(times[below[0]]) if below.size else None


def main() -> int:
    bench_dir = ROOT / "bench_results"
    bench_dir.mkdir(exist_ok=True)

    syn = synthetic_bench()
    (bench_dir / "synthetic_cif.json").write_text(json.dumps(syn, indent=2, default=str))
    print("BENCH-WROTE:", bench_dir / "synthetic_cif.json")

    real = real_bench()
    (bench_dir / "real_survival.json").write_text(json.dumps(real, indent=2, default=str))
    print("BENCH-WROTE:", bench_dir / "real_survival.json")
    print("BENCH-DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

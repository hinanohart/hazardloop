# Changelog

All notable changes to hazardloop are documented here.
This project adheres to [Semantic Versioning](https://semver.org/) (with PEP 440 pre-release tags).

## [0.1.0a3] — pre-alpha (post-/compact consistency audit)

### Fixed
- Stale version strings left behind by the previous release bump: the README status badge
  and the `compare` CLI help / deferred-backend docstrings still read `0.1.0a1`. Source and
  README now carry no hardcoded release number except the canonical `__version__` (prose
  refers to "this release").
- `fit_logistic_success` now guards a (near-)constant duration vector: a tiny but non-zero
  standard deviation no longer blows up the centred/scaled design.
- `decision_curve` clamps the implied failure probability strictly below 1, so a very large
  cumulative-hazard threshold no longer drives every net benefit to `-inf` (identity over
  the usual operating range).

### Added
- `check_honest_marketing.py` now enforces version consistency: every PEP 440 pre-release
  literal in `src/hazardloop/**/*.py` and `README.md` must equal `__version__` (paired
  positive/negative tests), so a stale leftover version can no longer ship silently.

## [0.1.0a2] — pre-alpha (post-publish audit patch)

### Fixed
- README install instructions now use a from-source command (the package is not yet on
  PyPI); the headline `pip install hazardloop` was unreachable.
- README reproducibility note explains that `scripts/run_bench.py` exits non-zero due to a
  `datasets`/pyarrow teardown segfault (files are written first); verify by file existence.
- `synthetic_competing_risks` default `n_clusters` raised 4 → 20 so `replay --backend mock`
  produces a non-degenerate cluster-bootstrap CI (headline real-data numbers unchanged).
- `weibull_aft` now raises on fewer than two distinct event times instead of silently
  returning a diverged shape (β→∞).

## [0.1.0a1] — pre-alpha (initial GitHub pre-release)

### Added
- Survival core (numpy, from scratch): Kaplan-Meier + Greenwood variance + complementary
  log-log confidence band; Nelson-Aalen cumulative hazard; Aalen-Johansen competing-risk
  cumulative incidence function; Weibull AFT shape-parameter MLE; cluster bootstrap CI.
- Intermediate representation: `StepRecord`, `SurvivalRecord`, `TerminationMode`,
  `ControlDecision`, `SurvivalReport`.
- `TrajectoryBackend` protocol with a deterministic synthetic `mock` adapter and a
  read-only `SWE-smith-trajectories` adapter (optional `[data]` extra).
- `ReplayEvaluator`: offline counterfactual decision-quality metrics (lead-time,
  premature-abort rate, saved-compute fraction, per-cause precision/recall, decision
  curve) with train/test split separation and cluster-bootstrap CIs.
- Fail-closed `ControlPolicy` interface and a reference hazard-threshold policy
  (rule-based, not learned).
- Typer CLI: `hazardloop fit | replay | control` (`compare` is a v0.2 stub).
- CI enforcement: GPL-import check, honest-marketing check (banned-phrase / required
  disclaimer / placeholder / METR-assertion), with paired positive/negative tests.

### Notes
- `compare` (anytime-valid sequential comparison) is a stub; deferred (see arXiv:2512.03109).
- Live middleware benefit is UNVERIFIED; deferred to a later release.

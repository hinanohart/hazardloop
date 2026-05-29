# hazardloop — CLAIMS and NON-CLAIMS

This file is the single source of truth for what hazardloop does and does not assert.
It is checked mechanically (`scripts/check_honest_marketing.py`, run in CI). The
required disclaimer strings below are asserted to be **present**; the banned phrases
are asserted to be **absent** from README / docs.

---

## CLAIM (falsifiable)

1. **Censoring-aware survival estimation on real agent trajectories.**
   Given a set of logged agent runs with a per-run duration (default: step count derived
   from the trajectory) and a terminal outcome, hazardloop estimates a Kaplan-Meier
   survival curve (with Greenwood variance and a complementary-log-log confidence band)
   and a Nelson-Aalen cumulative hazard, treating the chosen outcome as the event and the
   complementary outcome as right-censoring under an explicit, declared modelling mode
   (`mode-A` = failure-as-event, success-as-censoring; `mode-B` = completion-as-event).
   Falsifiable: the numbers are reproducible from the published `bench_results/*.json`
   and cross-checked against `lifelines` to numerical tolerance.

2. **Typed competing-risk cumulative incidence via Aalen-Johansen.**
   When per-cause terminal labels are available, hazardloop estimates a per-cause
   cumulative incidence function with the Aalen-Johansen estimator, which avoids the
   systematic over-estimation produced by the naive `1 - cause-specific Kaplan-Meier`
   under competing risks (Putter, Fiocco & Geskus, Stat. Med. 2007). Additivity
   `sum_c CIF_c(t) = 1 - S(t)` is asserted as a numerical invariant.

3. **Divergence between a censoring-blind static logistic horizon and KM/CIF.**
   On the same data, a static logistic success-probability fit (the METR-style horizon
   methodology) and the censoring-aware KM / competing-risk estimates **can diverge**,
   and the magnitude is reported with a bootstrap confidence interval. This is a
   statement about two estimators on the same data; it is **not** a statement that the
   METR methodology is incorrect.

4. **Offline-replay decision quality.**
   A fail-closed control policy parameterised by a survival/hazard threshold is applied
   counterfactually to logged runs (no agent re-execution). hazardloop reports lead-time
   to failure, premature-abort rate, saved-compute fraction, and per-cause precision /
   recall, each with a cluster-bootstrap confidence interval, with the abort threshold
   selected on a train split and evaluated on a disjoint test split.

---

## NON-CLAIM (explicitly out of scope; live-system benefit is UNVERIFIED)

- hazardloop is an **estimator plus a policy interface**; it is **not a trained controller**.
  Any benefit to a live agent is **UNVERIFIED** in this release: hazardloop does not run,
  retrain, or re-execute agents, and reports no live A/B result.
- hazardloop makes **no assertion about agent capability, task-completion gains, or pass@k**.
  Offline-replay numbers are counterfactual estimates on logged runs, not live trials.
- The `fork` decision's rescue outcome is counterfactual and unobserved, so its rescue
  rate is **not computed and not reported**; only fork firing-timing and checkpoint cost
  are measured.
- hazardloop performs **no tail / return-level extrapolation** (extreme-value theory is
  deliberately not used: the i.i.d./stationarity assumptions break for sequential agent
  failures, per arXiv:2511.02927).
- hazardloop is **not** a verifier, a reward auditor, or a formal safety guarantee.
- Anytime-valid sequential comparison is **not** in this release (see arXiv:2512.03109,
  E-valuator, which occupies the binary-correctness stopping problem); it is deferred.
- When per-cause labels are not available from the data, the typed competing-risk CIF is
  produced from **synthetic validation only** and is labelled as such; the real-trajectory
  typed CIF is deferred to a later release.

---

## Prior art this release builds on and differentiates from

- arXiv:2509.02360 (offline PRM-score replay) — scalar reward replay without censoring or
  typed competing risks. hazardloop's offline-replay consumes the typed survival core and
  is the evaluation face, not the headline.
- arXiv:2512.03109 (E-valuator) — anytime-valid testing for binary correctness; deferred,
  cited, not re-implemented.
- METR time-horizon methodology (metr.org) — logistic fit on binary success without
  censoring or competing-risk handling.

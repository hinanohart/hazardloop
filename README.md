# hazardloop

**Censoring-aware competing-risk survival analysis for long-horizon LLM-agent trajectories.**

Long-horizon agent evaluation usually collapses a run into a binary success/failure and
fits a static logistic curve to it. That throws away two things survival analysis has
handled for a century: **censoring** (a run cut off by a timeout or budget is not the same
as an observed failure) and **competing risks** (a wrong patch, a tool error, an infinite
loop, and a budget exhaustion are different terminal events, not one). hazardloop brings
Kaplan-Meier, Nelson-Aalen, the **Aalen-Johansen competing-risk cumulative incidence
function**, and Weibull AFT to agent trajectories — implemented from scratch on numpy,
Apache-2.0 clean, CPU-only, no GPU.

> **Status: 0.1.0a1 (pre-alpha).** hazardloop is an *estimator plus a fail-closed policy
> interface*; it is **not a trained controller**. Any benefit to a live agent is
> **UNVERIFIED** in this release — hazardloop never runs, retrains, or re-executes agents.
> See [`docs/CLAIMS.md`](docs/CLAIMS.md) for the exact CLAIM / NON-CLAIM boundary.

## Why

<!-- MOTIVATION@S7 -->

## Install

```bash
pip install hazardloop            # core: numpy, scipy, typer
pip install "hazardloop[data]"    # + Hugging Face dataset adapter (real trajectories)
pip install "hazardloop[test]"    # + pytest / hypothesis / lifelines (numeric cross-check)
```

## Quickstart

```python
# <!-- QUICKSTART@S7: replace with a runnable, measured example -->
```

```bash
hazardloop --help
```

## What it computes

| Estimator | Module | Notes |
|---|---|---|
| Kaplan-Meier survival + Greenwood + cloglog CI | `core.kaplan_meier` | from scratch (numpy) |
| Nelson-Aalen cumulative hazard | `core.nelson_aalen` | per-step instantaneous hazard for control |
| Aalen-Johansen competing-risk CIF | `core.aalen_johansen` | avoids the naive `1 - KM_c` over-estimation |
| Weibull AFT shape β | `core.weibull_aft` | β<1 early-mortality, β>1 wear-out |
| Cluster bootstrap CI | `core.bootstrap` | default; cluster = model / harness |
| Offline-replay decision quality | `replay` | lead-time, premature-abort, saved-compute, per-cause PR |

## Measured results

<!-- MEASURED@S6: KM curve summary, CIF by cause, logistic-vs-KM divergence, replay metrics -->
<!-- All numbers in this section are generated from bench_results/*.json at S6/S7. No hand-written numbers. -->

## Scope and honesty

hazardloop deliberately does **not**:

- claim any agent capability, task-completion, or pass@k gain (offline-replay numbers are
  counterfactual estimates on logged runs, not live trials);
- extrapolate tails / return levels (no extreme-value theory — the i.i.d. assumption
  breaks for sequential agent failures, arXiv:2511.02927);
- act as a verifier, reward auditor, or formal safety guarantee.

The `fork` decision's rescue rate is unobserved and is **not reported**.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). `scikit-survival` (GPL) is
**not** a dependency and is excluded by a CI check; KM / Nelson-Aalen / Aalen-Johansen /
Weibull-AFT are re-implemented on numpy.

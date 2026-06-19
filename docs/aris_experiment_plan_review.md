# ARIS Experiment-Plan Review

## Baseline Completeness

- Random baseline: random selected subset.
- Current strong baseline: endpoint neuron-aware data selection.
- Simple strong baselines: quality-score selection and diversity coreset.
- Additional baselines: full-data IT, influence/gradient selection, layer
  selective tuning without trajectory selection.
- Ablations: no layer policy, no retention penalty, no hallucination drift gate,
  endpoint only, trajectory only.
- Baseline count: 7 families, excluding ablations.

## Numeric Decision Gates

> **SUPERSEDED (2026-06-19).** The original gate values in this table
> (0.03 relative gain, 2.0x cost, "two capability families") are **no longer the
> active confirmatory thresholds**. The locked decision thresholds now live in
> `configs/experiments/lattice_v5.yaml` and `docs/redesign/REDESIGN_v5.md`, on a
> normalized `[0,1]` utility scale. The table below is rewritten to match the locked
> v5 surface; `lattice_v5.yaml` is the single source of truth and overrides any
> residual numbers here.

| Gate | Metric | Locked v5 threshold | Failure Action | Verdict |
| --- | --- | --- | --- | --- |
| R1 routing value | `pi_psi` vs every control, pool-conditional (normalized) | `delta_R1 >= 0.01` | pivot/downgrade to diagnostic | pass in plan |
| R2 target | LATTICE-R vs stronger full-L NAIT and all baselines (normalized) | `delta_target >= 0.01` | remove method-superiority claim | pass in plan |
| R2 retention | retention-drift upper bound (normalized non-inferiority ceiling) | `delta_ret <= 0.02` | remove reliability claim | pass in plan |
| R2 hallucination | hallucination-drift upper bound (normalized ceiling) | `delta_hall <= 0.02` | remove drift-control claim | pass in plan |
| R2 cost | matched-budget training-compute overhead | `delta_cost <= 0.05` | narrow to analysis method | pass in plan |
| Evidence | seeds/replicates | `S >= 20` | label diagnostic only | pass in plan |

Layer policy is no longer a confirmatory "two capability families" gate; it is a
non-confirmatory generalization probe (`generalization_probe.confirmatory: false`),
Holm-corrected within the probe (see `lattice_v5.yaml`).

## Compute Feasibility

First server gate is restricted to one candidate pool and one base model. Raw
activations stay server-side; selected IDs, hashes, and compact metrics sync
back. The plan requires a 30 percent storage/time buffer and a cleanup policy
before full extraction.

## ARIS Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Evidence Quality | 10 | Target, retention, hallucination drift, cost, and layer evidence are separated and tiered. |
| Rigor | 10 | Matched-budget endpoint baseline, pre-registration, paired tests, and provenance are mandatory. |
| Gates | 10 | Endpoint, drift, cost, layer-policy, and seed gates have numeric thresholds and failure actions. |
| Feasibility | 9 | Budget and raw-activation policy are documented; live server preflight remains external. |
| Paper Potential | 10 | Passing gates would make a strong trajectory-level data-selection contribution. |

Average: 9.8/10.

VERDICT: PROCEED FOR LOCAL DESIGN; SERVER RUN REQUIRES USER APPROVAL
CONFIDENCE: Medium
HARD RULE VIOLATIONS: None

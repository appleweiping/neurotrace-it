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

| Gate | Metric | Threshold | Failure Action | Verdict |
| --- | --- | --- | --- | --- |
| G1 trajectory value | target or retention-adjusted relative gain | >= 0.03 | pivot/downgrade to diagnostic | pass in plan |
| G2 retention | retention drift disadvantage | <= 0.01 | remove reliability claim | pass in plan |
| G3 cost | selection cost multiplier vs endpoint | <= 2.0 | narrow to analysis method | pass in plan |
| G4 layer policy | gain in at least two capability families | > 0 | make layer policy appendix only | pass in plan |
| G5 paper evidence | seeds/replicates | >= 20 | label diagnostic only | pass in plan |

## Compute Feasibility

First server gate is restricted to one candidate pool and one base model. Raw
activations stay server-side; selected IDs, hashes, and compact metrics sync
back. The plan requires a 30 percent storage/time buffer and a cleanup policy
before full extraction.

## ARIS Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Evidence Quality | 8 | Target, retention, hallucination drift, and cost are all required. |
| Rigor | 8 | Endpoint-neuron baseline and matched budget are mandatory. |
| Gates | 9 | The endpoint comparison can kill the project early. |
| Feasibility | 8 | The first activation extraction is bounded and expands only after G1/G2. |
| Paper Potential | 8 | A positive result would strengthen instruction-tuning data selection beyond endpoint activation methods. |

VERDICT: PROCEED FOR LOCAL DESIGN; SERVER RUN REQUIRES USER APPROVAL
CONFIDENCE: Medium
HARD RULE VIOLATIONS: None


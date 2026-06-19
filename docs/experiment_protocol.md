# Experiment Protocol

## Formal Questions

Q1. Does trajectory-level data selection outperform endpoint neuron-aware data
selection at equal data and training budget?

Q2. Does layer-function compatibility improve target-task learning or retention
compared with uniform LoRA/freeze policies?

Q3. Does the method reduce cross-task semantic drift and hallucination drift
relative to capability-only data selection?

## Data

Formal candidates:

- math reasoning instruction data;
- code instruction data;
- multi-hop QA instruction data;
- general instruction retention set;
- factuality/hallucination drift set.

All datasets require manifest, license, split hash, and contamination audit.

## Baselines

1. random subset;
2. full-data instruction tuning;
3. quality-score selection;
4. diversity coreset;
5. influence or gradient similarity;
6. endpoint neuron-aware activation selection;
7. layer-selective tuning without trajectory selection.

## Metrics

- target capability: accuracy/pass rate/exact match;
- retention: MMLU-style aggregate, general instruction following, semantic
  similarity to base model where appropriate;
- hallucination drift: factuality error rate and harmful overconfidence;
- efficiency: selection cost, training cost, memory footprint;
- layer audit: selected-layer stability and ablation curves.

## Statistical Gate

- 20 seeds or bootstrap replicates for paper-result claims;
- paired tests on matched evaluation examples;
- effect sizes and 95 percent confidence intervals;
- Holm correction for multi-metric claims.

## Early Kill Gates

> **SUPERSEDED (2026-06-19).** The relative/absolute thresholds first drafted here
> (>=3% relative, 1 absolute point, 2x cost, "two capability families") are
> **no longer the active confirmatory gates**. The locked decision thresholds now
> live in `configs/experiments/lattice_v5.yaml` and `docs/redesign/REDESIGN_v5.md`,
> expressed as additive margins on a normalized `[0,1]` utility scale. The values
> below have been rewritten to match the locked v5 surface; `lattice_v5.yaml` is the
> single source of truth and overrides any residual wording here.

1. Trajectory/routing win (`delta_R1 = 0.01`, normalized): `pi_psi` must beat every
   control's pool-conditional value by `>= 0.01` normalized utility (one normalized
   point) via the per-contrast marginal-alpha bound; the method gate
   (`delta_target = 0.01`) requires the same margin over the stronger full-L NAIT
   variant and every locked baseline.
2. Retention-drift ceiling (`delta_ret = 0.02`, normalized): the retention-loss
   upper bound must sit below `0.02` (<=2 normalized points) vs the matched
   reference; the hallucination-drift ceiling is `delta_hall = 0.02` likewise.
3. Cost-gap ceiling (`delta_cost = 0.05`): realized matched-budget training-compute
   overhead must stay below 5 percent relative, unless the paper claims a high-cost
   analysis method only.
4. Layer-policy contribution is a **non-confirmatory generalization probe**
   (`generalization_probe.confirmatory: false` in `lattice_v5.yaml`), Holm-corrected
   within the probe; it is no longer a "two capability families" confirmatory gate.


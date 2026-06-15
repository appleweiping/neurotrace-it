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

1. Trajectory selection must beat endpoint activation selection by >=3 percent
   relative on target capability or retention-adjusted score.
2. Retention drift must not be worse than endpoint selection by more than 1
   absolute point.
3. Layer-wise policy must beat uniform adaptation in at least two capability
   families or be downgraded to diagnostic.
4. Selection cost must remain within 2x endpoint-selection cost for the same
   candidate pool, unless the paper claims a high-cost analysis method only.


# Pre-Registration

## Scope

This pre-registration covers the first endpoint-vs-trajectory data-selection
gate. It does not claim that results exist.

## Frozen Primary Hypothesis

H1: trajectory selection beats endpoint-neuron selection by at least 0.03
relative on target or retention-adjusted score, with retention drift
disadvantage <= 0.01 and selection cost <= 2x.

## Primary Outcomes

- Target or retention-adjusted relative gain.
- Retention drift disadvantage.
- Selection cost multiplier.

## Secondary Outcomes

- Hallucination/factuality drift.
- Layer policy ablation gain.
- Selected-data diversity.

## Analysis Lock

Endpoint-neuron and trajectory selection must use the same candidate pool,
budget, base model, LoRA rank, training steps, validation policy, and evaluator.


# NeuroTrace-IT

NeuroTrace-IT is a research-grade project on trajectory- and layer-aware data
selection for instruction tuning. It studies whether full reasoning activation
trajectories identify training examples that improve target capabilities while
preserving general semantics and reducing cross-task hallucination drift.

The defended claim is:

> Instruction-tuning data should be selected by trajectory-level neural evidence
> and layer-function compatibility, not only by start/end-token activation
> similarity or surface quality scores.

This is not an activation-steering project and not a generic data-selection
wrapper.

## Research Spine

1. Represent each instruction example by activation trajectories over reasoning
   steps, tokens, and layers.
2. Select examples whose trajectories match target capability paths while
   minimizing stability drift from the base model.
3. Use layer-wise adaptation or freezing only when trajectory evidence predicts
   layer-function alignment.
4. Evaluate both target-task gain and retention/hallucination drift under the
   same backbone and splits.

## Non-Toy Standard

Paper-facing evidence requires:

- real instruction-tuning corpora and held-out capability/retention benchmarks;
- NAIT-style neuron activation baselines, quality-filter baselines,
  influence/gradient baselines, random and full-data baselines;
- same base model, LoRA budget, training steps, and evaluator;
- 20 seeds or bootstrap replicates for paper claims;
- paired tests, effect sizes, confidence intervals, and drift audits;
- ARIS gates before any strengthened claim.

Smoke tests in this repository are only contract tests.


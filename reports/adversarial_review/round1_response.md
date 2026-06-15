# Adversarial Review Round 1 Response

## Reviewer Objection 1

NeuroTrace-IT may be NAIT with more expensive features.

Response: endpoint-neuron selection is mandatory, matched-budget, and the first
kill gate. If trajectory does not beat endpoint, the project pivots.

## Reviewer Objection 2

Layer-wise policy may be unnecessary complexity.

Response: layer policy is an ablation and becomes appendix-only unless it helps
in at least two capability families.

## Reviewer Objection 3

Target gains may hide retention or hallucination drift.

Response: retention and hallucination drift are primary paper metrics and can
block the reliability claim.

## Post-Review Hardening

The adversarial review treats NAIT-style endpoint-neuron selection as the
central kill comparator. The current package replaces scorecard-string
validation with checks for concrete first-gate and provenance obligations:

- `configs/baselines/baseline_registry.yaml` now records the NAIT-style
  endpoint baseline with paper URL, implementation source, license check, input
  access, tuning policy, and matched-budget fairness;
- `configs/experiments/first_gate.yaml` requires trajectory-vs-endpoint margin,
  contamination/leakage audit, retention drift gate, hallucination drift gate,
  and server prohibition;
- `configs/experiments/formal_neurotrace_it.yaml` makes
  `endpoint_baseline_required: true` and
  `trajectory_endpoint_relative_gain` contract markers;
- `schemas/selection_manifest.schema.json` requires `endpoint_neuron_selection`
  and `server_authorized: false`.

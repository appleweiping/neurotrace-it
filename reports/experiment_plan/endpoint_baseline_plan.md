# NeuroTrace-IT Endpoint Baseline Plan

Status: design-only local packet. No server command is authorized.

## First Server Gate Draft

Objective: compare endpoint-neuron selection and trajectory selection under the
same candidate pool, selected-data budget, base model, LoRA budget, and
evaluation set.

## Preconditions

- Endpoint-neuron baseline source/commit is selected and recorded.
- ARIS experiment-plan review remains >=8 with no hard-rule violations.
- User approves exact command, model, dataset, output path, and stop condition.
- Server process/GPU/storage preflight is clean.

## Candidate Command Draft

```bash
# Draft only, do not run without approval
python scripts/run_selection_gate.py \
  --config configs/experiments/formal_neurotrace_it.yaml \
  --run-tier diagnostic \
  --output outputs/neurotrace_first_gate
```

## Stop Conditions

- Trajectory selection fails to beat endpoint selection by the 0.03 relative gate.
- Retention drift disadvantage exceeds 0.01.
- Selection cost exceeds 2x endpoint cost without a pre-approved analysis-only
  reframing.


# NeuroTrace-IT Endpoint Baseline Plan

> **STALE / SUPERSEDED (v1-era packet).** This early draft predates the LATTICE-R
> v5 redesign and is kept only for historical context. Its thresholds, command,
> and config below are **NOT the registered v5 decision rules** and must not be
> read as such. The binding, locked thresholds live in
> `configs/experiments/lattice_v5.yaml` and are listed in
> `reports/run_packet.md` (e.g. R1 margin `delta_R1 = 0.01`, R2 non-inferiority
> margins `delta_target/delta_rel/delta_ret/delta_hall/delta_cost`, `floor_partial`).
> In particular the `0.03 relative gate`, the `0.01` retention-drift line, the
> `2x` cost ceiling, the `run_selection_gate.py` script, and the
> `formal_neurotrace_it.yaml` config named in the Stop Conditions / Candidate
> Command below are all **obsolete v1 placeholders**, superseded by the v5
> bootstrap-`t` gates G2r/G2h/G2c and the closed-testing graph. See
> `docs/redesign/REDESIGN_v5.md` for the current design.

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


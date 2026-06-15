# Active TODO

## Current Phase

Phase 0: local non-server initialization and design package.

## Completed

- Private GitHub repository created and pushed.
- Initial README, AGENTS, ARIS audit, protocol, baseline contract, claim matrix,
  server runbook, configs, validator, and tests created.
- Seed IT tuning idea reviewed and abstracted into a trajectory-vs-endpoint
  selection question.
- Local trajectory selection schema, retention/cost gate helper code, and tests added.
- ARIS research-refine and experiment-plan self reviews updated to >=8.
- Design-stage schema and endpoint-baseline report entries created under `reports/`.

## Next Local Tasks

1. Fill endpoint-neuron baseline reproduction checklist with exact source and commit after citation/baseline audit.
2. Add selected-example manifest validator extensions when real selection outputs exist.
3. Prepare exact server command packet for user approval after ARIS plan review.

## Server-Stage TODO

1. Extract activations for a small approved candidate pool after design review.
2. Run endpoint-neuron and trajectory selection at identical budgets.
3. Train matched LoRA runs and sync compact metrics/provenance.

## Next Concrete Command

```powershell
python scripts\validate_project.py; python -m pytest -q
```

## Stop Condition

Do not run server training or activation extraction until ARIS experiment-plan
review passes and the exact command is approved.

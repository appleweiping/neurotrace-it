# Active TODO

## Current Phase

Phase 0: local non-server initialization and design package.

## Completed

- Private GitHub repository created and pushed.
- Initial README, AGENTS, ARIS audit, protocol, baseline contract, claim matrix,
  server runbook, configs, validator, and tests created.
- Seed IT tuning idea reviewed and abstracted into a trajectory-vs-endpoint
  selection question.

## Next Local Tasks

1. Draft trajectory signature schema for token/step/layer activations.
2. Draft endpoint-neuron baseline reproduction checklist.
3. Write ARIS experiment-plan packet with activation extraction cost estimate.
4. Add validators for selected-example manifest once schema exists.

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


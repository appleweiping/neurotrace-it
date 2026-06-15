# NeuroTrace-IT Agent Operating Contract

Project slug: `neurotrace-it`.

Read these files before nontrivial work:

1. `README.md`
2. `docs/research_brief.md`
3. `docs/experiment_protocol.md`
4. `docs/baseline_contract.md`
5. `docs/claim_evidence_matrix.md`
6. `docs/aris_research_refine_audit.md`
7. `docs/server_runbook.md`

Use `agentmemory` according to `D:\devtools\AGENT-MEMORY-PROTOCOL.md`.

## Hard Rules

- Do not turn this into activation steering at inference time.
- Do not claim novelty over NAIT unless trajectory-level and layer-wise evidence
  beats start/end activation similarity under fair budgets.
- Do not optimize target-task gain while ignoring semantic retention,
  calibration, and cross-task hallucination drift.
- No toy or tiny-only result may be paper evidence.
- Server training requires ARIS experiment-plan approval and exact user
  authorization.

## Evidence Labels

- `paper_result`: 20+ seeds/replicates, fair baselines, significance tests,
  effect sizes, and complete provenance.
- `official`: 10+ seeds/replicates, caveated paper or appendix claims.
- `diagnostic`: analysis only.
- `pilot`: local contract/smoke only.


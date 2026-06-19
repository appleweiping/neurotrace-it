# Active TODO

## Current Phase

Build-now / run-later: Stage-1 Registered-Report design **locked at REDESIGN_v5
(LATTICE-R)** and novel-core pipeline **implemented & frozen** (green 93-test
harness). Pre-registered **decision thresholds are now LOCKED** as design choices in
`configs/experiments/lattice_v5.yaml`; only empirical results remain `DATA_NEEDED`.
The deferred authorized GPU run is the only remaining stage (`server.authorized:
false`); the entrypoints' authorized branches are guarded stubs (training body wired,
not yet coded).

## Completed

- Private GitHub repository created and pushed.
- Initial README, AGENTS, ARIS audit, protocol, baseline contract, claim matrix,
  server runbook, configs, validator, and tests created.
- Seed IT tuning idea reviewed and abstracted into a trajectory-vs-endpoint
  selection question.
- Local trajectory selection schema, retention/cost gate helper code, and tests added.
- ARIS research-refine and experiment-plan self reviews updated to >=8.
- Design-stage schema and endpoint-baseline report entries created under `reports/`.
- **REDESIGN_v4 Stage-1 RR locked** (method + theory + pre-registered analysis +
  kill-gates), responding to the v3 review (full-`phi_end` ridge control,
  cross-fit + conditional-null permutation + BCa test, locked LOCI `Y` + G7,
  partial-R² vs overall-increment floors, two-layer Holm, extraction-parity cost).
- **REDESIGN_v5 (LATTICE-R) locked** (rounds 1–4 adversarial re-gate): policy-value
  routing estimand `V(pi)` (no SUTVA), score-free `capacity_match` map, deterministic
  controls, intersection-union test + paired bootstrap-`t`, closed-testing FWER bound,
  faithful NAIT over `L`. Phase-B routing modules implemented & unit-tested
  (`layer_function.py`, `cost_model.py`, `baselines/nait_layerwise.py`,
  `analysis/{layer_attribution,routing_intervention,matched_budget,closed_testing,pool_firewall}.py`,
  `schemas_v2.RouterOutputs`), config `configs/experiments/lattice_v5.yaml`.
- **Pre-registered decision thresholds LOCKED** (C+D hardening): `delta_R1=0.01`,
  `delta_target=0.01`, `R_tot=64`, `r_max=16`, `r_0=4`, control seeds
  (`seed_rand/seed_shuf/seed_ada`), closed-test alpha allocation
  (`0.05`; split `0.34/0.33/0.33`), `floor_partial=0.01`, plus the supporting
  margins/rules — all in `lattice_v5.yaml` with provenance; empirical results stay
  `DATA_NEEDED`.
- **Citation audit pass** (verified via arXiv/ACL Anthology/OpenReview 2026-06-19):
  NAIT real author list, Diddee/Ippolito → Findings of NAACL 2025, AdaLoRA
  (+Karampatziakis), NeFT COLING 2025 (+Yingpeng Ma), Gretton MMD → `@article`,
  Qwen2.5 (arXiv:2412.15115), MoE routing → Shazeer 2017; baseline registry
  `2605.26004` re-characterized as a multimodal/VLM coreset (not a text anchor).
- **Novel-core modules implemented & frozen** (pure-stdlib, build-now/run-later):
  `baselines/nait.py`, `trajectory.py`, `analysis/residual_test.py`,
  `analysis/residualize.py`, `analysis/outcome_y.py`, `analysis/pair_mining.py`,
  `analysis/drift.py`, `selection.py`, `schemas_v2.py`, and the frozen
  `configs/experiments/lattice_v4.yaml`; unit/contract tests under `tests/`.
- **Theory discharged where proved** (Brier propriety, dual-ridge feasibility,
  residual-orthogonality estimand, matched-pair identification lemma, `kappa`
  non-functionality, submodular (1−1/e)).
- **Conference paper skeleton written** at `paper/main.tex` (+ `paper/references.bib`)
  — all results are DATA_NEEDED placeholders; no fabricated numbers.

## Next Local Tasks

1. Fill endpoint-neuron baseline reproduction checklist with exact source and commit after citation/baseline audit.
2. (`layer_function.py` and `cost_model.py` are now implemented & unit-tested —
   superseded.) Implement the entrypoint training/extraction bodies that currently
   raise from the guarded `run_authorized(...)` stubs, behind `server.authorized`.
3. Prepare exact server command packet for user approval after ARIS plan review
   (`reports/run_packet.md` is the current hand-off).

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

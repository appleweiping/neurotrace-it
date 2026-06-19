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

## Current Status

The Stage-1 Registered-Report design is locked at **REDESIGN_v5 (LATTICE-R)**
(`docs/redesign/REDESIGN_v5.md`), which elevates the v4 core to a *coupled
selection-and-routing* method: one endpoint+NAIT-residualized trajectory attribution
`psi(x)` drives both data selection (its sum) and LoRA layer routing (its support),
tied by the coupling identity `sum_{l in A} psi_l = beta_T . T_tilde`. The routing
object is a well-posed **pool-conditional policy value** `V(pi)` over whole training
runs (no per-example potential outcome, no SUTVA), decided by an **intersection-union
test** over a paired studentized bootstrap-`t`, inside a closed-testing graph with an
in-document FWER bound.

The novel-core pipeline is **implemented and frozen** as pure-stdlib,
build-now/run-later modules (no model load, no server call, no training): the faithful
NAIT baseline over the full released layer set `L`, the trajectory operator (SW2 `D`,
curvature `kappa`), the frozen out-of-sample nuisance map and the residualized
attribution `psi` + coupling identity, the score-free `capacity_match` rank map and
deterministic control policies, the policy-value / IUT routing test, the matched-budget
comparison, the closed-testing graph, the three-pool firewall, the LOCI utility outcome +
reliability gate, the Brier calibrator + factuality gate, and the V2 auditable signature
record. These are exercised by a **green 93-test harness** (`python -m pytest -q`).
**Pre-registered decision thresholds are now LOCKED** as design choices in
`configs/experiments/lattice_v5.yaml` (margins `delta_R1`/`delta_target`, capacity
`R_tot`/`r_max`/`r_0`, control seeds, the closed-testing alpha allocation, the partial-R^2
floor, etc.); only genuinely empirical results remain `DATA_NEEDED`. The theory is
**discharged where proved**, and the conference paper lives in `paper/main.tex`
(+ `paper/references.bib`). The operational hand-off is `reports/run_packet.md`. See also
`docs/active_todo.md`, `docs/milestones.md`, `docs/paper_claims_status.md`, and
`docs/definition_of_done.md`. **The only remaining gate is the authorized GPU run
(training / activation extraction / evaluation), which is deferred:
`server.authorized` stays false and no empirical claim is made until it executes;** the
entrypoints' authorized branches are guarded stubs, so the training body is wired and
specified but not yet coded.

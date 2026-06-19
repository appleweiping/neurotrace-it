# LATTICE-R run packet (Stage-1 Registered Report)

Status: **BUILD-NOW / RUN-LATER.** `server.authorized` stays **false**. **No
experiment, training, extraction, model load, or GPU job is run by this packet.**
This document is the operational hand-off for the v5 LATTICE-R contribution: it
pins what is *already implemented and unit-tested* versus what a *future authorized
run* would execute. It is kept **mutually consistent** with
`docs/redesign/REDESIGN_v5.md` (design) and `paper/main.tex` (Stage-1 Registered
Report) on every implementation-status statement: the v5 modules **ARE
implemented** as additive, do-not-run, pure-Python (stdlib only), and the unit
harness is **green (93 passing)**. There are **no** "pending implementation"
contradictions across the three documents.

This packet fabricates **no** numeric result. Every empirical cell is a
`DATA_NEEDED` placeholder. No arXiv id is invented (see `paper/references.bib`
citation hygiene).

---

## 0. Headline (one screen)

**LATTICE-R**: trajectory-aware, layer-routed instruction-tuning data selection
from **one** endpoint+NAIT-residualized attribution `ψ(x) ∈ R^{|A|}`, certified by
an **interventional routing test**, beating the **faithfully reproduced NAIT**
(`arXiv:2603.13201`) comparator at a **measured compute-matched** budget.

- Selection score = the **sum** `Σ_{ℓ∈A} ψ_ℓ(x)`; routing policy = the **support**
  `supp(ψ(x))` over the anchor set `A`; tied by the coupling identity
  `Σ_{ℓ∈A} ψ_ℓ = β̂_T · T̃` (one orthogonalized coefficient, one out-of-sample
  residual).
- Routing object = a **pool-conditional policy value** `V(π) = E_seed[U_train(π) |
  locked pools]` over whole training runs (no per-example potential outcome, no
  cross-example SUTVA).
- "Beats every control/comparator" = an **intersection-union test** (Berger 1982):
  each contrast at the marginal level `α`, no multiplicity penalty, exact level
  `α`; each per-contrast bound is a paired studentized bootstrap-`t`
  (asymptotically valid).
- Headline `R2`: LATTICE-R beats `nait_layerwise` over the full released decoder
  set `L` **and** every locked baseline on retention-adjusted target utility at a
  **measured** compute-matched budget (ledger: params, optimizer slots, realized
  FLOPs, skip-flag).

---

## 1. Implementation status (authoritative; mirrors REDESIGN_v5 §5.5 and main.tex Table 1)

**All v5 modules below ARE implemented** under `src/neurotrace_it/` as additive,
do-not-run, pure-Python: **stdlib only, no model load, no server call, no
training**. The caller supplies already-measured tensors/contrasts at run-later;
the arithmetic (apportionment, residualization, bootstrap-`t`, IUT, closure) is
testable build-now / run-later. `server.authorized: false` throughout.

| Module | Method block | Status |
| --- | --- | --- |
| `layer_function.py` | `capacity_match` / `make_feasible_mask` / six deterministic control masks; `leave_one_layer_redistribute`; `frozen_layer_ablation_profile` + `validate_J_against_freeze` | implemented; green |
| `cost_model.py` | measured compute ledger, extraction-parity check, routing-savings (skip-flag) — Gate R3 | implemented; green |
| `baselines/nait_layerwise.py` | faithful NAIT over the full `L` (base Eq.2–6); gated 8-anchor secondary | implemented; green |
| `analysis/layer_attribution.py` | frozen nuisance map `B_λ`, out-of-sample `T̃`, per-layer `ψ_ℓ`, coupling identity in/out of sample | implemented; green |
| `analysis/routing_intervention.py` | pool-conditional policy value, paired bootstrap-`t`, intersection-union decision — Gate R1 | implemented; green |
| `analysis/matched_budget.py` | R2-target IUT over the baseline set + single-contrast margin sub-claims — Gate R2 | implemented; green |
| `analysis/closed_testing.py` | closed-testing graph, IUT union-null leaves, 63-intersection closure vs shortcut, FWER probe | implemented; green |
| `analysis/pool_firewall.py` | locked three-pool split, leakage assertions, non-confirmatory probe regenerator | implemented; green |
| `schemas_v2.py` | `RouterOutputs` (`anchor_mask`, `rank_per_anchor`), `policy_value`, `pool_hashes`, `routing_policy_value`, `control_provenance` (additive optional) | implemented; green |

**Config:** `configs/experiments/lattice_v5.yaml` is additive (does **not**
overwrite `lattice_v4.yaml`); multiplicity-structure constants
(weights/edges/split `(0.34,0.33,0.33)`, IUT family) are **pinned now**; numeric
margins/thresholds are `DATA_NEEDED`, pinned at run authorization;
`server.authorized: false`.

> Note on REDESIGN_v5 §5.5 wording. The design doc was written **before** the
> modules existed and lists several as "REQUIRED; currently absent" with a
> Phase-B plan. That plan is now **discharged**: the modules in the table above
> are present and unit-tested. This packet and `paper/main.tex` reflect the
> **current** state (implemented + green); the design doc's "absent" notes are
> historical and superseded by this status table. There is no live contradiction:
> design = "what v5 would lock"; packet/paper = "it is locked and implemented,
> not run".

---

## 2. Green unit harness (code-correctness checks, NOT evidence)

Command (run locally; **not** a server/experiment run — pure stdlib, no model,
no GPU):

```
python -m pytest -q
```

Last result: **93 passed** (exit code 0). Per-file breakdown:

| Test file | Count | What it pins (formula evaluations only) |
| --- | --- | --- |
| `tests/test_layer_routing.py` | 26 | coupling identity in/out of sample; `capacity_match` conservation + score-freeness across all feasible cardinalities; `make_feasible_mask` lift (empty ⇒ uniform-over-`A`); six-arm IUT (per-contrast level, IUT size ≤ α at the least-favorable config, margin enters); deterministic control maps (`π_ada` frozen by `seed_ada`, invariant to training seed); closed-testing shortcut == brute-force (63 intersections) + FWER at the binding LFC; pool firewall leakage assertions; R0 permutation placebo level; Gate R2 method-win = all sub-claims + matched ledger |
| `tests/test_trajectory_selection.py` | 35 | strictly-proper Brier calibrator (risk minimized at `q=p`), G6 gate sets `λ_f:=0` on failure; trajectory operator (`D_ℓ`, `κ_ℓ`); selection objective |
| `tests/test_residualize_gates.py` | 19 | dual-ridge FWL residual maker; cross-fit partial-R²; conditional-null block permutation; BCa |
| `tests/test_endpoint_baseline.py` | 7 | `nait_layerwise` reproduces base Eq.5 over `L`; per-layer PCA/sign-align; token-mean variant; gated 8-anchor restricted sum; endpoint control `φ_end` is the **distinct** object |
| `tests/test_selection_schema_metrics.py` | 3 | selection schema + metric contracts |
| `tests/test_project_contracts.py` | 3 | project-level invariants (incl. `server_authorized` stays false) |
| **Total** | **93** | all passing; **code-correctness, not evidence** |

These are **formula evaluations** of locked equations from REDESIGN_v5
(§3.4, §3.5, §3.6, §3.7, §4). They are **not** experimental evidence and license
**no** empirical claim.

---

## 3. What a FUTURE authorized run would execute (NOT run here)

Requires `server.authorized: true` and an explicit authorization step. None of
this is performed by this packet.

```
Phase B-0  faithful NAIT over L (control φ_end + comparator s_NAIT), both base variants
Phase B-1  trajectory signatures D_ℓ, κ_ℓ, T(x) per pool (activations only on P_dep)
Phase B-2  Gate R0: cross-fit (K=10) frozen-nuisance residual test on T̃ vs Ỹ;
           block permutation (P=5000) + cluster BCa (B=2000); decision vs floor
Phase B-3  Gate R1: six real masked-LoRA arms over A (shared r_0, shared capacity_match,
           seeds 0..19); seed-mean V̂(π_arm); per-control bootstrap-t L_c(α);
           intersection-union decision (reject iff every L_c(α) > δ_R1)
Phase B-4  Gate R2: greedy selection on Σψ + routing π_ψ; train vs stronger-NAIT(L)
           and every baseline at a MEASURED matched budget; R2-target IUT + margin sub-claims
Phase B-5  closed-testing decision (§4 graph) + honest two-sided cost ledger (R3)
```

Backbones (pinned, released): `Qwen2.5-7B-Instruct` (primary),
`Qwen2.5-1.5B-Instruct` (secondary), `Llama-3.1-8B-Instruct` /
`Mistral-7B-Instruct-v0.3` (alternates); LoRA only; matched rank/steps/tokens.

Datasets (pinned identities; hashes at authorization): MetaMathQA, a permissive
code-instruct set, HotpotQA-distractor (candidate pool); MMLU (retention);
TruthfulQA + FActScore-style atomic factuality (hallucination + G6 ground truth).

---

## 4. Gates and fail actions (locked; identical to REDESIGN_v5 §5.3 and main.tex)

| Gate | Test | Fail action |
| --- | --- | --- |
| **R0** | block-perm `p<α_R0` on `T̃=T−ZB_λ` (residual to `[φ_end, NAIT(L)]`, frozen nuisance) + BCa above `floor_partial`, stable across `r∈{8,16,32}` | `stop_main_novelty_claim` (reduce to NAIT, clean null) |
| **R1** | union null `⋃_c{g_c≤δ_R1}`; reject iff **every** `L_c(α)>δ_R1` (IUT, paired bootstrap-`t`), pool-conditional on `P_val` | `drop_routing_keep_selection` (fall back to v4) |
| **R2** | matched-budget win: R2-target IUT over stronger-NAIT(L) + every baseline; each margin sub-claim's bound clears its `δ_k` | `no_method_win_claim` |
| **R3** | extraction-parity ≤ 2.0× kept; savings only if measured (skip-flag) | savings forbidden / `high_cost_analysis` |
| **G6** | factuality precondition (Spearman/Pearson ρ ≥ 0.3, lower CI > 0, ECE ≤ 0.1) | `λ_f := 0` |
| **G7** | `Y_obs`-reliability (ICC(2,1) ≥ 0.6; proxy↔retrain ρ ≥ 0.3) | primary not run on proxy |
| **J-val** | `J_{c,ℓ}` vs real anchor layer-freeze `Δ_ℓ^{LOL}` | drop `J` from router |

---

## 5. Primary cells (named; all DATA_NEEDED until the authorized run)

- **R0 primary:** `Qwen2.5-7B (math)` median-of-folds `partial_R²_T` and its BCa CI
  vs `floor_partial`.
- **R1 primary:** the IUT verdict over the five controls `{unif, shuf, rand,
  global, ada}` at `δ_R1`, with the simultaneous bound `L_{R1} = min_c L_c(α)`.
- **R2 primary:** retention-adjusted-gain win of LATTICE-R vs
  `nait_layerwise`(`L`) under a **matched** ledger, with the R2-target IUT over the
  full baseline set.

No value above is filled in. `server.authorized: false`; **no run executed.**

---

## 6. Consistency assertion

`paper/main.tex`, `docs/redesign/REDESIGN_v5.md` (as superseded by §1's status
table for the implementation state), and this `run_packet.md` agree on:
modules **implemented** (do-not-run, pure-Python), unit harness **green (93
passing)**, `server.authorized: false`, **no run executed**, and **zero fabricated
numbers**. Any "pending implementation" phrasing in the design doc is historical
Phase-B planning, now discharged; there is no live status contradiction.

# REDESIGN v6 — neurotrace-it (Stage-1 delta over v5: top-conference baseline lock)

**Status: design-only delta.** `server.authorized` stays **false**; nothing runs;
every result cell stays `DATA_NEEDED`. This document is a **concise delta over
`docs/redesign/REDESIGN_v5.md`** — it does NOT restate the v5 method. The v5
LATTICE-R design (trajectory operator, coupling identity, three-pool firewall,
policy-value R1, closed-testing FWER, locked margins/capacity) is **preserved
verbatim**; v6 only (a) promotes the headline model, (b) locks a verified
official-repo baseline suite for a fair top-venue comparison, (c) pins the
fairness protocol, (d) adds an explicit method-iteration loop, and (e) records the
single-RTX-4090 serial schedule. All v5 locked constants in
`configs/experiments/lattice_v5.yaml` remain the single source of truth.

Companion artifacts (new/updated in v6):
`configs/baselines/baseline_registry.yaml` (11 verified-repo entries +
`v6_baseline_contract` block), `BASELINES.md` (root), and the appended baseline
section of `reports/run_packet.md`.

---

## v6 delta log — what changed vs v5 and why

1. **Headline promotion (A).** Qwen2.5-7B-Instruct is now the **single
   confirmatory headline**; Qwen2.5-1.5B-Instruct is demoted to a
   robustness/smoke row. (In v5 both sizes appeared in the results table without a
   designated confirmatory headline.)
2. **Baseline lock (B).** The earlier baseline set (family-level placeholders +
   two reimpl anchors) is **extended** with **11 baselines that each have a
   confirmed official code release** (2024–2026), satisfying a **>= 8
   official-repo** fair-comparison contract. The reimpl anchors (NAIT, AlpaGasus)
   are retained but explicitly **not counted** in the official-repo set.
3. **Fairness protocol (C).** The "what is matched" contract is stated explicitly
   against the 7B headline (model, 3 pools, budget B, identical LoRA/optimizer/
   steps, identical eval, deciding gate).
4. **Method-iteration loop (D).** An explicit, capped diagnose-and-improve loop is
   added for the case where LATTICE-R does not clear R0/R1/R2 vs the baselines —
   improve the **method only**, never the data, never fabricate.
5. **Single-GPU schedule (E).** A single-RTX-4090 serial schedule note (kill-gate
   first ordering) is added.

Nothing in §§3–4 of v5 (method equations, multiplicity graph) is altered.

---

## A. Headline model: promote Qwen2.5-7B-Instruct; demote 1.5B

- **Confirmatory headline:** `Qwen/Qwen2.5-7B-Instruct` (revision `main`,
  Apache-2.0). All confirmatory R0/R1/R2 decisions and the "beats SOTA selection"
  / "beats NAIT" claims are made **at 7B**. The three result rows
  `Qwen2.5-7B (math)`, `Qwen2.5-7B (code)`, `Qwen2.5-7B (QA)` (one per pool) carry
  the confirmatory family.
- **Robustness / smoke row:** `Qwen/Qwen2.5-1.5B-Instruct` becomes a single
  `Qwen2.5-1.5B (all)` robustness row — a directional consistency / smoke check,
  **not** part of the confirmatory `α`. A 1.5B-only result may never be paper
  evidence (AGENTS.md non-toy standard).
- The frozen estimand, pools, LoRA-only constraint, and the locked v5 margins are
  unchanged. This is a *reporting-altitude* change, not an estimand change.

> Paper alignment: `paper/main.tex` already pins Qwen2.5-7B-Instruct primary /
> 1.5B secondary and the per-pool 7B result rows; v6 makes the 7B headline status
> explicit in the design record.

---

## B. The verified-official-repo baseline contract (>= 8) + disclosed anchors

All entries are registered in `configs/baselines/baseline_registry.yaml`
(`v6_baseline_contract` block) and tabulated in `BASELINES.md`. Each was confirmed
to have **real official code** (`repo_verified: true`,
`implementation_source: official_repo`).

**Selection must-beat comparators run from official code (9):**

1. **LESS** — ICML 2024 — princeton-nlp/LESS — gradient/influence selection (MIT).
2. **DEITA** — ICLR 2024 — hkust-nlp/deita — quality × complexity × diversity (Apache-2.0).
3. **SelectIT** — EMNLP 2024 — Blue-Raincoat/SelectIT — model-intrinsic uncertainty (Apache-2.0).
4. **DataInf** — ICLR 2024 — ykwon0407/DataInf — closed-form LoRA influence (MIT).
5. **Nuggets** — ICLR 2024 — pldlgb/nuggets — one-shot in-context info gain (license not stated).
6. **MIG** — ACL 2025 Findings — yichengchen24/MIG — max information gain in semantic space (Apache-2.0).
7. **S2L / SmallToLarge** — NeurIPS 2024 — BigML-CS-UCLA/S2L — small-model
   training-trajectory clustering (MIT). **The closest trajectory competitor.**
8. **Cherry / IFD** — NAACL 2024 — tianyi-lab/Cherry_LLM — Instruction-Following
   Difficulty (license not stated).
9. **TAGCOS** — NAACL 2025 — 2003pro/TAGCOS — gradient-clustered coreset (license
   not stated). Supersedes `diversity_coreset` as the citable text-coreset must-beat.

**Routing-arm controls run from official code (2):**

10. **NeFT** — COLING 2025 — NLP2CT/NeFT — neuron-level fine-tuning (control;
    license not stated).
11. **AdaLoRA** — ICLR 2023 — QingruZhang/AdaLoRA — adaptive rank budget (control, MIT).

**Count: 11 verified official-repo baselines ⇒ the >= 8 contract is satisfied.**

**Disclosed reimplementation anchors (NOT counted in the >= 8):**

- **NAIT** (ICLR 2026) — no public repo — faithful reimpl of endpoint
  neuron-similarity (Eq. 1), already an R0 control, high confidence
  (`endpoint_neuron_selection`).
- **AlpaGasus** (ICLR 2024) — GitHub is a project webpage only, no runnable code —
  faithful reimpl of the LLM-as-judge scoring (`quality_score_selection`).

`repo_verified` is marked honestly: `true` for the 11, `false` (no runnable
official code) for the two anchors.

---

## C. Fairness protocol (matched everything except the selection rule)

Each baseline differs from LATTICE-R in **exactly one place** — its selection rule
(or, for NeFT/AdaLoRA, its per-layer capacity allocation). Held identical:

- **Model:** Qwen2.5-7B-Instruct headline (revision `main`). Any baseline needing
  a scorer/proxy uses a Qwen2.5 backbone (7B raters; 1.5B for S2L's small-model
  proxy) — never a foreign base.
- **Pools:** the same three example-disjoint, family-stratified candidate pools
  (MetaMathQA / glaive-code-assistant / HotpotQA-distractor) under the locked
  3-pool firewall (`P_train`/`P_val`/`P_dep`); selection reads activations/scores
  only through the frozen `P_train` operators.
- **Budget:** the same top-`B` selection budget (pinned in `lattice_v4.yaml`).
- **Training:** OUR identical LoRA pipeline — same rank/capacity grid, steps,
  token budget, optimizer, and seeds `0..19`; equal *realized* compute checked by
  `cost_model.compute_match_ledger`.
- **Evaluation:** OUR identical eval — target accuracy, MMLU retention, TruthfulQA
  + FActScore-style hallucination, honest cost ledger.
- **Deciding gate:** the v5 closed-testing graph at FWER `α = 0.05` over
  `{R0, R1, G2t, G2r, G2h, G2c}`. **G2t** ("beats SOTA selection") is the
  full-union **IUT over the entire comparator set** — LATTICE-R must beat *every*
  baseline by `δ_target`, not the weakest. **R1** is the IUT over routing controls
  (incl. AdaLoRA, NeFT) by `δ_R1`. All margins/floors are the locked v5 values.

This contract is the binding text for `docs/baseline_contract.md` and the
`fairness_adaptation` field of each registry entry.

---

## D. Method-iteration loop (improve the METHOD, never the data)

If, at the 7B headline, LATTICE-R does **not** clear a confirmatory gate vs the
baseline suite, follow a **capped, pre-registered** diagnose-and-improve loop. The
data, pools, budget, splits, evaluators, margins, and the gate definitions are
**frozen** and may not be touched; only the **method** may change.

Loop (max **2** method-iteration rounds, then honest report):

1. **Diagnose, do not tune.** Use only the existing diagnostic/exploratory
   surface (R0 partial-`R²`, per-layer `J_{c,ℓ}`, the ablation arms, the cost
   ledger) to localize *why* a gate failed — e.g. trajectory residual too weak
   (R0), routing allocation mis-targeted (R1), selection not separating from a
   specific competitor (G2t), or a retention/cost ceiling breached.
2. **Improve the METHOD only.** Permitted: the trajectory operator (SW2 / curvature
   parameterization, anchor set `A`), the residualization/attribution `ψ`, the
   mask→rank `capacity_match` policy, the routing `τ_sel`/capacity grid — i.e. the
   modules in v5 §3 / §5.5. **Forbidden:** changing the candidate pool or budget,
   re-selecting data to fit the gate, moving a margin, swapping the evaluator, or
   any post-hoc threshold tuning. No fabricated numbers, ever (AGENTS.md hard
   rules; run_packet "fabricates ZERO numeric results").
3. **Re-register and re-run** the changed method through the **same** R0/R1/R2
   pipeline on the **same** locked pools/seeds. A method change that touches a
   locked constant requires a fresh pre-registration entry before re-run.
4. **Cap.** After at most 2 method-iteration rounds, if a confirmatory gate still
   fails, **stop and report honestly**: apply the pre-registered failure action
   for that gate (R0 fail ⇒ `stop_main_novelty_claim`; R1 fail ⇒
   `drop_routing_keep_selection`; G2t fail ⇒ `no_method_win_claim` / narrow to an
   analysis-method contribution; drift/cost ceiling fail ⇒ remove the
   corresponding reliability/deployability claim). The negative result is reported
   as-is; no claim is upgraded past its evidence.

This loop changes nothing about the v5 estimand or multiplicity; it only formalizes
the "if it does not beat the baselines, fix the method, not the comparison" stance.

---

## E. Single-RTX-4090 serial schedule (kill-gate-first ordering)

The authorized run targets **one RTX-4090 (24 GB), serial**. To avoid burning the
compute envelope on a method that will not clear an early gate, run **kill-gates
first** and stop on the first hard failure:

1. **Cheapest gate first.** Faithful NAIT over `L` + endpoint signatures + the R0
   mechanism certificate (CPU/GPU-light selection-side work) **before** any
   six-arm training. If **R0** fails its partial-`R²` floor, stop:
   `stop_main_novelty_claim` — no training compute is spent.
2. **R1 routing arms next**, serially: train the six masked-LoRA arms over 20
   seeds one at a time (no parallel GPUs), checkpoint per `(arm, seed)`. If **R1**
   fails the IUT vs the routing controls (AdaLoRA/NeFT included), stop:
   `drop_routing_keep_selection`.
3. **Baseline selection runs**, serially, reusing the same single 4090: each
   official-repo selector scores OUR pool, then OUR identical LoRA training trains
   its top-B. Order the baselines cheapest-scoring first (e.g. IFD/Nuggets
   perplexity scores) so an early dominating result is visible sooner.
4. **R2 compute-matched comparison + closed-testing decision** last. Per-arm and
   per-baseline realized FLOPs/wall-clock are **measured** by the cost ledger, not
   predicted; an extraction-parity multiplier `> 2.0×` declares high-cost-analysis
   with no deployment claim.

This respects the frozen compute envelope (`first_gate_budget.yaml`,
`matched_lora_training_gpu_hours: 24`, `buffer_percent: 30`) on a single device:
gates that can kill the claim cheaply run before the expensive training arms, and
nothing runs until `server.authorized: true` AND `--i-have-authorization` (see
`reports/run_packet.md` §2).

---

## Invariants preserved from v5 (unchanged)

- The frozen estimand (pool-conditional policy value `V(π)`; two outcome symbols
  `Y_obs` / `U_train` never conflated).
- The trajectory operator, coupling identity, residualized attribution, and the
  score-free `capacity_match` map.
- The closed-testing graph, IUT leaves, and the in-document FWER `≤ 0.05` bound.
- All locked margins/capacity/floors/seeds in `configs/experiments/lattice_v5.yaml`.
- `server.authorized: false`; the green 93-test pure-stdlib harness; the
  build-now/run-later guard. **No result is claimed.**

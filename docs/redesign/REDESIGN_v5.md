# REDESIGN v5 — neurotrace-it (Stage-1 Registered Report, top-venue elevation)

Status: **BUILD-NOW / RUN-LATER Stage-1 Registered Report.** `server.authorized`
stays **false**. NO experiment, training, extraction, or model load is run by this
document. ZERO fabricated **results** — every empirical results slot is a
`DATA_NEEDED` placeholder. Pre-registered **decision thresholds** (margins, capacity,
seeds, the closed-test α-allocation, the partial-R² floor) are **design choices and
are now LOCKED** to concrete values in `configs/experiments/lattice_v5.yaml` (§5.6);
they are not results. This file was originally **design-only**; the Phase-B modules
and the populated `lattice_v5.yaml` it specifies **have since been implemented and
locked** (see the implementation-status note below). The `src/` and `paper/` artifacts
it describes are now in place; `docs/pre_registration.md` is unchanged.

> **Implementation-status update (post-design; authoritative, mutually consistent
> with `paper/main.tex` and `reports/run_packet.md`).** The Phase-B modules
> this design specifies **have since been implemented** under
> `src/neurotrace_it/` as additive, **do-not-run, pure-Python** (stdlib only; no
> model load, no server call, no training), and are exercised by a **GREEN unit
> harness of 93 passing** formula-evaluation tests
> (`python -m pytest -q` ⇒ `93 passed`, exit 0; per-file: `test_layer_routing.py`
> 26, `test_trajectory_selection.py` 35, `test_residualize_gates.py` 19,
> `test_endpoint_baseline.py` 7, `test_selection_schema_metrics.py` 3,
> `test_project_contracts.py` 3). `configs/experiments/lattice_v5.yaml` exists
> (additive; multiplicity-structure constants pinned; pre-registered DECISION
> thresholds/margins now LOCKED to concrete values with provenance; only empirical
> results remain `DATA_NEEDED`). **`server.authorized: false` and NO run has been
> executed** —
> the tests are code-correctness checks, not evidence. Therefore the §5.5 phrases
> "REQUIRED; currently absent" and the closing "confirmed absent" are **historical
> Phase-B planning, now discharged**; the *current* status is **implemented &
> unit-tested, not run**. See `reports/run_packet.md` §1 (status table) and `main.tex`
> Table 1. There is no live "pending implementation" contradiction across the
> three documents.

This is a **response-to-review revision** that incorporates two independent
adversarial reviews (GPT-5.5 and an independent Opus verifier) of the prior v5
draft, and preserves the verified-correct v4 core
(`docs/redesign/REDESIGN_v4.md`). The reviews converged on **one fatal flaw** in the
original headline estimand (it was **ill-posed**: per-example causal contrasts under
fixed capacity, with cross-example SUTVA failure). v5 (round 1) repaired the estimand
at the equation level by reframing it as a **policy value over whole training runs**.
A re-gate (round 2) added the policy-domain / out-of-sample / margin-test scaffolding.
**A subsequent statistical re-gate (round 3, this document) found that the round-2
*inference* was still unsound on four counts** — all purely statistical, not
conceptual — and round 3 fixes each precisely. The policy-value reframe itself is
**verified sound and is preserved unchanged** (no more SUTVA, no ill-posedness); the
remaining defects were in (a) the policy's internal specification, (b) the *direction*
of the multiplicity argument, (c) an over-claimed exactness, and (d) an
estimand/estimator mismatch.

---

## v5 fix log (round 4 / RE-FIX) — what changed and why it is now correct

A round-4 re-gate accepted rounds 1–3 (sound policy-value reframe, correct IUT
direction, honest asymptotics, pool-conditional scope) but found **three residual
soundness blockers** — two specification gaps and one missing in-doc proof. All three
are now closed. **Nothing from rounds 1–3 is reverted; the policy-value estimand and its
IUT/bootstrap-`t` inference stand.**

| # | Round-4 blocker (re-gate) | Why it was still fatal | Round-4 fix (where proven) |
| --- | --- | --- | --- |
| **RE-FIX-1** | **The factuality calibrator's strict propriety was not proven in-doc.** The doc only asserted "Brier propriety + G6" without the in-doc proof that the calibrator's scoring rule is strictly proper, so the calibrator's optimality (truthful reporting) was unestablished. | A scoring precondition (G6) that licenses the `λ_f f̂` term must be backed by an in-doc strict-propriety proof, not a bare citation; otherwise the calibrator's optimality (truthful reporting) is unestablished. | **§3.8a added in full:** NeuroTrace-IT's **strictly-proper Brier calibrator** (Eq.8-1, the v4 §2.6 rule carried into v5) with the **strict-propriety proof** `argmin_q E_{Bern(p)}(q−y)² = p` uniquely via `E(q−y)²=(q−p)²+p(1−p)` (**Prop. P-RC1**, Eq.8-2). Propriety proves calibration, not eval-drift prediction; G6 stays a precondition (fail ⇒ `λ_f:=0`). |
| **RE-FIX-2** | **`capacity_match` still read `ψ` and still had an infeasible branch.** The old `CAP_SPILL` grew support "by descending `ψ_ℓ`", but `ψ` is **not** an argument of `capacity_match(m_A,R_tot,r_max)`; and `|A|·r_max≥R_tot` only made the *all-anchor* mask feasible, not every cardinality `k`. For `k·r_max<R_tot` the branch fired ⇒ implementation-dependence reopened. | If the shared rank map's output depends on a non-argument (`ψ`) or has an undefined/implementation-chosen branch, two faithful implementations can disagree on `U_train(π)` ⇒ `V(π)` is again ill-defined — exactly the R3-B1 failure mode. | **`capacity_match` is now a pure function of `(m_A,R_tot,r_max)` on the FEASIBLE domain `DOM`** (Eq.6-0b0/0b/0d): it **reads no `ψ`**, **never grows support**, and has **no infeasible branch** (feasibility/termination lemma). Feasibility is guaranteed **upstream** by **`make_feasible_mask`** (Eq.6-0e), the *only* place an arm's score is read; the empty mask is lifted to uniform-over-`A` there. (§3.5, Props. in §3.5.) |
| **RE-FIX-3** | **R1 control policies were prose, not deterministic maps.** `π_shuf/π_rand/π_global/π_ada` left the random-mask RNG, shuffle seed, strata, global block, AdaLoRA top-`k`, and whether randomization is fixed or part of `V(π)` unpinned ⇒ not single implementation-independent objects. | A policy-value estimand `V(π_c)` requires each control to be **one** object; an unpinned RNG makes `V(π_c)` implementation-dependent and the gap `g_c` undefined. | **Each control is now a single deterministic map of `x`** (§3.5, Eq.6-0e arms 2–6): `π_rand/π_shuf` driven by **fixed pre-registered seeds** `seed_rand/seed_shuf ⊕ H(x)`; `π_global` a **fixed block `A_glob`**; `π_ada` a **fixed warm-up `W_ada`** then top-`k` by importance (ties ascending-`ℓ`). The fixed randomization is **part of the policy definition, NOT the `V(π)` seed-expectation** (which is over training seeds only). |
| **RE-FIX-4** | **`π_ada` determinism was internally inconsistent.** §3.5 declares every control a deterministic `x↦mask` map with control randomization **NOT** part of `V(π)`, but `π_ada`'s importance came from a warm-up run that read the **per-run training seed** (round-3 control 6 said so explicitly) ⇒ `π_ada(x)` was non-deterministic in `x` and a **second** seed (the warm-up) lived inside the `V(π)` expectation. | A deterministic-map framing and a seed-measurable warm-up cannot coexist: either `π_ada` is not a single object, or `V(π)` averages over two seeds while the gap `g_c` is defined for a single one. Both break the policy-value estimand. | **The AdaLoRA importance warm-up is frozen by a SEPARATE pre-registered seed `seed_ada` OUTSIDE the training-seed expectation** (§3.5 control 6; `lattice_v5.yaml controls.seed_ada`, `ada_warmup_seed_outside_V: true`). The warm-up runs **once**, its importance is **persisted** (`schemas_v2.control_provenance`) and reused by every confirmatory seed, so `π_ada(x)` is a deterministic function of `x` and the **only** seed inside `V(π)` is the per-run training seed. |
| **RE-FIX-5** | **Closed-test intersections containing the R1/G2t union null could drop to an in-scope subset of components.** §4.1b item 3 tested an intersection `H_I ⊇ H0^{R1}` by an IUT over only the controls "in scope" of `I`, claiming "an IUT over a subset is still an IUT." But `H0^{R1}=⋃_c H0^{R1,c}` is a **single** elementary union null; a level-α test of it must reject the **whole** union. | An IUT over a proper subset of `R1`'s components tests `⋃_{c∈subset}H0^{R1,c} ⊊ H0^{R1}`; when the **binding** control (`g_c=δ_R1`) is the omitted one, the subset test rejects while `H0^{R1}` is true ⇒ the local test is **not** level-α for `H_I` ⇒ the closed-test FWER proof has a real hole (anti-conservative). | **Every closed-test intersection containing `H0^{R1}` (or `H0^{G2t}`) now uses the FULL-union IUT — ALL five control contrasts (resp. all baseline comparators) must clear the margin** (§4.1b items 1, 3; Table 4.1; Props. P1-IUT/P1-FWER updated). `R1` and `G2t` are single union-null leaves of the 6-member closed family; their internal components are never separate members and are never sub-scoped. Requiring the full union inside every intersection only makes rejection **harder**, so FWER ≤ 0.05 is restored (`closed_testing.py` enforces the union-null leaf rule; `test_layer_routing.py` item (f) checks it at the least-favorable single-binding-control config). |

**Net effect of round 4.** The factuality calibrator's strict propriety is now
**proven in-doc** (RE-FIX-1, NeuroTrace-IT's strictly-proper Brier rule); the
shared rank map `capacity_match` is a **pure, score-free, total function** on a feasible
domain with the `ψ`-dependence quarantined in a single constructor (RE-FIX-2); every
R1 control is a **single deterministic implementation-independent policy** (RE-FIX-3),
with `π_ada`'s importance **frozen by a separate `seed_ada` outside `V(π)`** so the only
seed inside the policy-value expectation is the training seed (RE-FIX-4); and the
closed-testing FWER hole is closed by requiring the **FULL R1/G2t union null inside
every intersection that contains it** (RE-FIX-5). Combined with the preserved rounds
1–3, the central policy-value estimand `V(π)` is now fully specified (every arm one
object), its inference correct in direction and honest in guarantee, the multiplicity
control sound on the corrected leaves, and every scoring precondition proven where used.

---

## v5 fix log (round 3) — what changed and why it is now correct

Round 1 (estimand = policy value) and round 2 (policy domain `A`, frozen
out-of-sample nuisance map, margin tests) were accepted **in their conceptual
content** but the round-3 statistical re-gate found **four equation-level blockers**
that made the *confirmatory inference* invalid or the *policy* under-specified. Each
is now fixed and proven below where named. **The policy-value reframing is NOT
reverted — it is verified sound and preserved. Round 3 corrects only the statistics.**

| # | Round-3 blocker | Why it was still fatal | Round-3 fix (where proven) |
| --- | --- | --- | --- |
| **R3-B1** | **Policy under-specified inside `A`.** The map from the anchor mask `m_A(x)` to per-layer LoRA ranks `r_ℓ(x)` (`capacity_match`) was a deferred module/YAML name, not a mathematical definition. Zero-support, ties, variable mask cardinality, and a per-layer rank cap with redistribution were all unresolved ⇒ `U_train(π)` was **not a single implementation-independent functional**. | A policy value `V(π)=E[U_train(π)]` is only defined once `U_train` is a *single* number for each `π`. With `capacity_match` undefined, two faithful implementations could produce different rank vectors for the *same* mask ⇒ two different `U_train` ⇒ no well-defined `V(π)`. | **`capacity_match` is now a closed-form, deterministic apportionment given in full (Eq. 6-0b).** It is the **largest-remainder (Hamilton) rule** distributing `R_tot` over the `\|m_A(x)\|` selected anchor layers as evenly as possible, with a **deterministic lexicographic tie-break**, an explicit **zero-support fallback policy** (Eq. 6-0c), explicit **renormalization for any mask cardinality** (built into the apportionment), and a **per-layer rank cap `r_max` with deterministic spillover** (Eq. 6-0d). `U_train(π)` is therefore one implementation-independent functional. No deferred `capacity_match`. (§3.5, §3.5a.) **[Round-4 RE-FIX-2 completes this: the map is made `ψ`-free and total on a feasible domain, with the empty-mask fallback / cardinality lift moved upstream to `make_feasible_mask` (Eq.6-0e); see the round-4 fix log.]** |
| **R3-B2** | **Multiplicity in the WRONG DIRECTION (load-bearing).** R1 (`π_ψ` beats *every* control) and R2-target (beats *every* comparator) are **intersection-union** hypotheses (null = *union* of per-contrast nulls; alternative = *intersection* of per-contrast alternatives). The round-2 doc used `Δ_R1 = min_c g_c` with a **shared min-over-controls (max-T) critical value** calibrated to the *joint* null — that is the procedure for the *opposite* (union-alternative / global-null) problem and is **anti-conservative at the least-favorable configuration**, where one control binds and the others are far. CI under-coverage came from the same error. | A max-T / min-over-controls shared quantile tests "does `π_ψ` beat *at least the worst-calibrated* control"; it borrows strength across contrasts and rejects too easily when only one control is hard. For "beats EVERY control" the type-I rate at the boundary (one `g_c=δ`, others `≫δ`) can exceed `α`. | **R1 and R2-target are now decided by an INTERSECTION-UNION TEST (Berger 1982).** Reject the composite "beats all controls" **iff EACH pairwise contrast `g_c > δ` is individually significant at the MARGINAL level `α`** (Eq. 7-IUT). An IUT requires **no multiplicity penalty** and is exactly level-`α` (size ≤ `α`, proven in Prop. P1-IUT). There is **no shared min-over-controls quantile**. The simultaneous lower bound reported is the **min over per-contrast marginal-`α` lower confidence bounds** (Eq. 7-IUT-CI). (§3.5, §4.2.) |
| **R3-B3** | **Per-contrast test dishonest about exactness.** The round-2 doc called the paired seed-difference test *exact at finite `S`* via seed sign-flip, justified by "i.i.d. seeds ⇒ sign-symmetry." **That implication is false:** i.i.d. seeds make the paired differences `d_c,s` i.i.d., but **not symmetric about their mean**, so the Rademacher sign-flip reference is **not** the exact finite-`S` null. The exactness claim was an overclaim. | Sign-flip / permutation exactness needs the centered differences to be **sign-symmetric** (or an actual randomization that induces it). i.i.d. draws give exchangeability of the `S` differences among themselves, not sign-symmetry of each. So the sign-flip critical value is at best asymptotically valid. | **Each per-contrast test is restated as a paired studentized bootstrap-`t` (equivalently a studentized permutation/bootstrap) with ASYMPTOTIC validity (Eq. 7-Bt); the word "exact" is removed.** We additionally state the **one** way exactness *could* be recovered — an **actual randomized seed-to-arm assignment** that makes sign-symmetry hold by design — and explicitly note we do **not** adopt it (arms are deterministically paired on the same seed), so we claim only asymptotic validity. No overclaim. (§4.2, Prop. P1-Bt.) |
| **R3-B4** | **Estimand/estimator mismatch.** `V(π)` was defined as an expectation over training **seeds AND data draws**, but R1 estimates only a **seed-mean on one fixed `P_val`** (the pools are locked). The estimator targets a *different* (pool-conditional) quantity than the *marginal* estimand it was attached to ⇒ the CI does not cover the named estimand. | A seed-mean at fixed `P_train/P_val/P_dep` is unbiased for the **pool-conditional** value `V_cond(π)=E_seed[U_train(π)\|pools]`, not for the pool-marginal `E_{seed,draws}[U_train(π)]`. Reporting a CI for the latter from the former under-covers (it omits the across-pool variance component). | **`V(π)` is REDEFINED as the pool-conditional value** `V(π)=E_{seed}[U_train(π)\|P_train,P_val,P_dep]` (Eq. 6-2), and **every confirmatory claim is explicitly scoped to the locked pools**. The seed-mean estimator now matches the estimand exactly; the only resampled randomness is the seed, which is what the bootstrap-`t` resamples. A **named, non-confirmatory generalization probe** (multi-pool re-split, §3.5a) is offered as the route to the pool-marginal claim, with its own multiplicity accounting, and is **not** part of the confirmatory `α`. Estimand and estimator are aligned. (§3.5a, §3.0.) |

**Preserved (verified-sound; carried through unchanged):** the **policy-value
estimand reframing** (`V(π)`, run-level treated unit, **no SUTVA**, no per-example
potential outcome — verified sound, round 1); the **coupling identity**
`Σ_{ℓ∈A} ψ_ℓ = β̂_T · T̃` (selection = the sum, routing = the support of **one**
endpoint+NAIT-residualized attribution); the **compute-matched ledger**; the
**three-pool leakage firewall**; the **faithful NAIT reproduction over the released
layer set `L`**; the **expanded baseline set**; the frozen out-of-sample nuisance map
`B_λ` and the closed-form `ψ(x)`; the margin-test framing (gap `≤` margin);
`server.authorized: false`; zero fabricated numbers.

The net effect of round 3: the policy `π_ψ` is now a **fully-specified
implementation-independent functional** (R3-B1: `capacity_match` is closed-form); the
"beats every control / every comparator" decision uses the **correct
intersection-union test with no multiplicity penalty and exact level `α`** (R3-B2);
each per-contrast test claims only the **asymptotic validity it actually has** (R3-B3:
no false exactness); and the **estimand is the pool-conditional value the seed-mean
actually estimates** (R3-B4). The central policy-value estimand is sound and its
inference is now both **correct in direction** and **honest about its guarantees**.

---

## v5 fix log (round 2) — what changed and why it is now correct (retained)

Round 2 took the round-1 reframing and (a) pinned the routing **domain** to the anchor
set `A` with `L∖A` a fixed common substrate; (b) replaced the in-sample residual
operator `M_λ` by a **frozen out-of-sample coefficient map** `B_λ`; (c) framed every
confirmatory null as a **margin** null. These three deltas are **kept**; round 3 only
corrects the *statistical inference* layered on top of them. The round-2 table is
retained for provenance.

| # | Round-2 blocker (re-gate) | Round-2 fix (kept) |
| --- | --- | --- |
| **R2-B1** | Routed policy `π_ψ` ranged over `{0,1}^L` but `ψ`,`T` only scored `A ⊊ L` ⇒ `L∖A` masks/ranks undefined. | Routing domain made **exactly `A`** (Eq.6-0); every `ℓ∈L∖A` pinned at a fixed policy-independent baseline rank `r_0` shared by all arms (Eq.6-0a). `L∖A` cancels in every contrast. **Round 3 completes the *intra-`A`* spec** (capacity_match, R3-B1). |
| **R2-B2** | Residual operator `M_λ` was an in-sample `N×N` hat; undefined for new `P_val`/`P_dep` rows. | Replaced by a **frozen coefficient map** `B_λ=(ZᵀZ+λΩ)⁺ZᵀT`; `T̃(x)=T(x)−Z(x)B_λ` for any `x` (Eq.5-1/5-1d). Kept. |
| **R2-B3** | Margin claim from a CI that only cleared `0`. | Every confirmatory null is the **margin** null; rejection requires the lower CI bound to clear the **margin** (Eq.6-G). Kept; round 3 fixes *how* the bound is computed (R3-B2/B3). |
| **R2-B4** | FWER asserted conditionally; finite-seed validity and weights deferred. | Closed-testing lattice with pinned weights (§4.1) is kept; **round 3 replaces the round-2 max-T leaf and its false finite-seed exactness with the correct IUT + asymptotic bootstrap-`t`** (R3-B2/B3) and re-proves FWER on the corrected leaves (Prop. P1-IUT). |

---

## v5 fix log (round 1) — what changed and why it is now correct (retained, VERIFIED SOUND)

Both reviewers rejected the prior headline estimand
`τ_ℓ(x) = Y(x, m_ℓ=1) − Y(x, m_ℓ=0)` under fixed capacity as **ill-posed**, for four
equation-level reasons. The round-1 reframing that repaired this is **verified sound
and is the load-bearing core that round 2 and round 3 build on; it is preserved
unchanged.**

| # | Blocker (both reviewers) | Why it was fatal | Round-1 fix (PRESERVED, verified sound) |
| --- | --- | --- | --- |
| **B1** | `τ_ℓ(x)=Y(x,m_ℓ=1)−Y(x,m_ℓ=0)` under fixed capacity is **ill-posed**: turning layer `ℓ` off frees rank `r_ℓ` that **must be redistributed**, with no rule ⇒ `Y(x,m_ℓ=0)` is not a single potential outcome. | The "potential outcome" was a *set*, one per redistribution; the contrast is undefined. | **Estimand reframed as a POLICY VALUE** `V(π)=E[U_train(π)]` over whole training runs (§3.5). The per-layer notion, if kept, is a **descriptive** run-level leave-one-layer policy-value difference with an explicit redistribution rule (§3.6), labeled descriptive, not causal. The headline `τ_ℓ` claim is **deleted**. **Sound; preserved.** |
| **B2** | **Cross-example SUTVA fails.** A LoRA arm is **one shared training run**; example `x`'s outcome depends on every other example's mask. No per-example potential outcome exists. | SUTVA violated by construction. | The **unit of treatment is the WHOLE training run / policy** `π`. `V(π)` has no per-example potential outcome and no cross-example interference. SUTVA holds at the **run level**. **Sound; preserved.** |
| **B3** | R1 only identifies an aggregate policy-value contrast, not `τ_ℓ`; `supp_τ(ψ)` undischarged. | The measured object ≠ the named object. | Routing object redefined as **`supp(ψ)`** from the coupling identity (§3.4); R1 estimates `V(π_ψ)−V(π_control)` per control arm. **Sound; preserved.** |
| **B4** | `Y(x,m)` **conflated** the observational per-example LOCI utility (R0) and the model-training outcome (routing). | Hidden identification gap. | Symbol **split**: `Y_obs(x)` = R0 regression target; `U_train(π)` = R1/R2 training-outcome functional. `Y(x,m)` never appears again. **Sound; preserved.** |
| **B5** | Closed-testing FWER asserted, not established. | Type-I inflation from data-dependent maxima. | §4 specifies nodes, weights, local tests. **Round 3 supplies the correct leaf tests (IUT + bootstrap-`t`) and the FWER proof.** |
| **B6** | NAIT comparator possibly weakened (`s_NAIT` over 8 anchors vs base Eq.5 over all `L`). | Not "faithful NAIT." | §3.3 computes `s_NAIT` over the **released decoder layer set `L`**; the 8-anchor restriction is a secondary variant gated on full-`L` reproduction. **Sound; preserved.** |
| **B7** | R0 permutation called exact without specifying exchangeability or per-permutation ridge refit. | "Exact" unjustified. | §3.2 specifies a **cross-fit / frozen-nuisance** permutation; guarantee softened to asymptotically valid / finite-sample approximately-exact under within-stratum residual exchangeability (Prop. P0). **Sound; preserved.** |

**Note on the round-2 / round-1 interaction.** Round 2 sets the *routing domain* to
`A` (R2-B1). The base-paper-faithful NAIT comparator (B6) is **still computed over the
full `L`** — a property of the *control/comparator*, not of the *routing policy*.
Routing acts on `A`; NAIT's score is read over all of `L`. Both requirements hold
simultaneously (§3.3 vs §3.5).

---

## 0. TL;DR (one screen)

- **The sharpened central contribution, stated narrowly:** selection and routing
  are two readouts — the **sum** and the **support** — of **one**
  endpoint+NAIT-residualized trajectory attribution `ψ(x)`, validated by a
  **policy-value** routing test (six real training arms), not by a predictive
  decomposition alone. The coupling identity `Σ_{ℓ∈A} ψ_ℓ = β̂_T · T̃`
  (§3.4, Eq.5-4) is the *mathematical* spine; the policy-value gate R1 (§3.5) is the
  *empirical* spine.

- **The estimand is a POLICY VALUE, not a per-example causal contrast (verified
  sound).** A routing **policy** `π_ψ` allocates the fixed LoRA capacity `R_tot`
  **over the anchor set `A`** to the support of the top-`τ_sel` layers of `ψ`, while
  all non-anchor layers `L∖A` carry a **fixed, policy-independent baseline rank `r_0`**
  common to every arm. The treated unit is the **whole training run**, so there is
  **no per-example potential outcome and no cross-example SUTVA**. The confirmatory
  estimand is the **pool-conditional** value gap `V(π_ψ) − V(π_control)` per control
  arm — exactly what the six-arm R1 measures (R3-B4).

- **The policy is now a FULLY-SPECIFIED implementation-independent functional
  (R3-B1; RE-FIX-2/3).** The mask→rank map `capacity_match` is the closed-form
  largest-remainder apportionment of `R_tot` over the selected anchor layers, with a
  deterministic ascending-`ℓ` tie-break, mask-cardinality renormalization, and a
  per-layer cap with deterministic spillover (Eq.6-0b/0d). It is a **pure score-free
  function of `(m_A,R_tot,r_max)`** on the feasible domain `DOM` — it reads no `ψ` and
  has no infeasible branch; feasibility and the empty-mask fallback (uniform-over-`A`)
  are handled **upstream** by `make_feasible_mask` (Eq.6-0e), the single place an arm's
  score is read. Every R1 control (`π_rand/π_shuf/π_global/π_ada`) is likewise a single
  deterministic map of `x` under fixed pre-registered seeds/blocks (RE-FIX-3).
  `U_train(π)` is one number per `π`.

- **"Beats every control / every comparator" is an INTERSECTION-UNION TEST (R3-B2).**
  Reject iff **each** pairwise contrast clears its **marginal-`α`** bound — no
  multiplicity penalty, exact level `α` (Berger 1982). **No** shared min-over-controls
  quantile. The reported simultaneous lower bound is the **min of the per-contrast
  marginal-`α` lower bounds**.

- **Each per-contrast test is an asymptotically valid paired studentized bootstrap-`t`
  (R3-B3).** The earlier "exact finite-`S` sign-flip" claim is **withdrawn** —
  i.i.d. seeds do not give sign-symmetry of the paired differences. We claim only the
  asymptotic validity we have.

- **Estimand and estimator are aligned (R3-B4).** `V(π)` is the **pool-conditional**
  seed-expectation at fixed `P_train/P_val/P_dep`; the seed-mean estimates exactly
  that; the claim is scoped to the locked pools. A separate, non-confirmatory
  multi-pool probe is the route to a pool-marginal statement.

- **The deployed attribution is a closed-form function of any `x` (R2-B2).** Nuisance
  is a **frozen coefficient map** `B_λ` fit once; `T̃(x) = T(x) − Z(x)B_λ` and `ψ(x)`
  are computable on `P_val`/`P_dep` rows the map never saw.

- **Two outcome symbols, never conflated (B4).** `Y_obs(x)` = observational
  per-example LOCI utility (R0 regression target). `U_train(π)` = held-out
  training-outcome functional (R1/R2). `ψ` is fit to predict `Y_obs`; the routing
  *policy built from* `ψ` is scored by `U_train`.

- **Every confirmatory test is a MARGIN test (R2-B3).** The null is "gap `≤` margin";
  rejection requires the **lower** bound (per-contrast, marginal-`α`) to **exceed the
  margin**, not merely `0`.

- **Verified-correct v4 cores preserved** (Gate R0 mechanism certificate): dual-ridge
  FWL endpoint+NAIT-residualized partial-R² estimand; conditional-null block
  permutation on the **orthogonalized residuals** `T̃`, with **frozen cross-fit
  nuisance**; outcome sign `+`; two-layer Holm as one node of a closed-testing graph.

- **Three-pool leakage firewall:** learn `ψ` on `P_train`, estimate `V(π)` on
  `P_val`, deploy on an untouched `P_dep` where selection/routing read activations
  only.

- **Falsifiable non-stitch chain:** R0 (residual `ψ` exists) ⇒ R1
  (`V(π_ψ) > V(π_control) + δ_R1` for every control, each by its own marginal-`α`
  bound, in real matched-compute arms) ⇒ R2 (selection+routing wins at a measured
  compute-matched budget). Each arrow has a pre-registered fallback.

- **Multiplicity is a formal closed-testing graph whose data-dependent "beats all"
  leaves are intersection-union tests** (no penalty, exact level `α`) and whose
  per-contrast inference is an asymptotic bootstrap-`t`; **FWER ≤ 0.05 is proven
  in-document** (§4) on the corrected leaves.

- **Phase-B modules** (Phase A here is design-only): `layer_function.py`,
  `cost_model.py`, `baselines/nait_layerwise.py`, `analysis/layer_attribution.py`,
  `analysis/routing_intervention.py`, `analysis/matched_budget.py`,
  `analysis/closed_testing.py`, `analysis/pool_firewall.py`, plus the `schemas_v2.py`
  `RouterOutputs` population and `configs/experiments/lattice_v5.yaml`.
  `server.authorized: false` throughout.

---

## 1. Central contribution and its novelty

### 1.1 The sharpened thesis (one sentence, narrow)

> **Trajectory-Coupled Selection-and-Routing (LATTICE-R).** One per-example
> endpoint+NAIT-residualized trajectory attribution `ψ(x) ∈ R^{|A|}` drives **both**
> the data-selection decision (keep the budget-`B` examples scoring high on
> `Σ_{ℓ∈A} ψ_ℓ`) **and** the LoRA layer-routing **policy** `π_ψ` (allocate the fixed
> rank `R_tot` over the anchor set `A` to `supp(ψ)`, holding `L∖A` at a fixed common
> baseline); LATTICE-R beats a *faithfully reproduced* NAIT selector (scored over the
> full `L`) on retention-adjusted target utility at a **measured compute-matched**
> budget, and a **policy-value** routing test certifies that the *coupling* (same
> vector for both decisions) is load-bearing — measured as the value gap
> `V(π_ψ) − V(π_control)` of whole training runs, not as a per-example causal
> contrast.

The contribution is the **coupling validated as a policy value**, not either piece
alone. The selection score and the routing support are the **sum** and the
**support** of the *same* well-defined attribution `ψ`, tied by the coupling
identity `Σ_{ℓ∈A} ψ_ℓ = β̂_T · T̃` (proven in §3.4).

### 1.2 Novelty vs NAIT (the base paper) — precise, from the base algorithm

NAIT's actual algorithm (base paper Algorithm 1, §3.2.1–3.2.2; Eqs. 1–6) is:

1. For each in-domain sample, record decoder-layer activations across **all**
   decoder layers `L`; per layer the activation vector is
   `A^(ℓ)(t_k) = [a_j^{(k)}]_{j=1}^{J}` (base Eq.1); the relative change is
   `ΔA^(ℓ)(t_k) = A^(ℓ)(t_K) − A^(ℓ)(t_1)` (base Eq.2). §3.2.1 prose **also** states
   "compute the mean activation across all `K` tokens" — so the base paper is
   **internally inconsistent** (Eq.2 uses first/last; the prose uses a token mean).
2. Per layer, `v_ℓ = PCA(ΔA^(ℓ))` (first PC; base Eq.3), sign-aligned via
   `μ_diff·v_ℓ ≥ 0` with `μ_diff = |P|⁻¹ Σ (A^(ℓ)(t_K) − A^(ℓ)(t_1))` (base Eq.4).
3. Score each candidate by the **layer-sum** projection over **all** decoder layers
   `s_y = Σ_{ℓ=1}^{L} (A^(ℓ)(y) · v_ℓ)` (base Eq.5) and select `top-k` (base Eq.6).

(Base model: LLaMA-2-7b, so `L` is its 32 decoder layers.)

**Crucial correction for the comparator (B6):** base Eq.5 sums over `ℓ = 1..L`
(**every released decoder layer**), not over an 8-anchor set. v5's faithful NAIT
(§3.3) therefore sums over `L`; the 8-anchor restriction is only a *secondary*
variant gated on the full-`L` reproduction. **This is independent of the routing
domain** (which is `A`, R2-B1): NAIT *scoring* is over `L`; LATTICE-R *routing* is
over `A`.

NAIT's structural commitments and the open axis v5 exploits:

- **(N1) Sequence collapse to a low moment.** NAIT summarizes a sequence by a
  first/last difference (Eq.2) or a token mean (prose) — both permutation/shape
  blind. v5's `D_ℓ` (sliced-Wasserstein-2 to a target cloud, Eq.4) and `κ_ℓ`
  (ordered bending energy, Eq.5) are **provably not functions of that summary**
  (§3.4 Prop.; unit-tested in `trajectory.py`). *Kept v4 novelty.*
- **(N2) Layer collapse to a sum, then top-k.** NAIT's `s_y = Σ_ℓ (A^(ℓ)·v_ℓ)`
  destroys *which* layer carried alignment and tunes all layers uniformly. v5 keeps
  the per-layer axis as a **decision variable** for selection *and* capacity
  allocation. *Contribution NAIT structurally cannot express.*
- **(N3) Selection-only → joint selection+allocation, coupled through one
  attribution.** NAIT answers only "which examples." v5 also answers "given this
  example's internal evidence, **where** should its gradient land," and *ties both
  answers to one vector `ψ`*.

The narrow novelty unit is **(N2)+(N3) coupled through one attribution and certified
by a policy-value routing test** — distinct from NAIT, which never conditions the
training footprint on the per-layer evidence it already computes.

### 1.3 Named prior art and exactly how v5 differs

| Prior art (specific) | What it does | Why v5 is not it |
| --- | --- | --- |
| **NAIT** (base paper, ICLR 2026; Alg.1, Eq.1–6) | first/last-diff (or token-mean) per layer → per-layer PCA dir `v_ℓ` → **layer-sum over all `L`** → top-k | v5 keeps the per-layer axis as a *routing* decision; R0 residualizes `Y_obs` against a **faithful NAIT Eq.5 layer-sum over `L`** (§3.3), not just `concat[start,end]` |
| **LESS** (Xia et al., ICML 2024, 2402.04333) | low-rank gradient datastore → select examples whose grads align with target; example-level | v5 *reuses* an influence/gradient attribution only to **define `Y_obs`** (method-neutral firewall); the *selector* is trajectory-geometric and adds **routing** LESS never addresses; LESS is in the baseline set |
| **TracIn** (Pruthi et al., 2020) | training-run gradient-trace influence | used to *define* `Y_obs`, not as the method; no layer allocation |
| **AdaLoRA** (Zhang et al., 2023, 2303.10512) | SVD-parameterized adapters; reallocates rank **globally during training** | v5 reallocates rank **per-example, a priori, from trajectory geometry before training**, then *measures whether that beats AdaLoRA's global allocation* (AdaLoRA is a required R1 arm) |
| **Surgical Fine-Tuning** (Lee et al., ICLR 2023, 2210.11466) | choose one *fixed* layer block to tune for a *whole task* | v5 routes **per example**; global-selective tuning is a required R1 control arm |
| **NeFT** (Xu et al., COLING 2025, 2025.coling-main.630; arXiv:2403.11621) | neuron-level selective tuning, task-global | v5's decision is per-example and *coupled to the selection score*; NeFT added to baselines |
| **Sparsely-gated MoE routing** (Shazeer et al., 2017, arXiv:1701.06538) | route tokens/experts at inference/training via a learned gating network | v5's "router" is an **offline data-conditioned capacity mask** from frozen activations, not a learned inference-time gate; added to baselines |
| **"A Critical Look at Targeted Instruction Selection"** (2602.14696) | argues selectors are entangled, **random is often competitive**, gradient reps most stable | v5 **adopts this as a threat model**: random, full-data, zero-shot, gradient-rep selectors are all required baselines; R2 is gated on beating the *strongest* comparator, not NAIT alone |
| **Sliced-Wasserstein / MMD** (Bonneel 2015; Gretton 2012) | distributional distances | tools, not the contribution; reused for `D_ℓ` |
| **Double-ML / FWL** (Chernozhukov 2018; Frisch–Waugh–Lovell) | orthogonalized partial effects | reused **verbatim** as the mechanism-certificate estimand (§3.2) |
| **Intersection-Union Test** (Berger 1982, *Technometrics*) | level-`α` test of a union null = reject iff every component rejects at `α`, no penalty | reused **verbatim** as the "beats every control/comparator" decision (R3-B2, §4.2) |

**One-line differentiation:** *NAIT sums the layer axis away (over all `L`) and tunes
uniformly; LATTICE-R keeps the layer axis as a per-example decision variable for
selection and capacity allocation over `A`, derives both from one
endpoint+NAIT-residualized attribution (`Σ_{ℓ∈A} ψ_ℓ = β̂_T·T̃`), and proves under
real matched-compute training runs that the routing policy's value `V(π_ψ)` beats
**every** prior-art arm (random, AdaLoRA, surgical, NeFT, MoE-routing) at a
compute-matched budget — each by its own marginal-`α` margin bound (intersection-union
test).*

---

## 2. Response to every must-fix item (both reviews)

GPT-5.5 (6/10) raised nine must-fixes; the independent Opus verifier raised the
ill-posed-estimand cluster (B1–B4); round-2 raised R2-B1..R2-B4; the round-3
statistical re-gate raised R3-B1..R3-B4. The estimand cluster is the verified-sound
round-1 fix; round 2 added the domain/map/margin scaffolding; round 3 corrects the
inference. The nine GPT-5.5 items are dispositioned below.

| # | Must-fix | v5 disposition | Where |
| --- | --- | --- | --- |
| **MF-1** | Faithful NAIT (first/last + token-mean), resolve the prose/Alg.1 inconsistency, residualize R0 against NAIT Eq.5 **over `L`**. | **ACCEPTED.** §3.3 reproduces NAIT over the **full released `L`** (base Eq.5), pre-registers both variants, gates "beats NAIT" on the stronger; 8-anchor is secondary (B6). R0 control block extended to `[φ_end, s_NAIT, V_proj, C, 1]`. | §2.2/§3.3, §3.2 |
| **MF-2** | Define the routing estimand soundly; `ψ_ℓ` is a policy score. | **ACCEPTED + REPAIRED (R1+R2+R3).** §3.5 makes the estimand a **policy value** `V(π)` over `A` (R2-B1) with a **fully-specified** mask→rank map (R3-B1); `ψ_ℓ` is the **policy score**; routing support = `supp(ψ)` over `A`, deployed via the frozen map (R2-B2). | §3.4, §3.5, §3.5a |
| **MF-3** | R1 = actual training intervention (six arms, paired held-out utility). | **ACCEPTED.** §3.5: six real masked-LoRA arms, paired `U_train` contrast; estimates `V(π_ψ)−V(π_control)` directly; **intersection-union** margin decision (R3-B2). | §3.5 |
| **MF-4** | Prevent outcome leakage (three pools). | **ACCEPTED.** §3.1 three-pool firewall; `Y_obs`/`U_train` never touch `P_dep` before the decision. | §3.1 |
| **MF-5** | Downgrade/validate `I_{c,ℓ}` (off-manifold). | **ACCEPTED.** §3.6 renames it `J_{c,ℓ}` (frozen layer-ablation profile), validated-or-dropped vs a real layer-freeze policy-value difference. | §3.6 |
| **MF-6** | Formalize compute matching (params, optimizer state, FLOPs, wall-clock, skip-flag). | **ACCEPTED.** §3.7 + `cost_model.py` ledger; R2 requires measured equality; "savings" forbidden without measured reduction. | §3.7 |
| **MF-7** | Formal closed-testing/gatekeeping with **established** FWER. | **ACCEPTED + CORRECTED (R3-B2/B3).** §4 gives nodes, **pinned** α-weights, **every** local test; the "beats all" leaves are **intersection-union tests** (no penalty, exact level `α`) and per-contrast inference is an **asymptotic bootstrap-`t`**, with an in-document FWER ≤ 0.05 proof on the corrected leaves. | §4 |
| **MF-8** | Expand baselines (LESS, AdaLoRA, surgical, NeFT, MoE-routing, random, full-data, zero-shot, gradient-rep). | **ACCEPTED.** §1.3 + §5.4 lock the set; "Critical Look" threat model adopted. | §1.3, §5.4 |
| **MF-9** | Lock implementation (yaml, modules, hyperparameters, seeds, folds, margins, τ-rule, failure actions). | **ACCEPTED.** §5–§6 give equation-level specs, locked symbols, seeds, folds, margins, `τ_sel` rule, `capacity_match` rule, `failure_action` per gate. | §5, §6 |

### 2.2 Faithful NAIT reproduction — see §3.3 (computed over the full `L`, B6).

The **faithful-reproduction contract** (the Phase-A gap) is discharged by
`baselines/nait_layerwise.py` (§3.3, §5.5 module 3) and **independently asserted** by
`tests/test_endpoint_baseline.py` (§5.5 module 12): on a synthetic activation tensor the
layerwise NAIT score reproduces base Eq.5 summed over `L`
(`s_NAIT(y)=Σ_{ℓ∈L} A^(ℓ)(y)·v_ℓ`), the per-layer PCA/sign-align of Eq.2–4, the gated
8-anchor restricted sum, and the top-k selection of Eq.6, and the existing endpoint
control `baselines/nait.py` (`φ_end`) is shown to be the **distinct** endpoint-only
object (it must NOT equal the layerwise sum), so "beats NAIT" wording is gated on the
reproduced full-`L` comparator (B6).

---

## 3. Method (equation level)

Dependency order: faithful NAIT (control + comparator) → trajectory operator `T(x)`
→ endpoint+NAIT-residualized attribution `ψ` and the **coupling identity** → routing
**policy** `π_ψ` over `A` (with a fully-specified mask→rank map) and its
**pool-conditional policy value** `V(π)` → compute-matched method comparison.

### 3.0 Notation (additions to v4 §2.0) — two outcome symbols, never conflated

| Symbol | Meaning |
| --- | --- |
| `L` | the full **released decoder layer set** of the model (LLaMA-2-7b: 32 layers) |
| `A ⊆ L`, `|A|=8` | **anchor/probe layer set** for the trajectory operator `T`, the attribution `ψ`, **and the routing policy domain** (R2-B1) |
| `L∖A` | the non-anchor layers: a **fixed common substrate** held at baseline rank `r_0` by *every* arm (R2-B1) |
| `φ_end(x) ∈ R^{2d|A|}` | endpoint signature (v4 Eq.1) — FULL control |
| `s_NAIT(x) ∈ R`, `V_NAIT` | faithful NAIT layer-sum score over **`L`** and per-layer PCA dirs (§3.3) |
| `D_ℓ(x), κ_ℓ(x)` | per-layer SW2² and curvature (v4 Eq.4/5), for `ℓ ∈ A` |
| `T(x) ∈ R^{2|A|}` | trajectory signature `({D_ℓ},{κ_ℓ})_{ℓ∈A}` (v4 Eq.6) |
| `Z(x) ∈ R^{p}` | the per-example **control feature row** `[φ_end(x), s_NAIT(x), V_proj(x), C(x), 1]` (Eq.5-0); the same `Z(·)` is defined for **any** `x`, in or out of sample |
| `B_λ ∈ R^{p×2|A|}` | **frozen ridge nuisance coefficient map** for `T` (Eq.5-1); residualizes any `x` |
| `b_λ^Y ∈ R^{p}` | frozen ridge nuisance coefficient map for `Y_obs` (Eq.5-1e) |
| `T̃(x) = T(x) − Z(x)·B_λ` | endpoint+NAIT-orthogonalized trajectory residual, **out-of-sample-defined** (Eq.5-1d) |
| `β̂_T ∈ R^{2|A|}` | dual-ridge FWL orthogonalized trajectory coefficient (Eq.5-2) |
| `ψ_ℓ(x)`, `ℓ∈A` | per-layer **routing policy score** (Eq.5-3) — *predictive, not causal* |
| **`Y_obs(x)`** | **observational per-example LOCI utility** = R0 regression target (v4 Eq.16–17). The thing `ψ` is fit to predict. **Never** a training-outcome. |
| **`U_train(π)`** | **held-out training-outcome functional**: train with mask-policy `π` on the selected set, evaluate on the frozen held-out split (retention-adjusted target utility). The thing R1/R2 score. |
| `m_A(x) ∈ {0,1}^A` | the per-example **anchor mask** (which anchor layers are ON) |
| `r_ℓ(x) ∈ Z_{≥0}` | the per-example LoRA rank assigned to anchor layer `ℓ∈A` (output of `capacity_match`, Eq.6-0b) |
| `π` | a **routing policy**: a map from each selected `x` to `(m_A(x), \{r_ℓ(x)\}_{ℓ∈A})` with `Σ_{ℓ∈A} r_ℓ(x)=R_tot` on `A` and **fixed `r_0` on every `ℓ∈L∖A`** (Eq.6-0/6-0a/6-0b) |
| `π_ψ` | the policy: raw `1[ψ_ℓ(x)≥τ_sel]` → `make_feasible_mask(·,ψ)` (Eq.6-0e) → ranks by `capacity_match` (Eq.6-0b) |
| **`V(π) = E_{seed}[U_train(π) | P_train,P_val,P_dep]`** | the **pool-conditional policy value** (seed-expectation at the LOCKED pools; R3-B4) |
| `g_c = V(π_ψ) − V(π_c)` | the per-control policy-value gap (the confirmatory estimand, per arm) |
| `J_{c,ℓ}` | frozen layer-ablation profile, `ℓ∈A` (renamed from `I_{c,ℓ}`; Eq.6-3) |
| `P_train, P_val, P_dep` | three disjoint pools (§3.1), **LOCKED** for every confirmatory claim |
| `R_tot` | fixed total LoRA rank reallocated **over `A`** by every policy/arm |
| `r_0` | the fixed baseline rank carried by every `ℓ∈L∖A`, identical across arms |
| `r_max` | the per-anchor-layer rank cap used by `capacity_match` spillover (Eq.6-0d) |
| `k_min = ⌈R_tot/r_max⌉` | minimum ON-cardinality for a mask to hold `R_tot` under the cap; enforced by `make_feasible_mask` (Eq.6-0e) so `capacity_match` is total on `DOM` (RE-FIX-2) |
| `DOM = {m_A : \|supp(m_A)\|·r_max ≥ R_tot}` | the feasible-mask domain of `capacity_match` (Eq.6-0b0); the map reads **only** `(m_A, R_tot, r_max)`, never `ψ` (RE-FIX-2) |
| `make_feasible_mask(raw, score, R_tot, r_max)` | the **only** place an arm-specific score is read; projects a raw mask onto `DOM` (Eq.6-0e). `score=ψ` for `π_ψ`; a **fixed deterministic key** for every control (RE-FIX-3) |
| `seed_rand, seed_shuf` | **pre-registered fixed** control-randomness seeds; combined with the stable hash `H(x)` make `π_rand`/`π_shuf` deterministic maps of `x` (NOT part of `V(π)`; RE-FIX-3) |
| `A_glob, W_ada` | fixed `π_global` anchor block / fixed `π_ada` warm-up budget — locked control specifics (RE-FIX-3) |
| `ℓ_Brier(q,y), q, c*` | NeuroTrace-IT's strictly-proper **Brier** calibrator score `ℓ_Brier(q,y)=(q−y)²`; calibrated report `q=σ(scale·z+bias)`; decision threshold `c*` for `f̂(x)` (Eq.8-1; §3.8a, RE-FIX-1) |
| `δ_R1, δ_target, …` | locked confirmatory **margins** (R2-B3); every confirmatory null is "gap `≤` margin" |
| `α` | the **marginal** per-contrast level at which each IUT component is tested (R3-B2) |

The split of the old `Y(x,m)` into `Y_obs` (R0) and `U_train` (R1/R2) resolves **B4**.
The restriction of the policy domain to `A` resolves **R2-B1**; the closed-form
mask→rank map (Eq.6-0b) resolves **R2-B1's intra-`A` gap, i.e. R3-B1**. The frozen
coefficient map `B_λ` resolves **R2-B2**. The **pool-conditional** definition of
`V(π)` resolves **R3-B4**.

### 3.1 Three-pool leakage firewall (MF-4) — structural

Partition the candidate corpus by a persisted hash into three example-disjoint
pools, stratified by capability family. **These pools are LOCKED: every confirmatory
estimand and claim in this document is conditional on this single fixed partition
(R3-B4).**

- **`P_train` (learn `ψ`).** Fit the frozen maps `B_λ, b_λ^Y`, `λ_ridge`, the
  cross-fit `β̂_T`, the policy `ψ_ℓ`, the NAIT directions `V_NAIT` (over `L`), and the
  LOCI clustering centroids. **`Y_obs` is computed here** for the R0 regression.
- **`P_val` (estimate `V(π)`).** Used **only** to estimate the **pool-conditional**
  routing policy values `V(π_arm)` for the six arms (§3.5) and to validate `J_{c,ℓ}`
  (§3.6). No selector hyperparameter is tuned on `P_val` outcomes beyond the
  pre-registered `τ_sel` rule. `ψ(x)` on `P_val` is computed by applying the **frozen**
  `B_λ` to each row's `Z(x)` (Eq.5-1d) — no refit.
- **`P_dep` (deploy).** Final selection/routing pool. On `P_dep` the selector and
  router read **activations only** through the frozen `P_train`-fit operators
  (`B_λ, b_λ^Y, β̂_T, ψ via Eq.5-1d, V_NAIT`, centroids); **neither `Y_obs` nor
  `U_train` is computed on `P_dep` before the decision.** The headline R2 comparison
  (§3.5) trains on the `P_dep`-selected set and evaluates on the frozen held-out eval
  split.

Leakage rule (locked): no quantity estimated with `P_dep` outcomes may enter the
`P_dep` selection/routing decision. Pool hashes persist in `schemas_v2`
(`pool_hashes`, additive optional field). A *generalization beyond this fixed
partition* is a separate, **non-confirmatory** probe (§3.5a).

### 3.2 Mechanism certificate — dual-ridge FWL, residualized against `φ_end` AND NAIT (R0; PRESERVED + MF-1 + B7 + R2-B2)

This is the v4 co-primary estimand, **unchanged in math**, on the observational
target `Y_obs`, with the control block **extended** to include NAIT Eq.5 features
(MF-1), the permutation nuisance handling **made precise** (B7), and the residual
operator **replaced by an out-of-sample coefficient map** (R2-B2).

Define the per-example **control feature row** for **any** `x` (in or out of sample):

```
Z(x) = [ φ_end(x) , s_NAIT(x) , V_proj(x) , C(x) , 1 ] ∈ R^{p}.                 (Eq. 5-0)
```

Here `φ_end` is the full endpoint signature, `s_NAIT(x)` the faithful NAIT layer-sum
over **`L`** (§3.3), `V_proj(x) ∈ R^{|L|}` the per-layer NAIT projections
`A^(ℓ)(x)·v_ℓ` (sum = `s_NAIT`), `C(x)` covariates (length, difficulty, family
one-hots). Stacking `Z(x)` over the `P_train` rows gives the design matrix
`Z ∈ R^{N×p}`; stack `T ∈ R^{N×2|A|}` and `Y_obs ∈ R^{N}` likewise.

**Frozen ridge nuisance coefficient maps (R2-B2).** Penalize only the wide
`[φ_end, V_proj]` blocks via `Ω = blockdiag(I on [φ_end,V_proj], 0 on [s_NAIT,C,1])`.
Fit **once** on the training fold and **freeze**:

```
B_λ   = ( ZᵀZ + λ_ridge·Ω )⁺ Zᵀ T      ∈ R^{p×2|A|},                           (Eq. 5-1)
b_λ^Y = ( ZᵀZ + λ_ridge·Ω )⁺ Zᵀ Y_obs  ∈ R^{p}.                                 (Eq. 5-1e)
```

The **deployed, out-of-sample residuals** for **any** example `x` are closed-form
functions of `x` through its own feature row `Z(x)`:

```
T̃(x) = T(x) − Z(x)·B_λ ,        Ỹ(x) = Y_obs(x) − Z(x)·b_λ^Y .                 (Eq. 5-1d)
```

For the stacked training rows this reproduces the old in-sample residual-maker exactly
(`T̃ = T − Z B_λ = (I − Z(ZᵀZ+λΩ)⁺Zᵀ) T = M_λ T`), so **no R0 math changes**; the only
change is that `T̃(x)` is defined for rows `Z` never saw. The old symbol `M_λ` is
retained only as shorthand for the **in-sample** stacking; the **operative deployed
object is `B_λ`** (Eq.5-1/5-1d).

The orthogonalized trajectory coefficient (dual / `n`-space form for 7B width,
exactly v4 `dual_ridge_partial_out`), on the observational target:

```
β̂_T = ( T̃ᵀ T̃ )⁺ T̃ᵀ Ỹ          (T̃, Ỹ stacked over training rows).            (Eq. 5-2)
```

Registered residual quantity (v4 Eq.10a): partial-R²
`partial_R²_T = (RSS_red − RSS_full)/RSS_red` with `RSS_red` from the **extended**
reduced model `Z`; deployment increment `ΔR²_overall = (RSS_red − RSS_full)/TSS`.

**Inference — nuisance handling made precise (B7), now via frozen maps (R2-B2).**
All nuisance (`B_λ, b_λ^Y`, the CV-chosen `λ_ridge`, the NAIT directions `V_NAIT`) is
fit by **10-fold cross-fitting on `P_train` and FROZEN**: for fold `k`, fit
`B_λ^{(−k)}, b_λ^{Y,(−k)}` on the other 9 folds, form the out-of-fold residuals
`T̃^{(k)}(x) = T(x) − Z(x)B_λ^{(−k)}` and `Ỹ^{(k)}(x) = Y_obs(x) − Z(x)b_λ^{Y,(−k)}`
for the held-out rows `x` of fold `k`, and **never refit ridge inside a permutation**.
The permutation test operates on the **already-orthogonalized residuals**:

```
within each (family × fold) stratum s, permute the rows of T̃ relative to Ỹ;
recompute partial_R²_T on the permuted pairing; repeat P=5000 times.           (Eq. 5-2p)
```

> **Proposition P0 (validity, precise — softens "exact").** Condition on the
> frozen cross-fit nuisance maps `{B_λ^{(−k)}, b_λ^{Y,(−k)}}`. Within each stratum
> `s`, if the orthogonalized residual pairs `(T̃_i, Ỹ_i)` are **exchangeable under the
> null** "`T` adds nothing given `Z`" (i.e. `Ỹ ⟂ T̃ | stratum` after partialling out
> `Z`), then the stratified permutation distribution of `partial_R²_T` is the exact
> conditional reference distribution **given the frozen nuisance and the stratum
> labels**, so the permutation p-value satisfies `P(p ≤ α | H0) ≤ α` up to the
> Monte-Carlo error of `P` draws. Because the nuisance is frozen, the only
> approximations are (i) cross-fit estimation of `B_λ` (an `O(n^{-1/2})` first-stage
> error, controlled by Neyman-orthogonality of the FWL moment) and (ii) the
> within-stratum exchangeability assumption. We therefore **claim only**: an
> *asymptotically valid, finite-sample approximately-exact* conditional permutation
> test under within-stratum residual exchangeability — **not** a literally exact test.
> The exchangeability assumption is itself checked by a pre-registered placebo
> permutation on a synthetic null DGP (`test_layer_routing.py` item (g)). ∎

Cluster BCa bootstrap CI (`B=2000`) on the LOCI clusters. No nested-`F` p-value is
used inferentially.

**Gate R0 (mechanism certificate).** Reject "`T` adds nothing given
`[φ_end, NAIT(L)]`" iff permutation `p < α_R0` (closed-test allocation §4) AND BCa-95%
CI of `partial_R²_T` excludes `floor_partial`, stable across `r ∈ {8,16,32}` PCA
poles. **Fail R0 ⇒ trajectory carries no residual beyond NAIT; the method reduces to
NAIT — report a clean null** (`failure_action: stop_main_novelty_claim`).

Why R0 is *harder* than v4: v4 residualized against `φ_end` only; v5 adds the
faithful NAIT score and per-layer projections **over `L`** to the control, so a
"residual" NAIT's own layer-sum already explains can no longer pass (MF-1, B6).

### 3.3 Faithful NAIT reproduction over the full `L` (MF-1, B6) — equation level

Implemented in `baselines/nait_layerwise.py`. **Primary reproduction sums over the
full released decoder layer set `L`** (base Eq.5), resolving B6:

- **Per-layer difference (Alg.1, base Eq.2).**
  `ΔA^(ℓ)(P_i) = A^(ℓ)(t_K) − A^(ℓ)(t_1)` for **every** `ℓ ∈ L`;
  `v_ℓ = PCA₁({ΔA^(ℓ)(P_i)}_i)` (base Eq.3); sign-align
  `v_ℓ ← −v_ℓ if μ_diff·v_ℓ < 0`, `μ_diff = |P|⁻¹ Σ_i ΔA^(ℓ)(P_i)` (base Eq.4).
- **Token-mean variant (prose §3.2.1).** Replace `ΔA^(ℓ)` by the mean-over-`K`-tokens
  activation summary before PCA; else identical. Pre-registered as a **named
  alternate** because the base paper is self-inconsistent; both are run, the
  **stronger** is the comparator.
- **Scoring (base Eq.5, over `L`).** `s_NAIT(y) = Σ_{ℓ=1}^{L} (A^(ℓ)(y)·v_ℓ)`;
  per-layer projections `proj_ℓ(y)=A^(ℓ)(y)·v_ℓ` persisted (they feed `V_proj` in R0).
- **Selection (base Eq.6).** `top-k(s_NAIT)` at budget `B`.

**Secondary 8-anchor variant (gated, B6).** A restricted score
`s_NAIT^{A}(y) = Σ_{ℓ∈A}(A^(ℓ)·v_ℓ)` over the 8 trajectory anchors is computed
**only as a secondary diagnostic**, reported **only if** the full-`L` reproduction
first matches base Eq.5 on the synthetic-activation unit check
(`test_layer_routing.py` item (d)) and on a released-checkpoint sanity slot
(`DATA_NEEDED`). "Beats NAIT" wording is **always** gated on the **full-`L`** stronger
variant; the 8-anchor number can never be the comparator the headline uses.

**Routing domain vs comparator domain (R2-B1 / B6 are orthogonal).** NAIT is *scored*
over the full `L` here. The LATTICE-R *routing policy* (§3.5) acts only on the anchor
set `A`; the two domains serve different roles (comparator score vs capacity mask) and
there is no conflict.

`endpoint_neuron_selection` (existing `baselines/nait.py`, the `concat[start,end]`
reconstruction) is **retained** as the FULL endpoint control `φ_end`;
`nait_layerwise` over `L` is the **decisive comparator**.

### 3.4 Trajectory operator, policy score `ψ`, and the COUPLING IDENTITY (PRESERVED; out-of-sample via Eq.5-1d)

`D_ℓ`, `κ_ℓ`, `T(x)` are **unchanged** (v4 Eq.3–6; `trajectory.py`), defined over the
anchor set `A`. `κ` is provably not a function of the first-moment pool: reorder steps
⇒ a first-moment summary is invariant but `κ_ℓ` (ordered second differences) changes
⇒ `κ_ℓ` is not a function of that summary. ∎

**Per-layer routing policy score (MF-2), out-of-sample-defined (R2-B2):**

```
ψ_ℓ(x) = β̂_{D,ℓ}·D̃_ℓ(x) + β̂_{κ,ℓ}·κ̃_ℓ(x),     ℓ ∈ A,                       (Eq. 5-3)
```

where `D̃_ℓ(x), κ̃_ℓ(x)` are the per-layer coordinates of the **out-of-sample**
residual `T̃(x) = T(x) − Z(x)·B_λ` (Eq.5-1d), so `ψ(x)` is a **closed-form function
of `x` alone** and is computable on `P_val`/`P_dep`. `β̂_{D,ℓ}, β̂_{κ,ℓ}` are the
matching coordinates of `β̂_T` (Eq.5-2). No new model is fit at deployment.

> **Coupling identity (PRESERVED — verified-sound; the mathematical spine).**
> Because `T(x) = ({D_ℓ},{κ_ℓ})_{ℓ∈A}` stacks the per-layer coordinates and `β̂_T` is
> the single orthogonalized coefficient on `T̃`,
> ```
> Σ_{ℓ∈A} ψ_ℓ(x)  =  Σ_{ℓ∈A} ( β̂_{D,ℓ}·D̃_ℓ(x) + β̂_{κ,ℓ}·κ̃_ℓ(x) )  =  β̂_T · T̃(x).  (Eq. 5-4)
> ```
> So the **selection score** `Σ_{ℓ∈A} ψ_ℓ(x)` is *exactly* the endpoint+NAIT-residualized
> trajectory prediction `β̂_T·T̃(x)` (one scalar endpoint), and the **routing
> support** is `supp(ψ(x)) ⊆ A`. Selection = the **sum**; routing = the **support**;
> of **one** attribution `ψ` built from **one** orthogonalized coefficient `β̂_T`.
> The identity holds for **out-of-sample** `x` because both sides use the same frozen
> `B_λ, β̂_T`. Unit-tested (`test_layer_routing.py` item (a), in- and out-of-sample). ∎

**`ψ_ℓ` is explicitly a predictive policy score, not a causal layer-write effect.** It
is fit to predict the observational `Y_obs`; it *proposes* the mask `m_A(x)`. The
**certification that writing to `supp(ψ)` is the right place** is the **policy-value**
gate R1 (§3.5), not the decomposition. The headline routing object is `supp(ψ) ⊆ A`,
**resolving B3**: the experiment in §3.5 measures exactly `V(π_ψ)`.

### 3.5 The routing POLICY-VALUE estimand, the FULLY-SPECIFIED policy, and the kill-gates (R1, R2, R3; MF-2/3/6, B1–B4, R2-B1, R2-B3, R3-B1, R3-B2)

**The policy domain is exactly the anchor set `A` (R2-B1), and the policy is now
fully specified inside `A` (R3-B1).** A routing **policy** is

```
π : x ↦ ( m_A(x) ∈ {0,1}^A , {r_ℓ(x)}_{ℓ∈A} ) ,
     with  Σ_{ℓ∈A} r_ℓ(x) = R_tot   (rank reallocated on A),                       (Eq. 6-0)
and  r_ℓ(x) = r_0  FIXED for every ℓ ∈ L∖A, identical across ALL arms.               (Eq. 6-0a)
```

Only the anchor masks/ranks on `A` are degrees of freedom; the `|L∖A|` non-anchor
layers form a **policy-independent common substrate** at rank `r_0`, which **cancels
in every contrast** `V(π_ψ) − V(π_c)`.

**`capacity_match` — the closed-form mask→rank map (R3-B1; RE-FIX-2).** `capacity_match`
is a **pure function of exactly its three declared arguments `(m_A, R_tot, r_max)` and
NOTHING ELSE** — in particular it **never reads `ψ`** (the previous re-gate flaw: the
old CAP_SPILL grew support "by descending `ψ_ℓ`", but `ψ` is not an argument, reopening
implementation-dependence). It is defined on the **domain of FEASIBLE masks**

```
DOM = { m_A ∈ {0,1}^A : |support(m_A)| · r_max ≥ R_tot },                            (Eq. 6-0b0)
```

i.e. masks whose ON-cardinality `k = |support(m_A)|` is large enough to hold `R_tot`
under the per-layer cap. **Feasibility is GUARANTEED UPSTREAM by every arm's mask
constructor** (Eq.6-0e below), which enforces a minimum cardinality
`k_min = ⌈R_tot / r_max⌉` **before** calling `capacity_match`; the `ψ`-dependent choice
of *which* extra layers to turn on lives in the arm's constructor (where `ψ` is
legitimately available for `π_ψ`, and a fixed rule for every control), **not** inside the
shared map. Consequently `capacity_match` has **no infeasible branch at all** and is a
total function on `DOM`. Given a feasible mask with support `S` of size `k`:

```
capacity_match(m_A, R_tot, r_max):   # m_A ∈ DOM, so k·r_max ≥ R_tot               (Eq. 6-0b)
  S ← support(m_A);  k ← |S|.        # k ≥ 1 and k·r_max ≥ R_tot, both guaranteed by DOM
  # 1. Even base share (renormalizes automatically to ANY feasible cardinality k):
  q ← floor(R_tot / k);   rem ← R_tot − k·q.     # 0 ≤ rem < k ;  note q ≤ r_max (see Prop. below)
  r_ℓ ← q  for ℓ ∈ S ;   r_ℓ ← 0  for ℓ ∉ S.
  # 2. Largest-remainder top-up of the `rem` leftover units, one rank each:
  #    all ON layers share the SAME fractional target R_tot/k − q, so the +1 awards
  #    are resolved by a DETERMINISTIC key: ascending layer index ℓ (lexicographic).
  S_sorted ← sort(S, key = ℓ ascending)
  for j in 0 .. rem−1:  r_{S_sorted[j]} ← r_{S_sorted[j]} + 1
  # 3. Per-layer cap r_max with DETERMINISTIC spillover among the SAME ON layers only
  #    (no support growth — feasibility makes growth unnecessary; uses (r,S,r_max) only):
  r ← CAP_SPILL(r, S, r_max)
  assert  Σ_{ℓ∈A} r_ℓ = R_tot  and  r_ℓ = 0 for ℓ ∉ S  and  r_ℓ ≤ r_max ∀ℓ.
  return r
```

```
CAP_SPILL(r, S, r_max):   # cap enforcement, pure function of (r,S,r_max) — NO ψ    (Eq. 6-0d)
  # After steps 1–2 at most the `rem` ascending-ℓ layers carry q+1; all others carry q.
  # If q+1 > r_max (only possible at the boundary k = ⌈R_tot/r_max⌉, q = r_max), clip the
  # overflowing layers to r_max and re-award their excess +1 units, by the SAME
  # largest-remainder / ascending-ℓ rule, to ON layers still below r_max. Because the
  # mask is FEASIBLE (k·r_max ≥ R_tot), the total headroom Σ_{ℓ∈S}(r_max − r_ℓ_afterclip)
  # ≥ R_tot − k·r_max·[q=r_max] ≥ 0 always suffices, so the loop terminates with all
  # excess placed and NO support growth is ever required.
  loop:
    over ← {ℓ∈S : r_ℓ > r_max};  if over = ∅: return r
    excess ← Σ_{ℓ∈over}(r_ℓ − r_max);  r_ℓ ← r_max for ℓ∈over
    free ← {ℓ∈S : r_ℓ < r_max}        # nonempty whenever excess>0, by feasibility (Prop. below)
    distribute `excess` units of +1 over `free` by largest remainder,
      tie-break ascending ℓ (same rule as step 2).
```

**Zero-support / empty-mask is handled at the CONSTRUCTOR, not here (Eq. 6-0c).** If an
arm's raw mask is empty (`k = 0`, e.g. no anchor scored ON), the constructor (Eq.6-0e)
replaces it with the **fixed uniform-over-`A` mask `1_A`** before any rank call; `1_A` is
feasible iff `|A|·r_max ≥ R_tot`, which §5.6 pins to hold. Thus `capacity_match` itself
is only ever called on a feasible nonempty mask and needs no `k=0` branch.

> **Feasibility / no-overflow lemma (Eq. 6-0b is total, cap-respecting, and conservative
> on `DOM`; `CAP_SPILL` is in fact a no-op there).** Let `m_A ∈ DOM`, so `k·r_max ≥
> R_tot`, hence `q = ⌊R_tot/k⌋ ≤ R_tot/k ≤ r_max`. After steps 1–2 every ON layer carries
> either `q` (the `k − rem` lowest-index ON layers) or `q+1` (the `rem` lowest-index ON
> layers); the maximum value is therefore `q + 1[rem>0]`.
> *Case `q < r_max`:* then `q+1 ≤ r_max`, so the maximum value `≤ r_max` — **no layer
> exceeds the cap**, `CAP_SPILL` finds `over = ∅` and returns unchanged, and `Σ r_ℓ =
> (k−rem)q + rem(q+1) = kq + rem = R_tot`.
> *Case `q = r_max`:* then `r_max = ⌊R_tot/k⌋ ≤ R_tot/k ≤ r_max` forces `R_tot/k = r_max`
> exactly, so `R_tot = k·r_max` and `rem = R_tot − kq = 0`; every ON layer carries exactly
> `q = r_max` and **again no layer exceeds the cap**.
> Thus for **every** feasible mask the post-steps-1–2 vector already satisfies `r_ℓ ≤
> r_max`, `CAP_SPILL` triggers no redistribution and never needs to grow `S`, the sum is
> exactly `R_tot`, and `capacity_match` is a deterministic function of `(m_A, R_tot,
> r_max)` alone. (The `CAP_SPILL` loop is retained only as a defensive invariant-checker;
> the headroom bound `Σ_{ℓ∈S}(r_max − r_ℓ) = k·r_max − R_tot ≥ 0` shows that even if a
> caller passed a non-canonical vector, any excess could be re-placed on free ON layers in
> finitely many passes without support growth.) ∎

**Properties (R3-B1 + RE-FIX-2, all deterministic and implementation-independent):**
*(i)* For any **feasible** mask cardinality `k ∈ {k_min,…,|A|}`, the output conserves
`Σ_{ℓ∈A} r_ℓ = R_tot` (steps 1–2 are an exact integer partition; CAP_SPILL preserves the
sum by the lemma). *(ii)* Ties are broken by **ascending layer index `ℓ`** — a fixed
total order — so no implementation freedom remains. *(iii)* The empty mask is resolved
**at the constructor** by the fixed uniform-over-`A` mask (Eq.6-0c/0e), never inside the
map. *(iv)* The per-layer cap `r_max` is enforced by `CAP_SPILL` using **only**
`(r, S, r_max)` — **no `ψ`, no support growth, no infeasible branch**. Therefore
`capacity_match` is a **single function of `(m_A, R_tot, r_max)`** and `U_train(π)` is
**one** implementation-independent functional of `π`. (Unit-tested:
`test_layer_routing.py` item (b).)

**Mask constructor with guaranteed feasibility (Eq. 6-0e) — where `ψ` (and only the
arm-specific rule) legitimately lives.** Every arm builds its raw anchor mask, then
projects it onto `DOM` by a **fixed, arm-specific, deterministic** rule that needs the
arm's own scoring object but **never** leaks into `capacity_match`:

```
make_feasible_mask(raw_mask, score, R_tot, r_max):                                  (Eq. 6-0e)
  k_min ← ceil(R_tot / r_max)                       # minimum ON-cardinality for feasibility
  S ← support(raw_mask)
  if S = ∅:  S ← A                                  # empty ⇒ uniform-over-A (Eq. 6-0c)
  if |S| ≥ k_min:  return indicator(S)              # already feasible
  # else add the (k_min − |S|) highest-`score` OFF anchors, tie-break ascending ℓ:
  cand ← sort( A∖S, key = (−score(ℓ), ℓ) )          # `score` is the ARM's own object:
  S ← S ∪ { first (k_min − |S|) of cand }           #   ψ_ℓ for π_ψ; a FIXED key for controls
  return indicator(S)
```

`make_feasible_mask` is the **only** place an arm-specific score (`ψ` for `π_ψ`; a fixed
key for each control, §3.5) is consulted, and it produces a mask in `DOM`;
`capacity_match` then maps that feasible mask to ranks with no further score access.
This cleanly separates the `ψ`-dependent *mask choice* (legitimately per-arm) from the
*shared, score-free* rank apportionment (`capacity_match`), closing the RE-FIX-2 flaw.

`π_ψ` instantiates `raw_m_ℓ(x)=1[ψ_ℓ(x) ≥ τ_sel]`, then
`m_A(x)=make_feasible_mask(raw_m, ψ(x), R_tot, r_max)` (Eq.6-0e, score = `ψ`), then
`{r_ℓ(x)}=capacity_match(m_A(x), R_tot, r_max)` (Eq.6-0b); it allocates `R_tot` over `A`
to (a feasibility-completed) `supp(ψ)` and leaves `L∖A` at `r_0`. This makes `π_ψ`,
`U_train(π_ψ)`, and `V(π_ψ)` **fully defined as implemented**, with `capacity_match` a
pure score-free function.

**The estimand is a POOL-CONDITIONAL policy value over whole training runs (B1, B2,
B4, R3-B4).** The held-out **training-outcome functional** is

```
U_train(π) = retention-adjusted target utility of the model obtained by
             training the selected set with per-example anchor masks+ranks {π(x)}
             (and the fixed r_0 substrate on L∖A), evaluated on the frozen
             held-out split.                                                       (Eq. 6-1)
```

The **policy value** is the **seed-expectation at the LOCKED pools** (R3-B4), and the
confirmatory estimand is the per-control gap:

```
V(π) = E_{seed}[ U_train(π) | P_train, P_val, P_dep ],
estimand g_c = V(π_ψ) − V(π_c)   per control arm c.                                (Eq. 6-2)
```

The expectation is over **training seeds only**, with the three pools held at their
single locked partition. The **unit of treatment is the whole training run**, so:

- **No per-example potential outcome (B1).** We never write `Y(x, m_ℓ)`. A policy
  fixes the *entire* anchor mask+rank field at once via `capacity_match`, so
  `U_train(π)` is a single well-defined functional of `π`.
- **No cross-example SUTVA violation (B2).** A run treats the whole set under one
  policy; the treated unit is the run, not the example. Runs are independent across
  the pre-registered seeds.
- **The experiment estimates exactly the headline object (B3).** R1 estimates
  `g_c = V(π_ψ) − V(π_c)` per control; the headline routing object is `supp(ψ) = π_ψ`
  over `A`. `supp_τ(ψ)` (the old, unidentified object) is **deleted**.
- **The policy is fully specified (R2-B1, R3-B1).** `π_ψ` over `A` with `capacity_match`
  ranks and `L∖A` pinned at `r_0`; nothing about the run is left undefined.
- **Estimand matches estimator (R3-B4).** `V(π)` is the **pool-conditional**
  seed-expectation; the seed-mean `V̂(π)=S^{-1}Σ_s U_train,s(π)` is unbiased for it and
  the bootstrap-`t` (§4.2) resamples exactly the seed randomness it averages. The
  confirmatory claim is **explicitly scoped to the locked pools**.
- **Outcome symbols never conflated (B4).** `π_ψ` is *built from* `ψ` (fit to `Y_obs`
  on `P_train`); it is *scored by* `U_train` on `P_val`. `Y_obs` and `U_train` never
  appear in the same equation.

**Gate R1 — Routing policy-value coherence (headline mechanism; MF-3, R2-B3, R3-B2).**
Estimated on `P_val` with **six real masked-LoRA training arms** (no proxy). Same
selected examples, same budget, same seeds (`configs/seeds/paper_20.txt`); each arm
trains and evaluates a policy on the anchor set `A`, with the identical `r_0` substrate
on `L∖A` and the identical `capacity_match` map (Eq.6-0b):

**Every control is a SINGLE deterministic, implementation-independent map (RE-FIX-3).**
Each arm is `x ↦ make_feasible_mask(raw_mask_arm(x), score_arm, R_tot, r_max)` (Eq.6-0e)
followed by `capacity_match(·, R_tot, r_max)` (Eq.6-0b). All randomness used by a control
is driven by a **fixed, pre-registered seed** combined with the example's **stable
content hash** `H(x)` (persisted in `schemas_v2.pool_hashes`), so the control is a
**deterministic function of `x`** — the randomization is **part of the policy
definition, NOT part of the `V(π)` expectation** (which is over *training* seeds only,
Eq.6-2). Re-running any implementation reproduces the identical mask for each `x`. The
fixed seeds `(seed_rand, seed_shuf, seed_ada)`, the global block `A_glob`, the AdaLoRA
warm-up budget `W_ada`, and the stratum map are locked in `lattice_v5.yaml` (§5.6).
**`seed_ada` is the SEPARATE, pre-registered seed that freezes the AdaLoRA importance
warm-up OUTSIDE the `V(π)` training-seed expectation (RE-FIX-4), so `π_ada` too is a
deterministic map of `x`.**

1. `π_ψ` — `raw_m_ℓ(x)=1[ψ_ℓ(x) ≥ τ_sel]`; `make_feasible_mask(·, score=ψ(x))`; ranks by
   `capacity_match` (the policy under test).
2. `π_unif` — raw mask `1_A` (all anchors ON), `make_feasible_mask` is a no-op (`1_A∈DOM`
   when `|A|·r_max≥R_tot`, §5.6), ranks by `capacity_match` (capacity-matched).
3. `π_shuf` — apply a **fixed within-stratum permutation** `σ_g` of the anchor-layer
   labels `{1..|A|}` to `ψ(x)` for every `x` in stratum `g`: `ψ^shuf_ℓ(x)=ψ_{σ_g(ℓ)}(x)`,
   where `σ_g = PRNG(seed_shuf ⊕ g).permutation(|A|)` is a **single deterministic
   permutation per stratum** (strata = the family×fold strata of §3.1, locked). Then
   `raw_m_ℓ(x)=1[ψ^shuf_ℓ(x) ≥ τ_sel]`, `make_feasible_mask(·, score=ψ^shuf(x))`,
   `capacity_match`. (Breaks the layer↔example coupling while holding the `ψ` *marginal*
   fixed; deterministic given `seed_shuf`.)
4. `π_rand` — raw anchor mask = a **deterministic pseudo-random subset** of `A` of
   cardinality `k(x)=|supp(raw_m_ψ(x))|` (matched to `π_ψ`'s pre-feasibility cardinality),
   drawn by `PRNG(H(x) ⊕ seed_rand).sample(A, k(x))`. `make_feasible_mask(·,
   score = the same PRNG's ranking key over `A∖S`)` completes to `k_min` if needed; then
   `capacity_match`. (One fixed seed ⇒ one mask per `x`; the RNG, its seed, and the
   tie-key are all pinned, so no implementation freedom.)
5. `π_global` — one **fixed global-selective anchor block** `A_glob ⊆ A`, identical for
   **every** `x` (surgical-style), pre-registered with `|A_glob| ≥ k_min` so it is
   feasible by construction (`make_feasible_mask` is a no-op); `capacity_match`. `A_glob`
   is the locked top-`k_glob` anchors by the **pooled training-set mean `|ψ_ℓ|` on
   `P_train`** (a fixed statistic computed once, ties ascending-`ℓ`), persisted in the
   config — not re-chosen per implementation.
6. `π_ada` — AdaLoRA's adaptive importance over `A` from a **fixed warm-up budget**
   `W_ada` (locked), giving a per-anchor importance score `imp_ℓ(x)`; the **policy** mask
   is `raw_m(x)=` top-`k(x)` anchors by `imp_ℓ(x)` (tie-break ascending `ℓ`),
   `make_feasible_mask(·, score=imp(x))`, then `capacity_match` to integer ranks summing
   to `R_tot`. (AdaLoRA's continuous allocation is **projected to the same integer
   capacity grid** as every other arm; `W_ada` and the tie rule are pinned so `imp_ℓ` is
   a deterministic function of the warm-up run.) **The AdaLoRA importance warm-up is
   frozen by a SEPARATE pre-registered seed `seed_ada` (locked in `lattice_v5.yaml`),
   NOT by the per-run training seed (RE-FIX-4).** The warm-up is run **once**, before
   the confirmatory arms, under `seed_ada`; its importance scores `imp_ℓ(x)` are
   persisted (`schemas_v2.control_provenance`) and read identically by every confirmatory
   seed. Thus `π_ada(x)` is a **deterministic function of `x`** (its randomness is part
   of the policy definition, fixed by `seed_ada`), exactly like the other four controls,
   and the **only** seed inside the `V(π)` expectation (Eq.6-2) remains the per-run
   **training** seed — the warm-up seed is explicitly OUTSIDE that expectation. This
   removes the round-3 internal inconsistency (importance previously depended on the
   training seed, which would have made `π_ada` non-deterministic in `x` and put a
   second seed inside `V(π)`).

All arms reallocate the **same** `R_tot` over `A`, hold the **same** `r_0` on `L∖A`,
use the **same** `make_feasible_mask` (Eq.6-0e) and the **same** `capacity_match`
(Eq.6-0b), and pass the §3.7 compute-match ledger. Each control is thus **one
implementation-independent object**, satisfying the policy-value estimand's requirement
(RE-FIX-3). For each arm
estimate `V̂(π_arm)=S^{-1}Σ_{s=1}^{S} U_train,s(π_arm)` (seed-mean, `S≥20`). Define the
per-control paired gaps

```
g_c = V(π_ψ) − V(π_c),     ĝ_c = V̂(π_ψ) − V̂(π_c),    c ∈ {unif,shuf,rand,global,ada}.  (Eq. 6-2g)
```

**The composite claim "`π_ψ` beats EVERY control by margin `δ_R1`" is an
INTERSECTION-UNION alternative (R3-B2).** Its null is the **union** of the per-control
nulls and its alternative the **intersection** of the per-control alternatives:

```
H0^{R1} = ⋃_c { g_c ≤ δ_R1 }   (π_ψ fails to beat at least one control by δ_R1);
H1^{R1} = ⋂_c { g_c >  δ_R1 }   (π_ψ beats EVERY control by > δ_R1).               (Eq. 6-G)
```

**Decision rule — INTERSECTION-UNION TEST (Berger 1982; R3-B2).** Reject `H0^{R1}`
(routing coherence holds) **iff EACH per-control margin test rejects at the MARGINAL
level `α`** — i.e. for **every** `c`, the per-contrast asymptotic lower confidence
bound `L_c(α)` for `g_c` (Eq.7-Bt) satisfies `L_c(α) > δ_R1`. **No multiplicity
correction is applied across the five controls** (an IUT is exactly level-`α`,
Prop. P1-IUT), and **no shared min-over-controls quantile is used** (that is the
round-2 error). Equivalently the **simultaneous lower bound** on `min_c g_c` is the
**minimum of the per-contrast marginal-`α` lower bounds** (Eq.7-IUT-CI), and we reject
iff that minimum exceeds `δ_R1`. **Fail R1 ⇒ drop routing; fall back to v4
selection-only** (`failure_action: drop_routing_keep_selection`).

**Gate R2 — Compute-matched method win (headline; MF-6, R2-B3, R3-B2).** At a
**measured compute-matched** budget (§3.7 ledger), LATTICE-R's `U_train` must beat the
**stronger** full-`L` NAIT variant **and** the strongest non-NAIT baseline (§5.4) **by
the locked margins**. The R2-target "beats every comparator in the baseline set" claim
is, like R1, an **intersection-union** alternative and is decided by the **same IUT**:
reject iff **each** comparator's per-contrast margin test clears `δ_target` at the
marginal level `α` — no penalty across comparators (Eq.7-IUT applied to the baseline
set). The remaining R2 sub-claims `k ∈ {relative, retention, hallucination, cost}` are
**single** margin tests (one contrast each), each rejecting iff the per-contrast
asymptotic bound clears its `δ_k` (lower bound above `δ` for "win" margins
`δ_rel`; **upper** bound below the ceiling for the non-inferiority drift/cost margins
`δ_ret, δ_hall, δ_cost`). **Fail R2 ⇒ no method-win claim**
(`failure_action: no_method_win_claim`).

**Gate R3 — Honest two-sided cost (MF-6).** `cost_model.py` reports the
extraction-parity multiplier (kept: `> 2.0× ⇒ high-cost-analysis`) **and** the
training-side ledger. A "training savings" claim is licensed **only** if `cost_model`
records a **measured** FLOP/wall-clock reduction with masked anchor layers **actually
skipped** (skip-flag true); otherwise the savings claim is **forbidden** and only the
extraction cost is reported.

### 3.5a Estimand scope and the (non-confirmatory) generalization probe (R3-B4)

The confirmatory estimand `g_c = V(π_ψ) − V(π_c)` is the seed-expectation **at the
single locked partition** `(P_train, P_val, P_dep)` (Eq.6-2). Every confirmatory
rejection is therefore a statement about **that** partition: *under these fixed pools,
the routing policy's pool-conditional value beats each control's by `δ_R1`.* This is
the estimand the seed-mean and its bootstrap-`t` CI actually cover — there is **no**
across-pool variance component omitted, because there is **no** across-pool expectation
in the estimand (R3-B4).

To speak about the **pool-marginal** value `E_{seed, pool-draws}[U_train(π)]` would
require resampling the *partition*, which would add a between-pool variance component
the seed-mean does not see. We therefore offer this as an **explicitly
non-confirmatory generalization probe**, outside the confirmatory `α`:

- Re-draw `G` independent pool partitions (`G` a pre-registered LOCKED count,
  `partitions_G = 5` in `lattice_v5.yaml`, each with a fresh `pool_split_seed`),
  recompute `ĝ_c^{(g)}` per partition, and
  report a **cluster-robust** interval for the pool-marginal `g_c` that includes the
  between-partition variance. Any inference from this probe carries its **own**
  multiplicity accounting (`G` partitions × 5 controls, Holm within the probe) and is
  reported as **exploratory**, never feeding the confirmatory closed test of §4.

This keeps the estimand and estimator **aligned** for the confirmatory claim (scoped
to the locked pools) while giving an honest, separately-accounted route to a broader
statement.

### 3.6 Frozen layer-ablation profile `J_{c,ℓ}` and the DESCRIPTIVE per-layer notion (MF-5, B1-(ii))

Renamed from "causal layer importance `I_{c,ℓ}`" — activation replacement is
off-manifold, so no causal claim is made; defined for `ℓ∈A`:

```
J_{c,ℓ} = ( Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ) ) / Acc_c(model),  ℓ∈A.       (Eq. 6-3)
```

`J_{c,ℓ}` may enter the router prior **only if it passes a validation gate** on
`P_val`: it must correlate (Spearman ρ ≥ `ρ_J`, lower CI > 0) with the **real anchor
layer-freeze policy-value difference** (Eq.6-4 below). **Fail ⇒ `J` is dropped** and
reported as a descriptive diagnostic; the router then uses `ψ` alone.

**Descriptive per-layer policy-value difference (the only per-layer notion kept;
B1-(ii)).** If a per-layer summary is desired, it is a **run-level
leave-one-anchor-layer policy-value difference with the explicit, pre-registered
`capacity_match` redistribution** (Eq.6-0b) — *not* a per-example causal `τ_ℓ`:

```
Δ_ℓ^{LOL} = V(π_full) − V(π_{−ℓ}),   ℓ ∈ A,
  where π_{−ℓ} masks anchor layer ℓ off for ALL examples and re-runs capacity_match
  on the remaining ON anchors (Eq.6-0b: even base + largest-remainder + cap), so
  Σ_{ℓ∈A} r_ℓ = R_tot is preserved and L∖A stays r_0.                              (Eq. 6-4)
```

`Δ_ℓ^{LOL}` is a **descriptive, run-level** quantity (well-defined because
`capacity_match` is fixed and the unit is the whole run), explicitly **labeled
descriptive, not causal evidence**, and is **never** part of the confirmatory `α`. It
describes which anchor layers the policy leans on; the confirmatory routing claim is
the IUT over `{g_c}` (Eq.6-2g/6-G), and the routing support is `supp(ψ) ⊆ A`.

### 3.7 Compute-match ledger (MF-6) — what "matched budget" means

`cost_model.py::compute_match_ledger` records, per arm: parameter count;
optimizer-state slot count; **realized FLOPs** (measured, not rank-inferred);
wall-clock; batch/accumulation policy; and a **skip-flag** = whether masked anchor
layers' adapters are actually excluded from the backward pass. R2's "matched budget"
requires equality within a locked relative tolerance on params, optimizer slots, and
realized FLOPs; because every arm shares the identical `r_0` substrate on `L∖A`, the
same `R_tot` on `A`, and the same `capacity_match` map, the substrate contributes
equally to every arm's ledger and the match reduces to the anchor allocation. Any arm
that cannot match exactly is labeled **capacity-unmatched diagnostic**, never the
headline. Equal rank alone is explicitly **not** equal compute.

### 3.8 Selection objective (PRESERVED) — submodular (1−1/e) intact

Per-example utility stays a scalar
`u(x) = max(0, Σ_{ℓ∈A} ψ_ℓ(x) − λ_r r̂(x) − λ_f f̂(x))` (v4 §2.8), and by the coupling
identity `Σ_{ℓ∈A} ψ_ℓ(x) = β̂_T·T̃(x)` (Eq.5-4). Routing is applied **after** selection,
on the selected set, so it never enters the facility-location objective
`F(S) = Σ_x u(x) + μ·C(S)`; `F` stays monotone-submodular and the greedy (1−1/e) proof
(`selection.py`) is untouched.

### 3.8a Factuality calibrator — NeuroTrace-IT's strictly-proper Brier calibrator (G6)

The factuality term `λ_f f̂(x)` in `u(x)` reads a calibrated proxy `q_a = σ(z_a)`
(§3.8, v4 §2.6). The calibrator must be scored by a **strictly proper** rule so that its
risk is minimized exactly at the true conditional support probability and cannot be gamed
by over/under-confidence. NeuroTrace-IT scores the calibrator with the **Brier (quadratic)
rule** — the same strictly-proper rule introduced in v4 §2.6 and carried unchanged into v5.
Nothing here touches routing; it discharges G6's scoring precondition.

**Construction (NeuroTrace-IT's Brier calibrator).** For a binary claim label `y∈{0,1}`
and calibrated report `q∈[0,1]`, score the calibrator by the Brier loss

```
ℓ_Brier(q, y) = (q − y)² .                                                          (Eq. 8-1)
```

The report is `q_a = σ(scale·z_a + bias)` (an optional affine recalibration of the claim
logit `z_a`; identity `scale=1, bias=0` by default), and `f̂(x) = mean_a 1[q_a < c*]` is
the fraction of an example's claims whose calibrated support probability falls below the
calibrated threshold `c*` (§2.6).

> **Proposition P-RC1 (strict propriety of the Brier rule).** Fix the true label law
> `y ∼ Bernoulli(p)`, `p∈[0,1]`. The expected score of report `q` is
> ```
> E_{y∼Bern(p)} (q − y)² = p(q − 1)² + (1−p)q² = (q − p)² + p(1−p).                  (Eq. 8-2)
> ```
> The term `p(1−p)` does not depend on `q`, and `(q − p)² ≥ 0` with equality iff `q = p`;
> taking the expectation over contexts preserves strictness. Hence the population Brier
> risk is **uniquely minimized at `q = p`**: truthful reporting is the unique optimum, so
> the rule is **strictly proper**. ∎

**What this discharges.** The calibrator is scored by the strictly-proper Brier rule
(Eq. 8-1); P-RC1 gives **calibration** (the empirical-Brier minimizer drives `q` toward
the conditional support probability). As always, **propriety proves calibration, not
eval-drift prediction**: G6 remains a *precondition* gate (Spearman/Pearson ρ ≥ 0.3,
lower CI > 0, ECE ≤ 0.1); **fail G6 ⇒ `λ_f := 0`**, no factuality claim. None of this
enters the confirmatory routing α; it only licenses the `λ_f f̂` term in `u(x)`.
(Unit-tested in `test_trajectory_selection.py`: the Brier risk is minimized at the true
probability, `factuality_drift` counts unsupported claims, ECE is exact on bracketing
slices, and the G6 gate sets `λ_f := 0` on failure; **formula evaluation, not evidence**.)

### 3.9 Algorithm box (design-only; no server run)

```
Algorithm LATTICE-R v5  (design-only; server.authorized = false)
Inputs : pools P_train / P_val / P_dep (disjoint, hashed, LOCKED), budget B,
         anchor layers A, full decoder set L, baseline rank r_0 on L∖A,
         target clouds {Q_ℓ^T}_{ℓ∈A}, concept set E, R_tot (over A), cap r_max,
         locked (λ_ridge grid, τ_sel rule, margins δ_*, marginal level α), seeds 0..19

# Phase A (DESIGN — this document): specify everything below; touch no src/.

# Phase B-0 : faithful NAIT baselines FIRST, over the FULL L (control + comparator)
  on P_train: v_ℓ ← PCA(ΔA^(ℓ)) for ℓ∈L (+ token-mean variant)
  s_NAIT, proj_ℓ ← base Eq.5 summed over L ; (8-anchor variant: secondary, gated)
  φ_end ← v4 Eq.1   # retained endpoint reconstruction (FULL control)

# Phase B-1 : trajectory signatures on each pool (activations only on P_dep), over A
  for x: D_ℓ(x)←Eq.4 ; κ_ℓ(x)←Eq.5 ; T(x)←Eq.6   (ℓ∈A)

# Phase B-2 : mechanism certificate R0 on P_train (cross-fit, FROZEN coefficient maps)
  Z(x)=[φ_end,s_NAIT(L),V_proj(L),C,1] ; B_λ,b_λ^Y ← Eq.5-1/5-1e (frozen per fold)
  T̃(x)=T(x)−Z(x)B_λ (Eq.5-1d) ; β̂_T←Eq.5-2 on Ỹ
  partial_R²_T, ΔR²_overall ; perm on orthogonalized residuals T̃ vs Ỹ (P=5000, no ridge refit)
  if R0 fails: STOP (reduce to NAIT, clean null)

# Phase B-3 : policy score + POLICY-VALUE routing R1 on P_val (REAL training runs)
  ψ_ℓ(x)←Eq.5-3 via frozen B_λ (out-of-sample) ; coupling check Σ_{ℓ∈A}ψ_ℓ=β̂_T·T̃ (Eq.5-4)
  policy π_ψ: m_A(x)=1[ψ_ℓ≥τ_sel] ; {r_ℓ}=capacity_match(m_A,R_tot,r_max) (Eq.6-0b); L∖A=r_0
  validate J_{c,ℓ} (Eq.6-3) vs anchor layer-freeze Δ^LOL (Eq.6-4); drop J if fail
  train arms {π_ψ,π_unif,π_shuf,π_rand,π_global,π_ada} (compute-matched; same r_0; same capacity_match)
  V̂(π_arm) ← seed-mean U_train (POOL-CONDITIONAL, Eq.6-2) ; ĝ_c ← Eq.6-2g
  per-control margin tests: L_c(α) > δ_R1 via paired bootstrap-t (Eq.7-Bt)
  INTERSECTION-UNION decision: reject H0^{R1} iff ALL five L_c(α) > δ_R1 (Eq.7-IUT)
  if R1 fails: DROP routing, keep v4 selection-only

# Phase B-4 : compute-matched method win R2 on P_dep (activations-only selection)
  select S=greedy F(S) ; route π_ψ on S ; train LATTICE-R vs stronger-NAIT(L) vs baselines
  R2-target: IUT over the baseline set (each comparator's L_c(α) > δ_target, Eq.7-IUT)
  R2-{rel,ret,hall,cost}: single margin tests (each adjusted bound clears its δ_k)
  if R2 fails: no method-win claim

# Phase B-5 : closed-testing decision (§4) + honest cost ledger R3 (§3.7)
  emit TrajectorySignatureV2 records (RouterOutputs populated) ; server_authorized=false
```

---

## 4. Multiplicity — closed-testing graph with IUT leaves and a finite-sample FWER (MF-7, B5, R2-B4, R3-B2, R3-B3)

The "beats every control / every comparator" decisions are **intersection-union tests**
(no penalty, exact level `α`); the per-contrast inference is an **asymptotic paired
bootstrap-`t`** (honest about its guarantee); and the cross-family gating is a
pre-registered closed-testing / sequential-gatekeeping graph with **explicit, pinned**
α-weights. **FWER ≤ 0.05 is established in-document** on these corrected leaves.
(`analysis/closed_testing.py` implements it.)

### 4.1 Graph nodes, edges, and α-weights (LOCKED, in-document — not YAML placeholders)

Family-wise α = 0.05, allocated by a Maurer–Bretz **graphical** gatekeeping scheme.
**All weights below are the locked constants; the YAML merely mirrors them.** The
marginal per-contrast level used inside each IUT leaf is the α currently held by that
node.

| Node | Hypothesis (MARGIN null, R2-B3) | Local test | Initial α-weight |
| --- | --- | --- | --- |
| **G0 (R0)** | `H0^{R0}`: "`T` adds nothing given `[φ_end, NAIT(L)]`" | block-permutation on `T̃=T−ZB_λ` (Eq.5-2p) + BCa floor, **Holm** within {joint, D-only, κ-only}, joint as gatekeeper | `w_0 = 1.00` |
| **G1 (R1)** | `H0^{R1}=⋃_c{g_c ≤ δ_R1}` (a single elementary union-null leaf over the 5 controls) | **INTERSECTION-UNION TEST** over **ALL 5** paired control contrasts: reject iff each `L_c(α_{G1}) > δ_R1` (Eq.7-IUT), each contrast via paired bootstrap-`t` (Eq.7-Bt); inside any closed-test intersection containing `R1`, the FULL 5-control union is required (§4.1b item 3) | inherits α only if G0 rejects |
| **G2t (R2-target)** | `H0^{G2t}=⋃_{b}{gap_b ≤ δ_target}` over the baseline set (a single elementary union-null leaf) | **INTERSECTION-UNION TEST** over **ALL** comparators: reject iff each comparator's `L_b(α_{G2t}) > δ_target` (Eq.7-IUT); inside any closed-test intersection containing `G2t`, the FULL comparator union is required (§4.1b item 3) | inherits α only if G1 rejects |
| **G2r** | `H0: retention-drift ≥ δ_ret` (non-inferiority) | single paired bootstrap-`t`, adjusted **upper** bound below ceiling | weighted-Holm split |
| **G2h** | `H0: hallucination-drift ≥ δ_hall` | single paired bootstrap-`t`, adjusted **upper** bound | weighted-Holm split |
| **G2c** | `H0: cost-gap ≥ δ_cost` | ledger-paired bootstrap-`t`, adjusted **upper** bound | weighted-Holm split |

**Edges (α-propagation), pinned.** On rejection a node passes its α-weight to its
children along locked edges: `G0 → G1` (weight `1.0`), `G1 → G2t` (weight `1.0`),
`G2t → {G2r,G2h,G2c}` with the **locked split** `(w_r, w_h, w_c) = (0.34, 0.33, 0.33)`
(summing to 1), and a recycling edge `{G2r,G2h,G2c} → G2t` (weight `1.0`). These are
concrete constants, pinned here and mirrored in `lattice_v5.yaml`; **not** open
placeholders.

**Why the IUT leaves need NO internal multiplicity weight (R3-B2).** An
intersection-union test of a `J`-way "beats all" claim rejects iff **every** one of the
`J` component tests rejects at the **marginal** level `α`. Its size is `≤ α` *for any
`J`* (Prop. P1-IUT). So G1's five control contrasts and G2t's baseline contrasts each
consume only the **single** node-level α — there is no `1/J` Bonferroni split and no
shared max-T/min-T quantile inside the node. The α-graph above governs only the
**cross-family** gating (R0→R1→R2→drift), where the closed-testing principle applies.

### 4.1b The complete set of local tests on the closed lattice (R2-B4, R3-B2)

The confirmatory family is `{H0^{R0}, H0^{R1}, H0^{G2t}, H0^{G2r}, H0^{G2h}, H0^{G2c}}`
(six **elementary** nulls). Two of them are themselves union nulls:
`H0^{R1} = ⋃_{c} H0^{R1,c}` over the five control contrasts and
`H0^{G2t} = ⋃_{b} H0^{G2t,b}` over the baseline-set comparators. **Crucially, each is a
single elementary leaf of the closed family — its internal components `{H0^{R1,c}}` are
NOT separate members of the closed family** (they are not in the list of six). The
**closed test** rejects an elementary `H_i` iff **every** intersection hypothesis
`H_I` with `i ∈ I` is rejected by a level-α local test, where the local test must be
**level-α for the FULL intersection hypothesis `H_I` it is testing**. Via the graphical
shortcut (Bretz et al. 2009, Prop. 1) — provably equivalent to testing all
`2^6−1 = 63` intersections of the six elementary nulls — the local tests are:

1. **Singletons.** The node's own local test (Table 4.1) at the α it currently holds.
   For `{R1}` and `{G2t}` this is the **IUT** (Eq.7-IUT) over **ALL** of that node's
   components: reject `{R1}` iff **every one of the five control contrasts** clears the
   margin at the node's α (and `{G2t}` iff every baseline comparator clears it). A
   level-α test of `H0^{R1} = ⋃_c H0^{R1,c}` must reject the WHOLE union, so it must
   require ALL five components (Prop. P1-IUT); requiring only a subset would test a
   strictly smaller hypothesis and would NOT be level-α for `H0^{R1}` when the binding
   component is the one omitted.
2. **Any intersection containing R0.** Rejected only if the R0 permutation test
   (Eq.5-2p, Prop. P0) rejects at the α mass on R0 (= full 0.05 at entry). The graph
   routes **all** initial mass through G0, so every intersection containing R0 is gated
   by R0 first (gatekeeper structure).
3. **Any intersection `H_I` containing R1 but not R0.** Because `H_I ⊇ H0^{R1}` and
   `H0^{R1} = ⋃_c H0^{R1,c}` is the full union over all five controls, a level-α local
   test of `H_I` must reject the FULL R1 union null — i.e. the **IUT over ALL FIVE
   control contrasts** (Eq.7-IUT), at the α inherited on `G0→G1`. **The local test does
   NOT drop to the controls "in scope" of the other members of `I`:** an IUT over a
   *proper subset* of R1's components tests `⋃_{c∈subset} H0^{R1,c} ⊊ H0^{R1}`, which is
   anti-conservative for `H_I` whenever the binding control (the one with `g_c = δ_R1`)
   lies outside the subset (a real FWER hole — the round-4 RE-FIX). The same rule applies
   to any `H_I ⊇ H0^{G2t}`: the local test is the IUT over **all** baseline comparators.
   (Other members of `I`, e.g. a drift singleton, contribute their own component to the
   intersection's local test as in item 4; but the R1/G2t union components are tested in
   FULL, never sub-scoped.)
4. **Intersections among `{G2t,G2r,G2h,G2c}`.** `G2t` in scope uses its IUT over **all**
   baseline comparators (item 3 rule); the drift/cost singletons use **weighted
   Bonferroni/Holm** with the pinned `(w_r,w_h,w_c)` split plus the recycling edge. For
   each such intersection the weighted-Bonferroni critical value is `α·(Σ in-scope
   weights)` — the exact Maurer–Bretz local test — and the combined local test for `H_I`
   rejects iff the (full) G2t IUT clears its mass **and** each in-scope drift/cost
   singleton clears its weighted critical value.

> **Why testing R1/G2t in FULL inside every intersection is still valid (and not
> over-conservative for the closed test).** The closed test for elementary `H_i` requires
> rejecting every `H_I ∋ i`. For `i = R1` the only intersections are those with
> `R1 ∈ I`, all of which satisfy `H_I ⊇ H0^{R1}`; the full-union IUT is the *correct*,
> exactly-level-α local test for each (Prop. P1-IUT applied to `H0^{R1}` gives size ≤ α
> regardless of the other members of `I`). For an elementary `H_j` with `j ≠ R1` (e.g.
> `j = G2r`), the intersections `H_I ∋ j` that also contain R1 are tested by the
> *combined* local test (full R1-IUT **and** the G2r component) — requiring the full R1
> union to also reject only makes those intersections **harder** to reject, never easier,
> so it can only *shrink* the rejection region for `H_j` and cannot inflate any type-I
> error. Hence FWER ≤ 0.05 is preserved for **every** elementary null. ∎

The graphical equivalence theorem (Bretz et al. 2009, Prop. 1) states the shortcut
rejects `H_i` **iff** the full closed test on the 63 local tests rejects `H_i`;
`analysis/closed_testing.py` implements the closure with R1 and G2t treated as **single
union-null leaves whose local test, in every intersection that contains them, requires
ALL of their components** (the union-null leaf rule above), and includes an assertion
(`test_layer_routing.py` item (f)) that the shortcut and the brute-force
63-intersection closed test agree on random p-vectors, **including the union-null IUT
leaves and the least-favorable single-binding-control configuration**.

### 4.2 The correct inference for the "beats all" maxima (R3-B2, R3-B3) — the key fix

The R1 and R2-target statistics each assert "`π_ψ` beats a *whole set* of arms." This
is an **intersection-union** problem, and the round-2 min-over-controls (max-T) shared
quantile was the **wrong direction**. Two pieces:

**(A) Per-contrast inference — asymptotic paired bootstrap-`t` (R3-B3).** For control
(or comparator) `c`, with per-seed paired differences (at the locked pools)

```
d_c,s = U_train,s(π_ψ) − U_train,s(π_c) ,   s = 1..S  (S ≥ 20),
```

the gap estimate is `ĝ_c = mean_s d_c,s` and the studentized statistic is
`t_c = (ĝ_c − δ_R1) / se_s(d_c,s)`, `se_s` the seed-standard-error (cluster-robust over
seeds). The reference distribution is the **paired studentized bootstrap-`t`**: for
`b=1..B_bt`, resample seeds with replacement, recompute `t_c^{*(b)} = (ĝ_c^{*(b)} −
ĝ_c)/se^{*(b)}`, and read the `α`-quantile to form the one-sided **lower** confidence
bound

```
L_c(α) = ĝ_c − t^{*}_{c,1−α} · se_s(d_c,s) ,                                       (Eq. 7-Bt)
```

with `t^{*}_{c,1−α}` the bootstrap `(1−α)`-quantile. **This is ASYMPTOTICALLY valid**
(studentized bootstrap, `O(S^{-1})` higher-order accuracy under finite second moments
and i.i.d. seeds); **we claim only asymptotic validity — not exact finite-`S`.** The
earlier sign-flip "exact finite-`S`" claim is **withdrawn (R3-B3)**: i.i.d. seeds make
the `d_c,s` i.i.d. but **not symmetric about their mean**, so the Rademacher sign-flip
reference is not the exact null.

> **Note on recovering exactness (R3-B3, not adopted).** A genuinely *exact*
> finite-`S` test would require an **actual randomization** that induces sign-symmetry
> — e.g. **randomly assigning each seed's run to `π_ψ` vs `π_c`** so that, under the
> sharp null `g_c = δ_R1`, swapping the labels leaves the law of the centered
> difference invariant. Our design pairs *the same seed* across all arms (to remove
> seed variance from the contrast), so no such label randomization exists and
> sign-symmetry does **not** hold by construction. We therefore do **not** claim
> exactness and use the asymptotic bootstrap-`t` (Eq.7-Bt) as the primary, with a
> studentized multiplier bootstrap as the named numerical fallback.

**(B) The "beats all" decision — INTERSECTION-UNION TEST (R3-B2).** Given the
per-contrast lower bounds `{L_c(α)}`, the composite null `H0^{R1}=⋃_c\{g_c≤δ_R1\}` is
rejected by the **IUT**:

```
REJECT H0^{R1}  iff  L_c(α) > δ_R1  for EVERY c ∈ {unif,shuf,rand,global,ada}.     (Eq. 7-IUT)
```

Each component is tested at the **marginal** level `α` — **no** `1/5` split, **no**
shared min-over-controls quantile. The reported **simultaneous lower bound** on the
"beats-all" margin `min_c g_c` is the **minimum of the per-contrast marginal-`α` lower
bounds**:

```
L_{R1} = min_c L_c(α) ,     and we reject iff  L_{R1} > δ_R1.                      (Eq. 7-IUT-CI)
```

The identical construction, with `δ_target` and the baseline-set comparators, gives the
R2-target decision; the single-contrast R2 margins (rel/ret/hall/cost) use the
per-contrast bound (Eq.7-Bt) directly (lower bound above `δ` for wins, upper bound
below the ceiling for drift/cost).

> **Proposition P1-IUT (size of the IUT; exact CONDITIONAL on valid level-`α`
> component tests; Berger 1982).** *(The Berger construction is exact given that each
> `L_c(α)` is a valid level-`α` bound; our implemented `L_c(α)` is the asymptotic paired
> bootstrap-`t`, so the realized per-contrast — and hence IUT — level is asymptotic, not
> finite-sample exact; see the closing sentence.)* Let
> `H0 = ⋃_{c=1}^{J} H0,c` with `H0,c : g_c ≤ δ`, and let each component test reject
> `H0,c` iff `L_c(α) > δ` where `L_c(α)` is a level-`α` lower confidence bound for
> `g_c` (so `P_{g_c}(L_c(α) > g_c) ≤ α` for all `g_c`). The IUT rejects `H0` iff **all**
> `J` components reject. Then for **any** parameter point `θ ∈ H0` there exists at
> least one `c*` with `g_{c*} ≤ δ`, and
> `P_θ(reject H0) ≤ P_θ(L_{c*}(α) > δ) ≤ P_θ(L_{c*}(α) > g_{c*}) ≤ α`. Hence the IUT
> has size `≤ α` **with no multiplicity correction and for every `J`**, regardless of
> the (unknown) dependence among the contrasts. The supremum is attained at the
> least-favorable configuration `g_{c*}=δ`, `g_{c≠c*}→∞`, which is exactly the
> configuration where the round-2 min-over-controls shared quantile was
> anti-conservative. The per-contrast bound `L_c(α)` is the asymptotic bootstrap-`t`
> bound (Eq.7-Bt), so the IUT inherits its **asymptotic** validity (R3-B3). ∎

> **Proposition P1-FWER (closed-testing FWER ≤ 0.05 on the corrected leaves).** Each
> leaf local test holds its level: (a) R0 by Prop. P0 (frozen-nuisance within-stratum
> exchangeability); (b) every union-null leaf (`R1` and `G2t`) is tested, **in every
> intersection that contains it**, by the **FULL-union IUT** (Eq.7-IUT over ALL of its
> components) — by Prop. P1-IUT this has size ≤ its node α for the full union null
> `⋃_c H0,c`, asymptotically, *regardless of the other members of the intersection*; a
> *sub-scoped* IUT (a proper subset of the union's components) is **forbidden** here
> because it tests a strictly smaller hypothesis and is anti-conservative at the
> least-favorable single-binding-control configuration (the round-4 RE-FIX, §4.1b
> item 3); (c) every single margin leaf (drift/cost) by the asymptotic bootstrap-`t`
> (Eq.7-Bt) at its weighted-Holm α. All α-weights and edge splits are the pinned
> constants of §4.1; every intersection local test is the explicit one of §4.1b. By the
> closed-testing principle, the procedure rejects an elementary `H_i` only if **all**
> intersections containing `i` reject at their level; since each such local test holds
> its level (a)–(c) — and the union-null leaves are tested in FULL inside every such
> intersection — `P(reject any true H0) ≤ α = 0.05`. The data-dependent "max over arms"
> does **not** leak type-I error because the IUT (Eq.7-IUT) requires **every** arm to
> clear the margin — it can only make rejection *harder*, never easier, than any single
> contrast. The bound is therefore an **asymptotic** FWER ≤ 0.05 (inheriting the
> bootstrap-`t` guarantee of (b)/(c)); we do **not** claim a finite-`S` exact FWER. ∎

The difference from the round-2 statement: round 2 (i) used a **min-over-controls
shared quantile** (wrong direction, anti-conservative at the least-favorable point) and
(ii) called the leaf **exact at finite `S`** via sign-flip (false). Round 3 (i) uses an
**IUT** (correct direction, exact level `α`, no penalty) and (ii) claims only the
**asymptotic** validity of the bootstrap-`t` it actually has.

### 4.3 Exploratory descendants (outside the confirmatory α)

Ablation contrasts (D-only, κ-only, `J`-on/off, μ=0, the descriptive `Δ_ℓ^{LOL}` of
§3.6, and the §3.5a multi-pool generalization probe) are **exploratory** descendants,
**not** part of the confirmatory α. The α-weights, edges, the split
`(w_r,w_h,w_c)=(0.34,0.33,0.33)`, `B_bt`, the marginal level `α`, and the IUT family
are locked in `lattice_v5.yaml` (mirroring §4.1) before any run.

---

## 5. Changed-from-v4, Preserved-from-v4, and Phase-B modules

### 5.0 Preserved from v4 / prior v5 (verified-correct; re-derivation NOT required)

| Preserved core | Where (code) | Why preservation is safe under the harder bar |
| --- | --- | --- |
| **Policy-value estimand reframing** (`V(π)`, run-level treated unit, no SUTVA, no per-example potential outcome) | §3.5 | **Verified sound (round 1).** Round 3 only (i) makes the intra-`A` mask→rank map closed-form and (ii) scopes `V` to the locked pools; neither touches the SUTVA-free run-level logic. |
| **Coupling identity** `Σ_{ℓ∈A} ψ_ℓ = β̂_T·T̃` (selection = sum, routing = support of ONE endpoint+NAIT-residualized attribution) | Eq.5-4; `analysis/layer_attribution.py` | Verified-sound algebraic identity; unit-tested in- and out-of-sample. Round 3 unchanged. |
| **Dual-ridge FWL endpoint+NAIT-residualized partial-R² estimand** | `residual_test.py::dual_ridge_partial_out`, `cross_fit_partial_r2` | Operates on `Y_obs` only; dual-feasibility / residual-orthogonality lemmas stand. Frozen `B_λ` is algebraically identical in-sample (Eq.5-1d). |
| **Conditional-null block permutation** on orthogonalized residuals `T̃` within family×fold strata, **with frozen cross-fit nuisance** | `residual_test.py::block_permutation_test` | Validity argument identical; B7 sharpened (freeze nuisance, Prop. P0). R1 is a separate real-training test and does not reuse this permutation. |
| **Outcome `Y_obs` sign = +** | `analysis/outcome_y.py::loci_influence` (v4 Eq.17, `+`) | `ψ_ℓ` inherits the same `Y_obs`; `U_train` is a *separate* metric. |
| **Two-layer Holm multiplicity** | `analysis/residualize.py::two_layer_holm` | Reused as the **within-trajectory G0 node** of the §4 graph. |
| SW2 `D_ℓ`, curvature `κ_ℓ`, `T(x)` operator + κ non-functionality proof | `trajectory.py` | Unchanged; `ψ_ℓ` is built *from* these over `A`. |
| Brier propriety + G6 factuality precondition (NeuroTrace-IT's strictly-proper Brier calibrator, §3.8a) | `analysis/drift.py` | **Proven in-doc (RE-FIX-1):** strictly-proper Brier rule `ℓ_Brier(q,y)=(q−y)²`, `E(q−y)²=(q−p)²+p(1−p)` uniquely minimized at `q=p` (P-RC1). Orthogonal to routing. |
| Monotone-submodular (1−1/e) selector | `selection.py` | Reused; routing applied **after** selection (§3.8). |
| LOCI clustering, G7 `Y_obs`-reliability precondition | `outcome_y.py` | Unchanged; G7 (ICC ≥ 0.6; proxy↔retrain ρ ≥ 0.3) remains the precondition for any claim on the proxy `Y_obs`. |
| Endpoint reconstruction NAIT comparator | `baselines/nait.py` | Retained as the FULL endpoint control `φ_end`; **augmented** by faithful Eq.5 `nait_layerwise` over `L` (§3.3). |
| Compute-matched ledger | `cost_model.py` (§3.7) | Preserved; measured-FLOP/skip-flag is the MF-6 fix; shared `r_0` + shared `capacity_match` make the match reduce to `A`. |
| Three-pool leakage firewall | `pool_firewall.py` (§3.1) | Preserved; now also separates `Y_obs` (P_train) from `U_train` (P_val) and locks the partition (R3-B4). |
| Expanded baseline set | `baseline_registry.yaml` (§5.4) | Preserved. |

**Why re-derivation is not needed:** no preserved estimand's *assumptions* are
weakened. The partial-R² is a dataset-level partial effect of `T` on `Y_obs` over an
*enlarged* control `[φ_end, NAIT(L), C]` (strictly harder). The coupling identity is
pure algebra. The estimand reframing **replaced** the unsound `τ_ℓ` with a sound
`V(π)` (round 1, verified); rounds 2–3 only **complete and correct** its specification
and inference (policy domain `A`, closed-form `capacity_match`, out-of-sample `B_λ`,
margin tests, IUT, pool-conditional scope, honest asymptotics). No preserved object is
touched.

### 5.1 Changed from prior v5 (round-4 RE-FIX deltas, then round-3 deltas)

- **RD5 — Brier calibrator strict propriety proven in-doc (RE-FIX-1).** §3.8a adds the
  in-doc proof for NeuroTrace-IT's **strictly-proper Brier calibrator** `ℓ_Brier(q,y)=(q−y)²`
  (the v4 §2.6 rule), via `E(q−y)²=(q−p)²+p(1−p)` uniquely minimized at `q=p` (Prop.
  P-RC1). The bare assertion is replaced by proof. No new scoring rule is introduced.
- **RD6 — `capacity_match` is score-free and total (RE-FIX-2).** The map now reads only
  `(m_A,R_tot,r_max)` (no `ψ`), is defined on the feasible domain `DOM`, and has no
  infeasible/support-growth branch; feasibility is enforced upstream by
  `make_feasible_mask` (Eq.6-0e), the single place an arm's score is read. The old
  `ψ`-referencing CAP_SPILL growth branch is **deleted** (§3.5).
- **RD7 — every R1 control is a deterministic map (RE-FIX-3).** `π_rand/π_shuf` use fixed
  seeds ⊕ `H(x)`; `π_global` a fixed `A_glob`; `π_ada` a fixed `W_ada`; the fixed
  randomization is part of the policy, not the `V(π)` seed-expectation (§3.5).
- **RD8 — `π_ada` importance frozen by a separate seed (RE-FIX-4).** The AdaLoRA warm-up
  is run **once** under a pre-registered `seed_ada` and its importance scores are
  persisted and reused by every confirmatory seed, so `π_ada(x)` is a deterministic
  function of `x` and the **only** seed inside `V(π)` (Eq.6-2) is the per-run training
  seed. This closes the round-3 internal inconsistency where the warm-up read the
  training seed, which would have made `π_ada` non-deterministic in `x` and smuggled a
  second seed into the policy-value expectation (§3.5, control 6).

- **RD1 — `capacity_match` is now closed-form (R3-B1).** The mask→rank map is the
  largest-remainder apportionment with ascending-`ℓ` tie-break, automatic cardinality
  renormalization, and `r_max`-capped deterministic spillover (Eq.6-0b/0d) — and, after
  RE-FIX-2, **score-free on a feasible domain** with the empty-mask fallback moved to the
  constructor (Eq.6-0c/0e). `U_train(π)` is one implementation-independent
  functional (§3.5).
- **RD2 — IUT replaces the min-over-controls quantile (R3-B2).** "Beats every
  control/comparator" is decided by an intersection-union test: each contrast at the
  marginal level `α`, no penalty, exact level `α` (Eq.7-IUT, Prop. P1-IUT). The
  simultaneous bound is the **min of per-contrast marginal-`α` lower bounds**
  (Eq.7-IUT-CI). The round-2 shared max-T quantile is **deleted** (§3.5, §4.2).
- **RD3 — Honest asymptotic per-contrast test (R3-B3).** The false "exact finite-`S`
  sign-flip" is **withdrawn**; each contrast is a paired studentized bootstrap-`t` with
  **asymptotic** validity (Eq.7-Bt). The only route to exactness (an actual
  seed-to-arm randomization) is stated and explicitly **not** adopted (§4.2).
- **RD4 — Estimand scoped to the locked pools (R3-B4).** `V(π)` is the
  **pool-conditional** seed-expectation (Eq.6-2); the seed-mean matches it exactly; the
  claim is scoped to the locked partition. A separate, **non-confirmatory** multi-pool
  probe (§3.5a) is the route to a pool-marginal statement, with its own multiplicity.

### 5.1b Changed from v4 → prior v5 (round-1/2 deltas, retained)

- Round 1: routing estimand reframed as a POLICY VALUE `V(π)` (whole run as treated
  unit; B1,B2); headline routing object `supp(ψ)` (B3); outcome symbol split
  `Y_obs`/`U_train` (B4); per-layer `τ_ℓ` deleted, only descriptive `Δ_ℓ^{LOL}` kept;
  closed-testing FWER procedure added (B5); NAIT over full `L` (B6); R0 permutation
  made precise, frozen nuisance (B7).
- Round 2: routing domain pinned to `A` with `L∖A` a fixed `r_0` substrate (R2-B1);
  out-of-sample frozen coefficient map `B_λ` (R2-B2); margin-test framing (R2-B3);
  closed-testing lattice with pinned weights (R2-B4).

### 5.2 Changed from v4 (carried forward, unchanged by rounds 1–3)

- C1 headline is a narrowly-stated method claim (R0 + R1 + R2); C3 R0 control extended
  to NAIT; C4 R1 is real training arms; C5 three-pool firewall; C6 `J`
  renamed/validated; C7 measured compute ledger; C8 closed-testing graph; C9 expanded
  baselines; C10 faithful NAIT.

### 5.3 The falsifiable non-stitch chain and kill-gates

```
R0 (residual ψ beyond [φ_end, NAIT(L)] exists, on Y_obs)
   ⇒ R1 (pool-conditional policy value V(π_ψ) beats V(π_c) by ≥ δ_R1 for EVERY control
          c∈{unif,shuf,rand,global,ada}, each by its OWN marginal-α lower bound
          L_c(α)>δ_R1 — intersection-union test — in REAL matched-compute runs over A)
   ⇒ R2 (selection on Σ_{ℓ∈A} ψ + routing policy π_ψ beats stronger-NAIT(L) and EVERY
          baseline at a MEASURED compute-matched budget, R2-target as an IUT, each
          single sub-claim as a margin test).
```

| Gate | Test | Fail action |
| --- | --- | --- |
| **R0** | block-perm `p<α_R0` on `T̃=T−ZB_λ` (residual to `[φ_end, NAIT(L)]`, frozen nuisance) + BCa above `floor_partial`, stable across `r` | `stop_main_novelty_claim` (reduce to NAIT, clean null) |
| **R1** | union null `⋃_c{g_c≤δ_R1}`; reject iff **every** `L_c(α)>δ_R1` (IUT, paired bootstrap-`t`), pool-conditional on `P_val` (real runs over `A`) | `drop_routing_keep_selection` (fall back to v4) |
| **R2** | compute-matched win: R2-target IUT over stronger-NAIT(L) + every baseline; each margin sub-claim's bound clears its `δ_k` | `no_method_win_claim` |
| **R3** | extraction-parity ≤ 2.0× kept; savings claim only if measured (skip-flag) | savings claim forbidden / `high_cost_analysis` |
| **G6** | factuality precondition | `λ_f := 0` |
| **G7** | `Y_obs`-reliability precondition (ICC, proxy↔retrain) | primary not run on proxy |
| **J-val** | `J_{c,ℓ}` vs real anchor layer-freeze `Δ_ℓ^{LOL}` (§3.6) | drop `J` from router |

A bag of tricks cannot pass R1: R1 *is* the test that the support of the same vector
that drives selection yields a higher-value training policy (by margin `δ_R1`, against
**every** control individually) under matched compute.

### 5.4 Locked baseline set (MF-8)

Confirmatory comparators (`baseline_registry.yaml`, extended additively):
`nait_layerwise` over **`L`** (stronger NAIT, decisive), `endpoint_neuron_selection`
(FULL control), `random_subset`, `full_data_it` (upper bound),
`quality_score_selection`, `influence_gradient_selection` / **LESS** (gradient
representation — the "Critical Look" comparator), `zero_shot_select`. Routing-arm
comparators (R1, all over `A` with shared `r_0` on `L∖A` and shared `capacity_match`):
`uniform_lora`, `shuffled_psi`, `random_mask`, `global_surgical_layer`, `AdaLoRA`,
optional `NeFT_neuron_select`, `lora_moe_routing`. The **8-anchor NAIT**
(`nait_layerwise_anchor`) is a *secondary* diagnostic, reported only after the full-`L`
reproduction passes its unit check; **never** the comparator the headline uses. "Beats
NAIT" requires beating the **stronger full-`L`** variant; "beats SOTA selection"
requires beating **every** comparator in the confirmatory set under the §4.2
intersection-union test.

### 5.5 Phase-B modules to implement (names + spec)

All additive, pure-stdlib, build-now/run-later; no model load, no server call.

1. **`src/neurotrace_it/layer_function.py`** *(IMPLEMENTED; do-not-run, green)* —
   `frozen_layer_ablation_profile` (Eq.6-3, `J_{c,ℓ}` over `A`),
   `validate_J_against_freeze(...)` (§3.6 vs `Δ_ℓ^{LOL}` Eq.6-4),
   `routing_policy(ψ, J; τ_sel)` → per-example raw **anchor** mask `m_A(x) ∈ {0,1}^A`
   with `L∖A` pinned at `r_0` (Eq.6-0/6-0a);
   **`make_feasible_mask(raw, score, R_tot, r_max)`** (Eq.6-0e) → lifts any raw mask
   (incl. empty ⇒ uniform-over-`A`) into the feasible domain `DOM` using ONLY the arm's
   own `score` (the **single place** `ψ` or a control's fixed key is read);
   **`capacity_match(m_A, R_tot, r_max)`** → the closed-form largest-remainder rank
   vector with ascending-`ℓ` tie-break and capped same-support spillover
   (Eq.6-0b/0d), a **pure score-free function on `DOM` with no `ψ` argument and no
   infeasible branch** (RE-FIX-2); **single source of the rank map for all arms**;
   `control_mask(arm, x; seed_rand, seed_shuf, A_glob, W_ada)` → the deterministic
   per-control raw mask of RE-FIX-3; `leave_one_layer_redistribute(ℓ)` =
   `capacity_match` on a feasible `A∖{ℓ}` mask (Eq.6-4).
2. **`src/neurotrace_it/cost_model.py`** *(IMPLEMENTED; do-not-run, green)* —
   `gate1_multiplier`, `extraction_parity_check` (2.0× kill),
   `compute_match_ledger` (§3.7: params, optimizer slots, **realized FLOPs**,
   wall-clock, batch policy, skip-flag; shared `r_0` + shared `capacity_match`
   accounted equally), `routing_training_savings` (claim only on measured-reduction
   flag), `gate1b_deployability(R, R*)`.
3. **`src/neurotrace_it/baselines/nait_layerwise.py`** — faithful NAIT over **`L`**
   (§3.3): both Alg.1 first/last-diff and token-mean variants; per-layer PCA `v_ℓ`,
   sign-align, `s_NAIT` layer-sum over `L` (base Eq.5), persisted `proj_ℓ`, top-k;
   plus the **gated** secondary 8-anchor score `s_NAIT^{A}`.
4. **`src/neurotrace_it/analysis/layer_attribution.py`** —
   `frozen_nuisance_map(Z, T, Y_obs; λ, Ω)` → `(B_λ, b_λ^Y)` (Eq.5-1/5-1e);
   `residualize_out_of_sample(x; B_λ)` → `T̃(x)=T(x)−Z(x)B_λ` (Eq.5-1d);
   `per_layer_policy_score(...)`: decompose cross-fit `β̂_T` into `ψ_ℓ` (Eq.5-3, over
   `A`) and assert the coupling identity `Σ_{ℓ∈A} ψ_ℓ = β̂_T·T̃` **in- and
   out-of-sample** (Eq.5-4). Labeled **predictive policy, not causal**.
5. **`src/neurotrace_it/analysis/routing_intervention.py`** — Gate R1 (§3.5):
   harness spec for the six real
   masked-LoRA arms over `A` (shared `r_0`, shared `capacity_match`); the
   **pool-conditional** policy-value estimator `V̂(π)=seed-mean U_train | locked pools`
   (Eq.6-1/6-2); the per-control gaps `ĝ_c` (Eq.6-2g); the **paired studentized
   bootstrap-`t`** per-contrast lower bound `L_c(α)` (Eq.7-Bt); the
   **intersection-union decision** `reject iff ∀c L_c(α)>δ_R1` and the simultaneous
   bound `L_{R1}=min_c L_c(α)` (Eq.7-IUT/7-IUT-CI). **No per-example `τ` estimator**,
   **no min-over-controls shared quantile**.
6. **`src/neurotrace_it/analysis/matched_budget.py`** — Gate R2: paired
   compute-matched comparison vs stronger-NAIT(L) and **every** baseline; asserts the
   `compute_match_ledger` equality within tolerance and `Σ_{ℓ∈A} r_ℓ = R_tot` with
   `L∖A=r_0`; R2-target as an **IUT** over the baseline set (Eq.7-IUT); each single
   sub-claim as a margin test (Eq.7-Bt).
7. **`src/neurotrace_it/analysis/closed_testing.py`** — §4 graph: nodes
   {G0, G1, G2t, G2r, G2h, G2c}, **pinned** α-weights/edges/split
   `(0.34,0.33,0.33)`, the **IUT** leaves for G1/G2t (Eq.7-IUT), the per-contrast
   **bootstrap-`t`** (Eq.7-Bt), the **explicit 63-intersection local tests** (§4.1b)
   with a brute-force vs shortcut equivalence assertion, the Prop. P1-IUT/P1-FWER
   checks on synthetic nulls; wraps `two_layer_holm` as the G0 within-trajectory node.
8. **`src/neurotrace_it/schemas_v2.py` extension (additive optional only)** —
   populate `RouterOutputs` (`anchor_mask`, `rank_per_anchor`, `total_rank_A`,
   `r0_substrate`, `J_profile_hash`) and add additive fields `psi_per_anchor`,
   `policy_value` (per arm, pool-conditional), `pool_hashes {train,val,dep}` (locked
   partition), `nuisance_map_hash` (`B_λ, b_λ^Y`),
   `routing_policy_value {R1_pass, gaps_per_control, margin, per_contrast_lower_bounds,
   iut_simultaneous_bound}`, `compute_ledger`, and `control_provenance
   {seed_rand, seed_shuf, seed_ada, A_glob_hash, W_ada, ada_importance_hash,
   ada_warmup_seed_outside_V}` (persists the once-frozen AdaLoRA importance and the
   separate `seed_ada`, RE-FIX-4). V1/V2 records without them still validate.
9. **`src/neurotrace_it/analysis/pool_firewall.py`** — `split_pools(...)` (§3.1):
   disjoint hashed `P_train/P_val/P_dep` (**locked partition**), leakage assertions
   (no `P_dep` outcome in the `P_dep` decision path; `Y_obs` only on `P_train`,
   `U_train` only on `P_val`/`P_dep`-eval); asserts `ψ` on `P_val/P_dep` uses **frozen**
   `B_λ` only; optional `regenerate_partition(g)` for the §3.5a non-confirmatory probe.
10. **`tests/test_layer_routing.py`** — formula-evaluation unit checks (NOT
    evidence): (a) **coupling identity** `Σ_{ℓ∈A} ψ_ℓ = β̂_T·T̃` (Eq.5-4) **in-sample
    AND out-of-sample**; (b) **`capacity_match`** is a **pure function of
    `(m_A,R_tot,r_max)` that reads no `ψ`** (RE-FIX-2): on every **feasible** mask
    (`k·r_max≥R_tot`, `k∈{k_min,…,|A|}`) it yields `m_A∈{0,1}^A`, conserves
    `Σ_{ℓ∈A} r_ℓ=R_tot`, respects `r_ℓ≤r_max`, is **deterministic** under the
    ascending-`ℓ` tie-break, keeps `L∖A=r_0`, and **never grows support**; plus
    `make_feasible_mask` (Eq.6-0e) lifts every raw mask (incl. the empty mask ⇒
    uniform-over-`A`) to cardinality `≥k_min` using ONLY the arm's own score, and two
    different `score` inputs to `capacity_match` (it ignores them) give the identical
    rank vector; (c) the six-arm **IUT R1**
    procedure: the per-contrast bootstrap-`t` attains nominal one-sided level on a null
    DGP, the **IUT attains size ≤ α at the least-favorable configuration** (one
    `g_c=δ_R1`, others large) **where a min-over-controls quantile would inflate**, and
    the margin shift `δ_R1` enters the statistic (reject only when **every** gap exceeds
    `δ_R1`); (c2) **each control policy is a deterministic map of `x`** (RE-FIX-3 +
    RE-FIX-4): with fixed `seed_rand/seed_shuf/seed_ada/A_glob/W_ada`, re-running
    `π_rand,π_shuf,π_global,π_ada` twice on the same `x` returns the identical mask, the
    control output is invariant to the (unused-by-`V`) implementation RNG state, **and
    `π_ada`'s importance is frozen by `seed_ada` (NOT the training seed) so changing the
    training seed leaves `π_ada(x)` unchanged**; (d) `nait_layerwise` matches base
    Eq.5 **summed over `L`** on a synthetic activation tensor (and the gated 8-anchor
    variant matches its restricted sum);
    (e) `pool_firewall` rejects any `P_dep`-outcome dependent decision and any
    `Y_obs`/`U_train` cross-contamination, rejects any in-sample-only residualization on
    `P_val`, and confirms the partition is locked for confirmatory estimands;
    (f) `closed_testing` **shortcut equals brute-force 63-intersection closed test**
    (including the IUT leaves) and controls FWER (Prop. P1-FWER) on a synthetic
    correlated null; (g) R0 permutation placebo holds nominal level under the
    frozen-nuisance exchangeability of Prop. P0. The **factuality calibrator** (RE-FIX-1,
    §3.8a) is NeuroTrace-IT's strictly-proper Brier rule and is unit-checked separately in
    `test_trajectory_selection.py`: the Brier risk is minimized at the true probability
    (P-RC1), `factuality_drift` counts unsupported claims, ECE is exact on bracketing
    slices, and the G6 gate sets `λ_f := 0` on failure.
11. **`configs/experiments/lattice_v5.yaml`** *(additive; does NOT overwrite
    `lattice_v4.yaml`)* — `routing.load_bearing: true`; `routing.domain: A`;
    `routing.r0_substrate`; `routing.tau_sel`, `R1_arms`, `δ_R1`, `R1_seeds`,
    `R2_margins` (`δ_target, δ_rel, δ_ret, δ_hall, δ_cost`),
    `nait_variant: [alg1_L, token_mean_L]`, `nait_anchor_secondary: true`,
    `capacity_match: {rule: largest_remainder, tie_break: ascending_layer_index,
    cap: r_max, spillover: largest_remainder_same_support, reads_psi: false,
    domain: feasible_masks}`,
    `make_feasible_mask: {k_min: "ceil(R_tot/r_max)", empty_mask: uniform_over_A,
    score_source_per_arm: {psi: pi_psi, fixed_key: controls}}`,
    `controls: {seed_rand: 20250619, seed_shuf: 20250620, seed_ada: 20250621,
    A_glob: (rule: top-4 anchors by pooled |ψ_ℓ|), W_ada: 200,
    ada_warmup_seed_outside_V: true, randomization: part_of_policy_not_V}`
    (all LOCKED design choices),
    `factuality_calibrator: {score: brier, strictly_proper: true, recalibration: affine,
    threshold: 0.5}` (NeuroTrace-IT's own Brier calibrator; G6 thresholds in
    `lattice_v4.yaml:factuality_gate_g6`),
    `estimand_scope: pool_conditional_locked`, `compute_ledger.tolerance: 0.02`,
    the closed-testing α-graph mirroring §4.1 (`w_0=1.00`, edges `1.0`,
    `(w_r,w_h,w_c)=(0.34,0.33,0.33)`, recycle `1.0`),
    `inference: {decision: intersection_union_test, per_contrast: paired_bootstrap_t,
    marginal_level: α, B_bt: 10000 (LOCKED), fallback: studentized_multiplier}`,
    `generalization_probe: {confirmatory: false, partitions_G: 5 (LOCKED),
    accounting: holm_within_probe}`, `pool_split_seed=20250619`, `ρ_J=0.30`,
    `floor_partial=0.01`.
    **`server.authorized: false`** preserved. The numeric **decision
    margins/thresholds are now LOCKED** (pre-registered DESIGN CHOICES fixed before any
    run, per §5.6), as are the control seeds, `A_glob`/`W_ada` rules, `B_bt`, and
    `partitions_G`; the **multiplicity structure constants** (weights/edges) are pinned
    per §4.1. Only genuinely **empirical** quantities (resolved commit/tokenizer SHAs,
    measured FLOPs, observed gaps) remain `DATA_NEEDED`, resolved at run authorization.
12. **`tests/test_endpoint_baseline.py`** *(IMPLEMENTED; green; the Phase-A faithful-reproduction
    contract)* — formula-evaluation unit checks (NOT evidence) that pin the NAIT
    reproduction contract independently of `test_layer_routing.py`: (a) the layerwise
    score reproduces **base Eq.5 summed over `L`** (`s_NAIT=Σ_{ℓ∈L} A^(ℓ)·v_ℓ`) on a
    synthetic activation tensor; (b) the per-layer PCA direction `v_ℓ` (Eq.3) and its
    sign-alignment (Eq.4) are correct (sign flips when `μ_diff·v_ℓ<0`); (c) the Alg.1
    first/last-diff (Eq.2) and the token-mean variant are both reproduced and the
    **stronger** one is the comparator; (d) the gated **8-anchor** restricted score
    `s_NAIT^{A}=Σ_{ℓ∈A} A^(ℓ)·v_ℓ` equals the restricted sum and is never the headline
    comparator; (e) top-k selection (Eq.6) is the budget-`B` argmax with deterministic
    ties; (f) the existing endpoint control `baselines/nait.py` (`φ_end`,
    `concat[start,end]`) is the **distinct** endpoint-only object — it must NOT coincide
    with the layerwise full-`L` sum, so the comparator and the control are not conflated.

### 5.6 Locked specifics (MF-9 — exact, no fabricated results)

- **Seeds/folds.** Shared seeds `0..19` (`configs/seeds/paper_20.txt`); 10-fold
  family-stratified, example-disjoint cross-fit for R0 (frozen nuisance maps per
  fold); ≥20 seeds for R1/R2; `B_bt = 10000` bootstrap resamples for the per-contrast
  bootstrap-`t` (LOCKED design constant in `lattice_v5.yaml`).
- **Margins/thresholds (pre-registered DECISION values — now LOCKED in
  `lattice_v5.yaml`, not results).** These are design choices fixed before any run:
  `floor_partial=0.01`, `δ_R1=0.01`, `δ_target=0.01`, `δ_rel=0.00`, `δ_ret=0.02`,
  `δ_hall=0.02`, `δ_cost=0.05`, `ρ_J=0.30`, `τ_sel` (rule: `P_train` quantile for mean
  raw cardinality `k̄=4`), `r_0=4`, `r_max=16`, `R_tot=64`, marginal level `α=0.05`,
  `B_bt=10000`, `pool_split_seed=20250619`, control seeds
  `seed_rand=20250619, seed_shuf=20250620, seed_ada=20250621`, `A_glob` (rule: top-`4`
  anchors by pooled `|ψ_ℓ|`), `W_ada=200`; plus the frozen v4 constants `K=64` SW2
  projections, `|A|=8`, subsample cap `512`, `r ∈ {8,16,32}` PCA poles,
  `k_min=⌈R_tot/r_max⌉=4`, and the cross-validated `λ_ridge` grid. Each carries a
  provenance/power rationale inline in `lattice_v5.yaml`. Only genuinely EMPIRICAL
  outputs (resolved SHAs, measured FLOPs, observed gaps) stay `DATA_NEEDED`; this
  document fabricates no numeric result. **`r_max` is pinned in
  §5.6 so that `|A|·r_max ≥ R_tot` for the locked `R_tot`, hence the uniform mask
  `1_A ∈ DOM` and `k_min ≤ |A|`; together with `make_feasible_mask` (Eq.6-0e), which
  lifts EVERY arm's mask to cardinality `≥ k_min` BEFORE any rank call, `capacity_match`
  is a pure score-free largest-remainder map on `DOM` with NO infeasible and NO
  `ψ`-dependent branch (RE-FIX-2).** (The **multiplicity weights**
  `w_0=1.00`, edges `1.0`, `(w_r,w_h,w_c)=(0.34,0.33,0.33)`, recycle `1.0` are
  **structural constants**, pinned in §4.1, not data-derived.)
- **`capacity_match` rule (locked, Eq.6-0b; RE-FIX-2).** Largest-remainder
  apportionment of `R_tot` over `supp(m_A)`; remainder top-up tie-break = **ascending
  layer index `ℓ`**; per-layer cap `r_max` with largest-remainder spillover **among the
  same ON layers**. It is a **pure function of `(m_A, R_tot, r_max)` — it reads no `ψ`
  and has no infeasible branch**: feasibility (`|supp|·r_max ≥ R_tot`) is guaranteed by
  `make_feasible_mask` (Eq.6-0e) before any call; the empty mask is lifted to
  uniform-over-`A` **at the constructor**. **One** score-free function for every arm.
- **Control-policy specifics (locked, RE-FIX-3 + RE-FIX-4).** Each control is a single
  deterministic map: `π_rand`/`π_shuf` use **fixed seeds** `seed_rand`/`seed_shuf` ⊕
  `H(x)`; `π_global` uses the fixed block `A_glob` (top-`k_glob` by pooled `P_train`
  mean `|ψ_ℓ|`, ties ascending-`ℓ`); `π_ada` uses the fixed warm-up budget `W_ada` run
  **once** under the **separate pre-registered seed `seed_ada`** (NOT the training seed),
  then top-`k` by the persisted importance (ties ascending-`ℓ`). The fixed randomization
  is **part of the policy definition, not the `V(π)` seed-expectation**; the **only**
  seed inside `V(π)` is the per-run training seed.
- **`τ_sel` rule (locked).** raw `m_ℓ(x)=1[ψ_ℓ(x) ≥ τ_sel]` for `ℓ∈A`, with `τ_sel` the
  `P_train` quantile yielding the pre-registered mean per-example raw anchor-mask
  cardinality `k̄` (so `π_rand` can match pre-feasibility cardinality); fit on `P_train`,
  frozen, applied to `P_val`/`P_dep` via the frozen `B_λ`-derived `ψ`; then lifted to
  `DOM` by `make_feasible_mask` (Eq.6-0e).
- **Factuality calibrator (locked, RE-FIX-1, §3.8a).** NeuroTrace-IT's strictly-proper
  **Brier** rule `ℓ_Brier(q,y)=(q−y)²` (Eq.8-1, the v4 §2.6 calibrator); P-RC1 proves the
  population risk is uniquely minimized at `q=p`. G6 precondition unchanged; fail ⇒ `λ_f:=0`.
- **Estimand scope (locked, R3-B4).** Every confirmatory `V(π)` / `g_c` is the
  seed-expectation at the **single locked** `(P_train,P_val,P_dep)`; the multi-pool
  generalization probe (§3.5a) is **non-confirmatory** with its own Holm accounting.
- **Failure actions.** Every gate's `failure_action` is the one in §5.3 and is written
  into `lattice_v5.yaml`; no gate is open-ended.

---

## 6. Honest risks / limitations (and how the design bounds them)

1. **Routing policy may not beat uniform/AdaLoRA (R1 null).** **Bound:** R1 is a
   pre-registered **policy-value** kill-gate with a clean fallback (drop routing, keep
   v4 selection-only). Publishable either way.
2. **`ψ_ℓ` is predictive, not causal.** **Bound:** `ψ` is never claimed causal; it
   only *proposes* the policy `π_ψ`; the policy value `V(π_ψ)` (real runs) certifies
   it; collinearity/ridge/fold instability that move `ψ` mass without changing
   prediction fail the `π_shuf` arm of R1.
3. **No per-example routing effect exists (the prior fatal flaw).** **Bound:** the
   estimand is a **run-level policy value** `V(π)` (§3.5, verified sound); we never
   claim a per-example `τ_ℓ`. The only per-layer notion is the **descriptive**
   `Δ_ℓ^{LOL}` (§3.6), labeled non-causal and outside the confirmatory α.
4. **The policy could be ill-defined inside the anchor set (R3-B1, RE-FIX-2).**
   **Bound:** `capacity_match` is a **closed-form, deterministic, SCORE-FREE** (no `ψ`)
   largest-remainder map on the **feasible domain `DOM`**, with no infeasible/support-
   growth branch (feasibility/termination lemma, §3.5); the empty mask and the minimum
   cardinality `k_min` are handled **upstream** by `make_feasible_mask` (Eq.6-0e), the
   single place an arm's score is read. `U_train(π)` is one implementation-independent
   functional; unit-tested for every feasible cardinality and for score-invariance of the
   rank map (item (b)).
4b. **Controls could be implementation-dependent (RE-FIX-3).** **Bound:** every R1
   control is a **single deterministic map of `x`** under fixed pre-registered
   `seed_rand/seed_shuf/A_glob/W_ada`; the fixed randomization is **part of the policy**,
   not the `V(π)` seed-expectation; unit-tested for reproducibility (item (c2)).
4c. **The factuality calibrator's optimality was unproven (RE-FIX-1).** **Bound:** §3.8a
   proves NeuroTrace-IT's **Brier** calibrator is **strictly proper** (truthful `q=p`
   uniquely minimizes the population Brier risk, P-RC1, via `E(q−y)²=(q−p)²+p(1−p)`); G6
   stays a precondition (fail ⇒ `λ_f:=0`); outside the confirmatory routing α.
5. **Deployed attribution could be implementation-dependent (R2-B2).** **Bound:** the
   nuisance is a **frozen coefficient map** `B_λ`; `T̃(x)=T(x)−Z(x)B_λ` and `ψ(x)` are
   closed-form on any `P_val`/`P_dep` row (Eq.5-1d), unit-tested out-of-sample.
6. **Margin claims from a `0`-clearing CI would be invalid (R2-B3).** **Bound:** every
   confirmatory null is the **margin** null and the relevant bound must clear the
   **margin** (`L_c(α)>δ_R1` per control; drift upper bound below ceiling); the margin
   enters inside the studentized statistic (Eq.7-Bt).
7. **"Beats every control" could be tested anti-conservatively (R3-B2).** **Bound:**
   the decision is an **intersection-union test** (Berger 1982): each contrast at the
   marginal level `α`, **no** shared min-over-controls quantile, exact level `α` at the
   least-favorable configuration (Prop. P1-IUT). The simultaneous bound is the **min of
   the per-contrast marginal-`α` lower bounds**.
8. **The per-contrast test is not exact at finite `S` (R3-B3).** **Bound:** we claim
   only the **asymptotic** validity of the paired studentized bootstrap-`t`
   (Eq.7-Bt); the false "exact finite-`S` sign-flip" is withdrawn. The one route to
   exactness (an actual seed-to-arm randomization) is stated and **not** adopted, so no
   overclaim is made.
9. **Estimand/estimator mismatch (R3-B4).** **Bound:** `V(π)` is the
   **pool-conditional** seed-expectation at the locked pools — exactly what the
   seed-mean and its bootstrap-`t` estimate; the claim is scoped to those pools. The
   pool-marginal statement is a separate **non-confirmatory** probe (§3.5a) with its
   own multiplicity.
10. **Compute matching is fragile.** **Bound:** the §3.7 ledger requires *measured*
    FLOP/wall-clock equality and a skip-flag; the shared `r_0` substrate and shared
    `capacity_match` make the match reduce to `A`; unmatched arms are
    diagnostic-labeled; "savings" forbidden without measured reduction.
11. **Off-manifold `J`.** **Bound:** renamed and **validated-or-dropped** vs a real
    anchor layer-freeze policy-value difference (§3.6).
12. **Outcome leakage / symbol conflation.** **Bound:** three-pool firewall +
    `Y_obs`/`U_train` split; no `P_dep` outcome enters the `P_dep` decision; the two
    outcome symbols never co-occur in an equation (§3.0, §3.1).
13. **NAIT under-specification + comparator weakening.** **Bound:** both variants
    pre-registered and run **over the full `L`**; 8-anchor is secondary and gated;
    claims gated on the stronger full-`L` variant (§3.3). (Routing over `A` is a
    separate object and does not weaken the `L`-scored comparator.)
14. **Multiplicity inflation from data-dependent maxima.** **Bound:** formal
    closed-testing graph with **every** local intersection test written out (§4.1b),
    **all** weights pinned (§4.1), the "beats all" leaves as **IUTs** (which only make
    rejection harder), and an in-document **asymptotic FWER ≤ 0.05** proof
    (Prop. P1-FWER) for `S ≥ 20`.
15. **Permutation validity.** **Bound:** frozen cross-fit nuisance maps, no ridge
    refit in permutations; the R0 guarantee is the precise Prop. P0 (asymptotically
    valid, finite-sample approximately-exact under within-stratum residual
    exchangeability), not an over-stated "exact."
16. **Lateral relabeling risk.** The honest concession: routing existed in v4 §2.9 as
    an ablation. v5's *new* content is the policy-value certification (R1), the
    well-posed **and fully-specified** estimand `V(π)` over `A` (with closed-form
    `capacity_match`), the frozen out-of-sample attribution map, the leakage firewall,
    the compute ledger, and the closed-testing graph with **correct IUT leaves** and an
    honest asymptotic FWER. If R1/R2 null, v5 falls back to v4's cap; that is the
    intended falsification path.
17. **No empirical claim yet (RR Stage-1).** Everything here is design.
    `server.authorized: false`; the only numeric checks are formula evaluations
    (coupling identity in/out of sample, `capacity_match` conservation across all
    cardinalities, IUT size at the least-favorable configuration + margin-shift on a
    null DGP, closed-testing shortcut=brute-force + FWER on a correlated null, R0
    placebo), labeled **not evidence**.

---

## 7. What a 9+ reviewer should now see

- A **narrowly stated** method (selection+routing coupled through one
  endpoint+NAIT-residualized attribution, `Σ_{ℓ∈A} ψ_ℓ = β̂_T·T̃`) that does what NAIT
  structurally cannot.
- A **well-posed AND fully-specified routing estimand** — a **pool-conditional policy
  value** `V(π)=E_seed[U_train(π)|locked pools]` whose treated unit is the whole run
  (no per-example potential outcome, no cross-example SUTVA — verified sound), whose
  policy domain is **exactly the anchor set `A`** with `L∖A` a fixed common substrate,
  and whose intra-`A` mask→rank map `capacity_match` is a **closed-form, deterministic,
  SCORE-FREE** (reads no `ψ`) largest-remainder apportionment on a **feasible domain
  `DOM`** with **no infeasible branch** — feasibility and the empty-mask fallback live in
  a single upstream constructor `make_feasible_mask` (RE-FIX-2). **Every R1 control is a
  single deterministic implementation-independent map of `x`** under fixed pre-registered
  seeds/blocks (RE-FIX-3). The prior ill-posed `τ_ℓ` is gone and **each** policy (test
  and control) is one number per run.
- A clean **separation of outcomes**: `Y_obs` (R0 regression target) vs `U_train`
  (R1/R2 training metric), never conflated.
- A **proven-in-doc scoring precondition**: NeuroTrace-IT's factuality calibrator, the
  strictly-proper **Brier** rule `ℓ_Brier(q,y)=(q−y)²`, is shown **strictly proper**
  (truthful report `q=p` uniquely minimizes the population Brier risk, P-RC1, via
  `E(q−y)²=(q−p)²+p(1−p)`) — replacing a bare assertion with a full proof (RE-FIX-1,
  §3.8a).
- **Confirmatory inference that is correct in direction and honest about its
  guarantees**: "beats every control/comparator" is an **intersection-union test**
  (Berger 1982; no multiplicity penalty, exact level `α`, valid at the least-favorable
  configuration), each per-contrast bound is an **asymptotic** paired studentized
  bootstrap-`t` (no false finite-`S` exactness), and the estimand is **scoped to the
  locked pools** the seed-mean actually estimates.
- A **mechanism certificate** (the preserved, verified dual-ridge FWL partial-R² on
  `Y_obs`, residualized against NAIT over `L`, with a precise frozen-nuisance
  permutation) licensing the method non-circularly.
- A **faithful NAIT comparator over the full released `L`**, a **leakage firewall** with
  a **locked partition**, a **measured compute ledger**, a **validated-or-dropped**
  layer profile, and a **formal closed-testing graph with an in-document asymptotic
  FWER ≤ 0.05 proof** (Prop. P1-FWER) whose "beats all" leaves are intersection-union
  tests, with every local test written out and every weight pinned.
- A clean **null path at every gate**, so a negative result is a publishable
  falsification, not a buried run.

*Provenance:* Stage-1 Registered Report, design-only. No `src/`, `paper/`,
`pre_registration.md`, or config file is modified by this document; no experiment,
training, extraction, or model load is run; no git commit is made;
`server.authorized: false` preserved throughout. The only new artifact is this file.
`layer_function.py` and `cost_model.py` were the headline Phase-B modules of this
design; **they (and every other §5.5 module) have since been implemented** under
`src/neurotrace_it/` as do-not-run pure-Python with a **green 93-test harness**
(see the implementation-status banner at the top of this file, `run_packet.md` §1,
and `main.tex` Table 1). `server.authorized: false`; no run executed.

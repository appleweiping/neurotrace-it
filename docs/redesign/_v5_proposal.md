# REDESIGN v5 PROPOSAL — neurotrace-it (HARDER-BAR / top-venue elevation)

Status: **BUILD-NOW / RUN-LATER Stage-1 Registered Report.** `server.authorized`
stays **false**; NO experiment, training, extraction, or model load is run by this
proposal. ZERO fabricated numbers — every results slot stays `DATA_NEEDED`. This
document is **design-only and additive**: it does NOT modify `src/`, `paper/`,
`docs/pre_registration.md`, or any config; it specifies what v5 *would* lock and
which Phase-B modules would be added under a future authorization.

This file plays two roles in one voice:
1. **Hostile top-venue reviewer** naming the single biggest weakness that caps v4
   below 9 at NeurIPS/ICML/ICLR.
2. **Creative methodologist** proposing a concretely stronger v5 contribution that
   removes that cap *without* discarding the verified-correct v4 core.

---

## 0. TL;DR (one screen)

- **v4 is a measurement paper wearing a method-paper coat.** Its single headline
  estimand is a dataset-level **partial-R² of a trajectory block over endpoints**
  on a LOCI *proxy* outcome. Even if every gate passes, the deliverable is "a
  regression coefficient is non-zero" — *associational*, on a proxy, with the only
  genuinely *new* engineering idea (the user's **layer-wise routing**, idea #2)
  explicitly demoted to a NON-load-bearing ablation that **is not even
  implemented** (`layer_function.py` absent; `router_ablation.enabled: false`).
  A hostile reviewer reads this as "an expensive correlational re-confirmation that
  reasoning paths carry signal," not a selection method that beats NAIT.

- **v5 thesis (one sentence):** *A single per-example trajectory object — the
  layer-resolved distributional+curvature signature `T(x)` — is used **end to end**
  to (i) decide **which examples** to keep and (ii) decide **which layers each kept
  example is allowed to write to**, and this trajectory-aware **layer-routed**
  selector beats the faithfully-reproduced NAIT endpoint baseline on
  retention-adjusted utility **at matched data, parameter, optimizer-state, and
  extraction-parity compute budget**.* The residual partial-R² test is **demoted
  from the headline to the identifying mechanism check** that licenses the method.

- **One unifying principle (not a bag of tricks):** *the same residual trajectory
  geometry that predicts an example's utility beyond its endpoints also predicts
  **where** (which layers) that utility is realized.* Selection and routing are two
  readouts of one quantity `T(x)`; the kill-gate ties them so that if the routing
  signal is not the *same* signal as the selection signal, the method collapses to
  NAIT and the paper reports a clean null.

- **Preserved verified-correct cores (unchanged, re-used as-is):** dual-ridge FWL
  endpoint-residualized partial-R² estimand; conditional-null block permutation on
  endpoint-orthogonalized residuals; outcome `Y` sign = +(useful cluster → positive
  utility); two-layer Holm. These are *promoted* to the mechanism-identification
  layer, where their correctness is exactly what makes the routing claim non-
  circular.

---

## 1. The biggest weakness of v4's central claim (reviewer voice)

**The headline contribution is a partial-R², and a partial-R² is not a method.**

v4's central claim (REDESIGN_v4 §3.1; `paper/main.tex` "Falsifiable central
claim") is:

> "There exists a temporal/distributional component of the reasoning trajectory …
> that, after partialling out the FULL endpoint signature `φ_end` … still carries
> information predictive of retention-adjusted fine-tuning utility …"

Four compounding problems cap this below 9:

1. **It is associational, on a proxy, at the dataset level.** The estimand is
   `partial_R²_T` of `T` over `φ_end` against a **LOCI first-order influence**
   outcome `Y`. Even with G7 passing (ICC ≥ 0.6; proxy↔retrain ρ ≥ 0.3), the claim
   that survives is "a coefficient is non-zero on a proxy." The actual deployment
   claim a venue cares about — *does a selector built on `T` beat NAIT at matched
   budget?* — is buried in **P3 (aggregate gates), not the headline**, and is not
   the object the theory is built around. A reviewer asks: "you proved a regression
   coefficient; show me the *selector* that wins."

2. **The only genuinely new mechanism is missing.** The user's two ideas were
   (#1) use the full trajectory as a distribution, and (#2) **layer-wise selection:
   freeze some layers for some data to train others, because different layers have
   different functions.** v4 implements #1 (SW2 `D_ℓ`, curvature `κ_ℓ`) but **#2 is
   demoted to a "validated ablation, not load-bearing"** (§2.9), with `m_ℓ ≡ 1` in
   the main path (Eq. 11), `router_ablation.enabled: false`
   (`lattice_v4.yaml:175`), and **`layer_function.py` does not exist on disk**
   (confirmed: `src/neurotrace_it/` has no `layer_function.py`; the V2 schema even
   reserves a `RouterOutputs` block that is never populated by a real router).
   So the differentiating idea is vaporware and the shipped method is
   *NAIT-similarity + a residual test*.

3. **"Beyond endpoints" is a weaker frame than the base paper deserves.** Read
   literally, **NAIT is not an endpoint method.** From the base PDF (§3.2.1–3.2.2,
   Eq. 1–5): NAIT records activations over **all decoder layers**, takes the
   **mean activation across all K tokens**, extracts a **per-layer PCA direction**
   `v_ℓ = PCA(ΔA^(ℓ))`, and scores by **summing the per-layer projections**
   `s_y = Σ_ℓ (A^(ℓ) · v_ℓ)` (base Eq. 5). So NAIT already (a) uses a *mean over the
   whole sequence*, not start/end, and (b) **aggregates layers by summation**,
   discarding which layer carried the signal. v4's `φ_end = concat[h(start),
   h(end)]` is a *reconstruction* that is simultaneously richer at the token level
   and **strictly poorer at the layer level** than NAIT's own Eq. 5. A hostile
   NAIT author will say "your 'endpoint baseline' is a strawman — it is neither what
   I do nor my strongest configuration." This both threatens the faithful-baseline
   claim **and** points to the real gap NAIT left open: **NAIT collapses the layer
   axis.** That is the seam v5 should attack.

4. **The cost story makes the partial-R² self-defeating.** v4 already concedes
   (§2.7, extraction-parity kill) that if trajectory extraction pushes the honest
   multiplier > 2.0× the method is "high-cost-analysis, no deployment claim." So in
   the regime where the residual signal is *interesting* (rich full traces), the
   method may be *disallowed from claiming deployment*; and where it is cheap, the
   signal may vanish. A pure partial-R² paper has no answer to "why pay for this?"
   — because the answer ("you get a better selector + a cheaper training run via
   layer routing") is exactly the part v4 amputated.

**Single sentence:** *v4 proves that reasoning paths carry residual signal but
never converts that signal into a method that does something NAIT cannot — so the
contribution reads as a costly diagnostic, not a selection algorithm, and caps at
~7.*

---

## 2. The sharpened v5 central contribution

### 2.1 One crisp thesis

> **Trajectory-Aware, Layer-Routed Instruction-Tuning Data Selection (LATTICE-R).**
> The same per-example trajectory object `T(x) = ({D_ℓ}, {κ_ℓ})` that predicts an
> example's retention-adjusted utility **beyond its endpoint signature** also
> predicts **which layers** that example should be allowed to update. v5 therefore
> couples two decisions through one quantity: a **selection** decision (keep the
> budget-`B` examples) and a **per-example layer-routing** decision (a capacity-
> matched mask `m(x) ∈ {0,1}^L` reallocating a fixed LoRA rank budget to the layers
> the trajectory implicates). The claim is a **method claim at matched budget**:
> LATTICE-R beats the faithfully-reproduced NAIT selector on retention-adjusted
> target utility at matched data **B**, matched total trainable parameters / rank /
> optimizer-state, and **extraction-parity** compute — and the residual partial-R²
> test is the *mechanism certificate* that the win comes from trajectory geometry,
> not from spending more.

This is a **method paper**: the deliverable is *a selector+router that wins at
matched budget*, with the v4 estimand re-cast as the falsifiable *why-it-works*
gate.

### 2.2 Why it is novel vs the base paper (NAIT) — specific, no hand-waving

NAIT (base PDF, Eq. 1–6; ICLR 2026) has two structural commitments that LATTICE-R
*directly* breaks, and a third axis it never touches:

- **(N1) Sequence collapse → first moment.** NAIT summarizes each layer by the
  **mean activation across all K tokens** (base §3.2.1, "we compute the mean
  activation across all K tokens"). v5's `D_ℓ` (sliced-Wasserstein-2 to a target
  cloud) and `κ_ℓ` (ordered bending energy) are **provably not functions of that
  per-layer mean** (REDESIGN_v4 Prop. §2.5 / `paper` Prop. "Curvature is not a
  function of the first-moment pool"; implemented and unit-tested in
  `trajectory.py::trajectory_curvature`). This is the v4 novelty, *kept*.

- **(N2) Layer collapse → summation, then `top-k`.** NAIT scores by **summing**
  per-layer projections `s_y = Σ_ℓ (A^(ℓ)·v_ℓ)` (base Eq. 5) and selects `top-k`
  (base Eq. 6). **It throws away which layer carried the alignment and it tunes all
  layers uniformly.** No NAIT result conditions the *training footprint* on the
  *per-layer evidence it already computed*. LATTICE-R's **layer routing is the
  contribution NAIT structurally cannot express**: it keeps the per-layer axis as a
  *decision variable* for both selection and capacity allocation. **This is the
  user's idea #2, finally made load-bearing.**

- **(N3) Selection-only → joint selection+allocation.** NAIT (and LIMA, AlpaGasus,
  SelectIT, LESS — base Table 1) all answer only "which examples." None answer
  "given this example's internal evidence, **where** in the network should its
  gradient be allowed to land." v5 unifies the *coreset* question (which data) with
  a *capacity-allocation* question (which parameters) through one trajectory object.

### 2.3 Named prior art and exactly how v5 differs

| Prior art (specific) | What it does | Why v5 is not it |
| --- | --- | --- |
| **NAIT** (Chen et al., *Neuron-Aware Data Selection*, ICLR 2026; base paper) | per-layer mean activation → per-layer PCA dir `v_ℓ` → **sum over layers** → top-k examples (Eq. 1–6) | v5 keeps the per-layer axis as a *routing* decision and tests utility **residual to a faithfully reproduced NAIT score** (not a strawman); reproduces NAIT's *own* Eq. 5 layer-sum as the comparator, not just `concat[start,end]` |
| **LESS** (Xia et al., ICML 2024) | gradient/influence datastore → select examples whose grads align with a target; *example-level* | v5's `Y` *reuses* an influence-on-validation attribution (method-neutral) only to **define the outcome**; the *selector* is trajectory-geometric, and v5 adds **layer routing** LESS never addresses |
| **TracIn** (Pruthi et al., 2020) | training-run gradient-trace influence | same as LESS: used to *define* `Y`, not as the method; no layer allocation |
| **LoRA / AdaLoRA** (Hu et al., 2022; Zhang et al., AdaLoRA 2023) | low-rank adapters; AdaLoRA **prunes rank by importance during training** | AdaLoRA reallocates rank **globally from the training signal**; v5 reallocates rank **per-example, a priori, from the candidate's trajectory geometry before training** — a *data-conditioned* allocation, not a training-time pruning |
| **Surgical fine-tuning / layer-selective tuning** (Lee et al., "Surgical Fine-Tuning", ICLR 2023; Zhang et al., "Let's Focus on Neuron", arXiv 2403.11621 — cited in base refs) | choose a *fixed* layer block to tune for a *whole task* | v5 routes **per example**, conditioned on that example's measured trajectory, not a single task-global block; the allocation is a function `m(x)` of `T(x)`, capacity-matched |
| **Sliced-Wasserstein / MMD** (Bonneel 2015; Gretton 2012) | distributional distances | tools, not the contribution; reused for `D_ℓ` |
| **Double-ML / FWL partialling-out** (Chernozhukov 2018; Frisch–Waugh–Lovell) | orthogonalized partial effects | reused **verbatim** as the mechanism-certificate estimand (§4) |

**The one-line differentiation:** *NAIT sums the layer axis away and tunes
uniformly; LATTICE-R keeps the layer axis as a per-example decision variable for
both data selection and capacity allocation, and proves the routing signal is the
same residual-trajectory signal that beats NAIT's endpoints.*

---

## 3. The falsifiable test / kill-gate that makes v5 NON-stitched

The danger in adding routing is "bag of tricks": a reviewer says selection and
routing are two unrelated heuristics bolted together. v5 forecloses that with a
**single unifying principle made falsifiable**.

### 3.1 The unifying principle (stated as an identity to test)

> **Routing-selection coherence.** Let `T(x) = ({D_ℓ(x)}, {κ_ℓ(x)})` be the one
> trajectory object. Define the **per-layer residual utility attribution**
> `ψ_ℓ(x)` = the layer-`ℓ` coordinate's contribution to the endpoint-residualized
> utility coefficient (the per-layer decomposition of `β̂_T` from the v4 ridge-FWL
> estimand, §4). The principle is: *the layers whose residual coordinate predicts
> utility (`ψ_ℓ` large) are exactly the layers routing should write to.* Formally
> the **routing mask is a thresholding of the same `ψ_ℓ` that the selector sums**:
> `m_ℓ(x) = 1[ ψ_ℓ(x) ≥ τ ]`, with the selection score `Σ_ℓ ψ_ℓ(x)`. Selection and
> routing are the **sum** and the **support** of one vector `ψ(x)`.

This makes them *the same object*: there is no second model, no second feature set,
no independently tuned head. If `ψ(x)` is meaningless, *both* die together.

### 3.2 The kill-gate (decisive, pre-registered — falsifies the unification)

**Gate R0 — Mechanism certificate (PRESERVED v4 test, promoted).** The
endpoint-residualized cross-fit / block-permutation / cluster-BCa test of "`T` adds
nothing given full `φ_end`" must reject (two-layer Holm), with `partial_R²_T`
BCa-CI above floor and robustness-stable across `r ∈ {8,16,32}`. *This is exactly
the v4 co-primary test, unchanged.* If R0 fails, the trajectory carries no residual
signal and **the entire method reduces to NAIT** — report a clean null. (v4 §2.3–
§2.4, `residual_test.py`, `residualize.py` — reused verbatim.)

**Gate R1 — Routing-selection coherence (NEW, the unification falsifier).** The
per-layer attribution `ψ_ℓ` driving **selection** must be the same vector driving
**routing**. Pre-registered test: the routing mask derived from `ψ(x)` must, on
held-out folds, **explain the per-example utility better than (a) a layer-uniform
mask and (b) a `ψ`-shuffled mask** (layers permuted across examples). Concretely,
the held-out utility of the routed-training proxy under `m(x)=1[ψ_ℓ≥τ]` must exceed
both controls by a pre-registered margin, with a conditional-null permutation that
**shuffles the layer assignment of `ψ` within strata** (so the null is "routing
uses `ψ` no better than chance layer assignment"). **Fail R1 ⇒ the routing claim is
dropped; the paper falls back to v4's selection-only claim** (still publishable as
the mechanism result, but not the headline method). This gate is what makes
selection+routing *one principle*: it directly tests that the support of `ψ` is
load-bearing, not decorative.

**Gate R2 — Capacity-matched method win (NEW headline gate).** At **matched
budget** (data `B`, total LoRA rank `R_tot`, optimizer-state slots, steps, tokens,
**and** extraction-parity compute), LATTICE-R's retention-adjusted target utility
must beat the faithfully-reproduced NAIT selector by the pre-registered margins
(≥ 0.02 absolute / ≥ 0.03 relative; retention-drift disadvantage ≤ 0.01;
hallucination drift ≤ 0.01 — the existing G1–G3). The routing arm must **conserve
capacity exactly** (rank reallocated, not added): `Σ_ℓ r_ℓ = R_tot`. Any variant
that cannot match capacity exactly is reported as **capacity-unmatched diagnostic**,
never as the headline. **Fail R2 ⇒ no method-win claim.**

**Gate R3 — Honest extraction-parity cost (PRESERVED + sharpened).** The honest
three-line cost model (§2.7) is extended with a **routing line**: routing adds
*zero* extraction cost (it reuses the same `T(x)` already computed for selection)
and *reduces* training FLOPs/optimizer state by concentrating rank — so v5 can, for
the first time, argue a **net compute argument**: even if trajectory extraction
costs more up front, routing can *recoup* it at training time. The extraction-parity
kill (> 2.0× ⇒ high-cost-analysis) is retained unchanged; v5 additionally reports
the **training-side savings** so the end-to-end multiplier is honest in both
directions.

### 3.3 Why this is "one principle, not a stitch"

The chain is a single conditional: **R0 (residual exists) ⇒ R1 (its per-layer
support is the routing mask) ⇒ R2 (routing+selection on that support wins at matched
budget).** Each arrow is falsifiable and each failure has a pre-registered fallback.
There is exactly one feature object (`T(x)`), one attribution (`ψ_ℓ`), and one
estimand (the v4 ridge-FWL partial effect, now decomposed per layer). Selection is
`Σ_ℓ ψ_ℓ`; routing is `supp_τ(ψ)`. A bag of tricks cannot pass R1, because R1
*specifically* tests that the two uses share the same vector.

---

## 4. Exactly what changes from v4, and what is preserved (and why preservation is safe)

### 4.1 PRESERVED verbatim (verified-correct; re-derivation NOT required)

| Preserved core | Where (code) | Why preservation is safe under the harder bar |
| --- | --- | --- |
| **Dual-ridge FWL endpoint-residualized partial-R² estimand** | `analysis/residual_test.py::dual_ridge_partial_out`, `cross_fit_partial_r2` | The estimand's *correctness* is what makes R1/R2 non-circular: the per-layer `ψ_ℓ` is a decomposition of the **already-orthogonalized** `β̂_T`, so routing cannot be an artifact of `T` re-encoding `φ_end`. Promoting it from "headline" to "mechanism certificate" does not change a line of math — `Theorem (dual feasibility)` and `Lemma (residual orthogonality)` in `paper/main.tex` stand unchanged. |
| **Conditional-null block permutation** (permute endpoint-orthogonalized residuals `M_λ T` within family×fold strata) | `residual_test.py::block_permutation_test` | v5's R1 *reuses the same conditional-null machinery*, only changing what is permuted (layer assignment of `ψ`) — the validity argument (permute post-orthogonalization residuals, never raw rows) is identical and already proven. |
| **Outcome `Y` sign = +(useful cluster → positive utility)** | `analysis/outcome_y.py::loci_influence` (Eq. 17, `+`) | The routing target inherits the same `Y`; a sign error would invert both selection and routing identically, so the verified `+` convention is exactly what keeps `ψ_ℓ` interpretable as "this layer raises utility." |
| **Two-layer Holm multiplicity** (joint gatekeeper) | `analysis/residualize.py::two_layer_holm` | v5 adds R1/R2 as **new families** under the *same* Holm scaffold (see §4.3); the gatekeeper logic is reused, so family-wise α = 0.05 control is preserved by construction. |
| SW2 `D_ℓ`, curvature `κ_ℓ`, `T(x)` operator | `trajectory.py` | unchanged; `ψ_ℓ` is built *from* these, not instead of them |
| Brier propriety + G6 factuality gate | `analysis/drift.py` | unchanged; orthogonal to routing |
| Monotone-submodular `(1−1/e)` selector | `selection.py` | reused; routing changes only the per-layer *utility decomposition* feeding `u(x)`, not the submodular coverage structure (see §6 risk on submodularity) |
| LOCI clustering, G7 reliability | `outcome_y.py` | unchanged |
| Faithful NAIT comparator | `baselines/nait.py` | **upgraded** (see §4.2 change C5) to reproduce NAIT's *own* Eq. 5 layer-sum + per-layer PCA direction, not only `concat[start,end]` |

**Why re-derivation is not needed:** none of the preserved estimands' *assumptions*
are weakened by adding routing. The partial-R² is still a dataset-level partial
effect of `T` over `φ_end`; routing only *reads out* its per-layer structure. The
permutation null is still "exchangeable conditional on controls." Holm still
controls a finite family. The harder bar is met by **adding** a method layer on top
of correct estimands, not by re-opening them.

### 4.2 CHANGED / ADDED (the v5 deltas)

- **C1 — Headline re-framing.** The central claim moves from "residual partial-R²
  exists" (associational) to "**layer-routed trajectory selector beats NAIT at
  matched budget**" (method). The partial-R² becomes the **mechanism certificate
  (Gate R0)**, not the contribution. (`paper/main.tex` abstract + Contributions
  list would be rewritten under authorization; not edited now.)

- **C2 — Per-layer residual attribution `ψ_ℓ` (NEW estimand readout).** Decompose
  the orthogonalized coefficient `β̂_T` into per-layer contributions
  `ψ_ℓ(x) = β̂_{D,ℓ}·\tilde D_ℓ(x) + β̂_{κ,ℓ}·\tilde κ_ℓ(x)` (the residualized,
  per-layer coordinate products), where `\tilde·` are the `M_λ`-orthogonalized
  trajectory residuals already produced by the cross-fit. This is **pure post-
  processing of existing residuals** — no new model.

- **C3 — Causal layer-importance `I_{c,ℓ}` (the user's idea #2, IMPLEMENTED).**
  Build the validation-only, frozen, hashed layer-function profile
  `I_{c,ℓ} = (Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ)) / Acc_c(model)` (v4 Eq. 13,
  *specified but never coded*). v5 implements it in a new `layer_function.py`.

- **C4 — Capacity-matched router (LOAD-BEARING now).** `m_ℓ(x) = 1[ψ_ℓ(x)·I_{c,ℓ}
  ≥ τ]` with **exact rank reallocation** `r_ℓ = R_tot / |{ℓ: m_ℓ=1}|` (clipped),
  conserving `Σ_ℓ r_ℓ = R_tot`, optimizer-state slots, and effective rank. The V2
  schema's existing `RouterOutputs` block (`schemas_v2.py`) is finally populated.

- **C5 — Faithful NAIT reproduction upgraded.** Add a **NAIT-Eq.5 comparator**:
  per-layer mean activation → per-layer PCA direction `v_ℓ` → layer-sum score
  `Σ_ℓ (A^(ℓ)·v_ℓ)` → top-k (base Eq. 1–6), alongside the existing `concat[start,
  end]` endpoint reconstruction. Both are reported; "beats NAIT" wording is gated on
  beating the **stronger** of the two (closes the strawman risk in §1.3 of this
  doc). The existing `baselines/nait.py` stays as the endpoint reconstruction; a new
  `nait_layerwise` variant is added.

- **C6 — New kill-gates R1/R2/R3** (§3.2), added to the pre-registration amendments
  (recorded here, not written into `pre_registration.md` in this RR).

- **C7 — Cost model gains a routing/training-savings line** (§3.2 R3).

### 4.3 Multiplicity bookkeeping (so the new gates don't inflate α)

The two-layer Holm scaffold is extended, not replaced:
- **Within-trajectory family** {joint, D-only, κ-only} — unchanged (Layer 1).
- **R1 routing-coherence** and **R2 matched-budget win** enter **Layer 2 (across
  metric families)** as two additional families {target, retention, hallucination,
  **routing-coherence**, **layer-allocation**, cost}, Holm-corrected as before, and
  **conditioned on the R0 gatekeeper passing** (joint block). The composition rule
  ("within first, then across on gatekeeper-conditioned p-values") is reused from
  `residualize.py::two_layer_holm`, so family-wise α = 0.05 is preserved by the
  same proof.

---

## 5. New / expanded Phase-B modules (names + 1-line spec each)

All additive, pure-stdlib, build-now/run-later; no model load, no server call. Names
mirror the v4 deliverable order (endpoint baseline first; routing reads off existing
residuals).

1. **`baselines/nait_layerwise.py`** — faithful NAIT Eq. 1–6 comparator: per-layer
   mean activation → per-layer PCA direction `v_ℓ` → layer-sum score → top-k (the
   *stronger* NAIT baseline that closes the strawman risk).
2. **`analysis/layer_attribution.py`** — `per_layer_residual_attribution(...)`:
   decompose the cross-fit `β̂_T` into `ψ_ℓ(x)` from the *already-orthogonalized*
   `M_λ T` residuals (selection = `Σ_ℓ ψ_ℓ`; routing = `supp_τ ψ`).
3. **`layer_function.py`** — `causal_layer_importance` (v4 Eq. 13, finally coded),
   `route_layers(ψ, I; τ)`, `capacity_match(mask, R_tot)`: per-example mask with
   **exact** total-rank reallocation (the user's idea #2, made load-bearing).
4. **`analysis/routing_coherence.py`** — `routing_selection_coherence_test(...)`:
   Gate R1 — held-out, conditional-null permutation that **shuffles the layer
   assignment of `ψ`** within strata; rejects iff routed mask beats uniform AND
   `ψ`-shuffled controls by the locked margin.
5. **`analysis/matched_budget.py`** — `capacity_matched_compare(...)`: Gate R2 —
   paired matched-budget comparison (LATTICE-R vs stronger-NAIT) on retention-
   adjusted utility with effect sizes + cluster-BCa CIs over ≥20 seeds; asserts
   `Σ_ℓ r_ℓ = R_tot`.
6. **`cost_model.py`** — `gate1_multiplier` (incl. differential trajectory
   extraction/storage), `extraction_parity_check`, **`routing_training_savings`**
   (the new training-side recoupment line), `gate1b_deployability(R, R*)`.
7. **`schemas_v2.py` extension (additive optional fields only)** — populate the
   existing `RouterOutputs` with `psi_per_layer`, `mask`, `R_tot`, `r_per_layer`,
   `I_profile_hash`; add `routing_coherence {R1_pass, margin, perm_p}` as an
   additive optional block (V1/V2 records without it still validate).
8. **`tests/test_layer_routing.py`** — formula-evaluation unit checks (NOT
   evidence): (a) `ψ_ℓ` summed over layers recovers the scalar `β̂_T·\tilde T`
   contribution (decomposition identity); (b) `capacity_match` conserves
   `Σ_ℓ r_ℓ = R_tot` exactly; (c) R1 permutation attains nominal type-I on a null
   synthetic DGP where `ψ` carries no layer structure; (d) the stronger-NAIT
   layer-sum comparator matches base Eq. 5 on a synthetic activation tensor.
9. **`configs/experiments/lattice_v5.yaml`** (additive; does NOT overwrite
   `lattice_v4.yaml`) — flips `router_ablation.enabled → routing.load_bearing:
   true`, adds `routing.tau`, `R1_margin`, `R1_permutations`, `R2_margins`, the
   stronger-NAIT comparator id, `capacity_match: exact`; **`server.authorized:
   false`** preserved.

---

## 6. Honest risks / limitations + how the design bounds them

1. **Routing may not improve over uniform LoRA (R1/R2 null).** The honest prior is
   that per-example layer routing is hard and may not beat a well-tuned uniform
   adapter. **Bound:** R1 and R2 are pre-registered kill-gates with explicit
   fallbacks — fail R1 ⇒ drop routing, keep the v4 selection-only result; fail R2 ⇒
   no method-win claim. The paper remains publishable as the mechanism result even
   if routing nulls, so the design is **falsification-robust**.

2. **Capacity-matching is fragile (the v4 §2.9 caveat persists).** Exact rank,
   optimizer-state, and effective-rank matching is achievable; **exact FLOP equality
   under heterogeneous per-example masks is not.** **Bound:** any variant that
   cannot match capacity exactly is labeled **capacity-unmatched diagnostic**, never
   the headline; the matched arm reallocates a *fixed* `R_tot`, and FLOP residuals
   are reported, not hidden.

3. **`ψ_ℓ` decomposition can be unstable when `D`/`κ` columns are collinear across
   layers.** **Bound:** the decomposition is read off the **ridge-orthogonalized**
   residuals (regularization already conditions the system); R1's `ψ`-shuffle
   control directly tests whether the per-layer structure is real or noise — if
   `ψ_ℓ` is just noise, the shuffle control matches and R1 fails closed.

4. **Submodular `(1−1/e)` guarantee interaction.** Routing changes the *training
   footprint* per example but the **selection utility `u(x)` stays a scalar**
   (`u(x)=max(0, Σ_ℓ ψ_ℓ − λ_r r̂ − λ_f f̂)`), so the facility-location objective
   `F(S)` remains monotone-submodular and the `(1−1/e)` proof (`selection.py`,
   `paper` Thm) is untouched. **Bound:** routing is applied *after* selection, on the
   selected set, so it never enters the submodular objective and cannot break the
   guarantee. (Stated explicitly to pre-empt the reviewer worry.)

5. **NAIT-reproduction dispute.** Reproducing NAIT's Eq. 5 (per-layer PCA direction
   + layer-sum) requires choices NAIT under-specifies (which layers, PCA rank).
   **Bound:** v5 reports **both** NAIT variants (endpoint-reconstruction and
   Eq.5-layerwise), pins the anchor-layer grid and PCA rank in `lattice_v5.yaml`,
   and gates "beats NAIT" on beating the **stronger** variant — so the claim cannot
   rest on a weak baseline.

6. **Cost honesty cuts both ways.** Routing's training-side savings (R3) could be
   over-credited. **Bound:** the savings line is reported with the **same**
   amortization discipline (`R=1` default, break-even `R*`) as the extraction line,
   and the extraction-parity kill (> 2.0× ⇒ high-cost-analysis) is retained
   unchanged, so a net-compute claim must clear *both* directions honestly.

7. **Proxy outcome still upstream of everything.** Selection, routing, and `ψ_ℓ` all
   inherit the LOCI proxy `Y`. **Bound:** G7 (ICC ≥ 0.6; proxy↔retrain ρ ≥ 0.3,
   lower CI > 0) remains the precondition; fail G7 ⇒ primary not run on the proxy
   (unchanged from v4). Routing inherits the same gate, so no routing claim is made
   on an unreliable `Y`.

8. **No empirical claim yet (RR Stage-1).** Everything above is design.
   `server.authorized: false`; the only numeric checks are formula evaluations
   (decomposition identity, capacity conservation, R1 type-I on a null DGP), labeled
   **not evidence**. In-principle acceptance is conditional on executing the locked
   plan under authorization.

---

## 7. What a 9+ reviewer should now see

- A **method** (selector **+** load-bearing router) that does something NAIT
  structurally cannot (keep the layer axis as a per-example decision), **beating the
  strongest faithful NAIT reproduction at matched budget** (R2).
- A **single unifying principle** — selection is `Σ_ℓ ψ`, routing is `supp_τ ψ` —
  with a dedicated falsifier (R1) that *prevents* the "bag of tricks" reading.
- A **mechanism certificate** (the verified v4 residual partial-R²) that licenses
  the method and is correct by the already-discharged theory.
- An **honest, two-sided cost model** that can argue net compute via training-side
  recoupment, finally answering "why pay for trajectories?".
- A clean **null path** at every gate, so a negative result is a publishable
  falsification, not a buried run.

*Provenance:* Stage-1 Registered Report, design-only and additive. No `src/`,
`paper/`, `pre_registration.md`, or config file is modified by this proposal; no
experiment, training, extraction, or model load is run; no git commit is made;
`server.authorized: false` preserved throughout. The only new artifact is this file.

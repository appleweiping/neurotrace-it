# REDESIGN v2 — neurotrace-it

> **SUPERSEDED (2026-06-19) — historical snapshot.** This is the v2 entry in the
> numbered redesign series and is retained only as a changelog. The decision
> thresholds quoted below (trajectory–endpoint margin 0.02, relative gain 0.03,
> drift ≤ 0.01, cost ≤ 2.0×) are **no longer active**. The live, locked thresholds
> are in `configs/experiments/lattice_v5.yaml` and `docs/redesign/REDESIGN_v5.md`
> (normalized `[0,1]` scale: `delta_R1 = delta_target = 0.01`,
> `delta_ret = delta_hall = 0.02`, `delta_cost = 0.05`). `lattice_v5.yaml` is the
> single source of truth; nothing in this file should be read as a current gate.

Status: BUILD-NOW / RUN-LATER. Design + theory + code-readiness only.
No experiment is run here. `server.authorized: false` is preserved. No numbers
are fabricated; the only numeric checks below are **closed-form / quadrature
formula evaluations**, explicitly labeled "formula evaluation, not evidence."
This document is **additive** (adds only `docs/redesign/REDESIGN_v2.md`); no
other file is modified or deleted; no git commit.

This redesign reuses the existing governance verbatim: the seven-family
`configs/baselines/baseline_registry.yaml` (with the mandatory NAIT-style
`endpoint_neuron_selection`, arXiv 2603.13201), the numeric gates in
`configs/experiments/first_gate.yaml` / `formal_neurotrace_it.yaml`
(trajectory–endpoint margin 0.02, target/adjusted relative gain 0.03, retention
drift ≤ 0.01, hallucination drift ≤ 0.01, cost ≤ 2.0×), the metric API in
`src/neurotrace_it/metrics.py` (`retention_adjusted_gain`, `passes_drift_gate`,
`passes_cost_gate`), and the manifest contracts in
`src/neurotrace_it/schemas.py` and `schemas/selection_manifest.schema.json`.

v1 received a **Strong Reject** (external automated review; the original review
file lives outside this repository and is intentionally not referenced by an
absolute path here for anonymized-review hygiene).
v2 keeps the strong parts of v1 (the permutation-sensitive curvature statistic,
the sliced-Wasserstein distributional operator, the strictly-proper Brier proof,
and the single matched-endpoint/divergent-curvature kill construction) and
applies every one of the reviewer's corrections.

---

## Changelog vs v1 (each GPT-5.5 finding → what changed)

| GPT-5.5 finding (v1) | Fix applied in v2 | Where |
| --- | --- | --- |
| **F3 — theory overclaims `mean-pool ≡ endpoint`.** Endpoint = start/end token *positions*; a full-trajectory mean is a *different statistic*. | The formal claim is **reframed** to **endpoint-RESIDUALIZED trajectory information**: after regressing out the endpoint signature `φ_end`, the *residual* trajectory features still predict retention-adjusted gain. Tested by **partial correlation / residualized regression at matched budget**. §1.2 now proves only "first-moment temporal pooling is insufficient," never "endpoint-equivalent." | §1.2, §3.1–§3.2 |
| **F1 — `TrajectorySignature` stores only IDs/counts/hash; method is unauditable.** | Define **`TrajectorySignatureV2`** (versioned, `schema_version="2.0.0"`) that **persists the estimands themselves**: `D_l`, `kappa_l`, projection seeds, slice masks, selection/alignment scores, drift estimates `r̂,f̂`, calibrator provenance, layer-router outputs `m(x)` and `endpoint_signature` `φ_end`. The record stores **T(x)**, not just a hash; the hash becomes an integrity check *over* the stored estimands. | §2.1, §5 |
| **F2 — decisive NAIT/endpoint baseline still unimplemented.** | The **endpoint-neuron (NAIT) baseline is Deliverable #1**, written and unit-tested **before** any proposed-method operator code. It is the residualization basis `φ_end` *and* the decisive comparator. | §2.0a, §5 (Deliverable #1) |
| **F5 — submodular guarantee unsupported (`J(S)` subtracts undefined `Redund(S)`).** | Replaced with a **genuine monotone-submodular facility-location / coverage objective** `F(S)` over the trajectory-feature space. `F` is monotone + submodular ⇒ greedy retains the **(1−1/e)** guarantee, *proven* (§2.3). The old `J(S)−γ·Redund(S)` form is dropped. | §2.3 |
| **F6 — per-example LoRA gating breaks matched-budget fairness.** | Layer routing is demoted to a **validated ablation** with an explicit **capacity-matching contract**: active params, update opportunities, FLOPs, optimizer-state slots, and effective rank are matched to the uniform-LoRA baseline (capacity-conserving rank reallocation, §2.2). Main claim does **not** depend on routing. | §2.2, §4.5 |
| **F7 — Brier proof correct but doesn't show the proxy predicts eval drift.** | Brier propriety proof retained (it is correct), but the factuality proxy is **demoted to a validated ablation**: a **held-out proxy-vs-eval correlation + calibration diagnostic** (Spearman/Pearson + reliability curve vs TruthfulQA/FActScore) is a **precondition gate**, not an assumption. `λ_f` is set to 0 unless the diagnostic passes. | §2.4, §4.4 |
| **F8 — cost gate not credible (omits ablations, Fisher, SW2, routing, calibration).** | **Re-budgeted realistically**: the cost gate now itemizes causal-ablation measurement, Fisher estimation, SW2 projections, routing, and calibration overhead, separates **one-off validation cost** (amortized, outside the per-pool multiplier) from **per-example selection cost** (inside the multiplier), and states the resolution-sweep fallback that keeps the *per-pool* multiplier ≤ 2.0×. | §2.5, §4.6, §6 |
| **F4 — "depends on all moments" too strong for finite-sample/finite-projection SW2.** | Softened to "depends on **more than the first moment** (finite-sample, finite-`K` estimate)"; the *only* clean falsifier remains the matched-endpoint/divergent-curvature construction, now framed as the **residual** test. | §2.1(i), §3.2, §6 |

No gate threshold is changed. v2 changes the *claim framing, the persisted
record, the objective's guarantee, the fairness contract, the validation status
of two components, and the cost accounting* — not the evidence status (still
design-only) and not the numeric gates.

---

## 1. Defect being fixed

### 1.1 The audited critical defect (precise statement)

The base paper — **Neuron-Aware Data Selection (NAIT)** — scores an
instruction example `x` for selection using only the activations at the
**START and END tokens** of the sequence. Concretely, let a transformer have
`L` layers and hidden width `d`. For a token position `p` and layer `ℓ`, write
the post-block residual-stream activation as `h_ℓ(p) ∈ R^d`. NAIT compresses an
example to an **endpoint feature**

```
φ_end(x) = concat_{ℓ ∈ A} [ h_ℓ(p_start) , h_ℓ(p_end) ]            (Eq. 1)
```

over an anchor layer set `A`, and selects examples by similarity of `φ_end(x)`
to a target-capability anchor `μ_T` (e.g. cosine / class-mean neuron overlap).

The seed idea for neurotrace-it is to ask whether the **full reasoning
trajectory** — the activation tensor over (layers × decoding-steps × tokens) —
carries selection-relevant information **beyond what `φ_end` already captures**,
plus a **layer-wise** adapt/freeze rule (treated as an ablation).

The audited defect being fixed is that **as scaffolded, the trajectory novelty
was not yet a falsifiable contribution AND was over-claimed**. Two operators
were named but UNSPECIFIED, one baseline was not reproduced, and the central
formal claim was **stated too strongly** (it asserted the trajectory mean equals
the endpoint feature, which is false — see §1.2):

1. **Trajectory operator under-specified, and the v1 claim over-stated.** "Use
   the trajectory as a distribution" needs a defined summary map and distance
   *and* an honest statement of what it buys over `φ_end`. v2 specifies the
   operator (§2.1) and states the gain as a **residual** over endpoints (§3).
2. **Layer-routing rule unspecified and fairness-threatening.** "Different
   layers serve different functions, so adapt/freeze layer-wise" had no measured
   notion of layer function and, worse, per-example masking silently changes the
   trainable-parameter budget. v2 grounds it (§2.2) **and** demotes it to a
   capacity-matched ablation, so the main claim never leans on it.
3. **Drift metrics unspecified; factuality proxy unvalidated.** v2 gives both an
   estimator (§2.4) **and** a held-out proxy-vs-eval correlation/calibration
   precondition; the factuality term is an ablation, not load-bearing.
4. **NAIT endpoint baseline not reproduced.** Eq. 1 is registered
   (`endpoint_neuron_selection`) but no faithful reimplementation existed. v2
   makes it **Deliverable #1**, built and tested before any proposed-method code,
   because it is both the residualization basis and the decisive comparator.

### 1.2 What first-moment pooling loses (corrected — NOT "endpoint-equivalent")

> **Correction (GPT-5.5 F3/F4).** v1 wrote "mean-pool ≡ endpoint." That is
> false. The endpoint feature `φ_end` reads the **start/end token positions** —
> a specific pair of positions. A per-layer trajectory **mean** `ψ_1` averages
> over *all* steps/tokens. These are **different statistics**: neither is a
> function of the other in general. The defensible statement is only that
> **first-moment temporal pooling is insufficient** — it is permutation- and
> shape-invariant, so it discards exactly the temporal/distributional structure
> v2 exploits. We never claim mean-pool equals endpoint.

Let `H(x) = { h_ℓ,s,t }` be the trajectory set over layers `ℓ ∈ [L]`, decoding
steps `s ∈ [S]`, and tokens `t ∈ [T_s]`. Define the per-layer **first-moment
summary** `ψ_1(x)_ℓ = (1/N_ℓ) Σ_{s,t} h_ℓ,s,t`. Then for any score **linear in
the summary**, `score(x) = ⟨w, ψ_1(x)⟩`,

```
score(x) = Σ_ℓ ⟨w_ℓ , mean_{s,t} h_ℓ,s,t ⟩
         = mean_{s,t} ⟨w, h_·,s,t⟩.                                  (Eq. 2)
```

Two consequences, both **provable and modest** (this is all §1.2 claims):

- **(a) Permutation invariance.** `ψ_1` is invariant to any reordering of the
  decoding steps `s`. Hence any first-moment selector cannot distinguish a
  direct-answer path from a detour-then-correct path with the same multiset of
  activations. A statistic that *changes* under reordering (curvature `κ`, §2.1)
  is, by definition, **not a function of `ψ_1`**.
- **(b) Moment blindness.** `ψ_1` ignores spread/shape; a distributional
  distance (SW2, §2.1) responds to higher moments that `ψ_1` cannot see.

What §1.2 does **not** claim: it does not claim `ψ_1` and `φ_end` are equal, and
it does not claim SW2/`κ` are *guaranteed* to beat `φ_end` on real data — that is
an **empirical** question settled by the residual test (§3) and the kill-gate
(§4.6). The collapse to escape is not "endpoint" but "first-moment pooling"; the
contribution must live in the part of `T(x)` that survives **residualizing out
`φ_end`** (§3.1).

---

## 2. New method (equation-level)

**Name:** **LATTICE** — *Layer-Aware Trajectory-distribuTIon seleCtion with
rEtention-adjusted gain.* (Working name; the contribution, not the acronym,
matters.)

LATTICE has, in v2, a **clear dependency order** that mirrors the deliverable
order (§5): first the **endpoint baseline** `φ_end` (the residualization basis
and decisive comparator), then (A) a **trajectory operator**, (B) a
**capacity-matched layer router** (ablation), (C) a **monotone-submodular
retention-adjusted selection objective** (proven (1−1/e)), and a strictly-proper
calibrator (§2.4) whose factuality use is **gated by validation**.

### 2.0 Notation

| Symbol | Meaning |
| --- | --- |
| `x` | candidate instruction example (prompt + target reasoning trace) |
| `L, S_x, T_{x,s}` | #layers, #decoding steps for `x`, #tokens at step `s` |
| `h_ℓ,s,t ∈ R^d` | residual-stream activation, layer `ℓ`, step `s`, token `t` |
| `A ⊆ [L]` | anchor/probe layer set (fixed on validation) |
| `φ_end(x)` | endpoint signature, Eq. 1 (computed by Deliverable #1) |
| `μ_T, Σ_T` | target-capability anchor mean / cov of per-layer activations |
| `T(x)` | trajectory signature `({D_ℓ},{κ_ℓ})`, Eq. 6 |
| `r(·|φ_end)` | residual operator: regress `·` on `φ_end`, keep residual |
| `m ∈ {0,1}^L` | per-example layer mask (1 = route, 0 = freeze) — ablation only |
| `B` | selected-data budget (number of examples), shared by all methods |

### 2.0a Endpoint-neuron baseline (Deliverable #1 — built FIRST)

Per GPT-5.5 F2 and the recommendation, the endpoint baseline is implemented and
tested **before** any proposed-method operator. It is faithful NAIT (Eq. 1):

```
φ_end(x) = concat_{ℓ∈A} [ h_ℓ(p_start), h_ℓ(p_end) ];
score_end(x) = sim( φ_end(x), μ_T )      (cosine or class-mean neuron overlap)
```

`φ_end` plays **two** roles: (i) the **decisive matched-budget comparator** the
whole paper turns on; (ii) the **residualization basis** for the v2 formal
claim (§3.1). No proposed operator may be measured for "gain over endpoints"
until `score_end` and `φ_end` exist and pass their unit tests
(`tests/test_endpoint_baseline.py`, §5).

### 2.1 (A) Trajectory operator — the layer-resolved trajectory signature

We summarize `x` per layer by a **trajectory descriptor** that captures what
first-moment pooling throws away (§1.2): (i) the *distribution* of activations
across steps/tokens (more than its mean), (ii) the *temporal geometry*
(curvature/speed over decoding steps).

**(i) Per-layer empirical activation measure → sliced-Wasserstein distance.**
For layer `ℓ`, collect the cloud `P_ℓ(x) = { h_ℓ,s,t }` as an empirical
distribution on `R^d`; compare to the target cloud `Q_ℓ^T` (held-out
target-capability exemplars) with the **sliced-Wasserstein-2** distance:

```
SW2²(P,Q) = E_{u~Unif(S^{d-1})} [ W2²( u#P , u#Q ) ]                 (Eq. 3)
D_ℓ(x)    = SW2²( P_ℓ(x) , Q_ℓ^T ).                                  (Eq. 4)
```

estimated with `K` random projections (seeds **persisted** in the signature,
§2.6); each 1-D `W2²` is a sort-and-integrate, `O((nP+nQ) log n)`.

> **Honest scope (GPT-5.5 F4).** `D_ℓ` depends on **more than the first moment**
> (in the finite-sample, finite-`K` estimate); it is *not* a function of `ψ_1`
> alone. We do **not** claim it "depends on all moments" as a safe finite-sample
> statement, and we do **not** claim it must beat `φ_end` — empirically it could
> track endpoint/mean similarity (§6). Its endpoint-beyond value is settled only
> by the **residual** test (§3.1).

**(ii) Temporal curvature (uses ordering — Eq. 3 does not).** Let the per-step
mean path be `g_ℓ,s = mean_t h_ℓ,s,t`. With discrete velocity/acceleration
`v_ℓ,s = g_ℓ,s+1 − g_ℓ,s`, `a_ℓ,s = g_ℓ,s+1 − 2 g_ℓ,s + g_ℓ,s−1`, define the
**trajectory-curvature statistic** (normalized bending energy):

```
κ_ℓ(x) = (1/(S_x−2)) Σ_{s=2}^{S_x−1} ‖a_ℓ,s‖ / ( ‖v_ℓ,s‖·‖v_ℓ,s−1‖ + ε ).   (Eq. 5)
```

`κ_ℓ` is **permutation-sensitive** (reorder steps ⇒ `κ_ℓ` changes), whereas
`ψ_1` is permutation-invariant; therefore `κ_ℓ` is provably **not a function of
`ψ_1`** (§1.2a). This is the statistic the falsifiable prediction (§3.2) is
built on.

**(iii) Cross-layer signature.** Stack into the **trajectory signature**

```
T(x) = ( { D_ℓ(x) }_{ℓ∈A} , { κ_ℓ(x) }_{ℓ∈A} ) ∈ R^{2|A|}.          (Eq. 6)
```

The **endpoint-residualized** target-alignment read-out (the heart of the v2
claim) is defined in §3.1; the raw read-out (validation-fit, never on
test/drift) is

```
ρ_T(x) = − Σ_{ℓ∈A} α_ℓ D_ℓ(x) − Σ_{ℓ∈A} β_ℓ | κ_ℓ(x) − κ_ℓ^T |,     (Eq. 7)
```

with `κ_ℓ^T` the target exemplars' mean curvature, `(α,β) ≥ 0` fit on validation
only.

### 2.2 (B) Capacity-matched layer router (ABLATION, not load-bearing)

> **Demotion (GPT-5.5 F6).** Per-example LoRA layer gating changes the trainable
> footprint, threatening the matched-budget contract. In v2, layer routing is a
> **validated ablation**: the main result (§3) stands or falls on the
> *selection* claim with **uniform** LoRA. The router is reported only if it
> helps at **matched capacity** in ≥2 capability families (gate G4).

**Layer-function profile (measured once, validation only).** For capability `c`
and layer `ℓ`, a **causal/probing importance** `I_{c,ℓ} ∈ [0,1]` via mean-ablation
causal drop —

```
I_{c,ℓ} = ( Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ) ) / Acc_c(model)     (Eq. 8)
```

— or linear-probe accuracy (robustness variant). Computed once, frozen, hashed
into provenance, validation-only.

**Per-example routing rule.** With `w_ℓ(x) = D̃_ℓ(x)` (Eq. 4, min-max normalized
over `A`),

```
m_ℓ(x) = 1[  I_{c(x),ℓ} · w_ℓ(x)  ≥  τ  ]   for ℓ ∈ A,  else 0.       (Eq. 9)
```

**Capacity-matching contract (the fairness fix).** When the router activates
LoRA on a *subset* of layers for example `x`, the matched-budget comparison must
hold **active params, update opportunities, FLOPs, optimizer-state slots, and
effective capacity** equal to the uniform-LoRA baseline. v2 enforces this by
**capacity-conserving rank reallocation**: the per-example *total* LoRA rank
budget `R_tot = Σ_ℓ r_ℓ` is **fixed** and equal to the uniform baseline's
`|A|·r_unif`; routing redistributes that fixed budget across the routed layers
(`r_ℓ = R_tot / |{ℓ: m_ℓ(x)=1}|`, clipped) rather than adding parameters. Thus:

- **active params** per example = `R_tot·(...)` constant across methods;
- **optimizer-state slots** (Adam moments) scale with active params ⇒ matched;
- **FLOPs / update opportunities**: report both *per-step* and *total* and
  require equality within the fairness tolerance (`docs/baseline_contract.md`);
- **effective capacity**: reported via total rank, identical by construction.

If exact equality is infeasible for a variant, that variant is reported as
**capacity-unmatched diagnostic**, never as a matched-budget claim. Setting
`m_ℓ ≡ 1` (uniform) recovers `layer_selective_no_trajectory` / standard
uniform-LoRA — the registered ablation poles.

### 2.3 (C) Retention-adjusted selection objective — monotone submodular, proven (1−1/e)

> **Replacement (GPT-5.5 F5).** v1's `J(S) = Σ[ĝ−λ_r r̂−λ_f f̂] − γ·Redund(S)`
> subtracted an **undefined** `Redund(S)` and then asserted greedy `(1−1/e)`.
> That guarantee does not follow. v2 replaces it with a **genuine monotone
> non-negative submodular facility-location / coverage objective** and proves the
> guarantee.

**Per-example value (modular part).** Let `ĝ(x) = ρ̃_T(x)` be the
endpoint-residualized read-out (§3.1, centered to be non-negative via a constant
shift), and define the **per-example utility**

```
u(x) = max( 0,  ĝ(x) − λ_r·r̂(x) − λ_f·f̂(x) ),                      (Eq. 10a)
```

reusing `metrics.retention_adjusted_gain` as the kernel (`ĝ − λ_r r̂`) extended
by the factuality term (`λ_f = 0` unless the §2.4 diagnostic passes).

**Coverage / facility-location objective (submodular part).** Over the
trajectory-feature space, define a similarity `σ(x, e) ∈ [0,1]` between a
candidate `x` and a *ground element* `e` (a target-capability "concept" exemplar;
`E` = the held-out target exemplar set). The **facility-location coverage**

```
C(S) = Σ_{e∈E}  w_e · max_{x∈S} σ(x, e).                             (Eq. 10b)
```

`C` rewards a subset that **covers** the diverse target concepts (and so
penalizes near-duplicate on-path picks *implicitly*, the role v1's `Redund`
tried to play — but here as a **principled** monotone submodular term). Combine
into the **selection objective**

```
F(S) = Σ_{x∈S} u(x)  +  μ · C(S),       μ ≥ 0.                       (Eq. 11)
```

**Proposition (monotone submodularity ⇒ greedy is (1−1/e)).**
`F` is monotone non-decreasing and submodular on `2^X`; therefore the greedy
algorithm that adds, at each step, `argmax_x [F(S∪{x}) − F(S)]` until `|S| = B`
returns `S_greedy` with `F(S_greedy) ≥ (1 − 1/e)·F(S*)`, where `S*` is the
cardinality-`B` optimum (Nemhauser–Wolsey–Fisher 1978).

**Proof.**
(1) *Modular term is submodular & monotone.* `Σ_{x∈S} u(x)` with `u ≥ 0`
(Eq. 10a) is modular (hence submodular) and monotone non-decreasing.
(2) *Facility location is monotone submodular.* Fix `e`. The function
`S ↦ max_{x∈S} σ(x,e)` is monotone (adding elements cannot lower a max) and
submodular: for `S ⊆ T` and `x ∉ T`,
`max_{S∪{x}} σ − max_{S} σ ≥ max_{T∪{x}} σ − max_{T} σ`
because the marginal gain `max(0, σ(x,e) − max_{S}σ)` is non-increasing in the
running max, which is itself non-decreasing in the set. A non-negative weighted
sum (`w_e ≥ 0`) of monotone submodular functions is monotone submodular, so `C`
is. (3) *Closure.* A non-negative combination (`1·Σu + μ·C`, `μ≥0`) of monotone
submodular functions is monotone submodular. Hence `F` is monotone submodular,
the NWF theorem applies, and greedy is `(1−1/e)`-optimal. ∎

This guarantee is now **earned**, not asserted; greedy is `O(B·|pool|·|E|)` and
stays inside the re-budgeted cost gate (§2.5). If a future variant needs a term
that is *not* submodular, the guarantee is dropped and greedy is reported as a
**heuristic** (the reviewer's permitted alternative) — but the recommended,
shipped form (Eq. 11) keeps the proof.

### 2.4 Drift estimators + the strictly-proper calibrator (factuality = validated ablation)

**Retention drift estimator `r̂(x)`** — a Fisher-style stability penalty. With
`F̄_ℓ` the diagonal Fisher of the base model on a held-out *general* set,

```
r̂(x) = Σ_{ℓ∈A} F̄_ℓ · w_ℓ(x) · m_ℓ(x).                              (Eq. 12)
```

High `r̂` ⇒ the example writes to layers the base model relies on for general
ability ⇒ likely retention loss. (With uniform LoRA, `m_ℓ≡1`.)

**Factuality drift estimator `f̂(x)`** — a claim-support margin from a calibrated
proxy. For atomic factual claims in `x`'s target, `f̂(x) = mean_a 1[ q_a < c* ]`
where `q_a = σ(z_a)` is the calibrator output and `c*` a calibrated threshold.

**Propriety PROOF (the calibrator-propriety requirement — retained; it is correct).**
Score the calibrator with the **Brier (quadratic) rule**

```
ℓ_Brier(q_a, y_a) = ( q_a − y_a )².                                  (Eq. 13)
```

**Proposition (propriety).** Brier is strictly proper: for the true conditional
`p = Pr(y=1|z)`, the population risk is minimized **uniquely** at `q ≡ p`.

**Proof.** For any reported `q ∈ [0,1]` and `y ∼ Bernoulli(p)`,
```
E[(q−y)²] = p(q−1)² + (1−p)q² = q² − 2pq + p = (q − p)² + p(1 − p).   (Eq. 14)
```
`p(1−p)` is independent of `q`; `(q−p)² ≥ 0` with equality iff `q = p`. So the
population risk is uniquely minimized at `q = p`; taking expectation over
contexts preserves the strict inequality. ∎ The same decomposition yields the
**reliability–resolution** split, a checkable calibration diagnostic.

> **Demotion + precondition (GPT-5.5 F7).** Propriety proves the calibrator is
> *well-calibrated*; it does **not** prove the proxy predicts TruthfulQA/FActScore
> **drift**. Therefore the factuality term is a **validated ablation**, gated by a
> **held-out proxy-vs-eval diagnostic** (§4.4): on a held-out slice, require
> (a) Spearman/Pearson correlation between `f̂` and the eval factuality-drift
> signal above a pre-registered threshold, and (b) a passing reliability curve.
> **If the diagnostic fails, `λ_f := 0`** and no safety/hallucination reduction
> is claimed from `f̂`. The proof guarantees calibration, not proxy validity;
> proxy validity is an *empirical precondition*, never an assumption.

### 2.5 Cost model (re-budgeted realistically — GPT-5.5 F8)

v1's compute plan put endpoint selection at 2h and trajectory selection at 4h —
**exactly** the 2× limit — and omitted causal ablations, Fisher, SW2 projections,
routing, and calibration. v2 separates two cost classes:

- **One-off validation cost (amortized; OUTSIDE the per-pool multiplier):**
  causal layer importance `I_{c,ℓ}` (Eq. 8, `|A|·|capabilities|` mean-ablation
  passes), diagonal Fisher `F̄_ℓ` (Eq. 12), calibrator fit + proxy-vs-eval
  diagnostic (§4.4), `(α,β,μ,τ,λ)` fitting. These are paid **once** per
  (model, validation split), not per candidate pool; they are reported as a
  separate line and **must not** be hidden inside the 2× gate, but also must not
  be double-counted against per-pool selection.
- **Per-example selection cost (INSIDE the `passes_cost_gate` multiplier):**
  per-`x` SW2 (`K` projections × sort-integrate), `κ` (Eq. 5), residualization
  `r(·|φ_end)`, drift estimators, greedy coverage update. The multiplier is
  `cost_select / cost_endpoint_select` over the **same pool**.

**Keeping the per-pool multiplier ≤ 2.0×.** The dominant per-example term is SW2;
its cost is linear in `K` and in `n log n`. The **trajectory-resolution sweep**
(`docs/motivation_ablation_hparam_plan.md`, "cost vs trajectory resolution")
tunes `K`, the probe-layer count `|A|`, and step/token subsampling so the
*per-pool* multiplier stays ≤ 2.0×; if it cannot, the method is reported as a
**high-cost analysis method** per the gate's `failure_action:
require_efficiency_ablation` — not as a deployment method. The validation/amortized
line is reported transparently in the compute table (§4.6) so reviewers see the
true total, with the honest caveat that the per-pool gate covers only the
amortizable-away part.

### 2.6 Versioned, auditable signature record (GPT-5.5 F1)

> **Fix.** v1 falsely said `TrajectorySignature` "already stores exactly `T(x)`."
> It stores only `example_id, layer_ids, step_count, token_count,
> trajectory_hash` — IDs/counts/hash, **not the estimands**. v2 defines a
> **versioned** record that **persists the estimands themselves**, so every
> number that steers selection is auditable; the hash becomes an *integrity
> check over* the stored values, not a stand-in for them.

`TrajectorySignatureV2` (design; added in `src/neurotrace_it/schemas_v2.py`,
§5 — **not** a modification of `schemas.py`) persists, per example:

```
schema_version        : "2.0.0"
example_id            : str
layer_ids (A)         : tuple[int,...]
step_count, token_count
endpoint_signature    : φ_end(x)            # Eq.1 — the residualization basis
D                     : { ℓ: D_ℓ(x) }       # Eq.4   (the estimand, stored)
kappa                 : { ℓ: κ_ℓ(x) }       # Eq.5   (the estimand, stored)
projection_seeds      : tuple[int,...]       # SW2 reproducibility (K seeds)
slice_masks           : { ℓ: mask }          # step/token subsample masks
alignment_scores      : { rho_T, rho_residual }   # Eq.7 / §3.1
selection_scores      : { u(x), marginal_gain }   # Eq.10a / greedy
drift_estimates       : { r_hat, f_hat }     # Eq.12 / §2.4
calibrator_provenance : { rule:"brier", c*, reliability_hash, diag_pass:bool }
layer_router_outputs  : { m(x), I_profile_hash, R_tot, r_per_layer }
trajectory_hash       : H(all of the above)  # integrity check, not a substitute
```

`validate_selection_manifest` is **extended** (in the new module, additively) to
require, for V2 records: non-empty `D`/`kappa` over `layer_ids`, present
`endpoint_signature`, present `projection_seeds`, and a `trajectory_hash` that
**recomputes** from the stored estimands. `server_authorized` stays `false`;
`endpoint_neuron_selection` stays a required baseline id. The legacy V1 record
remains valid (back-compatible) but is flagged `audit_incomplete`.

### 2.7 Algorithm box

```
Algorithm LATTICE-Select  (design-only; no server run)
Inputs : pool X (budget B), probe layers A, target exemplars Q^T and concept set E,
         general/Fisher set G, layer-function profile I_{c,ℓ},
         weights (α,β,λ_r,λ_f,μ,τ) fit on VALIDATION only, K projections + seeds
Output : selected set S (|S|=B), [ablation] masks m(x),
         SelectionManifest with TrajectorySignatureV2 records (server_authorized=False)

# Phase -1 : ENDPOINT BASELINE FIRST (Deliverable #1)
  for x in X: φ_end(x) ← Eq.1 ;  score_end(x) ← sim(φ_end, μ_T)   # decisive comparator + residual basis

# Phase 0 (once, validation, AMORTIZED cost): measure layer function + Fisher + calibrator
  I_{c,ℓ} ← Eq.8 ;  F̄_ℓ ← diag Fisher on G ;  fit Brier calibrator ; run proxy-vs-eval diag → set λ_f

# Phase 1 : per-example trajectory signatures (PER-POOL cost)
  for x in X:
     D_ℓ(x) ← SW2²(P_ℓ,Q_ℓ^T)         # Eq.4   (store seeds, value)
     κ_ℓ(x) ← bending energy            # Eq.5   (store value)
     T(x)   ← ({D_ℓ},{κ_ℓ})            # Eq.6
     ρ̃_T(x) ← residualize ρ_T on φ_end # §3.1   (the v2 claim's read-out)
     [ablation] w_ℓ(x), m_ℓ(x)         # Eq.9 + capacity-matching §2.2

# Phase 2 : drift + monotone-submodular selection (PER-POOL cost)
  for x in X: r̂(x) ← Eq.12 ; f̂(x) ← §2.4 ; u(x) ← Eq.10a
  S ← greedy argmax of F(S)=Σ u(x) + μ·C(S)      # Eq.11 ; (1−1/e) by §2.3 proof

# Phase 3 : emit auditable contracts (LOCAL, additive, uncommitted)
  records ← { TrajectorySignatureV2(x) : x∈S }   # §2.6 — estimands persisted
  manifest ← SelectionManifest(project="neurotrace-it",
              baseline_ids⊇{"endpoint_neuron_selection"},
              signatures=records, selected_example_ids=S, server_authorized=False)
  assert validate_selection_manifest_v2(manifest) == []
  return S, m, manifest
# Training/eval with uniform LoRA (main) or matched-capacity masks (ablation) is a SEPARATE server step — NOT run here.
```

---

## 3. Why it is NOT stitching + the falsifiable prediction

### 3.1 The single crisp scientific contribution (RESIDUALIZED — GPT-5.5 F3)

> **There exists a temporal/distributional component of the reasoning trajectory
> — captured by curvature `κ` and distributional alignment `D` — that, AFTER
> regressing out the endpoint signature `φ_end`, still carries information
> predictive of retention-adjusted fine-tuning utility, at matched data,
> parameter, and compute budget.**

This is the **endpoint-residualized** claim. Formally, let `φ_end(x)` be the
endpoint feature (Deliverable #1) and `Y` the retention-adjusted fine-tuning
gain. Define the **residualized trajectory read-out**

```
ρ̃_T(x) = T(x) − Proj_{span(φ_end)} T(x)        (residual of T(x) on φ_end)
```

and the **residual selection score** as the projection of `Y`'s predictor onto
`ρ̃_T`. The honest, falsifiable statement is:

> **partial correlation** `corr( ρ̃_T(x), Y | φ_end(x) ) ≠ 0`, equivalently the
> **residualized regression** of `Y` on `T(x)` after `φ_end` has a non-zero,
> significant coefficient at matched budget.

This is one claim built from **one quantity** `T(x) = ({D_ℓ},{κ_ℓ})`: the
selection score (§2.1), the (ablation) router (§2.2), the retention penalty
(§2.4), and the coverage term (§2.3) are all functions of the same `D_ℓ`/`κ_ℓ`.
Crucially, the gain is measured **as a residual over endpoints**, so it cannot be
an artifact of `T(x)` merely re-encoding `φ_end` — that part is regressed out by
construction. Remove the trajectory signature and the residual claim is empty;
that shared origin (and the residualization) is what makes it a method, not a
stitch.

### 3.2 The falsifiable prediction (matched-endpoint / divergent-curvature)

> **Prediction P (residual form):** Construct (or mine) example pairs `(x, x′)`
> with **near-identical endpoint signatures** `φ_end(x) ≈ φ_end(x′)` (so endpoint
> *and* any first-moment selector score them ≈ equally), but **different
> trajectory curvature** `κ(x) ≠ κ(x′)` — e.g. a direct-answer solution vs a
> detour-then-correct solution reaching the same final state (equal endpoints by
> construction, different `a_ℓ,s`). LATTICE predicts they yield **measurably
> different retention-adjusted fine-tuning outcomes** (differing by > the 0.02
> trajectory–endpoint margin). Because `φ_end` is matched, any outcome difference
> is **purely residual** trajectory information.

- **If P holds:** endpoint-residualized trajectory information is real ⇒ the core
  novelty stands; the 0.02 margin gate is cleared by a mechanism endpoints cannot
  see.
- **If P fails** (matched-endpoint pairs train identically): the residual carries
  nothing ⇒ **stop main novelty claim** (`failure_action:
  stop_main_novelty_claim`), downgrade to diagnostic.

This is the sharp `trajectory ⊥ endpoint` test the brief demands, now stated as a
**residual** so it is immune to the "T(x) just re-encodes φ_end" objection.

---

## 4. Experiment design to confirm/kill the claim (design only — NO runs)

`server.authorized` stays **false**. Everything below is a plan; no command is
executed.

### 4.1 Datasets (reuse `docs/data_and_evaluation_plan.md`)

- **Target pools (train candidates):** math reasoning IT, code IT, multi-hop QA
  IT (frozen split hash + contamination audit, per `first_gate.yaml`).
- **Held-out target evals:** math (EM), executable code (pass rate), multi-hop QA
  with distractors.
- **Retention eval:** MMLU-style aggregate + general IF
  (`aggregate_retention_score`, `semantic_drift`).
- **Hallucination/factuality eval:** TruthfulQA + FActScore-style atomic-claim
  factuality (`factuality_error_rate`, `harmful_overconfidence_rate`) — also the
  ground truth for the §4.4 proxy diagnostic.

### 4.2 Base models (with sizes — non-toy)

A **~7–8B** instruction-tunable model (primary) and a **~1.5–3B** model
(secondary, scale-robustness). Exact IDs pending user approval
(`first_gate_open_model_pending_user_approval`). LoRA only; rank/steps/tokens
**matched across all methods**. No tiny-toy-only or synthetic-only evidence.

### 4.3 Baselines (reuse `baseline_registry.yaml`, all 7 + base paper)

`random_subset`, `full_data_it` (upper bound), `quality_score_selection`,
`diversity_coreset`, `influence_gradient_selection`,
**`endpoint_neuron_selection` (NAIT, Deliverable #1, the decisive comparator and
residual basis)**, `layer_selective_no_trajectory`. **Ablations:** endpoint-only;
`D`-only (no `κ`); `κ`-only (no `D`); **router on/off at matched capacity**
(§2.2); `λ_r=0`; `λ_f=0` (default unless §4.4 passes); coverage `μ=0`.

### 4.4 Metrics, estimators, and the proxy-validity precondition

Target (EM/pass-rate), retention (`retention_adjusted_gain`), hallucination
drift, selection+training cost. The **primary statistic for the core claim** is
the **partial correlation / residualized-regression coefficient** of
retention-adjusted gain on `T(x)` given `φ_end(x)` (§3.1).

**Proxy-vs-eval diagnostic (precondition for any factuality claim, GPT-5.5 F7).**
On a held-out slice, compute Spearman & Pearson correlation between `f̂(x)` and
the eval factuality-drift signal (TruthfulQA/FActScore), plus the
reliability–resolution curve from Eq. 14. Pre-register thresholds; **only if both
pass** is `λ_f > 0` permitted; otherwise `λ_f := 0` and no factuality claim.

### 4.5 Matched-budget protocol (reuse `baseline_contract.md` + `pre_registration.md`)

Same candidate pool, budget `B`, base model + tokenizer, **same LoRA rank**
(`R_tot`, §2.2 capacity-matching), same steps/tokens, validation-only selection,
same evaluator, **shared seeds 0..19** (`configs/seeds/paper_20.txt`). Paired
tests on matched eval items; Holm correction across {target, retention,
hallucination, layer, cost}; effect sizes + 95% CIs
(`docs/statistical_analysis_plan.md`). The router ablation is run **only at
matched capacity**; any unmatched variant is diagnostic-labeled.

### 4.6 The kill-gate (decisive, pre-registered) + honest cost table

Two layers, thresholds **unchanged**:

1. **Mechanism kill (Prediction P, §3.2):** on matched-endpoint / divergent-`κ`
   pairs, if **residual** fine-tuning outcomes do not differ by > 0.02 adjusted,
   the endpoint-residualized claim is false ⇒ **stop main novelty claim**.
2. **Aggregate kill (existing G1–G5):** trajectory_adjusted − endpoint_adjusted
   < 0.02 (margin), or target/adjusted relative gain < 0.03, or retention drift
   disadvantage > 0.01, or hallucination drift > 0.01, or **per-pool** cost
   multiplier > 2.0× ⇒ downgrade/forbid per each gate's `failure_action`. Router
   must help in ≥2 capability families **at matched capacity** (G4) or be
   appendix-only. Paper tier needs ≥20 seeds (G5).

**Honest cost table (re-budgeted, §2.5).** The compute plan must report two
lines: (i) **amortized validation** (causal `I`, Fisher, calibrator + diagnostic,
hparam fit) — paid once per (model, split); (ii) **per-pool selection** (SW2,
`κ`, residualization, drift, greedy) — the only line inside the 2× multiplier.
The old "2h vs 4h = exactly 2×" framing is withdrawn; the per-pool figure is set
by the resolution sweep, and the amortized line is shown separately so the true
total is visible.

No gate threshold is changed by this redesign; LATTICE supplies the *operators
and the honest accounting* the gates were always meant to test.

---

## 5. What changes in code vs the current scaffold

All changes are **additive and uncommitted**; nothing existing is modified or
deleted. The scaffold's `schemas.py`, `metrics.py`, `baseline_registry.yaml`, and
gate configs are reused as-is. **Deliverable order is enforced: #1 endpoint
baseline before any proposed-method operator.**

**Reused unchanged**
- `src/neurotrace_it/metrics.py` — `retention_adjusted_gain` is the kernel of
  Eq. 10a; `passes_drift_gate` / `passes_cost_gate` are the §4.6 gates.
- `src/neurotrace_it/schemas.py` — left **unmodified**; the V1
  `TrajectorySignature`/`SelectionManifest` remain valid (back-compatible).
- `configs/baselines/baseline_registry.yaml`, `configs/experiments/*.yaml`,
  `configs/seeds/paper_20.txt`, `schemas/selection_manifest.schema.json` — reused
  verbatim.

**To ADD later (not in this commit; design names only; in dependency order)**
- **Deliverable #1 — `src/neurotrace_it/endpoint_baseline.py`** — `endpoint_signature`
  (Eq. 1), `endpoint_score` (NAIT similarity). **Built and tested first.**
  `tests/test_endpoint_baseline.py` — faithful-NAIT contract + similarity tests.
- `src/neurotrace_it/schemas_v2.py` — **`TrajectorySignatureV2`** (§2.6, persists
  estimands) + `validate_selection_manifest_v2` (additive; does not touch
  `schemas.py`).
- `src/neurotrace_it/trajectory_ops.py` — `sliced_wasserstein2` (Eq. 3–4, returns
  value **and** seeds), `trajectory_curvature` (Eq. 5), `trajectory_signature` →
  `TrajectorySignatureV2`.
- `src/neurotrace_it/residualize.py` — `residualize_on_endpoint` (§3.1),
  `partial_correlation` / `residualized_regression` (the primary statistic).
- `src/neurotrace_it/layer_function.py` — `causal_layer_importance` (Eq. 8),
  `route_layers` (Eq. 9) **+ `capacity_match` (§2.2 rank reallocation)**.
- `src/neurotrace_it/drift.py` — `fisher_retention_penalty` (Eq. 12),
  `BrierCalibrator` (§2.4) + `factuality_drift` + `proxy_eval_diagnostic` (§4.4).
- `src/neurotrace_it/select.py` — `lattice_select` (greedy monotone-submodular
  `F(S)`, Eq. 11) emitting a V2 `SelectionManifest`.
- `configs/experiments/lattice_v2.yaml` — hparams `(K,α,β,λ_r,λ_f,μ,τ,R_tot)`,
  probe-layer set `A`, trajectory-resolution sweep, **two-line cost budget
  (amortized vs per-pool)**; **`server.authorized: false`**.
- `tests/test_trajectory_ops.py` — (a) `κ` permutation-sensitive while mean-pool
  invariant (encodes §1.2a); (b) Brier propriety numeric check via Eq. 14
  (**formula evaluation, not evidence**); (c) `F(S)` submodularity unit check
  (diminishing-returns inequality on a tiny synthetic set — **formula
  evaluation, not evidence**); (d) `validate_selection_manifest_v2` round-trips
  the persisted estimands and recomputes `trajectory_hash`.

No server command, no training, no extraction is added or run.

---

## 6. Open risks + honest limitations

1. **Residual signal may be small or null.** The core claim is now a **residual**
   (`corr(ρ̃_T, Y | φ_end) ≠ 0`); if endpoints already explain the variance, the
   partial correlation is ~0 and the claim is killed by §3.2/§4.6. This is the
   intended falsifier, not a flaw — but the honest prior is that the residual may
   be small on naturally-mined pairs; the clean construction (§3.2) is the only
   crisp test.
2. **Curvature signal-to-noise.** `κ` (Eq. 5) is noisy for short traces
   (`S_x` small). Mitigation: resolution sweep; if `κ-only` ablation is at chance,
   fall back to the `D`-only residual. The curvature half may not survive.
3. **SW2 vs the collapse — empirical, not guaranteed (GPT-5.5 F4).** SW2 escapes
   *first-moment* collapse but may empirically track endpoint/mean similarity on
   real data; only the residual test settles it. We do **not** claim "all
   moments."
4. **Router fairness is fragile (GPT-5.5 F6).** Capacity-conserving rank
   reallocation (§2.2) matches params/optimizer-state/total-rank, but exact FLOPs
   and "effective capacity" equality are hard to guarantee; any residual mismatch
   forces a capacity-unmatched **diagnostic** label, never a matched claim. The
   main result deliberately uses **uniform** LoRA so it never depends on this.
5. **Factuality proxy validity is a precondition, not a result (GPT-5.5 F7).** The
   Brier proof gives calibration, not eval-drift prediction. If the §4.4
   proxy-vs-eval diagnostic fails, `λ_f := 0` and no hallucination-reduction claim
   is made.
6. **Cost honesty (GPT-5.5 F8).** Amortized validation cost (causal `I`, Fisher,
   calibrator, hparam fit) is real and reported separately; only per-pool
   selection cost is inside the 2× gate. If the per-pool multiplier exceeds 2.0×
   after the resolution sweep, the method is reported as **high-cost analysis**,
   not deployment.
7. **No empirical claim yet.** Per `docs/paper_claims_status.md`, every result
   above is **design**. "Beats NAIT" remains forbidden wording until the
   matched-budget endpoint reproduction (Deliverable #1) runs under approval. The
   only numeric checks here (Eq. 14 Brier identity; the submodularity
   diminishing-returns inequality) are **formula evaluations, not evidence**.

---

*Provenance:* additive redesign; adds only `docs/redesign/REDESIGN_v2.md`; no
file modified/deleted; no git commit; no experiment executed;
`server.authorized: false` preserved throughout.

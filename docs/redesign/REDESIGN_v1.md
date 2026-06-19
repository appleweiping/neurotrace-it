# REDESIGN v1 — neurotrace-it

Status: BUILD-NOW / RUN-LATER. Design + theory + code-readiness only.
No experiment is run here. `server.authorized: false` is preserved. No numbers
are fabricated. This document is additive and uncommitted.

This redesign reuses the existing governance verbatim: the seven-family
`configs/baselines/baseline_registry.yaml` (with the mandatory NAIT-style
`endpoint_neuron_selection`, arXiv 2603.13201), the numeric gates in
`configs/experiments/first_gate.yaml` / `formal_neurotrace_it.yaml`
(trajectory–endpoint margin 0.02, target/adjusted relative gain 0.03, retention
drift ≤ 0.01, hallucination drift ≤ 0.01, cost ≤ 2.0×), the metric API in
`src/neurotrace_it/metrics.py` (`retention_adjusted_gain`, `passes_drift_gate`,
`passes_cost_gate`), and the manifest contracts in
`src/neurotrace_it/schemas.py` and `schemas/selection_manifest.schema.json`.

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

The seed idea for neurotrace-it is to replace the **endpoint** `φ_end` with the
**full reasoning trajectory** — the activation tensor over
(layers × decoding-steps × tokens) — used as a *distribution*, plus a
**layer-wise** adapt/freeze rule.

The audited defect is that **as currently scaffolded, the trajectory novelty is
not yet a falsifiable contribution**. Three operators are named but
**UNSPECIFIED**, and one baseline is **not reproduced**:

1. **Trajectory operator unspecified.** "Use the trajectory as a distribution"
   has no defined summary map nor distance. Any naive instantiation
   (mean-pool over steps, then cosine) is *provably* a smoothed endpoint
   feature — see §1.2 — so the method **collapses to a more EXPENSIVE
   endpoint-activation similarity**, contributing nothing over Eq. 1 except
   cost. This is exactly the ARIS "NAIT-with-an-expensive-feature" kill
   argument (`docs/literature_boundary.md`, `docs/risks_and_blockers.md`).
2. **Layer-routing rule unspecified.** "Different layers serve different
   functions, so adapt/freeze layer-wise" has no *measured* notion of layer
   function and no rule mapping a datum to a layer mask. Hand-waving here makes
   the layer claim untestable; the governance already flags that it must be an
   ablation unless grounded (`docs/idea_synthesis.md` §Selected Direction).
3. **Drift metrics unspecified.** Retention and hallucination/factuality drift
   are listed as metric *names* (`aggregate_retention_score`, `semantic_drift`,
   `factuality_error_rate`, `harmful_overconfidence_rate`) but have **no
   estimator and no place in the selection objective**. Without that, the
   reliability claim cannot be earned and "retention gain via shorter/easier
   examples" (Kill Argument 3, `docs/research_brief.md`) cannot be ruled out.
4. **NAIT endpoint baseline not reproduced.** Eq. 1 is registered
   (`endpoint_neuron_selection`) but no faithful reimplementation exists, so the
   matched-budget head-to-head that decides the whole paper cannot be run.

### 1.2 The collapse, made formal (why this is the real risk)

Let `H(x) = { h_ℓ,s,t }` be the trajectory set over layers `ℓ ∈ [L]`, decoding
steps `s ∈ [S]`, and tokens `t ∈ [T_s]`. Define a generic **first-moment
summary** `ψ_1(x)_ℓ = (1/N_ℓ) Σ_{s,t} h_ℓ,s,t` (per-layer mean over the
trajectory). Then for any selection score that is **linear in the summary**,
`score(x) = ⟨w, ψ_1(x)⟩`, we have

```
score(x) = Σ_ℓ ⟨w_ℓ , mean_{s,t} h_ℓ,s,t ⟩
         = mean_{s,t} ⟨w, h_·,s,t⟩,                                  (Eq. 2)
```

i.e. the trajectory score is an **average of per-position endpoint-style
scores**. Two examples with identical per-layer means but different temporal
*shape* receive the *same* score. So **any first-moment / mean-pool trajectory
operator is, up to a constant cost overhead, an endpoint operator.** This is the
collapse the redesign must provably escape: the trajectory operator must depend
on a statistic that **mean-pooling destroys** (temporal ordering / curvature /
higher moments / inter-step geometry), and the falsifiable prediction (§3) must
exploit precisely that gap.

---

## 2. New method (equation-level)

**Name:** **LATTICE** — *Layer-Aware Trajectory-distribuTIon seleCtion with
rEtention-adjusted gain.* (Working name; the contribution, not the acronym, is
what matters.)

LATTICE has three coupled components, each fixing one §1 defect: (A) a
**trajectory operator** that is provably non-collapsing to endpoints; (B) a
**measured layer-function router**; (C) a **retention-adjusted selection
objective** that folds drift into the score and is *proper* (§2.4 proof).

### 2.0 Notation

| Symbol | Meaning |
| --- | --- |
| `x` | candidate instruction example (prompt + target reasoning trace) |
| `L, S_x, T_{x,s}` | #layers, #decoding steps for `x`, #tokens at step `s` |
| `h_ℓ,s,t ∈ R^d` | residual-stream activation, layer `ℓ`, step `s`, token `t` |
| `A ⊆ [L]` | anchor/probe layer set (fixed on validation) |
| `μ_T, Σ_T` | target-capability anchor mean / cov of per-layer activations |
| `θ_0` | frozen base-model parameters; `θ` post-LoRA params |
| `m ∈ {0,1}^L` | per-example layer mask (1 = adapt, 0 = freeze) |
| `B` | selected-data budget (number of examples), shared by all methods |

### 2.1 (A) Trajectory operator — the layer-resolved trajectory signature

We summarize `x` per layer by a **trajectory descriptor** that captures three
things mean-pooling throws away: (i) the *distribution* of activations across
steps/tokens (not just its mean), (ii) the *temporal geometry* (curvature/speed
of the residual stream over decoding steps), and (iii) the *cross-layer*
coupling. We then score by a **distributional distance** to the target anchor,
combined with a **curvature** term.

**(i) Per-layer empirical activation measure.** For layer `ℓ`, collect the cloud
`P_ℓ(x) = { h_ℓ,s,t : s ∈ [S_x], t ∈ [T_{x,s}] }` as an empirical distribution
on `R^d`. The *trajectory-as-distribution* idea is realized by comparing `P_ℓ(x)`
to the target measure `Q_ℓ^T` (the activation cloud of held-out target-capability
exemplars) with a distance that is **not** a function of the mean alone. We use
the **sliced-Wasserstein-2** distance, which is cheap and metrizes weak
convergence:

```
SW2²(P,Q) = E_{u~Unif(S^{d-1})} [ W2²( u#P , u#Q ) ]                 (Eq. 3)
```

estimated with `K` random projections `u_k`; for each `u_k`, `W2²` of the two
1-D pushforwards is a sort-and-integrate, `O((nP+nQ) log n)`. Define the
**distributional alignment** of `x` at layer `ℓ`:

```
D_ℓ(x) = SW2²( P_ℓ(x) , Q_ℓ^T ).                                     (Eq. 4)
```

Crucially `D_ℓ` depends on **all moments** of `P_ℓ(x)`, so two examples with
equal per-layer means but different spread/shape get different `D_ℓ` —
escaping the Eq. 2 collapse.

**(ii) Temporal curvature (uses ordering, which Eq. 3 ignores).** Let the
per-step mean trajectory at layer `ℓ` be `g_ℓ,s = mean_t h_ℓ,s,t ∈ R^d`,
`s = 1..S_x`. Define the **discrete velocity and acceleration**

```
v_ℓ,s = g_ℓ,s+1 − g_ℓ,s,        a_ℓ,s = g_ℓ,s+1 − 2 g_ℓ,s + g_ℓ,s−1.
```

The **trajectory-curvature statistic** (a normalized bending energy):

```
κ_ℓ(x) = ( 1 / (S_x−2) ) Σ_{s=2}^{S_x−1} || a_ℓ,s || / ( ||v_ℓ,s||·||v_ℓ,s−1|| + ε ).   (Eq. 5)
```

`κ_ℓ` is **permutation-sensitive**: reorder the decoding steps and `κ_ℓ` changes,
whereas any mean-pool `ψ_1` is permutation-invariant. This is the second,
independent escape from collapse, and it is the statistic the falsifiable
prediction (§3) is built on.

**(iii) Cross-layer signature and example score.** Stack the per-layer terms
over the probe set `A` into the **trajectory signature**

```
T(x) = ( { D_ℓ(x) }_{ℓ∈A} , { κ_ℓ(x) }_{ℓ∈A} ) ∈ R^{2|A|}.          (Eq. 6)
```

The **target-alignment score** (higher = more on-path for the target capability)
is a validation-fit linear read-out with a curvature-shape penalty:

```
ρ_T(x) = − Σ_{ℓ∈A} α_ℓ D_ℓ(x) − Σ_{ℓ∈A} β_ℓ | κ_ℓ(x) − κ_ℓ^T |,     (Eq. 7)
```

where `κ_ℓ^T` is the target exemplars' mean curvature and `(α,β) ≥ 0` are fit on
a held-out validation split only (never on test; never on drift metrics — this
respects `docs/baseline_contract.md` "Forbidden Comparisons").

**Cost boundedness.** `D_ℓ` is `O(K · n log n)` per layer with `n = S_x·T̄`;
`κ_ℓ` is `O(S_x·d)`. With `|A|` probe layers (not all `L`), `K≈32` projections,
and the existing **trajectory resolution sweep** (`docs/motivation_ablation...`,
"cost vs trajectory resolution"), total selection cost is held under the **2.0×
endpoint gate** (`passes_cost_gate`, `configs/.../max_selection_cost_multiplier`).
The signature `T(x)` is exactly what `schemas.TrajectorySignature` already
records (`layer_ids = A`, `step_count = S_x`, `token_count = Σ_s T_{x,s}`,
`trajectory_hash = H(T(x))`).

### 2.2 (B) Layer-routing grounded in measured layer function

The router decides, **per example**, which layers adapt (`m_ℓ=1`) vs freeze
(`m_ℓ=0`). It is grounded in a **measured** per-layer, per-capability importance
— not assumed.

**Layer-function profile (measured once, on validation).** For capability `c`
and layer `ℓ`, define a **causal/probing importance** `I_{c,ℓ} ∈ [0,1]`
estimated by either (default) a **mean-ablation causal drop** —

```
I_{c,ℓ} = ( Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ) ) / Acc_c(model),   (Eq. 8)
```

i.e. the relative capability drop when layer `ℓ`'s activations are replaced by
their dataset mean — or a **linear-probe accuracy** of capability `c` decoded
from `h_ℓ` (reported as a robustness variant). `I_{c,ℓ}` is computed once,
frozen, hashed into provenance, and is **validation-only**.

**Per-example routing rule.** An example `x` carries a target capability `c(x)`
(from its pool tag) and a measured **trajectory mass profile** `w_ℓ(x) =
D̃_ℓ(x)` (the alignment of Eq. 4, min-max normalized over `A`). Route by
**function–evidence compatibility**:

```
m_ℓ(x) = 1[  I_{c(x),ℓ} · w_ℓ(x)  ≥  τ  ]   for ℓ ∈ A,  else 0.       (Eq. 9)
```

Interpretation: **adapt a layer only when it is both causally important for the
example's capability (`I`) AND the example actually exercises that layer's
function on-path (`w`).** Freeze layers that are causally load-bearing for
*retained* capabilities but not exercised by `x` — this is the mechanism by
which LATTICE limits drift (a frozen important layer cannot be over-written).
`τ` is one scalar selected on validation retention-adjusted gain. The mask `m(x)`
is realized as **per-layer LoRA gating** (LoRA on layer `ℓ` is enabled iff
`m_ℓ(x)=1` for examples routed there); the global LoRA rank/budget is **shared
with all baselines** (`docs/baseline_contract.md`), so the layer claim is tested
at *matched* parameter budget, satisfying ARIS gate G4.

**Baseline tie-in.** Setting `m_ℓ ≡ 1` recovers `layer_selective_no_trajectory`'s
uniform-adapt and the standard uniform-LoRA; setting `I·w` → endpoint similarity
recovers an endpoint-routed control. These are the registered ablations
("layer policy without trajectory", "trajectory without layer policy").

### 2.3 (C) Retention-adjusted selection objective

Selection picks the budget-`B` subset `S` maximizing **retention-adjusted
marginal gain**, reusing `metrics.retention_adjusted_gain` exactly.

**Per-example predicted target gain:** `ĝ(x) = ρ_T(x)` (Eq. 7), centered.

**Per-example predicted drift** uses two *estimators* (this is the fix for
defect 3 — drift now has math and enters the objective):

- **Retention drift estimator** `r̂(x)`: a **Fisher-style stability penalty**.
  Let `F̄_ℓ` be the diagonal Fisher information of the base model on a held-out
  *general* set at layer `ℓ`. Training on `x` perturbs layer `ℓ` proportionally
  to its on-path gradient mass `w_ℓ(x)·m_ℓ(x)`. Define

  ```
  r̂(x) = Σ_{ℓ∈A} F̄_ℓ · w_ℓ(x) · m_ℓ(x).                            (Eq. 10)
  ```

  High `r̂` = the example writes to layers the base model relies on for general
  ability ⇒ likely retention loss. (Eq. 9's freezing directly lowers `r̂`.)

- **Hallucination/factuality drift estimator** `f̂(x)`: a **claim-support
  margin**. For factual sub-spans in `x`'s target, `f̂(x)` is the fraction of
  atomic claims whose model-assigned support log-odds fall below a calibrated
  threshold (an inexpensive proxy for the eval-time FActScore/TruthfulQA signal,
  validated to correlate with it on a held-out slice before any selection use).

**Subset objective (retention-adjusted gain):**

```
J(S) = Σ_{x∈S} [ ĝ(x) − λ_r · r̂(x) − λ_f · f̂(x) ]  −  γ · Redund(S),   (Eq. 11)
```

where `Redund(S)` is a facility-location / determinantal diversity term over the
signatures `T(x)` (prevents picking `B` near-duplicate on-path examples; the
"selection diversity temperature" hparam already in the plan). `J` is **modular +
submodular**, so a greedy `O(B·|pool|)` selection is `(1−1/e)`-optimal and stays
inside the cost gate. The per-example bracket
`[ĝ − λ_r r̂ − λ_f f̂]` is literally `retention_adjusted_gain(ĝ, r̂ , drift_weight=λ_r)`
extended with the factuality term — the existing metric function is the kernel.

**Gate wiring (unchanged thresholds).** After matched-budget training, the
realized quantities feed the existing gates: `passes_drift_gate(Δretention)`
(≤0.01), the analogous `Δhallucination ≤ 0.01`, `passes_cost_gate(cost/endpoint)`
(≤2.0), and the **trajectory–endpoint margin** (Δadjusted ≥ 0.02) plus
target/adjusted relative gain ≥ 0.03. These are read straight from
`configs/experiments/first_gate.yaml`.

### 2.4 Propriety PROOF (the calibrator-propriety requirement)

The factuality/retention estimators only deserve to *steer selection* if the
calibration that converts model log-odds into a drift score is **incentive-
compatible**: the calibrator must be scored by a **proper scoring rule**, so its
risk is minimized exactly at the true conditional support probability and cannot
be gamed by over/under-confidence. We make this precise and prove it.

**Construction (the proper-scoring calibrator).** For an atomic factual claim
`a` in example `x`, let `y_a ∈ {0,1}` be its (held-out, human/automatic-verified)
support label, and let the calibrator output `q_a = σ(z_a) ∈ (0,1)` from model
support log-odds `z_a`. Score the calibrator with the **Brier (quadratic)
scoring rule**

```
ℓ_Brier(q_a, y_a) = ( q_a − y_a )².                                  (Eq. 12)
```

The drift estimator `f̂(x)` of §2.3 is then the *risk-calibrated* aggregate
`f̂(x) = mean_a 1[ q_a < c* ]` where `c*` is the threshold chosen on the
Brier-calibrated `q`. Define the calibrator's **risk**
`R(q) = E_{(a,y)}[ ℓ_Brier(q(a), y) ]`.

**Proposition (propriety).** Brier is a strictly proper scoring rule: for the
true conditional `p_a = Pr(y_a = 1 | z_a)`, the population risk `R(q)` is
minimized **uniquely** at `q ≡ p`, i.e.
`E_{y∼p}[ℓ_Brier(q,y)] ≥ E_{y∼p}[ℓ_Brier(p,y)]` with equality iff `q=p`.
Hence a calibrator trained to minimize Eq. 12 is incentive-compatible: its
optimum is the *true* support probability, so `f̂` measures genuine factual
support rather than confidence-gaming, and threshold `c*` has a calibrated
meaning.

**Proof.** Fix a context with true positive probability `p`. For any reported
`q ∈ [0,1]`,
```
E_{y∼Bernoulli(p)}[ (q−y)² ]
   = p(q−1)² + (1−p)(q−0)²
   = q² − 2pq + p
   = (q − p)² + p(1 − p).                                            (Eq. 13)
```
The term `p(1−p)` is independent of `q` (irreducible variance), and `(q−p)² ≥ 0`
with equality iff `q = p`. Therefore `E[ℓ_Brier(q,y)]` is minimized **uniquely**
at `q = p`. Taking expectation over contexts preserves the strict inequality
pointwise, so `R(q) ≥ R(p)` with equality iff `q = p` a.s. ∎

**Consequence for the method.** Because the calibrator is fit under a strictly
proper rule, the decomposition Eq. 13 also yields the standard
**reliability–resolution** split (`(q−p)²` aggregates to a calibration term),
giving a *checkable* calibration diagnostic on held-out data and ensuring
`λ_f·f̂` in Eq. 11 penalizes true hallucination propensity, not artifacts of
miscalibration. (Log loss is an equally valid strictly proper alternative; Brier
is chosen for boundedness, which keeps `J(S)` well-scaled for the greedy.)

### 2.5 Algorithm box

```
Algorithm LATTICE-Select  (design-only; no server run)
Inputs : candidate pool X (budget B), probe layers A, target exemplars Q^T,
         general/Fisher set G, measured layer-function profile I_{c,ℓ},
         weights (α,β,λ_r,λ_f,γ,τ) fit on VALIDATION only, K projections
Output : selected set S (|S|=B), per-example layer masks m(x),
         SelectionManifest (server_authorized=False)

# Phase 0 (once): measure layer function  ── validation only
  for capability c, layer ℓ∈A: I_{c,ℓ} ← Eq.8 (mean-ablation causal drop)
  for ℓ∈A: F̄_ℓ ← diag Fisher of base model on G

# Phase 1: per-example trajectory signatures
  for x in X:
     extract H(x)={h_ℓ,s,t} on layers A (raw activations stay server-side)
     D_ℓ(x)  ← SW2²(P_ℓ(x), Q_ℓ^T)                       # Eq.3–4  (all moments)
     κ_ℓ(x)  ← bending energy of step-mean path           # Eq.5    (ordering)
     T(x)    ← ({D_ℓ},{κ_ℓ})                              # Eq.6  → TrajectorySignature
     ρ_T(x)  ← −Σ α_ℓ D_ℓ − Σ β_ℓ|κ_ℓ−κ_ℓ^T|             # Eq.7
     w_ℓ(x)  ← normalize D_ℓ(x) over A
     m_ℓ(x)  ← 1[ I_{c(x),ℓ}·w_ℓ(x) ≥ τ ]                 # Eq.9 routing

# Phase 2: drift estimators + objective
  for x in X:
     r̂(x) ← Σ F̄_ℓ·w_ℓ(x)·m_ℓ(x)                          # Eq.10 retention
     f̂(x) ← mean_a 1[ q_a < c* ],  q from Brier-proper calibrator   # §2.4
     ĝ(x) ← ρ_T(x)
  S ← greedy argmax of  J(S)=Σ[ĝ−λ_r r̂−λ_f f̂] − γ·Redund(S)        # Eq.11
       (submodular ⇒ (1−1/e) guarantee; cost ≤ 2.0× endpoint gate)

# Phase 3: emit contracts (LOCAL, additive, uncommitted)
  manifest ← SelectionManifest(project="neurotrace-it",
              baseline_ids⊇{"endpoint_neuron_selection"},
              signatures={T(x)}, selected_example_ids=S,
              server_authorized=False)
  assert validate_selection_manifest(manifest) == []
  return S, m, manifest
# Training/eval with mask m(x) is a SEPARATE server step — NOT run here.
```

---

## 3. Why it is NOT stitching

### 3.1 The single crisp scientific contribution

> **There exists a temporal/distributional component of the reasoning
> trajectory — captured by curvature `κ` and distributional alignment `D` — that
> is provably invisible to any endpoint or mean-pool activation feature, and that
> component carries causal information about fine-tuning outcome (target gain and
> retention drift) over and above endpoint-neuron similarity, at matched data,
> parameter, and compute budget.**

This is one claim, not glued parts. LATTICE's three pieces are not independent
modules bolted together; they are **derivations from a single quantity**: the
per-example trajectory signature `T(x)=({D_ℓ},{κ_ℓ})`. The selection score
(Eq. 7), the layer router (Eq. 9), and the retention penalty (Eq. 10) are **all
functions of the same `D_ℓ` and `w_ℓ`** — the layer mass that aligns to the
target is the same mass that routes adaptation and the same mass that predicts
drift. Remove the trajectory signature and *all three collapse simultaneously to
the endpoint baseline*. That shared origin is what makes it a method rather than
a stitch.

### 3.2 The falsifiable prediction (built on the §1.2 collapse)

Because §1.2 proves mean-pool ≡ endpoint, the contribution is falsifiable by
construction:

> **Prediction P:** Construct (or mine) example pairs `(x, x′)` with
> **near-identical endpoint signatures** `φ_end(x) ≈ φ_end(x′)` and near-equal
> per-layer means (so any endpoint/mean-pool selector scores them equally),
> but **different trajectory curvature** `κ(x) ≠ κ(x′)` (e.g. a direct-answer
> solution vs. a detour-then-correct solution reaching the same final state;
> these have equal endpoints by construction but different `a_ℓ,s`). LATTICE
> predicts they yield **measurably different fine-tuning outcomes** (target gain
> and/or retention drift differing by > the 0.02 trajectory–endpoint margin).

- **If P holds:** trajectory information is real and endpoint-invisible ⇒ the
  core novelty stands; the 0.02 margin gate (`trajectory_endpoint_margin`) is
  cleared by a mechanism endpoints *cannot* see.
- **If P fails** (matched-endpoint pairs train identically): the trajectory adds
  nothing endpoints miss ⇒ **kill** the novelty, downgrade to diagnostic exactly
  as `failure_action: stop_main_novelty_claim` mandates.

This is the sharp `trajectory ≠ endpoint` argument the brief demands: a single
construction whose outcome confirms or kills the paper.

---

## 4. Experiment design to confirm/kill the claim (design only — NO runs)

`server.authorized` stays **false**. Everything below is a plan; no command is
executed.

### 4.1 Datasets (reuse `docs/data_and_evaluation_plan.md`)

- **Target capability pools (train candidates):** math reasoning IT pool, code
  IT pool, multi-hop QA IT pool (each with frozen split hash + contamination
  audit, per `configs/experiments/first_gate.yaml`).
- **Held-out target evals:** math reasoning (exact match), executable code
  (pass rate), multi-hop QA with distractors.
- **Retention eval:** MMLU-style aggregate + general instruction-following
  (`aggregate_retention_score`, `semantic_drift`).
- **Hallucination/factuality eval:** TruthfulQA + FActScore-style atomic-claim
  factuality (`factuality_error_rate`, `harmful_overconfidence_rate`).

### 4.2 Base models (with sizes — non-toy)

Two open backbones at meaningfully different scale so the claim is not
scale-accidental: a **~7–8B** instruction-tunable model (primary; e.g. an 8B
open LLM) and a **~1.5–3B** model (secondary, for the layer-function/scale
robustness curve). Exact IDs pending user approval
(`first_gate_open_model_pending_user_approval`). LoRA only; rank/steps/tokens
**matched across all methods**. No tiny-toy-only or synthetic-only evidence.

### 4.3 Baselines (reuse `configs/baselines/baseline_registry.yaml`, all 7 + base paper)

`random_subset`, `full_data_it` (upper-bound reference), `quality_score_selection`,
`diversity_coreset`, `influence_gradient_selection`,
**`endpoint_neuron_selection` (the base paper, NAIT, faithfully reproduced — the
decisive comparator)**, `layer_selective_no_trajectory`. **Ablations** (registered
in `docs/motivation_ablation_hparam_plan.md`): endpoint-only; trajectory-`D`-only
(no `κ`); `κ`-only (no `D`); trajectory without layer router (`m≡1`); router
without trajectory (endpoint-routed); no retention penalty (`λ_r=0`); no
factuality penalty (`λ_f=0`).

### 4.4 Metrics & estimators

Target (EM/pass-rate), retention (`retention_adjusted_gain`), hallucination
drift, selection+training cost. Drift estimators §2.3; calibrator scored by the
**strictly proper Brier rule** (§2.4) with reliability–resolution diagnostic.

### 4.5 Matched-budget protocol (reuse `docs/baseline_contract.md` + `pre_registration.md`)

Same candidate pool, same selected-data budget `B`, same base model + tokenizer,
**same LoRA rank**, same training steps/tokens, same validation-only selection
policy, same evaluator, **shared seeds 0..19** (`configs/seeds/paper_20.txt`).
Paired tests on matched eval items; Holm correction across {target, retention,
hallucination, layer, cost}; effect sizes + 95% CIs
(`docs/statistical_analysis_plan.md`). The `κ-only`/`D-only`/endpoint-routed
ablations isolate *which* part of `T(x)` does the work.

### 4.6 The kill-gate (decisive, pre-registered)

The **matched-budget endpoint-vs-trajectory experiment** is the kill-gate, with
two layers:

1. **Mechanism kill (Prediction P, §3.2):** on matched-endpoint / divergent-`κ`
   pairs, if fine-tuning outcomes do **not** differ by > 0.02 adjusted, the
   endpoint-invisibility claim is false ⇒ **stop main novelty claim**.
2. **Aggregate kill (existing G1–G5):** if
   `trajectory_adjusted − endpoint_adjusted < 0.02`
   (`trajectory_endpoint_margin`), or target/adjusted relative gain < 0.03, or
   retention drift disadvantage > 0.01, or hallucination drift > 0.01, or cost
   > 2.0×, the corresponding claim is downgraded/forbidden per each gate's
   `failure_action`. Layer router must help in ≥2 capability families (G4) or
   become appendix-only. Paper tier needs ≥20 seeds (G5).

No gate threshold is changed by this redesign; LATTICE only supplies the
*operators* the gates were always meant to test.

---

## 5. What changes in code vs the current scaffold

All changes are **additive and uncommitted**; nothing existing is modified or
deleted. Files marked ADD are new; the scaffold's `schemas.py`, `metrics.py`,
`baseline_registry.yaml`, and gate configs are **reused as-is**.

**Reused unchanged**
- `src/neurotrace_it/schemas.py` — `TrajectorySignature` already stores exactly
  `T(x)` (layer_ids=A, step_count=S_x, token_count=Σ T, trajectory_hash=H(T(x)));
  `SelectionManifest` + `validate_selection_manifest` enforce the
  endpoint-baseline + `server_authorized=False` invariants the algorithm asserts.
- `src/neurotrace_it/metrics.py` — `retention_adjusted_gain` is the per-example
  kernel of Eq. 11; `passes_drift_gate` / `passes_cost_gate` are the §4.6 gates.
- `configs/baselines/baseline_registry.yaml`, `configs/experiments/*.yaml`,
  `configs/seeds/paper_20.txt`, `schemas/selection_manifest.schema.json` — gates,
  baselines, seeds, manifest schema all reused verbatim.

**To ADD later (not in this commit; design names only)**
- `src/neurotrace_it/trajectory_ops.py` — `sliced_wasserstein2` (Eq. 3–4),
  `trajectory_curvature` (Eq. 5), `trajectory_signature` → returns the existing
  `TrajectorySignature`.
- `src/neurotrace_it/layer_function.py` — `causal_layer_importance` (Eq. 8),
  `route_layers` (Eq. 9) → per-example mask `m(x)`.
- `src/neurotrace_it/drift.py` — `fisher_retention_penalty` (Eq. 10),
  `BrierCalibrator` (§2.4, strictly proper) + `factuality_drift` (`f̂`).
- `src/neurotrace_it/select.py` — `lattice_select` (greedy submodular Eq. 11)
  emitting a `SelectionManifest`.
- `configs/experiments/lattice_v1.yaml` — operator hparams `(K,α,β,λ_r,λ_f,γ,τ)`,
  probe-layer set `A`, trajectory-resolution sweep; **`server.authorized: false`**.
- `tests/test_trajectory_ops.py` — contract tests: (a) `κ` is permutation-
  sensitive while mean-pool is invariant (encodes §1.2 escape); (b) Brier
  propriety numeric check via Eq. 13; (c) `validate_selection_manifest` passes.
- `docs/redesign/REDESIGN_v1.md` — this file.

No server command, no training, no extraction is added or run.

---

## 6. Open risks + honest limitations

1. **Curvature signal-to-noise.** `κ` (Eq. 5) can be noisy for short decoding
   traces (`S_x` small). Mitigation: trajectory-resolution sweep already planned;
   if `κ-only` ablation is at chance, fall back to the `D`-only distributional
   claim (which alone still escapes the Eq. 2 collapse). Honest: the curvature
   half of the contribution may not survive.
2. **`D_ℓ` vs the collapse — incomplete guarantee.** SW2 escapes *mean-only*
   collapse, but if target/non-target clouds happen to differ mostly in their
   means on real data, `D_ℓ` could empirically track endpoint similarity. The
   matched-endpoint/divergent-`κ` construction (§3.2) is the *only* clean
   falsifier; on naturally-mined pairs the separation may be small.
3. **Causal importance cost.** Eq. 8 mean-ablation over `|A|` layers ×
   capabilities adds measurement cost; counted inside the selection-cost gate,
   but if it pushes past 2.0× the router must be reported as analysis-only.
4. **Factuality estimator is a proxy.** `f̂` correlates with FActScore/TruthfulQA
   by design but is not the eval metric; the propriety proof guarantees
   *calibration*, not that the proxy's claim-decomposition matches the eval's. A
   held-out proxy-vs-eval correlation check is a precondition, not an assumption.
5. **Single-anchor target.** `Q^T` assumes a coherent target-capability cloud;
   for heterogeneous capabilities `μ_T/Q^T` may need mixtures, untested here.
6. **No empirical claim yet.** Per `docs/paper_claims_status.md`, every result
   above is *design*. "Beats NAIT" remains forbidden wording until the
   matched-budget endpoint reproduction runs under approval. This document
   changes the *operators and their falsifiability*, not the evidence status.

---

*Provenance:* additive redesign; no files modified/deleted; no git commit; no
experiment executed; `server.authorized: false` preserved throughout.

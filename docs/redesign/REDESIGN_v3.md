# REDESIGN v3 — neurotrace-it (Stage-1 Registered Report design)

Status: **BUILD-NOW / RUN-LATER**. This document is a **Stage-1 Registered
Report** (RR): a method + a *pre-registered* analysis plan, written **before**
any data are collected. Under the RR model the unit of review is the **question,
the method, and the pre-registered analysis** — not results — so "no results
yet" is the *expected* state at this stage, not a defect (cf. the v2 GPT-5.5
review, which graded a pre-data design as if it were a results paper and
scored 3/10 mainly because "there is no paper-result evidence"). Acceptance of a
Stage-1 RR is *in-principle acceptance* conditional on executing the locked plan.

No experiment is run here. `server.authorized: false` is preserved. No numbers
are fabricated; every numeric expression below is a **closed-form / quadrature
formula evaluation**, explicitly labeled "formula evaluation, not evidence."
This document is **additive** — it adds only
`docs/redesign/REDESIGN_v3.md`; no other file is modified or deleted; no git
commit is made.

This redesign reuses the existing governance verbatim: the seven-family
`configs/baselines/baseline_registry.yaml`, the numeric gates in
`configs/experiments/first_gate.yaml` / `formal_neurotrace_it.yaml`
(trajectory–endpoint margin 0.02, target/adjusted relative gain 0.03, retention
drift ≤ 0.01, hallucination drift ≤ 0.01, cost ≤ 2.0×), the metric API in
`src/neurotrace_it/metrics.py` (`retention_adjusted_gain`, `passes_drift_gate`,
`passes_cost_gate`), and the manifest contracts in
`src/neurotrace_it/schemas.py` and `schemas/selection_manifest.schema.json`. **No
gate threshold is changed by v3.**

v3 preserves the **verified-correct** parts of v2 (the permutation-sensitive
curvature statistic and its non-functionality proof; the sliced-Wasserstein
distributional operator; the monotone-submodular facility-location objective with
its earned (1−1/e) proof; the strictly-proper Brier proof) and **fixes the six
precise defects** the v2 reviewer flagged as the *fatal scientific* concerns
(A–F below), the most load-bearing of which is that the v2 residualization
operator was **ill-typed**.

---

## Changelog vs v2

| Fix | v2 defect (precise) | v3 change | Where |
| --- | --- | --- | --- |
| **A. Residualization operator was ill-typed** | v2 §3.1 wrote `ρ̃_T(x) = T(x) − Proj_{span(φ_end)} T(x)`. But `T(x) ∈ R^{2|A|}` and `φ_end(x) ∈ R^{2d|A|}` live in **different spaces**; a *per-example* projection of one vector onto the span of a single other vector in a different dimension is undefined (and even within a space, projecting one point onto one vector is not residualization). | Restated as a **DATASET-LEVEL residualized multiple regression** (Frisch–Waugh–Lovell / added-variable): across the labeled pool, regress utility `Y` on trajectory features `T(x)` **while controlling for** `φ_end(x)`, length, difficulty, and dataset-family. The estimand is a **partial regression coefficient / partial correlation**, not a per-example vector projection. The read-out `(α,β)` is **fit inside the residualized regression on a train fold and the test statistic is evaluated on a held-out fold** (no validation-fit reuse as the test statistic ⇒ no leakage). "Residual trajectory signal survives controls" is the **PRIMARY falsifiable statistic**. | §2.3, §3.1, §4.2 |
| **B. Cost-gate integrity** | v2 already split per-pool vs amortized cost, but the split could be *gamed*: pushing causal-ablation / Fisher / calibrator / proxy / hparam costs "amortized/outside" flatters the 2× gate because **NAIT carries none of that stack**. | (i) **Gate-1** = per-pool *selection* cost inside the pre-registered 2× multiplier, defined as the **incremental** cost over the *identical* extraction NAIT also pays (so the comparison is apples-to-apples). (ii) A pre-registration **AMENDMENT** adds **Gate-1b**, a *second* gate that **includes amortized validation cost** (causal `I`, Fisher, calibrator, proxy diagnostic, hparam fit) for **single-run deployability**, with a **concrete pool-reuse count `R`** that any amortization claim must name. Per-pool selection cost and amortized validation cost are reported on **distinct lines**. | §2.6, §4.6, §4.7 |
| **C. Matched-endpoint kill-gate made RUNNABLE on natural pairs** | v2's matched-`φ_end` / divergent-`κ` test risked being "artificial / synthetic-only." | A **concrete mining recipe** for **naturally occurring** pairs: bucket real pool examples by capability/answer, retrieve nearest-neighbor pairs under `‖φ_end(x)−φ_end(x')‖₂ ≤ τ_end` (explicit **percentile-1 tolerance**), keep only pairs whose **curvature gap `|κ(x)−κ(x')|` is in the top decile** (divergent). Target **n = 300** retained pairs; **power analysis** for the paired test at the 0.02 margin is given (`n=300` gives power ≈ 0.92 at d=0.2; `n≥197` suffices for 0.80). No contrived/synthetic construction. | §3.2, §4.5 |
| **D. κ-only vs D-only ablation decision rule pre-registered** | v2 noted the residual "may be carried entirely by D" but did not pre-declare the contingency/fallback. | A pre-registered **2×2 decision table** over {`D`-only significant?, `κ`-only significant?}: if only `D` survives controls, the **temporal-curvature novelty is dropped** and the claim **falls back** to "distributional (magnitude) trajectory signal beyond endpoints"; the paper does **not** claim curvature. The fallback claim and its allowed wording are fixed *now*. | §3.3, §4.4 |
| **E. Factuality proxy is a precondition GATE** | v2 demoted factuality to an ablation but the gate's *thresholds/curve* were under-specified. | Promoted to an explicit **precondition gate G6** evaluated **before** `λ_f` is allowed non-zero: held-out proxy-vs-eval **Spearman ρ ≥ 0.3 AND Pearson r ≥ 0.3 (lower 95% CI > 0)** *and* a **reliability curve** with ECE ≤ 0.1. **Fail ⇒ `λ_f := 0`**, no factuality/safety claim. The Brier propriety proof is retained (it proves calibration, *not* proxy validity). | §2.5, §4.4 (G6) |
| **F. Citation/model hygiene + concrete instantiation** | v2 inherited registry arXiv IDs `2603.13201` (NAIT) and `2605.26004` (coreset) — both **future-dated/implausible** — and left base-model IDs as `pending_user_approval` and SW2 settings loose. | The two arXiv IDs are **flagged UNVERIFIED** and **must be replaced** by a real citation before any must-beat/must-cite claim (verification checklist in §4.8); no "beats NAIT" wording until then. Base models **pinned**: primary `Qwen2.5-7B-Instruct`, secondary `Qwen2.5-1.5B-Instruct`, with named open alternates. SW2 settings **pinned**: `K=64` projections, `|A|=8` probe layers, subsample `≤512` (step,token) activations/example. The pins make the design concretely instantiable. | §1.4, §2.2, §4.8 |
| (preserved) | v2 curvature non-functionality proof, SW2 operator, submodular (1−1/e) proof, Brier propriety proof. | **Kept verbatim** (re-proved in §2.4, §2.7, §2.5), only re-numbered. | §2 |

Note: the **NAIT endpoint baseline is the FIRST Phase-2 code deliverable** (§5,
Deliverable #1) — it is both the *residualization control* `φ_end` and the
*decisive comparator*; no proposed operator may be measured for "gain over
endpoints" until it exists and passes its unit tests.

---

## 1. Defect / context

### 1.1 The base method and the seed question

The base paper — **Neuron-Aware Data Selection (NAIT)** — scores an instruction
example `x` for selection using only activations at the **START and END tokens**.
For a transformer with `L` layers and hidden width `d`, write the post-block
residual-stream activation at token position `p`, layer `ℓ` as `h_ℓ(p) ∈ R^d`.
NAIT compresses an example to an **endpoint feature**

```
φ_end(x) = concat_{ℓ ∈ A} [ h_ℓ(p_start) , h_ℓ(p_end) ]  ∈ R^{2 d |A|}     (Eq. 1)
```

over an anchor-layer set `A`, and selects by similarity of `φ_end(x)` to a
target-capability anchor `μ_T`.

NeuroTrace-IT asks whether the **full reasoning trajectory** — activations over
(layers × decoding-steps × tokens) — carries selection-relevant information
**beyond what `φ_end` already captures**, plus a layer-wise adapt/freeze rule
(treated as an ablation).

### 1.2 The v2 audited defects this RR repairs

The v2 design was correctly framed (endpoint-residualized claim, NAIT as decisive
baseline) but carried **six precise scientific defects**, which are exactly the
"fatal concerns" of the v2 review and the targets of this RR:

1. **(A) The residualization operator was ill-typed.** v2 defined the central
   estimand as a *per-example* projection `T(x) − Proj_{span(φ_end)} T(x)`. This
   is undefined: `T(x) ∈ R^{2|A|}` and `φ_end(x) ∈ R^{2 d |A|}` are vectors of
   **different dimension**, so "project `T(x)` onto `span(φ_end(x))`" has no
   meaning. Even inside one space, projecting a single point onto a single vector
   is not what "residualize out the endpoint signature" should mean. The fix
   (§2.3) makes the estimand a **dataset-level multiple regression** that controls
   for `φ_end` (and nuisance covariates), where residualization is the standard,
   well-typed Frisch–Waugh–Lovell operation on **columns across examples**.
2. **(B) Cost-gate integrity.** The amortized/per-pool split can be gamed by
   parking the method-only stack (causal ablation, Fisher, calibrator, proxy,
   hparams) outside the 2× gate that NAIT does not pay (§2.6, §4.6–§4.7).
3. **(C) Kill-gate runnability.** The matched-endpoint / divergent-curvature test
   needed a concrete *natural*-pair mining recipe, tolerance, target `n`, and
   power (§3.2, §4.5).
4. **(D) Magnitude-vs-curvature contingency.** The residual could be carried
   entirely by the magnitude term `D` and collapse the temporal-curvature
   novelty; the contingency must be pre-declared (§3.3, §4.4).
5. **(E) Factuality proxy validity.** Calibration (Brier) ≠ proxy predicts eval
   drift; proxy validity is a **precondition gate**, not an assumption
   (§2.5, §4.4).
6. **(F) Citation/model hygiene.** Two registry arXiv IDs are implausible and
   must be replaced; base models and SW2 settings must be pinned (§1.4, §2.2, §4.8).

### 1.3 What first-moment pooling loses (preserved from v2 — NOT "endpoint-equivalent")

We never claim mean-pool equals endpoint (v1's error). The endpoint feature reads
two **positions**; a per-layer trajectory **mean** averages over *all* steps and
tokens — different statistics. The defensible, modest statement is only that
**first-moment temporal pooling is insufficient**: it is permutation- and
shape-invariant and so discards the temporal/distributional structure v3 exploits.

Let `H(x) = { h_{ℓ,s,t} }` over layers `ℓ ∈ [L]`, steps `s ∈ [S_x]`, tokens
`t ∈ [T_{x,s}]`. Define the per-layer first-moment summary
`ψ₁(x)_ℓ = (1/N_ℓ) Σ_{s,t} h_{ℓ,s,t}`. For any score linear in the summary,
`score(x) = ⟨w, ψ₁(x)⟩`,

```
score(x) = Σ_ℓ ⟨w_ℓ , mean_{s,t} h_{ℓ,s,t}⟩ = mean_{s,t} ⟨w, h_{·,s,t}⟩.    (Eq. 2)
```

Two **provable, modest** consequences:

- **(a) Permutation invariance.** `ψ₁` is invariant to any reordering of steps
  `s`. A first-moment selector cannot distinguish a direct-answer path from a
  detour-then-correct path with the same activation multiset. A statistic that
  *changes* under reordering (curvature `κ`, §2.4) is therefore **not a function
  of `ψ₁`** (proof in §2.4).
- **(b) Moment blindness.** `ψ₁` ignores spread/shape; a distributional distance
  (SW2, §2.2) responds to higher moments `ψ₁` cannot see.

§1.3 does **not** claim `ψ₁ = φ_end`, nor that SW2/`κ` are *guaranteed* to beat
`φ_end` — that is the empirical question settled by the residual regression (§3.1)
and the kill-gate (§3.2). The collapse to escape is **first-moment pooling**; the
contribution must live in the part of the trajectory that survives **controlling
for `φ_end`** at the dataset level (§2.3).

### 1.4 Citation & model hygiene (fix F — must-do before any must-beat claim)

- **UNVERIFIED arXiv IDs.** `configs/baselines/baseline_registry.yaml` lists
  `https://arxiv.org/abs/2603.13201` (NAIT, `endpoint_neuron_selection`) and
  `https://arxiv.org/abs/2605.26004` (`diversity_coreset`). Both encode year
  **26xx** — these are **future-dated / implausible** arXiv identifiers and are
  treated here as **placeholders requiring verification or replacement** (§4.8).
  Until a real, checkable citation is pinned for the NAIT baseline, the forbidden
  wording in `docs/paper_claims_status.md` ("better than NAIT") stays forbidden,
  and `endpoint_neuron_selection` is implemented as a **faithful-reimplementation
  endpoint baseline** described by Eq. 1, not as "a reproduction of paper X."
- **Pinned base models (fix F).** Primary **`Qwen2.5-7B-Instruct`** (~7.6B);
  secondary **`Qwen2.5-1.5B-Instruct`** (~1.5B) for scale-robustness. Named open
  alternates if licensing/availability requires: `Llama-3.1-8B-Instruct` /
  `Llama-3.2-3B-Instruct`, or `Mistral-7B-Instruct-v0.3`. LoRA only; rank, steps,
  tokens matched across all methods. These replace
  `first_gate_open_model_pending_user_approval`; the config marker stays as the
  approval flag, but the design now names concrete checkpoints so the plan is
  instantiable.
- **Pinned trajectory settings (fix F).** `K = 64` SW2 random projections (seeds
  persisted); probe-layer set `|A| = 8` (an evenly spaced grid over depth, fixed
  on validation); per-example activation **subsample cap = 512** (step,token)
  vectors per layer, sampled with a persisted mask. These are the knobs the
  resolution sweep (§2.6) is allowed to move to hold the cost gate.

---

## 2. Method (equation-level, with proofs)

**Name:** **LATTICE** — *Layer-Aware Trajectory-distribuTIon seleCtion with
rEtention-adjusted gain* (working name; the contribution, not the acronym,
matters). Dependency order mirrors the deliverables (§5): endpoint baseline
`φ_end` (control + comparator) → trajectory operator `T(x)` → dataset-level
residualized regression (the claim) → capacity-matched router (ablation) →
monotone-submodular selection objective → strictly-proper calibrator (factuality
gated by validation).

### 2.0 Notation

| Symbol | Meaning |
| --- | --- |
| `x` | candidate instruction example (prompt + target reasoning trace) |
| `L, S_x, T_{x,s}` | #layers, #decoding steps for `x`, #tokens at step `s` |
| `h_{ℓ,s,t} ∈ R^d` | residual-stream activation, layer `ℓ`, step `s`, token `t` |
| `A ⊆ [L]`, `|A|=8` | anchor/probe layer set (fixed on validation) |
| `φ_end(x) ∈ R^{2 d |A|}` | endpoint signature, Eq. 1 (Deliverable #1) |
| `μ_T, Q_ℓ^T` | target anchor mean / target-capability exemplar cloud at layer `ℓ` |
| `D_ℓ(x)` | per-layer SW2² to target cloud, Eq. 4 (magnitude/distributional term) |
| `κ_ℓ(x)` | per-layer temporal curvature, Eq. 5 (temporal term) |
| `T(x) ∈ R^{2|A|}` | trajectory feature vector `({D_ℓ},{κ_ℓ})`, Eq. 6 |
| `Y` | retention-adjusted fine-tuning utility outcome (the regression target) |
| `c(x)` | nuisance covariates: length, difficulty, dataset-family (one-hot) |
| `m ∈ {0,1}^L` | per-example layer mask (1 = route) — ablation only |
| `B` | selected-data budget (#examples), shared by all methods |

### 2.1 Endpoint-neuron baseline (Deliverable #1 — built FIRST)

Faithful NAIT (Eq. 1): `φ_end(x) = concat_{ℓ∈A}[h_ℓ(p_start), h_ℓ(p_end)]`,
`score_end(x) = sim(φ_end(x), μ_T)` (cosine or class-mean neuron overlap).
`φ_end` plays **two** roles: the **decisive matched-budget comparator** and the
**control block** in the residualized regression (§2.3). No proposed operator may
be measured for "gain over endpoints" until `score_end`/`φ_end` exist and pass
`tests/test_endpoint_baseline.py` (§5).

### 2.2 (A) Trajectory operator — layer-resolved signature

**(i) Per-layer distributional term → sliced-Wasserstein-2 (preserved).** For
layer `ℓ`, collect the cloud `P_ℓ(x) = { h_{ℓ,s,t} }` (subsampled to ≤512) as an
empirical distribution on `R^d`; compare to the held-out target cloud `Q_ℓ^T`:

```
SW2²(P,Q) = E_{u~Unif(S^{d-1})} [ W2²( u#P , u#Q ) ]                          (Eq. 3)
D_ℓ(x)    = SW2²( P_ℓ(x) , Q_ℓ^T ),                                          (Eq. 4)
```

estimated with `K=64` random projections (seeds persisted); each 1-D `W2²` is a
sort-and-integrate, `O((n_P+n_Q) log n)`. `D_ℓ` is the **magnitude** term and
depends on **more than the first moment** (finite-sample, finite-`K`); it is *not*
a function of `ψ₁` alone. We do **not** claim "all moments," and we do **not**
claim it must beat `φ_end` — settled only by §3.1.

**(ii) Temporal curvature `κ` (uses ordering — Eq. 3 does not).** Let the per-step
mean path be `g_{ℓ,s} = mean_t h_{ℓ,s,t}`; velocity `v_{ℓ,s} = g_{ℓ,s+1} − g_{ℓ,s}`,
acceleration `a_{ℓ,s} = g_{ℓ,s+1} − 2 g_{ℓ,s} + g_{ℓ,s−1}`. Define the
**normalized bending energy**

```
κ_ℓ(x) = (1/(S_x−2)) Σ_{s=2}^{S_x−1}  ‖a_{ℓ,s}‖ / ( ‖v_{ℓ,s}‖·‖v_{ℓ,s−1}‖ + ε ).   (Eq. 5)
```

**(iii) Cross-layer signature.**

```
T(x) = ( { D_ℓ(x) }_{ℓ∈A} , { κ_ℓ(x) }_{ℓ∈A} ) ∈ R^{2|A|}.                  (Eq. 6)
```

The raw validation-fit read-out (never used as the test statistic) is

```
ρ_T(x) = − Σ_{ℓ∈A} α_ℓ D_ℓ(x) − Σ_{ℓ∈A} β_ℓ | κ_ℓ(x) − κ_ℓ^T |,           (Eq. 7)
```

with `κ_ℓ^T` the target exemplars' mean curvature, `(α,β) ≥ 0`. The **test**
read-out is the residualized regression coefficient of §2.3, fit on train and
evaluated on a held-out fold.

### 2.3 (A, PRIMARY) Dataset-level endpoint-residualized regression — the well-typed estimand

> **Fix A.** Residualization is a **column operation across examples**, not a
> per-example vector projection. We never project `T(x) ∈ R^{2|A|}` onto
> `φ_end(x) ∈ R^{2d|A|}` (different spaces). Instead, over the labeled pool we run
> a **multiple regression of the scalar outcome `Y` on the trajectory columns
> `T(·)` while controlling for the endpoint columns `φ_end(·)` and nuisance
> covariates `c(·)`**, and test whether the trajectory block carries non-zero
> partial signal.

Let the labeled set be `{(x_i, Y_i)}_{i=1}^N`, where `Y_i` is the
retention-adjusted fine-tuning utility (operationalized in §4.1). Stack design
matrices `T ∈ R^{N×2|A|}`, `Φ ∈ R^{N×p_Φ}` (a fixed low-dimensional **endpoint
summary** — see "dimension control" below), `C ∈ R^{N×p_C}` (length, difficulty,
dataset-family one-hots), and a column of ones. The **full** and **reduced**
models are

```
(full)    Y = T β_T + Φ β_Φ + C β_C + 1·b + ε,                              (Eq. 8a)
(reduced) Y =        Φ β_Φ + C β_C + 1·b + ε.                               (Eq. 8b)
```

**Frisch–Waugh–Lovell (well-typed residualization).** Let `M = I − Z(ZᵀZ)⁺Zᵀ` be
the residual-maker for `Z = [Φ, C, 1]`. FWL gives `β̂_T` from the **residualized**
regression of `MY` on `MT`:

```
β̂_T = ( (MT)ᵀ(MT) )⁺ (MT)ᵀ (MY).                                           (Eq. 9)
```

Here `MT` is `T` with the endpoint/nuisance signal **regressed out column-by-
column across the N examples** — a standard, dimension-safe operation (no
cross-space projection). The **partial correlation** of trajectory and outcome
given controls is `corr(MT_j, MY)` per column.

**PRIMARY falsifiable statistic (locked).** The trajectory block carries residual
signal iff the **block F-test** of `H₀: β_T = 0` in (Eq. 8a) vs (Eq. 8b) rejects,
i.e.

```
F = [ (RSS_red − RSS_full)/q ] / [ RSS_full/(N − k) ],   q = 2|A|,          (Eq. 10)
```

reported with its p-value, the partial-R² increment `ΔR² = (RSS_red − RSS_full)/RSS_red`,
and 95% CIs on `β̂_T`. **Primary success:** the held-out block test is significant
(Holm-corrected) **and** `ΔR²` exceeds a pre-registered floor (§4.2). This is the
operationalization of "residual trajectory signal survives controls."

**Leakage control (locked).** The read-out weights and the regression are **fit on
a train fold**; the **F-test / partial-R² is computed on a disjoint held-out
fold** by applying the train-estimated `M` and `β̂` (out-of-sample residualization
and prediction). We **never** reuse a validation fit as the test statistic. Folds
are example-disjoint and dataset-family-stratified; the 20 shared seeds index the
fold splits.

**Dimension control for `Φ` (so the control is honest, not degenerate).** Raw
`φ_end ∈ R^{2d|A|}` is too wide to use as `N` regression controls (it would
over-fit and trivially absorb everything). We instead control for a **fixed,
pre-registered endpoint *summary*** `Φ` of dimension `p_Φ ≪ N`: (1) `score_end(x)`
(the NAIT similarity itself), (2) the top-`r` PCA components of `φ_end` fit on the
**train fold only** with `r = 16` (persisted loadings), and (3) per-layer endpoint
norms. This guarantees `Φ` captures the endpoint *decision signal* NAIT actually
uses while keeping `k = p_Φ + p_C + 2|A| + 1 ≪ N` so (Eq. 10) is well-posed. The
choice of `r`, the PCA-on-train rule, and `p_Φ` are **locked** here.

### 2.4 (preserved) `κ` is provably not a function of `ψ₁`

**Proposition.** There exist trajectories with identical `ψ₁` but different `κ_ℓ`.
**Proof.** Take two step-sequences that are reorderings of each other:
`ψ₁` (a mean over the multiset) is invariant to the permutation, but `κ_ℓ`
(Eq. 5) depends on the *ordered* second differences `a_{ℓ,s}`, which change under
reordering whenever the path is not collinear. Hence `κ_ℓ` is not a function of
`ψ₁`. ∎ (This is the statistic the kill-gate, §3.2, is built on.)

### 2.5 Drift estimators + strictly-proper calibrator (factuality gated, fix E)

**Retention drift `r̂(x)`** — Fisher-style stability penalty. With `F̄_ℓ` the
diagonal Fisher of the base model on a held-out general set,

```
r̂(x) = Σ_{ℓ∈A} F̄_ℓ · w_ℓ(x) · m_ℓ(x),    w_ℓ(x) = D̃_ℓ(x) (min-max over A).  (Eq. 11)
```

High `r̂` ⇒ the example writes to layers the base model relies on for general
ability ⇒ likely retention loss. (Uniform LoRA ⇒ `m_ℓ ≡ 1`.)

**Factuality drift `f̂(x)`** — claim-support margin from a calibrated proxy:
`f̂(x) = mean_a 1[q_a < c*]`, `q_a = σ(z_a)` the calibrator output, `c*` a
calibrated threshold.

**Brier propriety (preserved; correct).** Score the calibrator with
`ℓ_Brier(q,y) = (q−y)²`. For `y ∼ Bernoulli(p)`,

```
E[(q−y)²] = p(q−1)² + (1−p)q² = (q − p)² + p(1 − p).                        (Eq. 12)
```

`p(1−p)` is independent of `q`; `(q−p)² ≥ 0` with equality iff `q=p`, so the
population risk is **uniquely** minimized at `q=p`; taking expectation over
contexts preserves strictness. ∎ The same decomposition yields the
reliability–resolution split used by the gate. **This proves calibration, not
that the proxy predicts eval drift.**

> **Precondition gate G6 (fix E).** Before `λ_f` may be non-zero, require on a
> held-out slice: **Spearman ρ(f̂, drift_eval) ≥ 0.3 AND Pearson r ≥ 0.3 with
> lower 95% CI > 0**, *and* a reliability curve with **ECE ≤ 0.1**. **Fail ⇒
> `λ_f := 0`** and no factuality/safety claim is made. Thresholds are locked
> (§4.4). Calibration is necessary but not sufficient; proxy validity is the gate.

### 2.6 (B) Cost model with integrity — incremental gate + amortized second gate

> **Fix B.** The 2× gate must compare like with like. NAIT pays the **same base
> extraction** LATTICE pays; what LATTICE *adds* is the selection stack. We
> therefore define the per-pool gate on the **incremental** selection cost over
> the shared extraction, and we **do not** hide the method-only validation stack —
> we put it in a *second*, named gate with a concrete reuse count.

- **Gate-1 (per-pool selection, INSIDE the pre-registered 2× multiplier).**
  `mult₁ = cost_select(LATTICE) / cost_select(NAIT)` over the **same pool and the
  same shared activation extraction**. `cost_select` counts only the *incremental*
  per-example selection work: SW2 (`K·n log n`), `κ` (Eq. 5), the residualized
  regression scoring, drift estimators, greedy coverage updates. This is the
  existing `passes_cost_gate` multiplier; threshold **2.0× unchanged**.
- **Amortized validation cost (REPORTED SEPARATELY, never inside Gate-1).** Causal
  layer importance `I_{c,ℓ}` (Eq. 13: `|A|·|capabilities|` mean-ablation passes),
  diagonal Fisher `F̄_ℓ`, calibrator fit + G6 proxy diagnostic,
  `(α,β,μ,τ,λ,r,K,|A|)` fitting. Paid **once** per (model, validation split).
- **Gate-1b (single-run deployability, pre-registration AMENDMENT — fix B).** A
  **second** gate that **includes** the amortized validation cost:
  `mult_dep = ( cost_amortized/R + cost_select(LATTICE) ) / cost_select(NAIT)`,
  where **`R` is the number of candidate pools the validation artifacts are reused
  across**. Any amortization claim **must name `R`**; the design pre-registers a
  **conservative default `R = 1`** (single-run deployability: a deployer who
  selects on *one* pool pays the whole validation stack) **and** reports the
  break-even `R*` such that `mult_dep ≤ 2.0×`. Reporting both `R=1` and `R*` makes
  the amortization auditable rather than self-serving.

**Holding Gate-1 ≤ 2.0×.** The dominant incremental term is SW2 (linear in `K`
and `n log n`). The **resolution sweep** (`docs/motivation_ablation_hparam_plan.md`,
"cost vs trajectory resolution") tunes `K`, `|A|`, and (step,token) subsample to
keep `mult₁ ≤ 2.0×`; if it cannot, the method is reported as a **high-cost
analysis method** per `failure_action: require_efficiency_ablation` — not a
deployment method. The honest two-line table (§4.6) shows both gates.

### 2.7 (preserved) Monotone-submodular selection objective — earned (1−1/e)

Per-example utility (modular, non-negative):
`u(x) = max(0, ĝ(x) − λ_r r̂(x) − λ_f f̂(x))`, where `ĝ(x)` is the
endpoint-residualized read-out (§2.3, centered to be non-negative by a constant
shift), reusing `metrics.retention_adjusted_gain` as the `ĝ − λ_r r̂` kernel; the
factuality term is added with `λ_f = 0` unless G6 (§2.5) passes.

Facility-location coverage over the trajectory-feature space, with
`σ(x,e) ∈ [0,1]` a similarity to ground element `e` (target-concept exemplar; `E`
the held-out exemplar set):

```
C(S) = Σ_{e∈E} w_e · max_{x∈S} σ(x,e);   F(S) = Σ_{x∈S} u(x) + μ·C(S),  μ≥0.   (Eq. 14)
```

**Proposition (NWF (1−1/e)).** `F` is monotone non-decreasing and submodular;
greedy (`argmax_x [F(S∪{x})−F(S)]` until `|S|=B`) returns `S_greedy` with
`F(S_greedy) ≥ (1−1/e)·F(S*)`. **Proof.** (1) `Σ u(x)`, `u≥0`, is modular and
monotone. (2) For each `e`, `S↦max_{x∈S}σ(x,e)` is monotone (a max cannot drop
when adding elements) and submodular (its marginal `max(0, σ(x,e) − running max)`
is non-increasing in the running max, which is non-decreasing in the set); a
non-negative weighted sum of monotone-submodular functions is
monotone-submodular, so `C` is. (3) A non-negative combination of
monotone-submodular functions is monotone-submodular, so `F` is; NWF applies. ∎
Greedy is `O(B·|pool|·|E|)` and stays inside Gate-1. (If a future variant uses a
non-submodular term, the guarantee is dropped and greedy is reported as a
heuristic; the shipped form keeps the proof.)

### 2.8 (B-ablation) Capacity-matched layer router (not load-bearing)

Per-example LoRA layer gating changes the trainable footprint, so it is a
**validated ablation** only; the main claim uses **uniform** LoRA. Layer-function
profile (validation-only, frozen, hashed):

```
I_{c,ℓ} = ( Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ) ) / Acc_c(model).        (Eq. 13)
```

Routing `m_ℓ(x) = 1[ I_{c(x),ℓ}·w_ℓ(x) ≥ τ ]`. **Capacity-matching contract:**
fix total LoRA rank `R_tot = |A|·r_unif` and **reallocate** it across routed
layers (`r_ℓ = R_tot/|{ℓ:m_ℓ=1}|`, clipped) so active params, optimizer-state
slots, FLOPs (per-step and total), and effective rank match the uniform baseline;
any variant that cannot match exactly is reported as **capacity-unmatched
diagnostic**, never a matched-budget claim. The router is kept only if it helps at
matched capacity in ≥2 capability families (G4).

### 2.9 Versioned, auditable signature record (preserved)

`TrajectorySignatureV2` (`schema_version="2.0.0"`, added additively in
`src/neurotrace_it/schemas_v2.py`) **persists the estimands**: `example_id`,
`layer_ids (A)`, `endpoint_signature` (Eq. 1, the control), `D = {ℓ:D_ℓ}`,
`kappa = {ℓ:κ_ℓ}`, `projection_seeds` (K), `slice_masks`,
`alignment_scores {ρ_T, residual β̂_T}`, `selection_scores {u(x), marginal_gain}`,
`drift_estimates {r̂, f̂}`, `calibrator_provenance {rule:"brier", c*,
reliability_hash, G6_pass:bool}`, `layer_router_outputs {m(x), I_profile_hash,
R_tot, r_per_layer}`, and `trajectory_hash` = `H(all of the above)` as an
**integrity check that recomputes from the stored estimands**.
`validate_selection_manifest_v2` requires non-empty `D`/`kappa` over `layer_ids`,
present `endpoint_signature`, present `projection_seeds`, and a recomputing hash.
`server_authorized` stays `false`; `endpoint_neuron_selection` stays required.

### 2.10 Algorithm box

```
Algorithm LATTICE-Select  (design-only; no server run)
Inputs : pool X (budget B), probe layers A (|A|=8), target clouds {Q_ℓ^T}, concept set E,
         general/Fisher set G, layer-function profile I_{c,ℓ}, K=64 projections+seeds,
         weights (α,β,λ_r,λ_f,μ,τ,r) fit on TRAIN fold only
Output : selected set S (|S|=B), [ablation] masks m(x),
         SelectionManifest with TrajectorySignatureV2 records (server_authorized=False)

# Phase -1 : ENDPOINT BASELINE FIRST (Deliverable #1) — control + comparator
  for x in X: φ_end(x) ← Eq.1 ; score_end(x) ← sim(φ_end, μ_T)

# Phase 0 (once, validation, AMORTIZED): layer fn + Fisher + calibrator + G6
  I_{c,ℓ} ← Eq.13 ; F̄_ℓ ← diag Fisher on G ; fit Brier calibrator ; run G6 → set λ_f

# Phase 1 : per-example trajectory signatures (PER-POOL, incremental cost)
  for x in X:
     D_ℓ(x) ← SW2²(P_ℓ,Q_ℓ^T)   # Eq.4 (store seeds, value)
     κ_ℓ(x) ← bending energy      # Eq.5 (store value)
     T(x)   ← ({D_ℓ},{κ_ℓ})       # Eq.6

# Phase 2 : dataset-level residualized regression (the CLAIM, train-fit/held-out-test)
  build Φ (score_end, PCA_16(φ_end|train), endpoint norms), C (len/diff/family)
  β̂_T ← FWL residualized regression (Eq.9) on TRAIN fold
  report block-F (Eq.10), ΔR², CIs on HELD-OUT fold     # PRIMARY statistic

# Phase 3 : drift + monotone-submodular selection (PER-POOL cost)
  for x in X: r̂(x) ← Eq.11 ; f̂(x) ← §2.5 ; u(x) ← max(0, ĝ − λ_r r̂ − λ_f f̂)
  S ← greedy argmax of F(S)=Σ u(x) + μ·C(S)   # Eq.14 ; (1−1/e) by §2.7

# Phase 4 : emit auditable contracts (LOCAL, additive, uncommitted)
  records ← { TrajectorySignatureV2(x) : x∈S }
  manifest ← SelectionManifest(project="neurotrace-it",
              baseline_ids⊇{"endpoint_neuron_selection"},
              signatures=records, selected_example_ids=S, server_authorized=False)
  assert validate_selection_manifest_v2(manifest) == []
  return S, m, manifest
# Training/eval (uniform LoRA main; matched-capacity masks ablation) is a SEPARATE server step — NOT run here.
```

---

## 3. Contribution + falsifiable predictions

### 3.1 The single crisp contribution (RESIDUALIZED, well-typed)

> **There exists a temporal/distributional component of the reasoning trajectory
> — captured by curvature `κ` and distributional alignment `D` — that, AFTER
> controlling at the dataset level for the endpoint signature `φ_end` (and
> length, difficulty, dataset-family), still carries information predictive of
> retention-adjusted fine-tuning utility, at matched data, parameter, and compute
> budget.**

Formally (fix A): the **block F-test** of `H₀: β_T = 0` in the multiple
regression `Y ~ T + Φ + C` (Eq. 8a–10) **rejects on a held-out fold**, with
`ΔR²` above the pre-registered floor. This is one claim from **one quantity**
`T(x)=({D_ℓ},{κ_ℓ})`: the selection score (§2.2), the router (§2.8, ablation), the
retention penalty (§2.5), and the coverage term (§2.7) are all functions of the
same `D_ℓ/κ_ℓ`. Because the gain is the **partial** signal over endpoints, it
cannot be an artifact of `T(x)` re-encoding `φ_end` — that variance is regressed
out across examples by FWL. Remove the trajectory block and the claim is empty;
that shared origin (plus the residualization) is what makes it a method, not a
stitch.

### 3.2 Falsifiable Prediction P (matched-endpoint / divergent-curvature, runnable on NATURAL pairs)

> **Prediction P.** Among **naturally occurring** pool pairs `(x, x')` with
> near-identical endpoint signatures `‖φ_end(x)−φ_end(x')‖₂ ≤ τ_end` but
> **divergent curvature** `|κ(x)−κ(x')|` in the top decile, LATTICE predicts
> **measurably different retention-adjusted fine-tuning outcomes** (paired
> difference > the 0.02 trajectory–endpoint margin). Because `φ_end` is matched,
> any outcome difference is **purely residual** trajectory information.

**Concrete mining recipe (fix C — no synthetic construction).**
1. From each real instruction pool (math, code, multi-hop QA), compute `φ_end` and
   `κ` for all candidates (reusing Phase-1 outputs).
2. **Bucket** by capability family and by coarse final-answer equivalence
   (exact-match answer / normalized output / AST-equal code) so paired examples
   teach "the same endpoint."
3. Within a bucket, retrieve nearest neighbors in `φ_end` (cosine / `ℓ₂`) and keep
   pairs with **`‖φ_end(x)−φ_end(x')‖₂ ≤ τ_end`**, where **`τ_end` = the 1st
   percentile** of the within-bucket pairwise endpoint-distance distribution
   (explicit, data-driven tolerance; persisted).
4. Among those, **keep pairs whose `|κ(x)−κ(x')|` is in the top decile**
   (divergent curvature) — e.g. a direct-answer path vs a detour-then-correct path
   reaching the same final state.
5. **Target `n = 300`** retained pairs (≈100 per family); if a family yields fewer,
   report the achieved `n` and re-run power.

**Outcomes.** If P **holds** ⇒ endpoint-residualized trajectory information is real
⇒ core novelty stands, 0.02 margin cleared by a mechanism endpoints cannot see. If
P **fails** (matched-endpoint pairs train identically) ⇒ residual carries nothing
⇒ `failure_action: stop_main_novelty_claim`, downgrade to diagnostic.

### 3.3 Pre-registered `κ`-only vs `D`-only contingency (fix D)

The residual could be carried **entirely by `D` (magnitude)**, collapsing the
temporal-curvature novelty. Pre-declared 2×2 decision table (evaluated on the
held-out fold, Holm-corrected):

| `D`-only block sig? | `κ`-only block sig? | Decision (locked) | Allowed claim |
| --- | --- | --- | --- |
| yes | yes | full claim | "distributional **and** temporal-curvature trajectory signal beyond endpoints" |
| yes | no | **fallback** | "**distributional (magnitude)** trajectory signal beyond endpoints"; **curvature novelty dropped** |
| no | yes | curvature-only | "temporal-curvature trajectory signal beyond endpoints"; magnitude term demoted |
| no | no | **kill** | no trajectory-beyond-endpoint claim; `stop_main_novelty_claim` |

The fallback wording is fixed **now** so the contingency cannot be rationalized
post hoc. `κ`-only and `D`-only are run as registered ablation poles (§4.3).

### 3.4 Predictions summary (all falsifiable, pre-data)

- **P1 (primary):** held-out block F-test of `β_T=0` rejects with `ΔR² ≥ floor`
  (§4.2). *Falsifier:* non-significant or `ΔR² < floor`.
- **P2 (mechanism):** matched-endpoint/divergent-`κ` paired difference > 0.02
  (§3.2). *Falsifier:* ≤ 0.02.
- **P3 (aggregate):** trajectory-selected − endpoint-selected adjusted gain
  ≥ 0.02 margin and ≥ 0.03 relative, retention drift disadvantage ≤ 0.01,
  hallucination drift ≤ 0.01 (existing G1–G3).
- **P4 (router, ablation):** helps in ≥2 families at matched capacity (G4), else
  appendix-only.
- **P5 (cost):** Gate-1 `mult₁ ≤ 2.0×`; Gate-1b reported with `R=1` and `R*`.

---

## 4. Pre-registered analysis plan + kill-gates + power

`server.authorized` stays **false**. Everything below is a locked plan; no command
is executed. This is the Stage-1 analysis lock.

### 4.1 Outcome `Y`, datasets, models

- **Outcome `Y` (locked):** per-example **retention-adjusted fine-tuning utility**,
  computed as `retention_adjusted_gain(target_gain, retention_drift,
  drift_weight)` (existing metric) measured by a **leave-one-cluster-in / influence-
  style** attribution at fixed budget (so each `x` gets a scalar `Y`). The exact
  attribution estimator is locked to the influence-on-validation procedure used by
  the `influence_gradient_selection` baseline, applied identically to all methods.
- **Datasets (reuse `data_and_evaluation_plan.md`):** math IT, code IT, multi-hop
  QA IT (frozen split hash + contamination audit); held-out evals: math EM, code
  pass-rate, multi-hop QA with distractors; retention: MMLU-style aggregate +
  general IF; hallucination: TruthfulQA + FActScore-style atomic-claim factuality
  (also the ground truth for G6).
- **Models (pinned, fix F):** `Qwen2.5-7B-Instruct` (primary),
  `Qwen2.5-1.5B-Instruct` (secondary); alternates per §1.4. LoRA only; matched
  rank/steps/tokens.

### 4.2 PRIMARY analysis (fix A) — locked

Fit (Eq. 8a–9) on the **train fold**; compute the **block F-test** (Eq. 10),
`ΔR²`, and 95% CIs on `β̂_T` on the **held-out fold**. **Pre-registered floors:**
held-out block-test **p < 0.05 after Holm** across {target, retention,
hallucination, layer, cost}, **and** held-out **`ΔR² ≥ 0.02`** (the same 0.02
scale as the margin gate; an effect smaller than this is treated as "endpoints
already explain it"). `Φ` is the locked endpoint summary (score_end + PCA-16 on
train + per-layer endpoint norms); `C` = length, difficulty, dataset-family. Folds
example-disjoint, family-stratified, indexed by the 20 shared seeds. **Leakage
rule locked:** train-estimated `M`,`β̂`,PCA loadings applied out-of-sample; no
validation fit is reused as a test statistic.

### 4.3 Baselines & ablations (reuse `baseline_registry.yaml`)

`random_subset`, `full_data_it` (upper bound), `quality_score_selection`,
`diversity_coreset`, `influence_gradient_selection`,
**`endpoint_neuron_selection` (NAIT, Deliverable #1; control + decisive
comparator)**, `layer_selective_no_trajectory`. **Ablations:** endpoint-only;
**`D`-only (no κ)**; **`κ`-only (no D)** (the §3.3 poles); router on/off at matched
capacity (§2.8); `λ_r=0`; `λ_f=0` (default unless G6 passes); coverage `μ=0`.

### 4.4 Secondary/contingency gates

- **§3.3 2×2 decision table** evaluated on the held-out fold (locked wording).
- **G6 factuality precondition (fix E):** Spearman ρ ≥ 0.3 **and** Pearson r ≥ 0.3
  (lower 95% CI > 0) between `f̂` and the eval factuality-drift signal, **and**
  reliability ECE ≤ 0.1. **Fail ⇒ `λ_f := 0`**, no factuality/safety claim.

### 4.5 Power analysis for Prediction P (fix C)

Paired test (Wilcoxon signed-rank / paired-t on matched pairs) at the **0.02
adjusted margin**. Treating the standardized paired effect as `d = Δ/σ_Δ`, the
required `n` for power `1−β` at two-sided `α=0.05` is
`n ≈ (z_{1−α/2}+z_{1−β})² / d²` (formula evaluation, not evidence):

| assumed `d` | `n` for power 0.80 | `n` for power 0.90 |
| --- | --- | --- |
| 0.20 (small) | **197** | **264** |
| 0.30 | 88 | 118 |
| 0.50 | 32 | 43 |

**At the pre-registered target `n = 300`**, power at `d=0.20` is
`Φ(√300·0.20 − 1.96) = Φ(3.46 − 1.96) = Φ(1.50) ≈ 0.93` (**formula evaluation, not
evidence**). Thus `n=300` is powered (~0.93) to detect even a *small* paired
effect at the 0.02 margin; the design pre-commits to re-running this formula with
the achieved `n` per family. If a family cannot reach `n≥197`, that family's P-test
is labeled **underpowered diagnostic**, not a kill.

### 4.6 Kill-gates (decisive, pre-registered) + honest two-line cost table

1. **Primary residual kill (P1, §4.2):** held-out block test non-significant *or*
   `ΔR² < 0.02` ⇒ **stop main novelty claim**.
2. **Mechanism kill (P2, §3.2):** matched-endpoint/divergent-`κ` paired difference
   ≤ 0.02 ⇒ **stop main novelty claim** (`failure_action: stop_main_novelty_claim`).
3. **Aggregate kill (existing G1–G5):** trajectory_adjusted − endpoint_adjusted
   < 0.02, or target/adjusted relative gain < 0.03, or retention drift
   disadvantage > 0.01, or hallucination drift > 0.01, or **Gate-1** `mult₁ > 2.0×`
   ⇒ downgrade/forbid per each `failure_action`. Router must help ≥2 families at
   matched capacity (G4) or be appendix-only. Paper tier needs ≥20 seeds (G5).
4. **Contingency (D, §3.3):** the 2×2 table selects the allowed claim/fallback.
5. **Factuality precondition (E, G6):** fail ⇒ `λ_f:=0`.

**Honest two-line cost table (fix B):** the compute plan reports (i) **per-pool
incremental selection** (SW2, `κ`, residualized regression, drift, greedy) — the
**only** line inside Gate-1's 2× multiplier; and (ii) **amortized validation**
(causal `I`, Fisher, calibrator + G6, hparam fit) — paid once per (model, split),
shown separately and fed into **Gate-1b** with `R=1` and the break-even `R*`. The
v1 "2h vs 4h = exactly 2×" framing is withdrawn; the per-pool figure is set by the
resolution sweep.

### 4.7 Pre-registration amendment (fix B, recorded)

> **AMENDMENT to `docs/pre_registration.md` (design-only, additive).** Add
> **Gate-1b (single-run deployability):** `mult_dep = (cost_amortized/R +
> cost_select(LATTICE)) / cost_select(NAIT)` with conservative default **`R=1`**
> and reported break-even **`R*`**. Gate-1 (per-pool, 2.0×) is unchanged. Rationale:
> NAIT carries no validation stack, so deployability must be judged including the
> amortized cost, not only the per-pool increment.

### 4.8 Citation/model verification checklist (fix F — blocking before must-beat)

- [ ] Replace/verify `endpoint_neuron_selection` `paper_url`
  (`arxiv.org/abs/2603.13201` is future-dated/implausible) with a real, checkable
  NAIT citation **or** relabel the baseline as a faithful endpoint reimplementation
  (Eq. 1) with no paper attribution. **Blocks** any "beats NAIT" wording.
- [ ] Replace/verify `diversity_coreset` `paper_url`
  (`arxiv.org/abs/2605.26004` is future-dated/implausible) with a real coreset
  citation.
- [ ] Confirm base-model checkpoint IDs, licenses, and tokenizer hashes for
  `Qwen2.5-7B-Instruct` / `Qwen2.5-1.5B-Instruct` (or pinned alternates) before
  approval.
- [ ] Persist SW2 settings (`K=64`, `|A|=8`, subsample 512) and PCA `r=16` in
  `configs/experiments/lattice_v3.yaml` (added later, additively).

### 4.9 Matched-budget protocol (reuse `baseline_contract.md` + `pre_registration.md`)

Same pool, budget `B`, base model + tokenizer, **same LoRA rank** (`R_tot`),
same steps/tokens, validation-only selection, same evaluator, **shared seeds
0..19** (`configs/seeds/paper_20.txt`). Paired tests on matched eval items; Holm
across {target, retention, hallucination, layer, cost}; effect sizes + 95% CIs
(`docs/statistical_analysis_plan.md`). Router ablation only at matched capacity;
any unmatched variant is diagnostic-labeled.

---

## 5. Code plan for Phase-2 (additive, uncommitted; deliverable order enforced)

All changes are **additive**; nothing existing is modified or deleted.
`schemas.py`, `metrics.py`, `baseline_registry.yaml`, and gate configs are reused
as-is. **Deliverable #1 (endpoint baseline) is built and tested before any
proposed-method operator.**

**Reused unchanged:** `src/neurotrace_it/metrics.py` (`retention_adjusted_gain`
kernel; `passes_drift_gate`/`passes_cost_gate` gates); `src/neurotrace_it/schemas.py`
(V1 records stay valid, back-compatible); `configs/baselines/baseline_registry.yaml`,
`configs/experiments/*.yaml`, `configs/seeds/paper_20.txt`,
`schemas/selection_manifest.schema.json`.

**To ADD later (design names only; dependency order):**
- **Deliverable #1 — `src/neurotrace_it/endpoint_baseline.py`** — `endpoint_signature`
  (Eq. 1), `endpoint_score` (NAIT similarity). **Built and tested first.**
  `tests/test_endpoint_baseline.py` — faithful-endpoint contract + similarity tests.
- `src/neurotrace_it/schemas_v2.py` — `TrajectorySignatureV2` (§2.9) +
  `validate_selection_manifest_v2` (additive; does not touch `schemas.py`).
- `src/neurotrace_it/trajectory_ops.py` — `sliced_wasserstein2` (Eq. 3–4, returns
  value **and** seeds), `trajectory_curvature` (Eq. 5), `trajectory_signature`.
- **`src/neurotrace_it/residualize.py`** — `build_endpoint_summary` (Φ: score_end +
  PCA-16-on-train + endpoint norms), `fwl_residualize` (Eq. 9), `block_f_test`
  (Eq. 10), `partial_r2` — **the PRIMARY statistic**, with strict train/held-out
  fold separation (leakage control).
- `src/neurotrace_it/match_pairs.py` — `mine_matched_endpoint_pairs` (§3.2 recipe:
  bucket → NN in φ_end → τ_end=p1 → top-decile |Δκ|), `paired_margin_test`,
  `power_for_pairs` (§4.5 formula).
- `src/neurotrace_it/layer_function.py` — `causal_layer_importance` (Eq. 13),
  `route_layers`, `capacity_match` (§2.8 rank reallocation).
- `src/neurotrace_it/drift.py` — `fisher_retention_penalty` (Eq. 11),
  `BrierCalibrator` (§2.5) + `factuality_drift` + `proxy_eval_diagnostic` (G6).
- `src/neurotrace_it/select.py` — `lattice_select` (greedy monotone-submodular
  `F(S)`, Eq. 14) emitting a V2 `SelectionManifest`.
- `configs/experiments/lattice_v3.yaml` — pinned hparams (`K=64,|A|=8,subsample=512,
  r=16,α,β,λ_r,λ_f,μ,τ,R_tot`), probe-layer grid `A`, resolution sweep, **two-gate
  cost budget (Gate-1 per-pool + Gate-1b amortized with R)**; **`server.authorized:
  false`**.
- `tests/test_trajectory_ops.py` — (a) `κ` permutation-sensitive while mean-pool
  invariant (encodes §2.4); (b) Brier propriety numeric check via Eq. 12
  (**formula evaluation, not evidence**); (c) `F(S)` submodularity unit check
  (diminishing-returns inequality on a tiny synthetic set — **formula evaluation,
  not evidence**); (d) FWL residualization **recovers the known coefficient** on a
  synthetic linear DGP where the trajectory column is orthogonalized against the
  control (**formula evaluation, not evidence**); (e)
  `validate_selection_manifest_v2` round-trips estimands + recomputes
  `trajectory_hash`.

No server command, no training, no extraction is added or run.

---

## 6. Honest limitations

1. **Residual signal may be small or null (intended falsifier).** The core claim is
   now a **dataset-level partial effect** (`β_T ≠ 0` given `Φ,C`); if endpoints +
   length + difficulty + family already explain the variance, the block test fails
   and the claim is killed by §4.2/§4.6. The honest prior is that the residual may
   be small on naturally mined pairs; the §3.2 natural-pair test is the crisp test.
2. **`Φ` is a *summary*, not all of `φ_end`.** To keep (Eq. 10) well-posed we
   control for a 16-dim PCA + score_end + norms, not raw `φ_end ∈ R^{2d|A|}`. If the
   endpoint *decision* signal lives outside that summary, the control is
   incomplete; mitigation: report sensitivity to `r ∈ {8,16,32}` (the main `r=16`
   is locked; others are sensitivity-only, not the registered test).
3. **Curvature SNR.** `κ` (Eq. 5) is noisy for short traces (`S_x` small);
   resolution sweep mitigates; if `κ`-only is at chance the §3.3 table drops the
   curvature novelty and the claim falls back to the `D` (magnitude) term.
4. **SW2 vs the collapse is empirical, not guaranteed.** SW2 escapes *first-moment*
   collapse but may empirically track endpoint/mean similarity; only the residual
   regression settles it. We do **not** claim "all moments."
5. **Matched-pair availability.** The §3.2 recipe depends on real pools containing
   enough near-matched-endpoint / divergent-`κ` pairs; if a family yields `n<197`
   its P-test is underpowered-diagnostic (not a kill), and the achieved `n` and
   re-run power are reported.
6. **Router fairness is fragile.** Capacity-conserving rank reallocation matches
   params/optimizer-state/total-rank, but exact FLOPs / "effective capacity"
   equality is hard; any residual mismatch forces a capacity-unmatched
   **diagnostic** label. The main result uses **uniform** LoRA so it never depends
   on this.
7. **Factuality proxy validity is a precondition (G6), not a result.** Brier proves
   calibration, not eval-drift prediction; fail G6 ⇒ `λ_f:=0`, no hallucination-
   reduction claim.
8. **Cost honesty (two gates).** Amortized validation cost is real and reported on
   its own line; only per-pool incremental selection is inside Gate-1's 2×.
   Gate-1b (with `R=1` and `R*`) governs single-run deployability. If `mult₁ > 2.0×`
   after the sweep, the method is **high-cost analysis**, not deployment.
9. **Citation/model risk (fix F).** Two registry arXiv IDs are future-dated and
   **unverified**; "beats NAIT" stays forbidden until a real NAIT citation is
   pinned or the baseline is relabeled a faithful endpoint reimplementation.
10. **No empirical claim yet (RR Stage-1).** Per `docs/paper_claims_status.md`,
    every result above is **design**; in-principle acceptance is conditional on
    executing the locked plan under approval. The only numeric checks here (Eq. 12
    Brier identity; the submodularity inequality; the §4.5 power formulas) are
    **formula evaluations, not evidence**.

---

*Provenance:* additive Stage-1 Registered Report; adds only
`docs/redesign/REDESIGN_v3.md`; no file modified/deleted; no git commit; no
experiment executed; `server.authorized: false` preserved throughout.

# REDESIGN v4 — neurotrace-it (Stage-1 Registered Report design)

Status: **BUILD-NOW / RUN-LATER — METHOD + ANALYSIS PIPELINE IMPLEMENTED &
FROZEN; THEORY DISCHARGED WHERE PROVED; GPU RUN DEFERRED (server.authorized:
false).** This document is a **Stage-1 Registered Report** (RR): a method + a
*pre-registered* analysis plan, written **before** any data are collected. Under
the RR model the unit of review is the **question, the method, and the
pre-registered analysis** — not results — so "no results yet" is the *expected*
state at this stage, not a defect. Acceptance of a Stage-1 RR is *in-principle
acceptance* conditional on executing the locked plan.

**Closure state (2026-06-19).** The novel-core pipeline named in §5 is now
**implemented and frozen as pure-stdlib, build-now/run-later modules** (no model
load, no server call, no training): the NAIT endpoint baseline
(`src/neurotrace_it/baselines/nait.py`, Deliverable #1), the trajectory operator
(`src/neurotrace_it/trajectory.py` — SW2 `D_ℓ`, curvature `κ_ℓ`, signature `T`),
the co-primary residual estimand + valid OOS inference
(`src/neurotrace_it/analysis/residual_test.py` — dual/`n`-space ridge-FWL,
cross-fit partial-R², conditional-null block permutation), the gates/CI machinery
(`analysis/residualize.py` — endpoint control + PCA-`r` poles, cluster-BCa CI,
two-layer Holm, contingency, robustness floor, achieved-power), the LOCI outcome
+ G7 (`analysis/outcome_y.py`), the matched-pair diagnostic
(`analysis/pair_mining.py`), the Brier calibrator + G6 (`analysis/drift.py`), the
monotone-submodular selector (`selection.py`), the V2 auditable signature record
(`schemas_v2.py`), and the frozen config (`configs/experiments/lattice_v4.yaml`).
The **theory below is discharged where provable** (Brier propriety,
dual-ridge feasibility/equivalence, residual-orthogonality estimand, matched-pair
identification lemma, `κ` non-functionality, submodular (1−1/e)); the paper
skeleton is written at `paper/main.tex` (+ `paper/references.bib`). What remains
is the **authorized GPU run** (training/extraction/eval), which is **deferred** —
`server.authorized: false` stays set and **no command in this design is
executed**.

No experiment is run here. `server.authorized: false` is preserved. No numbers
are fabricated; every numeric expression below is a **closed-form / quadrature
formula evaluation**, explicitly labeled "formula evaluation, not evidence."
This document was originally **additive**; the implementation it specifies has
since landed as the additive modules listed above (no existing public API meaning
is changed; APIs are only **extended**, and V1 records stay valid). All references
to existing code/config (`metrics.py`, `schemas.py`, `baseline_registry.yaml`,
`first_gate.yaml`, `formal_neurotrace_it.yaml`,
`schemas/selection_manifest.schema.json`, `configs/seeds/paper_20.txt`) remain
**reuse-as-is**.

v4 is a **response-to-review revision of v3**. It keeps the verified-correct
core of v3 (the FWL/partial-regression framing, the SW2 distributional operator,
the permutation-sensitive curvature statistic and its non-functionality proof,
the monotone-submodular (1−1/e) objective, the strictly-proper Brier proof, the
endpoint-baseline-first deliverable order, the two-gate cost honesty, the
factuality precondition gate, and the citation/model-hygiene discipline) and
**fixes the six precise defects the GPT-5.5 v3 review raised** (the review scored
v3 **7/10** and named four upgrades required to reach 9/10: a fully specified
`Y`, a valid cross-fitted statistical test, a stronger endpoint control, and a
cost gate that counts trajectory extraction). v4 implements all four plus two
additional pre-registration hardenings (ΔR² floor denominator; multiplicity
across the {D, κ, joint} family).

---

## Changelog vs v3 (each review finding → concrete fix)

| # | v3 defect (GPT-5.5 review, redesign_v3_gpt55.md) | v4 fix (this document) | Where |
| --- | --- | --- | --- |
| **(a)** | **"Beyond-endpoint" still partly under-controlled (HIGH).** v3 §2.3 controlled only an endpoint *summary* `Φ = score_end + PCA-16 + per-layer norms`, not the full endpoint geometry `φ_end ∈ R^{2d|A|}`; the *registered* test therefore rested on a **weaker** control, so a "trajectory beyond endpoints" rejection could be an artifact of endpoint variance the 16-PC summary dropped. | The endpoint control is made **HONEST and CO-PRIMARY**: the registered partialling-out is against the **FULL** `φ_end` via **high-dimensional ridge partialling-out** (ridge-FWL / double-ML-style orthogonalization), not a 16-PC summary. The summary-PCA control is **demoted to a sensitivity pole**. We **pre-register `r ∈ {8, 16, 32}`** PCA-rank controls *and* the full-ridge control, with a **robustness floor**: the trajectory effect must survive the *full-ridge* control AND be **monotone-stable** across `r` (no sign flip; CI overlap). Co-primary = the claim passes only if **both** the full-ridge partialling-out **and** the matched-pair diagnostic point the same way. | §2.3, §3.1, §4.2 (CP-1) |
| **(b)** | **Held-out block F-test is statistically invalid (HIGH).** v3 fit the regression/residualizer on a train fold and then reported a **classical nested-model F-test (Eq. 10) on held-out RSS**. A nested-model `F` is derived from *train-fitted* RSS under Gaussian-OLS nesting; it does **not** have an `F` reference distribution when both RSS are computed out-of-sample from train-frozen `M, β̂`. | The nested-model `F` is **removed as the inferential test** (kept only as a descriptive train-fold diagnostic). The locked out-of-sample test is a **valid resampling test**: (1) **K=10 repeated cross-fit** estimation of the held-out partial effect, aggregated by a **median-of-folds** point estimate; (2) a **block permutation test** (permute the trajectory block `T` rows *within* family/fold strata, refit, recompute the held-out statistic — exact null for "T adds nothing given controls"); (3) a **cluster/stratified bootstrap** 95% CI. Significance = permutation p (Holm) AND bootstrap-CI excludes the floor. No `F`-distribution p-values are used inferentially. | §2.4, §4.2, §4.3 |
| **(c)** | **Primary outcome `Y` not operationally locked (HIGH).** v3 defined `Y` as "leave-one-cluster-in / influence-style attribution" but did not specify cluster construction, the estimator equations, the train/eval budget, or a reliability precondition; the primary regression was thus hard to reproduce and potentially proxy-circular. | `Y` is **fully locked** (§4.1): exact **cluster construction** (embedding + HDBSCAN params, `K_clusters`, size floor, persisted assignment hash), the **leave-one-cluster-in (LOCI) influence estimator equations** (Eq. 15–17), and the **training/eval budget** (steps, tokens, LoRA rank, eval items). A new **Y-reliability precondition gate G7** (§2.6, §4.4) requires (i) **test–retest / seed stability** of `Y` (ICC ≥ 0.6 across ≥3 seeds), and (ii) **proxy↔ground-truth validity**: on a pre-registered subset, the LOCI influence proxy must correlate with **actual held-out fine-tuning deltas** (Spearman ρ ≥ 0.3, lower 95% CI > 0). **Fail G7 ⇒ the primary regression is not run on the influence proxy**; fall back to the direct retrain-delta `Y` on the subset (underpowered-diagnostic) — no primary claim on an unreliable `Y`. | §2.6, §4.1, §4.4 (G7) |
| **(d)** | **ΔR² floor denominator ambiguous.** v3 §2.3 wrote `ΔR² = (RSS_red − RSS_full)/RSS_red`, an **overall** increment normalized by the reduced RSS; it conflated "fraction of *remaining* variance explained" (a **partial-R²** quantity) with the **overall-R² increment**, and was matched to the 0.02 margin without a stated mapping. | The floor is **re-specified as a partial-R²** with the correct denominator: `partial_R² = (RSS_red − RSS_full) / RSS_red` is the **fraction of the residual (endpoint-controlled) variance** explained by the trajectory block — that is the registered quantity (it answers "of what endpoints leave unexplained, how much does T explain"). The **overall-R² increment** `ΔR²_overall = (RSS_red − RSS_full)/TSS` is reported **separately** and is the quantity **mapped to the 0.02 deployment margin** (§4.2): a 0.02 overall-R² increment ≈ the 0.02 adjusted-score margin scale. Both are locked; the kill-gate uses the partial-R² floor for the residual claim and the overall-R² increment for the deployment-margin map. | §2.3, §4.2 |
| **(e)** | **Multiplicity within the {D, κ, joint} family not pre-registered.** v3 ran `D`-only, `κ`-only, and joint blocks (the §3.3 2×2 table) and applied Holm across the *metric* family {target, retention, hallucination, layer, cost} but **did not** pre-register a correction across the **three trajectory-block hypotheses** themselves, inviting a "test all three, claim whichever wins" inflation. | A **two-layer multiplicity correction is pre-registered** (§4.5): (1) **within** the trajectory family {D-only, κ-only, joint}, Holm over the **3** block tests (the §3.3 decision table reads the Holm-adjusted significances, not raw); (2) **across** metric families {target, retention, hallucination, layer, cost}, Holm as before. The joint test is the **gatekeeper**: if the joint block fails, D-only/κ-only are reported as **exploratory** only. Family-wise α = 0.05 throughout. | §3.3, §4.5 |
| **(f1)** | **Cost gate still favors the proposed method (MEDIUM).** v3 Gate-1 excluded the "shared" activation extraction, but endpoint-only NAIT need **not** extract full step/token/layer trajectories — so the "shared extraction" premise is false and the differential extraction/storage burden was hidden. | The **"shared extraction" loophole is closed** with an **extraction-parity kill condition** *and* trajectory I/O accounting (§2.7): either (i) **both** methods are forced to consume the *identical* full-trace extraction for a justified reason (then extraction truly cancels) — pre-registered as the default — **or** (ii) if endpoint-only could run on a cheaper endpoint-only extraction, the **differential trajectory extraction + storage I/O is counted INSIDE Gate-1's 2× multiplier**. A new **extraction-parity kill**: if trajectory extraction/storage alone pushes the honest end-to-end multiplier > 2.0× and cannot be reduced by the resolution sweep, the method is **declared high-cost-analysis** (no deployment claim). | §2.7, §4.6 |
| **(f2)** | **Matched-pair mechanism claim overstated causality (MEDIUM).** v3 §3.2 said matched endpoints make outcome differences "**purely** residual trajectory information"; but natural-pair mining can still confound on length, difficulty, reasoning style, annotation quality, or pool artifacts. | The matched-pair claim is **softened to a strong DIAGNOSTIC, not causal proof** (§3.2). Wording changed from "purely residual" to "**consistent with** residual trajectory information **after** observed-covariate matching." Pairs are additionally **balanced on length/difficulty/family** (coarsened exact matching) and the **residual covariate imbalance is reported**; the test is positioned as **corroborating** the co-primary regression, never as standalone causal evidence. | §3.2 |
| **(f3)** | **Future-dated arXiv IDs; base models loosely pinned.** Registry IDs `2603.13201` (NAIT) and `2605.26004` (coreset) encode year 26xx (future-dated/implausible); base-model pins needed confirmation. | **Future-dated IDs explicitly flagged INVALID** and quarantined: no "beats NAIT" wording until replaced by a real, checkable citation **or** the baseline is relabeled a faithful endpoint reimplementation (Eq. 1). **Base models pinned** to released checkpoints: primary `Qwen2.5-7B-Instruct`, secondary `Qwen2.5-1.5B-Instruct`, alternates `Llama-3.1-8B-Instruct` / `Mistral-7B-Instruct-v0.3`; checkpoint + tokenizer hashes confirmed before approval (§4.8 checklist). | §1.4, §4.8 |
| (preserved) | v3 curvature non-functionality proof, SW2 operator, submodular (1−1/e) proof, Brier propriety proof, endpoint-first deliverable order, G6 factuality gate, Gate-1b deployability gate. | **Kept** (re-numbered); only the inferential test, endpoint control, `Y` lock, ΔR² denominator, multiplicity, cost parity, matched-pair wording, and citations change. | §2 |

Note: the **NAIT endpoint baseline remains the FIRST Phase-2 code deliverable**
(§5, Deliverable #1) — it is both the *partialling-out control* `φ_end` and the
*decisive comparator*; no proposed operator is measured for "gain over
endpoints" until it exists and passes its unit tests.

---

## 1. Defect / context

### 1.1 The base method and the seed question

The base method — **Neuron-Aware Data Selection (NAIT)** — scores an instruction
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

### 1.2 The v3 review findings this RR repairs

The GPT-5.5 v3 review (`redesign_v3_gpt55.md`, score 7/10) raised five findings
and four upgrades-to-9 requirements. v4 closes each (changelog above):

1. **(a) Endpoint control too weak** — registered test rested on a 16-PC summary,
   not full `φ_end`. v4 makes the **full-`φ_end` ridge partialling-out CO-PRIMARY**
   (§2.3), with `r ∈ {8,16,32}` as sensitivity poles and a robustness floor.
2. **(b) Held-out block F-test invalid** — a nested-model `F` on train-fitted
   out-of-sample RSS has no `F` reference law. v4 replaces it with a **cross-fit +
   block-permutation + bootstrap** test (§2.4, §4.3).
3. **(c) Primary outcome `Y` under-specified** — clusters, estimator equations,
   budget, and a reliability precondition were missing. v4 **locks `Y`** and adds
   the **Y-reliability gate G7** (§2.6, §4.1, §4.4).
4. **(d) ΔR² floor denominator ambiguous** — partial-R² vs overall increment
   conflated, mapping to 0.02 unstated. v4 **separates** the two and **maps the
   overall-R² increment to the 0.02 margin** (§2.3, §4.2).
5. **(e) Multiplicity within {D, κ, joint} not pre-registered** — v4 adds
   **two-layer Holm** with the joint block as gatekeeper (§3.3, §4.5).
6. **(f) Cost-gate extraction loophole, matched-pair over-claim, citation/model
   hygiene** — v4 adds an **extraction-parity kill / trajectory-I/O accounting**
   (§2.7), **softens the matched-pair claim to a diagnostic** (§3.2), and **pins
   models (repo IDs + revisions) / verifies the NAIT citation as real** (NAIT
   `2603.13201` confirmed; only `diversity_coreset` `2605.26004` still pending)
   (§1.4, §4.8).

### 1.3 What first-moment pooling loses (preserved from v3 — NOT "endpoint-equivalent")

We never claim mean-pool equals endpoint. The endpoint feature reads two
**positions**; a per-layer trajectory **mean** averages over *all* steps and
tokens — different statistics. The defensible, modest statement is only that
**first-moment temporal pooling is insufficient**: it is permutation- and
shape-invariant and so discards the temporal/distributional structure v4
exploits.

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
  *changes* under reordering (curvature `κ`, §2.5) is therefore **not a function
  of `ψ₁`** (proof in §2.5).
- **(b) Moment blindness.** `ψ₁` ignores spread/shape; a distributional distance
  (SW2, §2.2) responds to higher moments `ψ₁` cannot see.

§1.3 does **not** claim `ψ₁ = φ_end`, nor that SW2/`κ` are *guaranteed* to beat
`φ_end` — that is the empirical question settled by the co-primary regression
(§3.1) and the matched-pair diagnostic (§3.2). The contribution must live in the
part of the trajectory that survives **partialling out the FULL `φ_end`** at the
dataset level (§2.3).

### 1.4 Citation & model hygiene (fix f3 — must-do before any must-beat claim)

- **NAIT citation is REAL (verified 2026-06-19); diversity_coreset still pending.**
  `configs/baselines/baseline_registry.yaml` lists
  `https://arxiv.org/abs/2603.13201` (NAIT, `endpoint_neuron_selection`) and
  `https://arxiv.org/abs/2605.26004` (`diversity_coreset`). **Correction (fix f3):
  arXiv 2603.13201 (NAIT) is a real, checkable paper** — present on OpenReview as
  of 2026-06-19 — so it is **no longer treated as invalid/future-dated**; the
  earlier "26xx ⇒ invalid" heuristic was wrong for this ID. NAIT therefore stays a
  **must-cite / must-beat** baseline and "better than NAIT" wording is permissible
  **only after** the fair endpoint-neuron baseline results clear the gates (§4.6),
  not blocked on citation. `endpoint_neuron_selection` remains a faithful
  endpoint reimplementation of Eq. 1, now **attributed to NAIT (2603.13201)**.
  The `diversity_coreset` id `2605.26004` is **still unverified** and stays a
  placeholder requiring a real coreset citation before its must-beat use (§4.8).
- **Pinned base models (released checkpoints, repo IDs + revisions).** Primary
  **`Qwen/Qwen2.5-7B-Instruct`** (~7.6B); secondary **`Qwen/Qwen2.5-1.5B-Instruct`**
  (~1.5B) for scale-robustness — both pinned to a **fixed HuggingFace revision
  commit + tokenizer hash** recorded in `configs/experiments/lattice_v4.yaml`
  before any server run (§4.8 checklist). Named open alternates if
  licensing/availability requires: `meta-llama/Llama-3.1-8B-Instruct` /
  `meta-llama/Llama-3.2-3B-Instruct`, or `mistralai/Mistral-7B-Instruct-v0.3`,
  each likewise revision-pinned. LoRA only; rank, steps, tokens matched across all
  methods. These replace `first_gate_open_model_pending_user_approval`; the
  config marker stays as the approval flag, but the design names concrete
  released checkpoints so the plan is instantiable. **No unreleased / dated-after
  -today checkpoint is pinned.**
- **Pinned trajectory settings.** `K = 64` SW2 random projections (seeds
  persisted); probe-layer set `|A| = 8` (an evenly spaced grid over depth, fixed
  on validation); per-example activation **subsample cap = 512** (step,token)
  vectors per layer, sampled with a persisted mask. These are the knobs the
  resolution sweep (§2.7) is allowed to move to hold the cost gate.

---

## 2. Method (equation-level, with proofs)

**Name:** **LATTICE** — *Layer-Aware Trajectory-distribuTIon seleCtion with
rEtention-adjusted gain* (working name). Dependency order mirrors the
deliverables (§5): endpoint baseline `φ_end` (control + comparator) → trajectory
operator `T(x)` → **full-`φ_end` ridge-partialling-out co-primary regression**
(the claim) → cross-fit/permutation/bootstrap inference → capacity-matched router
(ablation) → monotone-submodular selection objective → strictly-proper
calibrator (factuality gated by validation) → Y-reliability gate.

### 2.0 Notation

| Symbol | Meaning |
| --- | --- |
| `x` | candidate instruction example (prompt + target reasoning trace) |
| `L, S_x, T_{x,s}` | #layers, #decoding steps for `x`, #tokens at step `s` |
| `h_{ℓ,s,t} ∈ R^d` | residual-stream activation, layer `ℓ`, step `s`, token `t` |
| `A ⊆ [L]`, `|A|=8` | anchor/probe layer set (fixed on validation) |
| `φ_end(x) ∈ R^{2 d |A|}` | endpoint signature, Eq. 1 (Deliverable #1; the FULL control) |
| `μ_T, Q_ℓ^T` | target anchor mean / target-capability exemplar cloud at layer `ℓ` |
| `D_ℓ(x)` | per-layer SW2² to target cloud, Eq. 4 (magnitude/distributional term) |
| `κ_ℓ(x)` | per-layer temporal curvature, Eq. 5 (temporal term) |
| `T(x) ∈ R^{2|A|}` | trajectory feature vector `({D_ℓ},{κ_ℓ})`, Eq. 6 |
| `Y` | retention-adjusted fine-tuning utility outcome via LOCI (Eq. 15–17) |
| `c(x)` | nuisance covariates: length, difficulty, dataset-family (one-hot) |
| `λ_ridge` | ridge penalty for full-`φ_end` partialling-out (Eq. 8c), CV-locked |
| `m ∈ {0,1}^L` | per-example layer mask (1 = route) — ablation only |
| `B` | selected-data budget (#examples), shared by all methods |

### 2.1 Endpoint-neuron baseline (Deliverable #1 — built FIRST)

Faithful NAIT (Eq. 1): `φ_end(x) = concat_{ℓ∈A}[h_ℓ(p_start), h_ℓ(p_end)]`,
`score_end(x) = sim(φ_end(x), μ_T)` (cosine or class-mean neuron overlap).
`φ_end` plays **two** roles: the **decisive matched-budget comparator** and the
**FULL control block** in the co-primary ridge-partialling-out regression (§2.3).
No proposed operator is measured for "gain over endpoints" until
`score_end`/`φ_end` exist and pass `tests/test_endpoint_baseline.py` (§5).

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
depends on **more than the first moment**; it is *not* a function of `ψ₁` alone.
We do **not** claim "all moments," and we do **not** claim it must beat `φ_end` —
settled only by §3.1.

**(ii) Temporal curvature `κ` (uses ordering — Eq. 3 does not).** Let the
per-step mean path be `g_{ℓ,s} = mean_t h_{ℓ,s,t}`; velocity
`v_{ℓ,s} = g_{ℓ,s+1} − g_{ℓ,s}`, acceleration
`a_{ℓ,s} = g_{ℓ,s+1} − 2 g_{ℓ,s} + g_{ℓ,s−1}`. Define the **normalized bending
energy**

```
κ_ℓ(x) = (1/(S_x−2)) Σ_{s=2}^{S_x−1}  ‖a_{ℓ,s}‖ / ( ‖v_{ℓ,s}‖·‖v_{ℓ,s−1}‖ + ε ).   (Eq. 5)
```

**(iii) Cross-layer signature.**

```
T(x) = ( { D_ℓ(x) }_{ℓ∈A} , { κ_ℓ(x) }_{ℓ∈A} ) ∈ R^{2|A|}.                  (Eq. 6)
```

The raw validation-fit read-out (never used as the inferential statistic) is

```
ρ_T(x) = − Σ_{ℓ∈A} α_ℓ D_ℓ(x) − Σ_{ℓ∈A} β_ℓ | κ_ℓ(x) − κ_ℓ^T |,           (Eq. 7)
```

with `κ_ℓ^T` the target exemplars' mean curvature, `(α,β) ≥ 0`. The **test**
read-out is the cross-fit ridge-partialled coefficient of §2.3/§2.4.

### 2.3 (a, CO-PRIMARY) Full-`φ_end` ridge partialling-out — the honest endpoint control

> **Fix (a).** v3 controlled only a 16-PC endpoint *summary*. v4 controls the
> **FULL** endpoint geometry `φ_end ∈ R^{2d|A|}` via **high-dimensional ridge
> partialling-out** (so `p_Φ` ≈ `2d|A|` ≫ N is handled by regularization, not by
> dropping dimensions). The PCA-summary control is retained only as a sensitivity
> pole at `r ∈ {8,16,32}`.

Let the labeled set be `{(x_i, Y_i)}_{i=1}^N`, `Y_i` the LOCI
retention-adjusted fine-tuning utility (§2.6, §4.1). Stack design matrices
`T ∈ R^{N×2|A|}`, `Φ_full ∈ R^{N×2d|A|}` (the **full** endpoint signature, Eq. 1,
standardized), `C ∈ R^{N×p_C}` (length, difficulty, dataset-family one-hots), and
a column of ones. The control block is `Z = [Φ_full, C, 1]`.

**Ridge residual-maker (well-typed, high-dim safe).** Because `Φ_full` is wide
(`2d|A| ≫ N`), the OLS residual-maker is rank-deficient; we use the **ridge**
residual-maker

```
M_λ = I − Z ( ZᵀZ + λ_ridge·Ω )⁺ Zᵀ,   Ω = blockdiag(I on Φ_full, 0 on [C,1]).   (Eq. 8c)
```

Only the `Φ_full` block is penalized (covariates `C` and intercept enter
unpenalized). `λ_ridge` is **selected by cross-validation on the TRAIN fold only**
and **persisted** (one locked value per model/seed split; the CV grid is
pre-registered in §4.2). The trajectory coefficient is the partialled regression
of `M_λ Y` on `M_λ T`:

```
β̂_T = ( (M_λ T)ᵀ(M_λ T) )⁺ (M_λ T)ᵀ (M_λ Y).                                 (Eq. 9)
```

This is **double-ML / orthogonalization-style partialling-out**: the endpoint
signal is regressed out of *both* `Y` and `T` by a regularized learner, and the
trajectory effect is read off the orthogonalized residuals. It controls the
**full** endpoint geometry, not a 16-PC shadow.

**Partial-R² floor (fix d — correct denominator).** Let `RSS_full`, `RSS_red` be
the residual sums of squares of the full vs endpoint-only (reduced) cross-fit
predictors on held-out data. The **registered residual quantity** is the
**partial-R²**

```
partial_R²_T = (RSS_red − RSS_full) / RSS_red                                (Eq. 10a)
```

— the fraction of the **endpoint-controlled residual variance** explained by the
trajectory block (denominator = `RSS_red`, the residual variance *after*
endpoints). The **overall-R² increment** is reported **separately** and is the
quantity **mapped to the 0.02 deployment margin**:

```
ΔR²_overall = (RSS_red − RSS_full) / TSS.                                    (Eq. 10b)
```

**Deployment-margin map (fix d):** a `ΔR²_overall ≥ 0.02` corresponds to the
same 0.02 adjusted-score margin scale used by the trajectory–endpoint gate
(`first_gate.yaml`); the residual claim uses `partial_R²_T ≥ floor_partial`
(§4.2), the deployment relevance uses `ΔR²_overall ≥ 0.02`. Both floors are
locked; an effect that clears the partial-R² residual floor but not the 0.02
overall-increment is reported as "real but deployment-immaterial."

**Inference is OOS-valid (fix b, §2.4):** all p-values come from the cross-fit /
permutation / bootstrap machinery, never from a nested-model `F` reference law.

### 2.4 (b) Valid out-of-sample inference — cross-fit + block permutation + bootstrap

> **Fix (b).** A classical nested `F` (Eq. 10, v3) is invalid on train-fitted
> out-of-sample RSS. v4 locks a **resampling test**. The nested `F` survives only
> as an in-fold descriptive number, never as the inferential p-value.

1. **Cross-fit estimation (K = 10).** Partition examples into 10
   family-stratified, example-disjoint folds (indexed by the 20 shared seeds).
   For each fold `k`: fit `M_λ`, `λ_ridge`, PCA loadings, and the partialled
   regression on the **other 9 folds**; compute the held-out partialled
   statistic (`β̂_T`, `partial_R²_T`, `ΔR²_overall`) on fold `k`. Aggregate by
   **median-of-folds** point estimate (robust to a single unstable split).

2. **Block permutation test (exact null "T adds nothing given controls").**
   Under H₀, the trajectory block `T` is exchangeable conditional on the
   controls. Permute the **rows of `M_λ T`** (the endpoint-orthogonalized
   trajectory residuals) **within family×fold strata** `P = 5000` times; recompute
   the cross-fit statistic each time; the permutation p-value is the right-tail
   rank of the observed statistic. This is an **exact, distribution-free** test of
   the registered null — no `F` law assumed.

3. **Cluster/stratified bootstrap CI.** Resample **clusters** (the §4.1 LOCI
   clusters, to respect within-cluster dependence) with replacement `B_boot =
   2000` times; recompute the cross-fit statistic; report the **BCa 95% CI**.

**Decision (locked):** the residual trajectory claim rejects H₀ iff **(permutation
p < 0.05 after the two-layer Holm of §4.5)** AND **(bootstrap-BCa 95% CI for
`partial_R²_T` excludes `floor_partial`)**. The co-primary status (fix a) further
requires this to hold under the **full-ridge** control AND be **robustness-stable**
across `r ∈ {8,16,32}` (§4.2).

### 2.5 (preserved) `κ` is provably not a function of `ψ₁`

**Proposition.** There exist trajectories with identical `ψ₁` but different `κ_ℓ`.
**Proof.** Take two step-sequences that are reorderings of each other: `ψ₁` (a
mean over the multiset) is invariant to the permutation, but `κ_ℓ` (Eq. 5)
depends on the *ordered* second differences `a_{ℓ,s}`, which change under
reordering whenever the path is not collinear. Hence `κ_ℓ` is not a function of
`ψ₁`. ∎ (This is the statistic the §3.2 diagnostic is built on.)

### 2.6 Drift estimators, calibrator (factuality gated, G6), Y-reliability (G7)

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
population risk is **uniquely** minimized at `q=p`; expectation over contexts
preserves strictness. ∎ This proves **calibration, not proxy validity**.

> **Precondition gate G6 (factuality, preserved).** Before `λ_f` may be non-zero,
> require on a held-out slice: **Spearman ρ(f̂, drift_eval) ≥ 0.3 AND Pearson
> r ≥ 0.3 with lower 95% CI > 0**, *and* reliability ECE ≤ 0.1. **Fail ⇒
> `λ_f := 0`**, no factuality/safety claim. Thresholds locked (§4.4).

> **Precondition gate G7 (Y-reliability, fix c — NEW).** Before the primary
> regression (§2.3) is run on the LOCI influence proxy `Y`, require both:
> **(i) Test–retest / seed stability.** Recompute `Y` under ≥3 independent seeds
> (cluster assignment + training seed); require **intraclass correlation
> ICC(2,1) ≥ 0.6** across seeds (per-example `Y` is stable, not seed noise).
> **(ii) Proxy↔ground-truth validity.** On a **pre-registered subset** of
> `n_sub = 60` clusters, compare the LOCI influence proxy `Y` to the **actual
> held-out fine-tuning delta** `Δ_retrain` from genuinely retraining with vs
> without that cluster (Eq. 16); require **Spearman ρ(Y, Δ_retrain) ≥ 0.3 with
> lower 95% CI > 0**. **Fail G7 ⇒** the primary regression is **not** run on the
> influence proxy; it falls back to the **direct retrain-delta `Y`** on the
> subset, reported as **underpowered-diagnostic** — no primary claim is made on
> an unreliable proxy. Thresholds locked (§4.4).

### 2.7 (f1) Cost model with integrity — extraction-parity kill + trajectory-I/O accounting

> **Fix (f1).** v3 Gate-1 assumed a "shared extraction" both methods pay. But
> endpoint-only NAIT need not extract full step/token/layer traces, so the
> shared-extraction premise can be false and the differential extraction/storage
> burden hidden. v4 closes the loophole two ways.

- **Default: extraction-parity (pre-registered).** Both methods are **forced to
  consume the identical full-trace extraction**, justified because the matched-
  budget protocol fixes a single extraction artifact for *all* selectors (fair-
  comparison requirement). Under parity, extraction truly cancels and Gate-1 is
  the per-pool *selection* increment only.

- **Extraction-parity KILL condition.** If the experiment cannot honestly force
  parity — i.e. endpoint-only could run on a strictly cheaper endpoint-only
  extraction — then the **differential trajectory extraction + storage I/O is
  counted INSIDE Gate-1's 2× multiplier**:
  `mult₁ = ( cost_extract_traj_diff + cost_select(LATTICE) ) / cost_select(NAIT)`.
  If `mult₁ > 2.0×` after the resolution sweep, the method is **declared
  high-cost-analysis** (`failure_action: require_efficiency_ablation`), **not** a
  deployment method, and the "practical cost" claim is forbidden.

- **Gate-1 (per-pool selection).** `cost_select` counts the incremental
  per-example selection work: SW2 (`K·n log n`), `κ` (Eq. 5), the ridge-
  partialled regression scoring, drift estimators, greedy coverage updates. This
  is the existing `passes_cost_gate` multiplier; threshold **2.0× unchanged**.

- **Amortized validation cost (Gate-1b, preserved).** Causal layer importance
  `I_{c,ℓ}` (Eq. 13), diagonal Fisher `F̄_ℓ`, calibrator + G6/G7 diagnostics,
  hparam fit. Paid **once** per (model, validation split). Gate-1b:
  `mult_dep = ( cost_amortized/R + cost_select(LATTICE) ) / cost_select(NAIT)`,
  with conservative default **`R = 1`** and reported break-even **`R*`**.

The honest cost table (§4.6) now shows **three** lines: per-pool selection,
**differential trajectory extraction/storage**, and amortized validation.

### 2.8 (preserved) Monotone-submodular selection objective — earned (1−1/e)

Per-example utility (modular, non-negative):
`u(x) = max(0, ĝ(x) − λ_r r̂(x) − λ_f f̂(x))`, where `ĝ(x)` is the
endpoint-residualized read-out (§2.3, centered non-negative by a constant shift),
reusing `metrics.retention_adjusted_gain` as the `ĝ − λ_r r̂` kernel; the
factuality term is added with `λ_f = 0` unless G6 (§2.6) passes.

Facility-location coverage over the trajectory-feature space, with
`σ(x,e) ∈ [0,1]` a similarity to ground element `e` (target-concept exemplar; `E`
the held-out exemplar set):

```
C(S) = Σ_{e∈E} w_e · max_{x∈S} σ(x,e);   F(S) = Σ_{x∈S} u(x) + μ·C(S),  μ≥0.   (Eq. 14)
```

**Proposition (NWF (1−1/e)).** `F` is monotone non-decreasing and submodular;
greedy (`argmax_x [F(S∪{x})−F(S)]` until `|S|=B`) returns `S_greedy` with
`F(S_greedy) ≥ (1−1/e)·F(S*)`. **Proof.** (1) `Σ u(x)`, `u≥0`, is modular and
monotone. (2) For each `e`, `S↦max_{x∈S}σ(x,e)` is monotone and submodular (its
marginal `max(0, σ(x,e) − running max)` is non-increasing in the running max,
which is non-decreasing in the set); a non-negative weighted sum of monotone-
submodular functions is monotone-submodular, so `C` is. (3) A non-negative
combination of monotone-submodular functions is monotone-submodular, so `F` is;
NWF applies. ∎ Greedy is `O(B·|pool|·|E|)` and stays inside Gate-1. (If a future
variant uses a non-submodular term, the guarantee is dropped and greedy is
reported as a heuristic; the shipped form keeps the proof.)

### 2.9 (B-ablation) Capacity-matched layer router (not load-bearing)

Per-example LoRA layer gating changes the trainable footprint, so it is a
**validated ablation** only; the main claim uses **uniform** LoRA. Layer-function
profile (validation-only, frozen, hashed):

```
I_{c,ℓ} = ( Acc_c(model) − Acc_c(model | h_ℓ ← h̄_ℓ) ) / Acc_c(model).        (Eq. 13)
```

Routing `m_ℓ(x) = 1[ I_{c(x),ℓ}·w_ℓ(x) ≥ τ ]`. **Capacity-matching contract:**
fix total LoRA rank `R_tot = |A|·r_unif` and **reallocate** it across routed
layers (`r_ℓ = R_tot/|{ℓ:m_ℓ=1}|`, clipped) so active params, optimizer-state
slots, FLOPs, and effective rank match the uniform baseline; any variant that
cannot match exactly is reported as **capacity-unmatched diagnostic**, never a
matched-budget claim. The router is kept only if it helps at matched capacity in
≥2 capability families (G4).

### 2.10 Versioned, auditable signature record (extended additively)

`TrajectorySignatureV2` (`schema_version="2.0.0"`, added additively in
`src/neurotrace_it/schemas_v2.py`) **persists the estimands**: `example_id`,
`layer_ids (A)`, `endpoint_signature` (Eq. 1, the FULL control), `D = {ℓ:D_ℓ}`,
`kappa = {ℓ:κ_ℓ}`, `projection_seeds` (K), `slice_masks`,
`alignment_scores {ρ_T, residual β̂_T, λ_ridge}`, `selection_scores {u(x),
marginal_gain}`, `drift_estimates {r̂, f̂}`, `calibrator_provenance {rule:"brier",
c*, reliability_hash, G6_pass:bool}`, `y_reliability {ICC, rho_proxy_retrain,
G7_pass:bool}`, `cluster_assignment_hash`, `layer_router_outputs {m(x),
I_profile_hash, R_tot, r_per_layer}`, and `trajectory_hash` = `H(all of the
above)` as an **integrity check that recomputes from the stored estimands**.
`validate_selection_manifest_v2` requires non-empty `D`/`kappa` over `layer_ids`,
present `endpoint_signature`, present `projection_seeds`, and a recomputing hash.
**The v4 fields (`λ_ridge`, `y_reliability`, `cluster_assignment_hash`) are
additive optional fields — V1/V2 records stay valid.** `server_authorized` stays
`false`; `endpoint_neuron_selection` stays required.

### 2.11 Algorithm box

```
Algorithm LATTICE-Select v4  (design-only; no server run)
Inputs : pool X (budget B), probe layers A (|A|=8), target clouds {Q_ℓ^T}, concept set E,
         general/Fisher set G, layer-function profile I_{c,ℓ}, K=64 projections+seeds,
         weights (α,β,λ_r,λ_f,μ,τ,r,λ_ridge) fit on TRAIN folds only

# Phase -1 : ENDPOINT BASELINE FIRST (Deliverable #1) — FULL control + comparator
  for x in X: φ_end(x) ← Eq.1 ; score_end(x) ← sim(φ_end, μ_T)

# Phase 0 (once, validation, AMORTIZED): layer fn + Fisher + calibrator + G6 + G7
  I_{c,ℓ} ← Eq.13 ; F̄_ℓ ← diag Fisher on G ; fit Brier calibrator ; run G6 → set λ_f
  build LOCI clusters (Eq.15) ; compute Y (Eq.16-17) ; run G7 (ICC + proxy↔retrain)
  if G7 fails: route primary to direct-retrain-delta Y on subset (diagnostic)

# Phase 1 : per-example trajectory signatures (PER-POOL)
  for x in X: D_ℓ(x) ← Eq.4 ; κ_ℓ(x) ← Eq.5 ; T(x) ← Eq.6

# Phase 2 : CO-PRIMARY regression (full-φ_end ridge partialling-out, cross-fit)
  for k in 1..10 folds:
     fit M_λ (Eq.8c, λ_ridge CV on other 9), PCA_r loadings (r∈{8,16,32}), β̂_T (Eq.9) on train
     compute held-out partial_R²_T (Eq.10a), ΔR²_overall (Eq.10b) on fold k
  median-of-folds estimate ; block-permutation p (P=5000) ; cluster-bootstrap BCa CI (B=2000)
  robustness: require stability across r AND full-ridge control (§4.2)

# Phase 3 : drift + monotone-submodular selection (PER-POOL)
  for x in X: r̂(x) ← Eq.11 ; f̂(x) ← §2.6 ; u(x) ← max(0, ĝ − λ_r r̂ − λ_f f̂)
  S ← greedy argmax of F(S)=Σ u(x) + μ·C(S)   # Eq.14 ; (1−1/e) by §2.8

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

### 3.1 The single crisp contribution (CO-PRIMARY: full-`φ_end` partialling-out + matched-pair diagnostic)

> **There exists a temporal/distributional component of the reasoning trajectory
> — captured by curvature `κ` and distributional alignment `D` — that, AFTER
> partialling out the FULL endpoint signature `φ_end` (high-dim ridge) and
> length/difficulty/dataset-family, still carries information predictive of
> retention-adjusted fine-tuning utility, at matched data, parameter, and compute
> budget.**

Formally (fix a, b): the **cross-fit, block-permutation test** of "T adds nothing
given the full `φ_end` control" rejects (Holm-corrected per §4.5), with
`partial_R²_T` BCa-CI above its floor and `ΔR²_overall ≥ 0.02` for deployment
relevance (fix d). **Co-primary** = this must hold under the **full-ridge**
control AND be **robustness-stable across `r ∈ {8,16,32}`** (§4.2) AND be
**corroborated by the matched-pair diagnostic** (§3.2). One claim from one
quantity `T(x)=({D_ℓ},{κ_ℓ})`: selection score (§2.2), router (§2.9, ablation),
retention penalty (§2.6), coverage (§2.8) are all functions of the same
`D_ℓ/κ_ℓ`. Because the gain is the **partial** signal over the **full** endpoint
geometry, it cannot be an artifact of `T(x)` re-encoding `φ_end` — that variance
is orthogonalized out by ridge-FWL. That shared origin plus the partialling-out
is what makes it a method, not a stitch.

### 3.2 Matched-endpoint / divergent-curvature DIAGNOSTIC (fix f2 — not causal proof)

> **Prediction P (diagnostic).** Among **naturally occurring** pool pairs
> `(x, x')` with near-identical endpoint signatures
> `‖φ_end(x)−φ_end(x')‖₂ ≤ τ_end` but **divergent curvature** `|κ(x)−κ(x')|` in
> the top decile, LATTICE predicts **measurably different retention-adjusted
> fine-tuning outcomes** (paired difference > the 0.02 trajectory–endpoint
> margin). Matched endpoints make the difference **consistent with residual
> trajectory information after observed-covariate matching** — a **strong
> corroborating diagnostic**, NOT standalone causal proof: pair mining can still
> confound on length, difficulty, reasoning style, annotation quality, or pool
> artifacts.

**Concrete mining recipe (no synthetic construction).**
1. From each real instruction pool (math, code, multi-hop QA), compute `φ_end`
   and `κ` for all candidates (reusing Phase-1 outputs).
2. **Bucket** by capability family and by coarse final-answer equivalence
   (exact-match answer / normalized output / AST-equal code) so paired examples
   teach "the same endpoint."
3. Within a bucket, retrieve nearest neighbors in `φ_end` (cosine / `ℓ₂`) and
   keep pairs with **`‖φ_end(x)−φ_end(x')‖₂ ≤ τ_end`**, `τ_end` = the **1st
   percentile** of within-bucket pairwise endpoint-distance (persisted).
4. Among those, **keep pairs whose `|κ(x)−κ(x')|` is in the top decile**.
5. **Covariate balancing (fix f2):** additionally apply **coarsened exact
   matching** on length/difficulty/family so the paired set is balanced on
   observed confounders; **report the residual standardized-mean-difference
   imbalance** for each covariate. Pairs that cannot be balanced are dropped and
   the achieved `n` re-powered.
6. **Target `n = 300`** retained pairs (≈100 per family); if a family yields
   fewer, report achieved `n` and re-run power.

**Outcomes.** P **corroborates** ⇒ residual trajectory information is consistent
with a mechanism endpoints cannot see (strengthens §3.1). P **fails** (matched,
balanced pairs train identically) ⇒ residual likely carries nothing ⇒
`failure_action: stop_main_novelty_claim`, downgrade to diagnostic. **P is never
the sole basis for the causal claim** — it must agree with §3.1.

### 3.3 Pre-registered `κ`-only vs `D`-only contingency + multiplicity (fix d, e)

The residual could be carried **entirely by `D` (magnitude)**, collapsing the
temporal-curvature novelty. The three block tests {**joint** (T), **D-only**,
**κ-only**} are corrected by **Holm within this trajectory family** (3 tests,
fix e), and the **joint test is the gatekeeper**: if the Holm-adjusted joint
block fails, D-only/κ-only are **exploratory only**. The locked 2×2 table reads
the **Holm-adjusted** significances on the held-out cross-fit statistic:

| `D`-only sig (Holm)? | `κ`-only sig (Holm)? | Decision (locked) | Allowed claim |
| --- | --- | --- | --- |
| yes | yes | full claim | "distributional **and** temporal-curvature trajectory signal beyond endpoints" |
| yes | no | **fallback** | "**distributional (magnitude)** trajectory signal beyond endpoints"; **curvature novelty dropped** |
| no | yes | curvature-only | "temporal-curvature trajectory signal beyond endpoints"; magnitude demoted |
| no | no | **kill** | no trajectory-beyond-endpoint claim; `stop_main_novelty_claim` |

(The table is consulted **only if** the joint gatekeeper passes.) Fallback
wording is fixed **now** so the contingency cannot be rationalized post hoc.
`κ`-only and `D`-only run as registered ablation poles (§4.3).

### 3.4 Predictions summary (all falsifiable, pre-data)

- **P1 (co-primary):** cross-fit/permutation test of "T adds nothing | full
  `φ_end`" rejects (Holm, §4.5) with `partial_R²_T` BCa-CI above floor, stable
  across `r∈{8,16,32}` (§4.2). *Falsifier:* non-significant, CI below floor, or
  unstable across `r`.
- **P1' (deployment relevance):** `ΔR²_overall ≥ 0.02` (fix d). *Falsifier:* < 0.02.
- **P2 (mechanism diagnostic):** matched-endpoint/divergent-`κ`,
  covariate-balanced paired difference > 0.02 (§3.2). *Falsifier:* ≤ 0.02 (kills
  only in agreement with P1).
- **P3 (aggregate):** trajectory-selected − endpoint-selected adjusted gain
  ≥ 0.02 margin and ≥ 0.03 relative, retention drift disadvantage ≤ 0.01,
  hallucination drift ≤ 0.01 (existing G1–G3).
- **P4 (router, ablation):** helps in ≥2 families at matched capacity (G4), else
  appendix-only.
- **P5 (cost):** Gate-1 `mult₁ ≤ 2.0×` **including differential trajectory
  extraction/storage** under the extraction-parity rule (§2.7); Gate-1b reported
  with `R=1` and `R*`.
- **P6 (Y-reliability):** G7 passes (ICC ≥ 0.6, proxy↔retrain ρ ≥ 0.3); else the
  primary regression is not run on the proxy (§2.6, §4.4).

---

## 4. Pre-registered analysis plan + kill-gates + power

`server.authorized` stays **false**. Everything below is a locked plan; no
command is executed. This is the Stage-1 analysis lock.

### 4.1 Outcome `Y` — FULLY LOCKED (fix c)

- **Definition.** Per-example **retention-adjusted fine-tuning utility**,
  `Y_i = retention_adjusted_gain(target_gain_i, retention_drift_i, drift_weight)`
  (existing metric, `metrics.py`, reused unchanged), attributed to example `x_i`
  by a **leave-one-cluster-in (LOCI)** influence estimator.

- **Cluster construction (locked).** Embed each candidate by its endpoint
  signature `φ_end` (L2-normalized); target **`K_clusters` ≈ N/200** with a
  **size floor of 25**; singletons/noise points are assigned to their nearest
  cluster centroid. The **cluster assignment and its hash are persisted**
  (`cluster_assignment_hash` in §2.10). Clustering is fit on the **train fold
  only**; held-out examples are assigned by nearest centroid (no leakage).

  **The LOCKED clustering method is the deterministic agglomerative surrogate**
  implemented in `analysis/outcome_y.py::build_loci_clusters` (furthest-point
  k-center seeding + Lloyd refinement + size-floor dissolution). It is locked —
  rather than HDBSCAN — to keep the pipeline pure-stdlib, fully deterministic, and
  zero-extra-dependency under build-now / run-later; it honours the identical §4.1
  contract (size floor 25, noise/singletons reassigned to the nearest centroid,
  persisted assignment hash). The surrogate's parameters are persisted in
  `LociClustering.params` alongside the HDBSCAN parameters for provenance.

  **HDBSCAN is an OPTIONAL drop-in** (`min_cluster_size = 25`, `min_samples = 10`,
  `metric = euclidean`, `cluster_selection_method = eom`) selectable *only* if the
  `hdbscan` dependency is available at run authorization; the persisted
  `HDBSCAN_PARAMS` make that swap one-to-one. Switching to HDBSCAN is a
  pre-registered alternate, not the default, and does not change the locked
  contract above.

  ```
  Clusters = AgglomerativeSurrogate(φ_end ; size_floor=25, K≈N/200).fit(train)  (Eq. 15)
           # optional drop-in: HDBSCAN(φ_end ; min_cluster_size=25, min_samples=10)
  ```

- **LOCI influence estimator (locked equations).** For cluster `g` containing
  `x_i`, the leave-one-cluster-in utility is the held-out validation-loss change
  from including vs excluding cluster `g` in the selected set, attributed to its
  members by a first-order influence approximation (so we need **one** reference
  fit, not one-fit-per-cluster):

  ```
  Δ_g  = L_val(θ̂_{−g}) − L_val(θ̂)                                            (Eq. 16)
  Y_i  = + ( |g|⁻¹ · Δ_g ) · drift_adjust_i,   x_i ∈ g                        (Eq. 17)
  ```

  `Y` is a **utility** (higher = more useful): removing a useful cluster `g`
  *raises* the held-out validation loss, so `Δ_g > 0` and the member utility
  `Y_i > 0`. (Eq. 17 was sign-corrected from `−` to `+` so that a useful cluster
  receives positive utility; the old `−` mapped a useful cluster to negative `Y`.)

  where `θ̂` is the reference LoRA fit on the full selected pool, `θ̂_{−g}` is the
  influence-approximated parameter with cluster `g` down-weighted (TracIn /
  influence-on-validation as used by `influence_gradient_selection`, applied
  **identically to all methods**), and `drift_adjust_i` folds in the
  retention/hallucination penalty consistent with `retention_adjusted_gain`. The
  **exact attribution procedure is locked to the `influence_gradient_selection`
  baseline implementation** so it is method-neutral.

- **Training/eval budget (locked).** Reference fit: LoRA rank `R_tot` (matched
  across methods), **fixed steps `S_train` and token budget `T_tok`** per the
  matched-budget protocol (§4.9); validation loss `L_val` on the frozen held-out
  split; eval items per `data_and_evaluation_plan.md`. The budget is **identical**
  for the reference fit, the G7 retrain subset, and every baseline.

- **Datasets (reuse `data_and_evaluation_plan.md`):** math IT, code IT, multi-hop
  QA IT (frozen split hash + contamination audit); held-out evals: math EM, code
  pass-rate, multi-hop QA with distractors; retention: MMLU-style aggregate +
  general IF; hallucination: TruthfulQA + FActScore-style atomic-claim factuality
  (also the G6 ground truth).

- **Models (pinned, fix f3):** `Qwen2.5-7B-Instruct` (primary),
  `Qwen2.5-1.5B-Instruct` (secondary); alternates per §1.4. LoRA only; matched
  rank/steps/tokens.

### 4.2 CO-PRIMARY analysis (fix a, b, d) — locked

Fit the **full-`φ_end` ridge partialling-out** (Eq. 8c–9) and PCA-`r` poles
within the **10-fold cross-fit** (§2.4); compute the held-out **partial-R²**
(Eq. 10a) and **overall-R² increment** (Eq. 10b); test by **block permutation**
(P=5000) and **cluster bootstrap** (B=2000). **Locked floors:**
- residual claim: **permutation p < 0.05 after two-layer Holm** (§4.5) AND
  **BCa-95% CI of `partial_R²_T` excludes `floor_partial = 0.02`**
  (partial-R² scale; the fraction of endpoint-residual variance);
- deployment relevance: **`ΔR²_overall ≥ 0.02`** (mapped to the 0.02 margin gate,
  fix d);
- **robustness floor (fix a):** the trajectory effect must (i) survive the
  **full-ridge** control and (ii) be **monotone-stable across `r ∈ {8,16,32}`**
  (no sign flip; overlapping BCa CIs). If full-ridge passes but a PCA pole
  disagrees, the result is reported as **control-sensitive** (weaker tier), not a
  clean co-primary pass.

`λ_ridge` CV grid (locked): `{1e-2, 1e-1, 1, 10, 100} × σ²_Φ` selected by 5-fold
CV on each train fold; the selected value persisted per split. `Φ_full` = full
Eq.1 signature (standardized); `C` = length, difficulty, dataset-family. Folds
example-disjoint, family-stratified, indexed by the 20 shared seeds. **Leakage
rule locked:** train-estimated `M_λ`, `β̂`, PCA loadings, cluster centroids,
`λ_ridge` applied out-of-sample; no in-fold fit is reused as a test statistic; no
nested-`F` p-value is used inferentially.

### 4.3 Baselines & ablations (reuse `baseline_registry.yaml`)

`random_subset`, `full_data_it` (upper bound), `quality_score_selection`,
`diversity_coreset`, `influence_gradient_selection`,
**`endpoint_neuron_selection` (NAIT, Deliverable #1; FULL control + decisive
comparator)**, `layer_selective_no_trajectory`. **Ablations:** endpoint-only;
**`D`-only (no κ)**; **`κ`-only (no D)** (§3.3 poles); full-ridge vs PCA-`r`
control (§4.2); router on/off at matched capacity (§2.9); `λ_r=0`; `λ_f=0`
(default unless G6); coverage `μ=0`. The **nested-model F** (v3 Eq. 10) is
reported as a **descriptive in-fold diagnostic only**, explicitly NOT an
inferential p-value.

### 4.4 Secondary/contingency gates

- **§3.3 2×2 decision table** evaluated on the held-out cross-fit statistic with
  **Holm within {D, κ, joint}** (fix e), joint as gatekeeper.
- **G6 factuality precondition:** Spearman ρ ≥ 0.3 **and** Pearson r ≥ 0.3 (lower
  95% CI > 0) between `f̂` and eval factuality-drift, **and** ECE ≤ 0.1. **Fail ⇒
  `λ_f := 0`**, no factuality/safety claim.
- **G7 Y-reliability precondition (fix c):** ICC(2,1) ≥ 0.6 across ≥3 seeds **and**
  Spearman ρ(Y, Δ_retrain) ≥ 0.3 (lower 95% CI > 0) on `n_sub = 60` clusters.
  **Fail ⇒** primary not run on proxy; fall back to direct retrain-delta `Y`
  (underpowered-diagnostic).

### 4.5 Multiplicity correction (fix e) — locked

Two layers of Holm, family-wise α = 0.05:
1. **Within the trajectory family** {joint-T, D-only, κ-only}: Holm over **3**
   block tests. The **joint test is the gatekeeper** — D-only/κ-only are
   interpreted only if joint passes; otherwise they are exploratory.
2. **Across metric families** {target, retention, hallucination, layer, cost}:
   Holm as in `statistical_analysis_plan.md`.
The two corrections compose (within-family first, then across-family on the
gatekeeper p-values). No "test three, claim the winner" — the registered primary
is the **joint** block.

### 4.6 Kill-gates (decisive, pre-registered) + honest THREE-line cost table

1. **Co-primary residual kill (P1, §4.2):** cross-fit/permutation test
   non-significant (Holm) *or* `partial_R²_T` BCa-CI below floor *or* unstable
   across `r` (full-ridge fails) ⇒ **stop main novelty claim**.
2. **Deployment-relevance kill (P1', fix d):** `ΔR²_overall < 0.02` ⇒ effect
   labeled "real but deployment-immaterial," no deployment claim.
3. **Mechanism diagnostic (P2, §3.2):** covariate-balanced matched-pair
   difference ≤ 0.02 ⇒ contributes to `stop_main_novelty_claim` **only in
   agreement with P1** (diagnostic, not standalone).
4. **Aggregate kill (existing G1–G5):** trajectory_adjusted − endpoint_adjusted
   < 0.02, or target/adjusted relative gain < 0.03, or retention drift
   disadvantage > 0.01, or hallucination drift > 0.01, or **Gate-1** `mult₁ >
   2.0×` (incl. differential trajectory extraction, §2.7) ⇒ downgrade/forbid per
   each `failure_action`. Router must help ≥2 families at matched capacity (G4).
   Paper tier needs ≥20 seeds (G5).
5. **Contingency (D/κ, §3.3):** Holm-adjusted 2×2 selects the allowed claim/fallback.
6. **Factuality precondition (G6):** fail ⇒ `λ_f:=0`.
7. **Y-reliability precondition (G7, fix c):** fail ⇒ primary not run on proxy.
8. **Extraction-parity kill (f1, §2.7):** if differential trajectory
   extraction/storage pushes honest `mult₁ > 2.0×` and the sweep cannot reduce it
   ⇒ **high-cost-analysis**, no deployment claim.

**Honest THREE-line cost table (fix f1):** (i) per-pool incremental selection;
(ii) **differential trajectory extraction + storage I/O** (counted inside Gate-1
unless honest extraction-parity holds); (iii) amortized validation (causal `I`,
Fisher, calibrator + G6/G7, hparam fit) — Gate-1b with `R=1` and break-even `R*`.

### 4.7 Pre-registration amendments (fix b, c, e, f1 — recorded)

> **AMENDMENTS to `docs/pre_registration.md` (design-only, additive — do NOT edit
> the file in this RR; recorded here for the locked plan):**
> 1. **Inference:** the primary OOS test is **cross-fit + block-permutation +
>    cluster-bootstrap**, not a nested-model `F`. The nested `F` is descriptive
>    only.
> 2. **Endpoint control:** the registered control is the **full-`φ_end` ridge
>    partialling-out** (co-primary), with `r∈{8,16,32}` PCA poles as robustness.
> 3. **Outcome `Y`:** locked per §4.1 (HDBSCAN clusters, Eq. 15–17, fixed
>    budget) and gated by **G7**.
> 4. **ΔR² floor:** partial-R² (Eq. 10a) for the residual claim; overall-R²
>    increment (Eq. 10b) mapped to the 0.02 margin.
> 5. **Multiplicity:** two-layer Holm with the **joint** block as gatekeeper.
> 6. **Cost:** Gate-1 includes **differential trajectory extraction** unless
>    extraction-parity is honestly forced; **extraction-parity kill** added.
> 7. **Gate-1b (deployability):** unchanged from v3 (`R=1` default, `R*`
>    reported).

### 4.8 Citation/model verification checklist (fix f3 — blocking before must-beat)

- [x] `endpoint_neuron_selection` `paper_url` **verified real**:
  `arxiv.org/abs/2603.13201` (NAIT) is a checkable paper (OpenReview, confirmed
  2026-06-19). Keep the citation; the baseline is a faithful Eq. 1 endpoint
  reimplementation **attributed to NAIT**. "beats NAIT" wording is gated on the
  fair-comparison gates (§4.6), **not** on citation validity.
- [x] **FREEZE — `diversity_coreset` DROPPED FROM MUST-BEAT.** `arxiv.org/abs/
  2605.26004` is **still unverified**; rather than block on it, the baseline is
  demoted to a **non-must-beat** secondary/contextual baseline
  (`must_beat: false`, `citation_status: unverified_2026-06-19` in
  `configs/baselines/baseline_registry.yaml`). No "beats diversity_coreset"
  claim is licensed; replace with a verified coreset citation **(pre-registered;
  pin exact revision/hash at run authorization)** if it is to be promoted back.
- [x] **FREEZE — base models pinned (pre-registered).** `Qwen/Qwen2.5-7B-Instruct`
  (primary) + `Qwen/Qwen2.5-1.5B-Instruct` (secondary), **Apache-2.0**, the
  two-size design; released checkpoints only. `repo_id` + `license` recorded in
  `configs/experiments/lattice_v4.yaml`; `revision` / `checkpoint_hash` /
  `tokenizer_hash` carry the literal note **"pre-registered; pin exact
  revision/hash at run authorization"** (`server.authorized` stays false).
- [x] **FREEZE — eval sets pinned (pre-registered).** Candidate pool =
  MetaMathQA (math) + a permissive code-instruct set + a multi-hop instruction
  set; **retention eval = held-out MMLU**; **hallucination eval = TruthfulQA +
  FActScore-style** (also G6 ground truth). Recorded under `candidate_pool` /
  `retention_eval` / `hallucination_eval` in `lattice_v4.yaml`; each labeled
  **"pre-registered; pin exact revision/hash at run authorization"**.
- [ ] Persist SW2 settings (`K=64`, `|A|=8`, subsample 512), PCA `r∈{8,16,32}`,
  `λ_ridge` grid, HDBSCAN params, and the G7 thresholds in
  `configs/experiments/lattice_v4.yaml` (added later, additively).

### 4.9 Matched-budget protocol (reuse `baseline_contract.md` + `pre_registration.md`)

Same pool, budget `B`, base model + tokenizer, **same LoRA rank** (`R_tot`), same
steps/tokens, validation-only selection, same evaluator, **shared seeds 0..19**
(`configs/seeds/paper_20.txt`). Paired tests on matched eval items; two-layer
Holm (§4.5); effect sizes + 95% CIs (`docs/statistical_analysis_plan.md`). Router
ablation only at matched capacity; any unmatched variant is diagnostic-labeled.
**Extraction-parity (fix f1):** all selectors consume the identical full-trace
extraction artifact, or the differential is charged to Gate-1.

---

## 5. Code plan for Phase-2 (additive; deliverable order enforced) — **IMPLEMENTED & FROZEN**

All changes are **additive**; nothing existing is modified or deleted.
`schemas.py`, `metrics.py`, `baseline_registry.yaml`, and gate configs are reused
as-is. Public APIs are **extended, not redefined**. **Deliverable #1 (endpoint
baseline) is built and tested before any proposed-method operator.**

**Status (2026-06-19): the modules below are now IMPLEMENTED and frozen** as
pure-stdlib build-now/run-later code (no model load, no server call, no training);
unit/contract tests live under `tests/`. The list is retained as the deliverable
manifest; "to ADD later" now reads "ADDED (frozen)". The remaining work is the
**deferred authorized GPU run** only (`server.authorized: false`).

**Reused unchanged:** `src/neurotrace_it/metrics.py` (`retention_adjusted_gain`
kernel; `passes_drift_gate`/`passes_cost_gate` gates — signatures untouched);
`src/neurotrace_it/schemas.py` (V1 records stay valid, back-compatible);
`configs/baselines/baseline_registry.yaml`, `configs/experiments/*.yaml`,
`configs/seeds/paper_20.txt`, `schemas/selection_manifest.schema.json`.

**ADDED (frozen; dependency order — paths as implemented):**
- **Deliverable #1 — `src/neurotrace_it/endpoint_baseline.py`** —
  `endpoint_signature` (Eq. 1), `endpoint_score` (NAIT similarity). **Built and
  tested first.** `tests/test_endpoint_baseline.py` — faithful-endpoint contract
  + similarity tests.
- **[DONE]** `src/neurotrace_it/schemas_v2.py` — `TrajectorySignatureV2` (§2.10) +
  `validate_selection_manifest_v2` (additive; does not touch `schemas.py`; the
  v4 fields `λ_ridge`, `y_reliability`, `cluster_assignment_hash` are optional).
- **[DONE]** `src/neurotrace_it/trajectory.py` — `sliced_wasserstein2` (Eq. 3–4,
  returns value **and** seeds), `trajectory_curvature` (Eq. 5),
  `trajectory_signature`, plus `rbf_mmd2` robustness pole.
- **[DONE]** `src/neurotrace_it/analysis/residual_test.py` +
  `analysis/residualize.py` — `build_endpoint_control` (full `Φ_full`
  + PCA-`r` poles), `ridge_partial_out` / `dual_ridge_partial_out` (Eq. 8c–9,
  `n`-space dual for 7B-scale width), `cross_fit_partial_r2` (Eq. 10a–b, 10-fold),
  `block_permutation_test` (P=5000), `cluster_bootstrap_ci` (BCa, B=2000),
  `two_layer_holm`, `contingency_decision`, `robustness_floor`, `achieved_power`
  — **the CO-PRIMARY statistic**, strict train/held-out fold separation; the
  nested `F` kept as a labeled descriptive helper only.
- **[DONE]** `src/neurotrace_it/analysis/outcome_y.py` — `build_loci_clusters`
  (Eq. 15; deterministic agglomerative surrogate, HDBSCAN optional drop-in),
  `loci_influence` (Eq. 16–17), `y_reliability_gate` (G7: ICC + proxy↔retrain).
- **[DONE]** `src/neurotrace_it/analysis/pair_mining.py` —
  `mine_matched_endpoint_pairs` (§3.2 recipe incl. coarsened exact matching +
  imbalance report), `paired_margin_test`, `power_for_pairs` (§4.5 formula).
- `src/neurotrace_it/layer_function.py` — `causal_layer_importance` (Eq. 13),
  `route_layers`, `capacity_match` (§2.9 rank reallocation). *(Router is an
  ablation, not load-bearing; remaining additive module for the deferred run.)*
- **[DONE]** `src/neurotrace_it/analysis/drift.py` — Fisher retention penalty
  (Eq. 11) kernel, `BrierCalibrator` (§2.6) + `factuality_drift` +
  `g6_factuality_gate` (G6: Spearman/Pearson + lower-CI + ECE).
- **[DONE]** `src/neurotrace_it/selection.py` — `greedy_submodular_select`
  (lazy-greedy monotone-submodular `F(S)`, Eq. 14, (1−1/e)); manifest emission via
  `schemas_v2.SelectionManifestV2`.
- `src/neurotrace_it/cost_model.py` — `gate1_multiplier` (incl. differential
  trajectory extraction/storage, §2.7), `extraction_parity_check`,
  `gate1b_deployability` (`R`, `R*`). *(Remaining additive module for the deferred
  run; the 2.0× gate kernel is reused from `metrics.passes_cost_gate`.)*
- `configs/experiments/lattice_v4.yaml` — pinned hparams (`K=64,|A|=8,
  subsample=512, r∈{8,16,32}, λ_ridge grid, HDBSCAN params, G7 thresholds,
  α,β,λ_r,λ_f,μ,τ,R_tot`), probe-layer grid `A`, resolution sweep, **three-line
  cost budget**; **`server.authorized: false`**.
- `tests/test_trajectory_ops.py` / `tests/test_residualize.py` — (a) `κ`
  permutation-sensitive while mean-pool invariant (encodes §2.5); (b) Brier
  propriety numeric check via Eq. 12 (**formula evaluation, not evidence**); (c)
  `F(S)` submodularity unit check (**formula evaluation, not evidence**); (d)
  ridge-FWL **recovers the known coefficient** on a synthetic linear DGP where the
  trajectory column is orthogonalized against the control (**formula evaluation,
  not evidence**); (e) block-permutation test attains nominal type-I on a null
  synthetic DGP (**formula evaluation, not evidence**); (f)
  `validate_selection_manifest_v2` round-trips estimands + recomputes
  `trajectory_hash`.

No server command, no training, no extraction is added or run. The modules above
are implemented as pure-stdlib, build-now/run-later code with unit/contract tests;
the only outstanding step is the **deferred authorized GPU run**
(`server.authorized: false`).

---

## 6. Honest limitations

1. **Residual signal may be small or null (intended falsifier).** The core claim
   is a **dataset-level partial effect** over the **full** endpoint control; if
   endpoints + length + difficulty + family already explain the variance, the
   cross-fit/permutation test fails and the claim is killed by §4.2/§4.6. The
   honest prior is that the residual may be small.
2. **Full-ridge control depends on `λ_ridge`.** Over-penalizing leaves residual
   endpoint variance in `T` (inflates the effect); under-penalizing over-fits and
   absorbs trajectory signal (deflates it). Mitigation: CV-locked `λ_ridge`,
   reported sensitivity across the grid, and the `r∈{8,16,32}` robustness floor.
   The full-ridge result is co-primary precisely so a single control choice is
   not load-bearing.
3. **Cross-fit/permutation power.** Repeated cross-fit + permutation is valid but
   lower-powered than an (invalid) parametric `F`; with `N` and 10 folds the
   design pre-commits to reporting achieved power and treating an underpowered
   family as diagnostic.
4. **LOCI `Y` is an influence approximation.** First-order influence (Eq. 16–17)
   can be inaccurate for large parameter moves; G7 (§2.6) is the explicit
   precondition — fail G7 ⇒ no primary claim on the proxy.
5. **Curvature SNR.** `κ` (Eq. 5) is noisy for short traces (`S_x` small);
   resolution sweep mitigates; if `κ`-only is at chance the §3.3 table drops the
   curvature novelty and the claim falls back to `D`.
6. **SW2 vs the collapse is empirical, not guaranteed.** SW2 escapes
   *first-moment* collapse but may empirically track endpoint/mean similarity;
   only the residual regression settles it. We do **not** claim "all moments."
7. **Matched-pair is a diagnostic, not causal proof (fix f2).** Even after
   coarsened exact matching, unobserved confounders (reasoning style, annotation
   quality) remain; P2 corroborates P1, never replaces it.
8. **Router fairness is fragile.** Capacity-conserving rank reallocation matches
   params/optimizer-state/total-rank, but exact FLOPs equality is hard; any
   residual mismatch forces a capacity-unmatched **diagnostic** label. The main
   result uses **uniform** LoRA so it never depends on this.
9. **Factuality proxy validity is a precondition (G6), not a result.** Brier
   proves calibration, not eval-drift prediction; fail G6 ⇒ `λ_f:=0`.
10. **Cost honesty (three lines, fix f1).** Differential trajectory extraction
    and storage are now charged to Gate-1 unless honest extraction-parity holds;
    if the honest multiplier exceeds 2.0× the method is high-cost-analysis, not
    deployment.
11. **Citation/model risk (fix f3, corrected).** The NAIT id `2603.13201` is a
    **real, checkable paper** (OpenReview, verified 2026-06-19) — NAIT stays a
    must-beat baseline and "beats NAIT" is gated on the fair-comparison results
    (§4.6), **not** on citation validity. Only `diversity_coreset` (`2605.26004`)
    remains unverified. Models pinned to released checkpoints with revision
    commits.
12. **No empirical claim yet (RR Stage-1).** Every result above is **design**;
    in-principle acceptance is conditional on executing the locked plan under
    approval. The only numeric checks here (Eq. 12 Brier identity; the
    submodularity inequality; the §4.5/power formulas) are **formula evaluations,
    not evidence**.

---

*Provenance:* Stage-1 Registered Report. The original v4 revision was additive
(added only this doc); the §5 novel-core modules it specifies have since been
**implemented and frozen** as additive pure-stdlib code (V1 records stay valid; no
existing public API meaning changed — APIs extended only), the theory is
**discharged where proved**, and the paper skeleton is written at `paper/main.tex`
(+ `paper/references.bib`). **No experiment executed; no training/extraction/model
load; no git commit made by this update; `server.authorized: false` preserved
throughout.** The only outstanding work is the deferred authorized GPU run.

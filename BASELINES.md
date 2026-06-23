# NeuroTrace-IT — Baselines (ARIS v6)

This document is the human-readable companion to
`configs/baselines/baseline_registry.yaml`. It locks the baseline suite for the
top-conference fair comparison on the v6 confirmatory headline model
**Qwen2.5-7B-Instruct**. It records *what each baseline is, where its official
code lives, how it is run, and what is held identical* so the comparison is fair.

`server.authorized` stays **false**. No baseline is run yet and **no result is
claimed**: every result cell elsewhere remains `DATA_NEEDED`. This file is
provenance + the fairness contract only.

The method under test is **LATTICE-R** (trajectory-as-distribution SW2/curvature
signature `ψ(x)` driving both data selection and per-layer LoRA routing; see
`docs/redesign/REDESIGN_v6.md` and `paper/main.tex`).

---

## 1. Locked baseline table

**Verified-official-repo suite (>= 8 contract; all 11 have confirmed real code).**
These are run from the OFFICIAL repository (`implementation_source: official_repo`).

| # | Name | Venue | Official repo | Role | Code? |
| --- | --- | --- | --- | --- | --- |
| 1 | LESS | ICML 2024 | https://github.com/princeton-nlp/LESS | must_beat (gradient/influence) | official repo (MIT) |
| 2 | DEITA | ICLR 2024 | https://github.com/hkust-nlp/deita | must_beat (quality × complexity × diversity) | official repo (Apache-2.0) |
| 3 | SelectIT | EMNLP 2024 | https://github.com/Blue-Raincoat/SelectIT | must_beat (model-intrinsic uncertainty) | official repo (Apache-2.0) |
| 4 | DataInf | ICLR 2024 | https://github.com/ykwon0407/DataInf | must_beat (closed-form LoRA influence) | official repo (MIT) |
| 5 | Nuggets | ICLR 2024 | https://github.com/pldlgb/nuggets | must_beat (one-shot in-context info gain) | official repo (license not stated) |
| 6 | MIG | ACL 2025 Findings | https://github.com/yichengchen24/MIG | must_beat (max info gain, semantic space) | official repo (Apache-2.0) |
| 7 | S2L (SmallToLarge) | NeurIPS 2024 | https://github.com/BigML-CS-UCLA/S2L | must_beat (training-trajectory clustering — closest trajectory competitor) | official repo (MIT) |
| 8 | Cherry / IFD | NAACL 2024 | https://github.com/tianyi-lab/Cherry_LLM | must_beat (Instruction-Following Difficulty) | official repo (license not stated) |
| 9 | TAGCOS | NAACL 2025 | https://github.com/2003pro/TAGCOS | must_beat (gradient-clustered coreset) | official repo (license not stated) |
| 10 | NeFT | COLING 2025 | https://github.com/NLP2CT/NeFT | control (neuron-level fine-tuning; R1 routing arm) | official repo (license not stated) |
| 11 | AdaLoRA | ICLR 2023 | https://github.com/QingruZhang/AdaLoRA | control (adaptive rank budget; R1 routing arm) | official repo (MIT) |

Official-repo selection must-beat comparators: **#1–#9** (9 methods). Official-repo
routing-arm controls: **#10–#11**. The **>= 8 official-repo** baseline contract is
satisfied (11 verified).

**Disclosed reimplementation anchors (base-paper, NOT counted in the >= 8).**
These are run as faithful reimplementations and are disclosed as such.

| Name | Venue | Code status | How we run it | Role |
| --- | --- | --- | --- | --- |
| NAIT | ICLR 2026 | No public repo | Faithful reimpl of endpoint neuron-similarity (Eq. 1); already an R0 control, high confidence | must-beat control |
| AlpaGasus | ICLR 2024 | GitHub is a project webpage only, no runnable code | Faithful reimpl of the LLM-as-judge quality scoring | must_beat |

These map to registry keys `endpoint_neuron_selection` (NAIT) and
`quality_score_selection` (AlpaGasus). They are honestly marked
`repo_verified: false` for runnable official code and are excluded from the
official-repo count.

**Project sanity / structural baselines (local, pre-existing).**
`random_subset`, `full_data_it` (upper bound), `diversity_coreset` (contextual,
non-must-beat — superseded for citation by TAGCOS), and the ablation arm
`layer_selective_no_trajectory`. See the registry for their contracts.

---

## 2. Fairness protocol

Every baseline differs from LATTICE-R in **exactly one place**: the *selection
rule* (or, for the two routing controls, the *per-layer capacity allocation*).
Everything downstream is held identical. This is enforced by
`docs/baseline_contract.md` and the 3-pool firewall (`analysis/pool_firewall.py`).

What is **matched** across all methods:

- **Model.** Confirmatory headline = `Qwen/Qwen2.5-7B-Instruct` (revision `main`,
  Apache-2.0). Secondary `Qwen/Qwen2.5-1.5B-Instruct` is a **robustness / smoke
  row only** (demoted from co-headline in v6). Any baseline whose method needs a
  scoring/proxy model uses a Qwen2.5 backbone (7B for raters, 1.5B for S2L's
  small-model proxy), never a foreign base, so features are backbone-matched.
- **Candidate pools.** The same three example-disjoint, family-stratified pools:
  MetaMathQA (math), `glaiveai/glaive-code-assistant` (code), HotpotQA-distractor
  (multi-hop QA), partitioned by the locked 3-pool firewall
  (`P_train` / `P_val` / `P_dep`). Selection reads activations/scores only through
  the frozen `P_train` operators.
- **Selection budget B.** The same top-`B` budget for every selector (pinned in
  `configs/experiments/lattice_v4.yaml` at run authorization).
- **Training pipeline.** OUR identical LoRA pipeline: same LoRA rank/capacity
  grid, same training steps, same token budget, same optimizer, same 20 seeds
  (`0..19`, `configs/seeds/paper_20.txt`). Equal rank is checked to be equal
  *realized compute* via `cost_model.compute_match_ledger`.
- **Evaluation pipeline.** OUR identical eval: target-task accuracy, MMLU
  retention drift, TruthfulQA + FActScore-style atomic-claim hallucination drift,
  and the honest cost ledger.

**Deciding gate.** The pre-registered closed-testing graph at FWER `α = 0.05`
over `{R0, R1, G2t, G2r, G2h, G2c}`:

- **R0** — the trajectory residual `ψ` explains target utility beyond
  `[φ_end, NAIT(L)]` (partial-`R²` floor + BCa lower bound).
- **R1** — the routing policy `π_ψ` beats every routing control (incl. AdaLoRA,
  NeFT) by `δ_R1`, decided by a full-union intersection-union test (IUT).
- **R2 / G2t** — LATTICE-R beats **every** selection comparator in the suite above
  (and the stronger full-`L` NAIT) by `δ_target` at a measured compute-matched
  budget, again by IUT — "beats SOTA selection" requires beating *all*, not the
  weakest. G2r/G2h/G2c are the retention/hallucination/cost non-inferiority
  ceilings.

If LATTICE-R does not clear these gates against the baselines at 7B, the method is
diagnosed and improved (never the data, never fabricated) and re-run under a cap;
see `docs/redesign/REDESIGN_v6.md` §D.

---

## 3. Per-baseline "how we run it"

Each note states: official repo vs reimpl, and exactly what is adapted to our
task. The full machine-readable contract (including `fairness_adaptation`) is in
`configs/baselines/baseline_registry.yaml`.

**LESS** (official repo, MIT) — Run the official optimizer-aware low-rank
gradient-similarity influence selection. Adapted: warm-up/projection on OUR
Qwen2.5-7B backbone, score OUR pool against the validation target, keep top-B,
train with OUR LoRA pipeline. (This is the v6 upgrade of the disclosed influence
anchor `influence_gradient_selection` from reimpl to official MIT code.)

**DEITA** (official repo, Apache-2.0) — Run the released complexity + quality
scorers and the diversity-aware Repr Filter greedy selection. Adapted: candidate
pool, budget B, and training/eval swapped to ours; scorers used as-is.

**SelectIT** (official repo, Apache-2.0) — Run the official token/sentence/model
-level self-reflection uncertainty scoring. Adapted: the rater model is OUR
headline Qwen2.5-7B-Instruct (model-intrinsic, no external judge); top-B over OUR
pool; OUR LoRA pipeline.

**DataInf** (official repo, MIT) — Compute the closed-form LoRA influence on OUR
Qwen2.5-7B LoRA gradients against the validation target. Adapted: influence
evaluated on the SAME LoRA parameterization used for training (pipeline-matched);
rank OUR pool, top-B, OUR pipeline.

**Nuggets** (official repo, license not stated) — Run the official one-shot
in-context "golden score" (zero-shot vs one-shot perplexity gain). Adapted: the
scoring LM is OUR Qwen2.5-7B-Instruct; top-B over OUR pool; OUR pipeline. Upstream
license is undeclared, so only the documented scoring procedure is reused on our
own backbone (no upstream artifact loaded).

**MIG** (official repo, Apache-2.0) — Run the official maximum-information-gain
selection over a semantic label graph. Adapted: the semantic-space graph is
rebuilt on OUR three pools; selected B examples trained with OUR pipeline.

**S2L / SmallToLarge** (official repo, MIT) — Run the official small-model
training-loss-trajectory clustering + balanced sampling. Adapted: the proxy small
model is OUR secondary backbone Qwen2.5-1.5B-Instruct (in-family); selected B
trained with OUR Qwen2.5-7B LoRA pipeline. **This is the closest trajectory
competitor**: S2L clusters scalar *loss* trajectories over steps, whereas
LATTICE-R uses full per-layer *activation-trajectory distributions* (SW2 /
curvature) — the contrast this baseline exists to expose.

**Cherry / IFD** (official repo, license not stated) — Run the official
Instruction-Following Difficulty score (conditioned vs unconditioned response
perplexity ratio). Adapted: scored with OUR Qwen2.5-7B-Instruct; top-B over OUR
pool; OUR pipeline. Undeclared upstream license → documented scoring reused on our
backbone.

**TAGCOS** (official repo, license not stated) — Run the official per-sample
gradient-representation + clustering + greedy coreset selection. Adapted: gradient
features from OUR Qwen2.5-7B LoRA; coreset of size B over OUR pool; OUR pipeline.
This is the **verified text-domain gradient-coreset comparator** that the dropped
MAGIC/VLM citation could not be; it supersedes `diversity_coreset` as the citable
text-coreset must-beat.

**NeFT** (official repo, license not stated) — **Routing-arm control.** Run the
official neuron-selection to choose the adapted parameter set. Adapted: projected
onto OUR fixed capacity grid (`R_tot` / `r_max`) so it trains the same effective
parameter count as every R1 arm; data selection held identical across arms.

**AdaLoRA** (official repo, MIT) — **Routing-arm control.** Run the official
SVD-importance adaptive rank reallocation under OUR matched total budget `R_tot`
(warm-up `W_ada = 200` steps, fixed seed) as an R1 arm; data selection held
identical across arms.

**NAIT** (reimpl anchor) — No public repo. Faithful reimplementation of endpoint
neuron-similarity (Eq. 1), written from the published equations; already wired as
an R0 control. Must-beat control; the headline "beats NAIT" requires beating the
stronger full-`L` reproduction, not the 8-anchor diagnostic.

**AlpaGasus** (reimpl anchor) — GitHub is a project webpage only (no runnable
code). Faithful reimplementation of the LLM-as-judge per-example quality scoring
under the project (MIT) license; no LLaMA-derived AlpaGasus model artifact is
loaded.

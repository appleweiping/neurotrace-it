# Pre-Registration

This is the consolidated Stage-1 analysis lock.
`server.authorized` stays **false**: no result is claimed to exist.

> **SUPERSEDED THRESHOLDS (C+D hardening, 2026-06-19).** The numeric decision
> thresholds below were first drafted against the v4 lock. The **binding source of
> the exact locked constants is now `configs/experiments/lattice_v5.yaml` plus
> `docs/redesign/REDESIGN_v5.md` §5.6** (LATTICE-R routing redesign). Where any
> threshold here disagrees with `lattice_v5.yaml`, **`lattice_v5.yaml` governs**.
> The values stated below have been resynced to the v5 locked design choices
> (`delta_target=0.01`, `delta_ret=0.02`, `delta_hall=0.02`, `delta_cost=0.05`,
> `floor_partial=0.01`); the older v4-era summary (`docs/redesign/REDESIGN_v4.md`
> §2-§4 and `configs/experiments/lattice_v4.yaml`) remains the historical record.

## Scope

Covers the endpoint-vs-trajectory data-selection gate and the REDESIGN_v4 NOVEL
CORE (the trajectory "lattice" operator + its residualized co-primary regression).
It does not claim any result exists; it locks the analysis before any run.

## Frozen Primary Hypothesis

H1 (v5 locked thresholds; binding source `configs/experiments/lattice_v5.yaml`):
trajectory selection / LATTICE-R routing beats endpoint-neuron selection (and every
locked baseline / control) by at least `delta_target = 0.01` normalized utility on
target or retention-adjusted score, with retention-drift disadvantage at most
`delta_ret = 0.02` and hallucination-drift disadvantage at most `delta_hall = 0.02`
(non-inferiority ceilings), and training-compute overhead at most `delta_cost = 0.05`
(5% relative). The routing win margin is `delta_R1 = 0.01`. All margins are on the
normalized `[0,1]` utility scale (1.0 = full retention-adjusted target accuracy).
These are pre-registered DESIGN CHOICES, not results.

## Pinned Models, Datasets, Evaluators (lattice_v4.yaml)

- **Models (Apache-2.0, released checkpoints only):** primary
  `Qwen/Qwen2.5-7B-Instruct`, secondary `Qwen/Qwen2.5-1.5B-Instruct`. Git
  `revision: main`; the exact commit SHA and tokenizer/checkpoint hashes are
  **resolved** from that revision via `huggingface_hub` at run authorization (not
  transcribed here, to avoid fabricating checksum values). LoRA only, matched rank.
- **Candidate pool (identities pinned):** math `meta-math/MetaMathQA`; code
  `glaiveai/glaive-code-assistant`; multihop `hotpotqa/hotpot_qa` (distractor
  config). Each pinned by `repo_id` + `revision: main`; split content-hash resolved
  at authorization.
- **Retention eval:** held-out MMLU aggregate. **Hallucination/factuality (also G6
  ground truth):** TruthfulQA + FActScore-style atomic-claim factuality.

## Primary Outcomes

- Target or retention-adjusted relative gain.
- Retention drift disadvantage.
- Selection cost multiplier (honest three-line cost table; extraction-parity kill).

## Secondary Outcomes

- Hallucination/factuality drift.
- Layer-policy ablation gain (router ablation; NOT load-bearing).
- Selected-data diversity.

## Locked Outcome `Y` (LOCI utility) — sign-corrected

`Y` is a **utility (higher = more useful)**: per-example retention-adjusted
fine-tuning utility, attributed by a **leave-one-cluster-in (LOCI)** influence
estimator. For cluster `g` containing `x_i`:

```
Delta_g = L_val(theta_{-g}) - L_val(theta)               (Eq. 16)
Y_i     = + (|g|^{-1} * Delta_g) * drift_adjust_i         (Eq. 17, sign-corrected)
```

Removing a **useful** cluster *raises* validation loss (`Delta_g > 0`), so its
members get **positive** `Y`. (The earlier `-` convention wrongly mapped useful
clusters to negative `Y`; corrected to `+`.)

### Cluster construction (locked method)

Embed candidates by L2-normalized endpoint signature `phi_end`; target
`K ≈ N/200` with a **size floor of 25**; noise/singletons reassigned to the
nearest centroid; assignment + hash persisted (`cluster_assignment_hash`);
clustering fit on the **train fold only**, held-out assigned by nearest centroid.

The **locked default clustering method is the deterministic agglomerative
surrogate** (`analysis/outcome_y.py::build_loci_clusters`): furthest-point
k-center seeding + Lloyd refinement + size-floor dissolution. It is locked (over
HDBSCAN) to keep the pipeline pure-stdlib, deterministic, and zero-extra-
dependency. **HDBSCAN** (`min_cluster_size=25, min_samples=10, metric=euclidean,
cluster_selection_method=eom`) is a pre-registered **optional drop-in**, used only
if the `hdbscan` package is available at authorization; its params are persisted
so the swap is one-to-one. The drop-in does not change the locked contract.

## Analysis Lock

Endpoint-neuron and trajectory selection must use the same candidate pool, budget,
base model, LoRA rank, training steps, validation policy, and evaluator. The
residualized co-primary regression, the kill-gates, and the multiplicity
correction are locked in `docs/statistical_analysis_plan.md`.

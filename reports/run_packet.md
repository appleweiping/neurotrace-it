# NeuroTrace-IT LATTICE-R v5 — Run Packet (BUILD-NOW / RUN-LATER)

**Status: design-only. `server.authorized: false`. NOTHING runs now.** This packet
specifies the exact, server-parameterized, ONE-COMMAND commands a later **AUTHORIZED**
run would execute for the full REDESIGN_v5 LATTICE-R pipeline. The **novel analysis
machinery is implemented and unit-tested** (green 93-test harness); the
**training/extraction body of each entrypoint is NOT yet coded** — every
`run_authorized(...)` branch is a guarded stub that raises `SystemExit` (see §3). The
resumability contract (§6) is therefore **scaffolding wired in `_run_guard.py`, not yet
exercised end-to-end**. Every command's default behaviour is a **dry run** that prints
the resolved plan JSON and exits 0, loading **no** model and touching **no** GPU. A
command leaves dry-run only when **BOTH** (a) the resolved config has
`server.authorized == true` **AND** (b) the explicit `--i-have-authorization` flag is
passed — and even then, today, it hits the guarded stub. **This packet fabricates ZERO
numeric results** — every quantitative slot is a `${...}` placeholder resolved from the
frozen configs at run authorization (see Section 0).

- Design spec: `docs/redesign/REDESIGN_v5.md` (LATTICE-R; verified-sound policy-value
  estimand; IUT leaves; closed-testing FWER; 3-pool firewall; faithful NAIT over `L`).
- Frozen routing config: `configs/experiments/lattice_v5.yaml` (`server.authorized: false`).
- Frozen models/datasets: `configs/experiments/lattice_v4.yaml`.
- Frozen compute budget: `configs/compute/first_gate_budget.yaml`.
- Shared seeds: `configs/seeds/paper_20.txt` (exactly `0..19`).
- Entrypoints: `scripts/extract_signatures.py`, `scripts/run_nait_baseline.py`,
  `scripts/run_r0_analysis.py`, `scripts/run_r1_routing.py`,
  `scripts/run_r2_matched_budget.py`, `scripts/eval_gates.py`
  (shared guard: `scripts/_run_guard.py`).

---

## 0. Placeholder dictionary — EVERY `${...}` used in this packet

Every placeholder below is a **locked symbol**. Two kinds:

1. **Pre-registered DECISION thresholds are now LOCKED to concrete values in
   `configs/experiments/lattice_v5.yaml`** (C+D hardening, 2026-06-19) — these are
   *design choices*, fixed before any run, NOT results. They include `${DELTA_R1}`,
   `${DELTA_TARGET}`/`${DELTA_REL}`/`${DELTA_RET}`/`${DELTA_HALL}`/`${DELTA_COST}`,
   `${R_TOT}`, `${R_MAX}`, `${R0_SUBSTRATE}` (`r_0`), `${TAU_SEL}` (rule + `k_bar`),
   `${SEED_RAND}`/`${SEED_SHUF}`/`${SEED_ADA}`, `${A_GLOB}` (rule + `k_glob`),
   `${W_ADA}`, `${B_BT}`, `${FLOOR_PARTIAL}`, `${RHO_J}`, `${POOL_SPLIT_SEED}`,
   `${LEDGER_TOL}`, and the closed-test α-allocation. The packet resolves each from the
   named config key (the value is in the YAML, not invented here).
2. **Genuinely EMPIRICAL slots stay `DATA_NEEDED`** and are resolved only at run
   authorization: resolved commit SHAs / tokenizer hashes (`${MODEL_*}`, `${DS_*}`),
   measured FLOPs/wall-clock, and every observed result cell. These are the only
   `DATA_NEEDED` quantities. (Structural multiplicity constants — closed-test
   weights/edges/split — were already pinned; see §4.) No value is invented here.

### 0.1 Run-tree / harness placeholders

| Placeholder | Meaning | Resolved from |
| --- | --- | --- |
| `${RUN_DIR}` | output + STATUS.json dir for an entrypoint | `--run-dir` flag, default `runs/<entrypoint>` |
| `${R0_RUN_DIR}` / `${R1_RUN_DIR}` / `${R2_RUN_DIR}` | the run dirs of the R0/R1/R2 entrypoints (consumed by `eval_gates.py`) | prior stages' `--run-dir` |
| `${SEED}` | one shared training seed | `configs/seeds/paper_20.txt` (`0..19`) |
| `${N_SEEDS}` | seed count for R1/R2 (`>= 20`) | `lattice_v5.yaml routing.R1_seeds` (the `paper_20.txt` manifest) |

### 0.2 Models / datasets (frozen in `lattice_v4.yaml`)

| Placeholder | Meaning | Resolved from (`configs/experiments/lattice_v4.yaml`) |
| --- | --- | --- |
| `${MODEL_PRIMARY}` | primary base model | `base_models.primary` = `Qwen/Qwen2.5-7B-Instruct` @ `revision: main` |
| `${MODEL_SECONDARY}` | secondary base model | `base_models.secondary` = `Qwen/Qwen2.5-1.5B-Instruct` @ `revision: main` |
| `${DS_MATH}` | math instruction pool | `candidate_pool.math` = `meta-math/MetaMathQA` @ `revision: main` |
| `${DS_CODE}` | code instruction pool | `candidate_pool.code` = `glaiveai/glaive-code-assistant` @ `revision: main` |
| `${DS_MULTIHOP}` | multi-hop QA pool | `candidate_pool.multihop` = `hotpotqa/hotpot_qa` (config `distractor`) @ `revision: main` |

> The exact commit SHA / tokenizer / split content-hashes are **resolved (not invented)**
> from each `revision: main` HEAD via `huggingface_hub` at run authorization
> (`pin_resolver` in `lattice_v4.yaml`). They are NOT transcribed here.

### 0.3 Trajectory-operator / extraction placeholders

| Placeholder | Meaning | Resolved from |
| --- | --- | --- |
| `${A_SIZE}` | anchor set size `|A|` (= 8) | `lattice_v4.yaml trajectory_operator.anchor_layers` |
| `${K_SW2}` | SW2 random projections `K` (= 64) | `lattice_v4.yaml trajectory_operator.sw2_projections` |
| `${SUBSAMPLE_CAP}` | (step,token) vectors/layer cap (= 512) | `lattice_v4.yaml trajectory_operator.subsample_cap` |
| `${BUDGET_B}` | selection budget `B` (top-k) | `lattice_v4.yaml` selection budget key (pinned at authorization) |

### 0.4 R0 (mechanism certificate) placeholders

| Placeholder | Meaning | Resolved from |
| --- | --- | --- |
| `${LAMBDA_RIDGE_GRID}` | ridge penalty grid `λ_ridge` | `lattice_v4.yaml` ridge grid / CV |
| `${P_PERM}` | block-permutation draws `P` (design value 5000) | REDESIGN_v5 §3.2 Eq.5-2p |
| `${B_BCA}` | cluster BCa bootstrap resamples (design value 2000) | REDESIGN_v5 §3.2 |
| `${FLOOR_PARTIAL}` | partial-R² floor the BCa lower bound must clear | `lattice_v5.yaml floor_partial` |
| `${ALPHA}` | confirmatory / marginal level `α` (= 0.05) | `lattice_v5.yaml closed_testing.alpha`, `inference.marginal_level` |

### 0.5 R1 (six-arm routing) placeholders

| Placeholder | Meaning | Resolved from (`configs/experiments/lattice_v5.yaml`) |
| --- | --- | --- |
| `${R0_SUBSTRATE}` | fixed baseline rank `r_0` on every `L\A` layer | `routing.r0_substrate` (LOCKED `4`) |
| `${R_TOT}` | total LoRA rank reallocated over `A` | `R_tot` (LOCKED `64`) |
| `${R_MAX}` | per-anchor rank cap `r_max` (with `|A|·r_max ≥ R_tot`) | `r_max` (LOCKED `16`) |
| `${TAU_SEL}` | raw mask threshold `τ_sel` (rule + `k_bar`) | `routing.tau_sel` (LOCKED rule, `k_bar=4`) |
| `${DELTA_R1}` | R1 win margin `δ_R1` | `routing.delta_R1` (LOCKED `0.01`) |
| `${SEED_RAND}` / `${SEED_SHUF}` / `${SEED_ADA}` | fixed control seeds | `controls.seed_rand/seed_shuf/seed_ada` |
| `${A_GLOB}` | fixed global-selective anchor block | `controls.A_glob` |
| `${W_ADA}` | fixed AdaLoRA warm-up budget | `controls.W_ada` |
| `${B_BT}` | per-contrast bootstrap-`t` resamples | `inference.B_bt` |

### 0.6 R2 (compute-matched) placeholders

| Placeholder | Meaning | Resolved from (`configs/experiments/lattice_v5.yaml`) |
| --- | --- | --- |
| `${DELTA_TARGET}` | R2-target IUT margin `δ_target` | `R2_margins.delta_target` |
| `${DELTA_REL}` | relative-win margin `δ_rel` | `R2_margins.delta_rel` |
| `${DELTA_RET}` | retention-drift ceiling `δ_ret` (non-inferiority) | `R2_margins.delta_ret` |
| `${DELTA_HALL}` | hallucination-drift ceiling `δ_hall` (non-inferiority) | `R2_margins.delta_hall` |
| `${DELTA_COST}` | cost-gap ceiling `δ_cost` (non-inferiority) | `R2_margins.delta_cost` |
| `${LEDGER_TOL}` | matched-budget relative tolerance | `compute_ledger.tolerance` |

### 0.7 J-profile placeholder

| Placeholder | Meaning | Resolved from |
| --- | --- | --- |
| `${RHO_J}` | Spearman ρ gate for `J_{c,ℓ}` validation | `lattice_v5.yaml rho_J` |
| `${POOL_SPLIT_SEED}` | locked 3-pool split salt/seed | `lattice_v5.yaml pool_split_seed` |

---

## 1. Primary confirmatory cells (named identically to the paper)

The confirmatory family is the **six elementary nulls** of the closed-testing graph
(§4). Each cell name below is used **identically** in `docs/redesign/REDESIGN_v5.md`
(Table 4.1), in `analysis/closed_testing.py` (`ELEMENTARY_NULLS`), in the entrypoint
`primary_cell(s)` plan field, and in the paper.

| Cell | Hypothesis (margin null) | Entrypoint | Decision |
| --- | --- | --- | --- |
| **R0** (`G0`) | `T` adds nothing given `[φ_end, NAIT(L)]` on `Y_obs` | `run_r0_analysis.py` | reject iff perm-`p < α_R0` AND BCa lower > `${FLOOR_PARTIAL}`, stable across `r∈{8,16,32}` |
| **R1** (`G1`) | `H0^{R1}=⋃_c{g_c ≤ δ_R1}` over 5 controls | `run_r1_routing.py` | **IUT**: reject iff every `L_c(α) > δ_R1` (Eq.7-IUT) |
| **G2t** (R2-target) | `H0^{G2t}=⋃_b{gap_b ≤ δ_target}` over baseline set | `run_r2_matched_budget.py` | **IUT**: reject iff every comparator `L_b(α) > δ_target` |
| **G2r** | retention-drift `≥ δ_ret` (non-inferiority) | `run_r2_matched_budget.py` | single bootstrap-`t`, upper bound `< δ_ret` |
| **G2h** | hallucination-drift `≥ δ_hall` | `run_r2_matched_budget.py` | single bootstrap-`t`, upper bound `< δ_hall` |
| **G2c** | cost-gap `≥ δ_cost` | `run_r2_matched_budget.py` | single bootstrap-`t`, upper bound `< δ_cost` |

R1 and G2t are **single union-null leaves**: in **every** closed-test intersection that
contains them, the local test requires the **FULL** union (all 5 controls / all
comparators), never an in-scope subset (RE-FIX-5). Their internal components are **not**
separate members of the closed family.

---

## 2. The hard guard (shared by every entrypoint)

Implemented once in `scripts/_run_guard.py::guard(...)`. For every entrypoint:

1. Top-level imports are **pure-python / stdlib only**. `torch`, `transformers`,
   `datasets`, `peft` are **lazy-imported inside the guarded branch** — so a dry run
   (and `--help`) loads no model and needs no GPU.
2. The guard reads `server.authorized` from `--config` (default `lattice_v5.yaml`,
   currently `false`) and the `--i-have-authorization` flag.
3. **Run iff `server.authorized == true` AND `--i-have-authorization`.** Otherwise:
   print the resolved plan JSON, exit 0, run nothing. **Even when both hold today, the
   `run_authorized(...)` body is a guarded stub that raises `SystemExit`** — the training
   body is wired but not yet coded.
4. `--resume` is designed to reload `${RUN_DIR}/STATUS.json` and skip every completed
   stage; a `plan_hash` mismatch refuses to resume into a changed plan. (Helper-level
   only; no authorized run has produced a `STATUS.json` to resume from yet.)

Verified (committed config, no fabrication): all six entrypoints dry-run with
`will_run=false`, `server_authorized=false`, exit 0; `--i-have-authorization` alone
does **not** bypass the config guard. The authorized branch is a stub that raises, so no
training, extraction, or model load occurs under any flag combination as committed.

---

## 3. RUN-LATER commands (exact, ONE-COMMAND; resumability scaffolding wired, not yet exercised)

> **Honest entrypoint status (read before §3.x).** The *novel analysis machinery* every
> entrypoint imports — `layer_function` (`capacity_match`/`make_feasible_mask`/controls),
> `analysis/routing_intervention` (policy value, bootstrap-`t`, IUT),
> `analysis/layer_attribution`, `analysis/matched_budget`, `analysis/closed_testing`,
> `analysis/pool_firewall`, `cost_model`, `baselines/nait_layerwise` — **is implemented
> and unit-tested** as do-not-run pure-Python (green 93-test harness). What is **NOT yet
> implemented is the training/extraction BODY**: each entrypoint's `run_authorized(...)`
> branch is a **guarded stub that raises `SystemExit`** ("authorized branch intentionally
> not implemented in this BUILD-NOW / RUN-LATER packet") after lazily importing the
> modules. Consequently the model load, the six real training arms, activation extraction,
> and the gate evaluation are **wired and specified but not coded**, and the
> resumability/idempotence contract (§6) is **scaffolding that exists in
> `_run_guard.py::StatusCheckpoint` but has never been exercised end-to-end** — no
> `STATUS.json` stage is ever marked `completed`, because the authorized branch raises
> before doing work. These commands are therefore **one-command and dry-run-safe today**,
> but they are **not "truly resumable" yet**: resumability becomes real only when the
> training body replaces each stub at authorization.

> Replace `python` with the project interpreter. The §3.0 preflight commands carry **no**
> `--i-have-authorization` flag. The §3.1–§3.6 commands below ARE written in their
> **RUN-LATER form**, i.e. they **do** include `--i-have-authorization` — this is the
> exact command an authorized operator would run. **As committed, every one of these
> commands still executes as a DRY RUN**, because the hard guard requires BOTH
> `server.authorization == true` in the resolved config AND the flag; the committed
> config has `server.authorized: false`, so the flag alone is inert and bypasses nothing
> (`scripts/_run_guard.py::guard`). To actually run later, an authorized operator flips
> `server.authorized: true` in the resolved config (the flag is already present). **Do
> not do that now.**

### 3.0 Preflight (dry-run plan inspection — safe now)

```bash
python scripts/extract_signatures.py     # prints plan JSON, exits 0, loads nothing
python scripts/run_nait_baseline.py
python scripts/run_r0_analysis.py
python scripts/run_r1_routing.py
python scripts/run_r2_matched_budget.py
python scripts/eval_gates.py
```

### 3.1 Phase B-0/B-1 — signatures + faithful NAIT directions

```bash
python scripts/extract_signatures.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/extract_signatures \
  --i-have-authorization            # ONLY effective when server.authorized==true
```
Resumable: `--resume --run-dir runs/extract_signatures`. Stages (idempotent):
`nait_directions_L`, `nait_scores_proj`, `trajectory_signatures_{train,val,dep}`.
On `P_dep`, activations only (firewall).

### 3.2 Phase B-0 — faithful NAIT endpoint baseline (decisive comparator over `L`)

```bash
python scripts/run_nait_baseline.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/run_nait_baseline \
  --i-have-authorization
```
Resume: `--resume --run-dir runs/run_nait_baseline`.

### 3.3 Phase B-2 — R0 mechanism certificate (PRIMARY CELL R0)

```bash
python scripts/run_r0_analysis.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/run_r0_analysis \
  --i-have-authorization
```
Resume: `--resume --run-dir runs/run_r0_analysis`. Fail action: `stop_main_novelty_claim`.

### 3.4 Phase B-3 — R1 six-arm masked-LoRA INTERVENTIONAL routing (PRIMARY CELL R1)

```bash
python scripts/run_r1_routing.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/run_r1_routing \
  --i-have-authorization
```
Resume (RUN-LATER design contract, not yet exercised): `--resume --run-dir
runs/run_r1_routing`. **Once the training body replaces the guarded stub**, each
`(arm, seed)` training cell is to be recorded into `STATUS.json` only after its
checkpoint + `U_train` are durably written, so a resumed run would skip finished arms.
Arms: `psi, unif, shuf, rand, global, ada` over 20 seeds; the `ada_warmup_seed_outside_V`
stage runs **once** before the arms. **As committed, the authorized branch raises before
training, so no arm is ever marked complete and the skip-finished-arms behaviour is
unexercised.** Fail action: `drop_routing_keep_selection`.

### 3.5 Phase B-4 — R2 compute-matched method win (PRIMARY CELLS G2t/G2r/G2h/G2c)

```bash
python scripts/run_r2_matched_budget.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/run_r2_matched_budget \
  --i-have-authorization
```
Resume: `--resume --run-dir runs/run_r2_matched_budget`. Fail action: `no_method_win_claim`.

### 3.6 Phase B-5 — closed-testing decision + honest cost ledger

```bash
python scripts/eval_gates.py \
  --config configs/experiments/lattice_v5.yaml \
  --run-dir runs/eval_gates \
  --i-have-authorization
```
Consumes `runs/run_r0_analysis/r0_partial_r2.json`, `runs/run_r1_routing/r1_iut.json`,
`runs/run_r2_matched_budget/r2_result.json`; emits the FWER-≤0.05 rejection set over
`{R0,R1,G2t,G2r,G2h,G2c}` and the R3 two-sided cost ledger.

---

## 4. Multiplicity — pinned closed-testing graph (structural; NOT placeholders)

Mirrors `lattice_v5.yaml closed_testing` and `analysis/closed_testing.py::GraphSpec`.

- Nodes: `{G0(R0), G1(R1), G2t, G2r, G2h, G2c}`; family-wise `α = 0.05`.
- Initial mass `w_0 = 1.00` on R0. Edges (pinned): `G0→G1 (1.0)`, `G1→G2t (1.0)`,
  `G2t→{G2r,G2h,G2c}` split `(w_r,w_h,w_c)=(0.34,0.33,0.33)`, recycle `{G2r,G2h,G2c}→G2t (1.0)`.
- R1 and G2t are **full-union IUT** leaves (Berger 1982; no penalty, exact level `α`).
- Per-contrast inference: paired studentized **bootstrap-`t`** (asymptotic; Eq.7-Bt).
- `eval_gates.py` runs the Maurer-Bretz shortcut AND asserts it equals the brute-force
  63-intersection closed test; FWER ≤ 0.05 (Prop. P1-FWER).

These weights/edges are **structural constants**, pinned now — they are not `DATA_NEEDED`.

---

## 5. Compute budget (incl. the six training arms)

From `configs/compute/first_gate_budget.yaml` (frozen; `buffer_percent: 30`). These are
the **budget envelope** the authorized run must fit; they are config values, not run
measurements. The realized FLOPs/wall-clock are **measured** by
`cost_model.compute_match_ledger` at run time (R2/§3.7) — not predicted here.

| Resource | Budget (frozen config) | Maps to |
| --- | --- | --- |
| activation_extraction_gpu_hours | 12 | `extract_signatures.py` (both models, 3 pools) |
| endpoint_selection_gpu_or_cpu_hours | 2 | `run_nait_baseline.py` (NAIT over `L`) |
| trajectory_selection_gpu_or_cpu_hours | 4 | `run_r0_analysis.py` + signature scoring |
| matched_lora_training_gpu_hours | 24 | `run_r1_routing.py` (6 arms × 20 seeds) **and** `run_r2_matched_budget.py` |
| storage_gb | 500 | activation tensors + arm checkpoints |
| wall_clock_hours | 72 | end-to-end envelope |

**Six-arm note.** R1 trains `6 arms × ${N_SEEDS}` (= 20) masked-LoRA runs, all sharing
`r_0`, `R_tot`, and the score-free `capacity_match` map, so the §3.7 compute-match
reduces to the anchor-`A` allocation. The `matched_lora_training_gpu_hours: 24` line is
the **frozen envelope** for this; exact per-arm FLOPs are measured by the ledger at run
authorization, never fabricated here. Preflight (`first_gate_budget.yaml`):
`gpu_available`, `output_dir_empty_or_resumable`, `disk_free_gb_at_least_650`,
`raw_activation_cleanup_policy_confirmed`.

---

## 6. Resumability / idempotence contract (DESIGN CONTRACT — scaffolding only, not yet exercised)

**Honest status.** The resumability machinery below is the *specified design contract*
that the `_run_guard.py::StatusCheckpoint` helper supports, and the helper itself is
unit-covered. But **no entrypoint's authorized branch is implemented** — each
`run_authorized(...)` raises `SystemExit` before any stage runs (see §3). So today **no
`STATUS.json` is ever written by an authorized run, no stage is ever marked `completed`,
and the `--resume` skip-path has never executed end-to-end.** The contract becomes real
only when the training/extraction body replaces each guarded stub at authorization.

- Every entrypoint *will* write `${RUN_DIR}/STATUS.json` (`scripts/_run_guard.py::StatusCheckpoint`):
  `{entrypoint, plan_hash, started_at, updated_at, completed[], pending[], server_authorized:false}`
  — **once the authorized body exists**. (The dry-run path writes nothing.)
- A stage is to be added to `completed` **only after** its output artifact is durably
  written (idempotent / restartable cells) — wired in `StatusCheckpoint.mark_done`, but
  no entrypoint calls it yet because the body is a stub.
- `--resume` is designed to skip every completed stage; a changed resolved plan
  (`plan_hash` mismatch) refuses to resume, forcing a fresh `--run-dir`
  (`StatusCheckpoint.load_or_new`, unit-tested at the helper level).
- `experiments/queue_manifest.yaml` lists the same cells with `idempotent: true` and the
  explicit dependency DAG so an external runner can restart any cell **once the bodies are
  implemented**.

---

## 7. Firewall + scope (carried from §3.1, §3.5a)

- 3-pool firewall (`pool_firewall.split_pools`, locked salt `${POOL_SPLIT_SEED}`):
  `Y_obs` only on `P_train`; `V(π)` only on `P_val`; `P_dep` reads **activations only**
  before the decision; no `P_dep` outcome enters the `P_dep` decision path.
- Every confirmatory `V(π)`/`g_c` is **pool-conditional** on the single locked partition.
  The multi-pool generalization probe (§3.5a) is **non-confirmatory** (Holm within probe),
  never part of the confirmatory `α`.

*Provenance:* design-only Stage-1 Registered Report. No experiment, training, extraction,
or model load is run by this packet. `server.authorized: false` throughout. Zero
fabricated numbers.

---

## 8. ARIS v6 baseline-suite addendum (what needs updating for the baseline runs)

**Appended 2026-06-21 (ARIS v6).** This section notes — does NOT yet implement —
what a later authorized run must add for the locked baseline comparison. Source of
truth: `configs/baselines/baseline_registry.yaml` (`v6_baseline_contract`),
`BASELINES.md`, and `docs/redesign/REDESIGN_v6.md` (§B/§C/§E). `server.authorized`
stays **false**; no baseline runs now.

### 8.1 Per-baseline run approach (official repo vs reimpl)

Each baseline contributes **one number per pool** to the Gate-R2 G2t IUT
comparator set. For every baseline the pipeline is: *its* selection over OUR pool
under budget `B` ⇒ OUR identical Qwen2.5-7B LoRA training ⇒ OUR identical eval.
Only the selection step is the baseline's own.

| Baseline | Run via | What is adapted to our task |
| --- | --- | --- |
| LESS | official repo (MIT) | gradient warm-up/projection on OUR Qwen2.5-7B; score OUR pool; top-B |
| DEITA | official repo (Apache-2.0) | released complexity+quality scorers + Repr Filter over OUR pool; top-B |
| SelectIT | official repo (Apache-2.0) | self-reflection uncertainty with OUR Qwen2.5-7B as rater; top-B |
| DataInf | official repo (MIT) | closed-form LoRA influence on OUR Qwen2.5-7B LoRA grads; top-B |
| Nuggets | official repo (license n/s) | one-shot golden score with OUR Qwen2.5-7B; top-B |
| MIG | official repo (Apache-2.0) | info-gain over a semantic graph rebuilt on OUR pools; select B |
| S2L | official repo (MIT) | loss-trajectory clustering on OUR Qwen2.5-1.5B proxy; select B; train at 7B |
| Cherry/IFD | official repo (license n/s) | IFD score with OUR Qwen2.5-7B; top-B |
| TAGCOS | official repo (license n/s) | gradient coreset on OUR Qwen2.5-7B LoRA grads; select B |
| NeFT | official repo (license n/s) | **R1 routing arm**: neuron-select projected to OUR capacity grid `R_tot`/`r_max` |
| AdaLoRA | official repo (MIT) | **R1 routing arm**: SVD adaptive rank under OUR matched `R_tot`, `W_ada=200` |
| NAIT | faithful reimpl (no repo) | endpoint neuron-similarity Eq. 1; already an R0/G2t control |
| AlpaGasus | faithful reimpl (webpage only) | LLM-as-judge quality scoring under project (MIT) license |

Integration notes for the later run:
- The nine selection comparators (LESS…TAGCOS) join the existing G2t comparator
  set; the closed-test G2t IUT requires LATTICE-R to beat **every** member by
  `δ_target` (no weakest-link shortcut). The two routing controls (NeFT, AdaLoRA)
  join the **R1** routing-arm IUT, projected to the shared capacity grid.
- Vendored/official baseline code is run in its own isolated env; only the
  **selected example IDs + hash** cross back into OUR pipeline (firewall-safe),
  exactly like the existing selectors. No baseline reads `P_dep` outcomes.
- Backbone-matching is mandatory: baselines that need a scorer/proxy model use a
  Qwen2.5 backbone (7B raters; 1.5B for S2L), never the baseline's paper base.

### 8.2 Single-GPU kill-gate-first order (one RTX-4090, serial)

Run order on the single 4090 (stop on first hard gate failure; see REDESIGN_v6 §E):

1. **NAIT(`L`) + signatures + R0** (cheap, selection-side). R0 fail ⇒
   `stop_main_novelty_claim`; **no training compute spent on baselines or arms.**
2. **R1 six routing arms** (incl. AdaLoRA/NeFT controls), serial, 20 seeds,
   checkpoint per `(arm, seed)`. R1 fail ⇒ `drop_routing_keep_selection`.
3. **Baseline selection + matched LoRA training**, serial, reusing the same 4090;
   cheapest-scoring selectors first (IFD/Nuggets perplexity) so a dominating
   result surfaces early.
4. **R2 compute-matched comparison + closed-testing decision** last; realized
   FLOPs/wall-clock **measured** by `cost_model.compute_match_ledger`, never
   predicted. `> 2.0×` extraction-parity ⇒ high-cost-analysis, no deployment claim.

This fits the frozen envelope (`first_gate_budget.yaml`,
`matched_lora_training_gpu_hours: 24`, `buffer_percent: 30`) on one device by
spending the killing-cheapest gates before the expensive arms. Nothing runs until
`server.authorized: true` AND `--i-have-authorization`.

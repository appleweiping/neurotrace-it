# Statistical Analysis Plan

Consolidated Stage-1 statistical lock (companion to `docs/pre_registration.md`).
`server.authorized` stays **false**.

> **SUPERSEDED THRESHOLDS (C+D hardening, 2026-06-19).** The binding source of the
> exact locked constants is now `configs/experiments/lattice_v5.yaml` plus
> `docs/redesign/REDESIGN_v5.md` §4-§5.6. Where any threshold here disagrees with
> `lattice_v5.yaml`, **`lattice_v5.yaml` governs**. The robustness floor below has
> been resynced to the v5 locked value (`floor_partial = 0.01`); the older v4-era
> constants (`configs/experiments/lattice_v4.yaml`, `docs/redesign/REDESIGN_v4.md`
> §4) remain the historical record.

## Main Test

Paired tests on matched evaluation examples. Report relative target gain,
retention-adjusted gain, drift difference, confidence intervals, and effect sizes.

### Residualized co-primary regression (NOVEL CORE)

The trajectory operator's incremental signal is tested by **partialling out the
full endpoint control** `[phi_end | covariates | intercept]` from both the outcome
and the trajectory columns via **ridge** (dual/kernel form for n-space scaling;
CV-locked lambda over `[1e-2, 1e-1, 1, 10, 100] x sigma^2_phi`), then a
**cross-fit partial-R^2** over `K=10` family-stratified, example-disjoint folds.
Significance is a **conditional-null block permutation** (`P=5000`, within
family x fold strata) on the orthogonalized residuals. The endpoint control is the
**full `phi_end` ridge** (co-primary), NOT a 16-PC summary; PCA-`r` ranks
`{8,16,32}` are sensitivity poles for the **robustness floor**
(`floor_partial = 0.01`, the v5 locked Gate-R0 partial-R^2 floor in
`configs/experiments/lattice_v5.yaml`). Cluster bootstrap (BCa, `B=2000`) gives the CI.

## Kill-Gates (preconditions; fail => fall back, no main claim)

- **G6 factuality:** `lambda_f` may be non-zero ONLY if, on a held-out slice,
  Spearman `rho(f_hat, drift_eval) >= 0.3` AND Pearson `r >= 0.3` **with lower 95%
  one-sided CI > 0** (Fisher-z, one-sided `z_{0.95}=1.6449`), AND `ECE <= 0.1`.
  Fail => `lambda_f := 0`, no factuality/safety claim.
- **G7 Y-reliability:** before the primary regression runs on the influence proxy
  `Y`, require (0) `>= 3` seeds, (i) `ICC(2,1) >= 0.6` across those seeds, (ii)
  Spearman `rho(Y, retrain_delta) >= 0.3` on the `n_sub = 60` cluster subset **with
  lower 95% one-sided CI > 0**. Fail => fall back to the direct retrain-delta `Y`
  (underpowered diagnostic), proxy regression NOT run.

Both CI legs use a one-sided lower 95% bound (`z_{0.95}=1.6448536...`), consistent
with the one-sided "reliably > 0" question (not the two-sided `z_{0.975}=1.96`).

## Multiple Comparisons (two-layer Holm)

Family-wise alpha = 0.05.

1. **Within the trajectory family** `{joint, D_only, kappa_only}`: Holm over the 3
   block tests. The **`joint` (joint-T) block is the gatekeeper** — `D_only` /
   `kappa_only` are interpreted only if `joint` passes; otherwise exploratory. The
   gatekeeper key is `joint` everywhere (config, `two_layer_holm()` default, and
   `holm_adjust` p-value keys are unified, so wiring cannot fail-closed on a name
   mismatch).
2. **Across metric families** `{target, retention, hallucination, layer, cost}`:
   Holm on the gatekeeper-conditioned p-values, only if the gatekeeper passes.

The §3.3 2x2 contingency table (`D_only` x `kappa_only`) is consulted **only if**
the `joint` gatekeeper passes, yielding decision in
`{full, fallback, curvature-only, kill}`.

## Seeds And Replicates

Paper-facing results require at least 20 shared seeds (sequential 0..19, shared by
all methods) or bootstrap replicates. Lower counts are diagnostic only. G7 ICC
requires `>= 3` seeds.

## Endpoint Baseline Rule

If endpoint-neuron selection is not runnable under the same budget, the main claim
cannot be evaluated.

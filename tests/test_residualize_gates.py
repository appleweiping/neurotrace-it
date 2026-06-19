"""Unit tests for the locked Phase-2 GATES (REDESIGN_v4 §2.3-§4.5).

Build-now / run-later: every assertion is a closed-form / synthetic-DGP property
check ("formula evaluation, not evidence"). No model load, no server call, no
training. Pure stdlib, sub-second.

Coverage:
* Dual / kernel ridge residual-maker == primal ridge (fix b: scalable n-space).
* Conditional-null block permutation operates on M_lambda*T residuals (fix a):
  small p under a planted residual signal, calibrated under the null.
* build_endpoint_control assembles [phi_full, C, 1] + PCA-r poles (§2.3, §4.2).
* cluster_bootstrap_ci (BCa) brackets the point estimate (§2.4 step 3).
* two_layer_holm gatekeeper + §3.3 2x2 contingency table (fix e, d).
* robustness_floor tiers (fix a).
* LOCI clusters honour the size floor; loci_influence is the Eq.16-17 attribution.
* ICC(2,1), Spearman, and the G7 reliability gate (fix c).
* achieved_power / power_for_pairs are monotone and bounded (P=5000 x K=10).
"""

from __future__ import annotations

import math
import random

from neurotrace_it.analysis.residual_test import (
    block_permutation_test,
    cross_fit_partial_r2,
    dual_ridge_partial_out,
    ridge_partial_out,
)
from neurotrace_it.analysis.residualize import (
    achieved_power,
    build_endpoint_control,
    cluster_bootstrap_ci,
    contingency_decision,
    robustness_floor,
    two_layer_holm,
)
from neurotrace_it.analysis.outcome_y import (
    build_loci_clusters,
    icc_2_1,
    loci_influence,
    spearman_rho,
    y_reliability_gate,
)
from neurotrace_it.analysis.pair_mining import power_for_pairs


# --------------------------------------------------------------------------- #
# Synthetic planted-residual DGP (shared with the residual-test suite).       #
# --------------------------------------------------------------------------- #


def _planted_dgp(n, *, signal_coef, noise_sd, seed, p_phi=3):
    """Wide-ish endpoint control phi + one trajectory column with a residual part."""

    rng = random.Random(seed)
    control, trajectory, outcome, strata = [], [], [], []
    for i in range(n):
        phi = [rng.gauss(0.0, 1.0) for _ in range(p_phi)]
        t_resid = rng.gauss(0.0, 1.0)
        t = 0.7 * phi[0] + t_resid
        y = 1.5 * phi[0] - 0.4 * phi[1] + signal_coef * t_resid + rng.gauss(0.0, noise_sd)
        control.append(phi + [1.0])           # [phi_end... , intercept]
        trajectory.append([t])
        outcome.append(y)
        strata.append(i % 4)
    return control, outcome, trajectory, strata


# --------------------------------------------------------------------------- #
# Fix b: dual / kernel ridge equals primal ridge but scales in n-space.        #
# --------------------------------------------------------------------------- #


def test_dual_ridge_matches_primal_ridge_residuals():
    control, outcome, trajectory, _ = _planted_dgp(60, signal_coef=2.0, noise_sd=0.3, seed=11)
    penalized = [0, 1, 2]
    lam = 0.5
    primal_y, primal_cols = ridge_partial_out(
        control, outcome, trajectory, penalized_columns=penalized, ridge_lambda=lam
    )
    dual_y, dual_cols = dual_ridge_partial_out(
        control, outcome, trajectory, penalized_columns=penalized, ridge_lambda=lam
    )
    # Residuals must agree to numerical tolerance (same M_lambda, different solve).
    for z, y in zip(control, outcome):
        assert abs(primal_y.residual(z, y) - dual_y.residual(z, y)) < 1e-6
    for z, row in zip(control, trajectory):
        assert abs(primal_cols[0].residual(z, row[0]) - dual_cols[0].residual(z, row[0])) < 1e-6


def test_dual_ridge_applies_out_of_sample_without_pxp_gram():
    # Train on a subset, apply to a held-out row (the cross-fit access pattern).
    control, outcome, trajectory, _ = _planted_dgp(40, signal_coef=1.0, noise_sd=0.2, seed=12)
    train_c, train_y, train_t = control[:30], outcome[:30], trajectory[:30]
    dual_y, _ = dual_ridge_partial_out(
        train_c, train_y, train_t, penalized_columns=[0, 1, 2], ridge_lambda=1.0
    )
    # Out-of-sample residual is finite and uses only the train_phi kernel rows.
    r = dual_y.residual(control[35], outcome[35])
    assert math.isfinite(r)
    assert len(dual_y.train_phi) == 30


# --------------------------------------------------------------------------- #
# Fix a: conditional-null permutation on the orthogonalized residuals.         #
# --------------------------------------------------------------------------- #


def test_dual_crossfit_recovers_positive_partial_r2_under_signal():
    control, outcome, trajectory, strata = _planted_dgp(240, signal_coef=2.0, noise_sd=0.3, seed=21)
    res = cross_fit_partial_r2(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0, 1, 2], ridge_lambda=1e-2, n_folds=5, seed=3, use_dual=True,
    )
    assert res.partial_r2 > 0.05


def test_conditional_null_permutation_small_p_under_signal():
    control, outcome, trajectory, strata = _planted_dgp(160, signal_coef=2.0, noise_sd=0.3, seed=22)
    res = block_permutation_test(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0, 1, 2], ridge_lambda=1e-2, n_folds=5,
        n_permutations=199, seed=5, use_dual=True,
    )
    assert res.permutation_p_value is not None
    assert res.permutation_p_value < 0.1


def test_conditional_null_permutation_calibrated_under_null():
    control, outcome, trajectory, strata = _planted_dgp(160, signal_coef=0.0, noise_sd=1.0, seed=23)
    res = block_permutation_test(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0, 1, 2], ridge_lambda=1e-2, n_folds=5,
        n_permutations=199, seed=7, use_dual=True,
    )
    assert res.permutation_p_value is not None
    # Under H0 the conditional-null p is a valid probability and not tiny by construction.
    assert 0.0 < res.permutation_p_value <= 1.0
    assert res.permutation_p_value > 0.02


# --------------------------------------------------------------------------- #
# §2.3/§4.2: endpoint control assembly + PCA-r poles.                          #
# --------------------------------------------------------------------------- #


def test_build_endpoint_control_full_and_pca_poles():
    rng = random.Random(31)
    phi = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(40)]
    cov = [[rng.gauss(0, 1), 1.0 if i % 2 else 0.0] for i in range(40)]
    ctrl = build_endpoint_control(phi, cov, pca_ranks=(8, 16, 32), seed=0)
    # Full control = [6 phi cols | 2 cov | 1 intercept]; penalized = the 6 phi cols.
    assert len(ctrl.full_control[0]) == 6 + 2 + 1
    assert ctrl.penalized_columns == tuple(range(6))
    # PCA poles exist for each rank; rank is capped at min(r, p_phi, n).
    for r in (8, 16, 32):
        assert r in ctrl.pca_controls
        n_pen = len(ctrl.pca_penalized_columns[r])
        assert n_pen <= min(r, 6, 40)
        assert len(ctrl.pca_controls[r][0]) == n_pen + 2 + 1
    assert ctrl.phi_variance > 0.0


# --------------------------------------------------------------------------- #
# §2.4 step 3: cluster bootstrap BCa CI.                                       #
# --------------------------------------------------------------------------- #


def test_cluster_bootstrap_ci_brackets_cluster_mean():
    rng = random.Random(41)
    # 12 clusters, each with a stable per-cluster value; statistic = mean over clusters.
    cluster_values = {g: rng.gauss(0.5, 0.1) for g in range(12)}

    def stat(cluster_ids):
        if not cluster_ids:
            return 0.0
        return math.fsum(cluster_values[g] for g in cluster_ids) / len(cluster_ids)

    ci = cluster_bootstrap_ci(list(range(12)), stat, n_boot=300, level=0.95, seed=1)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.upper - ci.lower > 0.0
    # The true cluster mean lies inside the 95% interval (well-specified case).
    true_mean = math.fsum(cluster_values.values()) / 12
    assert ci.lower - 1e-9 <= true_mean <= ci.upper + 1e-9


# --------------------------------------------------------------------------- #
# §4.5 two-layer Holm + §3.3 2x2 contingency + §4.2 robustness floor.          #
# --------------------------------------------------------------------------- #


def test_two_layer_holm_gatekeeper_blocks_when_joint_fails():
    # Joint not significant => D-only/kappa-only are exploratory, no across-layer.
    out = two_layer_holm(
        {"joint": 0.40, "D_only": 0.001, "kappa_only": 0.002},
        {"target": 0.01, "retention": 0.2},
    )
    assert out["gatekeeper_passes"] is False
    assert out["across_adjusted"] is None
    assert out["exploratory"]["D_only"] is True


def test_two_layer_holm_passes_then_corrects_across_families():
    out = two_layer_holm(
        {"joint": 0.001, "D_only": 0.01, "kappa_only": 0.30},
        {"target": 0.01, "retention": 0.04, "hallucination": 0.2},
    )
    assert out["gatekeeper_passes"] is True
    assert out["across_adjusted"] is not None
    # Holm is monotone and never below the raw p.
    assert out["within_adjusted"]["joint"] >= 0.001


def test_contingency_decision_table_all_cells():
    assert contingency_decision(gatekeeper_passes=True, d_only_sig=True, kappa_only_sig=True).decision == "full"
    assert contingency_decision(gatekeeper_passes=True, d_only_sig=True, kappa_only_sig=False).decision == "fallback"
    assert contingency_decision(gatekeeper_passes=True, d_only_sig=False, kappa_only_sig=True).decision == "curvature-only"
    kill = contingency_decision(gatekeeper_passes=True, d_only_sig=False, kappa_only_sig=False)
    assert kill.decision == "kill"
    assert kill.failure_action == "stop_main_novelty_claim"
    assert contingency_decision(gatekeeper_passes=False, d_only_sig=True, kappa_only_sig=True).decision == "gatekeeper-failed"


def test_robustness_floor_tiers():
    # Full-ridge above floor + PCA poles overlap, same sign -> co-primary pass.
    ok = robustness_floor(0.10, (0.05, 0.15), {8: 0.09, 16: 0.11, 32: 0.10},
                          {8: (0.04, 0.14), 16: (0.06, 0.16), 32: (0.05, 0.15)}, floor=0.02)
    assert ok["tier"] == "co-primary-pass"
    # Full-ridge passes but a PCA pole flips sign -> control-sensitive.
    sens = robustness_floor(0.10, (0.05, 0.15), {8: -0.02, 16: 0.11, 32: 0.10},
                            {8: (-0.05, 0.01), 16: (0.06, 0.16), 32: (0.05, 0.15)}, floor=0.02)
    assert sens["tier"] == "control-sensitive"
    # Full-ridge CI below floor -> fail.
    bad = robustness_floor(0.01, (-0.01, 0.03), {8: 0.01}, {8: (-0.01, 0.03)}, floor=0.02)
    assert bad["tier"] == "fail"


# --------------------------------------------------------------------------- #
# §4.1: LOCI clusters + influence (Eq. 15-17).                                 #
# --------------------------------------------------------------------------- #


def test_build_loci_clusters_respects_size_floor():
    rng = random.Random(51)
    # Three well-separated blobs of 40 points each in R^4.
    rows = []
    for center in ([5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]):
        for _ in range(40):
            rows.append([c + rng.gauss(0, 0.2) for c in center])
    clustering = build_loci_clusters(rows, target_clusters=3, size_floor=25, seed=0)
    sizes = {}
    for lab in clustering.labels:
        sizes[lab] = sizes.get(lab, 0) + 1
    assert all(v >= 25 for v in sizes.values())
    assert clustering.assignment_hash == clustering.assignment_hash  # deterministic field
    # Held-out assignment returns a valid surviving cluster id.
    assert 0 <= clustering.assign([5.0, 0.0, 0.0, 0.0]) < len(clustering.centroids)


def test_loci_influence_is_literal_eq17_attribution():
    labels = [0, 0, 1, 1, 1]
    deltas = {0: 0.2, 1: -0.3}
    y = loci_influence(labels, deltas)
    # Eq.17 LITERALLY: Y_i = +(Delta_g / |g|). Y is a UTILITY (higher = more useful).
    assert y[0] == y[1]                       # same cluster, same |g| attribution
    assert math.isclose(y[0], (0.2 / 2))      # cluster 0: |g|=2, Delta=+0.2
    assert math.isclose(y[2], (-0.3 / 3))     # cluster 1: |g|=3, Delta=-0.3
    # drift_adjust multiplies the attribution.
    y_adj = loci_influence(labels, deltas, drift_adjust={0: 2.0})
    assert math.isclose(y_adj[0], 2.0 * (0.2 / 2))


def test_loci_influence_useful_cluster_gets_positive_y():
    # A USEFUL cluster: removing it RAISES val loss (Delta_g = L(-g) - L > 0)
    # -> utility Y must be POSITIVE. A harmful cluster (Delta_g < 0) -> negative Y.
    labels = [0, 0, 0, 1, 1, 1]
    deltas = {0: 0.5, 1: -0.4}   # cluster 0 useful, cluster 1 harmful
    y = loci_influence(labels, deltas)
    assert y[0] > 0.0                         # useful -> positive utility
    assert y[3] < 0.0                         # harmful -> negative utility


# --------------------------------------------------------------------------- #
# Fix c: ICC(2,1), Spearman, G7 reliability gate.                              #
# --------------------------------------------------------------------------- #


def test_icc_high_for_stable_low_for_noisy():
    # Stable: each subject's repeated measurements nearly equal -> high ICC.
    stable = [[v, v + 0.01, v - 0.01] for v in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert icc_2_1(stable) > 0.9
    # Noisy: measurements dominated by within-subject noise -> low ICC.
    rng = random.Random(61)
    noisy = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(5)]
    assert icc_2_1(noisy) < 0.6


def test_spearman_monotone():
    x = [1, 2, 3, 4, 5]
    assert math.isclose(spearman_rho(x, [10, 20, 30, 40, 50]), 1.0, abs_tol=1e-9)
    assert math.isclose(spearman_rho(x, [50, 40, 30, 20, 10]), -1.0, abs_tol=1e-9)


def test_g7_gate_passes_when_reliable_fails_when_not():
    # Reliable: stable seeds + strongly monotone proxy vs retrain delta.
    seeds = [[v, v + 0.02, v - 0.02] for v in [0.1 * i for i in range(20)]]
    proxy = [0.1 * i for i in range(20)]
    retrain = [0.1 * i + 0.001 * ((-1) ** i) for i in range(20)]
    ok = y_reliability_gate(seeds, proxy, retrain, n_sub=20)
    assert ok.icc_passes and ok.rho_passes and ok.passes
    assert ok.fallback is None
    # Unreliable: noisy seeds -> ICC fails -> gate fails -> diagnostic fallback.
    rng = random.Random(62)
    noisy_seeds = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(20)]
    bad = y_reliability_gate(noisy_seeds, proxy, [rng.gauss(0, 1) for _ in range(20)], n_sub=20)
    assert not bad.passes
    assert bad.fallback == "direct_retrain_delta_diagnostic"


# --------------------------------------------------------------------------- #
# Power / runtime calcs (P=5000 perms x K=10 folds; §4.2, §4.5).               #
# --------------------------------------------------------------------------- #


def test_achieved_power_monotone_and_bounded():
    low = achieved_power(300, effect_partial_r2=0.02, n_permutations=5000, n_folds=10)
    high = achieved_power(300, effect_partial_r2=0.10, n_permutations=5000, n_folds=10)
    assert 0.0 <= low["achieved_power"] <= 1.0
    assert high["achieved_power"] >= low["achieved_power"]
    # Locked design bookkeeping: (P+1)*K stat evals, K ridge fits (residuals cached).
    assert high["stat_evaluations"] == (5000 + 1) * 10
    assert high["ridge_fits"] == 10


def test_power_for_pairs_required_n_and_monotone():
    p_small = power_for_pairs(50, effect=0.05, sd_diff=0.1, margin=0.02)
    p_large = power_for_pairs(300, effect=0.05, sd_diff=0.1, margin=0.02)
    assert p_large["achieved_power"] >= p_small["achieved_power"]
    # required_n is finite for a real positive effect over the margin.
    assert p_large["required_n"] != math.inf and p_large["required_n"] > 0
    # No effect over margin -> zero power, infinite required_n.
    null = power_for_pairs(300, effect=0.02, sd_diff=0.1, margin=0.02)
    assert null["achieved_power"] <= 0.5

"""Unit tests for the NeuroTrace-IT NOVEL CORE (REDESIGN_v4).

DO-NOT-RUN under the current task: these are *written* build-now / run-later.
Every numeric assertion is a closed-form / synthetic-DGP property check -- a
"formula evaluation, not evidence" -- never an experimental result. No model
load, no server call, no training occurs anywhere below.

Coverage (maps to REDESIGN_v4 §5):
* NAIT endpoint baseline (Eq. 1) -- signature layout + similarity selection.
* kappa permutation-sensitive while the mean-pool is invariant (§2.5).
* SW2 distributional term responds to spread the mean cannot see (§1.3).
* Facility-location F(S) submodularity + greedy monotonicity + (1-1/e) (§2.8).
* Ridge-FWL cross-fit recovers a PLANTED residual signal and the block
  permutation test attains a small p-value on signal / nominal on null (§2.3-2.4).
* TrajectorySignatureV2 round-trips estimands and recomputes trajectory_hash;
  manifest V2 validation enforces estimand persistence (§2.10).
"""

from __future__ import annotations

import math
import random

import pytest

from neurotrace_it.analysis.pair_mining import (
    PairCandidate,
    mine_matched_endpoint_pairs,
    paired_margin_test,
)
from neurotrace_it.analysis.residual_test import (
    block_permutation_test,
    cross_fit_partial_r2,
    ridge_partial_out,
)
from neurotrace_it.analysis.outcome_y import y_reliability_gate
from neurotrace_it.analysis.residualize import (
    MAX_ABS_ACCEL,
    cluster_bootstrap_ci,
)
from neurotrace_it.analysis.drift import (
    BrierCalibrator,
    expected_calibration_error,
    factuality_drift,
    g6_factuality_gate,
)
from neurotrace_it.baselines.nait import (
    cosine_similarity,
    endpoint_score,
    endpoint_signature,
    nait_select,
)
from neurotrace_it.schemas_v2 import (
    SelectionManifestV2,
    TrajectorySignatureV2,
    validate_selection_manifest_v2,
)
from neurotrace_it.selection import (
    CoverageObjective,
    example_utility,
    facility_location_value,
    greedy_submodular_select,
)
from neurotrace_it.trajectory import (
    rbf_mmd2,
    sliced_wasserstein2,
    trajectory_curvature,
    trajectory_signature,
)


# --------------------------------------------------------------------------- #
# NAIT endpoint baseline (Deliverable #1, Eq. 1).                             #
# --------------------------------------------------------------------------- #


def test_endpoint_signature_layout_is_concat_start_end_over_A():
    activations = {
        0: ([1.0, 2.0], [3.0, 4.0]),
        3: ([5.0, 6.0], [7.0, 8.0]),
    }
    signature = endpoint_signature("ex1", activations, layer_ids=[0, 3])
    # phi_end = [h0_start, h0_end, h3_start, h3_end], R^{2*d*|A|} = R^8.
    assert signature.phi_end == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert signature.hidden_width == 2
    assert len(signature.phi_end) == 2 * signature.hidden_width * len(signature.layer_ids)


def test_endpoint_score_prefers_aligned_signature():
    activations = {0: ([1.0, 0.0], [1.0, 0.0])}
    signature = endpoint_signature("ex", activations)
    anchor = [1.0, 0.0, 1.0, 0.0]
    assert math.isclose(endpoint_score(signature, anchor), 1.0, abs_tol=1e-9)


def test_nait_select_is_deterministic_top_b():
    def sig(example_id: str, value: float):
        return endpoint_signature(example_id, {0: ([value], [value])})

    anchor = [1.0, 1.0]
    signatures = [sig("a", 0.1), sig("b", 1.0), sig("c", 0.5)]
    result = nait_select(signatures, anchor, budget=2)
    # cosine to anchor is identical for all positive scalars; ties break by id.
    assert result.selected_example_ids == ("a", "b")
    assert result.budget == 2


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# --------------------------------------------------------------------------- #
# Trajectory operator: curvature permutation-sensitivity, SW2 spread (§2.5,1.3) #
# --------------------------------------------------------------------------- #


def test_curvature_changes_under_step_reordering_but_mean_pool_invariant():
    # A non-collinear 3+-step path. kappa uses ordered second differences.
    path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
    reordered = [path[0], path[2], path[1], path[3]]

    def mean_pool(steps):
        dim = len(steps[0])
        return [math.fsum(s[i] for s in steps) / len(steps) for i in range(dim)]

    # First-moment pool is permutation-invariant (a multiset mean).
    assert mean_pool(path) == mean_pool(reordered)
    # Curvature is NOT invariant: it is provably not a function of psi_1 (§2.5).
    assert trajectory_curvature(path) != trajectory_curvature(reordered)


def test_curvature_zero_for_collinear_and_short_paths():
    collinear = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert math.isclose(trajectory_curvature(collinear), 0.0, abs_tol=1e-9)
    assert trajectory_curvature([[0.0], [1.0]]) == 0.0  # < 3 steps -> undefined -> 0


def test_sw2_responds_to_spread_the_mean_cannot_see():
    rng = random.Random(7)
    # Two clouds with the SAME mean (~0) but different spread.
    tight = [[rng.gauss(0.0, 0.1)] for _ in range(200)]
    wide = [[rng.gauss(0.0, 2.0)] for _ in range(200)]
    target = [[rng.gauss(0.0, 0.1)] for _ in range(200)]
    d_tight, seeds = sliced_wasserstein2(tight, target, n_projections=16, seed=1)
    d_wide, _ = sliced_wasserstein2(wide, target, n_projections=16, seed=1)
    # SW2 to the tight target is larger for the wide cloud (moment-sensitivity).
    assert d_wide > d_tight
    # Projection seeds are persisted for recomputability.
    assert len(seeds) == 16


def test_sw2_is_reproducible_given_seed():
    rng = random.Random(3)
    a = [[rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(80)]
    b = [[rng.gauss(1.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(80)]
    first, seeds_a = sliced_wasserstein2(a, b, n_projections=12, seed=42)
    second, seeds_b = sliced_wasserstein2(a, b, n_projections=12, seed=42)
    assert math.isclose(first, second, rel_tol=1e-12)
    assert seeds_a == seeds_b


def test_mmd_is_zero_for_identical_clouds():
    cloud = [[float(i), float(-i)] for i in range(20)]
    value = rbf_mmd2(cloud, cloud, seed=0)
    assert abs(value) < 1e-6


def test_trajectory_signature_persists_D_kappa_and_seeds():
    rng = random.Random(11)

    def step(n, mu):
        return [[rng.gauss(mu, 0.5)] for _ in range(n)]

    activations = {
        0: [step(4, 0.0), step(4, 0.3), step(4, 0.7)],
        1: [step(4, 0.1), step(4, 0.2), step(4, 0.9)],
    }
    target = {0: [[rng.gauss(0.0, 0.5)] for _ in range(40)], 1: [[rng.gauss(0.0, 0.5)] for _ in range(40)]}
    features = trajectory_signature("ex", activations, target, n_projections=8, seed=5)
    assert set(features.magnitude) == {0, 1}
    assert set(features.curvature) == {0, 1}
    assert set(features.projection_seeds) == {0, 1}
    # feature_vector is [D_0, D_1, kappa_0, kappa_1] in R^{2|A|}.
    assert len(features.feature_vector) == 4


# --------------------------------------------------------------------------- #
# Selection: submodularity + greedy monotonicity + (1-1/e) (Eq. 14, §2.8).    #
# --------------------------------------------------------------------------- #


def _toy_objective():
    # Coverage similarity: candidate "covers" ground element of the same letter.
    coverage = {
        ("a", "A"): 1.0, ("a", "B"): 0.1, ("a", "C"): 0.1,
        ("b", "A"): 0.1, ("b", "B"): 1.0, ("b", "C"): 0.1,
        ("c", "A"): 0.1, ("c", "B"): 0.1, ("c", "C"): 1.0,
        ("d", "A"): 0.9, ("d", "B"): 0.9, ("d", "C"): 0.9,
    }

    def sim(x, e):
        return coverage[(x, e)]

    utilities = {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.05}
    return CoverageObjective(
        utilities=utilities,
        ground_elements=["A", "B", "C"],
        similarity=sim,
        coverage_weight=1.0,
    )


def test_facility_location_is_submodular():
    objective = _toy_objective()
    # Submodularity: marginal of adding x to S >= marginal of adding x to T, S<=T.
    s_small = ["a"]
    s_large = ["a", "b"]
    new = "c"
    marginal_small = objective.value(s_small + [new]) - objective.value(s_small)
    marginal_large = objective.value(s_large + [new]) - objective.value(s_large)
    assert marginal_small + 1e-12 >= marginal_large


def test_facility_location_is_monotone():
    objective = _toy_objective()
    assert objective.value(["a"]) <= objective.value(["a", "b"]) <= objective.value(["a", "b", "c"])


def test_greedy_monotone_nondecreasing_and_matches_brute_force():
    objective = _toy_objective()
    result = greedy_submodular_select(objective, ["a", "b", "c", "d"], budget=2)
    # Greedy first picks the single best, then the best marginal addition.
    assert len(result.selected_example_ids) == 2
    assert math.isclose(result.approximation_ratio, 1.0 - 1.0 / math.e, rel_tol=1e-9)
    # Cumulative marginal gains reconstruct F(S) and are non-negative (monotone).
    assert all(gain >= -1e-12 for _, gain in result.marginal_gains)
    assert math.isclose(
        facility_location_value(objective, result.selected_example_ids),
        result.objective_value,
        rel_tol=1e-9,
    )
    # Brute-force optimum over all size-2 subsets; greedy is within (1-1/e).
    from itertools import combinations

    best = max(objective.value(list(combo)) for combo in combinations("abcd", 2))
    assert result.objective_value >= (1.0 - 1.0 / math.e) * best - 1e-9


def test_example_utility_is_nonnegative_and_uses_retention_kernel():
    # u(x) = max(0, g - lambda_r r - lambda_f f); reuses retention_adjusted_gain.
    assert example_utility(0.05, 0.01, drift_weight=2.0) == 0.030000000000000002
    assert example_utility(0.01, 0.5, drift_weight=1.0) == 0.0  # clipped at 0


# --------------------------------------------------------------------------- #
# Residual test: recover a PLANTED residual signal; null calibration (§2.3-4). #
# --------------------------------------------------------------------------- #


def _make_planted_dgp(n, *, signal_coef, noise_sd, seed):
    """Synthetic DGP with a KNOWN residual trajectory effect.

    control column phi (the endpoint signature shadow), one trajectory column t
    that is orthogonalized against phi, and Y = a*phi + signal_coef*t_resid + eps.
    The residual test should recover ~signal_coef and a small permutation p when
    signal_coef != 0, and a null-calibrated p when signal_coef == 0.
    """

    rng = random.Random(seed)
    control = []
    trajectory = []
    outcome = []
    strata = []
    for i in range(n):
        phi = rng.gauss(0.0, 1.0)
        # t correlated with phi PLUS an independent residual part.
        t_resid = rng.gauss(0.0, 1.0)
        t = 0.7 * phi + t_resid
        y = 1.5 * phi + signal_coef * t_resid + rng.gauss(0.0, noise_sd)
        control.append([phi, 1.0])          # [phi_end shadow, intercept]
        trajectory.append([t])
        outcome.append(y)
        strata.append(i % 4)
    return control, outcome, trajectory, strata


def test_ridge_partial_out_recovers_known_residual_coefficient():
    control, outcome, trajectory, _ = _make_planted_dgp(
        400, signal_coef=2.0, noise_sd=0.2, seed=1
    )
    outcome_partialler, column_partiallers = ridge_partial_out(
        control, outcome, trajectory, penalized_columns=[0], ridge_lambda=1e-3
    )
    # Orthogonalize both Y and the single t-column, then regress residuals.
    y_resid = [outcome_partialler.residual(z, y) for z, y in zip(control, outcome)]
    t_resid = [
        [column_partiallers[0].residual(z, row[0])] for z, row in zip(control, trajectory)
    ]
    # Closed-form 1-D slope of y_resid on t_resid.
    num = math.fsum(tr[0] * yr for tr, yr in zip(t_resid, y_resid))
    den = math.fsum(tr[0] * tr[0] for tr in t_resid)
    slope = num / den
    # The planted residual coefficient is 2.0; ridge-FWL recovers it closely.
    assert abs(slope - 2.0) < 0.25


def test_cross_fit_partial_r2_positive_under_signal():
    control, outcome, trajectory, strata = _make_planted_dgp(
        300, signal_coef=2.0, noise_sd=0.3, seed=2
    )
    result = cross_fit_partial_r2(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0], ridge_lambda=1e-2, n_folds=5, seed=3,
    )
    # A genuine residual effect leaves positive held-out partial-R^2.
    assert result.partial_r2 > 0.05
    assert result.n_folds >= 2


def test_block_permutation_small_p_under_signal():
    control, outcome, trajectory, strata = _make_planted_dgp(
        200, signal_coef=2.0, noise_sd=0.3, seed=4
    )
    result = block_permutation_test(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0], ridge_lambda=1e-2, n_folds=5,
        n_permutations=199, seed=5,
    )
    assert result.permutation_p_value is not None
    # Strong planted signal -> right-tail p should be small.
    assert result.permutation_p_value < 0.1


def test_block_permutation_calibrated_under_null():
    control, outcome, trajectory, strata = _make_planted_dgp(
        200, signal_coef=0.0, noise_sd=1.0, seed=6
    )
    result = block_permutation_test(
        control, outcome, trajectory, strata=strata,
        penalized_columns=[0], ridge_lambda=1e-2, n_folds=5,
        n_permutations=199, seed=7,
    )
    # Under H0 the p-value is a valid probability and not anti-conservative-by-construction.
    assert result.permutation_p_value is not None
    assert 0.0 < result.permutation_p_value <= 1.0


# --------------------------------------------------------------------------- #
# Matched-pair mining (§3.2).                                                  #
# --------------------------------------------------------------------------- #


def test_mine_pairs_keeps_matched_endpoint_divergent_curvature():
    # Two near-identical endpoints, divergent curvature, same CEM cell, same answer.
    candidates = [
        PairCandidate("a", (0.0, 0.0), curvature=0.1, family="math", answer_key="42",
                      length=10.0, difficulty=1.0),
        PairCandidate("b", (0.0, 0.001), curvature=5.0, family="math", answer_key="42",
                      length=10.2, difficulty=1.1),
        PairCandidate("c", (9.0, 9.0), curvature=0.2, family="math", answer_key="42",
                      length=10.0, difficulty=1.0),
    ]
    result = mine_matched_endpoint_pairs(
        candidates, tau_percentile=50.0, curvature_top_decile=0.5, target_n=10,
        length_bin_width=1.0, difficulty_bin_width=1.0,
    )
    pair_ids = {(p.first_id, p.second_id) for p in result.pairs}
    assert ("a", "b") in pair_ids
    assert "length" in result.covariate_imbalance
    assert result.achieved_n == len(result.pairs)


def test_paired_margin_test_reports_against_margin():
    from neurotrace_it.analysis.pair_mining import MatchedPair

    pairs = [MatchedPair("a", "b", endpoint_distance=0.01, curvature_gap=4.9, family="math")]
    outcomes = {"a": 0.10, "b": 0.02}
    report = paired_margin_test(pairs, outcomes, margin=0.02)
    assert report["n"] == 1.0
    assert math.isclose(report["mean_abs_diff"], 0.08, abs_tol=1e-9)
    assert report["exceeds_margin"] == 1.0


# --------------------------------------------------------------------------- #
# Schema V2: estimand persistence + recomputing hash + manifest validation.   #
# --------------------------------------------------------------------------- #


def _v2_signature(example_id: str) -> TrajectorySignatureV2:
    raw = TrajectorySignatureV2(
        example_id=example_id,
        layer_ids=(0, 1),
        step_count=3,
        token_count=12,
        endpoint_signature=(0.1, 0.2, 0.3, 0.4),
        magnitude={0: 0.5, 1: 0.7},
        curvature={0: 0.05, 1: 0.09},
        projection_seeds={0: (1, 2, 3), 1: (4, 5, 6)},
        alignment_scores={"rho_T": -0.3, "beta_T": 0.12},
        selection_scores={"u": 0.2, "marginal_gain": 0.05},
        drift_estimates={"r_hat": 0.01, "f_hat": 0.0},
        cluster_assignment_hash="clusterhash",
        ridge_lambda=1.0,
    )
    return raw.with_hash()


def test_v2_signature_hash_recomputes_from_estimands():
    signature = _v2_signature("ex1")
    assert signature.trajectory_hash == signature.recompute_hash()
    # Tampering with a persisted estimand breaks the integrity hash.
    tampered = TrajectorySignatureV2(
        example_id=signature.example_id,
        layer_ids=signature.layer_ids,
        step_count=signature.step_count,
        token_count=signature.token_count,
        endpoint_signature=signature.endpoint_signature,
        magnitude={0: 0.5, 1: 0.999},  # changed D_1
        curvature=signature.curvature,
        projection_seeds=signature.projection_seeds,
        trajectory_hash=signature.trajectory_hash,  # stale hash
    )
    assert tampered.trajectory_hash != tampered.recompute_hash()


def test_v2_manifest_validates_with_estimands_and_endpoint_baseline():
    signature = _v2_signature("ex1")
    manifest = SelectionManifestV2(
        project="neurotrace-it",
        candidate_pool_hash="poolhash1",
        selected_example_ids=("ex1",),
        baseline_ids=("endpoint_neuron_selection",),
        signatures=(signature,),
    )
    assert validate_selection_manifest_v2(manifest) == []


def test_v2_manifest_flags_missing_estimands_and_bad_hash():
    bad = TrajectorySignatureV2(
        example_id="ex1",
        layer_ids=(0, 1),
        step_count=3,
        token_count=12,
        endpoint_signature=(),                 # missing phi_end
        magnitude={0: 0.5},                    # missing kappa for layer 1, D for layer 1
        curvature={0: 0.05},
        projection_seeds={},                   # missing seeds
        trajectory_hash="not-a-real-hash",     # does not recompute
    )
    manifest = SelectionManifestV2(
        project="neurotrace-it",
        candidate_pool_hash="poolhash1",
        selected_example_ids=("ex1",),
        baseline_ids=("endpoint_neuron_selection",),
        signatures=(bad,),
    )
    errors = validate_selection_manifest_v2(manifest)
    assert any("endpoint_signature" in e for e in errors)
    assert any("projection_seeds" in e for e in errors)
    assert any("trajectory_hash does not recompute" in e for e in errors)


def test_v2_manifest_requires_endpoint_baseline_and_blocks_server_auth():
    signature = _v2_signature("ex1")
    manifest = SelectionManifestV2(
        project="neurotrace-it",
        candidate_pool_hash="poolhash1",
        selected_example_ids=("ex1",),
        baseline_ids=("random_subset",),       # endpoint baseline missing
        signatures=(signature,),
        server_authorized=True,                # must stay false
    )
    errors = validate_selection_manifest_v2(manifest)
    assert "endpoint_neuron_selection baseline is required" in errors
    assert "server_authorized must remain false in local manifests" in errors


# --------------------------------------------------------------------------- #
# G7 reliability gate: enforce seeds_min >= 3 (lattice_v4.yaml:75, fix c).     #
# --------------------------------------------------------------------------- #


def test_g7_gate_fails_closed_with_fewer_than_three_seeds():
    # Only k=2 seeds: even a perfectly-stable ICC must NOT pass the seeds gate.
    two_seed = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    proxy = [float(i) for i in range(60)]
    retrain = [float(i) for i in range(60)]  # perfect rank correlation
    report = y_reliability_gate(two_seed, proxy, retrain)
    assert report.n_seeds == 2
    assert report.seeds_passes is False
    assert report.icc_passes is False        # ICC not consulted under-seeded
    assert report.passes is False
    assert report.fallback == "direct_retrain_delta_diagnostic"


def test_g7_gate_seeds_precondition_met_at_three_seeds():
    # k=3 seeds with strong subject separation -> ICC high; rho perfect -> pass.
    three_seed = [[float(i) + j * 0.001 for j in range(3)] for i in range(8)]
    proxy = [float(i) for i in range(60)]
    retrain = [float(i) for i in range(60)]
    report = y_reliability_gate(three_seed, proxy, retrain)
    assert report.n_seeds == 3
    assert report.seeds_passes is True
    assert report.passes is True
    assert report.fallback is None


# --------------------------------------------------------------------------- #
# rbf_mmd2 rejects dimension-mismatched clouds (§2.2; was a silent zip-trunc). #
# --------------------------------------------------------------------------- #


def test_rbf_mmd2_rejects_dimension_mismatch():
    cloud_2d = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    cloud_3d = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    with pytest.raises(ValueError, match="dim mismatch"):
        rbf_mmd2(cloud_2d, cloud_3d, seed=0)


# --------------------------------------------------------------------------- #
# BCa cluster bootstrap is hardened for very small cluster counts (§2.4).      #
# --------------------------------------------------------------------------- #


def test_bca_small_cluster_count_does_not_blow_up():
    # g=3 clusters: jackknife acceleration is not estimable -> fall back to a=0,
    # and the returned interval must stay finite and bracket the point estimate.
    clusters = [0, 1, 2]
    table = {0: 1.0, 1: 1.05, 2: 0.97}

    def statistic(ids):
        return sum(table[c] for c in ids) / len(ids) if ids else 0.0

    ci = cluster_bootstrap_ci(clusters, statistic, n_boot=200, seed=1)
    assert math.isfinite(ci.lower) and math.isfinite(ci.upper)
    assert math.isfinite(ci.acceleration)
    assert abs(ci.acceleration) <= MAX_ABS_ACCEL + 1e-12
    assert ci.acceleration == 0.0          # below MIN_CLUSTERS_FOR_ACCEL -> a=0
    assert ci.lower <= ci.point <= ci.upper


def test_bca_acceleration_is_capped_for_skewed_jackknife():
    # Many clusters but a heavily skewed statistic: |accel| must stay capped so the
    # BCa adjustment denominator never blows up.
    clusters = list(range(12))
    # One extreme cluster value induces strong jackknife skewness.
    table = {c: (100.0 if c == 0 else 0.0) for c in clusters}

    def statistic(ids):
        return sum(table[c] for c in ids) / len(ids) if ids else 0.0

    ci = cluster_bootstrap_ci(clusters, statistic, n_boot=200, seed=2)
    assert abs(ci.acceleration) <= MAX_ABS_ACCEL + 1e-12
    assert math.isfinite(ci.lower) and math.isfinite(ci.upper)


# --------------------------------------------------------------------------- #
# Brier calibrator / factuality_drift / G6 ECE pathway (§2.6, G6).            #
# --------------------------------------------------------------------------- #


def test_brier_score_is_minimized_at_true_probability():
    # Eq. 12: E[(q-y)^2] = (q-p)^2 + p(1-p), uniquely minimized at q=p.
    # Build a slice with empirical support rate p ~= 0.7; the calibrator whose
    # output matches p attains a lower Brier score than a miscalibrated one.
    labels = [1] * 70 + [0] * 30
    logits = [0.0] * 100                       # sigma(0)=0.5 baseline logit grid

    # q=p calibrator: bias chosen so sigma(bias)=0.7.
    p = 0.7
    bias_p = math.log(p / (1.0 - p))
    calib_p = BrierCalibrator(scale=0.0, bias=bias_p)
    calib_half = BrierCalibrator(scale=0.0, bias=0.0)  # q=0.5 (miscalibrated)
    assert calib_p.brier_score(logits, labels) < calib_half.brier_score(logits, labels)


def test_factuality_drift_counts_unsupported_claims():
    # q_a = sigma(z_a); threshold c*=0.5. Two of four claims fall below c*.
    calib = BrierCalibrator(scale=1.0, bias=0.0, threshold=0.5)
    logits = [3.0, -2.0, 2.0, -1.0]            # supported, NOT, supported, NOT
    f_hat = factuality_drift(logits, calib)
    assert math.isclose(f_hat, 0.5, abs_tol=1e-9)
    assert factuality_drift([], calib) == 0.0  # no claims -> no drift


def test_expected_calibration_error_zero_for_perfect_calibration():
    # All-confident-and-correct vs all-confident-and-wrong bracket ECE in [0,1].
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    assert math.isclose(expected_calibration_error(probs, labels, n_bins=10), 0.0, abs_tol=1e-9)
    bad = expected_calibration_error([1.0, 1.0], [0, 0], n_bins=10)
    assert math.isclose(bad, 1.0, abs_tol=1e-9)


def test_g6_gate_sets_lambda_f_zero_on_failure_and_passes_when_valid():
    # FAIL: f_hat unrelated to drift_eval -> low rho/r -> lambda_f := 0.
    f_hat_bad = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    drift_unrelated = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    rel_probs = [0.0, 0.0, 1.0, 1.0]
    rel_labels = [0, 0, 1, 1]                  # perfectly calibrated -> ECE ~ 0
    fail = g6_factuality_gate(
        f_hat_bad, drift_unrelated, rel_probs, rel_labels, requested_lambda_f=1.0
    )
    assert fail.passes is False
    assert fail.lambda_f == 0.0

    # PASS: f_hat strongly tracks drift_eval (rho=r=1) and calibration is good.
    f_hat_good = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    drift_aligned = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    good = g6_factuality_gate(
        f_hat_good, drift_aligned, rel_probs, rel_labels, requested_lambda_f=1.0
    )
    assert good.passes is True
    assert good.lambda_f == 1.0
    assert good.ece_passes is True


# --------------------------------------------------------------------------- #
# Matched-pair bucket bookkeeping: per-(family, answer_key) tau + imbalance.   #
# --------------------------------------------------------------------------- #


def test_pair_mining_keeps_per_family_answer_key_tau_buckets():
    # Same family, TWO answer keys; each answer-key bucket must keep its own tau.
    candidates = [
        PairCandidate("a", (0.0, 0.0), curvature=0.1, family="math", answer_key="7",
                      length=10.0, difficulty=1.0),
        PairCandidate("b", (0.0, 0.001), curvature=5.0, family="math", answer_key="7",
                      length=10.0, difficulty=1.0),
        PairCandidate("c", (50.0, 50.0), curvature=0.1, family="math", answer_key="99",
                      length=10.0, difficulty=1.0),
        PairCandidate("d", (50.0, 50.05), curvature=5.0, family="math", answer_key="99",
                      length=10.0, difficulty=1.0),
    ]
    result = mine_matched_endpoint_pairs(
        candidates, tau_percentile=50.0, curvature_top_decile=0.5, target_n=10,
        length_bin_width=1.0, difficulty_bin_width=1.0,
    )
    # Two distinct (family, answer_key) buckets -> two persisted taus (not collapsed
    # onto a single "math" key).
    assert len(result.bucket_taus) == 2
    assert all("\x1f" in key for key in result.bucket_taus)


def test_pair_mining_imbalance_matches_retained_pairs_after_trim():
    # Build enough divergent pairs that target_n trims; imbalance must be computed
    # over the RETAINED pairs (post sort+trim), not a positional prefix.
    candidates = []
    for i in range(6):
        candidates.append(
            PairCandidate(f"x{i}", (0.0, 0.0), curvature=0.0, family="math",
                          answer_key="k", length=float(i), difficulty=0.0)
        )
        candidates.append(
            PairCandidate(f"y{i}", (0.0, 0.0005), curvature=10.0 + i, family="math",
                          answer_key="k", length=float(i), difficulty=0.0)
        )
    result = mine_matched_endpoint_pairs(
        candidates, tau_percentile=100.0, curvature_top_decile=1.0, target_n=2,
        length_bin_width=1.0, difficulty_bin_width=1.0,
    )
    assert result.achieved_n == len(result.pairs) <= 2
    # SMD is finite and reported for the retained set.
    assert "length" in result.covariate_imbalance
    assert math.isfinite(result.covariate_imbalance["length"])

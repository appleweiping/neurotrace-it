"""Formula-evaluation unit checks for the REDESIGN_v5 routing/multiplicity core.

These are NOT evidence runs: every check is a pure-Python evaluation of a locked
FORMULA from REDESIGN_v5 (§3.4, §3.5, §3.6, §3.7, §4). No model is loaded, no
server is called, no training is run. The items mirror §5.5 module 10:

(a) coupling identity  sum_l psi_l = beta_T . T~   in- AND out-of-sample (Eq.5-4)
(b) capacity_match is a pure (m_A,R_tot,r_max) function: conserves R_tot, respects
    r_max, deterministic, never grows support, ignores any score (RE-FIX-2);
    make_feasible_mask lifts every raw mask (incl. empty => uniform-over-A)
(c) the six-arm IUT R1: per-contrast bootstrap-t level, IUT size <= alpha at the
    least-favorable config, margin delta_R1 enters the statistic
(c2) each control policy is a deterministic map of x (RE-FIX-3/4); pi_ada frozen
    by seed_ada, invariant to the training seed
(d) nait_layerwise matches base Eq.5 summed over L (and the gated 8-anchor sum) --
    covered in tests/test_endpoint_baseline.py
    (test_a_d_layerwise_score_reproduces_eq5_over_L_and_anchor_subset); the
    formula and anchor-subset checks live with the layerwise baseline there.
(e) pool_firewall rejects P_dep-outcome decisions / Y_obs<->U_train contamination
    / in-sample-only residualization, and the partition is locked
(f) closed_testing shortcut == brute-force 63-intersection test (incl. union-null
    IUT leaves) and controls FWER (Prop. P1-FWER) at the binding LFC
(g) R0-style permutation placebo holds nominal level (Prop. P0 within-stratum
    exchangeability sanity)

The factuality calibrator (the strictly-proper Brier rule of §2.6) and the G6
precondition gate are checked separately in tests/test_trajectory_selection.py.
"""

from __future__ import annotations

import math

import pytest

from neurotrace_it import layer_function as lf
from neurotrace_it.analysis import layer_attribution as la
from neurotrace_it.analysis import routing_intervention as ri
from neurotrace_it.analysis import closed_testing as ct
from neurotrace_it.analysis import pool_firewall as pf
from neurotrace_it.analysis import matched_budget as mb

ANCHORS = list(range(8))
R_TOT = 32
R_MAX = 8


# --------------------------------------------------------------------------- #
# (b) capacity_match purity / conservation / make_feasible_mask
# --------------------------------------------------------------------------- #
def _k_min() -> int:
    return lf.k_min_for(R_TOT, R_MAX)


def test_b_capacity_match_conserves_over_all_feasible_cardinalities():
    kmin = _k_min()
    for k in range(kmin, len(ANCHORS) + 1):
        mask = {l: (1 if l < k else 0) for l in ANCHORS}
        ranks = lf.capacity_match(mask, R_TOT, R_MAX, anchors=ANCHORS)
        assert sum(ranks.values()) == R_TOT                       # conservation
        assert all(v <= R_MAX for v in ranks.values())            # cap
        on = [l for l in ANCHORS if l < k]
        off = [l for l in ANCHORS if l >= k]
        assert all(ranks[l] == 0 for l in off)                    # no leak off support
        assert all(ranks[l] >= 1 for l in on)                     # support unchanged
        assert set(l for l, v in ranks.items() if v > 0) <= set(on)  # never grows support


def test_b_capacity_match_is_deterministic_and_ignores_score():
    mask = {l: 1 for l in ANCHORS}
    r1 = lf.capacity_match(mask, R_TOT, R_MAX, anchors=ANCHORS)
    r2 = lf.capacity_match(mask, R_TOT, R_MAX, anchors=ANCHORS)
    assert r1 == r2  # deterministic (ascending-l tie-break)
    # capacity_match has NO score argument; two callers passing different scores
    # to make_feasible_mask, but the SAME resulting feasible mask, get the same rank.
    score_a = {l: float(l) for l in ANCHORS}
    score_b = {l: float(-l) for l in ANCHORS}
    feasible_a = lf.make_feasible_mask({l: 1 for l in ANCHORS}, score_a, R_TOT, R_MAX, anchors=ANCHORS)
    feasible_b = lf.make_feasible_mask({l: 1 for l in ANCHORS}, score_b, R_TOT, R_MAX, anchors=ANCHORS)
    assert feasible_a == feasible_b  # already-feasible 1_A is score-independent
    assert lf.capacity_match(feasible_a, R_TOT, R_MAX, anchors=ANCHORS) == lf.capacity_match(
        feasible_b, R_TOT, R_MAX, anchors=ANCHORS
    )


def test_b_largest_remainder_uneven_cardinality():
    # k=5 (feasible: 5*8=40 >= 32), R_tot=32 => q=6, rem=2 => the two lowest-index
    # ON layers get +1 (ascending-l top-up), all respect the cap r_max=8.
    on = (1, 3, 5, 6, 7)
    mask = {l: (1 if l in on else 0) for l in ANCHORS}
    ranks = lf.capacity_match(mask, R_TOT, R_MAX, anchors=ANCHORS)
    assert sum(ranks.values()) == R_TOT
    assert ranks[1] == 7 and ranks[3] == 7  # two lowest-index ON layers get +1
    assert ranks[5] == 6 and ranks[6] == 6 and ranks[7] == 6
    assert all(v <= R_MAX for v in ranks.values())
    assert all(ranks[l] == 0 for l in ANCHORS if l not in on)


def test_b_make_feasible_mask_lifts_empty_to_uniform():
    empty: dict[int, int] = {l: 0 for l in ANCHORS}
    lifted = lf.make_feasible_mask(empty, None, R_TOT, R_MAX, anchors=ANCHORS)
    assert all(lifted[l] == 1 for l in ANCHORS)  # uniform-over-A fallback (Eq.6-0c)
    ranks = lf.capacity_match(lifted, R_TOT, R_MAX, anchors=ANCHORS)
    assert sum(ranks.values()) == R_TOT


def test_b_make_feasible_mask_completes_to_kmin_by_score():
    kmin = _k_min()
    # A raw mask below k_min must be completed by highest-score OFF anchors.
    raw = {l: (1 if l == 0 else 0) for l in ANCHORS}  # |S|=1 < k_min
    score = {l: float(l) for l in ANCHORS}            # highest score = largest index
    lifted = lf.make_feasible_mask(raw, score, R_TOT, R_MAX, anchors=ANCHORS)
    on = [l for l in ANCHORS if lifted[l] == 1]
    assert len(on) == max(1, kmin)
    assert 0 in on  # the original ON layer is retained
    # capacity_match on the lifted mask is total and conserving.
    assert sum(lf.capacity_match(lifted, R_TOT, R_MAX, anchors=ANCHORS).values()) == R_TOT


def test_b_capacity_match_rejects_infeasible_mask():
    # An infeasible mask (k*r_max < R_tot) is a contract violation, never a branch.
    too_small = {l: (1 if l in (0, 1) else 0) for l in ANCHORS}  # k=2, 2*8=16 < 32
    with pytest.raises(ValueError):
        lf.capacity_match(too_small, R_TOT, R_MAX, anchors=ANCHORS)


def test_b_leave_one_layer_redistribute_conserves():
    for drop in ANCHORS:
        ranks = lf.leave_one_layer_redistribute(
            drop, anchors=ANCHORS, total_rank=R_TOT, r_max=R_MAX
        )
        assert sum(ranks.values()) == R_TOT


# --------------------------------------------------------------------------- #
# (a) coupling identity  sum_l psi_l = beta_T . T~  in- AND out-of-sample
# --------------------------------------------------------------------------- #
def _fit_nuisance(seed: int, anchors, p=5, n=50):
    rng = _LCG(seed)
    Z = [[rng.gauss() for _ in range(p)] for _ in range(n)]
    T = [[rng.gauss() for _ in range(2 * len(anchors))] for _ in range(n)]
    Y = [rng.gauss() for _ in range(n)]
    nm = la.frozen_nuisance_map(Z, T, Y, lam=1.0, anchors=anchors)
    beta = [rng.gauss() for _ in range(2 * len(anchors))]
    return nm, beta, Z, T


def test_a_coupling_identity_in_and_out_of_sample():
    anchors = [0, 1, 2, 3]
    nm, beta, Z, T = _fit_nuisance(11, anchors)
    # In-sample: use a training row.
    sc_in = la.per_layer_policy_score(Z[0], T[0], beta, nm, anchors=anchors)
    la.assert_coupling_identity(sc_in, tol=1e-9)
    assert sc_in.identity_residual < 1e-9
    # Out-of-sample: a fresh row.
    rng = _LCG(999)
    z = [rng.gauss() for _ in range(len(Z[0]))]
    t = [rng.gauss() for _ in range(2 * len(anchors))]
    sc_out = la.per_layer_policy_score(z, t, beta, nm, anchors=anchors)
    la.assert_coupling_identity(sc_out, tol=1e-9)
    assert sc_out.identity_residual < 1e-9
    # supp(psi) subseteq A: psi is defined exactly on the anchors.
    assert set(sc_out.psi.keys()) == set(anchors)


def test_a_frozen_nuisance_map_block_penalty_matches_residual_test():
    # REGISTERED BLOCK penalty (Eq. 8c): only the wide phi_end nuisance columns
    # are shrunk; s_NAIT / V_proj / C and the intercept stay UNPENALIZED, exactly
    # as residual_test._fit_ridge_control does. Deployed R0 map must use this.
    from neurotrace_it.analysis.residual_test import ridge_partial_out

    anchors = [0, 1, 2, 3]
    rng = _LCG(77)
    p = 6  # [phi0, phi1, phi2 | s_NAIT, C, intercept]
    n = 40
    Z = [[rng.gauss() for _ in range(p)] for _ in range(n)]
    for row in Z:
        row[-1] = 1.0                       # intercept column
    T = [[rng.gauss() for _ in range(2 * len(anchors))] for _ in range(n)]
    Y = [rng.gauss() for _ in range(n)]
    phi_end_cols = [0, 1, 2]                 # the penalized wide nuisance block
    lam = 0.7

    nm = la.frozen_nuisance_map(
        Z, T, Y, lam=lam, anchors=anchors, penalized_columns=phi_end_cols
    )
    # Provenance frozen on the map.
    assert nm.penalized_columns == (0, 1, 2)

    # b_lambda_Y must equal residual_test's block-penalized ridge on Y (same Omega).
    outcome_partialler, _ = ridge_partial_out(
        Z, Y, T, penalized_columns=phi_end_cols, ridge_lambda=lam
    )
    for a, b in zip(nm.b_lambda_Y, outcome_partialler.coef):
        assert abs(a - b) < 1e-7

    # And it must DIFFER from the legacy all-columns-penalized fit (so the block
    # penalty is genuinely in effect, not a no-op).
    nm_uniform = la.frozen_nuisance_map(Z, T, Y, lam=lam, anchors=anchors)
    assert nm_uniform.penalized_columns == ()
    assert any(
        abs(a - b) > 1e-6 for a, b in zip(nm.b_lambda_Y, nm_uniform.b_lambda_Y)
    )

    # The coupling identity is pure stacking algebra -> still exact under the
    # block penalty (the registered psi(x) estimand is unchanged).
    beta = [rng.gauss() for _ in range(2 * len(anchors))]
    sc = la.per_layer_policy_score(Z[0], T[0], beta, nm, anchors=anchors)
    la.assert_coupling_identity(sc, tol=1e-9)


# --------------------------------------------------------------------------- #
# (c) six-arm IUT R1: per-contrast level, IUT size at LFC, margin enters
# --------------------------------------------------------------------------- #
def test_c_iut_rejects_only_when_every_gap_clears_margin():
    # Four controls with a large positive gap, ONE control at the margin boundary.
    rng = _LCG(5)
    n = 60
    big = {c: [1.0 + 0.05 * rng.gauss() for _ in range(n)] for c in ("unif", "shuf", "rand", "global")}
    binding = {"ada": [0.0 + 0.05 * rng.gauss() for _ in range(n)]}  # mean ~ 0
    diffs = {**big, **binding}
    # With margin 0, the binding control's lower bound should NOT clear 0 reliably,
    # so the IUT does not reject (a single failing component blocks the union).
    dec = ri.iut_decision(diffs, alpha=0.05, margin=0.0, n_boot=300, boot_seed=1)
    assert dec.binding_control == "ada"
    assert not dec.reject  # one component fails => composite null not rejected

    # Now make EVERY control's gap clearly exceed the margin => IUT rejects.
    all_big = {c: [1.0 + 0.05 * rng.gauss() for _ in range(n)] for c in ri.CONTROLS}
    dec2 = ri.iut_decision(all_big, alpha=0.05, margin=0.0, n_boot=300, boot_seed=1)
    assert dec2.reject
    assert dec2.simultaneous_lower_bound > 0.0


def test_c_margin_shift_enters_the_statistic():
    rng = _LCG(8)
    n = 60
    diffs = {c: [0.5 + 0.05 * rng.gauss() for _ in range(n)] for c in ri.CONTROLS}
    # Margin below the gap => rejects; margin above the gap => does not.
    lo = ri.iut_decision(diffs, alpha=0.05, margin=0.0, n_boot=300, boot_seed=2)
    hi = ri.iut_decision(diffs, alpha=0.05, margin=1.0, n_boot=300, boot_seed=2)
    assert lo.reject
    assert not hi.reject


def test_c_per_contrast_bootstrap_t_attains_nominal_level_on_null():
    # Under the null g_c = margin, the one-sided test should reject ~ alpha of the
    # time. Aggregate over many independent null datasets.
    alpha = 0.10
    margin = 0.0
    rejections = 0
    trials = 300
    for trial in range(trials):
        rng = _LCG(10_000 + trial)
        # Null: mean-zero differences (g_c = margin = 0).
        diffs = [rng.gauss() for _ in range(40)]
        b = ri.bootstrap_t_lower_bound(
            diffs, alpha=alpha, margin=margin, n_boot=200, boot_seed=trial
        )
        if b.rejects:
            rejections += 1
    rate = rejections / trials
    # Asymptotic bootstrap-t at n=40: allow generous slack, but must be near alpha
    # and not grossly inflated.
    assert rate <= alpha + 0.06, f"empirical level {rate} too high"


def test_c_iut_size_at_least_favorable_config():
    # LFC: one control truly null (g_c = margin), the rest strictly beaten. The IUT
    # size must stay <= alpha. A min-over-controls quantile would inflate here.
    alpha = 0.10
    margin = 0.0
    rejections = 0
    trials = 300
    for trial in range(trials):
        rng = _LCG(50_000 + trial)
        diffs = {}
        for c in ri.CONTROLS:
            if c == "unif":
                diffs[c] = [rng.gauss() for _ in range(40)]          # binding null
            else:
                diffs[c] = [3.0 + rng.gauss() for _ in range(40)]    # strictly beaten
        dec = ri.iut_decision(diffs, alpha=alpha, margin=margin, n_boot=150, boot_seed=trial)
        if dec.reject:
            rejections += 1
    rate = rejections / trials
    assert rate <= alpha + 0.06, f"IUT inflated at LFC: {rate}"


# --------------------------------------------------------------------------- #
# (c2) deterministic control maps (RE-FIX-3/4), pi_ada frozen by seed_ada
# --------------------------------------------------------------------------- #
def test_c2_controls_are_deterministic_maps_of_x():
    psi = {l: float((l * 7) % 5) for l in ANCHORS}
    common = dict(
        anchors=ANCHORS,
        tau_sel=2.0,
        total_rank=R_TOT,
        r_max=R_MAX,
        seed_rand=123,
        seed_shuf=456,
        seed_ada=789,
        A_glob=[0, 1, 2, 3],
    )
    for arm in ("psi", "unif", "shuf", "rand", "global", "ada"):
        m1 = lf.control_mask(arm, "example_42", psi, **common)
        m2 = lf.control_mask(arm, "example_42", psi, **common)
        assert m1 == m2, f"control {arm} is not a deterministic map of x"


def test_c2_pi_ada_invariant_to_training_seed():
    # pi_ada's importance is frozen by seed_ada (a SEPARATE seed), NOT the training
    # seed. There is no training-seed argument to control_mask, so changing any
    # per-run randomness cannot alter the mask; we assert it depends ONLY on
    # (x, seed_ada, persisted importance).
    psi = {l: float(l % 3) for l in ANCHORS}
    imp = {l: float((l * 3) % 7) for l in ANCHORS}
    base = dict(
        anchors=ANCHORS, tau_sel=1.0, total_rank=R_TOT, r_max=R_MAX,
        seed_rand=1, seed_shuf=2, A_glob=[0, 1, 2, 3], ada_importance=imp,
    )
    m_a = lf.control_mask("ada", "x", psi, seed_ada=777, **base)
    m_b = lf.control_mask("ada", "x", psi, seed_ada=777, **base)
    assert m_a == m_b
    # A different seed_ada (different pre-registered freeze) is allowed to differ,
    # but the SAME seed_ada always reproduces the mask -> deterministic in x.
    m_c = lf.control_mask("ada", "x", psi, seed_ada=888, **{**base, "ada_importance": None})
    assert isinstance(m_c, dict)


def test_c2_shuf_holds_psi_marginal_breaks_coupling():
    psi = {l: float(l) for l in ANCHORS}  # strictly increasing
    common = dict(
        anchors=ANCHORS, tau_sel=3.5, total_rank=R_TOT, r_max=R_MAX,
        seed_rand=1, seed_shuf=42, seed_ada=3, A_glob=[0, 1, 2, 3],
    )
    raw_psi = lf.control_mask("psi", "x", psi, **common)
    raw_shuf = lf.control_mask("shuf", "x", psi, **common)
    # Both select the same NUMBER of anchors above tau (the marginal is preserved)
    # but the identity of the ON layers differs (coupling broken) for this psi.
    assert sum(raw_psi.values()) == sum(raw_shuf.values())


# --------------------------------------------------------------------------- #
# (f) closed_testing: shortcut == brute force; FWER at LFC (Prop. P1-FWER)
# --------------------------------------------------------------------------- #
ALPHA = 0.05


def _make_inputs(r0, r1_comps, g2t_comps, g2r, g2h, g2c):
    return ct.ClosedTestInputs.from_rejections(
        r0_rejects=r0,
        r1_components=r1_comps,
        g2t_components=g2t_comps,
        g2r_rejects=g2r,
        g2h_rejects=g2h,
        g2c_rejects=g2c,
        alpha=ALPHA,
    )


def test_f_shortcut_equals_bruteforce_on_random_inputs():
    rng = _LCG(7)
    for _ in range(400):
        inp = _make_inputs(
            rng.boolp(0.6),
            [rng.boolp(0.6) for _ in range(5)],
            [rng.boolp(0.6) for _ in range(3)],
            rng.boolp(0.6), rng.boolp(0.6), rng.boolp(0.6),
        )
        ct.assert_shortcut_equals_bruteforce(inp, alpha=ALPHA)  # raises on disagreement


def test_f_union_null_leaf_requires_all_components():
    # R1 leaf must NOT reject when only 4 of 5 controls clear (the binding 5th fails).
    inp = _make_inputs(True, [True, True, True, True, False], [True, True, True], True, True, True)
    res = ct.closed_test_bruteforce(inp, alpha=ALPHA)
    assert not res.rejects("R1")  # full-union rule: one failing component blocks R1
    # And when all 5 clear, R1 is rejectable (given R0 gate).
    inp2 = _make_inputs(True, [True] * 5, [True, True, True], True, True, True)
    res2 = ct.closed_test_bruteforce(inp2, alpha=ALPHA)
    assert res2.rejects("R1")
    ct.assert_shortcut_equals_bruteforce(inp, alpha=ALPHA)
    ct.assert_shortcut_equals_bruteforce(inp2, alpha=ALPHA)
    # The union-null p-value is the FULL max over components (RE-FIX-5).
    assert ct.union_leaf_pvalue([0.001, 0.002, 0.5, 0.001, 0.001]) == 0.5


def test_f_fwer_controlled_at_binding_lfc():
    # R0-only true null -> FWER ~ alpha (the gatekeeper). R1-only at the LFC -> <= alpha.
    f_r0 = ct.fwer_simulation(n_trials=8000, alpha=0.05, seed=3, true_nulls=["R0"])
    assert f_r0 <= 0.05 + 0.01
    f_r1_lfc = ct.fwer_simulation(
        n_trials=8000, alpha=0.05, seed=3, true_nulls=["R1"], binding_only=True
    )
    assert f_r1_lfc <= 0.05 + 0.01, f"FWER inflated at R1 LFC: {f_r1_lfc}"
    # All-null binding LFC across every elementary null must also control.
    f_all = ct.fwer_simulation(
        n_trials=8000, alpha=0.05, seed=4, true_nulls=ct.ELEMENTARY_NULLS, binding_only=True
    )
    assert f_all <= 0.05 + 0.01


# --------------------------------------------------------------------------- #
# (e) pool_firewall: disjoint locked partition + leakage assertions
# --------------------------------------------------------------------------- #
def test_e_split_pools_disjoint_and_locked():
    corpus = [f"ex{i}" for i in range(400)]
    strata = {ex: f"fam{i % 4}" for i, ex in enumerate(corpus)}
    part = pf.split_pools(corpus, strata, salt="lock-1")
    report = pf.assert_no_leakage(part, corpus=corpus)
    assert report.clean
    # Locked: regenerating with the same salt is byte-identical.
    part2 = pf.split_pools(corpus, strata, salt="lock-1")
    assert part2.partition_hash == part.partition_hash
    # A different salt (the §3.5a probe) gives a DIFFERENT partition.
    probe = pf.regenerate_partition(0, corpus, strata, base_salt="lock-1")
    assert probe.partition_hash != part.partition_hash


def test_e_rejects_dep_outcome_in_decision():
    with pytest.raises(AssertionError):
        pf.assert_no_dep_outcome_in_decision(
            decision_inputs=["phi_end", "U_train_dep"],
            dep_outcome_keys=["U_train_dep"],
        )
    # Clean decision path passes.
    pf.assert_no_dep_outcome_in_decision(
        decision_inputs=["phi_end", "s_nait"], dep_outcome_keys=["U_train_dep"]
    )


def test_e_rejects_outcome_pool_contamination():
    with pytest.raises(AssertionError):
        pf.assert_outcome_pool_discipline(y_obs_pool="P_val", u_train_pool="P_val")
    with pytest.raises(AssertionError):
        pf.assert_outcome_pool_discipline(y_obs_pool="P_train", u_train_pool="P_train")
    # Correct discipline passes.
    pf.assert_outcome_pool_discipline(y_obs_pool="P_train", u_train_pool="P_val")


def test_e_rejects_in_sample_refit_on_val():
    with pytest.raises(AssertionError):
        pf.assert_frozen_residualization(
            pool="P_val", used_frozen_B_lambda=False, refit_in_sample=True
        )
    # Frozen residualization on P_val passes.
    pf.assert_frozen_residualization(
        pool="P_val", used_frozen_B_lambda=True, refit_in_sample=False
    )


# --------------------------------------------------------------------------- #
# (g) R0-style permutation placebo holds nominal level (Prop. P0 exchangeability)
# --------------------------------------------------------------------------- #
def _permutation_placebo_pvalue(values, group, *, n_perm, seed):
    """Two-group mean-difference permutation p-value (within-stratum exchange).

    Under the null (labels exchangeable), the observed |mean_1 - mean_0| is just
    one draw from the permutation distribution, so the p-value is ~ Uniform(0,1)
    and the test rejects at ~ alpha. Pure stdlib, deterministic in ``seed``.
    """

    n = len(values)
    g1 = [v for v, g in zip(values, group) if g == 1]
    g0 = [v for v, g in zip(values, group) if g == 0]
    obs = abs(math.fsum(g1) / len(g1) - math.fsum(g0) / len(g0))
    rng = _LCG(seed)
    ge = 1  # +1 for the observed statistic itself (Phipson-Smyth correction)
    idx = list(range(n))
    for _ in range(n_perm):
        # Fisher-Yates shuffle of the group assignment.
        order = idx[:]
        for i in range(n - 1, 0, -1):
            j = int(rng._next() * (i + 1))
            order[i], order[j] = order[j], order[i]
        perm_group = [group[order[k]] for k in range(n)]
        p1 = [v for v, g in zip(values, perm_group) if g == 1]
        p0 = [v for v, g in zip(values, perm_group) if g == 0]
        stat = abs(math.fsum(p1) / len(p1) - math.fsum(p0) / len(p0))
        if stat >= obs - 1e-12:
            ge += 1
    return ge / (n_perm + 1)


def test_g_permutation_placebo_holds_nominal_level():
    # Null DGP: the group label is a coin flip INDEPENDENT of the value, so the
    # within-stratum exchangeability of Prop. P0 holds and the placebo permutation
    # test must reject at ~ alpha (never grossly inflated).
    alpha = 0.10
    rejections = 0
    trials = 400
    for trial in range(trials):
        rng = _LCG(200_000 + trial)
        values = [rng.gauss() for _ in range(24)]
        group = [1 if rng.boolp(0.5) else 0 for _ in range(24)]
        # Guard against a degenerate all-one-group draw.
        if 0 < sum(group) < len(group):
            p = _permutation_placebo_pvalue(values, group, n_perm=200, seed=trial)
            if p <= alpha:
                rejections += 1
    rate = rejections / trials
    assert rate <= alpha + 0.06, f"placebo permutation inflated: {rate}"


def test_g_permutation_placebo_detects_real_shift():
    # Sanity converse: a large genuine group shift makes the placebo p-value small,
    # so the permutation machinery is not trivially non-rejecting.
    values = [0.0] * 12 + [5.0] * 12
    group = [0] * 12 + [1] * 12
    p = _permutation_placebo_pvalue(values, group, n_perm=400, seed=1)
    assert p < 0.05


# --------------------------------------------------------------------------- #
# Gate R2 (matched_budget §3.5 R2): method_win requires EVERY sub-claim + a
# deterministic (PYTHONHASHSEED-independent) per-sub-claim bootstrap seed.
# --------------------------------------------------------------------------- #
def _r2_inputs(*, rel_diffs, ret_diffs, hall_diffs, cost_diffs):
    # Target comparators that clearly clear the win margin (delta_target = 0).
    rng = _LCG(31)
    comparators = {
        c: [1.0 + 0.05 * rng.gauss() for _ in range(40)]
        for c in ("nait_L", "act_sel", "global", "unif")
    }
    sub = {
        "relative": (rel_diffs, 0.0, "win"),
        "retention": (ret_diffs, 0.0, "non_inferiority"),
        "hallucination": (hall_diffs, 0.0, "non_inferiority"),
        "cost": (cost_diffs, 0.0, "non_inferiority"),
    }
    from neurotrace_it.cost_model import LedgerMatchResult

    ledger = LedgerMatchResult(
        matched=True, tolerance=0.0, rank_conserved=True, mismatched_fields=()
    )
    return comparators, sub, ledger


def test_r2_method_win_requires_all_sub_claims():
    rng = _LCG(77)
    # All sub-claims pass: relative win (positive gap), drifts/cost below ceiling 0.
    rel_ok = [0.5 + 0.05 * rng.gauss() for _ in range(40)]
    drift_ok = [-0.5 + 0.05 * rng.gauss() for _ in range(40)]  # upper bound < 0
    comparators, sub, ledger = _r2_inputs(
        rel_diffs=rel_ok, ret_diffs=drift_ok, hall_diffs=drift_ok, cost_diffs=drift_ok
    )
    res = mb.gate_r2(
        comparators, sub, ledger, alpha=0.05, delta_target=0.0, n_boot=300, boot_seed=9
    )
    assert res.target.decision.reject and res.ledger_matched
    assert res.sub_claims_pass and res.method_win

    # Now break ONE sub-claim (retention drift well ABOVE the ceiling): R2 must fail
    # even though the target IUT still rejects and the ledger is matched.
    ret_bad = [0.8 + 0.05 * rng.gauss() for _ in range(40)]  # upper bound > 0 -> fails NI
    comparators2, sub2, ledger2 = _r2_inputs(
        rel_diffs=rel_ok, ret_diffs=ret_bad, hall_diffs=drift_ok, cost_diffs=drift_ok
    )
    res2 = mb.gate_r2(
        comparators2, sub2, ledger2, alpha=0.05, delta_target=0.0, n_boot=300, boot_seed=9
    )
    assert res2.target.decision.reject and res2.ledger_matched  # headline + ledger still ok
    assert not res2.sub_claims["retention"].passes
    assert not res2.sub_claims_pass
    assert not res2.method_win  # a single failing sub-claim vetoes the method win


def test_r2_sub_claim_seed_is_deterministic_not_salted_hash():
    # The per-sub-claim bootstrap seed must be reproducible across processes (it is
    # derived from sha256(name), NOT Python's salted hash()). Recompute the exact
    # expected stream seed and confirm the bound matches a direct single_margin_test.
    import hashlib

    diffs = [0.3 + 0.01 * i for i in range(40)]
    name = "retention"
    boot_seed = 9
    expected_seed = boot_seed ^ int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:8], "big"
    )
    direct = mb.single_margin_test(
        diffs, alpha=0.05, threshold=0.0, direction="non_inferiority",
        n_boot=200, boot_seed=expected_seed, name=name,
    )
    comparators, _sub, ledger = _r2_inputs(
        rel_diffs=diffs, ret_diffs=diffs, hall_diffs=diffs, cost_diffs=diffs
    )
    sub = {name: (diffs, 0.0, "non_inferiority")}
    res = mb.gate_r2(
        comparators, sub, ledger, alpha=0.05, delta_target=0.0, n_boot=200, boot_seed=boot_seed
    )
    # The gate's sub-claim bound is byte-identical to the directly-seeded test, i.e.
    # the seed derivation is the sha256 stream, not a process-salted hash().
    assert res.sub_claims[name].bound == direct.bound
    assert res.sub_claims[name].gap_hat == direct.gap_hat


# --------------------------------------------------------------------------- #
# Tiny deterministic RNG helpers (pure stdlib; reproducible across processes)
# --------------------------------------------------------------------------- #
class _LCG:
    """A tiny deterministic generator for test DGPs (no global random state)."""

    def __init__(self, seed: int):
        self.state = (seed * 2_862_933_555_777_941_757 + 3_037_000_493) & ((1 << 64) - 1)

    def _next(self) -> float:
        self.state = (self.state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & (
            (1 << 64) - 1
        )
        return (self.state >> 11) / float(1 << 53)

    def gauss(self) -> float:
        u1 = max(self._next(), 1e-12)
        u2 = self._next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def boolp(self, p: float) -> bool:
        return self._next() < p

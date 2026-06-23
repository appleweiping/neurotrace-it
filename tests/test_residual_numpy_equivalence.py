"""Numerical-equivalence test: numpy fast-path == pure-python fallback (Gate-R0).

The dual-ridge FWL cross-fit partial-R^2 test
(:mod:`neurotrace_it.analysis.residual_test`) ships a numpy fast-path for the
linear-algebra hot path (the n x n SPD kernel solves, the gram build, the kernel
APPLICATION ``k_* alpha``, the held-out RSS / partial-R^2, and the batched block
permutation) with a graceful pure-python fallback. This module pins the two paths
together: forcing ``_HAVE_NUMPY = False`` (exact stdlib Cholesky / dot products /
row-permutation) vs the default numpy path MUST agree to floating tolerance on

  * the partial-R^2 point statistic (< 1e-8 max abs diff),
  * the block-permutation p-value (seed-identical RNG draw order => IDENTICAL p),
  * the cluster-BCa CI bounds (same bootstrap/jackknife draws => match to ~1e-8),

because numpy ONLY swaps the underlying linear algebra -- same estimand, same
statistics, same RNG stream. A fast-but-wrong test is worse than a slow one; this
is the guard that the swap is numerically faithful.

Build-now / run-later: pure synthetic DGP, no model load, no server call.
"""

from __future__ import annotations

import random

import pytest

import neurotrace_it.analysis.residual_test as rt
from neurotrace_it.analysis.residualize import cluster_bootstrap_ci

# Skip the whole module if numpy is not installed: with no numpy there is only one
# path, so there is nothing to compare (the fallback IS the implementation).
np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------- #
# Synthetic planted-residual DGP (wide-ish endpoint control + signal column).  #
# --------------------------------------------------------------------------- #


def _planted_dgp(n, *, signal_coef, noise_sd, seed, p_phi=4):
    rng = random.Random(seed)
    control, trajectory, outcome, strata = [], [], [], []
    for i in range(n):
        phi = [rng.gauss(0.0, 1.0) for _ in range(p_phi)]
        t_resid = rng.gauss(0.0, 1.0)
        t0 = 0.7 * phi[0] + t_resid
        t1 = -0.3 * phi[1] + rng.gauss(0.0, 1.0)
        y = (
            1.5 * phi[0] - 0.4 * phi[1]
            + signal_coef * t_resid + rng.gauss(0.0, noise_sd)
        )
        control.append(phi + [1.0])           # [phi_end... , intercept]
        trajectory.append([t0, t1])
        outcome.append(y)
        strata.append(i % 6)                  # 6 clusters
    return control, outcome, trajectory, strata


def _run_block_perm(**kw):
    return rt.block_permutation_test(**kw)


def _max_abs(a, b):
    return abs(float(a) - float(b))


# --------------------------------------------------------------------------- #
# 1. Partial-R^2 + permutation-p equivalence (numpy vs forced pure-python).     #
# --------------------------------------------------------------------------- #


def test_partial_r2_and_perm_p_equivalence():
    control, outcome, trajectory, strata = _planted_dgp(
        90, signal_coef=1.4, noise_sd=0.4, seed=101
    )
    kw = dict(
        control=control, outcome=outcome, trajectory=trajectory, strata=strata,
        penalized_columns=[0, 1, 2, 3], ridge_lambda=0.5, n_folds=5,
        n_permutations=300, seed=7, use_dual=True,
    )

    assert rt._HAVE_NUMPY, "numpy must be present to compare the two paths"
    # numpy fast-path (default).
    res_np = _run_block_perm(**kw)

    # forced pure-python fallback.
    saved = rt._HAVE_NUMPY
    try:
        rt._HAVE_NUMPY = False
        res_py = _run_block_perm(**kw)
    finally:
        rt._HAVE_NUMPY = saved

    d_r2 = _max_abs(res_np.partial_r2, res_py.partial_r2)
    d_delta = _max_abs(res_np.delta_r2_overall, res_py.delta_r2_overall)
    d_beta = max(_max_abs(a, b) for a, b in zip(res_np.beta_t, res_py.beta_t))
    # partial-R^2 / delta-R^2 / beta agree to floating tolerance (only the inner
    # float64 reduction order differs between LAPACK/numpy and math.fsum).
    assert d_r2 < 1e-8, (d_r2, res_np.partial_r2, res_py.partial_r2)
    assert d_delta < 1e-8, d_delta
    assert d_beta < 1e-8, d_beta
    # The permutation p-value uses the SAME random.Random stream in the SAME order,
    # so it must be bit-identical (it is a rational k/(P+1)).
    assert res_np.permutation_p_value == res_py.permutation_p_value
    assert res_np.fold_partial_r2 == pytest.approx(res_py.fold_partial_r2, abs=1e-8)


# --------------------------------------------------------------------------- #
# 2. Null DGP: same equivalence holds with no planted signal.                   #
# --------------------------------------------------------------------------- #


def test_equivalence_under_null():
    control, outcome, trajectory, strata = _planted_dgp(
        80, signal_coef=0.0, noise_sd=1.0, seed=202
    )
    kw = dict(
        control=control, outcome=outcome, trajectory=trajectory, strata=strata,
        penalized_columns=[0, 1, 2, 3], ridge_lambda=1.0, n_folds=4,
        n_permutations=250, seed=13, use_dual=True,
    )
    res_np = _run_block_perm(**kw)
    saved = rt._HAVE_NUMPY
    try:
        rt._HAVE_NUMPY = False
        res_py = _run_block_perm(**kw)
    finally:
        rt._HAVE_NUMPY = saved

    assert _max_abs(res_np.partial_r2, res_py.partial_r2) < 1e-8
    assert res_np.permutation_p_value == res_py.permutation_p_value


def test_numpy_only_fallback_matches_scipy_path():
    """The numpy fast-path uses SciPy's ``cho_solve`` when available and a
    ``numpy.linalg.solve`` fallback otherwise (the server may have numpy without
    scipy). Both kernel-solve back-ends, and the pure-python path, must agree to
    floating tolerance and reproduce the permutation-p identically."""

    control, outcome, trajectory, strata = _planted_dgp(
        85, signal_coef=1.2, noise_sd=0.4, seed=505
    )
    kw = dict(
        control=control, outcome=outcome, trajectory=trajectory, strata=strata,
        penalized_columns=[0, 1, 2, 3], ridge_lambda=0.5, n_folds=5,
        n_permutations=200, seed=11, use_dual=True,
    )
    assert rt._HAVE_NUMPY
    res_scipy = _run_block_perm(**kw)               # numpy + scipy (default here)
    saved_sci = rt._HAVE_SCIPY
    try:
        rt._HAVE_SCIPY = False
        res_np_only = _run_block_perm(**kw)          # numpy, no scipy
    finally:
        rt._HAVE_SCIPY = saved_sci

    assert _max_abs(res_scipy.partial_r2, res_np_only.partial_r2) < 1e-8
    assert res_scipy.permutation_p_value == res_np_only.permutation_p_value


# --------------------------------------------------------------------------- #
# 3. dual_ridge_partial_out residuals: numpy fit == pure-python fit.            #
# --------------------------------------------------------------------------- #


def test_dual_ridge_residual_equivalence():
    control, outcome, trajectory, _ = _planted_dgp(
        70, signal_coef=1.0, noise_sd=0.3, seed=303
    )
    pen = [0, 1, 2, 3]
    lam = 0.7

    np_y, np_cols = rt.dual_ridge_partial_out(
        control, outcome, trajectory, penalized_columns=pen, ridge_lambda=lam
    )
    saved = rt._HAVE_NUMPY
    try:
        rt._HAVE_NUMPY = False
        py_y, py_cols = rt.dual_ridge_partial_out(
            control, outcome, trajectory, penalized_columns=pen, ridge_lambda=lam
        )
    finally:
        rt._HAVE_NUMPY = saved

    # Residual-maker outputs (applied out-of-sample) must coincide.
    dmax = 0.0
    for z, y in zip(control, outcome):
        dmax = max(dmax, _max_abs(np_y.residual(z, y), py_y.residual(z, y)))
    for col in range(len(np_cols)):
        for z, row in zip(control, trajectory):
            dmax = max(
                dmax,
                _max_abs(np_cols[col].residual(z, row[col]), py_cols[col].residual(z, row[col])),
            )
    assert dmax < 1e-8, dmax


# --------------------------------------------------------------------------- #
# 4. End-to-end BCa-CI equivalence via the R0-style cluster bootstrap.          #
# --------------------------------------------------------------------------- #


def test_bca_ci_equivalence_end_to_end():
    # Build an R0-style design and run the SAME cluster-BCa CI of the cross-fit
    # partial-R^2 under both linalg paths; the bootstrap/jackknife draws are stdlib
    # random with the same seed, so the bounds match to floating tolerance.
    control, outcome, trajectory, strata = _planted_dgp(
        96, signal_coef=1.3, noise_sd=0.4, seed=404
    )
    pen = [0, 1, 2, 3]
    lam = 0.5
    n_folds = 4

    rows_by_cluster: dict[int, list[int]] = {}
    for idx, c in enumerate(strata):
        rows_by_cluster.setdefault(c, []).append(idx)
    unique_clusters = sorted(rows_by_cluster)

    def make_stat():
        def partial_r2_on_clusters(cluster_ids):
            rows = []
            for c in cluster_ids:
                rows.extend(rows_by_cluster.get(c, ()))
            if len(rows) < max(4, n_folds):
                return 0.0
            sub_c = [control[i] for i in rows]
            sub_y = [outcome[i] for i in rows]
            sub_t = [trajectory[i] for i in rows]
            sub_s = [strata[i] for i in rows]
            sub = rt.residualized_regression_test(
                sub_c, sub_y, sub_t, strata=sub_s, penalized_columns=pen,
                ridge_lambda=lam, n_folds=n_folds, n_permutations=0, seed=5,
            )
            return sub.partial_r2
        return partial_r2_on_clusters

    ci_np = cluster_bootstrap_ci(unique_clusters, make_stat(), n_boot=120, level=0.95, seed=9)
    saved = rt._HAVE_NUMPY
    try:
        rt._HAVE_NUMPY = False
        ci_py = cluster_bootstrap_ci(unique_clusters, make_stat(), n_boot=120, level=0.95, seed=9)
    finally:
        rt._HAVE_NUMPY = saved

    assert _max_abs(ci_np.point, ci_py.point) < 1e-8, (ci_np.point, ci_py.point)
    assert _max_abs(ci_np.lower, ci_py.lower) < 1e-8, (ci_np.lower, ci_py.lower)
    assert _max_abs(ci_np.upper, ci_py.upper) < 1e-8, (ci_np.upper, ci_py.upper)
    # Same RNG stream -> identical bias-correction / acceleration too.
    assert _max_abs(ci_np.bias_correction, ci_py.bias_correction) < 1e-8
    assert _max_abs(ci_np.acceleration, ci_py.acceleration) < 1e-8

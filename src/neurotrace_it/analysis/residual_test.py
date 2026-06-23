"""Residualized multiple regression with cross-fit + permutation inference.

The CO-PRIMARY statistic of ``docs/redesign/REDESIGN_v4.md`` (§2.3-§2.4): does the
trajectory block ``T`` carry information about the outcome ``Y`` *after*
partialling out the FULL endpoint signature ``phi_end`` (plus nuisance covariates
``C``)? The endpoint control is the full ``phi_end in R^{2 d |A|}`` handled by
**ridge** partialling-out (ridge-FWL / double-ML orthogonalization), NOT a
low-rank PCA summary; inference is a **K-fold cross-fit** held-out partial-R^2
with an out-of-sample **block permutation** p-value -- explicitly NOT a
nested-model F-test (fix b).

Pipeline
--------
1. **Ridge residual-maker** (Eq. 8c). With control block ``Z = [phi_end, C, 1]``
   and ``Omega`` penalizing only the ``phi_end`` columns::

       M_lambda = I - Z (Z^T Z + lambda * Omega)^+ Z^T

   ``M_lambda`` orthogonalizes both ``Y`` and ``T`` against the (regularized)
   control. Only ``phi_end`` is penalized; ``C`` and the intercept enter
   unpenalized. **Scalability (fix b, review §2.3):** the penalized ``phi_end``
   block is wide (``2 d |A| >> N`` at 7B-scale hidden dims), so forming the
   ``p x p`` Gram is infeasible. We solve the ridge in **n-space (dual / kernel
   form)**: the penalized contribution enters only through the linear kernel
   ``G = Phi Phi^T`` (an ``n x n`` Gram in the *number of examples*, not
   features), and unpenalized covariates ``[C, 1]`` are profiled out with a
   small ``p_C x p_C`` solve. Cost is ``O(n^2 p + n^3)`` and **independent of the
   ``phi_end`` width**, so the FULL ``phi_end`` control is feasible.
2. **Partialled regression** (Eq. 9): ``beta_T = ((M T)^T (M T))^+ (M T)^T (M Y)``.
3. **Partial-R^2** (Eq. 10a) -- fraction of endpoint-controlled residual variance
   explained by ``T``; **overall-R^2 increment** (Eq. 10b) mapped to the 0.02
   deployment margin.
4. **Cross-fit** (K=10 default): fit ``M_lambda`` and ``beta_T`` on the other
   ``K-1`` folds; evaluate the held-out residual statistic on the left-out fold;
   aggregate by **median-of-folds**.
5. **Block permutation test (locked conditional null, REDESIGN_v4 §2.4 step 2,
   line 323).** Under H0 the trajectory block is exchangeable **conditional on
   the controls**, so the permutation is applied to the **endpoint-orthogonalized
   residuals ``M_lambda T``** -- the held-out cross-fit residuals after the SAME
   train-frozen orthogonalization -- permuted **within family x fold strata**,
   NOT to raw trajectory rows before cross-fit. Permuting raw rows would break
   the conditioning on ``phi_end`` and re-introduce the very endpoint covariance
   the null holds fixed. We recompute the held-out partial-R^2 ``P`` times and
   report the right-tail rank as an exact, distribution-free p-value.

Linear algebra is a tiny, dependency-free core (Cholesky / Gaussian elimination
with ridge on the diagonal, which makes the system SPD and the pseudo-inverse a
plain solve). The wide ``phi_end`` block is handled by the **dual** ridge above,
so ``p_phi >> N`` is fine.

DO-NOT-RUN: no model load, no server call, no training. Build-now / run-later.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

# Optional numpy acceleration for the SPD-solve / gram / RSS hot path. The
# bottleneck is the n x n SPD kernel solves (``_solve_spd`` -> pure-python
# Cholesky, O(n^3) with ~100ns/op overhead -> minutes at n=2000); numpy routes
# them to LAPACK (``numpy.linalg.cholesky`` + triangular solves), ~1000x faster.
#
# NUMERICALLY EQUIVALENT to the pure-Python path (verified by the equivalence
# test in ``tests/test_residual_numpy_equivalence.py`` to < 1e-8 on the
# partial-R^2 point stat, and to seed-identical reproduction of the permutation-p
# and BCa bounds): the fast-path replicates the SAME ridge system, the SAME
# incremental-jitter retry on a non-positive pivot, and -- crucially -- consumes
# the RNG in the SAME ORDER (folds, permutation row-shuffles, bootstrap draws are
# all stdlib ``random`` exactly as before), so results are reproducible-identical.
# Only the inner float64 reductions differ (LAPACK / numpy reduction vs
# ``math.fsum``), giving machine-epsilon agreement. If numpy is unavailable the
# exact stdlib path is used, so the module stays import-safe and the unit tests
# pass either way. No model load, no server call.
try:  # pragma: no cover - exercised both ways by the equivalence test
    import numpy as _np

    _HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    _np = None
    _HAVE_NUMPY = False

# Prefer SciPy's LAPACK Cholesky-solve (``cho_factor`` / ``cho_solve``) for the
# n x n kernel solves. This matters concretely: several recent numpy wheels ship a
# ``numpy.linalg.solve`` gufunc that is pathologically slow on Windows
# (~2s for a single 400x400 solve here -- a broken generalized-ufunc path), which
# would erase the whole speedup. SciPy's ``cho_solve`` reuses the Cholesky factor
# and routes to ``?potrs`` (~5ms for the same size, ~380x faster). SciPy is
# optional: without it we fall back to ``numpy.linalg.solve`` (correct, just slow
# on the affected builds) so the module stays import-safe with numpy alone.
try:  # pragma: no cover - presence-dependent
    from scipy.linalg import cho_factor as _scipy_cho_factor
    from scipy.linalg import cho_solve as _scipy_cho_solve

    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _scipy_cho_factor = None
    _scipy_cho_solve = None
    _HAVE_SCIPY = False

Matrix = list[list[float]]
Vector = list[float]

__all__ = [
    "DEFAULT_FOLDS",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_RIDGE",
    "RIDGE_LAMBDA_GRID_MULTIPLIERS",
    "DEFAULT_LAMBDA_CV_FOLDS",
    "RidgePartialOut",
    "DualRidgePartialOut",
    "CrossFitResidualResult",
    "RidgeLambdaCVResult",
    "ridge_partial_out",
    "dual_ridge_partial_out",
    "select_ridge_lambda_cv",
    "cross_fit_partial_r2",
    "block_permutation_test",
    "residualized_regression_test",
]

DEFAULT_FOLDS = 10            # K-fold cross-fit (§2.4)
DEFAULT_PERMUTATIONS = 5000   # block permutation count P (§2.4)
DEFAULT_RIDGE = 1.0           # placeholder lambda; CV-locked per split in the plan

# Registered ridge-lambda CV grid (REDESIGN_v4 §4.2 line 772, inherited by
# REDESIGN_v5 §3.2 "the CV-chosen lambda_ridge"):
#   "lambda_ridge CV grid (locked): {1e-2, 1e-1, 1, 10, 100} x sigma^2_Phi
#    selected by 5-fold CV on each train fold; the selected value persisted per
#    split."
# These are the MULTIPLIERS of sigma^2_Phi (the mean per-column phi_end variance,
# EndpointControl.phi_variance); the actual grid passed to the ridge is each
# multiplier x sigma^2_phi. Do NOT invent values -- this is the frozen grid.
# DOCUMENTED DEVIATION (2026-06-21): the registered multipliers are of sigma^2_phi
# (~per-column variance, here ~2.0). In the DUAL/kernel ridge the penalty enters as
# (G + lambda*I) with G = Phi Phi^T whose diagonal is ~p_phi (~2.4e4 for the 1.5B
# standardized endpoint block), so the registered max (100 x sigma^2_phi ~ 2e2) is
# ~100x too small to shrink AT ALL: the held-out 5-fold CV criterion is monotone-
# decreasing across the ENTIRE registered range and always pins lambda to the grid
# boundary, making the CV inert and partial_r2 a lambda-boundary artifact. We EXTEND
# the grid upward (adding 1e3..1e6) to bracket the kernel eigenvalue scale so the CV
# finds a GENUINE INTERIOR optimum -- this realizes the registered INTENT ("CV-select
# lambda by 5-fold held-out fit") rather than changing it. The selection criterion,
# folds, and estimand are unchanged; only the search range is widened to be adequate.
RIDGE_LAMBDA_GRID_MULTIPLIERS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
DEFAULT_LAMBDA_CV_FOLDS = 5  # "selected by 5-fold CV" (REDESIGN_v4 §4.2)


def _stable_seed(*parts: object) -> int:
    """Stable non-negative ``int`` seed from arbitrary components (fix c).

    ``random.Random`` rejects tuple seeds (``TypeError`` on Python 3.11+); hash
    the components into a reproducible 63-bit integer instead. ``blake2b`` is
    used (not the salted built-in ``hash``) so the stream is stable across
    processes -- required for the reproducibility ledger.
    """

    payload = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


# --------------------------------------------------------------------------- #
# Minimal dependency-free linear algebra (ridge-regularized => SPD solves).    #
# --------------------------------------------------------------------------- #


def _matmul_at_a(matrix: Matrix) -> Matrix:
    """Gram matrix ``A^T A`` (p x p) for an n x p design ``A``."""

    n = len(matrix)
    p = len(matrix[0]) if n else 0
    if _HAVE_NUMPY and n and p:
        a = _np.asarray(matrix, dtype=_np.float64)
        gram = a.T @ a
        gram = 0.5 * (gram + gram.T)  # exact symmetry for the downstream SPD factor
        return [list(row) for row in gram]
    gram = [[0.0] * p for _ in range(p)]
    for row in matrix:
        for i in range(p):
            ri = row[i]
            if ri == 0.0:
                continue
            gram_i = gram[i]
            for j in range(i, p):
                gram_i[j] += ri * row[j]
    for i in range(p):
        for j in range(i + 1, p):
            gram[j][i] = gram[i][j]
    return gram


def _matvec_at_b(matrix: Matrix, vector: Vector) -> Vector:
    """``A^T b`` for an n x p design ``A`` and length-n vector ``b``."""

    n = len(matrix)
    p = len(matrix[0]) if n else 0
    if _HAVE_NUMPY and n and p:
        a = _np.asarray(matrix, dtype=_np.float64)
        b = _np.asarray(vector, dtype=_np.float64)
        return [float(v) for v in (a.T @ b)]
    out = [0.0] * p
    for row, bv in zip(matrix, vector):
        if bv == 0.0:
            continue
        for j in range(p):
            out[j] += row[j] * bv
    return out


def _solve_spd(matrix: Matrix, rhs: Vector, *, jitter: float = 1e-10) -> Vector:
    """Solve ``A x = b`` for symmetric positive-(semi)definite ``A`` via Cholesky.

    A tiny diagonal jitter is added for numerical safety; because callers ridge-
    regularize ``A``, it is SPD and Cholesky succeeds. Falls back to incremental
    jitter if a non-positive pivot is encountered (rank-deficient edge cases).

    numpy fast-path: LAPACK Cholesky + triangular solves (``_np_solve_spd``), the
    EXACT same algorithm (same jitter retry) routed through C -- ~1000x faster on
    the n x n kernel solves that dominate the FWL test. Pure-python fallback below
    keeps the module import-safe without numpy.
    """

    size = len(matrix)
    if _HAVE_NUMPY and size:
        a = _np.asarray(matrix, dtype=_np.float64)
        b = _np.asarray(rhs, dtype=_np.float64)
        return [float(v) for v in _np_solve_spd(a, b, jitter=jitter)]
    work = [row[:] for row in matrix]
    current_jitter = jitter
    for _ in range(8):
        try:
            lower = _cholesky([[work[i][j] + (current_jitter if i == j else 0.0)
                                for j in range(size)] for i in range(size)])
            return _chol_solve(lower, rhs)
        except ValueError:
            current_jitter *= 10.0
    raise ValueError("SPD solve failed: matrix not positive definite even after jitter")


def _cholesky(matrix: Matrix) -> Matrix:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise ValueError("non-positive pivot in Cholesky")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _chol_solve(lower: Matrix, rhs: Vector) -> Vector:
    size = len(lower)
    # Forward solve L y = b.
    y = [0.0] * size
    for i in range(size):
        y[i] = (rhs[i] - math.fsum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    # Back solve L^T x = y.
    x = [0.0] * size
    for i in reversed(range(size)):
        x[i] = (y[i] - math.fsum(lower[k][i] * x[k] for k in range(i + 1, size))) / lower[i][i]
    return x


# --------------------------------------------------------------------------- #
# numpy fast-path for the SPD-solve hot path (LAPACK Cholesky, ~1000x faster). #
# Each helper is the EXACT vectorization of the stdlib primitive directly above #
# (same ridge system, same incremental-jitter retry); only the inner float64    #
# reductions differ. Guarded by ``_HAVE_NUMPY`` -> pure-python fallback.        #
# --------------------------------------------------------------------------- #


def _np_chol_factor(matrix: "object", jitter: float = 1e-10) -> "object":
    """SPD Cholesky factor with the SAME incremental-jitter retry as
    :func:`_solve_spd`: add ``jitter`` to the diagonal, ``x10`` up to 8 times on a
    non-positive pivot, then raise. Returns an OPAQUE factor handle reused by
    :func:`_np_chol_solve` for every right-hand-side (factor once, solve many).

    With SciPy the handle is the ``cho_factor`` tuple ``(c, lower)`` routed to
    LAPACK ``?potrf`` / ``?potrs``; without SciPy it is the lower factor ``L`` from
    ``numpy.linalg.cholesky``. Either way LAPACK ``?potrf`` raises on the first
    non-positive pivot exactly as the stdlib ``_cholesky`` does, so the retry
    semantics (and thus the produced solution) match the pure-python path.
    """

    size = matrix.shape[0]
    eye = _np.eye(size, dtype=_np.float64)
    current_jitter = jitter
    for _ in range(8):
        try:
            m = matrix + current_jitter * eye
            if _HAVE_SCIPY:
                return _scipy_cho_factor(m, lower=True, check_finite=False)
            return ("np", _np.linalg.cholesky(m))
        except (_np.linalg.LinAlgError, ValueError):
            # SciPy's cho_factor raises LinAlgError on a non-positive pivot too.
            current_jitter *= 10.0
    raise ValueError("SPD solve failed: matrix not positive definite even after jitter")


def _np_chol_solve(factor: "object", rhs: "object") -> "object":
    """Solve ``A x = rhs`` from the :func:`_np_chol_factor` handle. ``rhs`` may be a
    1-D vector or a 2-D column-stacked matrix, so a single factorization serves
    every RHS in a fold. With SciPy this is one ``?potrs`` call; without it, two
    LAPACK triangular solves via ``numpy.linalg.solve`` (correct, slower on the
    affected numpy builds -- see the import note)."""

    if _HAVE_SCIPY:
        return _scipy_cho_solve(factor, rhs, check_finite=False)
    lower = factor[1]
    y = _np.linalg.solve(lower, rhs)        # forward solve L y = rhs
    return _np.linalg.solve(lower.T, y)     # back solve L^T x = y


def _np_solve_spd(matrix: "object", rhs: "object", *, jitter: float = 1e-10) -> "object":
    """numpy/scipy SPD solve == :func:`_solve_spd` (same jitter retry). ``rhs`` may
    be a vector or a column-stacked matrix (batched solve over the shared factor)."""

    factor = _np_chol_factor(matrix, jitter)
    return _np_chol_solve(factor, rhs)


# --------------------------------------------------------------------------- #
# Ridge partialling-out (Eq. 8c-9).                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RidgePartialOut:
    """A fitted ridge residual-maker on a TRAIN fold (Eq. 8c).

    Stores the solved control coefficients so the SAME orthogonalization can be
    applied out-of-sample (no leakage): for a held-out control row ``z`` and
    target value ``y``, the residual is ``y - z @ coef``.
    """

    penalized_columns: tuple[int, ...]
    ridge_lambda: float
    coef: tuple[float, ...]  # control coefficients beta_Z, length p_Z

    def residual(self, control_row: Sequence[float], value: float) -> float:
        prediction = math.fsum(c * z for c, z in zip(self.coef, control_row))
        return value - prediction


def _fit_ridge_control(
    control: Matrix,
    target: Vector,
    penalized_columns: Sequence[int],
    ridge_lambda: float,
) -> RidgePartialOut:
    """Fit ``beta_Z = (Z^T Z + lambda Omega)^+ Z^T target`` (the Eq. 8c core).

    ``Omega`` is identity on ``penalized_columns`` (the ``phi_end`` block) and
    zero elsewhere (covariates ``C`` and the intercept enter unpenalized).
    """

    p = len(control[0]) if control else 0
    gram = _matmul_at_a(control)
    penalized = set(penalized_columns)
    for j in range(p):
        if j in penalized:
            gram[j][j] += ridge_lambda
    rhs = _matvec_at_b(control, target)
    coef = _solve_spd(gram, rhs)
    return RidgePartialOut(
        penalized_columns=tuple(sorted(penalized)),
        ridge_lambda=ridge_lambda,
        coef=tuple(coef),
    )


def ridge_partial_out(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    penalized_columns: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE,
) -> tuple[RidgePartialOut, list[RidgePartialOut]]:
    """Fit the ridge residual-makers for ``Y`` and each column of ``T`` (Eq. 8c-9).

    Returns ``(outcome_partialler, [trajectory_partialler_per_column])``. The
    same fitted partiallers are applied out-of-sample by the cross-fit caller.

    Parameters
    ----------
    control:
        ``Z = [phi_end, C, 1]`` design (n x p_Z), standardized upstream.
    outcome:
        ``Y`` (length n).
    trajectory:
        ``T`` design (n x q), one column per trajectory feature.
    penalized_columns:
        Indices of the ``phi_end`` columns in ``Z`` to penalize (Omega).
    ridge_lambda:
        The ridge penalty (CV-locked per split in the registered plan).
    """

    outcome_partialler = _fit_ridge_control(control, outcome, penalized_columns, ridge_lambda)
    q = len(trajectory[0]) if trajectory else 0
    column_partiallers: list[RidgePartialOut] = []
    for col in range(q):
        column = [row[col] for row in trajectory]
        column_partiallers.append(
            _fit_ridge_control(control, column, penalized_columns, ridge_lambda)
        )
    return outcome_partialler, column_partiallers


# --------------------------------------------------------------------------- #
# Dual / kernel ridge residual-maker in n-space (fix b -- 7B-scale feasible).  #
# --------------------------------------------------------------------------- #


def _split_control(
    control_row: Sequence[float], penalized: frozenset[int]
) -> tuple[list[float], list[float]]:
    """Split a control row into (penalized phi_end block, unpenalized [C, 1])."""

    phi: list[float] = []
    unpen: list[float] = []
    for j, value in enumerate(control_row):
        if j in penalized:
            phi.append(value)
        else:
            unpen.append(value)
    return phi, unpen


@dataclass(frozen=True)
class DualRidgePartialOut:
    """A fitted ridge residual-maker solved in **n-space** (dual / kernel form).

    Mathematically identical to :class:`RidgePartialOut` (same ``M_lambda``,
    Eq. 8c) but never forms the ``p x p`` Gram of the wide ``phi_end`` block, so
    it scales to ``2 d |A| >> N`` (7B hidden dims). The penalized block enters
    only through the linear kernel ``G = Phi Phi^T`` (``n x n``); the unpenalized
    covariates ``U = [C, 1]`` are profiled out with a ``p_U x p_U`` solve.

    The fit is frozen on a TRAIN fold and applied out-of-sample with the SAME
    parameters (no leakage): for a held-out row it computes
    ``y - u @ b - k_*^T alpha`` where ``k_* = Phi_test Phi_train^T``.
    """

    penalized: frozenset[int]
    ridge_lambda: float
    train_phi: tuple[tuple[float, ...], ...]   # Phi_train (n_train x p_phi)
    unpen_coef: tuple[float, ...]              # b: unpenalized [C, 1] coefficients
    dual_coef: tuple[float, ...]               # alpha: dual weights (length n_train)

    def residual(self, control_row: Sequence[float], value: float) -> float:
        phi, unpen = _split_control(control_row, self.penalized)
        prediction = math.fsum(c * u for c, u in zip(self.unpen_coef, unpen))
        for alpha, phi_train_row in zip(self.dual_coef, self.train_phi):
            prediction += alpha * math.fsum(a * b for a, b in zip(phi, phi_train_row))
        return value - prediction

    def residuals_batch(
        self, control_rows: Sequence[Sequence[float]], values: Sequence[float]
    ) -> list[float]:
        """Vectorized :meth:`residual` over many rows -- the kernel APPLICATION hot
        path. The dual prediction ``k_* alpha`` with ``k_* = Phi_rows Phi_train^T``
        is the second O(n^2 p_phi) cost in the FWL test (the first is the solve);
        applying it row-by-row in pure Python pins a core at n=2000. numpy fast-path
        batches it as ``(Phi_rows @ Phi_train^T) @ alpha + U_rows @ b`` in float64;
        IDENTICAL math (the per-row stdlib path differs only in summation order).
        Falls back to the scalar :meth:`residual` without numpy.
        """

        if not _HAVE_NUMPY:
            return [self.residual(row, val) for row, val in zip(control_rows, values)]

        penalized = self.penalized
        n_rows = len(control_rows)
        if n_rows == 0:
            return []
        phi_rows: list[list[float]] = []
        unpen_rows: list[list[float]] = []
        for row in control_rows:
            phi_r, unpen_r = _split_control(row, penalized)
            phi_rows.append(phi_r)
            unpen_rows.append(unpen_r)

        vals = _np.asarray(values, dtype=_np.float64)
        prediction = _np.zeros(n_rows, dtype=_np.float64)
        if self.unpen_coef:
            u_mat = _np.asarray(unpen_rows, dtype=_np.float64)
            b = _np.asarray(self.unpen_coef, dtype=_np.float64)
            prediction = prediction + u_mat @ b
        if self.dual_coef and self.train_phi and phi_rows[0]:
            phi_mat = _np.asarray(phi_rows, dtype=_np.float64)         # (n_rows, p_phi)
            phi_train = _np.asarray(self.train_phi, dtype=_np.float64)  # (n_train, p_phi)
            alpha = _np.asarray(self.dual_coef, dtype=_np.float64)      # (n_train,)
            # k_* alpha = (Phi_rows Phi_train^T) alpha = Phi_rows (Phi_train^T alpha).
            prediction = prediction + phi_mat @ (phi_train.T @ alpha)
        return [float(v) for v in (vals - prediction)]


def _fit_dual_ridge_control(
    phi: Matrix,
    unpen: Matrix,
    gram: Matrix,
    target: Vector,
    penalized: frozenset[int],
    ridge_lambda: float,
) -> DualRidgePartialOut:
    """Dual ridge with unpenalized covariates profiled out (the Eq. 8c core).

    Minimizes ``||y - Phi a - U b||^2 + lambda ||a||^2``. With the representer
    ``a = Phi^T alpha`` (so ``Phi a = G alpha``), the KKT stationarity conditions
    are the saddle system::

        [ G + lambda I   U ] [alpha]   [y]
        [ U^T            0 ] [  b  ] = [0]

    (the ``b`` row is the unpenalized normal equation ``U^T residual = 0``, which
    reduces to ``U^T alpha = 0``). Solving by the Schur complement on the small
    ``p_u x p_u`` block: with ``K = G + lambda I``,
    ``(U^T K^{-1} U) b = U^T K^{-1} y`` and ``alpha = K^{-1}(y - U b)``. The
    pre-computed ``gram = Phi Phi^T`` is shared across all targets (Y and every
    T column) for the same train fold. Cost is independent of ``p_phi``.
    """

    n = len(target)
    p_u = len(unpen[0]) if unpen and unpen[0] else 0

    if _HAVE_NUMPY:
        return _fit_dual_ridge_control_np(
            phi, unpen, gram, target, penalized, ridge_lambda, n, p_u
        )

    # K = (G + lambda I) -- SPD kernel ridge system (n x n, independent of p_phi).
    kernel = [row[:] for row in gram]
    for i in range(n):
        kernel[i][i] += ridge_lambda

    k_inv_y = _solve_spd(kernel, list(target))
    if p_u:
        k_inv_u_cols: list[Vector] = []
        for c in range(p_u):
            u_col = [unpen[i][c] for i in range(n)]
            k_inv_u_cols.append(_solve_spd(kernel, u_col))
        # Schur reduced system: (U^T K^{-1} U) b = U^T K^{-1} y.
        a_mat = [[0.0] * p_u for _ in range(p_u)]
        rhs_b = [0.0] * p_u
        for r in range(p_u):
            u_r = [unpen[i][r] for i in range(n)]
            rhs_b[r] = math.fsum(u_r[i] * k_inv_y[i] for i in range(n))
            for c in range(p_u):
                a_mat[r][c] = math.fsum(u_r[i] * k_inv_u_cols[c][i] for i in range(n))
            a_mat[r][r] += 1e-12  # tiny jitter for SPD safety
        b = _solve_spd(a_mat, rhs_b)
    else:
        b = []

    # alpha = K^{-1}(y - U b).
    residual_target = [
        target[i] - math.fsum(b[c] * unpen[i][c] for c in range(p_u)) for i in range(n)
    ]
    alpha = _solve_spd(kernel, residual_target)

    return DualRidgePartialOut(
        penalized=penalized,
        ridge_lambda=ridge_lambda,
        train_phi=tuple(tuple(row) for row in phi),
        unpen_coef=tuple(b),
        dual_coef=tuple(alpha),
    )


def _fit_dual_ridge_control_np(
    phi: Matrix,
    unpen: Matrix,
    gram: "object",
    target: Vector,
    penalized: frozenset[int],
    ridge_lambda: float,
    n: int,
    p_u: int,
) -> DualRidgePartialOut:
    """Single-target numpy fast-path of :func:`_fit_dual_ridge_control` -- a thin
    wrapper over the batched :func:`_fit_dual_ridge_shared_np` (one target). Kept so
    a direct ``_fit_dual_ridge_control`` call under numpy is still correct; the hot
    path (``dual_ridge_partial_out``) uses the shared, multi-target form directly so
    the kernel is factored ONCE per fold rather than once per target."""

    return _fit_dual_ridge_shared_np(
        phi, unpen, gram, [list(target)], penalized, ridge_lambda, n
    )[0]


def dual_ridge_partial_out(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    penalized_columns: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE,
) -> tuple[DualRidgePartialOut, list[DualRidgePartialOut]]:
    """Dual (n-space) ridge residual-makers for ``Y`` and each ``T`` column.

    Drop-in scalable replacement for :func:`ridge_partial_out`: identical
    ``M_lambda`` but never forms the ``p x p`` Gram of the penalized ``phi_end``
    block. The ``n x n`` linear kernel ``G = Phi Phi^T`` is built ONCE and reused
    across ``Y`` and every trajectory column (they share the same control).
    """

    penalized = frozenset(penalized_columns)
    phi = [_split_control(row, penalized)[0] for row in control]
    unpen = [_split_control(row, penalized)[1] for row in control]
    n = len(control)
    q = len(trajectory[0]) if trajectory and trajectory[0] else 0

    if _HAVE_NUMPY:
        # numpy fast-path: build the n x n kernel G = Phi Phi^T ONCE, factor
        # K = G + lambda I ONCE (Cholesky), and solve EVERY target (Y + each T
        # column) against the SAME shared factor. This collapses the per-fold cost
        # to a single Cholesky + batched ?potrs instead of (1 + q) re-factorizations.
        phi_mat = _np.asarray(phi, dtype=_np.float64) if (n and phi[0]) else _np.zeros((n, 0))
        gram = phi_mat @ phi_mat.T
        gram = 0.5 * (gram + gram.T)              # exact symmetry for the SPD factor
        targets = [list(outcome)] + [[row[col] for row in trajectory] for col in range(q)]
        fits = _fit_dual_ridge_shared_np(phi, unpen, gram, targets, penalized, ridge_lambda, n)
        return fits[0], list(fits[1:])

    # Pure-python fallback: kernel built once, per-target SPD solves (as before).
    gram = [[0.0] * n for _ in range(n)]
    for i in range(n):
        phi_i = phi[i]
        gram[i][i] = math.fsum(v * v for v in phi_i)
        for j in range(i + 1, n):
            val = math.fsum(a * b for a, b in zip(phi_i, phi[j]))
            gram[i][j] = val
            gram[j][i] = val

    outcome_partialler = _fit_dual_ridge_control(
        phi, unpen, gram, list(outcome), penalized, ridge_lambda
    )
    column_partiallers: list[DualRidgePartialOut] = []
    for col in range(q):
        column = [row[col] for row in trajectory]
        column_partiallers.append(
            _fit_dual_ridge_control(phi, unpen, gram, column, penalized, ridge_lambda)
        )
    return outcome_partialler, column_partiallers


def _fit_dual_ridge_shared_np(
    phi: Matrix,
    unpen: Matrix,
    gram: "object",
    targets: list[Vector],
    penalized: frozenset[int],
    ridge_lambda: float,
    n: int,
) -> list[DualRidgePartialOut]:
    """Fit the dual ridge for MANY targets against ONE shared kernel factor.

    Numerically IDENTICAL to calling :func:`_fit_dual_ridge_control_np` per target
    (same K = G + lambda I, same Cholesky-with-jitter, same Schur complement with
    1e-12 jitter, same alpha = K^{-1}(y - U b)); it merely factors K once and
    batches the ``?potrs`` solves across the stacked right-hand-sides. ``train_phi``
    is converted to a tuple once and shared across all returned partiallers (it is
    the same Phi_train for every target in the fold)."""

    p_u = len(unpen[0]) if unpen and unpen[0] else 0
    eye = _np.eye(n, dtype=_np.float64)
    factor = _np_chol_factor(gram + ridge_lambda * eye)
    train_phi = tuple(tuple(row) for row in phi)

    Y = _np.asarray(targets, dtype=_np.float64).T  # (n, n_targets)
    n_t = Y.shape[1]

    if p_u:
        u_mat = _np.asarray(unpen, dtype=_np.float64)             # (n, p_u)
        # Solve K^{-1} [Y | U] once against the shared factor.
        rhs = _np.concatenate([Y, u_mat], axis=1)                # (n, n_t + p_u)
        sol = _np_chol_solve(factor, rhs)
        k_inv_Y = sol[:, :n_t]                                   # (n, n_t)
        k_inv_u = sol[:, n_t:]                                   # (n, p_u)
        a_mat = u_mat.T @ k_inv_u + 1e-12 * _np.eye(p_u, dtype=_np.float64)
        # Schur solve uses the SAME default 1e-10 jitter as the per-target path
        # (_np_solve_spd -> _np_chol_factor default), so the b coefficients match.
        a_factor = _np_chol_factor(a_mat)                       # tiny SPD p_u x p_u
        rhs_b = u_mat.T @ k_inv_Y                                # (p_u, n_t)
        B = _np_chol_solve(a_factor, rhs_b)                     # (p_u, n_t) coefficients b per target
        resid_targets = Y - u_mat @ B                            # (n, n_t)
        alpha = _np_chol_solve(factor, resid_targets)           # (n, n_t)
        B_cols = [B[:, t] for t in range(n_t)]
    else:
        alpha = _np_chol_solve(factor, Y)
        B_cols = [_np.zeros(0, dtype=_np.float64) for _ in range(n_t)]

    out: list[DualRidgePartialOut] = []
    for t in range(n_t):
        out.append(
            DualRidgePartialOut(
                penalized=penalized,
                ridge_lambda=ridge_lambda,
                train_phi=train_phi,
                unpen_coef=tuple(float(v) for v in B_cols[t]),
                dual_coef=tuple(float(v) for v in alpha[:, t]),
            )
        )
    return out


@dataclass(frozen=True)
class RidgeLambdaCVResult:
    """The registered 5-fold ridge-lambda CV selection (REDESIGN_v4 §4.2).

    ``selected_lambda`` is ``selected_multiplier * sigma2_phi`` -- the value that
    minimizes the held-out control-fit criterion over the registered grid. The
    per-lambda criterion (mean out-of-fold control-residual SS, summed over the
    ``Y`` and ``T`` nuisance fits) is persisted for the result JSON so the
    selection is auditable.
    """

    selected_lambda: float
    selected_multiplier: float
    sigma2_phi: float
    grid_multipliers: tuple[float, ...]
    grid_lambdas: tuple[float, ...]
    per_lambda_criterion: tuple[float, ...]
    n_cv_folds: int


def select_ridge_lambda_cv(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    penalized_columns: Sequence[int],
    sigma2_phi: float,
    grid_multipliers: Sequence[float] = RIDGE_LAMBDA_GRID_MULTIPLIERS,
    n_cv_folds: int = DEFAULT_LAMBDA_CV_FOLDS,
    strata: Sequence[int] | None = None,
    seed: int = 0,
    use_dual: bool = True,
) -> RidgeLambdaCVResult:
    """Registered 5-fold ridge-lambda CV (REDESIGN_v4 §4.2 line 772; v5 §3.2).

    Selects ``lambda_ridge`` from the **frozen** grid
    ``{1e-2, 1e-1, 1, 10, 100} * sigma^2_Phi`` by 5-fold cross-validation on the
    **held-out control-fit criterion**: for each candidate lambda, fit the
    registered dual-ridge nuisance maps (``b_lambda^Y`` for ``Y`` and ``B_lambda``
    for each ``T`` column, Eq. 5-1/5-1e) on the CV-train rows and measure the
    out-of-fold residual sum-of-squares of those control fits, i.e. how well the
    ridge-penalized control block ``Z`` predicts ``Y`` and ``T`` on rows it never
    saw. The lambda with the smallest mean out-of-fold control-residual SS wins.

    This is purely a hyperparameter selector for the nuisance; it does NOT touch
    the partial-R^2 / permutation / BCa statistic, the estimand, or the LOCI
    clustering. It mirrors the registered leakage rule (train-estimated lambda
    applied out-of-sample). ``sigma2_phi`` is ``EndpointControl.phi_variance``.

    Returns the selected lambda plus the per-lambda criterion vector for the JSON.
    """

    n = len(outcome)
    if strata is None:
        strata = [0] * n
    cv_folds = max(2, min(n_cv_folds, n))
    rng = random.Random(_stable_seed(seed, "lambda_cv"))
    folds = _stratified_folds(n, strata, cv_folds, rng)

    grid_lambdas = tuple(float(m) * float(sigma2_phi) for m in grid_multipliers)
    criteria: list[float] = []

    for lam in grid_lambdas:
        fold_scores: list[float] = []
        for fold in folds:
            if not fold:
                continue
            fold_members = set(fold)
            train_idx = [i for i in range(n) if i not in fold_members]
            if not train_idx:
                continue
            train_control = [control[i] for i in train_idx]
            train_outcome = [outcome[i] for i in train_idx]
            train_traj = [trajectory[i] for i in train_idx]
            if use_dual:
                outcome_partialler, column_partiallers = dual_ridge_partial_out(
                    train_control, train_outcome, train_traj,
                    penalized_columns=penalized_columns, ridge_lambda=lam,
                )
            else:
                outcome_partialler, column_partiallers = ridge_partial_out(
                    train_control, train_outcome, train_traj,
                    penalized_columns=penalized_columns, ridge_lambda=lam,
                )
            # Held-out control-fit residual SS for Y and each T column. Residuals
            # are batched per-partialler (numpy kernel application); the SS reduction
            # keeps the SAME per-row, per-column term order via math.fsum.
            r_y = _apply_residuals(outcome_partialler, control, outcome, fold)
            r_t_cols = [
                _apply_residuals_col(column_partiallers[col], control, trajectory, col, fold)
                for col in range(len(column_partiallers))
            ]
            terms: list[float] = []
            for pos in range(len(fold)):
                terms.append(r_y[pos] * r_y[pos])
                for col in range(len(column_partiallers)):
                    rt = r_t_cols[col][pos]
                    terms.append(rt * rt)
            ss = math.fsum(terms)
            fold_scores.append(ss / max(1, len(fold)))
        criteria.append(math.fsum(fold_scores) / len(fold_scores) if fold_scores else math.inf)

    # Argmin over the grid (ties -> smaller lambda, i.e. lower index, for stability).
    best_idx = min(range(len(grid_lambdas)), key=lambda k: (criteria[k], k))
    return RidgeLambdaCVResult(
        selected_lambda=grid_lambdas[best_idx],
        selected_multiplier=float(grid_multipliers[best_idx]),
        sigma2_phi=float(sigma2_phi),
        grid_multipliers=tuple(float(m) for m in grid_multipliers),
        grid_lambdas=grid_lambdas,
        per_lambda_criterion=tuple(criteria),
        n_cv_folds=cv_folds,
    )


def _ols_predict_rss(design: Matrix, target: Vector) -> tuple[Vector, float]:
    """OLS fit (tiny ridge for stability) returning ``(coef, RSS)`` in-sample."""

    if not design or not design[0]:
        # No predictors: best constant is the mean (but residuals already centered).
        rss = math.fsum(value * value for value in target)
        return [], rss
    gram = _matmul_at_a(design)
    for i in range(len(gram)):
        gram[i][i] += 1e-9
    rhs = _matvec_at_b(design, target)
    coef = _solve_spd(gram, rhs)
    rss = 0.0
    for row, value in zip(design, target):
        prediction = math.fsum(c * x for c, x in zip(coef, row))
        rss += (value - prediction) ** 2
    return coef, rss


# --------------------------------------------------------------------------- #
# Cross-fit partial-R^2 (Eq. 10a-b, §2.4).                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CrossFitResidualResult:
    """Held-out cross-fit residual statistics + permutation inference."""

    partial_r2: float          # median-of-folds partial-R^2 (Eq. 10a)
    delta_r2_overall: float    # median-of-folds overall-R^2 increment (Eq. 10b)
    beta_t: tuple[float, ...]  # median-of-folds partialled coefficient (Eq. 9)
    n_folds: int
    permutation_p_value: float | None = None
    n_permutations: int = 0
    fold_partial_r2: tuple[float, ...] = field(default_factory=tuple)
    nested_f_descriptive: float | None = None  # in-fold descriptive ONLY (never inferential)


def _stratified_folds(
    n: int, strata: Sequence[int], n_folds: int, rng: random.Random
) -> list[list[int]]:
    """Family-stratified, example-disjoint fold assignment (§2.4)."""

    by_stratum: dict[int, list[int]] = {}
    for index in range(n):
        by_stratum.setdefault(strata[index], []).append(index)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for stratum_indices in by_stratum.values():
        shuffled = stratum_indices[:]
        rng.shuffle(shuffled)
        for position, index in enumerate(shuffled):
            folds[position % n_folds].append(index)
    return folds


@dataclass(frozen=True)
class FoldResiduals:
    """Held-out, endpoint-orthogonalized residuals for ONE cross-fit fold.

    These are the ``M_lambda Y`` and ``M_lambda T`` residuals on the test fold,
    computed with the train-frozen ridge partialling-out and partialled
    regression. The locked conditional-null permutation (§2.4 step 2) permutes
    the ROWS of ``t_resid`` (i.e. ``M_lambda T``) within strata and recomputes
    the held-out statistic from THESE residuals -- no re-orthogonalization, no
    raw-row reshuffle before cross-fit.
    """

    test_idx: tuple[int, ...]
    y_resid: tuple[float, ...]               # M_lambda Y on the test fold
    t_resid: tuple[tuple[float, ...], ...]   # M_lambda T on the test fold (n_test x q)
    beta_t: tuple[float, ...]                # train-frozen partialled coefficient
    tss: float                               # held-out TSS about the train mean


def _apply_residuals(
    partialler: object,
    control: Matrix,
    target: Vector,
    idx: Sequence[int],
) -> list[float]:
    """Residuals of ``target`` against ``partialler`` over rows ``idx`` (batched if
    the partialler is a dual one with a numpy ``residuals_batch``)."""

    rows = [control[i] for i in idx]
    vals = [target[i] for i in idx]
    batch = getattr(partialler, "residuals_batch", None)
    if batch is not None:
        return batch(rows, vals)
    return [partialler.residual(control[i], target[i]) for i in idx]


def _apply_residuals_col(
    partialler: object,
    control: Matrix,
    trajectory: Matrix,
    col: int,
    idx: Sequence[int],
) -> list[float]:
    """Residuals of trajectory column ``col`` against ``partialler`` over ``idx``."""

    rows = [control[i] for i in idx]
    vals = [trajectory[i][col] for i in idx]
    batch = getattr(partialler, "residuals_batch", None)
    if batch is not None:
        return batch(rows, vals)
    return [partialler.residual(control[i], trajectory[i][col]) for i in idx]


def _fold_residuals(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    penalized_columns: Sequence[int],
    ridge_lambda: float,
    use_dual: bool,
) -> FoldResiduals:
    """Fit ``M_lambda`` + ``beta_T`` on TRAIN; return the TEST-fold residuals.

    Uses the **dual / kernel** ridge by default (``use_dual``) so the FULL
    ``phi_end`` control is feasible at 7B scale; the primal path is kept for
    small-control sanity checks. The returned residuals are the inputs the
    conditional-null permutation operates on.
    """

    train_control = [control[i] for i in train_idx]
    train_outcome = [outcome[i] for i in train_idx]
    train_traj = [trajectory[i] for i in train_idx]

    if use_dual:
        outcome_partialler, column_partiallers = dual_ridge_partial_out(
            train_control, train_outcome, train_traj,
            penalized_columns=penalized_columns, ridge_lambda=ridge_lambda,
        )
    else:
        outcome_partialler, column_partiallers = ridge_partial_out(
            train_control, train_outcome, train_traj,
            penalized_columns=penalized_columns, ridge_lambda=ridge_lambda,
        )

    # Orthogonalize TRAIN residuals and fit beta_T on them (Eq. 9). The kernel
    # APPLICATION (k_* alpha over all rows x all columns) is batched per-partialler
    # when the dual partialler supports it (numpy fast-path); otherwise per-row.
    train_y_resid = _apply_residuals(outcome_partialler, control, outcome, train_idx)
    q = len(column_partiallers)
    train_t_cols = [
        _apply_residuals_col(column_partiallers[col], control, trajectory, col, train_idx)
        for col in range(q)
    ]
    train_t_resid = [[train_t_cols[col][r] for col in range(q)] for r in range(len(train_idx))]
    beta_t, _ = _ols_predict_rss(train_t_resid, train_y_resid)

    # Apply the SAME train-frozen partiallers out-of-sample on TEST: M_lambda Y, M_lambda T.
    test_y_resid = _apply_residuals(outcome_partialler, control, outcome, test_idx)
    test_t_cols = [
        _apply_residuals_col(column_partiallers[col], control, trajectory, col, test_idx)
        for col in range(q)
    ]
    test_t_resid = [
        tuple(test_t_cols[col][r] for col in range(q)) for r in range(len(test_idx))
    ]

    train_mean = math.fsum(train_outcome) / len(train_outcome)
    tss = math.fsum((outcome[i] - train_mean) ** 2 for i in test_idx)
    return FoldResiduals(
        test_idx=tuple(test_idx),
        y_resid=tuple(test_y_resid),
        t_resid=tuple(test_t_resid),
        beta_t=tuple(beta_t),
        tss=tss,
    )


def _statistic_from_residuals(
    y_resid: Sequence[float],
    t_resid: Sequence[Sequence[float]],
    beta_t: Sequence[float],
    tss: float,
) -> tuple[float, float]:
    """Held-out ``(partial_R^2, delta_R^2_overall)`` from frozen residuals (Eq. 10).

    Reduced model = endpoint-only (predict the Y residual by 0); full model adds
    the train-frozen ``beta_T`` applied to the (possibly permuted) ``M_lambda T``
    residual rows. Permuting ``t_resid`` rows here IS the conditional-null
    permutation -- nothing is re-orthogonalized.
    """

    rss_red = math.fsum(v * v for v in y_resid)
    rss_full = 0.0
    for row, value in zip(t_resid, y_resid):
        prediction = math.fsum(b * x for b, x in zip(beta_t, row)) if beta_t else 0.0
        rss_full += (value - prediction) ** 2
    partial_r2 = (rss_red - rss_full) / rss_red if rss_red > 0 else 0.0
    delta_r2_overall = (rss_red - rss_full) / tss if tss > 0 else 0.0
    return partial_r2, delta_r2_overall


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _crossfit_folds(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    strata: Sequence[int],
    penalized_columns: Sequence[int],
    ridge_lambda: float,
    n_folds: int,
    seed: int,
    use_dual: bool,
) -> list[FoldResiduals]:
    """Compute and cache the held-out endpoint-orthogonalized residuals per fold."""

    n = len(outcome)
    if n < n_folds:
        n_folds = max(2, n)
    rng = random.Random(_stable_seed(seed, "folds"))
    folds = _stratified_folds(n, strata, n_folds, rng)

    fold_residuals: list[FoldResiduals] = []
    for fold in folds:
        if not fold:
            continue
        fold_members = set(fold)
        train_idx = [i for i in range(n) if i not in fold_members]
        if not train_idx:
            continue
        fold_residuals.append(
            _fold_residuals(
                control, outcome, trajectory,
                train_idx=train_idx, test_idx=fold,
                penalized_columns=penalized_columns, ridge_lambda=ridge_lambda,
                use_dual=use_dual,
            )
        )
    return fold_residuals


def _aggregate(fold_residuals: Sequence[FoldResiduals], q: int) -> CrossFitResidualResult:
    """Median-of-folds aggregation of the cached residual statistics."""

    fold_partials: list[float] = []
    fold_overall: list[float] = []
    for fr in fold_residuals:
        partial_r2, delta_overall = _statistic_from_residuals(
            fr.y_resid, fr.t_resid, fr.beta_t, fr.tss
        )
        fold_partials.append(partial_r2)
        fold_overall.append(delta_overall)
    median_beta = tuple(
        _median([fr.beta_t[col] for fr in fold_residuals if col < len(fr.beta_t)])
        for col in range(q)
    )
    return CrossFitResidualResult(
        partial_r2=_median(fold_partials),
        delta_r2_overall=_median(fold_overall),
        beta_t=median_beta,
        n_folds=len(fold_partials),
        fold_partial_r2=tuple(fold_partials),
    )


def cross_fit_partial_r2(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    strata: Sequence[int] | None = None,
    penalized_columns: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = 0,
    use_dual: bool = True,
) -> CrossFitResidualResult:
    """K-fold cross-fit held-out partial-R^2 (Eq. 10a-b), median-of-folds (§2.4).

    No permutation p-value is computed here (call :func:`block_permutation_test`
    or :func:`residualized_regression_test` for the OOS p-value). The nested-model
    F is NOT reported as an inferential statistic anywhere. ``use_dual`` selects
    the scalable n-space ridge (default) so the FULL ``phi_end`` control is
    feasible at 7B hidden dims.
    """

    n = len(outcome)
    if strata is None:
        strata = [0] * n
    fold_residuals = _crossfit_folds(
        control, outcome, trajectory,
        strata=strata, penalized_columns=penalized_columns, ridge_lambda=ridge_lambda,
        n_folds=n_folds, seed=seed, use_dual=use_dual,
    )
    q = len(trajectory[0]) if trajectory and trajectory[0] else 0
    return _aggregate(fold_residuals, q)


def block_permutation_test(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    strata: Sequence[int] | None = None,
    penalized_columns: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE,
    n_folds: int = DEFAULT_FOLDS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
    use_dual: bool = True,
) -> CrossFitResidualResult:
    """Cross-fit partial-R^2 with the LOCKED conditional-null permutation p (§2.4).

    **The conditional null (REDESIGN_v4 §2.4 step 2, line 323).** Under H0 "T adds
    nothing given the controls", the trajectory block is exchangeable *conditional
    on the endpoint control*. We therefore permute the rows of the
    **endpoint-orthogonalized residuals ``M_lambda T``** -- the train-frozen
    held-out cross-fit residuals -- **within family x fold strata**, recompute the
    held-out partial-R^2 from the SAME residuals, and report the right-tail rank.

    This is the fix for the earlier bug that permuted *raw* trajectory rows before
    cross-fit: that reshuffles the endpoint covariance the null must hold fixed and
    re-orthogonalizes the controls against shuffled data, so it does not test the
    registered conditional null. Here the orthogonalization is computed ONCE per
    fold and only the residual rows are permuted -- the exact, distribution-free
    conditional test. The strata are ``(family, fold)`` so exchangeability holds
    within each held-out fold separately.

    The permutation p-value uses the standard ``(1 + #{perm >= obs}) / (1 + P)``
    add-one estimator (never zero).
    """

    n = len(outcome)
    if strata is None:
        strata = [0] * n
    q = len(trajectory[0]) if trajectory and trajectory[0] else 0

    fold_residuals = _crossfit_folds(
        control, outcome, trajectory,
        strata=strata, penalized_columns=penalized_columns, ridge_lambda=ridge_lambda,
        n_folds=n_folds, seed=seed, use_dual=use_dual,
    )
    observed = _aggregate(fold_residuals, q)

    # Pre-group, PER FOLD, the test-row positions by family stratum: the locked
    # strata are family x fold, so permutation is within (family, fold).
    fold_strata_groups: list[list[list[int]]] = []
    for fr in fold_residuals:
        groups: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(fr.test_idx):
            groups.setdefault(strata[global_idx], []).append(local_pos)
        fold_strata_groups.append(list(groups.values()))

    # The held-out statistic depends on the (possibly permuted) M_lambda T rows
    # ONLY through the scalar projection proj_i = beta_t . t_resid_i (the full-model
    # prediction). Permuting t_resid rows within a stratum is therefore IDENTICAL to
    # permuting the precomputed scalars proj_i within that stratum. We precompute
    # proj / rss_red ONCE per fold; each permutation then reduces to an O(n_test)
    # gather + squared-error sum instead of O(n_test * q) row copies + re-projection.
    # The RNG draw ORDER is preserved EXACTLY (same per-stratum rng.shuffle calls in
    # the same order), so the permutation distribution -- and thus the p-value -- is
    # reproducible-identical to the pure-row-permutation path. (Only partial_r2 drives
    # the right-tail rank; delta_R^2_overall was computed-but-unused in the old loop.)
    fold_proj: list[list[float]] = []
    fold_y: list[list[float]] = []
    fold_rss_red: list[float] = []
    for fr in fold_residuals:
        beta_t = fr.beta_t
        if beta_t:
            proj = [math.fsum(b * x for b, x in zip(beta_t, row)) for row in fr.t_resid]
        else:
            proj = [0.0] * len(fr.y_resid)
        fold_proj.append(proj)
        fold_y.append(list(fr.y_resid))
        fold_rss_red.append(math.fsum(v * v for v in fr.y_resid))

    rng = random.Random(_stable_seed(seed, "perm"))
    observed_stat = observed.partial_r2

    if _HAVE_NUMPY:
        np_proj = [_np.asarray(p, dtype=_np.float64) for p in fold_proj]
        np_y = [_np.asarray(y, dtype=_np.float64) for y in fold_y]
        at_least_as_extreme = 0
        for _ in range(n_permutations):
            perm_partials: list[float] = []
            for fi, groups in enumerate(fold_strata_groups):
                proj = np_proj[fi]
                # Build the permutation index: dest position `positions[k]` receives
                # the source at `shuffled[k]` (matches permuted_t[dest]=t_resid[src]).
                perm_idx = _np.arange(proj.shape[0])
                for positions in groups:
                    if len(positions) < 2:
                        continue
                    shuffled = positions[:]
                    rng.shuffle(shuffled)
                    for src, dest in zip(positions, shuffled):
                        perm_idx[dest] = src
                proj_perm = proj[perm_idx]
                diff = np_y[fi] - proj_perm
                rss_full = float(diff @ diff)
                rss_red = fold_rss_red[fi]
                perm_partials.append((rss_red - rss_full) / rss_red if rss_red > 0 else 0.0)
            if _median(perm_partials) >= observed_stat:
                at_least_as_extreme += 1
    else:
        at_least_as_extreme = 0
        for _ in range(n_permutations):
            perm_partials = []
            for fi, groups in enumerate(fold_strata_groups):
                proj = fold_proj[fi]
                proj_perm = list(proj)
                for positions in groups:
                    if len(positions) < 2:
                        continue
                    shuffled = positions[:]
                    rng.shuffle(shuffled)
                    source_vals = [proj[p] for p in positions]
                    for dest, val in zip(shuffled, source_vals):
                        proj_perm[dest] = val
                y = fold_y[fi]
                rss_full = math.fsum((y[i] - proj_perm[i]) ** 2 for i in range(len(y)))
                rss_red = fold_rss_red[fi]
                perm_partials.append((rss_red - rss_full) / rss_red if rss_red > 0 else 0.0)
            if _median(perm_partials) >= observed_stat:
                at_least_as_extreme += 1

    p_value = (1 + at_least_as_extreme) / (1 + n_permutations)
    return CrossFitResidualResult(
        partial_r2=observed.partial_r2,
        delta_r2_overall=observed.delta_r2_overall,
        beta_t=observed.beta_t,
        n_folds=observed.n_folds,
        permutation_p_value=p_value,
        n_permutations=n_permutations,
        fold_partial_r2=observed.fold_partial_r2,
    )


def residualized_regression_test(
    control: Matrix,
    outcome: Vector,
    trajectory: Matrix,
    *,
    strata: Sequence[int] | None = None,
    penalized_columns: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE,
    n_folds: int = DEFAULT_FOLDS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
    use_dual: bool = True,
) -> CrossFitResidualResult:
    """End-to-end co-primary test: cross-fit partial-R^2 + block-permutation p.

    Public entry point. Returns the partial effect (``partial_r2``, ``beta_t``),
    the deployment-relevance increment (``delta_r2_overall``), and the
    out-of-sample permutation p-value (``permutation_p_value``). The caller
    applies the locked decision rule (permutation p < 0.05 after Holm AND BCa-CI
    of ``partial_r2`` excludes the floor; §4.2) and the robustness floor across
    PCA-``r`` controls. ``use_dual`` keeps the scalable n-space ridge (default).
    """

    return block_permutation_test(
        control,
        outcome,
        trajectory,
        strata=strata,
        penalized_columns=penalized_columns,
        ridge_lambda=ridge_lambda,
        n_folds=n_folds,
        n_permutations=n_permutations,
        seed=seed,
        use_dual=use_dual,
    )

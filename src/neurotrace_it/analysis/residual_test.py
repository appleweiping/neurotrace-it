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

Matrix = list[list[float]]
Vector = list[float]

__all__ = [
    "DEFAULT_FOLDS",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_RIDGE",
    "RidgePartialOut",
    "DualRidgePartialOut",
    "CrossFitResidualResult",
    "ridge_partial_out",
    "dual_ridge_partial_out",
    "cross_fit_partial_r2",
    "block_permutation_test",
    "residualized_regression_test",
]

DEFAULT_FOLDS = 10            # K-fold cross-fit (§2.4)
DEFAULT_PERMUTATIONS = 5000   # block permutation count P (§2.4)
DEFAULT_RIDGE = 1.0           # placeholder lambda; CV-locked per split in the plan


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
    """

    size = len(matrix)
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
    # Linear kernel G = Phi Phi^T (n x n) -- the only object touching p_phi.
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
    q = len(trajectory[0]) if trajectory and trajectory[0] else 0
    column_partiallers: list[DualRidgePartialOut] = []
    for col in range(q):
        column = [row[col] for row in trajectory]
        column_partiallers.append(
            _fit_dual_ridge_control(phi, unpen, gram, column, penalized, ridge_lambda)
        )
    return outcome_partialler, column_partiallers


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

    # Orthogonalize TRAIN residuals and fit beta_T on them (Eq. 9).
    train_y_resid = [outcome_partialler.residual(control[i], outcome[i]) for i in train_idx]
    train_t_resid = [
        [p.residual(control[i], trajectory[i][col]) for col, p in enumerate(column_partiallers)]
        for i in train_idx
    ]
    beta_t, _ = _ols_predict_rss(train_t_resid, train_y_resid)

    # Apply the SAME train-frozen partiallers out-of-sample on TEST: M_lambda Y, M_lambda T.
    test_y_resid = [outcome_partialler.residual(control[i], outcome[i]) for i in test_idx]
    test_t_resid = [
        tuple(p.residual(control[i], trajectory[i][col]) for col, p in enumerate(column_partiallers))
        for i in test_idx
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

    rng = random.Random(_stable_seed(seed, "perm"))
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        perm_partials: list[float] = []
        perm_overall: list[float] = []
        for fr, groups in zip(fold_residuals, fold_strata_groups):
            # Permute M_lambda T rows within each (family, fold) stratum.
            permuted_t = list(fr.t_resid)
            for positions in groups:
                if len(positions) < 2:
                    continue
                shuffled = positions[:]
                rng.shuffle(shuffled)
                source_rows = [fr.t_resid[p] for p in positions]
                for dest, row in zip(shuffled, source_rows):
                    permuted_t[dest] = row
            partial_r2, delta_overall = _statistic_from_residuals(
                fr.y_resid, permuted_t, fr.beta_t, fr.tss
            )
            perm_partials.append(partial_r2)
            perm_overall.append(delta_overall)
        if _median(perm_partials) >= observed.partial_r2:
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

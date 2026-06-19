"""Per-layer routing policy score ``psi_l`` and the COUPLING IDENTITY (Eq. 5-1..5-4).

REDESIGN_v5 §3.4, DO-NOT-RUN. This module exposes the frozen ridge nuisance map,
the OUT-OF-SAMPLE residualization, and the per-layer decomposition of the
orthogonalized trajectory coefficient ``beta_T`` into ``psi_l``, with the
mathematical spine asserted both in- and out-of-sample:

    psi_l(x) = beta_{D,l} . D~_l(x) + beta_{kappa,l} . kappa~_l(x),   l in A   (Eq. 5-3)

    sum_{l in A} psi_l(x) = beta_T . T~(x)                                     (Eq. 5-4)

where ``T~(x) = T(x) - Z(x) . B_lambda`` (Eq. 5-1d) is the
endpoint+NAIT-orthogonalized trajectory residual, defined for ANY ``x`` (in or
out of sample) via the FROZEN coefficient map ``B_lambda``. The identity holds
out-of-sample because both sides use the same frozen ``B_lambda, beta_T``.

``psi_l`` is explicitly a PREDICTIVE policy score, not a causal layer-write
effect: it is fit to predict the observational ``Y_obs`` and merely *proposes*
the mask; the certification that writing to ``supp(psi)`` is the right place is
the policy-value Gate R1 (§3.5), not this decomposition.

The trajectory ``T(x) = ({D_l}, {kappa_l})_{l in A}`` stacks the per-layer
coordinates; ``beta_T`` is the single orthogonalized coefficient on ``T~``. The
coupling identity is therefore pure algebra of the stacking (Eq. 5-4).

DO-NOT-RUN: pure stdlib; no model load, no server call, no training. The caller
supplies the fitted ridge maps / coefficients (run-later); the arithmetic here is
testable build-now / run-later. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "FrozenNuisanceMap",
    "PerLayerScore",
    "frozen_nuisance_map",
    "residualize_out_of_sample",
    "per_layer_policy_score",
    "coupling_residual",
    "assert_coupling_identity",
]


def _as_floats(vector: Sequence[float]) -> list[float]:
    return [float(c) for c in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """``matrix^T . vector``-free plain ``M . v`` where ``M`` is row-major (rows x cols)."""

    return [_dot(row, vector) for row in matrix]


def _zt_times_B(z: Sequence[float], B: Sequence[Sequence[float]]) -> list[float]:
    """Compute ``Z(x) . B_lambda`` where ``B`` is ``p x (2|A|)`` row-major and ``z`` is length ``p``.

    Returns the length-``2|A|`` row vector ``z . B``.
    """

    if len(z) != len(B):
        raise ValueError(f"Z length {len(z)} != B rows {len(B)} (p mismatch)")
    out_dim = len(B[0]) if B else 0
    out = [0.0] * out_dim
    for i, zi in enumerate(z):
        row = B[i]
        if len(row) != out_dim:
            raise ValueError("ragged B_lambda matrix")
        for j in range(out_dim):
            out[j] += zi * row[j]
    return out


def _solve_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    lam: float,
    *,
    penalized_columns: Sequence[int] | None = None,
) -> list[list[float]]:
    """Closed-form multi-output ridge ``B = (X^T X + lam Omega)^{-1} X^T Y`` (pure stdlib).

    ``features`` is ``n x p``; ``targets`` is ``n x m``; returns ``B`` of shape
    ``p x m`` (row-major) so ``x_row . B`` predicts a length-``m`` target row.
    Used to FREEZE the nuisance map; small ``p`` is expected (control row).

    ``Omega`` is the REGISTERED BLOCK penalty (REDESIGN_v5 §3.2, Eq. 8c, matching
    ``residual_test._fit_ridge_control``): identity on ``penalized_columns`` (the
    wide ``phi_end`` nuisance block) and zero elsewhere, so the controls
    ``s_NAIT(L) / V_proj(L) / C`` and the intercept enter UNPENALIZED. When
    ``penalized_columns is None`` every column is penalized (the legacy uniform
    ridge), retained only for callers that have no nuisance/control split.
    """

    n = len(features)
    if n == 0:
        raise ValueError("ridge needs at least one observation")
    p = len(features[0])
    m = len(targets[0]) if targets else 0
    penalized = set(range(p)) if penalized_columns is None else set(penalized_columns)

    # Gram = X^T X + lam Omega  (p x p);  rhs = X^T Y  (p x m).
    gram = [[0.0] * p for _ in range(p)]
    rhs = [[0.0] * m for _ in range(p)]
    for i in range(n):
        xi = _as_floats(features[i])
        yi = _as_floats(targets[i])
        for a in range(p):
            xa = xi[a]
            for b in range(p):
                gram[a][b] += xa * xi[b]
            for c in range(m):
                rhs[a][c] += xa * yi[c]
    for a in range(p):
        if a in penalized:
            gram[a][a] += lam

    return _gauss_solve_multi(gram, rhs)


def _gauss_solve_multi(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    """Solve ``matrix . X = rhs`` for ``X`` via Gaussian elimination with partial pivoting."""

    p = len(matrix)
    m = len(rhs[0]) if rhs else 0
    # Augment.
    aug = [matrix[i][:] + rhs[i][:] for i in range(p)]
    for col in range(p):
        # Partial pivot.
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-15:
            aug[pivot][col] += 1e-12  # tiny ridge fallback for singularity
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for j in range(col, p + m):
            aug[col][j] /= piv
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, p + m):
                aug[r][j] -= factor * aug[col][j]
    return [[aug[i][p + j] for j in range(m)] for i in range(p)]


@dataclass(frozen=True)
class FrozenNuisanceMap:
    """Frozen ridge nuisance coefficient maps (Eq. 5-1 / 5-1e).

    ``B_lambda`` is the ``p x (2|A|)`` map for the trajectory ``T``;
    ``b_lambda_Y`` is the length-``p`` map for ``Y_obs``. Both are FROZEN once
    on ``P_train`` and applied unchanged to any ``x`` (out-of-sample), so the
    residual ``T~`` and ``psi`` are closed-form functions of ``x``.

    ``penalized_columns`` records the REGISTERED BLOCK penalty support: the
    indices of the wide ``phi_end`` nuisance columns in ``Z`` that the ridge
    ``Omega`` penalizes (controls ``s_NAIT/V_proj/C`` and the intercept stay
    unpenalized). It is the frozen provenance of which columns were shrunk; an
    empty tuple means the legacy uniform ridge (every column penalized).
    """

    B_lambda: tuple[tuple[float, ...], ...]   # p x 2|A|
    b_lambda_Y: tuple[float, ...]             # length p
    lam: float
    anchor_order: tuple[int, ...]             # the 2|A| coordinate layout key
    penalized_columns: tuple[int, ...] = ()   # frozen phi_end penalty support (Omega)


def frozen_nuisance_map(
    Z: Sequence[Sequence[float]],
    T: Sequence[Sequence[float]],
    Y_obs: Sequence[float],
    *,
    lam: float,
    anchors: Sequence[int],
    penalized_columns: Sequence[int] | None = None,
) -> FrozenNuisanceMap:
    """Fit and FREEZE the ridge nuisance maps ``(B_lambda, b_lambda_Y)`` (Eq. 5-1/5-1e).

    ``Z`` is the ``n x p`` control feature matrix ``[phi_end, s_NAIT, V_proj, C, 1]``;
    ``T`` is the ``n x (2|A|)`` trajectory matrix; ``Y_obs`` the length-``n`` R0
    target. The maps are frozen here and never refit at deployment.

    ``penalized_columns`` is the REGISTERED BLOCK penalty (REDESIGN_v5 §3.2,
    Eq. 8c -- the same ``Omega`` as ``residual_test._fit_ridge_control``): pass
    the indices of the wide ``phi_end`` nuisance columns so ONLY they are shrunk,
    leaving ``s_NAIT(L)``, ``V_proj(L)``, ``C`` and the intercept UNPENALIZED. The
    deployed R0 map MUST pass this (``run_r0_analysis`` builds it from the control
    layout) so the frozen ``B_lambda`` matches the residual-test pattern and the
    registered ``psi(x)`` estimand is unchanged. ``None`` keeps the legacy uniform
    ridge (every column penalized) for callers with no nuisance/control split;
    note the coupling identity (Eq. 5-4) is pure stacking algebra and holds for
    ANY frozen ``B_lambda`` regardless of the penalty support.
    """

    B = _solve_ridge(Z, T, lam, penalized_columns=penalized_columns)
    bY_mat = _solve_ridge(Z, [[y] for y in Y_obs], lam, penalized_columns=penalized_columns)
    bY = tuple(row[0] for row in bY_mat)
    return FrozenNuisanceMap(
        B_lambda=tuple(tuple(r) for r in B),
        b_lambda_Y=bY,
        lam=lam,
        anchor_order=tuple(anchors),
        penalized_columns=tuple(penalized_columns) if penalized_columns is not None else (),
    )


def residualize_out_of_sample(
    z: Sequence[float],
    t: Sequence[float],
    nuisance: FrozenNuisanceMap,
) -> list[float]:
    """OUT-OF-SAMPLE residual ``T~(x) = T(x) - Z(x) . B_lambda`` (Eq. 5-1d).

    Defined for ANY ``x`` (in or out of sample) using the FROZEN ``B_lambda``;
    no model is refit. Returns the length-``2|A|`` residual vector.
    """

    predicted = _zt_times_B(_as_floats(z), [list(r) for r in nuisance.B_lambda])
    t = _as_floats(t)
    if len(t) != len(predicted):
        raise ValueError(f"T dim {len(t)} != 2|A| = {len(predicted)}")
    return [ti - pi for ti, pi in zip(t, predicted)]


@dataclass(frozen=True)
class PerLayerScore:
    """Per-layer policy score ``psi_l`` and the coupling-identity witness (Eq. 5-3/5-4)."""

    psi: Mapping[int, float]             # {l: psi_l(x)}
    selection_score: float               # sum_l psi_l(x)
    coupling_rhs: float                  # beta_T . T~(x)
    identity_residual: float             # |sum_l psi_l - beta_T . T~|


def per_layer_policy_score(
    z: Sequence[float],
    t: Sequence[float],
    beta_T: Sequence[float],
    nuisance: FrozenNuisanceMap,
    *,
    anchors: Sequence[int],
) -> PerLayerScore:
    """Decompose ``beta_T`` into ``psi_l`` and assert the coupling identity (Eq. 5-3/5-4).

    The trajectory residual ``T~`` stacks per-layer coordinates in the order
    ``[D_{a0}, kappa_{a0}, D_{a1}, kappa_{a1}, ...]`` over ``anchors``; ``beta_T``
    is the matching orthogonalized coefficient. Then

        psi_l = beta_{D,l} . D~_l + beta_{kappa,l} . kappa~_l                 (Eq. 5-3)

    and ``sum_l psi_l = beta_T . T~`` (Eq. 5-4) by the stacking algebra -- the
    selection score is EXACTLY the endpoint+NAIT-residualized trajectory
    prediction, and the routing support is ``supp(psi) subseteq A``.
    """

    t_tilde = residualize_out_of_sample(z, t, nuisance)
    beta = _as_floats(beta_T)
    if len(beta) != 2 * len(anchors):
        raise ValueError(f"beta_T dim {len(beta)} != 2|A| = {2 * len(anchors)}")
    if len(t_tilde) != 2 * len(anchors):
        raise ValueError(f"T~ dim {len(t_tilde)} != 2|A| = {2 * len(anchors)}")

    psi: dict[int, float] = {}
    for idx, layer in enumerate(anchors):
        d_coord = 2 * idx
        k_coord = 2 * idx + 1
        psi[layer] = beta[d_coord] * t_tilde[d_coord] + beta[k_coord] * t_tilde[k_coord]

    selection = math.fsum(psi.values())
    rhs = _dot(beta, t_tilde)
    return PerLayerScore(
        psi=psi,
        selection_score=selection,
        coupling_rhs=rhs,
        identity_residual=abs(selection - rhs),
    )


def coupling_residual(score: PerLayerScore) -> float:
    """Convenience: ``|sum_l psi_l - beta_T . T~|`` (zero up to float error, Eq. 5-4)."""

    return score.identity_residual


def assert_coupling_identity(score: PerLayerScore, *, tol: float = 1e-9) -> None:
    """Raise if the coupling identity (Eq. 5-4) does not hold within ``tol``."""

    if score.identity_residual > tol:
        raise AssertionError(
            f"coupling identity violated: |sum psi_l - beta_T . T~| = "
            f"{score.identity_residual} > {tol}"
        )

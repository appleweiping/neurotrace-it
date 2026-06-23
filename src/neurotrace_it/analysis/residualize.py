"""Endpoint-control assembly, cluster bootstrap (BCa), multiplicity + decision.

The locked GATES around the co-primary residual statistic
(:mod:`neurotrace_it.analysis.residual_test`), per
``docs/redesign/REDESIGN_v4.md`` §2.3, §3.3, §4.2, §4.5:

* :func:`build_endpoint_control` -- assemble the control block ``Z = [phi_full,
  C, 1]`` (the FULL endpoint signature, standardized) AND the PCA-``r`` robustness
  poles for ``r in {8, 16, 32}``, returning the penalized-column indices and the
  per-pole controls so the co-primary regression can be run under the full-ridge
  control and each PCA pole (the §4.2 robustness floor: full-ridge survives AND
  monotone-stable across ``r`` with no sign flip / overlapping CIs).
* :func:`cluster_bootstrap_ci` -- the **BCa** 95% CI of the partial-R^2 (or any
  scalar functional), resampling **LOCI clusters** with replacement ``B = 2000``
  times to respect within-cluster dependence (§2.4 step 3, §4.2).
* :func:`two_layer_holm` -- the §4.5 two-layer Holm: within the trajectory family
  ``{joint, D-only, kappa-only}`` with **joint as gatekeeper**, then across
  metric families.
* :func:`contingency_decision` -- the §3.3 2x2 decision table as code.
* :func:`robustness_floor` -- the §4.2 robustness floor (full-ridge passes AND
  monotone-stable / no sign flip / overlapping CIs across PCA ``r``).
* :func:`achieved_power` -- the achieved-power / runtime calc for the
  ``P = 5000`` permutations x ``K = 10`` folds design (§4.2, §4.5).

Pure stdlib; build-now / run-later. No server call, no training, no model load.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

# Optional numpy acceleration for the PCA robustness-pole projection
# (:func:`_top_pca`). The pure-python deflated power iteration forms the n x n
# Gram and runs up to 100 iterations per component IN PYTHON -- O(n^2) per
# iteration, which dominates design assembly at n=2000 (minutes). numpy routes the
# Gram to a float64 matmul and the eigendecomposition to LAPACK ``eigh``. The PCA
# scores define a ridge-PENALIZED control SUBSPACE (a robustness pole); the
# downstream residual-maker M_lambda is invariant to the basis chosen within that
# top-r eigenspace, so the fast-path and the stdlib path yield the same control
# (and the same robustness verdict) -- only the within-subspace basis/sign differs.
# Without numpy the exact stdlib path is used, so the module stays import-safe.
try:  # pragma: no cover - presence-dependent
    import numpy as _np

    _HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    _np = None
    _HAVE_NUMPY = False

Matrix = list[list[float]]
Vector = list[float]

__all__ = [
    "PCA_RANKS",
    "DEFAULT_BOOTSTRAP",
    "MIN_CLUSTERS_FOR_ACCEL",
    "MAX_ABS_ACCEL",
    "EndpointControl",
    "BootstrapCI",
    "ContingencyDecision",
    "build_endpoint_control",
    "cluster_bootstrap_ci",
    "two_layer_holm",
    "contingency_decision",
    "robustness_floor",
    "achieved_power",
]

PCA_RANKS = (8, 16, 32)          # §4.2 robustness poles r in {8, 16, 32}
DEFAULT_BOOTSTRAP = 2000         # B_boot (§2.4 step 3)
# BCa small-g hardening: below this many clusters the jackknife acceleration is
# not estimable (we fall back to a=0, the bias-corrected interval); the cap keeps
# the BCa adjustment denominator 1 - a*(z0+z) well away from zero.
MIN_CLUSTERS_FOR_ACCEL = 4
MAX_ABS_ACCEL = 0.25


# --------------------------------------------------------------------------- #
# Endpoint control assembly + PCA-r robustness poles (§2.3, §4.2).            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EndpointControl:
    """The assembled control block(s) for the co-primary regression.

    ``full_control`` is ``Z = [phi_full (standardized), C, 1]`` with
    ``penalized_columns`` indexing the ``phi_full`` block (the only ridge-
    penalized columns; ``C`` and the intercept are unpenalized). ``pca_controls``
    maps each robustness rank ``r`` to ``Z_r = [phi_PCA_r, C, 1]`` with its own
    ``penalized_columns`` -- the §4.2 sensitivity poles. ``phi_variance`` is the
    mean per-column variance ``sigma^2_phi`` used to scale the locked
    ``lambda_ridge`` CV grid (§4.2: ``{1e-2,1e-1,1,10,100} * sigma^2_phi``).
    """

    full_control: Matrix
    penalized_columns: tuple[int, ...]
    pca_controls: dict[int, Matrix]
    pca_penalized_columns: dict[int, tuple[int, ...]]
    phi_variance: float


def _standardize_columns(matrix: Matrix) -> tuple[Matrix, list[float], list[float]]:
    """Z-score each column; return (standardized, means, stds). Zero-var -> std 1."""

    n = len(matrix)
    p = len(matrix[0]) if n else 0
    means = [math.fsum(matrix[i][j] for i in range(n)) / n for j in range(p)] if n else []
    stds: list[float] = []
    for j in range(p):
        var = math.fsum((matrix[i][j] - means[j]) ** 2 for i in range(n)) / n if n else 0.0
        stds.append(math.sqrt(var) if var > 1e-18 else 1.0)
    out = [[(matrix[i][j] - means[j]) / stds[j] for j in range(p)] for i in range(n)]
    return out, means, stds


def _top_pca(standardized_phi: Matrix, rank: int, *, seed: int = 0) -> Matrix:
    """Project standardized phi onto its top-``rank`` principal directions.

    Dependency-free: deflated power iteration on the ``p x p`` covariance via the
    ``n``-space Gram (we operate on scores directly to avoid forming ``p x p`` for
    wide phi). Returns the ``n x min(rank, p, n)`` score matrix (the PCA-``r``
    endpoint summary used as a robustness control pole).
    """

    n = len(standardized_phi)
    p = len(standardized_phi[0]) if n else 0
    eff = min(rank, p, n)
    if eff <= 0:
        return [[] for _ in range(n)]

    if _HAVE_NUMPY:
        # numpy fast-path: top-``eff`` eigenpairs of G = Phi Phi^T via LAPACK eigh.
        # Scores = u_k * sqrt(lambda_k) for the largest eigenvalues (same as the
        # converged power-iteration components, up to within-eigenspace basis/sign,
        # which is immaterial for a ridge-penalized control pole). Truncates at the
        # same lambda > 1e-12 positivity guard as the stdlib deflation.
        phi_mat = _np.asarray(standardized_phi, dtype=_np.float64)
        gram_np = phi_mat @ phi_mat.T
        gram_np = 0.5 * (gram_np + gram_np.T)
        eigvals, eigvecs = _np.linalg.eigh(gram_np)        # ascending
        order = eigvals[::-1][:eff]
        vecs = eigvecs[:, ::-1][:, :eff]
        keep = order > 1e-12
        order = order[keep]
        vecs = vecs[:, keep]
        scores = vecs * _np.sqrt(order)                     # (n, n_kept)
        return [[float(v) for v in row] for row in scores]

    # n x n Gram G = Phi Phi^T (avoids p x p for wide phi). Top eigenvectors of G
    # give the principal-component SCORES directly (score_k = sqrt(lambda_k) u_k).
    gram = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = standardized_phi[i]
        gram[i][i] = math.fsum(v * v for v in row_i)
        for j in range(i + 1, n):
            val = math.fsum(a * b for a, b in zip(row_i, standardized_phi[j]))
            gram[i][j] = val
            gram[j][i] = val

    rng = random.Random(seed)
    components: list[list[float]] = []  # each is a length-n score vector
    work = [row[:] for row in gram]
    for _ in range(eff):
        # Power iteration for the top eigenpair of the current deflated `work`.
        vec = [rng.gauss(0.0, 1.0) for _ in range(n)]
        norm = math.sqrt(math.fsum(c * c for c in vec)) or 1.0
        vec = [c / norm for c in vec]
        eigval = 0.0
        for _ in range(100):
            nxt = [math.fsum(work[i][k] * vec[k] for k in range(n)) for i in range(n)]
            norm = math.sqrt(math.fsum(c * c for c in nxt))
            if norm < 1e-12:
                break
            nxt = [c / norm for c in nxt]
            if math.fsum(abs(a - b) for a, b in zip(nxt, vec)) < 1e-10:
                vec = nxt
                break
            vec = nxt
        eigval = math.fsum(
            vec[i] * math.fsum(work[i][k] * vec[k] for k in range(n)) for i in range(n)
        )
        if eigval <= 1e-12:
            break
        score = [vec[i] * math.sqrt(eigval) for i in range(n)]
        components.append(score)
        # Deflate: work -= eigval * vec vec^T.
        for i in range(n):
            for k in range(n):
                work[i][k] -= eigval * vec[i] * vec[k]

    # Assemble n x len(components) score matrix.
    return [[components[c][i] for c in range(len(components))] for i in range(n)]


def build_endpoint_control(
    phi_end: Matrix,
    covariates: Matrix | None = None,
    *,
    pca_ranks: Sequence[int] = PCA_RANKS,
    seed: int = 0,
) -> EndpointControl:
    """Assemble ``Z = [phi_full, C, 1]`` plus PCA-``r`` robustness poles (§2.3, §4.2).

    Parameters
    ----------
    phi_end:
        The FULL endpoint signature design ``Phi in R^{n x 2d|A|}`` (Eq. 1).
        Standardized column-wise here.
    covariates:
        Nuisance covariates ``C`` (length, difficulty, dataset-family one-hots),
        ``n x p_C``; ``None`` => no covariates (just phi + intercept).
    pca_ranks:
        Robustness ranks ``r`` for the PCA control poles (default ``{8,16,32}``).
    seed:
        Seed for the (deterministic) PCA power iteration.

    Returns
    -------
    EndpointControl
        Full control + per-``r`` PCA controls + penalized-column indices + the
        mean phi variance used to scale the ridge grid.
    """

    n = len(phi_end)
    std_phi, _, _ = _standardize_columns(phi_end)
    p_phi = len(std_phi[0]) if n else 0
    cov = covariates if covariates is not None else [[] for _ in range(n)]
    p_c = len(cov[0]) if n and cov[0] else 0

    # Mean per-column phi variance after standardization is ~1; report the raw
    # mean variance for the lambda-grid scale (sigma^2_phi).
    raw_var = 0.0
    if n and p_phi:
        col_vars = []
        for j in range(p_phi):
            mean = math.fsum(phi_end[i][j] for i in range(n)) / n
            col_vars.append(math.fsum((phi_end[i][j] - mean) ** 2 for i in range(n)) / n)
        raw_var = math.fsum(col_vars) / len(col_vars)

    # Full control Z = [phi_full | C | 1]; phi columns are 0..p_phi-1 (penalized).
    full_control: Matrix = []
    for i in range(n):
        row = list(std_phi[i]) + list(cov[i]) + [1.0]
        full_control.append(row)
    penalized = tuple(range(p_phi))

    pca_controls: dict[int, Matrix] = {}
    pca_penalized: dict[int, tuple[int, ...]] = {}
    for r in pca_ranks:
        scores = _top_pca(std_phi, r, seed=seed)
        r_eff = len(scores[0]) if n and scores[0] else 0
        ctrl: Matrix = []
        for i in range(n):
            ctrl.append(list(scores[i]) + list(cov[i]) + [1.0])
        pca_controls[r] = ctrl
        pca_penalized[r] = tuple(range(r_eff))

    return EndpointControl(
        full_control=full_control,
        penalized_columns=penalized,
        pca_controls=pca_controls,
        pca_penalized_columns=pca_penalized,
        phi_variance=raw_var,
    )


# --------------------------------------------------------------------------- #
# Cluster bootstrap BCa CI (§2.4 step 3, §4.2).                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapCI:
    """A BCa bootstrap confidence interval for a scalar functional."""

    point: float
    lower: float
    upper: float
    level: float
    n_boot: int
    bias_correction: float
    acceleration: float


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    rr = q * q
    return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / \
           (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1.0)


def cluster_bootstrap_ci(
    clusters: Sequence[int],
    statistic: Callable[[Sequence[int]], float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """BCa 95% CI of a scalar functional by resampling LOCI **clusters** (§2.4).

    Resamples the *clusters* (not individual rows) with replacement to respect
    within-cluster dependence, recomputing ``statistic(resampled_cluster_ids)``
    each time. The BCa interval applies bias-correction ``z0`` (from the fraction
    of bootstrap replicates below the point estimate) and acceleration ``a``
    (from the jackknife-over-clusters skewness), the standard
    bias-corrected-and-accelerated percentile interval.

    Parameters
    ----------
    clusters:
        The unique cluster ids to resample (each appears once; the resampler
        draws ``len(clusters)`` ids with replacement).
    statistic:
        Callable mapping a multiset of cluster ids -> scalar (e.g. the cross-fit
        partial-R^2 restricted to rows in those clusters).
    n_boot, level, seed:
        Bootstrap count (default 2000), CI level (default 0.95), RNG seed.
    """

    unique = list(dict.fromkeys(clusters))
    g = len(unique)
    point = statistic(unique)
    if g < 2:
        return BootstrapCI(point, point, point, level, n_boot, 0.0, 0.0)

    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_boot):
        resampled = [unique[rng.randrange(g)] for _ in range(g)]
        replicates.append(statistic(resampled))
    replicates.sort()

    # Bias correction z0 from the fraction of replicates < point.
    below = sum(1 for r in replicates if r < point)
    frac = below / n_boot
    frac = min(max(frac, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = _normal_ppf(frac)

    # Acceleration a from the jackknife (leave-one-cluster-out) skewness.
    #
    # Small-g hardening: with very few clusters the jackknife sum-of-squares can
    # be near-zero (e.g. all LOO statistics nearly equal), so ``num / den`` -- a
    # ratio of a cubic to a 3/2-power of that tiny sum -- explodes, and a huge
    # ``accel`` makes the BCa map ``z0 + (z0+z)/(1 - accel*(z0+z))`` blow up or
    # flip sign, yielding a garbage interval. The jackknife skewness is also
    # simply not estimable with too few clusters. We therefore (a) require at
    # least ``MIN_CLUSTERS_FOR_ACCEL`` clusters and a non-degenerate jackknife
    # spread before trusting ``accel``, falling back to ``a = 0`` (the
    # bias-corrected, BC, interval) otherwise, and (b) clamp ``|accel|`` to a
    # finite cap so the BCa denominator stays well-conditioned.
    accel = 0.0
    jack: list[float] = []
    for k in range(g):
        loo = unique[:k] + unique[k + 1:]
        jack.append(statistic(loo))
    jack_mean = math.fsum(jack) / g
    ss = math.fsum((jack_mean - v) ** 2 for v in jack)
    num = math.fsum((jack_mean - v) ** 3 for v in jack)
    den = 6.0 * (ss ** 1.5)
    # Scale-aware floor: only trust the ratio when the jackknife spread is a
    # meaningful fraction of the point estimate's magnitude, not numerical dust.
    spread_floor = 1e-12 * (1.0 + abs(point)) ** 2
    if g >= MIN_CLUSTERS_FOR_ACCEL and ss > spread_floor and den > 0.0:
        accel = num / den
        if not math.isfinite(accel):
            accel = 0.0
        accel = max(-MAX_ABS_ACCEL, min(MAX_ABS_ACCEL, accel))

    alpha = 1.0 - level
    z_lo, z_hi = _normal_ppf(alpha / 2.0), _normal_ppf(1.0 - alpha / 2.0)

    def adjusted(z: float) -> float:
        denom = 1.0 - accel * (z0 + z)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return _normal_cdf(z0 + (z0 + z) / denom)

    p_lo = adjusted(z_lo)
    p_hi = adjusted(z_hi)

    def percentile(p: float) -> float:
        p = min(max(p, 0.0), 1.0)
        pos = p * (n_boot - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n_boot - 1)
        frac_ = pos - lo
        return replicates[lo] * (1.0 - frac_) + replicates[hi] * frac_

    return BootstrapCI(
        point=point,
        lower=percentile(p_lo),
        upper=percentile(p_hi),
        level=level,
        n_boot=n_boot,
        bias_correction=z0,
        acceleration=accel,
    )


# --------------------------------------------------------------------------- #
# Two-layer Holm + §3.3 2x2 contingency + §4.2 robustness floor (fix e, d, a). #
# --------------------------------------------------------------------------- #


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""

    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        candidate = (m - rank) * p
        running = max(running, candidate)
        adjusted[key] = min(running, 1.0)
    return adjusted


def two_layer_holm(
    trajectory_family: Mapping[str, float],
    metric_families: Mapping[str, float] | None = None,
    *,
    alpha: float = 0.05,
    gatekeeper: str = "joint",
) -> dict[str, object]:
    """Two-layer Holm with the **joint** block as gatekeeper (§4.5, fix e).

    Layer 1 -- within the trajectory family ``{joint, D-only, kappa-only}``: Holm
    over the 3 block tests. The ``gatekeeper`` (``joint``) must pass at ``alpha``;
    if it fails, ``D-only`` / ``kappa-only`` are flagged **exploratory only**.
    Layer 2 -- across metric families ``{target, retention, hallucination, layer,
    cost}``: Holm on the gatekeeper-conditioned p-values. The two corrections
    compose (within-family first, then across-family).

    Returns a dict with ``within_adjusted``, ``gatekeeper_passes``,
    ``across_adjusted`` (or ``None``), and ``exploratory`` flags.
    """

    within = holm_adjust(trajectory_family)
    gate_p = within.get(gatekeeper, 1.0)
    gate_passes = gate_p < alpha

    result: dict[str, object] = {
        "within_adjusted": within,
        "gatekeeper": gatekeeper,
        "gatekeeper_passes": gate_passes,
        "exploratory": {k: (not gate_passes) for k in within if k != gatekeeper},
    }
    if metric_families is not None and gate_passes:
        result["across_adjusted"] = holm_adjust(metric_families)
    else:
        result["across_adjusted"] = None
    return result


@dataclass(frozen=True)
class ContingencyDecision:
    """Outcome of the §3.3 2x2 decision table."""

    decision: str          # full | fallback | curvature-only | kill | gatekeeper-failed
    allowed_claim: str
    failure_action: str | None


def contingency_decision(
    *,
    gatekeeper_passes: bool,
    d_only_sig: bool,
    kappa_only_sig: bool,
) -> ContingencyDecision:
    """The locked §3.3 2x2 table as code (consulted only if joint passes).

    | D-only | kappa-only | decision        | claim                                   |
    | yes    | yes        | full            | distributional AND temporal-curvature   |
    | yes    | no         | fallback        | distributional (magnitude); curvature dropped |
    | no     | yes        | curvature-only  | temporal-curvature; magnitude demoted    |
    | no     | no         | kill            | no trajectory-beyond-endpoint claim      |
    """

    if not gatekeeper_passes:
        return ContingencyDecision(
            decision="gatekeeper-failed",
            allowed_claim="D-only/kappa-only are exploratory only; no primary trajectory claim",
            failure_action="stop_main_novelty_claim",
        )
    if d_only_sig and kappa_only_sig:
        return ContingencyDecision(
            "full",
            "distributional and temporal-curvature trajectory signal beyond endpoints",
            None,
        )
    if d_only_sig and not kappa_only_sig:
        return ContingencyDecision(
            "fallback",
            "distributional (magnitude) trajectory signal beyond endpoints; curvature novelty dropped",
            None,
        )
    if kappa_only_sig and not d_only_sig:
        return ContingencyDecision(
            "curvature-only",
            "temporal-curvature trajectory signal beyond endpoints; magnitude demoted",
            None,
        )
    return ContingencyDecision(
        "kill",
        "no trajectory-beyond-endpoint claim",
        "stop_main_novelty_claim",
    )


def robustness_floor(
    full_ridge_point: float,
    full_ridge_ci: tuple[float, float],
    pca_points: Mapping[int, float],
    pca_cis: Mapping[int, tuple[float, float]],
    *,
    floor: float = 0.02,
) -> dict[str, object]:
    """The §4.2 robustness floor (fix a): full-ridge survives AND PCA-stable.

    Requires (i) the full-ridge BCa CI lower bound exceeds ``floor``; (ii) no sign
    flip across ``r in {8,16,32}`` (all points share the full-ridge sign); (iii)
    monotone-stable / overlapping CIs (each PCA CI overlaps the full-ridge CI).
    If full-ridge passes but a PCA pole disagrees, the result is labelled
    **control-sensitive** rather than a clean co-primary pass.
    """

    full_lo, full_hi = full_ridge_ci
    full_passes = full_lo > floor
    sign = 1.0 if full_ridge_point >= 0 else -1.0
    no_sign_flip = all((sign * v) >= 0 for v in pca_points.values())

    def overlaps(ci: tuple[float, float]) -> bool:
        lo, hi = ci
        return not (hi < full_lo or lo > full_hi)

    all_overlap = all(overlaps(pca_cis[r]) for r in pca_cis)
    stable = no_sign_flip and all_overlap
    if full_passes and stable:
        tier = "co-primary-pass"
    elif full_passes and not stable:
        tier = "control-sensitive"
    else:
        tier = "fail"
    return {
        "tier": tier,
        "full_ridge_passes": full_passes,
        "no_sign_flip": no_sign_flip,
        "cis_overlap": all_overlap,
    }


# --------------------------------------------------------------------------- #
# Achieved-power / runtime calc for P=5000 perms x K=10 folds (§4.2, §4.5).   #
# --------------------------------------------------------------------------- #


def achieved_power(
    n: int,
    *,
    effect_partial_r2: float,
    n_permutations: int = 5000,
    n_folds: int = 10,
    alpha: float = 0.05,
    seconds_per_fold_fit: float = 0.0,
) -> dict[str, float]:
    """Achieved power + runtime estimate for the locked permutation design.

    Power model: the held-out partialled effect against the conditional null is
    approximated by a non-central signal-to-noise on the per-fold residual. With
    held-out sample ``n`` and partial-R^2 ``rho`` the per-test z-shift is
    ``delta = sqrt(n * rho / (1 - rho))``; a one-sided permutation test at
    ``alpha`` (granularity ``1/(P+1)``) has approximate power
    ``Phi(delta - z_{1-alpha})`` floored at the achievable permutation resolution.

    Runtime: ``(P + 1) * K`` cross-fit evaluations; if ``seconds_per_fold_fit`` is
    supplied (measured, not assumed), the wall-clock estimate is reported. The
    permutation reuses cached per-fold residuals, so only ``K`` ridge FITS occur
    (``(P+1)`` cheap residual re-aggregations), which is reflected in the
    ``ridge_fits`` vs ``stat_evaluations`` split.
    """

    rho = max(min(effect_partial_r2, 0.999999), 0.0)
    delta = math.sqrt(n * rho / (1.0 - rho)) if rho < 1.0 else math.inf
    z_alpha = _normal_ppf(1.0 - alpha)
    power = _normal_cdf(delta - z_alpha)
    perm_resolution = 1.0 / (n_permutations + 1)
    # The permutation p cannot be more powerful than its granularity allows.
    power = min(power, 1.0 - perm_resolution)

    stat_evaluations = (n_permutations + 1) * n_folds
    ridge_fits = n_folds  # residuals cached; permutations re-aggregate only
    wall_seconds = ridge_fits * seconds_per_fold_fit if seconds_per_fold_fit else 0.0
    return {
        "achieved_power": power,
        "z_shift": delta,
        "alpha": alpha,
        "min_resolvable_p": perm_resolution,
        "stat_evaluations": float(stat_evaluations),
        "ridge_fits": float(ridge_fits),
        "wall_seconds_estimate": wall_seconds,
    }

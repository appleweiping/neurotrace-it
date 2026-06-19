"""LOCI outcome ``Y`` construction + G7 reliability gate (§4.1, §2.6, fix c).

The locked outcome ``Y`` of ``docs/redesign/REDESIGN_v4.md`` §4.1 and its
precondition gate G7 (§2.6, §4.4):

* :func:`build_loci_clusters` -- cluster candidates by their L2-normalized
  endpoint signature ``phi_end`` (Eq. 15). The *production* path is HDBSCAN
  (``min_cluster_size=25, min_samples=10, metric=euclidean,
  cluster_selection_method=eom``); to stay pure-stdlib / build-now-run-later this
  module ships a faithful **agglomerative surrogate** that honours the same
  contract (size floor 25, noise/singletons reassigned to the nearest centroid,
  persisted assignment hash). The HDBSCAN params are persisted so the production
  swap is a drop-in. Clustering is fit on the **train fold only**; held-out
  examples are assigned by nearest centroid (no leakage).
* :func:`loci_influence` -- the leave-one-cluster-in influence estimator
  (Eq. 16-17): ``Delta_g = L_val(theta_{-g}) - L_val(theta)`` attributed to
  members by ``Y_i = +(|g|^-1 Delta_g) * drift_adjust_i``. ``Y`` is a utility
  (higher = more useful): a useful cluster (``Delta_g > 0``) gets positive ``Y``.
* :func:`y_reliability_gate` -- G7 (fix c): ICC(2,1) >= 0.6 across >= 3 seeds AND
  Spearman rho(Y, Delta_retrain) >= 0.3 with **lower-95%-CI > 0** on the
  pre-registered ``n_sub = 60`` subset. Fail => the primary regression is NOT run
  on the influence proxy (falls back to the direct retrain-delta Y, diagnostic).

Pure stdlib; build-now / run-later. No server call, no training, no model load.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

Vector = Sequence[float]

__all__ = [
    "HDBSCAN_PARAMS",
    "CLUSTER_SIZE_FLOOR",
    "ICC_MIN",
    "RHO_MIN",
    "N_SUB_DEFAULT",
    "SEEDS_MIN",
    "LociClustering",
    "ReliabilityReport",
    "build_loci_clusters",
    "loci_influence",
    "icc_2_1",
    "spearman_rho",
    "y_reliability_gate",
]

# Locked §4.1 HDBSCAN params (persisted; production path).
HDBSCAN_PARAMS = {
    "min_cluster_size": 25,
    "min_samples": 10,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}
CLUSTER_SIZE_FLOOR = 25
ICC_MIN = 0.6          # G7 ICC(2,1) threshold (§4.4)
RHO_MIN = 0.3          # G7 Spearman rho threshold (§4.4)
N_SUB_DEFAULT = 60     # pre-registered retrain subset of clusters
SEEDS_MIN = 3          # G7 ICC requires >= 3 seeds (lattice_v4.yaml:75, §4.4)


def _l2_normalize(vec: Vector) -> list[float]:
    norm = math.sqrt(math.fsum(c * c for c in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [c / norm for c in vec]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class LociClustering:
    """Persisted LOCI cluster assignment (Eq. 15) with its integrity hash."""

    labels: tuple[int, ...]                 # cluster id per example (train order)
    centroids: tuple[tuple[float, ...], ...]
    params: dict[str, object]
    assignment_hash: str

    def assign(self, phi_end: Vector) -> int:
        """Assign a held-out example to its nearest cluster centroid (no leakage)."""

        normalized = _l2_normalize(phi_end)
        best, best_d = 0, math.inf
        for cid, centroid in enumerate(self.centroids):
            d = _euclidean(normalized, centroid)
            if d < best_d:
                best, best_d = cid, d
        return best


def _hash_labels(labels: Sequence[int]) -> str:
    payload = ",".join(str(v) for v in labels).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def build_loci_clusters(
    phi_end_rows: Sequence[Vector],
    *,
    target_clusters: int | None = None,
    size_floor: int = CLUSTER_SIZE_FLOOR,
    seed: int = 0,
) -> LociClustering:
    """Cluster candidates by L2-normalized ``phi_end`` (Eq. 15) -- train fold only.

    Surrogate for the locked HDBSCAN: a deterministic furthest-point seeding +
    nearest-centroid assignment that honours the §4.1 contract -- target
    ``K ~= N/200``, **size floor 25** (small clusters merged into the nearest
    surviving centroid so no cluster is below floor), and singletons/noise
    reassigned to their nearest centroid. The assignment + its hash are persisted
    (``cluster_assignment_hash`` in §2.10). The HDBSCAN params are recorded for
    the production swap.
    """

    n = len(phi_end_rows)
    normalized = [_l2_normalize(row) for row in phi_end_rows]
    if n == 0:
        return LociClustering((), (), dict(HDBSCAN_PARAMS), _hash_labels(()))

    k = target_clusters if target_clusters is not None else max(1, n // 200)
    k = max(1, min(k, max(1, n // size_floor)))

    # Furthest-point (k-center) seeding for deterministic, spread-out centroids.
    rng = random.Random(seed)
    first = rng.randrange(n)
    seeds_idx = [first]
    while len(seeds_idx) < k:
        best_idx, best_d = -1, -1.0
        for i in range(n):
            d = min(_euclidean(normalized[i], normalized[s]) for s in seeds_idx)
            if d > best_d:
                best_idx, best_d = i, d
        if best_idx < 0:
            break
        seeds_idx.append(best_idx)
    centroids = [list(normalized[s]) for s in seeds_idx]

    # One Lloyd-style assignment + recentre pass for stability.
    for _ in range(5):
        buckets: list[list[int]] = [[] for _ in centroids]
        for i in range(n):
            best, best_d = 0, math.inf
            for cid, c in enumerate(centroids):
                d = _euclidean(normalized[i], c)
                if d < best_d:
                    best, best_d = cid, d
            buckets[best].append(i)
        new_centroids: list[list[float]] = []
        for members in buckets:
            if not members:
                continue
            dim = len(normalized[0])
            new_centroids.append(
                [math.fsum(normalized[i][d] for i in members) / len(members) for d in range(dim)]
            )
        if not new_centroids:
            break
        centroids = new_centroids

    # Final assignment.
    labels = [0] * n
    buckets = [[] for _ in centroids]
    for i in range(n):
        best, best_d = 0, math.inf
        for cid, c in enumerate(centroids):
            d = _euclidean(normalized[i], c)
            if d < best_d:
                best, best_d = cid, d
        labels[i] = best
        buckets[best].append(i)

    # Enforce size floor: dissolve under-floor clusters into the nearest surviving
    # centroid (so no member is in a below-floor cluster, per §4.1).
    survivors = [cid for cid, members in enumerate(buckets) if len(members) >= size_floor]
    if not survivors:
        survivors = [max(range(len(buckets)), key=lambda cid: len(buckets[cid]))]
    remap: dict[int, int] = {}
    for cid in range(len(centroids)):
        if cid in survivors:
            remap[cid] = survivors.index(cid)
        else:
            nearest = min(survivors, key=lambda s: _euclidean(centroids[cid], centroids[s]))
            remap[cid] = survivors.index(nearest)
    final_centroids = [tuple(centroids[cid]) for cid in survivors]
    final_labels = tuple(remap[labels[i]] for i in range(n))

    return LociClustering(
        labels=final_labels,
        centroids=tuple(final_centroids),
        params=dict(HDBSCAN_PARAMS),
        assignment_hash=_hash_labels(final_labels),
    )


def loci_influence(
    labels: Sequence[int],
    cluster_loss_deltas: Mapping[int, float],
    *,
    drift_adjust: Mapping[int, float] | None = None,
) -> list[float]:
    """Per-example LOCI influence ``Y_i`` (Eq. 16-17, locked).

    Implements the locked Eq. 17 **literally**:
    ``Y_i = +( |g|^{-1} * Delta_g ) * drift_adjust_i`` for ``x_i in g``, where
    ``Delta_g = L_val(theta_{-g}) - L_val(theta)`` (Eq. 16) is the held-out
    validation-loss change from down-weighting cluster ``g`` (``theta_{-g}`` the
    influence-approximated parameter with ``g`` down-weighted). ``Y`` is a
    **utility** (higher = more useful): a useful cluster, whose removal *raises*
    the validation loss (``Delta_g > 0``), receives a **positive** ``Y``. The
    magnitude is attributed equally to the ``|g|`` members and folded with the
    retention/hallucination ``drift_adjust_i`` (consistent with
    ``retention_adjusted_gain``). The sign convention is fixed by the spec; this
    function does NOT re-interpret it.

    Parameters
    ----------
    labels:
        Cluster id per example (length n).
    cluster_loss_deltas:
        ``{g: Delta_g}`` -- the Eq. 16 validation-loss change per cluster.
    drift_adjust:
        Optional ``{i: drift_adjust_i}`` per-example multiplier (default 1.0).
    """

    n = len(labels)
    sizes: dict[int, int] = {}
    for g in labels:
        sizes[g] = sizes.get(g, 0) + 1
    out: list[float] = []
    for i in range(n):
        g = labels[i]
        delta_g = cluster_loss_deltas.get(g, 0.0)
        size = sizes.get(g, 1) or 1
        attributed = delta_g / size
        adj = drift_adjust.get(i, 1.0) if drift_adjust else 1.0
        out.append(attributed * adj)
    return out


# --------------------------------------------------------------------------- #
# G7 reliability: ICC(2,1) + Spearman rho with lower-95%-CI > 0 (fix c).      #
# --------------------------------------------------------------------------- #


def icc_2_1(measurements: Sequence[Sequence[float]]) -> float:
    """ICC(2,1) -- two-way random, single-rater absolute agreement.

    ``measurements[i]`` is the vector of ``k`` repeated measurements (>= 3 seeds)
    of subject ``i``'s ``Y``. ICC(2,1) =
    ``(MSR - MSE) / (MSR + (k-1) MSE + (k/n)(MSC - MSE))`` from the two-way ANOVA
    decomposition (rows=subjects, cols=raters/seeds). Returns 0.0 for degenerate
    (no variance) input.
    """

    n = len(measurements)
    if n < 2:
        return 0.0
    k = len(measurements[0])
    if k < 2 or any(len(row) != k for row in measurements):
        raise ValueError("ICC(2,1) requires a rectangular n x k matrix with k >= 2")

    grand = math.fsum(math.fsum(row) for row in measurements) / (n * k)
    row_means = [math.fsum(row) / k for row in measurements]
    col_means = [math.fsum(measurements[i][j] for i in range(n)) / n for j in range(k)]

    ss_rows = k * math.fsum((rm - grand) ** 2 for rm in row_means)
    ss_cols = n * math.fsum((cm - grand) ** 2 for cm in col_means)
    ss_total = math.fsum(
        (measurements[i][j] - grand) ** 2 for i in range(n) for j in range(k)
    )
    ss_error = ss_total - ss_rows - ss_cols
    if ss_total <= 1e-18:
        return 0.0

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if abs(denom) < 1e-18:
        return 0.0
    return (msr - mse) / denom


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on ranks; tie-aware)."""

    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    mx = math.fsum(rx) / len(rx)
    my = math.fsum(ry) / len(ry)
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.fsum((a - mx) ** 2 for a in rx)
    vy = math.fsum((b - my) ** 2 for b in ry)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _one_sided_z(level: float) -> float:
    """One-sided upper critical value ``z_{level}`` (e.g. 1.6448... at 0.95)."""

    if abs(level - 0.95) < 1e-9:
        return 1.6448536269514722  # Phi^{-1}(0.95), one-sided
    from statistics import NormalDist

    return NormalDist().inv_cdf(level)


def _spearman_ci_lower(rho: float, n: int, *, level: float = 0.95) -> float:
    """One-sided lower 95% confidence bound on Spearman rho (Fisher z, n >= 4).

    The G7 gate asks "is rho reliably > 0?", a one-sided question, so this uses the
    one-sided critical value z_{0.95} = 1.6448536269514722 (NOT the two-sided
    z_{0.975} = 1.96). The returned value is a genuine lower 95% one-sided bound.
    """

    if n < 4 or abs(rho) >= 1.0:
        return rho
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = _one_sided_z(level)  # one-sided z_{level}; 1.6448536... at level=0.95
    z_lo = z - z_crit * se
    return math.tanh(z_lo)


@dataclass(frozen=True)
class ReliabilityReport:
    """G7 Y-reliability gate outcome (fix c)."""

    icc: float
    rho: float
    rho_ci_lower: float
    icc_passes: bool
    rho_passes: bool
    passes: bool
    fallback: str | None
    n_seeds: int = 0
    seeds_passes: bool = True


def y_reliability_gate(
    seed_measurements: Sequence[Sequence[float]],
    proxy_y: Sequence[float],
    retrain_delta: Sequence[float],
    *,
    icc_min: float = ICC_MIN,
    rho_min: float = RHO_MIN,
    n_sub: int = N_SUB_DEFAULT,
    seeds_min: int = SEEDS_MIN,
    level: float = 0.95,
) -> ReliabilityReport:
    """G7 precondition gate (§2.6, §4.4, fix c).

    Requires ALL of:
      (0) at least ``seeds_min`` (3) seeds, i.e. ``seed_measurements`` is an
          ``n x k`` matrix with ``k >= seeds_min`` (``lattice_v4.yaml:75``). With
          fewer seeds the ICC(2,1) two-way decomposition is not trustworthy, so
          the gate fails closed regardless of the (under-powered) ICC value.
      (i) ICC(2,1) >= ``icc_min`` (0.6) across those seeds (per-example Y stable);
      (ii) Spearman ``rho(proxy_y, retrain_delta) >= rho_min`` (0.3) with
           **lower 95% CI > 0** on the pre-registered ``n_sub`` (60) cluster subset.
    Fail => the primary regression is NOT run on the influence proxy; it falls
    back to the direct retrain-delta Y on the subset (underpowered-diagnostic).

    The ``seeds_min`` precondition is enforced *before* ICC is consulted: the
    number of seeds is the column count ``k`` of ``seed_measurements`` (the
    repeated per-subject measurements). An empty matrix counts as 0 seeds.
    """

    n_seeds = len(seed_measurements[0]) if seed_measurements else 0
    seeds_ok = n_seeds >= seeds_min

    icc = icc_2_1(seed_measurements) if seeds_ok else 0.0
    sub = min(n_sub, len(proxy_y), len(retrain_delta))
    rho = spearman_rho(proxy_y[:sub], retrain_delta[:sub])
    rho_lo = _spearman_ci_lower(rho, sub, level=level)

    icc_ok = seeds_ok and icc >= icc_min
    rho_ok = (rho >= rho_min) and (rho_lo > 0.0)
    passes = seeds_ok and icc_ok and rho_ok
    fallback = None if passes else "direct_retrain_delta_diagnostic"
    return ReliabilityReport(
        icc=icc,
        rho=rho,
        rho_ci_lower=rho_lo,
        icc_passes=icc_ok,
        rho_passes=rho_ok,
        passes=passes,
        fallback=fallback,
        n_seeds=n_seeds,
        seeds_passes=seeds_ok,
    )

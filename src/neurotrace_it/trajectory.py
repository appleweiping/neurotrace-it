"""Trajectory operator over (layer x step) activations.

Implements the layer-resolved trajectory signature ``T(x)`` of
``docs/redesign/REDESIGN_v4.md`` §2.2, plus the auxiliary distributional
signatures used by the matched-pair and selection machinery:

* per-layer **distributional magnitude** ``D_l`` via sliced-Wasserstein-2
  (``SW2^2``) to a held-out target cloud, with **fixed, persisted projection
  seeds** (Eq. 3-4);
* per-layer **temporal curvature** ``kappa_l``, the normalized bending energy of
  the per-step mean path (Eq. 5) -- a statistic that *uses step ordering* and is
  provably NOT a function of the first-moment pool (§2.5);
* the cross-layer signature ``T(x) = ({D_l}, {kappa_l}) in R^{2|A|}`` (Eq. 6);
* a sliced-Wasserstein-2 / MMD **distributional signature** with fixed projection
  seeds, for the matched-pair diagnostic and coverage similarity.

Design notes
------------
* **Pure stdlib.** No numpy/scipy/torch. Activations are nested Python sequences
  ``h_{l,s,t} in R^d``. This keeps the operator import-safe and faithful to the
  existing dependency-free codebase; it is build-now / run-later and performs no
  model load and no server call.
* **Deterministic projections.** SW2 uses ``K`` random unit directions drawn from
  a seeded :class:`random.Random`; the seed list is returned alongside the value
  so it can be persisted in ``TrajectorySignatureV2`` (§2.10) and recomputed.
* **First-moment insufficiency, not equivalence (§1.3).** We never claim mean-pool
  equals endpoint; ``kappa`` and ``D`` are designed to respond to temporal /
  higher-moment structure that the first-moment pool discards.

DO-NOT-RUN: no experiment, no extraction, no training is performed.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

# h_{l,s,t} in R^d.  An example's activations are indexed [layer][step][token].
Vector = Sequence[float]
StepCloud = Sequence[Vector]          # tokens at one (layer, step)
LayerSteps = Sequence[StepCloud]      # steps for one layer
# Per-example activations: {layer_id: [ [tok0, tok1, ...]_step0, ..._step1, ... ]}
LayerStepActivations = Mapping[int, LayerSteps]
# Held-out target cloud per layer: {layer_id: [h0, h1, ...]} (an empirical sample).
TargetClouds = Mapping[int, Sequence[Vector]]

__all__ = [
    "DEFAULT_SW2_PROJECTIONS",
    "DEFAULT_SUBSAMPLE_CAP",
    "CURVATURE_EPS",
    "LayerMagnitude",
    "TrajectoryFeatures",
    "sliced_wasserstein2",
    "layer_magnitude",
    "trajectory_curvature",
    "trajectory_signature",
    "sw2_signature",
    "rbf_mmd2",
]

# Locked defaults (REDESIGN_v4 §1.4). Overridable for the resolution sweep (§2.7).
DEFAULT_SW2_PROJECTIONS = 64           # K random projections
DEFAULT_SUBSAMPLE_CAP = 512            # (step, token) vectors per layer
CURVATURE_EPS = 1e-8                   # epsilon in Eq. 5 denominator


def _stable_seed(*parts: object) -> int:
    """Derive a stable, non-negative ``int`` seed from arbitrary components.

    ``random.Random`` only accepts ``None | int | float | str | bytes |
    bytearray`` (Python 3.11+ rejects tuples with a ``TypeError``). Several
    call-sites want to derive an *independent but reproducible* stream from a
    base integer ``seed`` plus a string tag (e.g. ``"P"`` / ``"Q"``). We hash the
    components into a single deterministic 63-bit integer so the stream is stable
    across processes (``hash()`` is salted for ``str``/``bytes`` and would NOT
    be reproducible across runs, so it must not be used here).
    """

    payload = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _as_floats(vector: Vector) -> list[float]:
    return [float(component) for component in vector]


def _check_dim(vectors: Sequence[Sequence[float]]) -> int:
    dim: int | None = None
    for vector in vectors:
        if dim is None:
            dim = len(vector)
        elif len(vector) != dim:
            raise ValueError(f"inconsistent vector dim {len(vector)} != {dim}")
    if dim is None or dim == 0:
        raise ValueError("empty or zero-dimensional cloud")
    return dim


def _subsample(
    cloud: Sequence[Sequence[float]],
    cap: int,
    rng: random.Random,
) -> list[list[float]]:
    """Persisted-mask subsample to at most ``cap`` vectors (deterministic).

    Uses :meth:`random.Random.sample` over indices so the retained mask is a
    function of ``rng`` state only -- the caller persists the seed, so the mask is
    reproducible (the §2.10 ``slice_masks`` field).
    """

    materialized = [list(_as_floats(vector)) for vector in cloud]
    if len(materialized) <= cap:
        return materialized
    keep_indices = sorted(rng.sample(range(len(materialized)), cap))
    return [materialized[index] for index in keep_indices]


def _unit_direction(dim: int, rng: random.Random) -> list[float]:
    """A uniformly random unit vector on S^{d-1} via normalized Gaussian."""

    while True:
        direction = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(math.fsum(component * component for component in direction))
        if norm > 1e-12:
            return [component / norm for component in direction]
        # Degenerate all-zero draw (astronomically unlikely): resample.


def _w2_1d_squared(proj_p: list[float], proj_q: list[float]) -> float:
    """Squared 1-D Wasserstein-2 between two empirical samples.

    The 1-D ``W2^2`` is the L2 distance between the (sorted) quantile functions.
    For samples of unequal size we resample both onto a common grid of
    ``lcm``-free common length via the standard sorted-quantile interpolation at
    ``n = max(len_p, len_q)`` equally spaced quantile levels. This is the exact
    sort-and-integrate estimator referenced in Eq. 3.
    """

    sorted_p = sorted(proj_p)
    sorted_q = sorted(proj_q)
    n_p = len(sorted_p)
    n_q = len(sorted_q)
    if n_p == 0 or n_q == 0:
        raise ValueError("cannot compute W2 against an empty projected sample")
    if n_p == n_q:
        return math.fsum((a - b) ** 2 for a, b in zip(sorted_p, sorted_q)) / n_p
    # Unequal sizes: integrate squared quantile gap over a shared quantile grid.
    grid = max(n_p, n_q)

    def quantile(sorted_values: list[float], level: float) -> float:
        # level in [0, 1]; linear interpolation between order statistics.
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = level * (len(sorted_values) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(sorted_values) - 1)
        frac = position - low
        return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac

    total = 0.0
    for index in range(grid):
        level = (index + 0.5) / grid
        gap = quantile(sorted_p, level) - quantile(sorted_q, level)
        total += gap * gap
    return total / grid


def sliced_wasserstein2(
    cloud_p: Sequence[Vector],
    cloud_q: Sequence[Vector],
    *,
    n_projections: int = DEFAULT_SW2_PROJECTIONS,
    seed: int = 0,
    subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
) -> tuple[float, tuple[int, ...]]:
    """Estimate ``SW2^2(P, Q)`` and return ``(value, projection_seeds)`` (Eq. 3).

    ``SW2^2(P,Q) = E_{u~Unif(S^{d-1})}[ W2^2(u#P, u#Q) ]`` estimated with
    ``n_projections`` random unit directions; each 1-D ``W2^2`` is a
    sort-and-integrate, ``O((n_P + n_Q) log n)``.

    The per-projection integer seeds are derived deterministically from ``seed``
    and returned so they can be **persisted** (``projection_seeds`` in §2.10) and
    the value recomputed bit-for-bit.

    Returns
    -------
    (value, projection_seeds)
        ``value`` is the mean over projections of the 1-D ``W2^2``.
        ``projection_seeds`` is the tuple of per-projection seeds actually used.
    """

    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    # Subsample both clouds with independent but seed-derived RNGs (persisted).
    rng_p = random.Random(_stable_seed(seed, "P"))
    rng_q = random.Random(_stable_seed(seed, "Q"))
    sample_p = _subsample(cloud_p, subsample_cap, rng_p)
    sample_q = _subsample(cloud_q, subsample_cap, rng_q)
    dim_p = _check_dim(sample_p)
    dim_q = _check_dim(sample_q)
    if dim_p != dim_q:
        raise ValueError(f"cloud dim mismatch {dim_p} != {dim_q}")

    projection_seeds: list[int] = []
    accumulated = 0.0
    for index in range(n_projections):
        proj_seed = (seed * 1_000_003 + index) & 0x7FFF_FFFF
        projection_seeds.append(proj_seed)
        direction = _unit_direction(dim_p, random.Random(proj_seed))
        proj_p = [math.fsum(d * x for d, x in zip(direction, point)) for point in sample_p]
        proj_q = [math.fsum(d * x for d, x in zip(direction, point)) for point in sample_q]
        accumulated += _w2_1d_squared(proj_p, proj_q)
    return accumulated / n_projections, tuple(projection_seeds)


def rbf_mmd2(
    cloud_p: Sequence[Vector],
    cloud_q: Sequence[Vector],
    *,
    bandwidth: float | None = None,
    subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
    seed: int = 0,
) -> float:
    """Biased (V-statistic) RBF-kernel ``MMD^2`` between two clouds.

    Provided alongside SW2 as an alternative distributional distance for the
    matched-pair diagnostic / robustness poles. ``bandwidth`` defaults to the
    median-heuristic over the pooled pairwise squared distances.

    We use the **biased V-statistic** (all three kernel means include their full
    matrices, diagonals retained) rather than the unbiased U-statistic so that
    ``MMD^2(P, P) = 0`` holds **exactly** on identical samples -- the defining
    property of an MMD. The unbiased estimator drops within-sample diagonals but
    not the cross diagonal, so it returns a small negative value on identical
    finite clouds (a finite-sample artifact, not a real discrepancy). The biased
    form is non-negative and is the standard choice when the identity property is
    required; the O(1/n) bias is immaterial for the auxiliary use here.
    """

    rng_p = random.Random(_stable_seed(seed, "mmd-P"))
    rng_q = random.Random(_stable_seed(seed, "mmd-Q"))
    sample_p = _subsample(cloud_p, subsample_cap, rng_p)
    sample_q = _subsample(cloud_q, subsample_cap, rng_q)
    dim_p = _check_dim(sample_p)
    dim_q = _check_dim(sample_q)
    # Reject dimension-mismatched clouds: ``sq_dist`` uses ``zip``, which would
    # otherwise SILENTLY truncate to the shorter vector and return a meaningless
    # MMD across point clouds living in different spaces.
    if dim_p != dim_q:
        raise ValueError(f"cloud dim mismatch {dim_p} != {dim_q}")

    def sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
        return math.fsum((x - y) ** 2 for x, y in zip(a, b))

    if bandwidth is None:
        pooled = sample_p + sample_q
        dists = [
            sq_dist(pooled[i], pooled[j])
            for i in range(len(pooled))
            for j in range(i + 1, len(pooled))
        ]
        median = sorted(dists)[len(dists) // 2] if dists else 1.0
        bandwidth = math.sqrt(median / 2.0) if median > 0 else 1.0
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel_mean(left: list[list[float]], right: list[list[float]], *, drop_diag: bool) -> float:
        total = 0.0
        count = 0
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if drop_diag and left is right and i == j:
                    continue
                total += math.exp(-gamma * sq_dist(a, b))
                count += 1
        return total / count if count else 0.0

    # Biased V-statistic: retain all diagonals so MMD^2(P,P)=0 exactly.
    k_pp = kernel_mean(sample_p, sample_p, drop_diag=False)
    k_qq = kernel_mean(sample_q, sample_q, drop_diag=False)
    k_pq = kernel_mean(sample_p, sample_q, drop_diag=False)
    return k_pp + k_qq - 2.0 * k_pq


@dataclass(frozen=True)
class LayerMagnitude:
    """Per-layer SW2 magnitude term ``D_l`` with its persisted projection seeds."""

    layer_id: int
    value: float
    projection_seeds: tuple[int, ...]


def layer_magnitude(
    layer_cloud: Sequence[Vector],
    target_cloud: Sequence[Vector],
    *,
    layer_id: int,
    n_projections: int = DEFAULT_SW2_PROJECTIONS,
    seed: int = 0,
    subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
) -> LayerMagnitude:
    """Compute ``D_l(x) = SW2^2(P_l(x), Q_l^T)`` (Eq. 4) for one layer."""

    value, seeds = sliced_wasserstein2(
        layer_cloud,
        target_cloud,
        n_projections=n_projections,
        seed=seed,
        subsample_cap=subsample_cap,
    )
    return LayerMagnitude(layer_id=layer_id, value=value, projection_seeds=seeds)


def trajectory_curvature(per_step_path: Sequence[Vector], *, eps: float = CURVATURE_EPS) -> float:
    """Per-layer temporal curvature ``kappa_l`` -- normalized bending energy (Eq. 5).

    Given the per-step mean path ``g_{l,s} = mean_t h_{l,s,t}`` (one vector per
    decoding step, in temporal order), with velocity ``v_s = g_{s+1} - g_s`` and
    acceleration ``a_s = g_{s+1} - 2 g_s + g_{s-1}``::

        kappa_l = (1/(S-2)) * sum_{s=2}^{S-1} ||a_s|| / (||v_s|| * ||v_{s-1}|| + eps)

    This statistic depends on the *ordered* second differences and therefore
    changes under step reordering whenever the path is non-collinear -- so it is
    provably NOT a function of the (permutation-invariant) first-moment pool
    (§2.5). Returns ``0.0`` for paths shorter than 3 steps (curvature undefined).
    """

    path = [_as_floats(step) for step in per_step_path]
    n_steps = len(path)
    if n_steps < 3:
        return 0.0
    dim = _check_dim(path)

    def diff(a: list[float], b: list[float]) -> list[float]:
        return [ai - bi for ai, bi in zip(a, b)]

    def norm(vector: list[float]) -> float:
        return math.sqrt(math.fsum(c * c for c in vector))

    # Indices follow Eq. 5: s runs 2..S-1 (1-based) => 1..n_steps-2 (0-based interior).
    contributions: list[float] = []
    for s in range(1, n_steps - 1):
        velocity = diff(path[s + 1], path[s])
        prev_velocity = diff(path[s], path[s - 1])
        acceleration = [
            path[s + 1][i] - 2.0 * path[s][i] + path[s - 1][i] for i in range(dim)
        ]
        denom = norm(velocity) * norm(prev_velocity) + eps
        contributions.append(norm(acceleration) / denom)
    return math.fsum(contributions) / len(contributions)


def _per_step_mean_path(layer_steps: LayerSteps) -> list[list[float]]:
    """g_{l,s} = mean_t h_{l,s,t} for each step s, in temporal order."""

    path: list[list[float]] = []
    for step_cloud in layer_steps:
        tokens = [_as_floats(token) for token in step_cloud]
        if not tokens:
            raise ValueError("empty step cloud: a decoding step has no tokens")
        dim = _check_dim(tokens)
        mean = [math.fsum(token[i] for token in tokens) / len(tokens) for i in range(dim)]
        path.append(mean)
    return path


@dataclass(frozen=True)
class TrajectoryFeatures:
    """The cross-layer trajectory signature ``T(x) = ({D_l}, {kappa_l})`` (Eq. 6).

    ``magnitude`` and ``curvature`` are keyed by layer id over the anchor set ``A``;
    ``projection_seeds`` records the SW2 seeds per layer for persistence (§2.10).
    ``feature_vector`` flattens to ``R^{2|A|}`` in ``[D_l for l in A] + [kappa_l for
    l in A]`` order (the design-matrix column order used by the residual test).
    """

    example_id: str
    layer_ids: tuple[int, ...]
    magnitude: dict[int, float]
    curvature: dict[int, float]
    projection_seeds: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def feature_vector(self) -> tuple[float, ...]:
        magnitudes = [self.magnitude[layer] for layer in self.layer_ids]
        curvatures = [self.curvature[layer] for layer in self.layer_ids]
        return tuple(magnitudes + curvatures)


def trajectory_signature(
    example_id: str,
    activations: LayerStepActivations,
    target_clouds: TargetClouds,
    *,
    layer_ids: Sequence[int] | None = None,
    n_projections: int = DEFAULT_SW2_PROJECTIONS,
    seed: int = 0,
    subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
    curvature_eps: float = CURVATURE_EPS,
) -> TrajectoryFeatures:
    """Compute the full trajectory signature ``T(x)`` for one example (Eq. 4-6).

    Parameters
    ----------
    example_id:
        Candidate example id.
    activations:
        ``{layer_id: [[tok...]_step0, [tok...]_step1, ...]}`` -- the per-(layer,
        step) token clouds for ``x``. Step order is the decoding order (used by
        curvature).
    target_clouds:
        ``{layer_id: [h0, h1, ...]}`` -- the held-out target-capability cloud
        ``Q_l^T`` per layer for the SW2 magnitude term.
    layer_ids:
        Anchor set ``A`` (fixed grid). Defaults to the sorted intersection of the
        activation and target-cloud layer keys.
    n_projections, seed, subsample_cap, curvature_eps:
        Locked operator knobs (§1.4); ``seed`` is offset per layer so each layer's
        SW2 draw is independent yet reproducible.

    Returns
    -------
    TrajectoryFeatures
        ``T(x) = ({D_l}, {kappa_l})`` with persisted projection seeds.
    """

    if layer_ids is None:
        ordered_layers = tuple(sorted(set(activations) & set(target_clouds)))
    else:
        ordered_layers = tuple(layer_ids)
    if not ordered_layers:
        raise ValueError("trajectory_signature requires a non-empty anchor set A")

    magnitude: dict[int, float] = {}
    curvature: dict[int, float] = {}
    seeds: dict[int, tuple[int, ...]] = {}

    for offset, layer in enumerate(ordered_layers):
        if layer not in activations:
            raise KeyError(f"activations missing layer {layer}")
        if layer not in target_clouds:
            raise KeyError(f"target_clouds missing layer {layer}")
        layer_steps = activations[layer]
        # D_l: flatten all (step, token) vectors into the per-layer cloud P_l(x).
        flat_cloud = [token for step_cloud in layer_steps for token in step_cloud]
        layer_seed = seed * 7919 + offset
        magnitude_term = layer_magnitude(
            flat_cloud,
            target_clouds[layer],
            layer_id=layer,
            n_projections=n_projections,
            seed=layer_seed,
            subsample_cap=subsample_cap,
        )
        magnitude[layer] = magnitude_term.value
        seeds[layer] = magnitude_term.projection_seeds
        # kappa_l: curvature of the per-step mean path.
        path = _per_step_mean_path(layer_steps)
        curvature[layer] = trajectory_curvature(path, eps=curvature_eps)

    return TrajectoryFeatures(
        example_id=example_id,
        layer_ids=ordered_layers,
        magnitude=magnitude,
        curvature=curvature,
        projection_seeds=seeds,
    )


def sw2_signature(
    activations: LayerStepActivations,
    target_clouds: TargetClouds,
    *,
    layer_ids: Sequence[int] | None = None,
    n_projections: int = DEFAULT_SW2_PROJECTIONS,
    seed: int = 0,
    subsample_cap: int = DEFAULT_SUBSAMPLE_CAP,
) -> dict[int, LayerMagnitude]:
    """Per-layer SW2 magnitude signature ``{l: D_l}`` with fixed projection seeds.

    A thin convenience wrapper returning the full :class:`LayerMagnitude` records
    (value + persisted seeds) for every anchor layer, for callers that want the
    distributional signature without curvature.
    """

    if layer_ids is None:
        ordered_layers = tuple(sorted(set(activations) & set(target_clouds)))
    else:
        ordered_layers = tuple(layer_ids)
    out: dict[int, LayerMagnitude] = {}
    for offset, layer in enumerate(ordered_layers):
        flat_cloud = [token for step_cloud in activations[layer] for token in step_cloud]
        out[layer] = layer_magnitude(
            flat_cloud,
            target_clouds[layer],
            layer_id=layer,
            n_projections=n_projections,
            seed=seed * 7919 + offset,
            subsample_cap=subsample_cap,
        )
    return out

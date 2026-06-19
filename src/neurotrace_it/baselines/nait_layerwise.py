"""Faithful layerwise NAIT reproduction over the FULL released layer set ``L``.

REDESIGN_v5 §3.3 (MF-1, B6), DO-NOT-RUN. This is the *decisive* NAIT comparator:
the layerwise score summed over the full released decoder layer set ``L`` (base
Eq. 5), as opposed to the endpoint-only control ``phi_end`` in
:mod:`neurotrace_it.baselines.nait` (which is the FULL endpoint *control*, a
distinct object that must NOT coincide with this layerwise sum).

Base-paper equations reproduced (faithful transcription, verified against the
v5 audit):

* **Eq. 2 (Alg. 1, per-layer difference).**
  ``Delta A^(l)(P_i) = A^(l)(t_K) - A^(l)(t_1)`` for every ``l in L``.
  A pre-registered named alternate (prose) replaces the first/last difference by
  the mean-over-K-tokens activation summary; both variants are computed and the
  STRONGER is the comparator.
* **Eq. 3 (per-layer PCA direction).** ``v_l = PCA_1({Delta A^(l)(P_i)}_i)``.
* **Eq. 4 (sign-align).** ``v_l <- -v_l if mu_diff . v_l < 0``,
  ``mu_diff = |P|^{-1} sum_i Delta A^(l)(P_i)``.
* **Eq. 5 (scoring, over L).** ``s_NAIT(y) = sum_{l=1}^{L} (A^(l)(y) . v_l)``;
  per-layer projections ``proj_l(y) = A^(l)(y) . v_l`` are persisted (they feed
  ``V_proj`` in R0).
* **Eq. 6 (selection).** ``top-k(s_NAIT)`` at budget ``B``.

A **gated** secondary 8-anchor variant ``s_NAIT^{A}(y) = sum_{l in A} ...`` is
computed only as a secondary diagnostic; "beats NAIT" wording is ALWAYS gated on
the full-``L`` variant.

DO-NOT-RUN: pure stdlib; no model load, no extraction, no server call. The caller
supplies already-extracted activation tensors so the score arithmetic is testable
build-now / run-later. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "LayerwiseNaitModel",
    "NaitLayerwiseScores",
    "principal_direction",
    "sign_align",
    "layer_difference",
    "token_mean_summary",
    "fit_layer_directions",
    "score_layerwise",
    "select_layerwise",
]

Vector = Sequence[float]
# Per-example, per-layer activation summary {layer_id: vector in R^d}.
LayerActivations = Mapping[int, Vector]
# A token trajectory for one layer: ordered list of per-token vectors.
TokenTrajectory = Sequence[Vector]


def _as_floats(vector: Vector) -> list[float]:
    return [float(component) for component in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def _l2(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(c * c for c in vector))


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return [x - y for x, y in zip(a, b)]


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty set of vectors")
    d = len(vectors[0])
    return [math.fsum(v[j] for v in vectors) / len(vectors) for j in range(d)]


def layer_difference(trajectory: TokenTrajectory) -> list[float]:
    """Eq. 2 (Alg. 1 first/last): ``A^(l)(t_K) - A^(l)(t_1)`` for one layer.

    ``trajectory`` is the ordered per-token activation list for one layer; the
    difference is the last-token minus first-token vector.
    """

    if len(trajectory) < 2:
        raise ValueError("layer_difference needs at least two token positions (t_1, t_K)")
    return _sub(_as_floats(trajectory[-1]), _as_floats(trajectory[0]))


def token_mean_summary(trajectory: TokenTrajectory) -> list[float]:
    """Prose token-mean variant: mean-over-K-tokens activation summary for one layer."""

    return _mean([_as_floats(v) for v in trajectory])


def principal_direction(
    samples: Sequence[Sequence[float]],
    *,
    iterations: int = 256,
    tol: float = 1e-12,
) -> list[float]:
    """First principal component ``PCA_1`` of mean-centered ``samples`` (Eq. 3).

    Pure-stdlib power iteration on the sample covariance (no numpy). Returns a
    unit-norm direction; deterministic (fixed all-ones start, renormalized).
    Centering is applied so this is PCA, not a raw second-moment direction.
    """

    if not samples:
        raise ValueError("principal_direction needs at least one sample")
    d = len(samples[0])
    centered = [_sub(_as_floats(s), _mean([_as_floats(x) for x in samples])) for s in samples]

    # Power iteration: v <- normalize( sum_i (x_i . v) x_i ).
    v = [1.0 / math.sqrt(d)] * d
    for _ in range(iterations):
        acc = [0.0] * d
        for x in centered:
            coeff = _dot(x, v)
            for j in range(d):
                acc[j] += coeff * x[j]
        norm = _l2(acc)
        if norm <= tol:
            # Degenerate (all samples identical after centering): return a stable
            # canonical unit vector so downstream sign-align/scoring is defined.
            out = [0.0] * d
            out[0] = 1.0
            return out
        new_v = [a / norm for a in acc]
        delta = _l2(_sub(new_v, v))
        v = new_v
        if delta <= tol:
            break
    return v


def sign_align(direction: Sequence[float], mu_diff: Sequence[float]) -> list[float]:
    """Eq. 4 sign alignment: flip ``v`` if ``mu_diff . v < 0``."""

    v = _as_floats(direction)
    if _dot(mu_diff, v) < 0.0:
        return [-c for c in v]
    return v


@dataclass(frozen=True)
class LayerwiseNaitModel:
    """Fitted per-layer NAIT directions over a layer set (Eq. 3-4).

    ``directions`` maps each layer id to its sign-aligned unit PCA direction
    ``v_l``. ``layer_set`` records the layers (the FULL ``L`` for the primary
    comparator; the 8 anchors for the gated secondary). ``variant`` is
    ``"alg1"`` (first/last diff) or ``"token_mean"``.
    """

    directions: Mapping[int, tuple[float, ...]]
    layer_set: tuple[int, ...]
    variant: str


def fit_layer_directions(
    differences_per_layer: Mapping[int, Sequence[Sequence[float]]],
    *,
    layer_set: Sequence[int],
    variant: str = "alg1",
) -> LayerwiseNaitModel:
    """Fit ``v_l`` for every layer in ``layer_set`` (Eq. 3 PCA + Eq. 4 sign-align).

    ``differences_per_layer[l]`` is the set ``{Delta A^(l)(P_i)}_i`` (already the
    per-layer differences from :func:`layer_difference`, or token-mean summaries
    for the alternate variant). The sign is aligned to the mean difference
    ``mu_diff`` (Eq. 4).
    """

    directions: dict[int, tuple[float, ...]] = {}
    for layer in layer_set:
        if layer not in differences_per_layer:
            raise KeyError(f"missing per-layer differences for layer {layer}")
        samples = [_as_floats(s) for s in differences_per_layer[layer]]
        v = principal_direction(samples)
        mu = _mean(samples)
        v = sign_align(v, mu)
        directions[layer] = tuple(v)
    return LayerwiseNaitModel(
        directions=directions,
        layer_set=tuple(layer_set),
        variant=variant,
    )


@dataclass(frozen=True)
class NaitLayerwiseScores:
    """Layerwise NAIT score for one candidate (Eq. 5) over a layer set.

    ``s_nait`` is ``sum_l (A^(l)(y) . v_l)``; ``proj`` persists the per-layer
    projections ``proj_l = A^(l)(y) . v_l`` (they feed ``V_proj`` in R0).
    """

    example_id: str
    s_nait: float
    proj: Mapping[int, float]
    layer_set: tuple[int, ...]


def score_layerwise(
    example_id: str,
    activations: LayerActivations,
    model: LayerwiseNaitModel,
) -> NaitLayerwiseScores:
    """Eq. 5 layerwise score ``s_NAIT(y) = sum_{l in layer_set} A^(l)(y) . v_l``.

    Summed over ``model.layer_set`` -- the FULL ``L`` for the primary comparator,
    or the 8 anchors for the gated secondary variant. Per-layer projections are
    persisted for R0's ``V_proj``.
    """

    proj: dict[int, float] = {}
    for layer in model.layer_set:
        if layer not in activations:
            raise KeyError(f"missing activation for layer {layer} in example {example_id!r}")
        v = model.directions[layer]
        a = _as_floats(activations[layer])
        proj[layer] = _dot(a, v)
    s = math.fsum(proj.values())
    return NaitLayerwiseScores(
        example_id=example_id,
        s_nait=s,
        proj=proj,
        layer_set=model.layer_set,
    )


def select_layerwise(
    scores: Sequence[NaitLayerwiseScores],
    *,
    budget: int,
) -> tuple[str, ...]:
    """Eq. 6 selection: budget-``B`` top-k by ``s_NAIT`` (ties ascending id)."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    ordered = sorted(scores, key=lambda s: (-s.s_nait, s.example_id))
    kept = min(budget, len(ordered))
    return tuple(s.example_id for s in ordered[:kept])

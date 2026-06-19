"""NAIT-style endpoint-neuron baseline (Deliverable #1).

Faithful reimplementation of Neuron-Aware Data Selection (NAIT) as described by
Eq. 1 of ``docs/redesign/REDESIGN_v4.md``. Per the corrected v4 citation-hygiene
rule (§1.4, fix f3), the registry arXiv id for this baseline -- arXiv 2603.13201
(NAIT) -- has been **verified as a real, checkable paper** (OpenReview, confirmed
2026-06-19); the earlier "future-dated => invalid" flag was a false positive and
is withdrawn. This module is a *faithful endpoint reimplementation of NAIT's
Eq. 1*; NAIT is a must-beat baseline, and any "beats NAIT" claim is gated on the
fair-comparison gates (§4.6), NOT on citation validity.

What NAIT does (the start/end activation signature + selection score)
---------------------------------------------------------------------
For a transformer with ``L`` layers and hidden width ``d`` and an anchor-layer
set ``A`` (``|A| = 8`` in the locked config), NAIT compresses an example ``x`` to
an *endpoint feature* read at only the START and END token positions::

    phi_end(x) = concat_{l in A} [ h_l(p_start) , h_l(p_end) ]   in R^{2 d |A|}   (Eq. 1)

and scores ``x`` for selection by similarity of ``phi_end(x)`` to a
target-capability anchor ``mu_T``::

    score_end(x) = sim(phi_end(x), mu_T)                                          (NAIT score)

``phi_end`` plays TWO roles in the registered design (§2.1, §2.3):

* the **decisive matched-budget comparator** (this baseline), and
* the **FULL control block** that the co-primary ridge-partialling-out
  regression (:mod:`neurotrace_it.analysis.residual_test`) orthogonalizes the
  trajectory features against.

This module is *pure* (stdlib only) and performs **no** model load and **no**
server call. Callers supply the already-extracted endpoint activations; this is
intentional so the selection arithmetic is testable build-now / run-later.

DO-NOT-RUN: no training, no extraction, no GPU/CPU heavy job is performed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

# A single layer's residual-stream activation vector h_l(p) in R^d.
Vector = Sequence[float]
# Endpoint activations for one example: {layer_id: (h_l(p_start), h_l(p_end))}.
EndpointActivations = Mapping[int, "tuple[Vector, Vector]"]

__all__ = [
    "EndpointNeuronSignature",
    "NaitSelectionResult",
    "endpoint_signature",
    "endpoint_score",
    "select_score",
    "nait_select",
]


def _as_floats(vector: Vector) -> list[float]:
    return [float(component) for component in vector]


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in vector))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float], *, eps: float = 1e-12) -> float:
    """Cosine similarity in ``[-1, 1]`` (numerically guarded).

    Returns ``0.0`` when either argument is the zero vector, matching the
    convention that an all-zero (degenerate) signature carries no directional
    selection signal rather than raising.
    """

    denom = _l2_norm(a) * _l2_norm(b)
    if denom <= eps:
        return 0.0
    return _dot(a, b) / denom


@dataclass(frozen=True)
class EndpointNeuronSignature:
    """Persisted NAIT endpoint signature for one example (Eq. 1).

    Attributes
    ----------
    example_id:
        Identifier of the candidate instruction example.
    layer_ids:
        The anchor-layer set ``A`` in a fixed, sorted order. The ``phi_end``
        layout concatenates ``[h_l(p_start), h_l(p_end)]`` in exactly this order.
    phi_end:
        The flattened endpoint feature in ``R^{2 d |A|}`` (Eq. 1).
    hidden_width:
        The per-layer hidden width ``d`` (so ``len(phi_end) == 2 * d * |A|``).
    """

    example_id: str
    layer_ids: tuple[int, ...]
    phi_end: tuple[float, ...]
    hidden_width: int

    def __post_init__(self) -> None:
        expected = 2 * self.hidden_width * len(self.layer_ids)
        if len(self.phi_end) != expected:
            raise ValueError(
                "phi_end length "
                f"{len(self.phi_end)} != 2*d*|A| = {expected} "
                f"(d={self.hidden_width}, |A|={len(self.layer_ids)})"
            )


@dataclass(frozen=True)
class NaitSelectionResult:
    """Output of greedy top-``B`` NAIT selection.

    ``scores`` is the per-example ``score_end`` (Eq.: NAIT score) for every
    candidate; ``selected_example_ids`` is the budget-``B`` argmax subset.
    """

    selected_example_ids: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    budget: int


def endpoint_signature(
    example_id: str,
    endpoint_activations: EndpointActivations,
    *,
    layer_ids: Sequence[int] | None = None,
) -> EndpointNeuronSignature:
    """Build the NAIT endpoint signature ``phi_end`` for one example (Eq. 1).

    Parameters
    ----------
    example_id:
        Candidate example id.
    endpoint_activations:
        Mapping ``{layer_id: (h_l(p_start), h_l(p_end))}``. Each pair holds the
        residual-stream activation at the START and END token positions for that
        layer. All vectors must share the hidden width ``d``.
    layer_ids:
        Optional explicit anchor set ``A`` (a fixed, validation-selected grid).
        When omitted, the sorted keys of ``endpoint_activations`` are used so the
        concatenation order is deterministic and persisted.

    Returns
    -------
    EndpointNeuronSignature
        The flattened ``phi_end in R^{2 d |A|}``.
    """

    if layer_ids is None:
        ordered_layers = tuple(sorted(endpoint_activations))
    else:
        ordered_layers = tuple(layer_ids)
        missing = [layer for layer in ordered_layers if layer not in endpoint_activations]
        if missing:
            raise KeyError(f"endpoint_activations missing layers {missing}")
    if not ordered_layers:
        raise ValueError("endpoint_signature requires a non-empty anchor-layer set A")

    flat: list[float] = []
    hidden_width: int | None = None
    for layer in ordered_layers:
        start_vec, end_vec = endpoint_activations[layer]
        start = _as_floats(start_vec)
        end = _as_floats(end_vec)
        if len(start) != len(end):
            raise ValueError(
                f"layer {layer}: start/end width mismatch {len(start)} vs {len(end)}"
            )
        if hidden_width is None:
            hidden_width = len(start)
        elif len(start) != hidden_width:
            raise ValueError(
                f"layer {layer}: hidden width {len(start)} != {hidden_width} (must match across A)"
            )
        flat.extend(start)
        flat.extend(end)

    assert hidden_width is not None  # guaranteed: ordered_layers is non-empty
    return EndpointNeuronSignature(
        example_id=example_id,
        layer_ids=ordered_layers,
        phi_end=tuple(flat),
        hidden_width=hidden_width,
    )


def endpoint_score(
    signature: EndpointNeuronSignature,
    target_anchor: Sequence[float],
    *,
    similarity: str = "cosine",
) -> float:
    """NAIT selection score: similarity of ``phi_end`` to the target anchor ``mu_T``.

    Parameters
    ----------
    signature:
        The endpoint signature ``phi_end`` (Eq. 1).
    target_anchor:
        The target-capability anchor ``mu_T`` in the *same* ``R^{2 d |A|}`` space
        (e.g. the mean ``phi_end`` of target-capability exemplars).
    similarity:
        ``"cosine"`` (default; class-mean neuron-overlap proxy) or ``"dot"``.

    Returns
    -------
    float
        ``score_end(x) = sim(phi_end(x), mu_T)``. Higher = more target-aligned.
    """

    anchor = _as_floats(target_anchor)
    if len(anchor) != len(signature.phi_end):
        raise ValueError(
            f"target_anchor dim {len(anchor)} != phi_end dim {len(signature.phi_end)}"
        )
    if similarity == "cosine":
        return cosine_similarity(signature.phi_end, anchor)
    if similarity == "dot":
        return _dot(signature.phi_end, anchor)
    raise ValueError(f"unknown similarity {similarity!r}; use 'cosine' or 'dot'")


# Public alias: the score used as the per-example selection key.
select_score = endpoint_score


def nait_select(
    signatures: Sequence[EndpointNeuronSignature],
    target_anchor: Sequence[float],
    *,
    budget: int,
    similarity: str = "cosine",
) -> NaitSelectionResult:
    """Greedy top-``B`` NAIT selection by endpoint-similarity score.

    This is the *decisive comparator*: it selects the budget-``B`` examples whose
    endpoint signature is most similar to ``mu_T``. Ties break by ``example_id``
    so the result is deterministic and reproducible across seeds.

    Parameters
    ----------
    signatures:
        Endpoint signatures for every candidate in the pool.
    target_anchor:
        Target-capability anchor ``mu_T``.
    budget:
        Selection budget ``B`` (number of examples to keep). ``B`` is clamped to
        the pool size.
    similarity:
        Passed through to :func:`endpoint_score`.

    Returns
    -------
    NaitSelectionResult
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    scored = [
        (signature.example_id, endpoint_score(signature, target_anchor, similarity=similarity))
        for signature in signatures
    ]
    # Sort by descending score, then ascending id for deterministic tie-breaks.
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))
    kept = min(budget, len(ordered))
    selected = tuple(example_id for example_id, _ in ordered[:kept])
    return NaitSelectionResult(
        selected_example_ids=selected,
        scores=tuple(scored),
        budget=kept,
    )

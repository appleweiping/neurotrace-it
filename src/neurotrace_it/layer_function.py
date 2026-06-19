"""Layer-routing primitives (REDESIGN_v5 §3.5, §3.6) -- ADDITIVE, DO-NOT-RUN.

This module implements the *closed-form, score-free* mask -> rank apportionment
``capacity_match`` and its feasibility constructor ``make_feasible_mask``, the
per-control deterministic mask maps of the six R1 arms (RE-FIX-3 / RE-FIX-4), the
frozen layer-ablation profile ``J_{c,l}`` (Eq. 6-3), its validation gate
(§3.6), and the descriptive leave-one-anchor-layer redistribution (Eq. 6-4).

The mathematical contracts these helpers must satisfy (verified by
``tests/test_layer_routing.py``):

* **Capacity conservation (Eq. 6-0b, Prop. (i)).** For every *feasible* anchor
  mask ``m_A`` (``|support| * r_max >= R_tot``) of any cardinality
  ``k in {k_min, .., |A|}``, ``capacity_match`` returns an integer rank vector
  ``r`` with ``sum_{l in A} r_l = R_tot``, ``r_l = 0`` off the support,
  ``r_l <= r_max`` everywhere, and it **never grows the support**.
* **Score-free / pure (RE-FIX-2).** ``capacity_match`` reads only
  ``(m_A, R_tot, r_max)``; it has no ``psi`` argument and no implementation
  freedom -- ties are broken by ascending layer index.
* **Single constructor for the score (RE-FIX-2).** ``make_feasible_mask`` is the
  ONLY place an arm-specific score (``psi`` for ``pi_psi``; a fixed key for each
  control) is consulted; it lifts any raw mask (including the empty mask =>
  uniform-over-``A``) into the feasible domain ``DOM`` before any rank call.
* **Deterministic controls (RE-FIX-3 / RE-FIX-4).** Each control is a
  deterministic function of ``x`` (its stable content hash) and fixed,
  pre-registered seeds; ``pi_ada`` reads a SEPARATE ``seed_ada`` so its
  importance is frozen OUTSIDE the per-run training-seed expectation.

DO-NOT-RUN: pure stdlib, no model load, no server call, no training. This file
specifies the *arithmetic* of the routing policy so it is testable build-now /
run-later. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

__all__ = [
    "RankVector",
    "capacity_match",
    "make_feasible_mask",
    "routing_policy",
    "control_mask",
    "leave_one_layer_redistribute",
    "frozen_layer_ablation_profile",
    "validate_J_against_freeze",
    "JValidation",
    "stable_example_hash",
    "k_min_for",
]

# A rank vector is an immutable mapping {anchor_layer_id: integer_rank}.
RankVector = "dict[int, int]"


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------
def k_min_for(total_rank: int, r_max: int) -> int:
    """Minimum ON-cardinality ``k_min = ceil(R_tot / r_max)`` (Eq. 6-0e).

    A mask must turn ON at least ``k_min`` anchor layers for the per-layer cap
    ``r_max`` to be able to hold ``R_tot`` total rank.
    """

    if r_max <= 0:
        raise ValueError("r_max must be a positive integer")
    if total_rank < 0:
        raise ValueError("total_rank (R_tot) must be non-negative")
    return math.ceil(total_rank / r_max)


def _support(mask: Mapping[int, int] | Sequence[int], anchors: Sequence[int]) -> list[int]:
    """Return the ON anchor layers (ascending) of ``mask`` restricted to ``anchors``.

    ``mask`` may be a ``{layer: 0/1}`` mapping or an explicit iterable of ON
    layer ids. Only layers in ``anchors`` are considered (the policy domain is
    exactly ``A``).
    """

    anchor_set = set(anchors)
    if isinstance(mask, Mapping):
        on = [layer for layer, bit in mask.items() if bit and layer in anchor_set]
    else:
        on = [layer for layer in mask if layer in anchor_set]
    return sorted(set(on))


# ---------------------------------------------------------------------------
# capacity_match -- the closed-form, score-free, TOTAL rank map (Eq. 6-0b/0d)
# ---------------------------------------------------------------------------
def capacity_match(
    m_A: Mapping[int, int] | Sequence[int],
    total_rank: int,
    r_max: int,
    *,
    anchors: Sequence[int],
) -> dict[int, int]:
    """Closed-form largest-remainder rank apportionment over ``support(m_A)``.

    Pure function of ``(m_A, R_tot, r_max, anchors)`` -- it **reads no ``psi``**
    (RE-FIX-2). ``anchors`` is the fixed policy domain ``A`` (a structural
    constant, not a score); it is required so the returned vector spans exactly
    ``A`` with zeros off the support.

    Steps (Eq. 6-0b):

    1. Even base share ``q = floor(R_tot / k)`` on every ON layer (``k`` = ON
       cardinality), which renormalizes automatically to any feasible ``k``.
    2. Largest-remainder top-up of the ``rem = R_tot - k*q`` leftover units, one
       rank each, resolved by the DETERMINISTIC ascending-layer-index key.
    3. Per-layer cap ``r_max`` enforced by :func:`_cap_spill` among the SAME ON
       layers only (no support growth). By the feasibility lemma (§3.5) this is a
       no-op on ``DOM``; it is retained as a defensive invariant-checker.

    Raises ``ValueError`` if the mask is infeasible
    (``k * r_max < R_tot``) -- feasibility must be guaranteed UPSTREAM by
    :func:`make_feasible_mask`, so this is a contract violation, never an
    expected branch.
    """

    support = _support(m_A, anchors)
    k = len(support)
    if k == 0:
        raise ValueError(
            "capacity_match called on an empty mask; lift it via make_feasible_mask first"
        )
    if k * r_max < total_rank:
        raise ValueError(
            f"infeasible mask: k={k} * r_max={r_max} < R_tot={total_rank}; "
            "call make_feasible_mask to project onto DOM first"
        )

    # Step 1: even base share.
    q = total_rank // k
    rem = total_rank - k * q
    rank = {layer: q for layer in support}

    # Step 2: largest-remainder top-up by ascending layer index (the fixed key).
    # All ON layers share the same fractional target, so the +1 awards go to the
    # `rem` lowest-index ON layers.
    for layer in support[:rem]:
        rank[layer] += 1

    # Step 3: cap enforcement among the same ON layers (no-op on DOM).
    rank = _cap_spill(rank, support, r_max)

    # Span exactly A with zeros off the support.
    out = {layer: 0 for layer in anchors}
    out.update(rank)

    # Invariants (the contract `tests/test_layer_routing.py` item (b) asserts).
    assert sum(out.values()) == total_rank, "capacity conservation violated"
    assert all(v <= r_max for v in out.values()), "per-layer cap violated"
    assert all(out[layer] == 0 for layer in anchors if layer not in support), (
        "support grew or leaked off the mask"
    )
    return out


def _cap_spill(rank: dict[int, int], support: Sequence[int], r_max: int) -> dict[int, int]:
    """Cap enforcement among the SAME ON layers (Eq. 6-0d), pure in ``(r, S, r_max)``.

    Clips any layer above ``r_max`` and re-awards the excess to ON layers still
    below ``r_max`` by the same ascending-index largest-remainder rule. By the
    feasibility lemma this never triggers on ``DOM`` (q <= r_max already), so it
    is a defensive invariant-checker; it NEVER grows the support.
    """

    rank = dict(rank)
    support = list(support)
    # Bounded loop: total rank is conserved and headroom only shrinks; guard
    # against any pathological caller with a finite pass cap.
    for _ in range(len(support) + 1):
        over = [layer for layer in support if rank[layer] > r_max]
        if not over:
            return rank
        excess = sum(rank[layer] - r_max for layer in over)
        for layer in over:
            rank[layer] = r_max
        free = [layer for layer in sorted(support) if rank[layer] < r_max]
        if not free:
            raise ValueError("cap spillover failed: no free ON layer (mask was infeasible)")
        # Distribute `excess` units of +1 over `free`, ascending index, wrapping.
        i = 0
        placed = 0
        while placed < excess:
            layer = free[i % len(free)]
            if rank[layer] < r_max:
                rank[layer] += 1
                placed += 1
            i += 1
            if i > excess * (len(free) + 1) + len(free):  # pragma: no cover - guard
                raise ValueError("cap spillover did not converge")
    raise ValueError("cap spillover exceeded pass bound")  # pragma: no cover - guard


# ---------------------------------------------------------------------------
# make_feasible_mask -- the ONLY place an arm's score is read (Eq. 6-0e)
# ---------------------------------------------------------------------------
def make_feasible_mask(
    raw_mask: Mapping[int, int] | Sequence[int],
    score: Mapping[int, float] | None,
    total_rank: int,
    r_max: int,
    *,
    anchors: Sequence[int],
) -> dict[int, int]:
    """Project a raw anchor mask onto the feasible domain ``DOM`` (Eq. 6-0e).

    This is the **single** place an arm-specific ``score`` is consulted:

    * empty raw mask  => uniform-over-``A`` (the ``1_A`` fallback, Eq. 6-0c);
    * already feasible (``|S| >= k_min``)  => returned unchanged;
    * otherwise add the ``(k_min - |S|)`` highest-``score`` OFF anchors, ties
      broken by ascending layer index.

    ``score`` is the arm's own object -- ``psi`` for ``pi_psi``; a fixed
    deterministic key for each control. ``capacity_match`` then maps the result
    to ranks with NO further score access (RE-FIX-2). Returns a ``{layer: 0/1}``
    mask over ``A`` guaranteed to lie in ``DOM`` (when ``|A| * r_max >= R_tot``,
    pinned in §5.6).
    """

    anchors = list(anchors)
    kmin = k_min_for(total_rank, r_max)
    if kmin > len(anchors):
        raise ValueError(
            f"k_min={kmin} > |A|={len(anchors)}: no feasible mask exists; "
            "pin r_max so |A|*r_max >= R_tot (§5.6)"
        )
    support = _support(raw_mask, anchors)
    if not support:
        support = list(anchors)  # empty => uniform-over-A (Eq. 6-0c)
    if len(support) >= kmin:
        return {layer: (1 if layer in set(support) else 0) for layer in anchors}

    # Add highest-score OFF anchors to reach k_min, ties ascending layer index.
    off = [layer for layer in anchors if layer not in set(support)]

    def key(layer: int) -> tuple[float, int]:
        s = 0.0 if score is None else float(score.get(layer, 0.0))
        return (-s, layer)  # descending score, then ascending layer index

    off_ranked = sorted(off, key=key)
    need = kmin - len(support)
    support = set(support) | set(off_ranked[:need])
    return {layer: (1 if layer in support else 0) for layer in anchors}


# ---------------------------------------------------------------------------
# routing_policy -- pi_psi raw mask (Eq. 6-0); L\A pinned at r_0
# ---------------------------------------------------------------------------
def routing_policy(
    psi: Mapping[int, float],
    tau_sel: float,
    *,
    anchors: Sequence[int],
) -> dict[int, int]:
    """``pi_psi`` raw anchor mask ``raw_m_l(x) = 1[psi_l(x) >= tau_sel]`` (Eq. 6-0).

    Returns the *pre-feasibility* mask over ``A``; the caller lifts it via
    :func:`make_feasible_mask` (score = ``psi``) and then :func:`capacity_match`.
    The non-anchor layers ``L\\A`` are held at ``r_0`` by every arm and are NOT
    part of this mask (they cancel in every contrast ``V(pi_psi) - V(pi_c)``).
    """

    return {layer: (1 if float(psi.get(layer, 0.0)) >= tau_sel else 0) for layer in anchors}


# ---------------------------------------------------------------------------
# Deterministic control masks (RE-FIX-3 / RE-FIX-4)
# ---------------------------------------------------------------------------
def stable_example_hash(example_id: str) -> int:
    """Stable content hash ``H(x)`` (persisted in ``schemas_v2.pool_hashes``).

    Uses SHA-256 of the example id so the value is identical across processes
    and Python invocations (``hash()`` is salted per-process and would NOT be
    reproducible). Returns a 64-bit non-negative integer seed component.
    """

    digest = hashlib.sha256(example_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _seeded_permutation(seed: int, n: int) -> list[int]:
    """Deterministic permutation of ``range(n)`` from ``seed`` (Fisher-Yates / SHA stream)."""

    order = list(range(n))
    # Draw a deterministic stream of bytes from the seed and Fisher-Yates shuffle.
    counter = 0
    for i in range(n - 1, 0, -1):
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        counter += 1
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def _seeded_subset(seed: int, anchors: Sequence[int], k: int) -> list[int]:
    """Deterministic size-``k`` subset of ``anchors`` from ``seed`` (ascending)."""

    perm = _seeded_permutation(seed, len(anchors))
    chosen = sorted(anchors[perm[i]] for i in range(min(k, len(anchors))))
    return chosen


def control_mask(
    arm: str,
    example_id: str,
    psi: Mapping[int, float],
    *,
    anchors: Sequence[int],
    tau_sel: float,
    total_rank: int,
    r_max: int,
    seed_rand: int,
    seed_shuf: int,
    seed_ada: int,
    A_glob: Sequence[int],
    ada_importance: Mapping[int, float] | None = None,
    stratum: int = 0,
    stratum_perm: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Deterministic per-control RAW anchor mask of the six R1 arms (RE-FIX-3/4).

    Every control is a deterministic function of ``x`` (its stable hash) and the
    fixed pre-registered seeds; the randomness is part of the *policy
    definition*, NOT part of the ``V(pi)`` training-seed expectation.

    Arms (§3.5):

    * ``"psi"``     -- ``1[psi_l >= tau_sel]`` (the policy under test).
    * ``"unif"``    -- ``1_A`` (all anchors ON).
    * ``"shuf"``    -- a fixed within-stratum permutation of the anchor labels
      applied to ``psi`` before thresholding (breaks the layer<->example
      coupling while holding the ``psi`` marginal fixed).
    * ``"rand"``    -- a deterministic pseudo-random subset of ``A`` of
      cardinality matched to ``pi_psi``'s pre-feasibility cardinality, drawn from
      ``seed_rand XOR H(x)``.
    * ``"global"``  -- the fixed global-selective block ``A_glob`` for EVERY
      ``x`` (surgical-style).
    * ``"ada"``     -- top-``k`` anchors by the FROZEN AdaLoRA importance
      ``ada_importance`` (computed ONCE under ``seed_ada``, OUTSIDE ``V(pi)``;
      RE-FIX-4), tie-break ascending layer index.

    Returns the *raw* (pre-feasibility) ``{layer: 0/1}`` mask; the caller lifts
    it with :func:`make_feasible_mask` (score = the arm's own object) and then
    :func:`capacity_match`.
    """

    anchors = list(anchors)
    hx = stable_example_hash(example_id)

    if arm == "psi":
        return routing_policy(psi, tau_sel, anchors=anchors)

    if arm == "unif":
        return {layer: 1 for layer in anchors}

    if arm == "shuf":
        # Fixed within-stratum permutation sigma_g of the anchor labels.
        if stratum_perm is None:
            perm = _seeded_permutation(seed_shuf ^ stratum, len(anchors))
            sigma = {anchors[i]: anchors[perm[i]] for i in range(len(anchors))}
        else:
            sigma = dict(stratum_perm)
        psi_shuf = {layer: float(psi.get(sigma[layer], 0.0)) for layer in anchors}
        return routing_policy(psi_shuf, tau_sel, anchors=anchors)

    if arm == "rand":
        # Match pi_psi's pre-feasibility cardinality k(x).
        k_x = sum(1 for layer in anchors if float(psi.get(layer, 0.0)) >= tau_sel)
        k_x = max(0, min(k_x, len(anchors)))
        chosen = _seeded_subset(seed_rand ^ hx, anchors, k_x)
        return {layer: (1 if layer in set(chosen) else 0) for layer in anchors}

    if arm == "global":
        glob = set(A_glob)
        return {layer: (1 if layer in glob else 0) for layer in anchors}

    if arm == "ada":
        if ada_importance is None:
            # No persisted importance => fall back to the seed_ada-frozen ordering
            # (a deterministic key over A; never the training seed, RE-FIX-4).
            perm = _seeded_permutation(seed_ada, len(anchors))
            imp = {anchors[perm[i]]: float(len(anchors) - i) for i in range(len(anchors))}
        else:
            imp = {layer: float(ada_importance.get(layer, 0.0)) for layer in anchors}
        k_x = sum(1 for layer in anchors if float(psi.get(layer, 0.0)) >= tau_sel)
        k_x = max(1, min(k_x if k_x > 0 else 1, len(anchors)))
        ranked = sorted(anchors, key=lambda l: (-imp[l], l))  # noqa: E741
        chosen = set(ranked[:k_x])
        return {layer: (1 if layer in chosen else 0) for layer in anchors}

    raise ValueError(f"unknown control arm {arm!r}")


# ---------------------------------------------------------------------------
# Descriptive leave-one-anchor-layer redistribution (Eq. 6-4)
# ---------------------------------------------------------------------------
def leave_one_layer_redistribute(
    layer: int,
    *,
    anchors: Sequence[int],
    total_rank: int,
    r_max: int,
) -> dict[int, int]:
    """``capacity_match`` on the feasible ``A\\{layer}`` mask (Eq. 6-4).

    The run-level leave-one-anchor-layer policy ``pi_{-l}`` masks anchor ``layer``
    OFF for ALL examples and re-runs ``capacity_match`` on the remaining ON
    anchors, so ``sum r_l = R_tot`` is preserved and ``L\\A`` stays ``r_0``. This
    underpins the DESCRIPTIVE per-layer ``Delta_l^{LOL}`` (never confirmatory).
    """

    remaining = [l for l in anchors if l != layer]  # noqa: E741
    mask = {l: 1 for l in remaining}  # noqa: E741
    feasible = make_feasible_mask(mask, None, total_rank, r_max, anchors=anchors)
    # The drop must not be re-added by feasibility completion; assert it stays off.
    return capacity_match(feasible, total_rank, r_max, anchors=anchors)


# ---------------------------------------------------------------------------
# Frozen layer-ablation profile J_{c,l} (Eq. 6-3) and its validation gate (§3.6)
# ---------------------------------------------------------------------------
def frozen_layer_ablation_profile(
    acc_full: Mapping[str, float],
    acc_ablated: Mapping[str, Mapping[int, float]],
    *,
    anchors: Sequence[int],
) -> dict[str, dict[int, float]]:
    """Frozen layer-ablation profile ``J_{c,l}`` over ``A`` (Eq. 6-3).

    ``J_{c,l} = (Acc_c(model) - Acc_c(model | h_l <- mean)) / Acc_c(model)`` for
    each capability ``c`` and anchor ``l in A``. Inputs are the (caller-supplied,
    already-measured) full and mean-ablated accuracies; this is pure arithmetic
    -- activation replacement is off-manifold so NO causal claim is made (the
    profile is renamed from the v4 ``I_{c,l}`` for exactly this reason).
    """

    profile: dict[str, dict[int, float]] = {}
    for c, acc_c in acc_full.items():
        denom = float(acc_c)
        if denom == 0.0:
            raise ValueError(f"Acc_c(model) is zero for capability {c!r}; J undefined")
        row: dict[int, float] = {}
        ablated_c = acc_ablated.get(c, {})
        for layer in anchors:
            if layer not in ablated_c:
                raise KeyError(f"missing ablated accuracy for capability {c!r}, layer {layer}")
            row[layer] = (denom - float(ablated_c[layer])) / denom
        profile[c] = row
    return profile


@dataclass(frozen=True)
class JValidation:
    """Outcome of the §3.6 validation gate for ``J_{c,l}``."""

    spearman_rho: float
    rho_lower_ci: float | None
    rho_threshold: float
    passes: bool
    action: str  # "use_J_prior" | "drop_J_use_psi_alone"


def validate_J_against_freeze(
    j_row: Mapping[int, float],
    delta_lol: Mapping[int, float],
    *,
    anchors: Sequence[int],
    rho_threshold: float,
    rho_lower_ci: float | None = None,
    spearman_fn: Callable[[Sequence[float], Sequence[float]], float] | None = None,
) -> JValidation:
    """§3.6 gate: ``J`` enters the router prior ONLY if it correlates with the real
    anchor leave-one-layer policy-value difference ``Delta_l^{LOL}`` (Eq. 6-4).

    Requires Spearman ``rho >= rho_threshold`` with ``lower CI > 0`` (when a CI is
    supplied). **Fail => ``J`` is dropped** and reported as a descriptive
    diagnostic; the router then uses ``psi`` alone.
    """

    if spearman_fn is None:
        from .analysis.outcome_y import spearman_rho as _spearman

        spearman_fn = _spearman  # type: ignore[assignment]

    xs = [float(j_row[layer]) for layer in anchors]
    ys = [float(delta_lol[layer]) for layer in anchors]
    rho = float(spearman_fn(xs, ys))
    ci_ok = True if rho_lower_ci is None else (rho_lower_ci > 0.0)
    passes = (rho >= rho_threshold) and ci_ok
    return JValidation(
        spearman_rho=rho,
        rho_lower_ci=rho_lower_ci,
        rho_threshold=rho_threshold,
        passes=passes,
        action="use_J_prior" if passes else "drop_J_use_psi_alone",
    )

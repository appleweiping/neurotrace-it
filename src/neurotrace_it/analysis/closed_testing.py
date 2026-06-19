"""Closed-testing graph with IUT union-null leaves (REDESIGN_v5 §4, RE-FIX-5).

ADDITIVE, DO-NOT-RUN. The confirmatory family is the SIX elementary nulls

    {H0^{R0}, H0^{R1}, H0^{G2t}, H0^{G2r}, H0^{G2h}, H0^{G2c}}

Two of them are themselves UNION nulls, and they are *single* elementary leaves
of the closed family:

    H0^{R1}  = union_c   H0^{R1,c}    over the five control contrasts
    H0^{G2t} = union_b   H0^{G2t,b}   over the baseline-set comparators

THE FWER-CRITICAL RULE (RE-FIX-5). A valid p-value for the FULL union null is the
MAXIMUM of its components' marginal p-values (the IUT: reject the union iff every
component clears its marginal level, i.e. max_c p_c <= alpha). Dropping to an
"in-scope subset" of the components -- equivalently using ``max`` over only a
subset, which is stochastically SMALLER -- is anti-conservative whenever the
binding component is the omitted one (a real FWER hole). This module always uses
the FULL-union ``max`` for the R1/G2t leaves.

Inference is modelled on per-node p-values (singletons R0/G2r/G2h/G2c) and the
union-null leaf p-values (``p_R1 = max_c p_{R1,c}``, ``p_G2t = max_b p_{G2t,b}``).
The closed test uses the pinned Maurer-Bretz graphical gatekeeping graph
``R0 -> R1 -> G2t -> {G2r,G2h,G2c}`` (§4.1) whose local test of an intersection
``H_I`` is the WEIGHTED-BONFERRONI test with the graph's current alpha-weights on
``I``. Provides:

* :func:`closed_test_shortcut` -- the Maurer-Bretz sequentially-rejective
  shortcut (Bretz et al. 2009, Prop. 1).
* :func:`closed_test_bruteforce` -- the explicit 2^6 - 1 = 63 intersection
  closed test with the graphical weighted-Bonferroni local tests.
* :func:`assert_shortcut_equals_bruteforce` -- the equivalence assertion
  (``test_layer_routing.py`` item (f)).
* :func:`fwer_simulation` -- a synthetic-null FWER probe (Prop. P1-FWER),
  including the least-favorable single-binding-control configuration.

DO-NOT-RUN: pure stdlib; the caller supplies the per-node/per-component p-values
(from the bootstrap-``t`` / permutation tests, run-later). The closure arithmetic
is testable build-now / run-later. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "ELEMENTARY_NULLS",
    "ClosedTestInputs",
    "ClosedTestResult",
    "GraphSpec",
    "DEFAULT_GRAPH",
    "iut_leaf_rejects",
    "union_leaf_pvalue",
    "closed_test_shortcut",
    "closed_test_bruteforce",
    "assert_shortcut_equals_bruteforce",
    "fwer_simulation",
    "ALPHA_GRAPH",
]

# The six elementary nulls of the confirmatory closed family (§4.1b).
ELEMENTARY_NULLS: tuple[str, ...] = ("R0", "R1", "G2t", "G2r", "G2h", "G2c")

# Pinned alpha-graph constants (§4.1): the cross-family gating structure.
ALPHA_GRAPH: dict[str, object] = {
    "alpha": 0.05,
    "w_0": 1.00,
    "edges": {("R0", "R1"): 1.0, ("R1", "G2t"): 1.0},
    "split": {"G2r": 0.34, "G2h": 0.33, "G2c": 0.33},  # G2t -> {G2r,G2h,G2c}
    "recycle": 1.0,
}


def iut_leaf_rejects(component_rejects: Sequence[bool]) -> bool:
    """A union-null leaf (R1/G2t) rejects iff EVERY one of its components rejects.

    This is the IUT (Eq. 7-IUT): the FULL union null ``union_c H0,c`` is rejected
    only when all components clear the margin. Requiring only a SUBSET would test
    a strictly smaller hypothesis and is forbidden (RE-FIX-5).
    """

    components = list(component_rejects)
    if not components:
        raise ValueError("union-null leaf needs at least one component")
    return all(components)


def union_leaf_pvalue(component_pvalues: Sequence[float]) -> float:
    """Valid p-value for a FULL union null = ``max_c p_c`` (the IUT p-value).

    The union null ``union_c H0,c`` is rejected at level ``alpha`` iff every
    component is below ``alpha`` -- equivalently iff ``max_c p_c <= alpha``. Using
    a ``max`` over a SUBSET of components is stochastically smaller and is
    forbidden (RE-FIX-5: the binding component must be inside the max).
    """

    comps = list(component_pvalues)
    if not comps:
        raise ValueError("union-null leaf needs at least one component p-value")
    return max(comps)


@dataclass(frozen=True)
class GraphSpec:
    """Pinned Maurer-Bretz gatekeeping graph (§4.1).

    ``weights`` is the initial alpha-allocation over the six elementary nulls
    (all mass starts on R0). ``edges`` is the directed propagation graph: on
    rejection of a node, its weight is transferred to its children by the listed
    fractions (each node's outgoing fractions sum to <= 1).
    """

    nodes: tuple[str, ...] = ELEMENTARY_NULLS
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"R0": 1.0, "R1": 0.0, "G2t": 0.0, "G2r": 0.0, "G2h": 0.0, "G2c": 0.0}
    )
    edges: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "R0": {"R1": 1.0},
            "R1": {"G2t": 1.0},
            "G2t": {"G2r": 0.34, "G2h": 0.33, "G2c": 0.33},
            "G2r": {"G2t": 1.0},  # recycling edge
            "G2h": {"G2t": 1.0},
            "G2c": {"G2t": 1.0},
        }
    )


DEFAULT_GRAPH = GraphSpec()


@dataclass(frozen=True)
class ClosedTestInputs:
    """Per-node p-values for one realized dataset (§4.1b).

    * ``p_r0`` -- the R0 permutation-test p-value.
    * ``r1_component_pvalues`` -- per-CONTROL p-values for the FULL R1 union;
      the leaf p-value is ``max`` over ALL five (RE-FIX-5).
    * ``g2t_component_pvalues`` -- per-COMPARATOR p-values for the FULL G2t union.
    * ``p_g2r/p_g2h/p_g2c`` -- the single drift/cost p-values.

    A convenience constructor :meth:`from_rejections` builds inputs from boolean
    rejections at a fixed alpha (used by the FWER probe), mapping a rejection to a
    p-value of 0 and a non-rejection to 1.
    """

    p_r0: float
    r1_component_pvalues: tuple[float, ...]
    g2t_component_pvalues: tuple[float, ...]
    p_g2r: float
    p_g2h: float
    p_g2c: float

    def node_pvalue(self, node: str) -> float:
        if node == "R0":
            return self.p_r0
        if node == "R1":
            return union_leaf_pvalue(self.r1_component_pvalues)
        if node == "G2t":
            return union_leaf_pvalue(self.g2t_component_pvalues)
        if node == "G2r":
            return self.p_g2r
        if node == "G2h":
            return self.p_g2h
        if node == "G2c":
            return self.p_g2c
        raise ValueError(f"unknown node {node!r}")

    @classmethod
    def from_rejections(
        cls,
        *,
        r0_rejects: bool,
        r1_components: Sequence[bool],
        g2t_components: Sequence[bool],
        g2r_rejects: bool,
        g2h_rejects: bool,
        g2c_rejects: bool,
        alpha: float,
    ) -> "ClosedTestInputs":
        """Build inputs from booleans by mapping reject->p<alpha and not->p=1.

        A rejecting component is given a p-value of ``alpha / 2`` (strictly below
        the marginal level) and a non-rejecting one ``1.0``; the union-null leaf p
        is then ``max`` over components, exactly reproducing the IUT.
        """

        rj = alpha / 2.0
        return cls(
            p_r0=(rj if r0_rejects else 1.0),
            r1_component_pvalues=tuple(rj if b else 1.0 for b in r1_components),
            g2t_component_pvalues=tuple(rj if b else 1.0 for b in g2t_components),
            p_g2r=(rj if g2r_rejects else 1.0),
            p_g2h=(rj if g2h_rejects else 1.0),
            p_g2c=(rj if g2c_rejects else 1.0),
        )


@dataclass(frozen=True)
class ClosedTestResult:
    """Closed-test rejection set over the six elementary nulls."""

    rejected: frozenset[str]

    def rejects(self, null: str) -> bool:
        return null in self.rejected


# ---------------------------------------------------------------------------
# Brute-force closed test with graphical weighted-Bonferroni local tests
# ---------------------------------------------------------------------------
def _subset_weights(subset: tuple[str, ...], graph: GraphSpec) -> dict[str, float]:
    """Alpha-weights the graph assigns to the intersection ``H_I`` (Maurer-Bretz).

    For the intersection over ``subset``, repeatedly propagate the initial weight
    mass of nodes OUTSIDE ``subset`` into ``subset`` along the (renormalized)
    edges, until all mass lives on ``subset``. The resulting per-node weights
    define the weighted-Bonferroni local test ``reject iff exists i: p_i <=
    alpha * w_i`` (the exact Maurer-Bretz local test for ``H_I``).
    """

    subset_set = set(subset)
    w = {n: float(graph.weights.get(n, 0.0)) for n in graph.nodes}

    # Iteratively push weight from outside-subset nodes into subset along edges.
    # The graph is a finite DAG-with-recycling; mass strictly moves toward subset
    # because every node outside subset has an outgoing path into subset under the
    # pinned graph. Bounded iteration with convergence guard.
    for _ in range(4 * len(graph.nodes) + 8):
        outside = [n for n in graph.nodes if n not in subset_set and w[n] > 1e-15]
        if not outside:
            break
        moved = False
        for n in outside:
            children = graph.edges.get(n, {})
            total = math.fsum(children.values())
            if total <= 0:
                continue
            mass = w[n]
            w[n] = 0.0
            for child, frac in children.items():
                w[child] = w.get(child, 0.0) + mass * (frac / total)
            moved = True
        if not moved:
            break
    return {n: w.get(n, 0.0) for n in subset}


def _intersection_rejects(subset: tuple[str, ...], inputs: ClosedTestInputs, graph: GraphSpec, alpha: float) -> bool:
    """Maurer-Bretz weighted-Bonferroni local test of ``H_I`` (§4.1b).

    Reject ``H_I`` iff some node ``i in subset`` has ``p_i <= alpha * w_i`` with
    ``w_i`` the graph weight the intersection assigns to ``i``. The R1/G2t node
    p-values are the FULL-union ``max`` (RE-FIX-5).
    """

    weights = _subset_weights(subset, graph)
    for node in subset:
        w = weights.get(node, 0.0)
        if w <= 0.0:
            continue
        if inputs.node_pvalue(node) <= alpha * w + 1e-15:
            return True
    return False


def closed_test_bruteforce(
    inputs: ClosedTestInputs,
    *,
    graph: GraphSpec = DEFAULT_GRAPH,
    alpha: float = 0.05,
) -> ClosedTestResult:
    """Explicit closed test over all ``2^6 - 1 = 63`` intersections (§4.1b).

    Rejects an elementary ``H_i`` iff EVERY intersection ``H_I`` with ``i in I`` is
    rejected by its graphical weighted-Bonferroni local test. The union-null
    leaves R1/G2t use the FULL-union ``max`` p-value inside every intersection
    (RE-FIX-5).
    """

    nodes = graph.nodes
    all_subsets = [
        subset
        for r in range(1, len(nodes) + 1)
        for subset in itertools.combinations(nodes, r)
    ]
    rejected: set[str] = set()
    for elem in nodes:
        containing = [s for s in all_subsets if elem in s]
        if all(_intersection_rejects(s, inputs, graph, alpha) for s in containing):
            rejected.add(elem)
    return ClosedTestResult(rejected=frozenset(rejected))


# ---------------------------------------------------------------------------
# Graphical shortcut (Bretz et al. 2009, Prop. 1)
# ---------------------------------------------------------------------------
def closed_test_shortcut(
    inputs: ClosedTestInputs,
    *,
    graph: GraphSpec = DEFAULT_GRAPH,
    alpha: float = 0.05,
) -> ClosedTestResult:
    """Sequentially-rejective Maurer-Bretz shortcut (Bretz et al. 2009, Prop. 1).

    Start with the initial weights; repeatedly reject any remaining node ``i``
    with ``p_i <= alpha * w_i``, remove it, and propagate its weight to its
    children by the (renormalized) edge fractions, updating the edge graph by the
    standard Bretz update. Provably equivalent to the brute-force closed test
    (:func:`assert_shortcut_equals_bruteforce`). The R1/G2t node p-values are the
    FULL-union ``max`` (RE-FIX-5).
    """

    remaining = set(graph.nodes)
    w = {n: float(graph.weights.get(n, 0.0)) for n in graph.nodes}
    # Mutable edge graph G[i][j].
    g: dict[str, dict[str, float]] = {
        n: {k: float(v) for k, v in graph.edges.get(n, {}).items() if k in remaining}
        for n in graph.nodes
    }
    pvals = {n: inputs.node_pvalue(n) for n in graph.nodes}
    rejected: set[str] = set()

    progress = True
    while progress and remaining:
        progress = False
        for node in list(remaining):
            if pvals[node] <= alpha * w[node] + 1e-15 and w[node] > 0.0:
                # Reject `node`; remove it and update the graph (Bretz update).
                rejected.add(node)
                remaining.discard(node)
                _bretz_update(node, w, g, remaining)
                progress = True
                break
    return ClosedTestResult(rejected=frozenset(rejected))


def _bretz_update(node: str, w: dict[str, float], g: dict[str, dict[str, float]], remaining: set[str]) -> None:
    """The standard Maurer-Bretz weight/edge update on rejecting ``node``."""

    children = {k: v for k, v in g.get(node, {}).items() if k in remaining}
    total = math.fsum(children.values())
    # Transfer node weight to children.
    if total > 0:
        for child, frac in children.items():
            w[child] = w.get(child, 0.0) + w[node] * (frac / total)
    w[node] = 0.0
    # Update edges: G'[i][k] = (G[i][k] + G[i][node]*G[node][k]) / (1 - G[i][node]*G[node][i])
    g_node = {k: (children[k] / total if total > 0 else 0.0) for k in children}
    for i in list(remaining):
        gi = g.get(i, {})
        g_i_node = gi.get(node, 0.0)
        if g_i_node <= 0.0:
            gi.pop(node, None)
            continue
        denom = 1.0 - g_i_node * g_node.get(i, 0.0)
        new_row: dict[str, float] = {}
        for k in remaining:
            if k == i:
                continue
            val = gi.get(k, 0.0) + g_i_node * g_node.get(k, 0.0)
            if denom > 1e-15:
                val /= denom
            if val > 0.0:
                new_row[k] = val
        g[i] = new_row
    g[node] = {}


def assert_shortcut_equals_bruteforce(
    inputs: ClosedTestInputs,
    *,
    graph: GraphSpec = DEFAULT_GRAPH,
    alpha: float = 0.05,
) -> None:
    """Assert the shortcut and the 63-intersection brute-force agree (§4.1b).

    Raises ``AssertionError`` if the two procedures disagree on the rejection set
    -- including the union-null IUT leaves and any single-binding-control
    configuration (``test_layer_routing.py`` item (f)).
    """

    short = closed_test_shortcut(inputs, graph=graph, alpha=alpha).rejected
    brute = closed_test_bruteforce(inputs, graph=graph, alpha=alpha).rejected
    if short != brute:
        raise AssertionError(
            f"shortcut != brute-force closed test: shortcut={sorted(short)} "
            f"brute={sorted(brute)}"
        )


# ---------------------------------------------------------------------------
# Prop. P1-FWER synthetic-null probe (deterministic, pure stdlib)
# ---------------------------------------------------------------------------
def _hashlib_sha(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _uniform(seed: int, salt: int) -> float:
    """Deterministic uniform(0,1) from a SHA-256 stream (reproducible across runs)."""

    digest = _hashlib_sha(f"{seed}:{salt}")
    return int.from_bytes(digest[:8], "big") / 2**64


def fwer_simulation(
    *,
    n_trials: int,
    alpha: float,
    seed: int = 0,
    true_nulls: Sequence[str] = ELEMENTARY_NULLS,
    binding_only: bool = False,
    graph: GraphSpec = DEFAULT_GRAPH,
) -> float:
    """Estimate the closed-test FWER on a synthetic null DGP (Prop. P1-FWER).

    Each trial draws independent per-component uniform p-values. For a TRUE null,
    each component p ~ Uniform(0,1) (so the marginal test rejects with prob
    ``alpha``); for an ALTERNATIVE, p = 0 (always rejects). With
    ``binding_only=True`` the R1/G2t unions are at the LEAST-FAVORABLE
    configuration: exactly one component is the true null (p ~ U) and the rest are
    alternatives (p=0). FWER is the fraction of trials rejecting AT LEAST ONE true
    null; the corrected full-union ``max`` p-value must keep FWER <= alpha.

    Returns the empirical FWER in [0, 1]; pure deterministic stdlib.
    """

    n_r1, n_g2t = 5, 3
    true_set = set(true_nulls)
    any_rejection = 0

    for trial in range(n_trials):
        base = seed * 1_000_003 + trial

        p_r0 = _uniform(base, 0) if "R0" in true_set else 0.0

        r1_p: list[float] = []
        for j in range(n_r1):
            if "R1" in true_set:
                if binding_only and j != 0:
                    r1_p.append(0.0)  # non-binding alternatives
                else:
                    r1_p.append(_uniform(base, 10 + j))  # binding true-null component
            else:
                r1_p.append(0.0)  # whole R1 is an alternative

        g2t_p: list[float] = []
        for j in range(n_g2t):
            if "G2t" in true_set:
                if binding_only and j != 0:
                    g2t_p.append(0.0)
                else:
                    g2t_p.append(_uniform(base, 20 + j))
            else:
                g2t_p.append(0.0)

        p_g2r = _uniform(base, 30) if "G2r" in true_set else 0.0
        p_g2h = _uniform(base, 31) if "G2h" in true_set else 0.0
        p_g2c = _uniform(base, 32) if "G2c" in true_set else 0.0

        inputs = ClosedTestInputs(
            p_r0=p_r0,
            r1_component_pvalues=tuple(r1_p),
            g2t_component_pvalues=tuple(g2t_p),
            p_g2r=p_g2r,
            p_g2h=p_g2h,
            p_g2c=p_g2c,
        )
        result = closed_test_bruteforce(inputs, graph=graph, alpha=alpha)
        if any(result.rejects(null) for null in true_set):
            any_rejection += 1

    return any_rejection / n_trials if n_trials else 0.0

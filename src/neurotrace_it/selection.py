"""Monotone-submodular selection objective + greedy with the (1-1/e) guarantee.

Implements the facility-location / coverage objective ``F(S)`` of
``docs/redesign/REDESIGN_v4.md`` §2.8 (Eq. 14) and its lazy-greedy maximizer.

Objective (Eq. 14)
-------------------
Per-example modular utility ``u(x) = max(0, g_hat(x) - lambda_r r_hat(x) -
lambda_f f_hat(x)) >= 0`` (the ``g_hat - lambda_r r_hat`` kernel reuses
:func:`neurotrace_it.metrics.retention_adjusted_gain`; the factuality term enters
only when the G6 gate passes, else ``lambda_f = 0``). Facility-location coverage
over the trajectory-feature space::

    C(S) = sum_{e in E} w_e * max_{x in S} sigma(x, e)
    F(S) = sum_{x in S} u(x) + mu * C(S),   mu >= 0                              (Eq. 14)

Guarantee (NWF (1-1/e))
-----------------------
``F`` is monotone non-decreasing and submodular (proof in §2.8): ``sum u(x)`` is
modular+monotone for ``u >= 0``; each ``S -> max_{x in S} sigma(x, e)`` is
monotone+submodular; a non-negative weighted sum of monotone-submodular functions
is monotone-submodular; hence ``F`` is, and greedy returns ``S_greedy`` with
``F(S_greedy) >= (1 - 1/e) F(S*)``.

We implement the standard accelerated **lazy greedy** (Minoux): because ``F`` is
submodular, marginal gains are non-increasing as ``S`` grows, so a max-heap of
stale upper bounds can be re-evaluated lazily without changing the result. The
selection is identical to naive greedy and inherits the ``(1 - 1/e)`` bound.

DO-NOT-RUN: pure stdlib; no model load, no server call, no training.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .metrics import retention_adjusted_gain

# A similarity function sigma(x, e) in [0, 1] between candidate id x and ground
# element id e (a target-concept exemplar).
SimilarityFn = Callable[[str, str], float]

__all__ = [
    "GROUND_ELEMENT_DEFAULT_WEIGHT",
    "ONE_MINUS_INV_E",
    "CoverageObjective",
    "GreedySelection",
    "example_utility",
    "facility_location_value",
    "greedy_submodular_select",
]

GROUND_ELEMENT_DEFAULT_WEIGHT = 1.0
ONE_MINUS_INV_E = 1.0 - 1.0 / math.e  # the NWF approximation ratio ~= 0.632


def example_utility(
    gain: float,
    retention_drift: float,
    *,
    drift_weight: float = 1.0,
    factuality_drift: float = 0.0,
    factuality_weight: float = 0.0,
) -> float:
    """Per-example modular utility ``u(x) >= 0`` (§2.8).

    ``u(x) = max(0, g_hat - lambda_r r_hat - lambda_f f_hat)``. The
    ``g_hat - lambda_r r_hat`` core reuses
    :func:`neurotrace_it.metrics.retention_adjusted_gain` so the kernel is shared
    with the rest of the pipeline; ``factuality_weight`` (``lambda_f``) must stay
    ``0.0`` unless the G6 precondition gate passes.
    """

    adjusted = retention_adjusted_gain(gain, retention_drift, drift_weight=drift_weight)
    adjusted -= factuality_weight * factuality_drift
    return max(0.0, adjusted)


@dataclass(frozen=True)
class CoverageObjective:
    """Facility-location coverage + modular utility objective ``F`` (Eq. 14).

    Attributes
    ----------
    utilities:
        ``{candidate_id: u(x)}`` with ``u(x) >= 0`` (build via
        :func:`example_utility`).
    ground_elements:
        The held-out exemplar set ``E`` (ids of target-concept exemplars).
    similarity:
        ``sigma(x, e) in [0, 1]`` between a candidate and a ground element.
    ground_weights:
        Optional ``{e: w_e >= 0}``; defaults to weight 1 for every element.
    coverage_weight:
        ``mu >= 0``, the facility-location mixing coefficient.
    """

    utilities: Mapping[str, float]
    ground_elements: Sequence[str]
    similarity: SimilarityFn
    ground_weights: Mapping[str, float] = field(default_factory=dict)
    coverage_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.coverage_weight < 0:
            raise ValueError("coverage_weight (mu) must be non-negative")
        for candidate_id, value in self.utilities.items():
            if value < 0:
                raise ValueError(f"utility for {candidate_id} is negative; u(x) must be >= 0")

    def weight_of(self, element: str) -> float:
        return float(self.ground_weights.get(element, GROUND_ELEMENT_DEFAULT_WEIGHT))

    def coverage_value(self, selected: Sequence[str]) -> float:
        """``C(S) = sum_e w_e * max_{x in S} sigma(x, e)``."""

        if not selected:
            return 0.0
        total = 0.0
        for element in self.ground_elements:
            best = max(self.similarity(candidate, element) for candidate in selected)
            total += self.weight_of(element) * best
        return total

    def utility_value(self, selected: Sequence[str]) -> float:
        return math.fsum(self.utilities.get(candidate, 0.0) for candidate in selected)

    def value(self, selected: Sequence[str]) -> float:
        """``F(S) = sum_{x in S} u(x) + mu * C(S)`` (Eq. 14)."""

        return self.utility_value(selected) + self.coverage_weight * self.coverage_value(selected)


def facility_location_value(objective: CoverageObjective, selected: Sequence[str]) -> float:
    """Convenience: evaluate ``F(S)`` for an explicit set (used in unit tests)."""

    return objective.value(selected)


@dataclass(frozen=True)
class GreedySelection:
    """Result of greedy maximization with the recorded approximation ratio."""

    selected_example_ids: tuple[str, ...]
    objective_value: float
    marginal_gains: tuple[tuple[str, float], ...]
    approximation_ratio: float = ONE_MINUS_INV_E


def greedy_submodular_select(
    objective: CoverageObjective,
    candidate_ids: Sequence[str],
    *,
    budget: int,
) -> GreedySelection:
    """Lazy-greedy maximization of ``F`` with the ``(1 - 1/e)`` guarantee (§2.8).

    Selects up to ``budget`` candidates by repeatedly taking the element of
    maximal marginal gain ``F(S u {x}) - F(S)``. Because ``F`` is monotone
    submodular, the accelerated lazy-greedy (max-heap of stale marginal bounds)
    yields the *same* set as naive greedy while re-evaluating far fewer marginals;
    ties break by ascending id for determinism.

    Returns
    -------
    GreedySelection
        The selected ids (in selection order), the achieved ``F(S)``, the
        per-step marginal gains, and ``approximation_ratio = 1 - 1/e``.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")

    selected: list[str] = []
    selected_set: set[str] = set()
    base_value = 0.0  # F(S) for the current S; starts at F(empty) = 0.
    marginal_gains: list[tuple[str, float]] = []

    def marginal(candidate: str) -> float:
        return objective.value(selected + [candidate]) - base_value

    # Lazy-greedy heap: entries are (-stale_gain, generation, candidate_id).
    # generation marks the |S| at which stale_gain was computed.
    heap: list[tuple[float, int, str]] = []
    for candidate in sorted(candidate_ids):
        if candidate in selected_set:
            continue
        # Initial upper bound at |S| = 0: the candidate's standalone F value.
        heapq.heappush(heap, (-objective.value([candidate]), 0, candidate))

    target = min(budget, len({cid for cid in candidate_ids}))
    while heap and len(selected) < target:
        neg_gain, generation, candidate = heapq.heappop(heap)
        if candidate in selected_set:
            continue
        current_generation = len(selected)
        if generation == current_generation:
            # Bound is fresh for the current S: accept this candidate.
            gain = -neg_gain
            selected.append(candidate)
            selected_set.add(candidate)
            base_value += gain
            marginal_gains.append((candidate, gain))
        else:
            # Stale bound: recompute the true marginal and re-insert.
            heapq.heappush(heap, (-marginal(candidate), current_generation, candidate))

    return GreedySelection(
        selected_example_ids=tuple(selected),
        objective_value=base_value,
        marginal_gains=tuple(marginal_gains),
        approximation_ratio=ONE_MINUS_INV_E,
    )

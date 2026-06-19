"""Compute-match ledger and two-sided cost gates (REDESIGN_v5 §3.7, §3.5 R3, §3.8).

ADDITIVE, DO-NOT-RUN. This module turns the §3.7 "matched budget" definition and
the §3.5 Gate R3 honest-cost contract into pure arithmetic:

* :func:`gate1_multiplier` -- the extraction-cost multiplier vs the cheapest
  baseline extraction.
* :func:`extraction_parity_check` -- the kept ``> 2.0x => high-cost-analysis``
  kill (§3.5 R3).
* :func:`compute_match_ledger` -- the per-arm ledger (params, optimizer-state
  slots, REALIZED FLOPs, wall-clock, batch policy, skip-flag) and the
  matched-within-tolerance verdict (§3.7).
* :func:`routing_training_savings` -- a training-savings claim is licensed ONLY
  if a MEASURED reduction with masked anchor layers actually skipped is recorded
  (skip-flag true); otherwise it is forbidden (§3.5 R3, two-sided honesty).
* :func:`gate1b_deployability` -- the deployability ratio ``R / R*`` gate.

Because every arm shares the identical ``r_0`` substrate on ``L\\A``, the same
``R_tot`` on ``A``, and the same ``capacity_match`` map, the substrate
contributes equally to every arm's ledger and the match reduces to the anchor
allocation (§3.7). Equal rank alone is explicitly NOT equal compute.

DO-NOT-RUN: pure stdlib; no model load, no FLOP measurement, no server call. The
caller supplies the already-MEASURED ledger numbers (none are fabricated here).
``server.authorized`` stays ``false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "ArmLedger",
    "LedgerMatchResult",
    "ExtractionParity",
    "SavingsClaim",
    "DeployabilityGate",
    "gate1_multiplier",
    "extraction_parity_check",
    "compute_match_ledger",
    "routing_training_savings",
    "gate1b_deployability",
    "EXTRACTION_PARITY_KILL",
]

# §3.5 R3: extraction parity kill threshold (kept from v4).
EXTRACTION_PARITY_KILL = 2.0


# ---------------------------------------------------------------------------
# Gate 1: extraction-parity multiplier
# ---------------------------------------------------------------------------
def gate1_multiplier(method_extraction_cost: float, baseline_extraction_cost: float) -> float:
    """Extraction-cost multiplier ``method / cheapest-baseline`` (§3.5 R3).

    A multiplier of ``m`` means the proposed method's *extraction* (trajectory /
    activation harvesting) costs ``m`` times the cheapest baseline's extraction.
    """

    if baseline_extraction_cost <= 0:
        raise ValueError("baseline_extraction_cost must be positive")
    if method_extraction_cost < 0:
        raise ValueError("method_extraction_cost must be non-negative")
    return method_extraction_cost / baseline_extraction_cost


@dataclass(frozen=True)
class ExtractionParity:
    """Result of the extraction-parity kill check (§3.5 R3)."""

    multiplier: float
    kill_threshold: float
    high_cost_analysis: bool  # multiplier > threshold => the result is a high-cost analysis


def extraction_parity_check(
    method_extraction_cost: float,
    baseline_extraction_cost: float,
    *,
    kill_threshold: float = EXTRACTION_PARITY_KILL,
) -> ExtractionParity:
    """Kept ``> 2.0x => high-cost-analysis`` kill (§3.5 R3).

    The flag does NOT suppress the analysis; it labels the result as a high-cost
    analysis so any efficiency wording is honest about the extraction overhead.
    """

    mult = gate1_multiplier(method_extraction_cost, baseline_extraction_cost)
    return ExtractionParity(
        multiplier=mult,
        kill_threshold=kill_threshold,
        high_cost_analysis=mult > kill_threshold,
    )


# ---------------------------------------------------------------------------
# Compute-match ledger (§3.7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmLedger:
    """Per-arm measured compute ledger (§3.7).

    All fields are MEASURED quantities supplied by the caller (run-later); this
    module performs no measurement. ``skip_flag`` records whether masked anchor
    layers' adapters are actually excluded from the backward pass.
    """

    arm: str
    param_count: int
    optimizer_slots: int
    realized_flops: float            # measured, NOT rank-inferred
    wall_clock_s: float
    batch_policy: str
    skip_flag: bool
    total_rank_A: int                # sum_{l in A} r_l (must equal R_tot)
    r0_substrate_rank: int           # rank on each L\A layer (shared, cancels)


@dataclass(frozen=True)
class LedgerMatchResult:
    """Verdict of the compute-match ledger comparison (§3.7)."""

    matched: bool
    tolerance: float
    rank_conserved: bool
    mismatched_fields: tuple[str, ...]
    detail: Mapping[str, float] = field(default_factory=dict)


def _within(a: float, b: float, tol: float) -> bool:
    """Relative-tolerance equality ``|a - b| <= tol * max(|a|, |b|, 1)``."""

    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


def compute_match_ledger(
    method: ArmLedger,
    comparator: ArmLedger,
    *,
    tolerance: float,
    total_rank: int,
) -> LedgerMatchResult:
    """Compare two arms' ledgers for the §3.7 matched-budget verdict.

    "Matched budget" requires equality within ``tolerance`` on **params,
    optimizer slots, and realized FLOPs**, plus the rank invariant
    ``sum_{l in A} r_l = R_tot`` for both arms (with the shared ``r_0`` substrate
    cancelling). Any arm that cannot match exactly is labeled
    capacity-unmatched diagnostic by the caller, never the headline.
    """

    mismatches: list[str] = []
    detail: dict[str, float] = {}
    for name, a, b in (
        ("param_count", method.param_count, comparator.param_count),
        ("optimizer_slots", method.optimizer_slots, comparator.optimizer_slots),
        ("realized_flops", method.realized_flops, comparator.realized_flops),
    ):
        ok = _within(float(a), float(b), tolerance)
        detail[name] = abs(float(a) - float(b))
        if not ok:
            mismatches.append(name)

    rank_conserved = (
        method.total_rank_A == total_rank and comparator.total_rank_A == total_rank
    )
    if not rank_conserved:
        mismatches.append("total_rank_A")

    return LedgerMatchResult(
        matched=not mismatches,
        tolerance=tolerance,
        rank_conserved=rank_conserved,
        mismatched_fields=tuple(mismatches),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Two-sided honest training-savings claim (§3.5 R3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SavingsClaim:
    """Two-sided honest training-savings verdict (§3.5 R3)."""

    licensed: bool
    flop_reduction: float | None
    wall_clock_reduction: float | None
    reason: str


def routing_training_savings(
    method: ArmLedger,
    dense_baseline: ArmLedger,
) -> SavingsClaim:
    """A training-savings claim is licensed ONLY with a MEASURED reduction AND a
    true skip-flag (§3.5 R3).

    If the masked anchor layers are not actually skipped (``skip_flag`` false),
    or the measured FLOPs/wall-clock are not reduced, the savings claim is
    **forbidden** and only the extraction cost is reported. This enforces the
    two-sided honesty of Gate R3 (no "savings" from un-skipped masked layers).
    """

    if not method.skip_flag:
        return SavingsClaim(
            licensed=False,
            flop_reduction=None,
            wall_clock_reduction=None,
            reason="skip_flag false: masked anchor layers not actually skipped; "
            "savings claim forbidden",
        )
    flop_reduction = dense_baseline.realized_flops - method.realized_flops
    wall_reduction = dense_baseline.wall_clock_s - method.wall_clock_s
    measured_reduction = flop_reduction > 0.0
    if not measured_reduction:
        return SavingsClaim(
            licensed=False,
            flop_reduction=flop_reduction,
            wall_clock_reduction=wall_reduction,
            reason="no measured FLOP reduction; savings claim forbidden",
        )
    return SavingsClaim(
        licensed=True,
        flop_reduction=flop_reduction,
        wall_clock_reduction=wall_reduction,
        reason="measured FLOP reduction with masked anchor layers skipped",
    )


# ---------------------------------------------------------------------------
# Gate 1b: deployability ratio
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeployabilityGate:
    """Deployability verdict ``R / R*`` (gate 1b)."""

    ratio: float
    target_ratio: float
    deployable: bool


def gate1b_deployability(
    realized_metric: float,
    reference_metric: float,
    *,
    target_ratio: float,
) -> DeployabilityGate:
    """Gate 1b: the realized/reference deployability ratio must clear ``target_ratio``.

    ``R`` is the realized deployment metric (e.g. retained target accuracy under
    the routed model) and ``R*`` the reference; deployability requires
    ``R / R* >= target_ratio``.
    """

    if reference_metric <= 0:
        raise ValueError("reference_metric (R*) must be positive")
    ratio = realized_metric / reference_metric
    return DeployabilityGate(
        ratio=ratio,
        target_ratio=target_ratio,
        deployable=ratio >= target_ratio,
    )

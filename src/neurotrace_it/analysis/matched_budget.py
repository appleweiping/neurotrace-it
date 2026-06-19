"""Gate R2 -- compute-matched method win (REDESIGN_v5 §3.5 R2, §3.7).

ADDITIVE, DO-NOT-RUN. At a MEASURED compute-matched budget (§3.7 ledger),
LATTICE-R's ``U_train`` must beat the stronger full-``L`` NAIT variant AND every
baseline by the locked margins. The "beats every comparator" claim is, like R1,
an INTERSECTION-UNION alternative decided by the SAME IUT (reject iff each
comparator's per-contrast margin test clears ``delta_target`` at the marginal
level ``alpha`` -- no penalty across comparators, Eq. 7-IUT). The remaining R2
sub-claims (relative / retention / hallucination / cost) are SINGLE margin tests:
a lower bound above ``delta`` for "win" margins; an UPPER bound below the ceiling
for the non-inferiority drift/cost margins.

This module reuses the per-contrast bootstrap-``t`` and the IUT of
:mod:`neurotrace_it.analysis.routing_intervention`, and asserts the
``compute_match_ledger`` equality (§3.7) plus the rank invariant
``sum_{l in A} r_l = R_tot`` with ``L\\A = r_0``.

DO-NOT-RUN: pure stdlib; the caller supplies the already-measured per-seed
contrasts and ledgers (run-later). ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .routing_intervention import (
    ContrastBound,
    IUTDecision,
    bootstrap_t_lower_bound,
    iut_decision,
)
from ..cost_model import ArmLedger, LedgerMatchResult, compute_match_ledger
from ..layer_function import capacity_match

__all__ = [
    "MarginTest",
    "R2TargetResult",
    "R2Result",
    "single_margin_test",
    "r2_target_iut",
    "assert_capacity_matched",
    "gate_r2",
]


@dataclass(frozen=True)
class MarginTest:
    """A single-contrast margin test (Eq. 7-Bt)."""

    name: str
    gap_hat: float
    bound: float                  # lower bound (win) or upper bound (non-inferiority)
    threshold: float              # delta (win) or ceiling (drift/cost)
    direction: str                # "win" (lower>delta) | "non_inferiority" (upper<ceiling)
    passes: bool


def single_margin_test(
    diffs: Sequence[float],
    *,
    alpha: float,
    threshold: float,
    direction: str,
    n_boot: int,
    boot_seed: int,
    name: str = "",
) -> MarginTest:
    """One single-contrast margin test (Eq. 7-Bt).

    ``direction == "win"``: pass iff the one-sided LOWER bound exceeds
    ``threshold`` (delta). ``direction == "non_inferiority"``: pass iff the
    one-sided UPPER bound is below the ceiling ``threshold`` (drift/cost). The
    upper bound is obtained by negating the differences and re-using the
    lower-bound machinery (symmetry of the studentized bootstrap).
    """

    if direction == "win":
        b = bootstrap_t_lower_bound(
            diffs, alpha=alpha, margin=threshold, n_boot=n_boot, boot_seed=boot_seed
        )
        return MarginTest(
            name=name,
            gap_hat=b.gap_hat,
            bound=b.lower_bound,
            threshold=threshold,
            direction=direction,
            passes=b.lower_bound > threshold,
        )
    if direction == "non_inferiority":
        # Upper bound on the drift/cost gap = -(lower bound on the negated gap).
        neg = [-float(d) for d in diffs]
        b = bootstrap_t_lower_bound(
            neg, alpha=alpha, margin=-threshold, n_boot=n_boot, boot_seed=boot_seed
        )
        upper = -b.lower_bound
        return MarginTest(
            name=name,
            gap_hat=-b.gap_hat,
            bound=upper,
            threshold=threshold,
            direction=direction,
            passes=upper < threshold,
        )
    raise ValueError(f"unknown direction {direction!r}; use 'win' or 'non_inferiority'")


@dataclass(frozen=True)
class R2TargetResult:
    """IUT over the baseline set for the R2-target "beats every comparator" claim."""

    decision: IUTDecision


def r2_target_iut(
    diffs_per_comparator: Mapping[str, Sequence[float]],
    *,
    alpha: float,
    delta_target: float,
    n_boot: int,
    boot_seed: int,
) -> R2TargetResult:
    """R2-target as an IUT over the WHOLE baseline set (Eq. 7-IUT).

    Reject the union null iff each comparator's per-contrast lower bound clears
    ``delta_target`` at the marginal level ``alpha`` -- no penalty across
    comparators (identical construction to R1, with the baseline comparators in
    place of the controls).
    """

    decision = iut_decision(
        diffs_per_comparator,
        alpha=alpha,
        margin=delta_target,
        n_boot=n_boot,
        boot_seed=boot_seed,
        controls=tuple(diffs_per_comparator.keys()),
    )
    return R2TargetResult(decision=decision)


def assert_capacity_matched(
    method: ArmLedger,
    comparator: ArmLedger,
    *,
    tolerance: float,
    total_rank: int,
    r_max: int,
    anchors: Sequence[int],
    method_mask: Mapping[int, int],
    comparator_mask: Mapping[int, int],
) -> LedgerMatchResult:
    """Assert the §3.7 ledger match AND the rank invariant for both arms.

    Verifies ``sum_{l in A} capacity_match(mask) = R_tot`` for each arm (so the
    ``r_0`` substrate on ``L\\A`` cancels) and the compute-match ledger equality
    within ``tolerance``.
    """

    for label, mask in (("method", method_mask), ("comparator", comparator_mask)):
        ranks = capacity_match(mask, total_rank, r_max, anchors=anchors)
        if sum(ranks.values()) != total_rank:
            raise AssertionError(f"{label} mask does not conserve R_tot under capacity_match")

    return compute_match_ledger(
        method, comparator, tolerance=tolerance, total_rank=total_rank
    )


@dataclass(frozen=True)
class R2Result:
    """Aggregate Gate R2 verdict (§3.5 R2)."""

    target: R2TargetResult
    sub_claims: Mapping[str, MarginTest]
    ledger_matched: bool
    sub_claims_pass: bool            # True iff EVERY relative/retention/hallucination/cost sub-claim passes
    method_win: bool                 # licensed only if target IUT rejects AND ledger matched AND every sub-claim passes


def gate_r2(
    diffs_per_comparator: Mapping[str, Sequence[float]],
    sub_claim_diffs: Mapping[str, tuple[Sequence[float], float, str]],
    ledger: LedgerMatchResult,
    *,
    alpha: float,
    delta_target: float,
    n_boot: int,
    boot_seed: int,
) -> R2Result:
    """Decide Gate R2: target IUT + single-contrast sub-claims + ledger match.

    ``sub_claim_diffs`` maps each sub-claim name -> ``(diffs, threshold,
    direction)``. A method-win is licensed ONLY if ALL of the following hold:

      * the target IUT rejects (LATTICE-R beats EVERY comparator, §3.5 R2);
      * EVERY R2 margin sub-claim passes -- the relative-win margin (``win``
        direction, lower bound > ``delta_rel``) AND each non-inferiority drift /
        cost margin (``non_inferiority`` direction, upper bound < ceiling) for
        retention, hallucination, and cost (§3.5 R2: R2 FAILS if ANY sub-claim
        fails, so a single failing sub-claim must veto the headline win);
      * the compute-match ledger is matched (§3.7).

    Failing R2 yields ``no_method_win_claim`` (the caller's failure action).
    """

    target = r2_target_iut(
        diffs_per_comparator,
        alpha=alpha,
        delta_target=delta_target,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    sub: dict[str, MarginTest] = {}
    for name, (diffs, threshold, direction) in sub_claim_diffs.items():
        # Distinct deterministic bootstrap stream per sub-claim (boot_seed XOR
        # sha256(name)). MUST be reproducible across processes / PYTHONHASHSEED, so
        # we derive the per-sub-claim seed from a SHA-256 digest -- NOT Python's
        # salted builtin ``hash()`` -- mirroring routing_intervention.iut_decision
        # (line 242), layer_function.stable_example_hash, and closed_testing.
        c_seed = boot_seed ^ int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
        sub[name] = single_margin_test(
            diffs,
            alpha=alpha,
            threshold=threshold,
            direction=direction,
            n_boot=n_boot,
            boot_seed=c_seed,
            name=name,
        )
    sub_claims_pass = all(t.passes for t in sub.values())
    method_win = target.decision.reject and sub_claims_pass and ledger.matched
    return R2Result(
        target=target,
        sub_claims=sub,
        ledger_matched=ledger.matched,
        sub_claims_pass=sub_claims_pass,
        method_win=method_win,
    )

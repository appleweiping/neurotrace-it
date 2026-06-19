"""Gate R1 -- routing policy-value coherence via the INTERSECTION-UNION TEST.

REDESIGN_v5 §3.5, §4.2, DO-NOT-RUN. The headline routing claim is the composite
"``pi_psi`` beats EVERY control by margin ``delta_R1``", whose null is the UNION
of the per-control nulls (an intersection-union structure, Berger 1982):

    H0^{R1} = union_c { g_c <= delta_R1 }   (fails to beat at least one control)
    H1^{R1} = inter_c { g_c >  delta_R1 }   (beats EVERY control by > delta_R1)

The valid decision rule is the IUT: reject iff EACH per-control margin test
rejects at the MARGINAL level ``alpha`` -- NO multiplicity tightening, NO shared
min-over-controls quantile (that was the round-2 wrong-direction error). Each
per-contrast lower bound ``L_c(alpha)`` is an ASYMPTOTIC paired studentized
bootstrap-``t`` bound (Eq. 7-Bt); we claim asymptotic validity only (the
sign-flip "exact finite-S" claim is withdrawn, R3-B3).

The simultaneous lower bound on the "beats-all" margin ``min_c g_c`` is the
MINIMUM of the per-contrast marginal-``alpha`` lower bounds (Eq. 7-IUT-CI):
``L_{R1} = min_c L_c(alpha)``, and we reject iff ``L_{R1} > delta_R1``.

This module also exposes the six-arm harness spec (the arm names + the
deterministic per-control mask via :mod:`neurotrace_it.layer_function`) and the
pool-conditional seed-mean policy-value estimator ``V_hat(pi) = mean_s
U_train,s(pi)``. The treated unit is the WHOLE training run, so there is no
per-example potential outcome and no cross-example SUTVA assumption.

DO-NOT-RUN: pure stdlib; the caller supplies the already-measured per-seed
``U_train`` values for each arm (run-later); the inference arithmetic is testable
build-now / run-later. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "R1_ARMS",
    "ContrastBound",
    "IUTDecision",
    "policy_value_seed_mean",
    "paired_differences",
    "bootstrap_t_lower_bound",
    "iut_decision",
    "simultaneous_lower_bound",
]

# The five controls the policy under test must beat (the IUT components).
CONTROLS: tuple[str, ...] = ("unif", "shuf", "rand", "global", "ada")
# All six R1 arms (the policy under test plus the five controls).
R1_ARMS: tuple[str, ...] = ("psi",) + CONTROLS


def policy_value_seed_mean(u_train_per_seed: Sequence[float]) -> float:
    """Pool-conditional policy-value estimate ``V_hat(pi) = mean_s U_train,s(pi)`` (Eq. 6-2).

    Unbiased for the pool-conditional seed-expectation ``V(pi)`` at the locked
    pools; the bootstrap-``t`` (below) resamples exactly the seed randomness it
    averages.
    """

    seeds = [float(u) for u in u_train_per_seed]
    if not seeds:
        raise ValueError("need at least one seed's U_train")
    return math.fsum(seeds) / len(seeds)


def paired_differences(
    u_psi_per_seed: Sequence[float],
    u_control_per_seed: Sequence[float],
) -> list[float]:
    """Per-seed PAIRED differences ``d_c,s = U_train,s(pi_psi) - U_train,s(pi_c)`` (§4.2).

    Same seed across arms removes seed variance from the contrast; the result is
    the cluster-robust (over seeds) input to the bootstrap-``t``.
    """

    if len(u_psi_per_seed) != len(u_control_per_seed):
        raise ValueError("psi and control must have the same number of paired seeds")
    if not u_psi_per_seed:
        raise ValueError("need at least one paired seed")
    return [float(a) - float(b) for a, b in zip(u_psi_per_seed, u_control_per_seed)]


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _std_error(xs: Sequence[float]) -> float:
    """Seed standard error ``sd / sqrt(n)`` with the unbiased (n-1) variance."""

    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    var = math.fsum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _bootstrap_index_stream(seed: int, n: int, draws: int) -> list[int]:
    """Deterministic resample-with-replacement indices in ``[0, n)`` from ``seed``.

    Uses a SHA-256 byte stream so the bootstrap is reproducible across processes
    (no salted ``hash()``, no ``random`` global-state dependence).
    """

    out: list[int] = []
    counter = 0
    while len(out) < draws:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        counter += 1
        for k in range(0, len(digest), 8):
            if len(out) >= draws:
                break
            out.append(int.from_bytes(digest[k : k + 8], "big") % n)
    return out


@dataclass(frozen=True)
class ContrastBound:
    """Per-contrast asymptotic bootstrap-``t`` lower bound (Eq. 7-Bt)."""

    control: str
    gap_hat: float            # g_hat_c = mean_s d_c,s
    se: float                 # seed standard error of d_c,s
    lower_bound: float        # L_c(alpha)
    margin: float             # delta_R1
    rejects: bool             # L_c(alpha) > delta_R1


def bootstrap_t_lower_bound(
    diffs: Sequence[float],
    *,
    alpha: float,
    margin: float,
    n_boot: int,
    boot_seed: int,
    control: str = "",
) -> ContrastBound:
    """One-sided asymptotic paired studentized bootstrap-``t`` lower bound (Eq. 7-Bt).

    For per-seed paired differences ``d_c,s`` with gap ``g_hat_c = mean(d)`` and
    seed standard error ``se``, resample seeds with replacement ``n_boot`` times,
    recompute ``t*^(b) = (g_hat*^(b) - g_hat) / se*^(b)``, and form

        L_c(alpha) = g_hat_c - t*_{1-alpha} . se ,

    rejecting the per-control null iff ``L_c(alpha) > margin``. Asymptotically
    valid (studentized bootstrap); NOT claimed exact at finite ``S`` (R3-B3).
    """

    diffs = [float(d) for d in diffs]
    n = len(diffs)
    g_hat = _mean(diffs)
    se = _std_error(diffs)

    if se == 0.0 or n < 2:
        # Degenerate: no seed variance. The lower bound collapses to the point
        # estimate; reject iff the gap itself clears the margin.
        return ContrastBound(
            control=control,
            gap_hat=g_hat,
            se=se,
            lower_bound=g_hat,
            margin=margin,
            rejects=g_hat > margin,
        )

    idx = _bootstrap_index_stream(boot_seed, n, n * n_boot)
    t_stars: list[float] = []
    for b in range(n_boot):
        sample = [diffs[idx[b * n + j]] for j in range(n)]
        g_star = _mean(sample)
        se_star = _std_error(sample)
        if se_star == 0.0:
            # Degenerate resample (all identical): t* is 0 by convention.
            t_stars.append(0.0)
        else:
            t_stars.append((g_star - g_hat) / se_star)

    t_stars.sort()
    # The (1-alpha) quantile of the bootstrap t-distribution.
    q_index = min(len(t_stars) - 1, max(0, int(math.ceil((1.0 - alpha) * len(t_stars))) - 1))
    t_quantile = t_stars[q_index]
    lower = g_hat - t_quantile * se
    return ContrastBound(
        control=control,
        gap_hat=g_hat,
        se=se,
        lower_bound=lower,
        margin=margin,
        rejects=lower > margin,
    )


@dataclass(frozen=True)
class IUTDecision:
    """Intersection-union decision for the "beats all" composite null (Eq. 7-IUT)."""

    reject: bool                                   # reject H0^{R1} iff every contrast rejects
    margin: float
    alpha: float
    simultaneous_lower_bound: float                # L_{R1} = min_c L_c(alpha)
    binding_control: str                           # the control with the smallest L_c
    per_contrast: tuple[ContrastBound, ...]


def simultaneous_lower_bound(bounds: Sequence[ContrastBound]) -> float:
    """``L_{R1} = min_c L_c(alpha)`` -- the min of per-contrast marginal-alpha bounds (Eq. 7-IUT-CI)."""

    if not bounds:
        raise ValueError("need at least one contrast bound")
    return min(b.lower_bound for b in bounds)


def iut_decision(
    diffs_per_control: Mapping[str, Sequence[float]],
    *,
    alpha: float,
    margin: float,
    n_boot: int,
    boot_seed: int,
    controls: Sequence[str] = CONTROLS,
) -> IUTDecision:
    """INTERSECTION-UNION decision: reject iff EVERY control clears the margin (Eq. 7-IUT).

    Each control is tested at the MARGINAL level ``alpha`` -- no 1/J split, no
    shared min-over-controls quantile. The simultaneous "beats-all" lower bound is
    ``min_c L_c(alpha)`` and we reject iff it exceeds ``margin``. By Prop. P1-IUT
    the IUT has size <= alpha at the least-favorable configuration (one
    ``g_c = margin``, the rest large), exactly where a min-over-controls quantile
    would inflate type-I error.
    """

    bounds: list[ContrastBound] = []
    for c in controls:
        if c not in diffs_per_control:
            raise KeyError(f"missing paired differences for control {c!r}")
        # Distinct deterministic bootstrap stream per control (boot_seed XOR name).
        c_seed = boot_seed ^ int.from_bytes(hashlib.sha256(c.encode()).digest()[:8], "big")
        bounds.append(
            bootstrap_t_lower_bound(
                diffs_per_control[c],
                alpha=alpha,
                margin=margin,
                n_boot=n_boot,
                boot_seed=c_seed,
                control=c,
            )
        )

    reject = all(b.rejects for b in bounds)
    l_r1 = simultaneous_lower_bound(bounds)
    binding = min(bounds, key=lambda b: b.lower_bound).control
    return IUTDecision(
        reject=reject and (l_r1 > margin),
        margin=margin,
        alpha=alpha,
        simultaneous_lower_bound=l_r1,
        binding_control=binding,
        per_contrast=tuple(bounds),
    )

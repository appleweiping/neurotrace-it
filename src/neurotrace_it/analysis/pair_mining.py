"""Matched-endpoint / divergent-curvature pair mining (DIAGNOSTIC, not causal).

Implements the §3.2 recipe of ``docs/redesign/REDESIGN_v4.md``: mine
*naturally-occurring* pool pairs ``(x, x')`` with near-identical endpoint
signatures ``||phi_end(x) - phi_end(x')||_2 <= tau_end`` but **divergent
curvature** ``|kappa(x) - kappa(x')|`` in the top decile, additionally balanced on
length / difficulty / family via **coarsened exact matching (CEM)**. The mined set
is a STRONG corroborating DIAGNOSTIC for the co-primary regression, never
standalone causal proof (fix f2): the reported paired difference is "consistent
with residual trajectory information after observed-covariate matching", and the
residual covariate imbalance is reported alongside.

Recipe (§3.2)
-------------
1. Bucket by capability family and coarse final-answer equivalence (so paired
   examples "teach the same endpoint").
2. Within a bucket, find nearest neighbours in ``phi_end`` and keep pairs with
   ``||phi_end(x) - phi_end(x')||_2 <= tau_end`` (``tau_end`` = the 1st percentile
   of within-bucket pairwise endpoint distance, persisted).
3. Among those, keep pairs whose ``|kappa(x) - kappa(x')|`` is in the top decile.
4. Coarsened exact matching on length / difficulty / family; report residual
   standardized-mean-difference imbalance; drop pairs that cannot be balanced.
5. Target ``n = 300`` retained pairs; report achieved ``n``.

DO-NOT-RUN: pure stdlib; no model load, no server call, no training.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

Vector = Sequence[float]

__all__ = [
    "DEFAULT_TAU_PERCENTILE",
    "DEFAULT_CURVATURE_TOP_DECILE",
    "DEFAULT_TARGET_PAIRS",
    "PairCandidate",
    "MatchedPair",
    "PairMiningResult",
    "endpoint_distance",
    "mine_matched_endpoint_pairs",
    "paired_margin_test",
    "power_for_pairs",
]

DEFAULT_TAU_PERCENTILE = 1.0       # 1st percentile of endpoint distance (tau_end)
DEFAULT_CURVATURE_TOP_DECILE = 0.10  # keep top-decile |delta kappa| pairs
DEFAULT_TARGET_PAIRS = 300         # target retained pairs (~100/family)


@dataclass(frozen=True)
class PairCandidate:
    """One pool candidate for pair mining.

    Attributes
    ----------
    example_id:
        Candidate id.
    phi_end:
        The endpoint signature ``phi_end`` (Eq. 1) used for matching.
    curvature:
        A scalar curvature summary ``kappa(x)`` (e.g. mean over layers of
        :func:`neurotrace_it.trajectory.trajectory_curvature`).
    family:
        Capability family bucket (``"math"`` / ``"code"`` / ``"multihop_qa"``).
    answer_key:
        Coarse final-answer-equivalence key (exact-match answer / normalized
        output / AST hash) so paired examples teach the same endpoint.
    length:
        Token/step length (a CEM covariate).
    difficulty:
        Difficulty score (a CEM covariate).
    outcome:
        Optional realized retention-adjusted fine-tuning outcome (filled at
        run-time; ``None`` in the build-now design).
    """

    example_id: str
    phi_end: tuple[float, ...]
    curvature: float
    family: str
    answer_key: str
    length: float = 0.0
    difficulty: float = 0.0
    outcome: float | None = None


@dataclass(frozen=True)
class MatchedPair:
    """A retained matched-endpoint / divergent-curvature pair.

    The CEM covariates (``first_length``/``second_length``/``first_difficulty``/
    ``second_difficulty``) and ``answer_key`` travel with the pair so that the
    residual-imbalance SMD is computed over **exactly the retained pairs** after
    the deterministic sort + trim to ``target_n`` (rather than over a positional
    prefix of the pre-sort acceptance order).
    """

    first_id: str
    second_id: str
    endpoint_distance: float
    curvature_gap: float
    family: str
    answer_key: str = ""
    first_length: float = 0.0
    second_length: float = 0.0
    first_difficulty: float = 0.0
    second_difficulty: float = 0.0


@dataclass(frozen=True)
class PairMiningResult:
    """Mined pairs + the persisted tolerance and reported covariate imbalance."""

    pairs: tuple[MatchedPair, ...]
    tau_end: float
    curvature_gap_threshold: float
    covariate_imbalance: Mapping[str, float]  # standardized mean difference per covariate
    achieved_n: int
    target_n: int
    dropped_for_balance: int = 0
    # Per-(family, answer_key) endpoint tolerance, keyed by "family\x1fanswer_key"
    # so each (family, answer_key) bucket keeps its OWN tau_end (a single string
    # "family" key would collapse distinct answer-key buckets and lose tolerances).
    bucket_taus: Mapping[str, float] = field(default_factory=dict)


def endpoint_distance(a: Vector, b: Vector) -> float:
    """Euclidean ``||phi_end(x) - phi_end(x')||_2``."""

    if len(a) != len(b):
        raise ValueError(f"phi_end dim mismatch {len(a)} vs {len(b)}")
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in ``[0, 100]``)."""

    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _coarsen(value: float, bin_width: float) -> int:
    if bin_width <= 0:
        return 0
    return int(math.floor(value / bin_width))


def _standardized_mean_difference(
    group_a: Sequence[float], group_b: Sequence[float]
) -> float:
    """Absolute standardized mean difference between two covariate samples."""

    if not group_a or not group_b:
        return 0.0
    mean_a = statistics.fmean(group_a)
    mean_b = statistics.fmean(group_b)
    var_a = statistics.pvariance(group_a) if len(group_a) > 1 else 0.0
    var_b = statistics.pvariance(group_b) if len(group_b) > 1 else 0.0
    pooled = math.sqrt((var_a + var_b) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return abs(mean_a - mean_b) / pooled


def mine_matched_endpoint_pairs(
    candidates: Sequence[PairCandidate],
    *,
    tau_percentile: float = DEFAULT_TAU_PERCENTILE,
    curvature_top_decile: float = DEFAULT_CURVATURE_TOP_DECILE,
    target_n: int = DEFAULT_TARGET_PAIRS,
    length_bin_width: float = 1.0,
    difficulty_bin_width: float = 1.0,
) -> PairMiningResult:
    """Mine matched-endpoint / divergent-curvature pairs (§3.2 recipe).

    Steps 1-5 of §3.2. ``tau_end`` is computed PER BUCKET as the
    ``tau_percentile``-th percentile of within-bucket pairwise endpoint distance
    and persisted; a single representative (median over buckets) ``tau_end`` is
    also returned for reporting. Coarsened exact matching on length/difficulty/
    family balances observed confounders; the residual standardized-mean-
    difference imbalance is reported per covariate.

    Returns
    -------
    PairMiningResult
        Retained pairs + the persisted tolerances + the reported imbalance.
    """

    # Step 1: bucket by (family, coarse answer key).
    buckets: dict[tuple[str, str], list[PairCandidate]] = {}
    for candidate in candidates:
        buckets.setdefault((candidate.family, candidate.answer_key), []).append(candidate)

    bucket_taus: dict[str, float] = {}
    accepted: list[MatchedPair] = []
    dropped_for_balance = 0

    for (family, answer_key), members in buckets.items():
        if len(members) < 2:
            continue
        # Step 2: within-bucket pairwise endpoint distances -> tau_end (percentile).
        distances: list[float] = []
        pair_records: list[tuple[int, int, float, float]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                dist = endpoint_distance(members[i].phi_end, members[j].phi_end)
                gap = abs(members[i].curvature - members[j].curvature)
                distances.append(dist)
                pair_records.append((i, j, dist, gap))
        tau_end = _percentile(distances, tau_percentile)
        # Key tau by the FULL (family, answer_key) bucket identity. A bare
        # ``f"{family}"`` key collapses every answer-key bucket of a family onto a
        # single slot (``setdefault`` keeps only the first), silently dropping the
        # tolerances of the other answer-key buckets. The "\x1f" separator is the
        # same record-separator used elsewhere so the two parts cannot collide.
        bucket_key = f"{family}\x1f{answer_key}"
        bucket_taus[bucket_key] = tau_end

        # Step 2 (cont.): keep near-endpoint pairs.
        near = [rec for rec in pair_records if rec[2] <= tau_end]
        if not near:
            continue
        # Step 3: keep top-decile |delta kappa| among the near pairs.
        gaps = [rec[3] for rec in near]
        gap_threshold = _percentile(gaps, 100.0 * (1.0 - curvature_top_decile))
        divergent = [rec for rec in near if rec[3] >= gap_threshold]

        # Step 4: coarsened exact matching on length/difficulty (family already fixed).
        for i, j, dist, gap in divergent:
            first = members[i]
            second = members[j]
            same_cell = (
                _coarsen(first.length, length_bin_width) == _coarsen(second.length, length_bin_width)
                and _coarsen(first.difficulty, difficulty_bin_width)
                == _coarsen(second.difficulty, difficulty_bin_width)
            )
            if not same_cell:
                dropped_for_balance += 1
                continue
            # Carry the CEM covariates ON the pair so the residual-imbalance SMD is
            # computed over exactly the pairs that SURVIVE the sort+trim below.
            accepted.append(
                MatchedPair(
                    first_id=first.example_id,
                    second_id=second.example_id,
                    endpoint_distance=dist,
                    curvature_gap=gap,
                    family=family,
                    answer_key=answer_key,
                    first_length=first.length,
                    second_length=second.length,
                    first_difficulty=first.difficulty,
                    second_difficulty=second.difficulty,
                )
            )

    # Deterministic ordering, then trim to target_n by largest curvature gap.
    accepted.sort(key=lambda pair: (-pair.curvature_gap, pair.first_id, pair.second_id))
    retained = accepted[:target_n] if target_n >= 0 else accepted

    # Imbalance is reported over the RETAINED pairs (post sort+trim), reading the
    # covariates carried on each pair -- not a positional prefix of the unsorted
    # acceptance lists, which would mismatch the retained set after re-sorting.
    covariate_imbalance = {
        "length": _standardized_mean_difference(
            [pair.first_length for pair in retained],
            [pair.second_length for pair in retained],
        ),
        "difficulty": _standardized_mean_difference(
            [pair.first_difficulty for pair in retained],
            [pair.second_difficulty for pair in retained],
        ),
    }
    representative_tau = (
        statistics.median(bucket_taus.values()) if bucket_taus else 0.0
    )
    representative_gap = (
        statistics.median([pair.curvature_gap for pair in retained]) if retained else 0.0
    )

    return PairMiningResult(
        pairs=tuple(retained),
        tau_end=representative_tau,
        curvature_gap_threshold=representative_gap,
        covariate_imbalance=covariate_imbalance,
        achieved_n=len(retained),
        target_n=target_n,
        dropped_for_balance=dropped_for_balance,
        bucket_taus=bucket_taus,
    )


def paired_margin_test(
    pairs: Sequence[MatchedPair],
    outcomes: Mapping[str, float],
    *,
    margin: float = 0.02,
) -> dict[str, float]:
    """Paired-difference diagnostic against the trajectory-endpoint margin (§3.2).

    Given realized per-example ``outcomes`` (filled at run-time; ``None`` in the
    build-now design), computes the mean absolute paired difference and a
    one-sample t-style statistic against ``margin``. This is a DIAGNOSTIC summary
    consumed by the §3.2 decision, never standalone causal evidence; it corroborates
    P1 (the co-primary regression) and is only kill-licensing in agreement with it.

    Returns a dict with ``mean_abs_diff``, ``n``, ``margin``, and (when ``n > 1``)
    ``t_stat`` and ``exceeds_margin`` (mean_abs_diff > margin).
    """

    diffs: list[float] = []
    for pair in pairs:
        if pair.first_id in outcomes and pair.second_id in outcomes:
            diffs.append(abs(outcomes[pair.first_id] - outcomes[pair.second_id]))
    n = len(diffs)
    result: dict[str, float] = {"n": float(n), "margin": float(margin)}
    if n == 0:
        result["mean_abs_diff"] = 0.0
        result["exceeds_margin"] = 0.0
        return result
    mean_abs_diff = statistics.fmean(diffs)
    result["mean_abs_diff"] = mean_abs_diff
    result["exceeds_margin"] = 1.0 if mean_abs_diff > margin else 0.0
    if n > 1:
        sd = statistics.stdev(diffs)
        if sd > 1e-12:
            result["t_stat"] = (mean_abs_diff - margin) / (sd / math.sqrt(n))
        else:
            result["t_stat"] = math.inf if mean_abs_diff > margin else 0.0
    return result


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def power_for_pairs(
    n_pairs: int,
    *,
    effect: float,
    sd_diff: float,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Achieved power of the §3.2 one-sided paired-margin test (§4.5 formula).

    Models the paired-difference test "mean paired |outcome| difference exceeds
    the ``margin``" as a one-sample, one-sided ``t``/``z`` test with true mean
    ``effect`` and per-pair sd ``sd_diff``. The non-centrality is
    ``ncp = sqrt(n) * (effect - margin) / sd_diff`` and the approximate power is
    ``Phi(ncp - z_{1-alpha})``. Used to (i) check the ``n = 300`` target gives
    adequate power for the registered ``effect``, and (ii) **re-power** a family
    that yields fewer balanced pairs than targeted (§3.2 step 6).

    Returns ``achieved_power``, ``ncp``, ``required_n`` (the smallest ``n`` that
    reaches 0.80 power for this effect), and the inputs.
    """

    if sd_diff <= 1e-12 or n_pairs < 1:
        return {
            "achieved_power": 1.0 if effect > margin else 0.0,
            "ncp": math.inf if effect > margin else 0.0,
            "required_n": 0.0,
            "n_pairs": float(n_pairs),
            "effect": effect,
            "margin": margin,
        }
    z_alpha = 1.6448536269514722 if abs(alpha - 0.05) < 1e-9 else _inv_norm(1.0 - alpha)
    standardized = (effect - margin) / sd_diff
    ncp = math.sqrt(n_pairs) * standardized
    power = _normal_cdf(ncp - z_alpha)

    required_n = math.inf
    if standardized > 0:
        z_power = 0.8416212335729143  # Phi^{-1}(0.80)
        required_n = ((z_alpha + z_power) / standardized) ** 2
    return {
        "achieved_power": power,
        "ncp": ncp,
        "required_n": math.ceil(required_n) if required_n != math.inf else math.inf,
        "n_pairs": float(n_pairs),
        "effect": effect,
        "margin": margin,
    }


def _inv_norm(p: float) -> float:
    """Minimal inverse-normal for non-default alpha (Beasley-Springer-Moro tail)."""

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Symmetric rational approximation; adequate for alpha in (0, 0.5).
    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    z = t - num / den
    return -z if p < 0.5 else z

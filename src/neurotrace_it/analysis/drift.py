"""Factuality calibrator + drift + the G6 precondition gate (§2.6, §4.4).

Implements the named pieces of ``docs/redesign/REDESIGN_v4.md`` §2.6 that gate the
factuality term ``lambda_f f_hat(x)`` of the selection utility (§2.8, Eq. 14):

* :class:`BrierCalibrator` -- a strictly-proper **Brier** calibrator over per-claim
  logits ``z_a``. Output ``q_a = sigma(z_a)`` (optionally affine-recalibrated
  ``sigma(scale * z_a + bias)``); scored by ``ell_Brier(q, y) = (q - y)^2``
  (Eq. 12). The population Brier risk is uniquely minimized at ``q = p`` -- this
  proves **calibration, not proxy validity** (which is exactly why proxy validity
  must be checked separately by the G6 gate below). This strictly-proper Brier
  rule is the whole of NeuroTrace-IT's factuality calibrator: it is what licenses
  the G6 scoring precondition that gates the ``lambda_f f_hat(x)`` term.
* :func:`factuality_drift` -- ``f_hat(x) = mean_a 1[q_a < c*]`` (§2.6): the fraction
  of an example's claims whose calibrated support probability falls below the
  calibrated threshold ``c*``.
* :func:`expected_calibration_error` -- the reliability **ECE** used by G6.
* :func:`pearson_r` / :func:`pearson_ci_lower` -- Pearson correlation + Fisher-z
  lower 95% CI bound, the second leg of G6.
* :func:`g6_factuality_gate` -- the G6 pathway: ``lambda_f`` may be non-zero ONLY
  if, on a held-out slice, **Spearman rho(f_hat, drift_eval) >= 0.3 AND Pearson
  r >= 0.3 with lower 95% CI > 0**, *and* reliability **ECE <= 0.1**. Fail =>
  ``lambda_f := 0`` and no factuality/safety claim (``on_fail: set_lambda_f_zero``,
  ``lattice_v4.yaml:77-81``).

Pure stdlib; build-now / run-later. No server call, no training, no model load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .outcome_y import _one_sided_z, spearman_rho

__all__ = [
    "G6_SPEARMAN_MIN",
    "G6_PEARSON_MIN",
    "G6_ECE_MAX",
    "BrierCalibrator",
    "G6Report",
    "factuality_drift",
    "expected_calibration_error",
    "pearson_r",
    "pearson_ci_lower",
    "g6_factuality_gate",
]

# Locked G6 thresholds (REDESIGN_v4 §4.4; configs/experiments/lattice_v4.yaml:77-81).
G6_SPEARMAN_MIN = 0.3   # Spearman rho(f_hat, drift_eval) >= 0.3
G6_PEARSON_MIN = 0.3    # Pearson r(f_hat, drift_eval) >= 0.3, lower 95% CI > 0
G6_ECE_MAX = 0.1        # reliability ECE <= 0.1


def _sigmoid(z: float) -> float:
    # Numerically stable logistic sigma(z).
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class BrierCalibrator:
    """A strictly-proper Brier calibrator over per-claim logits (§2.6).

    The calibrator maps a claim logit ``z_a`` to a support probability
    ``q_a = sigma(scale * z_a + bias)``. ``scale``/``bias`` default to the identity
    recalibration (``1.0``/``0.0``); a fitted affine recalibration can be passed in.
    ``threshold`` is the calibrated decision threshold ``c*`` used by
    :func:`factuality_drift`.

    Scoring is the proper Brier rule ``(q - y)^2`` (Eq. 12); the mean Brier score
    over a labelled reliability slice is :meth:`brier_score`.
    """

    scale: float = 1.0
    bias: float = 0.0
    threshold: float = 0.5

    def probability(self, logit: float) -> float:
        """``q_a = sigma(scale * z_a + bias)`` in ``[0, 1]``."""

        return _sigmoid(self.scale * float(logit) + self.bias)

    def probabilities(self, logits: Sequence[float]) -> list[float]:
        return [self.probability(z) for z in logits]

    def brier_score(self, logits: Sequence[float], labels: Sequence[int]) -> float:
        """Mean Brier loss ``mean_a (q_a - y_a)^2`` (Eq. 12) over a labelled slice."""

        if len(logits) != len(labels):
            raise ValueError("logits and labels must have equal length")
        if not logits:
            return 0.0
        probs = self.probabilities(logits)
        return math.fsum((q - float(y)) ** 2 for q, y in zip(probs, labels)) / len(probs)


def factuality_drift(
    claim_logits: Sequence[float],
    calibrator: BrierCalibrator,
    *,
    threshold: float | None = None,
) -> float:
    """``f_hat(x) = mean_a 1[q_a < c*]`` -- fraction of unsupported claims (§2.6).

    ``q_a = calibrator.probability(z_a)`` is the calibrated support probability of
    claim ``a``; ``c*`` is ``threshold`` (defaults to ``calibrator.threshold``).
    Returns ``0.0`` for an example with no claims (no evidence of factuality drift).
    """

    if not claim_logits:
        return 0.0
    c_star = calibrator.threshold if threshold is None else threshold
    probs = calibrator.probabilities(claim_logits)
    unsupported = sum(1 for q in probs if q < c_star)
    return unsupported / len(probs)


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    """Reliability ECE over ``n_bins`` equal-width probability bins (G6 leg).

    ``ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|`` -- the standard
    equal-width-binned expected calibration error. ``probs[i] in [0, 1]`` is the
    predicted support probability and ``labels[i] in {0, 1}`` the realized support.
    """

    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    n = len(probs)
    if n == 0:
        return 0.0
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_count = [0] * n_bins
    for p, y in zip(probs, labels):
        pp = min(max(float(p), 0.0), 1.0)
        # Right-closed last bin so p == 1.0 lands in the final bin.
        idx = min(int(pp * n_bins), n_bins - 1)
        bin_conf[idx] += pp
        bin_acc[idx] += float(y)
        bin_count[idx] += 1
    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        conf = bin_conf[b] / bin_count[b]
        acc = bin_acc[b] / bin_count[b]
        ece += (bin_count[b] / n) * abs(acc - conf)
    return ece


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson product-moment correlation; 0.0 for degenerate (no variance)."""

    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = math.fsum(x) / len(x)
    my = math.fsum(y) / len(y)
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.fsum((a - mx) ** 2 for a in x)
    vy = math.fsum((b - my) ** 2 for b in y)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    return cov / math.sqrt(vx * vy)


def pearson_ci_lower(r: float, n: int, *, level: float = 0.95) -> float:
    """One-sided lower 95% confidence bound on Pearson r (Fisher z, n >= 4).

    G6 asks "is r reliably > 0?", a one-sided question, so this uses the one-sided
    critical value z_{0.95} = 1.6448536269514722 (NOT the two-sided z_{0.975} =
    1.96). The returned value is a genuine lower 95% one-sided bound.
    """

    if n < 4 or abs(r) >= 1.0:
        return r
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = _one_sided_z(level)  # one-sided z_{level}; 1.6448536... at level=0.95
    return math.tanh(z - z_crit * se)


@dataclass(frozen=True)
class G6Report:
    """G6 factuality precondition outcome (§2.6, §4.4)."""

    spearman: float
    pearson: float
    pearson_ci_lower: float
    ece: float
    spearman_passes: bool
    pearson_passes: bool
    ece_passes: bool
    passes: bool
    lambda_f: float          # 0.0 unless the gate passes; else the requested weight


def g6_factuality_gate(
    f_hat: Sequence[float],
    drift_eval: Sequence[float],
    reliability_probs: Sequence[float],
    reliability_labels: Sequence[int],
    *,
    requested_lambda_f: float = 1.0,
    spearman_min: float = G6_SPEARMAN_MIN,
    pearson_min: float = G6_PEARSON_MIN,
    ece_max: float = G6_ECE_MAX,
    n_bins: int = 10,
    level: float = 0.95,
) -> G6Report:
    """The G6 factuality precondition gate -> ``lambda_f`` (§2.6, §4.4).

    ``lambda_f`` may be the ``requested_lambda_f`` ONLY when, on a held-out slice,
    ALL hold:
      (i)   Spearman ``rho(f_hat, drift_eval) >= spearman_min`` (0.3);
      (ii)  Pearson ``r(f_hat, drift_eval) >= pearson_min`` (0.3) with lower 95% CI
            > 0 (proxy *validity*, which the proper Brier score does NOT establish);
      (iii) reliability ``ECE <= ece_max`` (0.1) on ``(reliability_probs,
            reliability_labels)`` (proxy *calibration*).
    Fail => ``lambda_f := 0`` and no factuality/safety claim
    (``on_fail: set_lambda_f_zero``). The reliability slice (calibrator outputs vs
    realized support) is distinct from the ``(f_hat, drift_eval)`` validity slice.
    """

    rho = spearman_rho(f_hat, drift_eval)
    r = pearson_r(f_hat, drift_eval)
    r_lo = pearson_ci_lower(r, min(len(f_hat), len(drift_eval)), level=level)
    ece = expected_calibration_error(reliability_probs, reliability_labels, n_bins=n_bins)

    spearman_ok = rho >= spearman_min
    pearson_ok = (r >= pearson_min) and (r_lo > 0.0)
    ece_ok = ece <= ece_max
    passes = spearman_ok and pearson_ok and ece_ok
    return G6Report(
        spearman=rho,
        pearson=r,
        pearson_ci_lower=r_lo,
        ece=ece,
        spearman_passes=spearman_ok,
        pearson_passes=pearson_ok,
        ece_passes=ece_ok,
        passes=passes,
        lambda_f=requested_lambda_f if passes else 0.0,
    )

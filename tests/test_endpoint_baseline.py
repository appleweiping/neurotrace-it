"""Faithful NAIT reproduction contract (REDESIGN_v5 §2.2, §3.3; the Phase-A gap).

Formula-evaluation unit checks (NOT evidence) that pin the layerwise-NAIT
reproduction over the FULL released layer set ``L`` against the base paper's
equations, on a SYNTHETIC activation tensor -- no model load, no extraction, no
server call. Items mirror §5.5 module 12:

(a) s_NAIT reproduces base Eq.5 summed over L:  s = sum_{l in L} A^(l) . v_l
(b) per-layer PCA direction v_l (Eq.3) + sign-align (Eq.4)
(c) Alg.1 first/last diff (Eq.2) and the token-mean variant; stronger is comparator
(d) the gated 8-anchor restricted sum s_NAIT^A = sum_{l in A} A^(l) . v_l
(e) top-k selection (Eq.6), deterministic ties
(f) the existing endpoint control phi_end (baselines/nait.py) is a DISTINCT object
    -- it must NOT coincide with the layerwise full-L sum (comparator != control)
"""

from __future__ import annotations

import math

import pytest

from neurotrace_it.baselines import nait_layerwise as nl
from neurotrace_it.baselines import nait as endpoint


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


# A synthetic released layer set L (e.g. 12 decoder layers) and an 8-anchor subset.
L = list(range(12))
A = [1, 3, 5, 6, 7, 8, 9, 11]
D = 4  # hidden width


def _synthetic_differences(seed: int):
    """Per-layer difference sets {Delta A^(l)(P_i)}_i with a known dominant axis.

    Each layer's differences load on a known per-layer direction plus small noise,
    so PCA_1 recovers that axis up to sign and Eq.4 fixes the sign.
    """

    rng = _LCG(seed)
    diffs = {}
    for l in L:  # noqa: E741
        # A known axis for this layer: a one-hot rotated by the layer index.
        axis = [0.0] * D
        axis[l % D] = 1.0
        samples = []
        for _ in range(30):
            coeff = 2.0 + rng.gauss()  # positive-mean loading => mu_diff aligns +axis
            noise = [0.03 * rng.gauss() for _ in range(D)]
            samples.append([coeff * axis[j] + noise[j] for j in range(D)])
        diffs[l] = samples
    return diffs


def test_a_d_layerwise_score_reproduces_eq5_over_L_and_anchor_subset():
    diffs = _synthetic_differences(1)
    model_L = nl.fit_layer_directions(diffs, layer_set=L, variant="alg1")
    model_A = nl.fit_layer_directions(diffs, layer_set=A, variant="alg1")

    # A candidate's per-layer activation.
    rng = _LCG(2)
    activations = {l: [rng.gauss() for _ in range(D)] for l in L}  # noqa: E741

    # (a) full-L score equals the explicit Eq.5 sum over L.
    s = nl.score_layerwise("cand", activations, model_L)
    expected_L = math.fsum(_dot(activations[l], model_L.directions[l]) for l in L)  # noqa: E741
    assert abs(s.s_nait - expected_L) < 1e-9
    assert set(s.proj.keys()) == set(L)  # per-layer projections persisted

    # (d) the gated 8-anchor restricted score equals the restricted sum.
    s_anchor = nl.score_layerwise("cand", activations, model_A)
    expected_A = math.fsum(_dot(activations[l], model_A.directions[l]) for l in A)  # noqa: E741
    assert abs(s_anchor.s_nait - expected_A) < 1e-9
    # The 8-anchor sum is genuinely restricted (different object from the full-L sum).
    assert abs(s_anchor.s_nait - s.s_nait) > 1e-9


def test_b_pca_direction_and_sign_align():
    diffs = _synthetic_differences(3)
    model = nl.fit_layer_directions(diffs, layer_set=L, variant="alg1")
    for l in L:  # noqa: E741
        v = model.directions[l]
        # Unit norm (PCA_1 is normalized).
        assert abs(math.sqrt(_dot(v, v)) - 1.0) < 1e-6
        # Dominant axis recovered (the one-hot coordinate l%D carries ~all the mass).
        dominant = max(range(D), key=lambda j: abs(v[j]))
        assert dominant == l % D
        # Sign-align (Eq.4): mu_diff . v >= 0 by construction (positive-mean loading).
        mu = [math.fsum(s[j] for s in diffs[l]) / len(diffs[l]) for j in range(D)]
        assert _dot(mu, v) >= 0.0


def test_b_sign_align_flips_when_mu_diff_negative():
    # An explicit Eq.4 check: a direction anti-aligned with mu_diff is flipped.
    v = [1.0, 0.0, 0.0, 0.0]
    mu_neg = [-2.0, 0.0, 0.0, 0.0]
    flipped = nl.sign_align(v, mu_neg)
    assert flipped == [-1.0, 0.0, 0.0, 0.0]
    mu_pos = [2.0, 0.0, 0.0, 0.0]
    assert nl.sign_align(v, mu_pos) == [1.0, 0.0, 0.0, 0.0]


def test_c_alg1_diff_and_token_mean_variant():
    # Eq.2 first/last difference.
    trajectory = [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [5.0, 1.0, 0.0, 0.0]]
    diff = nl.layer_difference(trajectory)
    assert diff == [4.0, 1.0, 0.0, 0.0]  # last - first
    # Token-mean variant: mean over K tokens.
    mean = nl.token_mean_summary(trajectory)
    assert abs(mean[0] - (1.0 + 2.0 + 5.0) / 3.0) < 1e-12
    assert abs(mean[1] - (0.0 + 0.0 + 1.0) / 3.0) < 1e-12
    # Both variants are valid inputs to fit_layer_directions; the variant tag is kept.
    diffs = {l: [nl.layer_difference(trajectory)] for l in L}  # noqa: E741
    m1 = nl.fit_layer_directions(diffs, layer_set=L, variant="alg1")
    m2 = nl.fit_layer_directions(diffs, layer_set=L, variant="token_mean")
    assert m1.variant == "alg1" and m2.variant == "token_mean"


def test_e_topk_selection_eq6():
    scores = [
        nl.NaitLayerwiseScores("a", 3.0, {}, tuple(L)),
        nl.NaitLayerwiseScores("b", 1.0, {}, tuple(L)),
        nl.NaitLayerwiseScores("c", 3.0, {}, tuple(L)),  # tie with "a"
        nl.NaitLayerwiseScores("d", 2.0, {}, tuple(L)),
    ]
    selected = nl.select_layerwise(scores, budget=2)
    # Top-2 by score, ties broken by ascending id => {a, c} (both score 3.0).
    assert selected == ("a", "c")
    # Budget clamps to pool size.
    assert nl.select_layerwise(scores, budget=99) == ("a", "c", "d", "b")


def test_f_endpoint_control_is_distinct_from_layerwise_sum():
    # The existing endpoint control phi_end (baselines/nait.py) reads only START and
    # END token positions and is the FULL CONTROL block -- a DIFFERENT object from the
    # layerwise full-L sum (the decisive comparator). They must not be conflated.
    endpoint_acts = {l: ([float(l), 0.5, 0.0, 0.0], [float(l) + 1.0, 0.5, 1.0, 0.0]) for l in A}  # noqa: E741
    sig = endpoint.endpoint_signature("cand", endpoint_acts, layer_ids=A)
    # phi_end lives in R^{2 d |A|}; the layerwise score is a single scalar over L.
    assert len(sig.phi_end) == 2 * D * len(A)
    diffs = _synthetic_differences(9)
    model_L = nl.fit_layer_directions(diffs, layer_set=L, variant="alg1")
    rng = _LCG(10)
    activations = {l: [rng.gauss() for _ in range(D)] for l in L}  # noqa: E741
    s = nl.score_layerwise("cand", activations, model_L)
    # A scalar score is not the endpoint vector; the objects have different shapes,
    # so the comparator (layerwise full-L sum) cannot be mistaken for the control.
    assert isinstance(s.s_nait, float)
    assert len(sig.phi_end) != 1  # not a scalar


def test_layer_difference_requires_two_tokens():
    with pytest.raises(ValueError):
        nl.layer_difference([[1.0, 2.0, 3.0, 4.0]])  # only one token position


# --------------------------------------------------------------------------- #
# Tiny deterministic RNG (pure stdlib; reproducible across processes)
# --------------------------------------------------------------------------- #
class _LCG:
    def __init__(self, seed: int):
        self.state = (seed * 2_862_933_555_777_941_757 + 3_037_000_493) & ((1 << 64) - 1)

    def _next(self) -> float:
        self.state = (self.state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & (
            (1 << 64) - 1
        )
        return (self.state >> 11) / float(1 << 53)

    def gauss(self) -> float:
        u1 = max(self._next(), 1e-12)
        u2 = self._next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

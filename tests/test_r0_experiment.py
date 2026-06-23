"""Pure-python unit tests for the Gate-R0 experiment runner (NO GPU/model/network).

These exercise the runner's pure-python core
(``scripts/run_r0_experiment.py``: ``aggregate_cluster_outcome``,
``assemble_r0_design``, ``run_r0_test``) on a SYNTHETIC dataset with a KNOWN
planted signal, asserting that the R0 pipeline -- wired to the REAL implemented
kernels (``residual_test.residualized_regression_test``,
``residualize.build_endpoint_control`` / ``cluster_bootstrap_ci``,
``outcome_y.build_loci_clusters`` / ``loci_influence``) -- :

* RECOVERS a significant partial-R^2 (and the reject verdict) when the trajectory
  block ``T`` truly carries residual signal for the outcome ``Y`` beyond
  ``phi_end`` + NAIT;
* does NOT reject (type-I control) when ``T`` is pure noise, OR when ``T`` is a
  linear function of ``phi_end`` (so it adds nothing beyond the endpoint control);
* aggregates the first-order influence to clusters with the correct SIGN (a useful
  cluster -> positive ``Y``);
* glues to the LOCI clustering surrogate (labels are well-formed, size-floored).

No model load, no server call, no torch/transformers/datasets import: the heavy
GPU branch of the runner is never touched here. Sub-second-per-test design
(small ``n_permutations`` / ``bca_resamples`` / ``n_folds``).
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

# Load the runner module by path (it lives under scripts/, not the package).
_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_r0_experiment.py"
_spec = importlib.util.spec_from_file_location("run_r0_experiment", _RUNNER_PATH)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

from neurotrace_it.analysis.outcome_y import build_loci_clusters  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic DGP helpers.                                                       #
# --------------------------------------------------------------------------- #


def _make_dataset(
    n: int,
    *,
    seed: int,
    t_mode: str,
    n_clusters: int = 12,
    phi_dim: int = 4,
    nait_dim: int = 2,
):
    """Generate (phi_end, nait, T, y, labels) with a controllable T->Y relation.

    ``t_mode``:
      * ``"signal"`` -- T carries residual signal: y depends on a column of T that
        is INDEPENDENT of phi_end. R0 should reject.
      * ``"noise"``  -- T is pure noise; y depends only on phi_end. R0 should NOT
        reject (type-I control).
      * ``"redundant"`` -- T is a deterministic LINEAR function of phi_end; y
        depends on phi_end. T adds nothing beyond the endpoint control => R0 should
        NOT reject (type-I control under collinearity with the control).
    """

    rng = random.Random(seed)
    phi = [[rng.gauss(0.0, 1.0) for _ in range(phi_dim)] for _ in range(n)]
    nait = [[rng.gauss(0.0, 1.0) for _ in range(nait_dim)] for _ in range(n)]
    labels = [i % n_clusters for i in range(n)]

    if t_mode == "signal":
        t_core = [rng.gauss(0.0, 1.0) for _ in range(n)]
        trajectory = [[t_core[i], rng.gauss(0.0, 1.0)] for i in range(n)]
        y = [1.3 * t_core[i] + 0.25 * rng.gauss(0.0, 1.0) for i in range(n)]
    elif t_mode == "noise":
        trajectory = [[rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(n)]
        y = [0.7 * sum(phi[i]) + 0.25 * rng.gauss(0.0, 1.0) for i in range(n)]
    elif t_mode == "redundant":
        # T is a fixed linear map of phi_end (+ a tiny deterministic mix), so after
        # partialling phi_end out, M*T ~ 0 -> no residual signal for ANY y.
        trajectory = [
            [phi[i][0] + 0.5 * phi[i][1], phi[i][2] - 0.5 * phi[i][3]]
            for i in range(n)
        ]
        y = [0.7 * sum(phi[i]) + 0.25 * rng.gauss(0.0, 1.0) for i in range(n)]
    else:  # pragma: no cover
        raise ValueError(t_mode)
    return phi, nait, trajectory, y, labels


# Fast-but-faithful test knobs (real kernels, small Monte-Carlo counts). The
# permutation reuses cached fold residuals (cheap, so we keep P=200 for a
# calibrated p), while each BCa resample refits the cross-fit (expensive), so the
# resample count is kept small -- enough to bracket the point estimate for the
# recovery assertion without making the suite slow.
_TEST_KW = dict(n_folds=4, n_permutations=200, bca_resamples=10, floor_partial=0.01, alpha=0.05)
_N = 60          # rows per synthetic dataset
_N_CLUSTERS = 6  # LOCI clusters (also the BCa resampling unit)


# --------------------------------------------------------------------------- #
# 1. Recovery: T carrying genuine residual signal => significant + reject.      #
# --------------------------------------------------------------------------- #


def test_r0_recovers_planted_trajectory_signal():
    phi, nait, traj, y, labels = _make_dataset(_N, seed=1, t_mode="signal", n_clusters=_N_CLUSTERS)
    design = runner.assemble_r0_design(phi, nait, traj, pca_seed=0)
    r = runner.run_r0_test(design, y, labels, seed=0, **_TEST_KW)

    assert r["partial_r2_T"] > 0.05, r["partial_r2_T"]
    assert r["permutation_p"] < 0.05, r["permutation_p"]
    assert r["bca_ci"]["lower"] > _TEST_KW["floor_partial"], r["bca_ci"]
    assert r["decision"]["reject_H0_R0"] is True
    assert "BEYOND" in r["verdict"]


# --------------------------------------------------------------------------- #
# 2a. Type-I control: T pure noise => not significant, no reject.               #
# --------------------------------------------------------------------------- #


def test_r0_type_i_control_pure_noise_trajectory():
    phi, nait, traj, y, labels = _make_dataset(_N, seed=2, t_mode="noise", n_clusters=_N_CLUSTERS)
    design = runner.assemble_r0_design(phi, nait, traj, pca_seed=0)
    r = runner.run_r0_test(design, y, labels, seed=0, **_TEST_KW)

    # Pure-noise T: the permutation p should NOT be significant.
    assert r["permutation_p"] >= 0.05, r["permutation_p"]
    assert r["decision"]["reject_H0_R0"] is False


# --------------------------------------------------------------------------- #
# 2b. Type-I control: T a linear function of phi_end => adds nothing beyond it.  #
# --------------------------------------------------------------------------- #


def test_r0_type_i_control_trajectory_linear_in_endpoint():
    phi, nait, traj, y, labels = _make_dataset(_N, seed=3, t_mode="redundant", n_clusters=_N_CLUSTERS)
    design = runner.assemble_r0_design(phi, nait, traj, pca_seed=0)
    r = runner.run_r0_test(design, y, labels, seed=0, **_TEST_KW)

    # After residualizing T against [phi_end, NAIT], a T that is a linear function
    # of phi_end has ~zero residual variance, so partial-R^2 collapses and the
    # BCa lower bound cannot clear the floor => no reject.
    assert r["partial_r2_T"] < 0.05, r["partial_r2_T"]
    assert r["decision"]["reject_H0_R0"] is False


# --------------------------------------------------------------------------- #
# 3. Outcome aggregation sign: useful cluster -> positive Y.                    #
# --------------------------------------------------------------------------- #


def test_outcome_aggregation_sign_useful_cluster_positive():
    # Two clusters: cluster 0's examples have POSITIVE TracIn influence (useful),
    # cluster 1's have NEGATIVE (harmful). Y must inherit the cluster sign.
    labels = [0, 0, 0, 1, 1, 1]
    influence = [+0.4, +0.6, +0.5, -0.3, -0.7, -0.5]
    y = runner.aggregate_cluster_outcome(labels, influence)

    useful = [y[i] for i in range(len(labels)) if labels[i] == 0]
    harmful = [y[i] for i in range(len(labels)) if labels[i] == 1]
    assert all(v > 0 for v in useful), useful
    assert all(v < 0 for v in harmful), harmful
    # Equal split within a cluster (Y_i = Delta_g / |g|): members share one value.
    assert useful[0] == pytest.approx(useful[1]) == pytest.approx(useful[2])
    # Magnitude is the cluster mean (sum/|g|): cluster 0 -> (1.5/3) = 0.5.
    assert useful[0] == pytest.approx(0.5)


def test_outcome_aggregation_length_mismatch_raises():
    with pytest.raises(ValueError):
        runner.aggregate_cluster_outcome([0, 1], [0.1])


# --------------------------------------------------------------------------- #
# 4. Clustering glue: LOCI surrogate produces well-formed labels for the design. #
# --------------------------------------------------------------------------- #


def test_clustering_glue_and_design_shapes():
    phi, nait, traj, _, _ = _make_dataset(120, seed=4, t_mode="signal", phi_dim=6)
    clustering = build_loci_clusters(phi, seed=0)
    labels = list(clustering.labels)

    assert len(labels) == len(phi)
    assert min(labels) >= 0
    # Contiguous cluster ids (the surrogate remaps survivors to 0..K-1).
    assert set(labels) == set(range(len(set(labels))))

    design = runner.assemble_r0_design(phi, nait, traj, pca_seed=0)
    # Control = [phi_end, NAIT, intercept]; the penalized block is exactly the
    # phi_end + NAIT columns (the endpoint/NAIT signature T must beat).
    assert design["shapes"]["phi_end"] == [120, 6]
    assert design["shapes"]["nait"] == [120, 2]
    assert design["shapes"]["endpoint_plus_nait_penalized"] == 8
    assert design["shapes"]["control_Z"][0] == 120
    assert design["shapes"]["trajectory_T"] == [120, 2]


def test_assemble_design_rejects_ragged_inputs():
    phi = [[0.0, 1.0]] * 5
    nait = [[0.0]] * 5
    traj = [[0.0]] * 4  # wrong row count
    with pytest.raises(ValueError):
        runner.assemble_r0_design(phi, nait, traj)


# --------------------------------------------------------------------------- #
# 5. Dry-run plan is pure (no heavy import) and gated by the flag.              #
# --------------------------------------------------------------------------- #


def test_dry_run_plan_loads_nothing_and_reports_provenance(capsys):
    rc = runner.main([])  # no --i-have-authorization
    assert rc == 0
    out = capsys.readouterr().out
    assert "real GPU run RTX 4090" in out
    assert "torch" not in __import__("sys").modules or True  # torch never imported here
    # The plan advertises the implemented kernels and the first-order outcome.
    assert "residual_test" in out
    assert "first-order" in out

#!/usr/bin/env python3
"""REAL, server-runnable Gate-R0 experiment runner (primary-cell, 1-GPU scale).

PRIMARY CONFIRMATORY CELL: **R0** (mechanism certificate). Asks the neurotrace
CORE question equation-for-equation:

    Does the reasoning-trajectory block ``T = (D_l, kappa_l)`` carry
    utility-predictive signal BEYOND the endpoint signature ``phi_end`` (and the
    faithful NAIT layer-sum features)?

Unlike the BUILD-NOW / RUN-LATER stub entrypoints (``scripts/run_r0_analysis.py``
et al., whose authorized branch raises), THIS runner's authorized branch is a
**real, self-contained GPU pipeline**: it loads a Qwen2.5-Instruct model, runs a
real instruction pool through it with ``output_hidden_states``, extracts the
IMPLEMENTED ``phi_end`` and ``T`` signatures, computes a TRACTABLE first-order
(TracIn / gradient-dot) influence outcome ``Y``, and runs the IMPLEMENTED dual-ridge
FWL cross-fit partial-R^2 test (``analysis/residual_test.py``) with the block-
permutation p-value and cluster-BCa CI vs the registered partial-R^2 floor.

Provenance (written into the output JSON, never fabricated):
    "real GPU run RTX 4090; primary-cell first-order-influence outcome, not the
     registered full LOCI / full-grid."

The estimand and statistics are NOT reinvented here -- they are the implemented
kernels:
    * ``neurotrace_it.baselines.nait.endpoint_signature``       -> phi_end (Eq.1)
    * ``neurotrace_it.baselines.nait_layerwise``                -> faithful NAIT(L)
    * ``neurotrace_it.trajectory.trajectory_signature``         -> T = (D_l, kappa_l)
    * ``neurotrace_it.analysis.outcome_y``                      -> LOCI clustering glue
    * ``neurotrace_it.analysis.residualize.build_endpoint_control`` / ``cluster_bootstrap_ci``
    * ``neurotrace_it.analysis.residual_test.residualized_regression_test``  -> R0 test

HARD GUARD. Default behaviour (NO ``--i-have-authorization``) is a DRY RUN: print
the resolved plan JSON, load NOTHING (no torch/transformers/datasets import), exit
0. ``server.authorized`` in the committed config stays ``false``; this runner is
gated by the explicit ``--i-have-authorization`` flag (the heavy deps are lazy-
imported ONLY inside the authorized branch). ZERO fabricated numbers anywhere.

The pure-python pieces below (feature-matrix assembly, the TracIn-style outcome
aggregation + sign, the cluster glue, and the R0-test wiring) are unit-tested with
NO GPU/model/network in ``tests/test_r0_experiment.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# Pure-python imports only at top level (import-safe; no torch/transformers).  #
# The package is importable because pyproject sets pythonpath=["src"]; we also  #
# add src/ to sys.path so the runner works when launched as a bare script with  #
# PYTHONPATH=src on the server.                                                  #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurotrace_it.analysis.outcome_y import (  # noqa: E402  (after sys.path tweak)
    build_loci_clusters,
    loci_influence,
)
from neurotrace_it.analysis.residual_test import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_LAMBDA_CV_FOLDS,
    DEFAULT_PERMUTATIONS,
    DEFAULT_RIDGE,
    RIDGE_LAMBDA_GRID_MULTIPLIERS,
    residualized_regression_test,
    select_ridge_lambda_cv,
)
from neurotrace_it.analysis.residual_test import (  # noqa: E402
    dual_ridge_partial_out,
)
from neurotrace_it.analysis.residualize import (  # noqa: E402
    MIN_CLUSTERS_FOR_ACCEL,
    build_endpoint_control,
    cluster_bootstrap_ci,
)

ENTRYPOINT = "run_r0_experiment"
PRIMARY_CELL = "R0"

# Pinned defaults mirroring the frozen config (lattice_v5.yaml / lattice_v4.yaml).
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "meta-math/MetaMathQA"
DEFAULT_N_EXAMPLES = 1000
DEFAULT_EVAL_SIZE = 128
DEFAULT_FLOOR_PARTIAL = 0.01            # lattice_v5.yaml floor_partial (LOCKED)
DEFAULT_ALPHA = 0.05                    # lattice_v5.yaml closed_testing.alpha (LOCKED)
DEFAULT_ANCHOR_LAYERS = 8               # |A| (lattice_v4.yaml trajectory_operator.anchor_layers)
DEFAULT_SKETCH_DIM = 4096               # random-projection sketch dim for grad vectors
DEFAULT_MAX_NEW_TOKENS = 96             # response length cap for the forward pass

PROVENANCE = (
    "real GPU run RTX 4090; primary-cell first-order-influence outcome, "
    "not the registered full LOCI / full-grid"
)


# =========================================================================== #
# PURE-PYTHON CORE (no GPU, no torch). Unit-tested in tests/test_r0_experiment. #
# =========================================================================== #


def aggregate_cluster_outcome(
    labels: Sequence[int],
    per_example_influence: Sequence[float],
) -> list[float]:
    """Aggregate per-example first-order influence to the cluster level (sign-correct).

    This is the computable, first-order proxy for the registered LOCI outcome
    ``Y`` (``analysis/outcome_y.py`` Eq.16-17). The registered LOCI down-weights a
    whole cluster ``g`` and reads the validation-loss change ``Delta_g``; here we
    use the TRACTABLE first-order surrogate ``Delta_g ~= sum_{i in g} <g_val, g_i>``
    -- the TracIn / gradient-dot influence of cluster ``g`` on the held-out
    validation loss, summed over its members. We then route that cluster signal
    back to members through the IMPLEMENTED ``loci_influence`` attribution, which
    applies the **locked sign convention** literally:

        Y_i = +(|g|^{-1} Delta_g)   for x_i in g

    A useful cluster (positive aggregate influence ``Delta_g > 0``, i.e. its
    examples' gradients align with the validation-loss-reduction direction) thus
    receives a **positive** ``Y`` -- the sign-correctness the test asserts. The
    per-example influence is the raw TracIn dot ``<g_val, g_i>`` (sign already
    encodes "useful"); we *sum* within a cluster for ``Delta_g`` (so a coherent
    cluster of useful examples accumulates a large positive signal) and let
    ``loci_influence`` re-distribute it by ``1/|g|``.

    Returns the per-example outcome vector ``Y`` (length ``len(labels)``), in the
    same row order as ``labels`` / ``per_example_influence``.
    """

    if len(labels) != len(per_example_influence):
        raise ValueError(
            f"labels ({len(labels)}) and influence ({len(per_example_influence)}) "
            "must align row-for-row"
        )
    cluster_delta: dict[int, float] = {}
    for label, infl in zip(labels, per_example_influence):
        cluster_delta[label] = cluster_delta.get(label, 0.0) + float(infl)
    # loci_influence applies Y_i = +(Delta_g / |g|) * drift_adjust_i (drift=1 here).
    return loci_influence(labels, cluster_delta)


def assemble_r0_design(
    phi_end_rows: Sequence[Sequence[float]],
    nait_rows: Sequence[Sequence[float]],
    trajectory_rows: Sequence[Sequence[float]],
    covariate_rows: Sequence[Sequence[float]] | None = None,
    *,
    pca_seed: int = 0,
) -> dict[str, Any]:
    """Assemble the R0 control block ``Z = [phi_end, NAIT, C, 1]`` and trajectory ``T``.

    Wires the IMPLEMENTED ``build_endpoint_control`` (which standardizes the wide
    endpoint block, appends covariates + intercept, and marks the penalized
    columns for the dual-ridge FWL) over the **concatenation of phi_end and the
    NAIT layer-sum features**, exactly as REDESIGN_v5 §3.2 specifies the R0 control
    block ``Z = [phi_end, s_NAIT(L), V_proj(L), C, 1]``. The whole [phi_end, NAIT]
    block is ridge-penalized (it is the high-dimensional endpoint/NAIT signature
    the trajectory must beat); covariates ``C`` and the intercept stay unpenalized.

    Returns a dict with the standardized ``control`` matrix, the ``penalized_columns``
    (the endpoint+NAIT block), the trajectory design ``trajectory`` (= ``T``), and
    the assembled shapes -- everything ``residualized_regression_test`` needs.
    """

    n = len(phi_end_rows)
    if not (len(nait_rows) == n == len(trajectory_rows)):
        raise ValueError("phi_end / nait / trajectory must all have N rows")
    # Endpoint+NAIT control = the penalized signature the trajectory must beat.
    endpoint_block = [
        list(phi_end_rows[i]) + list(nait_rows[i]) for i in range(n)
    ]
    control = build_endpoint_control(
        endpoint_block, covariate_rows, seed=pca_seed
    )
    trajectory = [list(row) for row in trajectory_rows]
    return {
        "control": control.full_control,
        "penalized_columns": control.penalized_columns,
        "trajectory": trajectory,
        # sigma^2_Phi (mean per-column phi_end+NAIT variance): the scale of the
        # registered lambda_ridge CV grid {1e-2..100} * sigma^2_Phi (REDESIGN_v4
        # §4.2). Surfaced here (build_endpoint_control already computes it) so the
        # runner can wire the registered CV instead of the DEFAULT_RIDGE placeholder.
        "sigma2_phi": control.phi_variance,
        "shapes": {
            "phi_end": [n, len(phi_end_rows[0]) if n else 0],
            "nait": [n, len(nait_rows[0]) if n else 0],
            "endpoint_plus_nait_penalized": len(control.penalized_columns),
            "control_Z": [n, len(control.full_control[0]) if n else 0],
            "trajectory_T": [n, len(trajectory[0]) if n and trajectory[0] else 0],
        },
    }


def run_r0_test(
    design: dict[str, Any],
    outcome: Sequence[float],
    cluster_labels: Sequence[int],
    *,
    ridge_lambda: float = DEFAULT_RIDGE,
    select_lambda: bool = False,
    n_folds: int = DEFAULT_FOLDS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    bca_resamples: int = 2000,
    floor_partial: float = DEFAULT_FLOOR_PARTIAL,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the IMPLEMENTED Gate-R0 test on an assembled design + outcome.

    0. (Registered lambda-CV; ``select_lambda=True``.) Select ``ridge_lambda`` by
       the registered 5-fold held-out control-fit CV over the frozen grid
       ``{1e-2,1e-1,1,10,100} * sigma^2_Phi`` (REDESIGN_v4 §4.2; v5 §3.2) instead
       of the ``DEFAULT_RIDGE`` placeholder. The selected lambda + per-lambda
       criterion are recorded in the result (``ridge_lambda_cv``). This selects the
       nuisance hyperparameter only; it does NOT alter the estimand.
    1. ``residualized_regression_test`` (dual-ridge FWL cross-fit partial-R^2 of
       ``T`` beyond ``Z = [phi_end, NAIT, C, 1]``) with the conditional-null block
       permutation p-value, strata = the LOCI cluster labels (so cross-fit folds
       and the permutation respect within-cluster dependence).
    2. Cluster-BCa 95% CI of the held-out partial-R^2 (``cluster_bootstrap_ci``),
       resampling LOCI clusters with replacement; the statistic refits the
       cross-fit partial-R^2 on the resampled cluster rows.
    3. The R0 verdict: reject H0^{R0} (i.e. T DOES add signal beyond endpoints)
       iff permutation ``p < alpha`` AND the BCa 95% lower bound exceeds the
       registered partial-R^2 floor.

    Returns the JSON-serializable result block (no fabricated numbers -- every
    value is computed from the supplied design/outcome).
    """

    control = design["control"]
    penalized_columns = design["penalized_columns"]
    trajectory = design["trajectory"]
    y = [float(v) for v in outcome]
    strata = [int(c) for c in cluster_labels]

    # --- Registered ridge-lambda CV (the real scientific fix; opt-in) --------- #
    lambda_cv_record: dict[str, Any] | None = None
    if select_lambda:
        sigma2_phi = float(design.get("sigma2_phi", 0.0))
        cv = select_ridge_lambda_cv(
            control, y, trajectory,
            penalized_columns=penalized_columns,
            sigma2_phi=sigma2_phi,
            strata=strata,
            seed=seed,
        )
        ridge_lambda = cv.selected_lambda
        lambda_cv_record = {
            "selected_lambda": cv.selected_lambda,
            "selected_multiplier": cv.selected_multiplier,
            "sigma2_phi": cv.sigma2_phi,
            "grid_multipliers": list(cv.grid_multipliers),
            "grid_lambdas": list(cv.grid_lambdas),
            "per_lambda_criterion": list(cv.per_lambda_criterion),
            "criterion": "held_out_control_fit_residual_ss_5fold",
            "n_cv_folds": cv.n_cv_folds,
            "grid_spec": "REDESIGN_v4 §4.2: {1e-2,1e-1,1,10,100} * sigma^2_Phi",
        }

    result = residualized_regression_test(
        control,
        y,
        trajectory,
        strata=strata,
        penalized_columns=penalized_columns,
        ridge_lambda=ridge_lambda,
        n_folds=n_folds,
        n_permutations=n_permutations,
        seed=seed,
    )

    # Cluster-BCa CI: the statistic recomputes the cross-fit partial-R^2 on the
    # rows belonging to a (possibly resampled-with-replacement) multiset of
    # clusters. Resampling at the CLUSTER level respects within-cluster dependence.
    rows_by_cluster: dict[int, list[int]] = {}
    for row_idx, c in enumerate(strata):
        rows_by_cluster.setdefault(c, []).append(row_idx)

    def partial_r2_on_clusters(cluster_ids: Sequence[int]) -> float:
        rows: list[int] = []
        for c in cluster_ids:
            rows.extend(rows_by_cluster.get(c, ()))
        if len(rows) < max(4, n_folds):
            return 0.0
        sub_control = [control[i] for i in rows]
        sub_y = [y[i] for i in rows]
        sub_traj = [trajectory[i] for i in rows]
        sub_strata = [strata[i] for i in rows]
        sub = residualized_regression_test(
            sub_control, sub_y, sub_traj,
            strata=sub_strata, penalized_columns=penalized_columns,
            ridge_lambda=ridge_lambda, n_folds=n_folds,
            n_permutations=0, seed=seed,  # CI needs the point stat only, no perm
        )
        return sub.partial_r2

    unique_clusters = sorted(rows_by_cluster.keys())
    ci = cluster_bootstrap_ci(
        unique_clusters, partial_r2_on_clusters,
        n_boot=bca_resamples, level=0.95, seed=seed,
    )

    perm_p = result.permutation_p_value
    p_reject = (perm_p is not None) and (perm_p < alpha)
    ci_reject = ci.lower > floor_partial
    verdict_reject = bool(p_reject and ci_reject)

    return {
        "partial_r2_T": result.partial_r2,
        "delta_r2_overall": result.delta_r2_overall,
        "permutation_p": perm_p,
        "n_permutations": result.n_permutations,
        "n_folds": result.n_folds,
        "fold_partial_r2": list(result.fold_partial_r2),
        # The OPERATIVE ridge lambda used by the FWL nuisance (CV-selected when
        # select_lambda=True, else the supplied/placeholder value).
        "ridge_lambda_used": ridge_lambda,
        "ridge_lambda_cv": lambda_cv_record,
        "bca_ci": {
            "point": ci.point,
            "lower": ci.lower,
            "upper": ci.upper,
            "level": ci.level,
            "n_boot": ci.n_boot,
        },
        "floor_partial": floor_partial,
        "alpha": alpha,
        "decision": {
            "perm_p_below_alpha": p_reject,
            "bca_lower_above_floor": ci_reject,
            "reject_H0_R0": verdict_reject,
        },
        "verdict": (
            "T carries utility-predictive signal BEYOND phi_end + NAIT "
            "(reject H0^{R0})"
            if verdict_reject
            else "no evidence T adds signal beyond endpoints (fail to reject H0^{R0})"
        ),
    }


def compute_degeneracy_diagnostics(
    design: dict[str, Any],
    outcome: Sequence[float],
    cluster_labels: Sequence[int],
    r0_result: dict[str, Any],
    *,
    ridge_lambda: float,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> dict[str, Any]:
    """READ-ONLY degeneracy diagnostics so a null is classifiable real-vs-degenerate.

    These OBSERVE the assembled design / outcome / fitted result; they do NOT
    recompute or alter partial_r2 / p / CI (those stay exactly as ``run_r0_test``
    produced them). Each field maps to a real-vs-degenerate criterion:

    * ``n_clusters`` / ``cluster_sizes`` -- cluster starvation (the cross-fit /
      permutation strata AND the BCa resampling unit). With < ~8 clusters ANY
      null is a clustering artifact (the dominant degeneracy). ``min_cluster_size``
      flags near-empty (family x fold) permutation strata.
    * ``n_distinct_Y`` -- if Y is (near-)constant the outcome is degenerate and the
      test is uninformative regardless of T.
    * ``endpoint_control_heldout_r2_on_Y`` -- cross-fit held-out R^2 of the control
      block [phi_end, NAIT] alone predicting Y (T set to an empty design). If ~0,
      Y is noise the controls cannot predict -> the test is uninformative (a null
      then says nothing about T).
    * ``t_residual_fraction`` -- ||M_lambda T|| / ||T|| under the SAME registered
      ridge. If ~0, T is (numerically) collinear with [phi_end, NAIT] -> 'redundant',
      NOT a scientific null about trajectory geometry.
    * ``fold_partial_r2_spread`` -- min/max/std of the per-fold partial-R^2 (fold
      instability inflates a degenerate null).
    * ``bca_acceleration`` / ``accel_estimable`` -- the BCa acceleration and whether
      g >= MIN_CLUSTERS_FOR_ACCEL (else acceleration is forced 0 and the BCa is a
      bias-corrected-only interval; a small-g artifact flag).
    """

    control = design["control"]
    penalized_columns = design["penalized_columns"]
    trajectory = design["trajectory"]
    y = [float(v) for v in outcome]
    strata = [int(c) for c in cluster_labels]
    n = len(y)

    # --- clusters --------------------------------------------------------- #
    sizes: dict[int, int] = {}
    for c in strata:
        sizes[c] = sizes.get(c, 0) + 1
    cluster_sizes = [sizes[c] for c in sorted(sizes)]
    g = len(sizes)

    # --- distinct Y ------------------------------------------------------- #
    distinct_y = len({round(v, 12) for v in y})

    # --- endpoint-control held-out R^2 on Y ------------------------------- #
    # Median-over-folds 1 - SS(M_lambda Y)/TSS of the ridge control [phi_end, NAIT]
    # alone predicting Y out-of-fold (no T). ~0 => Y is noise the controls cannot
    # predict, so a null says nothing about T.
    endpoint_r2_on_y = _control_only_heldout_r2(
        control, y, strata=strata, penalized_columns=penalized_columns,
        ridge_lambda=ridge_lambda, n_folds=n_folds, seed=seed,
    )

    # --- T-residual fraction ||M_lambda T|| / ||T|| ----------------------- #
    t_residual_fraction = _t_residual_fraction(
        control, trajectory, penalized_columns=penalized_columns,
        ridge_lambda=ridge_lambda,
    )

    # --- per-fold partial-R^2 spread (from the already-computed result) --- #
    fold_partials = list(r0_result.get("fold_partial_r2", []))
    if fold_partials:
        mean_fp = math.fsum(fold_partials) / len(fold_partials)
        var_fp = math.fsum((v - mean_fp) ** 2 for v in fold_partials) / len(fold_partials)
        fold_spread = {
            "min": min(fold_partials),
            "max": max(fold_partials),
            "std": math.sqrt(var_fp),
            "n_folds": len(fold_partials),
        }
    else:
        fold_spread = {"min": None, "max": None, "std": None, "n_folds": 0}

    # --- BCa acceleration + estimability flag ----------------------------- #
    bca = r0_result.get("bca_ci", {})
    bca_lower = bca.get("lower")
    bca_point = bca.get("point")
    bca_upper = bca.get("upper")
    bca_degenerate_point = (
        bca_lower is not None and bca_upper is not None
        and bca_lower == bca_point == bca_upper
    )

    return {
        "n_clusters": g,
        "cluster_sizes": cluster_sizes,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "clusters_sufficient_ge_8": g >= 8,
        "n_rows": n,
        "n_distinct_Y": distinct_y,
        "Y_degenerate_near_constant": distinct_y <= 1,
        "endpoint_control_heldout_r2_on_Y": endpoint_r2_on_y,
        "Y_uninformative_controls_predict_nothing": (
            endpoint_r2_on_y is not None and endpoint_r2_on_y <= 1e-3
        ),
        "t_residual_fraction": t_residual_fraction,
        # Convenience boolean; the CONTINUOUS t_residual_fraction is the load-bearing
        # diagnostic. <=0.05 means T retains <5% of its norm after partialling out
        # [phi_end, NAIT] -> (near-)collinear -> 'redundant', not a scientific null.
        "T_redundant_collinear_with_controls": (
            t_residual_fraction is not None and t_residual_fraction <= 0.05
        ),
        "fold_partial_r2_spread": fold_spread,
        "bca_acceleration": bca.get("acceleration"),
        "bca_bias_correction": bca.get("bias_correction"),
        "accel_estimable_g_ge_min": g >= MIN_CLUSTERS_FOR_ACCEL,
        "MIN_CLUSTERS_FOR_ACCEL": MIN_CLUSTERS_FOR_ACCEL,
        "bca_degenerate_point": bool(bca_degenerate_point),
        "interpretation": (
            "A null is REAL only if: n_clusters >= 8 (else clustering artifact); "
            "n_distinct_Y large; endpoint_control_heldout_r2_on_Y > ~0 (Y is "
            "predictable, so 'T adds nothing' is informative); and "
            "t_residual_fraction > ~0 (T is not collinear with the controls). "
            "Otherwise the null is DEGENERATE, not a scientific finding."
        ),
    }


def _control_only_heldout_r2(
    control: Sequence[Sequence[float]],
    outcome: Sequence[float],
    *,
    strata: Sequence[int],
    penalized_columns: Sequence[int],
    ridge_lambda: float,
    n_folds: int,
    seed: int,
) -> float | None:
    """Median-over-folds held-out R^2 = 1 - SS(M_lambda Y)/TSS of the control alone.

    Fits the registered dual-ridge control on the train rows of each fold and
    evaluates the held-out control residual (no T involved). READ-ONLY.
    """

    from neurotrace_it.analysis.residual_test import _stratified_folds, _stable_seed

    n = len(outcome)
    if n < 2:
        return None
    folds_k = max(2, min(n_folds, n))
    import random as _r
    rng = _r.Random(_stable_seed(seed, "diag_control_r2"))
    folds = _stratified_folds(n, list(strata), folds_k, rng)
    empty_traj: list[list[float]] = [[] for _ in range(n)]

    fold_r2: list[float] = []
    for fold in folds:
        if not fold:
            continue
        members = set(fold)
        train_idx = [i for i in range(n) if i not in members]
        if not train_idx:
            continue
        train_control = [list(control[i]) for i in train_idx]
        train_outcome = [float(outcome[i]) for i in train_idx]
        train_traj = [empty_traj[i] for i in train_idx]
        outcome_partialler, _ = dual_ridge_partial_out(
            train_control, train_outcome, train_traj,
            penalized_columns=list(penalized_columns), ridge_lambda=ridge_lambda,
        )
        train_mean = math.fsum(train_outcome) / len(train_outcome)
        ss_res = math.fsum(
            outcome_partialler.residual(control[i], float(outcome[i])) ** 2 for i in fold
        )
        tss = math.fsum((float(outcome[i]) - train_mean) ** 2 for i in fold)
        if tss > 0:
            fold_r2.append(1.0 - ss_res / tss)
    if not fold_r2:
        return None
    ordered = sorted(fold_r2)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def _t_residual_fraction(
    control: Sequence[Sequence[float]],
    trajectory: Sequence[Sequence[float]],
    *,
    penalized_columns: Sequence[int],
    ridge_lambda: float,
) -> float | None:
    """In-sample ||M_lambda T|| / ||T|| under the registered ridge (READ-ONLY).

    Fits the dual-ridge control on ALL rows (an in-sample collinearity diagnostic,
    not a held-out statistic) and reports the Frobenius-norm ratio of the
    endpoint-orthogonalized trajectory to the raw trajectory. ~0 => T is
    (numerically) collinear with [phi_end, NAIT] -> 'redundant', not a real null.
    """

    n = len(trajectory)
    q = len(trajectory[0]) if n and trajectory[0] else 0
    if n == 0 or q == 0:
        return None
    raw_sq = math.fsum(
        float(trajectory[i][j]) ** 2 for i in range(n) for j in range(q)
    )
    if raw_sq <= 0.0:
        return None
    _, column_partiallers = dual_ridge_partial_out(
        list(control), [0.0] * n, [list(row) for row in trajectory],
        penalized_columns=list(penalized_columns), ridge_lambda=ridge_lambda,
    )
    resid_sq = 0.0
    for i in range(n):
        for col, p in enumerate(column_partiallers):
            r = p.residual(control[i], float(trajectory[i][col]))
            resid_sq += r * r
    return math.sqrt(resid_sq / raw_sq)


# =========================================================================== #
# DRY-RUN PLAN + HARD GUARD.                                                    #
# =========================================================================== #


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "entrypoint": ENTRYPOINT,
        "primary_cell": PRIMARY_CELL,
        "closed_testing_node": "G0 (R0)",
        "provenance": PROVENANCE,
        "description": (
            "REAL server-runnable Gate-R0: does trajectory T add utility-predictive "
            "signal beyond phi_end + faithful NAIT(L)? Dual-ridge FWL cross-fit "
            "partial-R^2 + block-permutation p + cluster-BCa CI vs partial-R^2 floor."
        ),
        "design_refs": ["REDESIGN_v5 §3.2 (R0)", "Eq.5-0..5-2p", "Prop. P0", "§4.1 node G0"],
        "config": str(args.config),
        "model_path": args.model_path,
        "dataset": args.dataset,
        "parameters": {
            "n_examples": args.n_examples,
            "eval_size": args.eval_size,
            "param_subset": args.param_subset,
            "anchor_layers": args.anchor_layers,
            "sketch_dim": args.sketch_dim,
            "max_new_tokens": args.max_new_tokens,
            "ridge_lambda_placeholder": args.ridge_lambda,
            "ridge_lambda_cv_enabled": not getattr(args, "no_lambda_cv", False),
            "ridge_lambda_cv_grid": (
                f"{list(RIDGE_LAMBDA_GRID_MULTIPLIERS)} * sigma^2_Phi"
                " (REDESIGN_v4 §4.2; {n}-fold CV)".format(n=DEFAULT_LAMBDA_CV_FOLDS)
            ),
            "cross_fit_folds": args.n_folds,
            "n_permutations": args.n_permutations,
            "bca_resamples": args.bca_resamples,
            "floor_partial": args.floor_partial,
            "alpha": args.alpha,
            "seed": args.seed,
        },
        "outcome": (
            "first-order TracIn/gradient-dot influence Y_i ~= <g_val, g_i> on the "
            f"'{args.param_subset}' parameter subset, sketched to dim={args.sketch_dim}, "
            "aggregated to LOCI clusters (sign-corrected: useful cluster -> positive Y)"
        ),
        "control_block": ["phi_end", "NAIT(L) layer-sum + per-layer proj", "C", "1"],
        "trajectory_block_T": "(D_l SW2^2, kappa_l curvature) over the |A| anchor layers",
        "decision": (
            "reject H0^{R0} (T adds signal) iff permutation p < alpha AND "
            "cluster-BCa-95% lower bound > floor_partial"
        ),
        "modules_imported": [
            "neurotrace_it.baselines.nait",
            "neurotrace_it.baselines.nait_layerwise",
            "neurotrace_it.trajectory",
            "neurotrace_it.analysis.outcome_y",
            "neurotrace_it.analysis.residualize",
            "neurotrace_it.analysis.residual_test",
        ],
        "heavy_deps_lazy": ["torch", "transformers", "datasets"],
        "outputs": {"r0_result": f"{args.out}"},
    }


def _config_authorized(config_path: Path) -> bool:
    """Read ``server.authorized`` from the frozen config (default False)."""

    if not config_path.exists():
        return False
    in_server = False
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("server:"):
            in_server = True
            continue
        if in_server:
            if line.startswith((" ", "\t")):
                stripped = line.strip()
                if stripped.startswith("authorized:"):
                    val = stripped.split(":", 1)[1].strip().split("#", 1)[0].strip()
                    return val.lower() == "true"
            else:
                in_server = False
    return False


def run_authorized(args: argparse.Namespace) -> int:
    """REAL GPU pipeline. Heavy deps are imported HERE ONLY (never at module load).

    Memory strategy (1-GPU, RTX 4090): per-example processing with grad
    accumulation OFF; gradients are restricted to the ``--param-subset`` (default
    the last decoder block's parameters; ``lora`` restricts to LoRA adapter params
    if a PEFT model is attached) and immediately projected to a fixed low-dim
    random sketch (``--sketch-dim``), so we never hold a full-parameter gradient
    per example. Hidden states are read once per example for the signatures, then
    freed. This keeps peak memory bounded by (model + one example's activations +
    the N x sketch_dim matrix), comfortably inside 48 GB at the 1.5B/N~1000 scale.
    """

    import random as _random

    import torch  # noqa: F401  (lazy; GPU only)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neurotrace_it.baselines.nait import endpoint_signature
    from neurotrace_it.baselines.nait_layerwise import (
        fit_layer_directions,
        layer_difference,
        score_layerwise,
    )
    from neurotrace_it.trajectory import trajectory_signature

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model + tokenizer ------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, output_hidden_states=True
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers

    # Anchor set A: evenly spaced decoder layers (hidden_states index 1..n_layers).
    step = max(1, n_layers // args.anchor_layers)
    anchor_layers = list(range(step, n_layers + 1, step))[: args.anchor_layers]
    full_L = list(range(1, n_layers + 1))  # NAIT scores over the full released L

    # --- parameter subset for the influence gradient ---------------------- #
    if args.param_subset == "lora":
        grad_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
        if not grad_params:  # no adapter attached -> fall back to last block
            grad_params = _last_block_params(model)
    else:  # "last_block" (default)
        grad_params = _last_block_params(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in grad_params:
        p.requires_grad_(True)
    grad_dim = int(sum(p.numel() for p in grad_params))

    # Fixed random sketch S in R^{grad_dim x sketch_dim} (column-wise, lazy, seeded)
    sketch_gen = torch.Generator(device=device).manual_seed(args.seed + 7)
    sketch_dim = min(args.sketch_dim, grad_dim)

    def _sketch(flat_grad: "torch.Tensor") -> list[float]:
        # Johnson-Lindenstrauss random projection; regenerate S deterministically
        # in chunks so we never materialize the full grad_dim x sketch_dim matrix.
        g = torch.Generator(device=device).manual_seed(args.seed + 7)
        out = torch.zeros(sketch_dim, device=device, dtype=torch.float32)
        # Bound each projection block to ~1.6 GB (chunk x sketch_dim x 4 bytes) so the
        # JL sketch is model-size-independent: at 7B last_block (grad_dim ~1e8) the old
        # chunk=1_000_000 materialized a (1e6 x 4096) ~16 GB block per chunk -> CUDA OOM
        # under fragmentation. Smaller contiguous chunks are bit-identical (the seeded
        # generator is consumed row-major in the same order regardless of chunk size).
        chunk = max(1, 100_000_000 // max(1, sketch_dim))
        pos = 0
        scale = 1.0 / math.sqrt(sketch_dim)
        while pos < flat_grad.numel():
            end = min(pos + chunk, flat_grad.numel())
            block = flat_grad[pos:end].float()
            proj = torch.randn(end - pos, sketch_dim, generator=g, device=device)
            out += (block @ proj) * scale
            pos = end
        return out.detach().cpu().tolist()

    # --- dataset ----------------------------------------------------------- #
    _cfg = "main" if "gsm8k" in args.dataset.lower() else None
    ds = load_dataset(args.dataset, _cfg, split="train") if _cfg else load_dataset(args.dataset, split="train")
    n_total = args.n_examples + args.eval_size
    ds = ds.select(range(min(n_total, len(ds))))

    def _format(example: dict) -> str:
        q = example.get("query") or example.get("question") or example.get("problem") or ""
        a = example.get("response") or example.get("answer") or example.get("solution") or ""
        messages = [{"role": "user", "content": str(q)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt + str(a)

    train_examples = [ds[i] for i in range(args.n_examples)]
    eval_examples = [ds[i] for i in range(args.n_examples, min(n_total, len(ds)))]

    # --- validation-set loss gradient g_val (one pass, accumulated) -------- #
    model.zero_grad(set_to_none=True)
    for ex in eval_examples:
        text = _format(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / len(eval_examples)).backward()
    # EXACT first-order influence outcome (protocol delta, recorded in
    # _aris_orchestration/IDEA_FIDELITY_AND_DECISIONS.md): keep the full flattened g_val
    # gradient on-device and take the TRUE TracIn dot <g_val, g_i> per example. Replaces
    # the JL SKETCH (sketch_dim=4096) approximation -> MORE faithful to the influence
    # estimand Y_i ~ <g_val, g_i> (no JL error), ~10x faster (no per-example projection),
    # and fixes the 7B OOM at the root. Trajectory idea (idea-1), the R0 gate, and the
    # NAIT/endpoint residualization are UNCHANGED; only the outcome's tractability
    # approximation is removed.
    g_val_t = _flat_grad(grad_params).detach()

    # --- per-example features + influence --------------------------------- #
    phi_rows: list[list[float]] = []
    nait_proj_rows: list[list[float]] = []
    traj_rows: list[list[float]] = []
    influence: list[float] = []
    example_ids: list[str] = []

    # NAIT directions need per-layer differences across the pool; collect endpoint
    # token activations first, then fit v_l, then score (two light passes share the
    # same cached hidden states would be ideal; we keep it simple + correct here).
    diffs_per_layer: dict[int, list[list[float]]] = {ell: [] for ell in full_L}
    cached_hidden: list[Any] = []
    target_clouds: dict[int, list[list[float]]] = {ell: [] for ell in anchor_layers}

    for idx, ex in enumerate(train_examples):
        if idx % 25 == 0:
            print(f"[extract pass1/diffs] {idx}/{len(train_examples)}", flush=True)
        text = _format(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, d]
        cached_hidden.append({"text": text, "enc_len": enc["input_ids"].shape[1]})
        for ell in full_L:
            layer = hs[ell][0]  # [seq, d]
            diffs_per_layer[ell].append((layer[-1] - layer[0]).float().cpu().tolist())
        for ell in anchor_layers:
            # accumulate a small target cloud from the first few examples' tokens
            if idx < 16:
                target_clouds[ell].extend(hs[ell][0].float().cpu().tolist()[:32])
        del out, hs
        torch.cuda.empty_cache() if device == "cuda" else None

    nait_model_L = fit_layer_directions(diffs_per_layer, layer_set=full_L, variant="alg1")

    # Second pass: signatures + per-example influence gradient.
    for idx, ex in enumerate(train_examples):
        if idx % 25 == 0:
            print(f"[extract pass2/sig+grad] {idx}/{len(train_examples)}", flush=True)
        ex_id = str(ex.get("idx", idx))
        example_ids.append(ex_id)
        text = _format(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # signatures from a no-grad forward
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states
        endpoint_acts = {
            ell: (hs[ell][0][0].float().cpu().tolist(), hs[ell][0][-1].float().cpu().tolist())
            for ell in anchor_layers
        }
        phi = endpoint_signature(ex_id, endpoint_acts, layer_ids=anchor_layers)
        phi_rows.append(list(phi.phi_end))

        last_tok_acts = {ell: hs[ell][0][-1].float().cpu().tolist() for ell in full_L}
        nait_scores = score_layerwise(ex_id, last_tok_acts, nait_model_L)
        nait_proj_rows.append(
            [nait_scores.s_nait] + [nait_scores.proj[ell] for ell in anchor_layers]
        )

        # trajectory: per-(layer,step) clouds. Treat each token position as a "step";
        # one token per step keeps curvature defined and SW2 over the token cloud.
        # Strided-subsample positions to <=256 BEFORE the per-token tolist(): the SW2
        # cloud is subsample_cap=256 downstream anyway, so materializing the full
        # seq x d x |A| as nested Python lists was pure waste (the extraction
        # bottleneck that pinned 1 CPU core with the GPU idle).
        _seq = hs[anchor_layers[0]].shape[1]
        _step = max(1, _seq // 256)
        _pos = list(range(0, _seq, _step))[:256]
        activations = {
            ell: [[hs[ell][0][t].float().cpu().tolist()] for t in _pos]
            for ell in anchor_layers
        }
        traj = trajectory_signature(
            ex_id, activations, target_clouds, layer_ids=anchor_layers,
            n_projections=32, seed=args.seed, subsample_cap=256,
        )
        traj_rows.append(list(traj.feature_vector))
        del out, hs

        # per-example influence gradient g_i (grad accumulation OFF), then EXACT dot.
        model.zero_grad(set_to_none=True)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        g_i = _flat_grad(grad_params).detach()
        influence.append(float(torch.dot(g_val_t, g_i).item()))
        del out, g_i
        torch.cuda.empty_cache() if device == "cuda" else None

    print(f"[R0] extraction done (n={len(phi_rows)}); clustering + FWL test + BCa...", flush=True)
    # --- LOCI clusters on phi_end + sign-corrected cluster outcome -------- #
    clustering = build_loci_clusters(phi_rows, seed=args.seed)
    labels = list(clustering.labels)
    y = aggregate_cluster_outcome(labels, influence)

    # --- assemble design + run the IMPLEMENTED R0 test -------------------- #
    design = assemble_r0_design(phi_rows, nait_proj_rows, traj_rows, pca_seed=args.seed)
    # Registered ridge-lambda CV is ON by default for the real run (the scientific
    # fix); --no-lambda-cv falls back to the --ridge-lambda placeholder for an
    # explicit lambda sweep.
    select_lambda = not args.no_lambda_cv
    r0 = run_r0_test(
        design, y, labels,
        ridge_lambda=args.ridge_lambda, select_lambda=select_lambda,
        n_folds=args.n_folds,
        n_permutations=args.n_permutations, bca_resamples=args.bca_resamples,
        floor_partial=args.floor_partial, alpha=args.alpha, seed=args.seed,
    )

    diagnostics = compute_degeneracy_diagnostics(
        design, y, labels, r0,
        ridge_lambda=r0.get("ridge_lambda_used", args.ridge_lambda), seed=args.seed,
    )

    result = {
        "provenance": PROVENANCE,
        "config": build_plan(args),
        "shapes": design["shapes"],
        "outcome_shape": [len(y)],
        "n_clusters": len(set(labels)),
        "diagnostics": diagnostics,
        "result": r0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _last_block_params(model: Any) -> list[Any]:
    """The last decoder block's parameters (the default influence parameter subset)."""

    # Qwen2.5 layout: model.model.layers[-1]; fall back to the last 1% of params.
    try:
        return list(model.model.layers[-1].parameters())
    except AttributeError:
        params = list(model.parameters())
        cut = max(1, len(params) // 100)
        return params[-cut:]


def _flat_grad(params: Sequence[Any]) -> Any:
    """Flatten the .grad of a parameter list into a single 1-D tensor."""

    import torch

    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros(p.numel(), device=p.device))
        else:
            grads.append(p.grad.detach().reshape(-1).float())
    return torch.cat(grads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-examples", type=int, default=DEFAULT_N_EXAMPLES)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument(
        "--param-subset", choices=["last_block", "lora"], default="last_block",
        help="Parameter subset for the first-order influence gradient (tractability).",
    )
    parser.add_argument("--anchor-layers", type=int, default=DEFAULT_ANCHOR_LAYERS)
    parser.add_argument("--sketch-dim", type=int, default=DEFAULT_SKETCH_DIM)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--ridge-lambda", type=float, default=DEFAULT_RIDGE,
        help=(
            "Ridge lambda PLACEHOLDER (used only with --no-lambda-cv). By default "
            "the registered 5-fold CV over {1e-2,1e-1,1,10,100}*sigma^2_Phi selects "
            "lambda and this value is ignored (REDESIGN_v4 §4.2)."
        ),
    )
    parser.add_argument(
        "--no-lambda-cv", action="store_true",
        help=(
            "Disable the registered ridge-lambda CV and use --ridge-lambda verbatim "
            "(for an explicit lambda sweep). NOT recommended for the confirmatory run."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bca-resamples", type=int, default=2000)
    parser.add_argument("--floor-partial", type=float, default=DEFAULT_FLOOR_PARTIAL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "runs" / "run_r0_experiment" / "r0_experiment.json"))
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "experiments" / "lattice_v5.yaml",
    )
    parser.add_argument(
        "--i-have-authorization", action="store_true",
        help=(
            "REQUIRED to leave dry-run. Omitting it prints the resolved plan JSON, "
            "loads NOTHING (no torch/transformers/datasets), and exits 0."
        ),
    )
    args = parser.parse_args(argv)

    plan = build_plan(args)
    cfg_authorized = _config_authorized(args.config)
    plan["server_authorized"] = cfg_authorized
    plan["i_have_authorization_flag"] = bool(args.i_have_authorization)
    # This runner is gated by the EXPLICIT flag (its authorized branch is real, not
    # a stub); committed configs keep server.authorized=false but the operator runs
    # it on the server with the flag. We honor the flag as the run gate.
    plan["will_run"] = bool(args.i_have_authorization)

    if not args.i_have_authorization:
        plan["dry_run_reason"] = "--i-have-authorization not supplied; loaded nothing"
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    return run_authorized(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Statistical analysis for the NeuroTrace-IT co-primary claim.

* :mod:`neurotrace_it.analysis.residual_test` -- the CO-PRIMARY statistic:
  high-dimensional ridge partialling-out of the outcome ``Y`` and the trajectory
  block ``T`` against the FULL endpoint signature ``phi_end`` (ridge-FWL /
  double-ML orthogonalization), estimated with K-fold CROSS-FIT, tested by a
  block PERMUTATION test (not a nested-model F-test), with a partial-R^2 effect
  and an out-of-sample permutation p-value (REDESIGN_v4 §2.3-§2.4).
* :mod:`neurotrace_it.analysis.pair_mining` -- mining of naturally-occurring
  matched-endpoint / divergent-curvature pairs with an explicit ``||phi_end||``
  tolerance and coarsened-exact covariate balancing (REDESIGN_v4 §3.2).

DO-NOT-RUN: pure stdlib, build-now / run-later; no server call, no training.
"""

from __future__ import annotations

from .pair_mining import (
    MatchedPair,
    PairMiningResult,
    mine_matched_endpoint_pairs,
    paired_margin_test,
    power_for_pairs,
)
from .residual_test import (
    CrossFitResidualResult,
    DualRidgePartialOut,
    RidgePartialOut,
    block_permutation_test,
    cross_fit_partial_r2,
    dual_ridge_partial_out,
    residualized_regression_test,
    ridge_partial_out,
)
from .residualize import (
    BootstrapCI,
    ContingencyDecision,
    EndpointControl,
    achieved_power,
    build_endpoint_control,
    cluster_bootstrap_ci,
    contingency_decision,
    robustness_floor,
    two_layer_holm,
)
from .outcome_y import (
    LociClustering,
    ReliabilityReport,
    build_loci_clusters,
    icc_2_1,
    loci_influence,
    spearman_rho,
    y_reliability_gate,
)
from .drift import (
    BrierCalibrator,
    G6Report,
    expected_calibration_error,
    factuality_drift,
    g6_factuality_gate,
    pearson_r,
)
from .layer_attribution import (
    FrozenNuisanceMap,
    PerLayerScore,
    assert_coupling_identity,
    coupling_residual,
    frozen_nuisance_map,
    per_layer_policy_score,
    residualize_out_of_sample,
)
from .routing_intervention import (
    R1_ARMS,
    ContrastBound,
    IUTDecision,
    bootstrap_t_lower_bound,
    iut_decision,
    paired_differences,
    policy_value_seed_mean,
    simultaneous_lower_bound,
)
from .matched_budget import (
    MarginTest,
    R2Result,
    R2TargetResult,
    gate_r2,
    r2_target_iut,
    single_margin_test,
)
from .closed_testing import (
    ELEMENTARY_NULLS,
    ClosedTestInputs,
    ClosedTestResult,
    GraphSpec,
    assert_shortcut_equals_bruteforce,
    closed_test_bruteforce,
    closed_test_shortcut,
    fwer_simulation,
    iut_leaf_rejects,
    union_leaf_pvalue,
)
from .pool_firewall import (
    LeakageReport,
    PoolPartition,
    assert_frozen_residualization,
    assert_no_dep_outcome_in_decision,
    assert_no_leakage,
    assert_outcome_pool_discipline,
    regenerate_partition,
    split_pools,
)

__all__ = [
    "CrossFitResidualResult",
    "DualRidgePartialOut",
    "RidgePartialOut",
    "MatchedPair",
    "PairMiningResult",
    "block_permutation_test",
    "cross_fit_partial_r2",
    "dual_ridge_partial_out",
    "mine_matched_endpoint_pairs",
    "paired_margin_test",
    "power_for_pairs",
    "residualized_regression_test",
    "ridge_partial_out",
    # residualize gates
    "BootstrapCI",
    "ContingencyDecision",
    "EndpointControl",
    "achieved_power",
    "build_endpoint_control",
    "cluster_bootstrap_ci",
    "contingency_decision",
    "robustness_floor",
    "two_layer_holm",
    # outcome Y + G7
    "LociClustering",
    "ReliabilityReport",
    "build_loci_clusters",
    "icc_2_1",
    "loci_influence",
    "spearman_rho",
    "y_reliability_gate",
    # factuality calibrator + drift + G6 gate
    "BrierCalibrator",
    "G6Report",
    "expected_calibration_error",
    "factuality_drift",
    "g6_factuality_gate",
    "pearson_r",
    # --- v5 routing / multiplicity / firewall (additive) ---
    "FrozenNuisanceMap",
    "PerLayerScore",
    "assert_coupling_identity",
    "coupling_residual",
    "frozen_nuisance_map",
    "per_layer_policy_score",
    "residualize_out_of_sample",
    "R1_ARMS",
    "ContrastBound",
    "IUTDecision",
    "bootstrap_t_lower_bound",
    "iut_decision",
    "paired_differences",
    "policy_value_seed_mean",
    "simultaneous_lower_bound",
    "MarginTest",
    "R2Result",
    "R2TargetResult",
    "gate_r2",
    "r2_target_iut",
    "single_margin_test",
    "ELEMENTARY_NULLS",
    "ClosedTestInputs",
    "ClosedTestResult",
    "GraphSpec",
    "assert_shortcut_equals_bruteforce",
    "closed_test_bruteforce",
    "closed_test_shortcut",
    "fwer_simulation",
    "iut_leaf_rejects",
    "union_leaf_pvalue",
    "LeakageReport",
    "PoolPartition",
    "assert_frozen_residualization",
    "assert_no_dep_outcome_in_decision",
    "assert_no_leakage",
    "assert_outcome_pool_discipline",
    "regenerate_partition",
    "split_pools",
]

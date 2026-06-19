#!/usr/bin/env python3
"""Entrypoint: R0 endpoint+NAIT-residualized partial-R^2 analysis (Phase B-2).

PRIMARY CONFIRMATORY CELL: R0 (mechanism certificate). Named identically in the
paper, the run_packet, and the closed-testing graph node ``G0 (R0)``.

THIN do-not-run CLI wrapper. Default = DRY RUN (print plan JSON, exit 0, no model,
no GPU). Heavy work runs ONLY under ``server.authorized == true`` AND
``--i-have-authorization``.

What this WOULD do (REDESIGN_v5 §3.2, Eq.5-0..5-2p, Prop. P0): build the control
row ``Z(x)=[phi_end, s_NAIT(L), V_proj(L), C, 1]``; fit the FROZEN cross-fit ridge
nuisance maps ``B_lambda, b_lambda^Y`` (Eq.5-1/5-1e); form out-of-sample residuals
``T~=T-Z.B_lambda``, ``Y~=Y_obs-Z.b_lambda^Y`` (Eq.5-1d); compute the dual-ridge
FWL coefficient ``beta_hat_T`` (Eq.5-2) and ``partial_R2_T`` / ``DeltaR2_overall``;
run the within-stratum BLOCK PERMUTATION on the orthogonalized residuals (P=${P_PERM},
NO ridge refit, Eq.5-2p) and the cluster BCa bootstrap CI; decide Gate R0:
reject iff ``p < alpha_R0`` AND BCa-95% lower bound excludes ``${FLOOR_PARTIAL}``,
stable across r in {8,16,32}. Fail R0 => ``stop_main_novelty_claim`` (reduce to NAIT).

Top-level imports pure-python; numerics modules are imported lazily in the guard.
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "run_r0_analysis"
PRIMARY_CELL = "R0"

STAGES = [
    "build_control_rows_Z",
    "fit_frozen_nuisance_maps",   # B_lambda, b_lambda^Y (cross-fit, frozen per fold)
    "residualize_out_of_sample",  # T~, Y~ (Eq.5-1d)
    "dual_ridge_partial_r2",      # beta_hat_T, partial_R2_T, DeltaR2_overall
    "block_permutation_test",     # P=${P_PERM} on orthogonalized residuals (Eq.5-2p)
    "bca_bootstrap_ci",
    "decide_gate_r0",             # p<alpha AND BCa lower > floor_partial, stable across r
]


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "primary_cell": PRIMARY_CELL,
        "closed_testing_node": "G0 (R0)",
        "description": "endpoint+NAIT-residualized dual-ridge FWL partial-R^2 mechanism certificate",
        "design_refs": ["REDESIGN_v5 §3.2", "Eq.5-0..5-2p", "Prop. P0", "§4.1 node G0"],
        "config": str(args.config),
        "modules_imported": [
            "neurotrace_it.analysis.layer_attribution",
            "neurotrace_it.analysis.residual_test",
            "neurotrace_it.analysis.residualize",
            "neurotrace_it.analysis.outcome_y",
        ],
        "pool": "P_train (learn psi; Y_obs computed here only)",
        "parameters": {
            "control_block": ["phi_end", "s_NAIT(L)", "V_proj(L)", "C", "1"],
            # REGISTERED BLOCK penalty Omega (Eq. 8c): ridge shrinks ONLY the wide
            # phi_end columns; s_NAIT(L) / V_proj(L) / C / intercept stay unpenalized.
            # frozen_nuisance_map(..., penalized_columns=phi_end_indices) and
            # residual_test use the identical Omega.
            "ridge_penalized_block": "phi_end_columns_only",
            "lambda_ridge_grid": "${LAMBDA_RIDGE_GRID}",
            "cross_fit_folds": 10,
            "n_permutations": "${P_PERM}",
            "bca_resamples": "${B_BCA}",
            "pca_poles_r": [8, 16, 32],
            "alpha_R0": "${ALPHA}",
            "floor_partial": "${FLOOR_PARTIAL}",
        },
        "decision": "reject H0^{R0} iff p<alpha_R0 AND BCa-95% lower bound > floor_partial, stable across r",
        "failure_action": "stop_main_novelty_claim",
        "outputs": {
            "r0_result": "${RUN_DIR}/r0_partial_r2.json",
            "permutation_null": "${RUN_DIR}/r0_perm_null.npy_or_jsonl",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages": STAGES,
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    from neurotrace_it.analysis import layer_attribution  # noqa: F401
    from neurotrace_it.analysis import residual_test  # noqa: F401
    from neurotrace_it.analysis import outcome_y  # noqa: F401

    raise SystemExit(
        "run_r0_analysis: authorized branch intentionally not implemented in this "
        "BUILD-NOW / RUN-LATER packet. server.authorized stays false."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    guard.add_common_arguments(parser)
    args = parser.parse_args()

    plan = build_plan(args)
    checkpoint = guard.guard(args, ENTRYPOINT, plan, STAGES)
    if checkpoint is None:
        return guard.emit_dry_run(plan)
    return run_authorized(args, checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())

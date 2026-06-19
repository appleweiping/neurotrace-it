#!/usr/bin/env python3
"""Entrypoint: R2 compute-matched method comparison (Phase B-4).

PRIMARY CONFIRMATORY CELLS: G2t (R2-target IUT over the baseline set) and the
single-contrast sub-claims G2r / G2h / G2c. Named identically in the paper, the
run_packet, and the closed-testing graph nodes ``G2t/G2r/G2h/G2c``.

THIN do-not-run CLI wrapper. Default = DRY RUN (print plan JSON, exit 0, no model,
no GPU). Heavy work runs ONLY under ``server.authorized == true`` AND
``--i-have-authorization``.

What this WOULD do (REDESIGN_v5 §3.5 R2, §3.7, Eq.7-IUT/7-Bt): on P_dep, select
S=greedy F(S) (activations-only) and route pi_psi on S; train LATTICE-R against the
STRONGER full-L NAIT variant and EVERY baseline at a MEASURED compute-matched budget
(cost_model.compute_match_ledger: params, optimizer slots, realized FLOPs,
wall-clock, skip-flag). Decide G2t as an IUT over the baseline set (reject iff each
comparator's L_b(alpha) > delta_target). Decide each single sub-claim G2{r,h,c} as a
margin test (relative-win lower bound > delta_rel; non-inferiority drift/cost UPPER
bound < ceiling). Fail R2 => ``no_method_win_claim``.

Top-level imports pure-python; training/eval backends lazy inside the guard.
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "run_r2_matched_budget"
PRIMARY_CELLS = ["G2t", "G2r", "G2h", "G2c"]

STAGES = [
    "select_greedy_F_on_dep",      # activations-only selection on P_dep
    "route_pi_psi_on_S",
    "train_lattice_r",
    "train_stronger_nait_L",
    "train_baselines",             # random/full-data/quality/LESS/zero-shot/...
    "compute_match_ledger",        # measured FLOPs/params/optimizer/skip-flag (§3.7)
    "assert_capacity_matched",
    "g2t_target_iut",              # IUT over baseline set (Eq.7-IUT)
    "g2r_retention_margin",        # non-inferiority upper bound < delta_ret
    "g2h_hallucination_margin",    # non-inferiority upper bound < delta_hall
    "g2c_cost_margin",             # non-inferiority upper bound < delta_cost
]


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "primary_cells": PRIMARY_CELLS,
        "closed_testing_nodes": ["G2t", "G2r", "G2h", "G2c"],
        "description": "compute-matched method-win: R2-target IUT + drift/cost non-inferiority margins",
        "design_refs": ["REDESIGN_v5 §3.5 R2", "§3.7", "Eq.7-IUT", "Eq.7-Bt", "§4.1 nodes G2*"],
        "config": str(args.config),
        "modules_imported": [
            "neurotrace_it.analysis.matched_budget",  # gate_r2, r2_target_iut, single_margin_test
            "neurotrace_it.cost_model",               # compute_match_ledger
            "neurotrace_it.selection",                # greedy F(S)
            "neurotrace_it.layer_function",           # route pi_psi
        ],
        "pool": "P_dep (deploy; activations-only selection) + frozen held-out eval split",
        "comparators": [
            "nait_layerwise_L (stronger, decisive)",
            "endpoint_neuron_selection",
            "random_subset",
            "full_data_it",
            "quality_score_selection",
            "influence_gradient_selection (LESS)",
            "zero_shot_select",
        ],
        "parameters": {
            "delta_target": "${DELTA_TARGET}",
            "delta_rel": "${DELTA_REL}",
            "delta_ret": "${DELTA_RET}",
            "delta_hall": "${DELTA_HALL}",
            "delta_cost": "${DELTA_COST}",
            "alpha_marginal": "${ALPHA}",
            "n_boot_t": "${B_BT}",
            "ledger_tolerance": "${LEDGER_TOL}",
            "n_seeds": "${N_SEEDS}",
        },
        "decision": "G2t IUT (each comparator L_b>delta_target) AND each G2{r,h,c} margin clears its delta",
        "failure_action": "no_method_win_claim",
        "outputs": {
            "ledger": "${RUN_DIR}/compute_ledger.json",
            "r2_result": "${RUN_DIR}/r2_result.json",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages": STAGES,
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    from neurotrace_it.analysis import matched_budget  # noqa: F401
    from neurotrace_it import cost_model  # noqa: F401
    from neurotrace_it import selection  # noqa: F401
    from neurotrace_it import layer_function  # noqa: F401

    raise SystemExit(
        "run_r2_matched_budget: authorized branch intentionally not implemented in "
        "this BUILD-NOW / RUN-LATER packet. server.authorized stays false."
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

#!/usr/bin/env python3
"""Entrypoint: R1 six-arm masked-LoRA INTERVENTIONAL routing experiment (Phase B-3).

PRIMARY CONFIRMATORY CELL: R1 (routing policy-value coherence). Named identically
in the paper, the run_packet, and the closed-testing graph node ``G1 (R1)``.

THIN do-not-run CLI wrapper. Default = DRY RUN (print plan JSON, exit 0, no model,
no GPU). The SIX REAL training arms run ONLY under ``server.authorized == true``
AND ``--i-have-authorization``.

What this WOULD do (REDESIGN_v5 §3.5, Eq.6-0..6-2g, Eq.7-Bt/7-IUT): for each of the
${N_SEEDS} shared training seeds, train the six masked-LoRA arms over the anchor set
``A`` with the identical r_0 substrate on ``L\\A`` and the identical score-free
``capacity_match`` map -- arms {psi, unif, shuf, rand, global, ada}. ``psi`` uses
``make_feasible_mask(.,score=psi)``; every CONTROL is a single deterministic map of
``x`` under fixed pre-registered seeds (seed_rand, seed_shuf, seed_ada), block
A_glob, warm-up W_ada (seed_ada freezes the AdaLoRA importance OUTSIDE the V(pi)
training-seed expectation). Compute the pool-conditional seed-mean V_hat(pi_arm) on
P_val, the per-control gaps g_hat_c, the paired studentized bootstrap-t lower bounds
L_c(alpha), and the INTERSECTION-UNION decision: reject H0^{R1} iff EVERY one of the
five L_c(alpha) > delta_R1 (no multiplicity penalty; no min-over-controls quantile).
Fail R1 => ``drop_routing_keep_selection``.

Top-level imports pure-python (the IUT machinery is stdlib); ``torch``/``peft``/
``transformers`` lazy inside the guard.
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "run_r1_routing"
PRIMARY_CELL = "R1"

ARMS = ["psi", "unif", "shuf", "rand", "global", "ada"]
CONTROLS = ["unif", "shuf", "rand", "global", "ada"]

# One STATUS.json key per (arm x seed) training cell + the analysis cells. The
# manifest cells are idempotent: a (arm, seed) cell is recorded only after its
# checkpoint + U_train metric are durably written, so --resume skips finished arms.
STAGES = (
    ["ada_warmup_seed_outside_V"]  # once-frozen AdaLoRA importance under seed_ada
    + [f"train::{arm}::seed${{SEED}}" for arm in ARMS]  # expands per seed at run time
    + [
        "validate_J_profile",      # J_{c,l} vs anchor layer-freeze Delta^LOL (drop if fail)
        "policy_value_seed_means",  # V_hat(pi_arm) pool-conditional on P_val
        "paired_gaps",             # g_hat_c (Eq.6-2g)
        "bootstrap_t_lower_bounds",  # L_c(alpha) (Eq.7-Bt)
        "iut_decision_R1",         # reject iff ALL five L_c(alpha) > delta_R1 (Eq.7-IUT)
    ]
)


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "primary_cell": PRIMARY_CELL,
        "closed_testing_node": "G1 (R1)",
        "description": "six-arm masked-LoRA interventional routing policy-value IUT over A",
        "design_refs": ["REDESIGN_v5 §3.5", "Eq.6-0..6-2g", "Eq.7-Bt", "Eq.7-IUT", "§4.1 node G1"],
        "config": str(args.config),
        "modules_imported": [
            "neurotrace_it.layer_function",  # capacity_match, make_feasible_mask, control_mask
            "neurotrace_it.analysis.routing_intervention",  # bootstrap_t, iut_decision
            "neurotrace_it.analysis.layer_attribution",     # psi via frozen B_lambda
            "neurotrace_it.cost_model",                     # compute-match ledger
        ],
        "pool": "P_val (estimate V(pi); pool-conditional, locked partition)",
        "arms": ARMS,
        "controls": CONTROLS,
        "parameters": {
            "domain": "A",
            "r0_substrate": "${R0_SUBSTRATE}",
            "R_tot": "${R_TOT}",
            "r_max": "${R_MAX}",
            "tau_sel": "${TAU_SEL}",
            "delta_R1": "${DELTA_R1}",
            "n_seeds": "${N_SEEDS}",
            "seeds_file": str(guard.SEEDS_FILE),
            "alpha_marginal": "${ALPHA}",
            "n_boot_t": "${B_BT}",
            "capacity_match": {"rule": "largest_remainder", "reads_psi": False},
            "control_seeds": {
                "seed_rand": "${SEED_RAND}",
                "seed_shuf": "${SEED_SHUF}",
                "seed_ada": "${SEED_ADA}",
                "A_glob": "${A_GLOB}",
                "W_ada": "${W_ADA}",
                "ada_warmup_seed_outside_V": True,
            },
        },
        "decision": "INTERSECTION-UNION: reject H0^{R1}=union_c{g_c<=delta_R1} iff every L_c(alpha)>delta_R1",
        "failure_action": "drop_routing_keep_selection",
        "outputs": {
            "arm_checkpoints": "${RUN_DIR}/arms/{arm}/seed{seed}/",
            "u_train_metrics": "${RUN_DIR}/u_train.jsonl",
            "r1_result": "${RUN_DIR}/r1_iut.json",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages_template": STAGES,
        "compute_match": "all arms share r_0, R_tot, capacity_match; matched on params/optimizer-slots/realized-FLOPs",
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    from neurotrace_it import layer_function  # noqa: F401
    from neurotrace_it.analysis import routing_intervention  # noqa: F401
    from neurotrace_it.analysis import layer_attribution  # noqa: F401
    from neurotrace_it import cost_model  # noqa: F401

    raise SystemExit(
        "run_r1_routing: authorized branch intentionally not implemented in this "
        "BUILD-NOW / RUN-LATER packet. server.authorized stays false; the six real "
        "training arms are wired here at run authorization, each (arm, seed) cell "
        f"recorded into {status.path} as it completes (idempotent --resume)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    guard.add_common_arguments(parser)
    args = parser.parse_args()

    plan = build_plan(args)
    checkpoint = guard.guard(args, ENTRYPOINT, plan, list(STAGES))
    if checkpoint is None:
        return guard.emit_dry_run(plan)
    return run_authorized(args, checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())

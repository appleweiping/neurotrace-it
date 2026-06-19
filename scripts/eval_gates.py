#!/usr/bin/env python3
"""Entrypoint: closed-testing decision + honest cost ledger (Phase B-5; §4, §3.7).

Aggregates the realized per-node p-values from R0 / R1 / R2 into the pinned
Maurer-Bretz closed-testing graph {G0,G1,G2t,G2r,G2h,G2c} and emits the
confirmatory rejection set with FWER <= 0.05 (Prop. P1-FWER). Also emits the
two-sided cost ledger (Gate R3) and the G6/G7/J-val precondition statuses.

THIN do-not-run CLI wrapper. Default = DRY RUN (print plan JSON, exit 0, no model,
no GPU). This entrypoint loads NO model in EITHER branch -- it consumes the JSON
artifacts written by the R0/R1/R2 entrypoints -- but it still honours the same hard
guard so it cannot finalize a confirmatory decision unless authorization is present.

What this WOULD do (§4.1b, §4.2): read r0_partial_r2.json, r1_iut.json,
r2_result.json; build ClosedTestInputs (R1 and G2t are SINGLE union-null leaves
whose node p-value is the FULL-union max over ALL components, RE-FIX-5); run the
sequentially-rejective shortcut AND assert it equals the brute-force 63-intersection
closed test; report which of {R0,R1,G2t,G2r,G2h,G2c} reject. cost_model reports the
extraction-parity multiplier (>2.0x => high_cost_analysis) and the training ledger;
a savings claim is licensed ONLY on a measured skip-flag reduction.

Top-level imports pure-python (closed_testing/cost_model are stdlib-only).
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "eval_gates"

STAGES = [
    "load_r0_r1_r2_artifacts",
    "build_closed_test_inputs",     # union-null leaves: full-max for R1, G2t
    "closed_test_shortcut",
    "assert_shortcut_equals_bruteforce",
    "fwer_report",                  # rejection set over {R0,R1,G2t,G2r,G2h,G2c}
    "cost_ledger_r3",               # extraction parity + training ledger + skip-flag
    "preconditions_g6_g7_jval",
]


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "description": "closed-testing FWER decision over the 6 primary cells + honest cost ledger",
        "design_refs": ["REDESIGN_v5 §4.1", "§4.1b", "§4.2", "§3.7", "Prop. P1-FWER"],
        "config": str(args.config),
        "modules_imported": [
            "neurotrace_it.analysis.closed_testing",  # shortcut + brute-force + FWER
            "neurotrace_it.cost_model",               # R3 ledger
            "neurotrace_it.analysis.drift",           # G6 factuality calibrator
        ],
        "primary_cells": list(guard.PRIMARY_CELLS),
        "closed_testing_graph": {
            "nodes": ["G0", "G1", "G2t", "G2r", "G2h", "G2c"],
            "alpha": "${ALPHA}",
            "w_0": 1.0,
            "edges": [["G0", "G1", 1.0], ["G1", "G2t", 1.0]],
            "split": {"w_r": 0.34, "w_h": 0.33, "w_c": 0.33},
            "recycle": 1.0,
            "union_null_leaves": {"R1": "full_union_iut(5)", "G2t": "full_union_iut(baseline_set)"},
        },
        "inputs": {
            "r0": "${R0_RUN_DIR}/r0_partial_r2.json",
            "r1": "${R1_RUN_DIR}/r1_iut.json",
            "r2": "${R2_RUN_DIR}/r2_result.json",
        },
        "parameters": {
            "extraction_parity_kill": 2.0,
            "savings_requires_measured_skip_flag": True,
        },
        "outputs": {
            "decision": "${RUN_DIR}/closed_test_decision.json",
            "cost_ledger": "${RUN_DIR}/cost_ledger_r3.json",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages": STAGES,
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    from neurotrace_it.analysis import closed_testing  # noqa: F401
    from neurotrace_it import cost_model  # noqa: F401
    from neurotrace_it.analysis import drift  # noqa: F401

    raise SystemExit(
        "eval_gates: authorized branch intentionally not implemented in this "
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

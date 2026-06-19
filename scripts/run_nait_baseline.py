#!/usr/bin/env python3
"""Entrypoint: faithful NAIT endpoint baseline (Phase B-0; REDESIGN_v5 §3.3, B6).

THIN do-not-run CLI wrapper. Default = DRY RUN (print plan JSON, exit 0, no model,
no GPU). Heavy work runs ONLY under ``server.authorized == true`` AND
``--i-have-authorization``.

What this WOULD do (§3.3): reproduce NAIT over the FULL released decoder set ``L``
(base Eq.5 layer-sum ``s_NAIT = sum_{l in L} A^(l)(y).v_l``), both the Alg.1
first/last-diff and the token-mean variant, select top-k at budget B (base Eq.6),
and compute the gated SECONDARY 8-anchor restricted score (never the headline
comparator). It also asserts the existing endpoint control ``baselines/nait.py``
(phi_end, concat[start,end]) is the DISTINCT endpoint-only object. The STRONGER
full-L variant is the decisive comparator used by every "beats NAIT" claim.

Top-level imports pure-python; ``torch``/``transformers`` lazy inside the guard.
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "run_nait_baseline"

STAGES = [
    "fit_directions_L_alg1",
    "fit_directions_L_token_mean",
    "score_layerwise_L",
    "select_topk_B",
    "anchor_secondary_gated",   # 8-anchor diagnostic, gated on full-L unit check
    "endpoint_control_distinctness",  # phi_end must NOT equal the layerwise L-sum
]


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "description": "faithful NAIT endpoint baseline over full L (control + decisive comparator)",
        "design_refs": ["REDESIGN_v5 §3.3", "B6", "base Eq.2-6"],
        "config": str(args.config),
        "modules_imported": [
            "neurotrace_it.baselines.nait_layerwise",
            "neurotrace_it.baselines.nait",
        ],
        "models": ["${MODEL_PRIMARY}", "${MODEL_SECONDARY}"],
        "datasets": ["${DS_MATH}", "${DS_CODE}", "${DS_MULTIHOP}"],
        "parameters": {
            "scope": "full_L",
            "nait_variants": ["alg1_L", "token_mean_L"],
            "comparator_rule": "stronger_full_L_variant",
            "budget_B": "${BUDGET_B}",
            "anchor_secondary": True,
            "anchor_size_A": "${A_SIZE}",
        },
        "outputs": {
            "nait_scores": "${RUN_DIR}/nait_scores_L.jsonl",
            "selected_ids": "${RUN_DIR}/nait_selected.json",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages": STAGES,
        "gate_note": "'beats NAIT' wording is gated on the stronger full-L variant; 8-anchor never the headline",
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    from neurotrace_it.baselines import nait_layerwise  # noqa: F401
    from neurotrace_it.baselines import nait  # noqa: F401

    raise SystemExit(
        "run_nait_baseline: authorized branch intentionally not implemented in this "
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

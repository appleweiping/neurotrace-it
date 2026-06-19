#!/usr/bin/env python3
"""Entrypoint: trajectory-signature extraction (Phase B-1) + faithful NAIT directions.

THIN do-not-run CLI wrapper. Default behaviour is a DRY RUN that prints the
resolved plan JSON and exits 0, loading NO model and touching NO GPU. The heavy
extraction runs ONLY when the resolved config has ``server.authorized == true``
AND ``--i-have-authorization`` is passed (the hard guard in ``_run_guard.py``).

What this stage WOULD do under authorization (REDESIGN_v5 Phase B-0 / B-1, §3.3, §3.4):

  * fit the faithful NAIT per-layer PCA directions ``v_l`` over the FULL released
    decoder set ``L`` (both Alg.1 first/last-diff and token-mean variants), persist
    ``s_NAIT`` and ``proj_l`` (which feed ``V_proj`` in R0) -- via
    ``neurotrace_it.baselines.nait_layerwise``;
  * compute the per-anchor trajectory signature ``T(x)=({D_l},{kappa_l})_{l in A}``
    over the anchor set ``A`` for every example in each pool -- via
    ``neurotrace_it.trajectory``; on ``P_dep`` the extractor reads ACTIVATIONS ONLY.

Top-level imports are pure-python/stdlib; ``torch``/``transformers``/``datasets``
are lazy-imported inside the guarded branch only.
"""

from __future__ import annotations

import argparse

import _run_guard as guard

ENTRYPOINT = "extract_signatures"

# Idempotent, restartable stages (one STATUS.json key each).
STAGES = [
    "nait_directions_L",       # v_l over full L (both variants), persisted
    "nait_scores_proj",        # s_NAIT + proj_l over L (feeds V_proj)
    "trajectory_signatures_train",
    "trajectory_signatures_val",
    "trajectory_signatures_dep",  # activations-only on P_dep
]


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "entrypoint": ENTRYPOINT,
        "description": "trajectory-signature extraction + faithful NAIT directions (Phase B-0/B-1)",
        "design_refs": ["REDESIGN_v5 §3.3", "§3.4", "Algorithm box Phase B-0/B-1"],
        "config": str(args.config),
        "frozen_configs": {
            "v5_routing": str(guard.LATTICE_V5),
            "v4_models_datasets": str(guard.LATTICE_V4),
            "compute_budget": str(guard.COMPUTE_BUDGET),
            "seeds": str(guard.SEEDS_FILE),
        },
        "modules_imported": [
            "neurotrace_it.baselines.nait_layerwise",
            "neurotrace_it.trajectory",
            "neurotrace_it.analysis.pool_firewall",
        ],
        "models": ["${MODEL_PRIMARY}", "${MODEL_SECONDARY}"],
        "datasets": ["${DS_MATH}", "${DS_CODE}", "${DS_MULTIHOP}"],
        "parameters": {
            "anchor_layers_A": "${A_SIZE}",
            "sw2_projections_K": "${K_SW2}",
            "subsample_cap": "${SUBSAMPLE_CAP}",
            "nait_variants": ["alg1_L", "token_mean_L"],
            "nait_anchor_secondary": True,
        },
        "outputs": {
            "nait_directions": "${RUN_DIR}/nait_directions.json",
            "trajectory_signatures": "${RUN_DIR}/signatures_{train,val,dep}.jsonl",
            "status": "${RUN_DIR}/STATUS.json",
        },
        "stages": STAGES,
        "pool_firewall": "P_dep reads activations only; no Y_obs/U_train on P_dep",
    }


def run_authorized(args: argparse.Namespace, status: "guard.StatusCheckpoint") -> int:
    # HEAVY, lazy imports -- reached ONLY past the hard guard.
    from neurotrace_it.baselines import nait_layerwise  # noqa: F401
    from neurotrace_it import trajectory  # noqa: F401
    from neurotrace_it.analysis import pool_firewall  # noqa: F401

    raise SystemExit(
        "extract_signatures: authorized branch is intentionally not implemented in "
        "this BUILD-NOW / RUN-LATER packet. server.authorized must stay false; the "
        "real extraction is wired here at run authorization, recording each of "
        f"{STAGES} into {status.path} as it completes."
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

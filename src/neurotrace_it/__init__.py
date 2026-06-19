"""NeuroTrace-IT research contract helpers.

The V1 surface (contracts, metrics, V1 schemas) is unchanged. The NOVEL CORE
(REDESIGN_v4) is exposed additively: the NAIT endpoint baseline, the trajectory
operator, the residualized cross-fit/permutation test, the matched-pair miner,
the monotone-submodular selector, and the V2 auditable signature record. All are
build-now / run-later: pure stdlib, no model load, no server call.
"""

from .baselines import (
    EndpointNeuronSignature,
    NaitSelectionResult,
    endpoint_score,
    endpoint_signature,
    nait_select,
)
from .contracts import EvidenceTier, ProjectContract, validate_manifest
from .metrics import passes_cost_gate, passes_drift_gate, retention_adjusted_gain
from .schemas import SelectionManifest, TrajectorySignature, validate_selection_manifest
from .schemas_v2 import (
    ComputeLedgerRecord,
    ControlProvenance,
    PoolHashes,
    RouterOutputs,
    RoutingPolicyValue,
    SelectionManifestV2,
    TrajectorySignatureV2,
    compute_trajectory_hash_v2,
    validate_selection_manifest_v2,
)
from .layer_function import (
    capacity_match,
    control_mask,
    frozen_layer_ablation_profile,
    leave_one_layer_redistribute,
    make_feasible_mask,
    routing_policy,
    validate_J_against_freeze,
)
from .cost_model import (
    ArmLedger,
    compute_match_ledger,
    extraction_parity_check,
    gate1_multiplier,
    gate1b_deployability,
    routing_training_savings,
)
from .selection import (
    CoverageObjective,
    GreedySelection,
    example_utility,
    greedy_submodular_select,
)
from .trajectory import (
    TrajectoryFeatures,
    sliced_wasserstein2,
    trajectory_curvature,
    trajectory_signature,
)

__all__ = [
    "EvidenceTier",
    "ProjectContract",
    "SelectionManifest",
    "TrajectorySignature",
    "passes_cost_gate",
    "passes_drift_gate",
    "retention_adjusted_gain",
    "validate_manifest",
    "validate_selection_manifest",
    # --- NOVEL CORE (additive) ---
    "EndpointNeuronSignature",
    "NaitSelectionResult",
    "endpoint_signature",
    "endpoint_score",
    "nait_select",
    "TrajectoryFeatures",
    "sliced_wasserstein2",
    "trajectory_curvature",
    "trajectory_signature",
    "CoverageObjective",
    "GreedySelection",
    "example_utility",
    "greedy_submodular_select",
    "SelectionManifestV2",
    "TrajectorySignatureV2",
    "compute_trajectory_hash_v2",
    "validate_selection_manifest_v2",
    # --- v5 routing primitives + cost model + schema records (additive) ---
    "ComputeLedgerRecord",
    "ControlProvenance",
    "PoolHashes",
    "RouterOutputs",
    "RoutingPolicyValue",
    "capacity_match",
    "control_mask",
    "frozen_layer_ablation_profile",
    "leave_one_layer_redistribute",
    "make_feasible_mask",
    "routing_policy",
    "validate_J_against_freeze",
    "ArmLedger",
    "compute_match_ledger",
    "extraction_parity_check",
    "gate1_multiplier",
    "gate1b_deployability",
    "routing_training_savings",
]

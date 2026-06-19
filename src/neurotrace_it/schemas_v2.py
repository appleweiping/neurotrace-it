"""Versioned, auditable trajectory signature record -- additive V2 (REDESIGN_v4 §2.10).

This module **extends** the V1 contracts of :mod:`neurotrace_it.schemas` WITHOUT
modifying them. ``TrajectorySignatureV2`` persists the *estimands* (not just ids /
hashes): the endpoint signature ``phi_end`` (the FULL control), per-layer ``D_l``
and ``kappa_l``, the SW2 projection seeds, the subsample masks, selection scores,
drift estimates, calibrator provenance, router outputs, the ridge penalty
``lambda_ridge``, the Y-reliability block, and the cluster-assignment hash. The
``trajectory_hash`` is an integrity check that **recomputes from the stored
estimands**.

Back-compatibility contract
---------------------------
* V1 :class:`neurotrace_it.schemas.TrajectorySignature` records stay valid; this
  module does not touch them.
* The v4 fields (``ridge_lambda``, ``y_reliability``, ``cluster_assignment_hash``)
  are **additive optional** fields -- a V2 record without them still validates.
* :func:`validate_selection_manifest_v2` accepts a manifest whose ``signatures``
  are V1 *or* V2 records; V1 records are validated by the V1 validator, V2 records
  additionally require non-empty ``D`` / ``kappa`` over ``layer_ids``, a present
  ``endpoint_signature``, present ``projection_seeds``, and a recomputing hash.
* ``server_authorized`` stays ``false``; ``endpoint_neuron_selection`` stays a
  required baseline.

DO-NOT-RUN: pure stdlib; no model load, no server call, no training.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping

from .schemas import TrajectorySignature, validate_selection_manifest

SCHEMA_VERSION_V2 = "2.0.0"

__all__ = [
    "SCHEMA_VERSION_V2",
    "CalibratorProvenance",
    "YReliability",
    "RouterOutputs",
    "PoolHashes",
    "ControlProvenance",
    "RoutingPolicyValue",
    "ComputeLedgerRecord",
    "TrajectorySignatureV2",
    "SelectionManifestV2",
    "compute_trajectory_hash_v2",
    "validate_selection_manifest_v2",
]


@dataclass(frozen=True)
class CalibratorProvenance:
    """Strictly-proper calibrator provenance (Brier; §2.6, G6)."""

    rule: str = "brier"
    threshold: float | None = None      # c*
    reliability_hash: str = ""
    g6_pass: bool = False


@dataclass(frozen=True)
class YReliability:
    """Y-reliability gate evidence (G7; §2.6, fix c) -- additive optional block."""

    icc: float | None = None                 # ICC(2,1) across >=3 seeds
    rho_proxy_retrain: float | None = None   # Spearman rho(Y, Delta_retrain)
    g7_pass: bool = False


@dataclass(frozen=True)
class RouterOutputs:
    """Capacity-matched layer router outputs (§2.9 / v5 §3.5; ablation only).

    The v4 fields (``layer_mask``, ``importance_profile_hash``, ``total_rank``,
    ``rank_per_layer``) are preserved. The v5 additive fields persist the
    anchor-only routing policy outputs (``anchor_mask`` / ``rank_per_anchor`` over
    the anchor set ``A``, ``total_rank_A`` = ``R_tot`` on ``A``, the shared
    ``r0_substrate`` carried by every ``L\\A`` layer, and the frozen
    ``J_profile_hash``); all default to empty/None so old records still validate.
    """

    layer_mask: tuple[int, ...] = ()         # m(x) in {0,1}^L over layer_ids
    importance_profile_hash: str = ""        # I_{c,l} profile hash (v4)
    total_rank: int | None = None            # R_tot (v4)
    rank_per_layer: tuple[int, ...] = ()     # r_l after reallocation (v4)
    # --- v5 additive (anchor-only routing policy; absent => still valid) ---
    anchor_mask: tuple[int, ...] = ()        # m_A(x) in {0,1}^A over the anchor set A
    rank_per_anchor: tuple[int, ...] = ()    # r_l on A from capacity_match
    total_rank_A: int | None = None          # sum_{l in A} r_l = R_tot
    r0_substrate: int | None = None          # shared baseline rank on every L\A layer
    J_profile_hash: str = ""                 # frozen J_{c,l} ablation profile hash


@dataclass(frozen=True)
class PoolHashes:
    """Locked three-pool partition hashes (v5 §3.1, R3-B4; additive optional)."""

    train: str = ""
    val: str = ""
    dep: str = ""
    partition_hash: str = ""


@dataclass(frozen=True)
class ControlProvenance:
    """R1 control-policy provenance (v5 §3.5, RE-FIX-3/4; additive optional).

    Persists the fixed control seeds and the ONCE-frozen AdaLoRA importance so
    every control is a deterministic map of ``x``. ``seed_ada`` is the SEPARATE
    pre-registered seed that freezes the AdaLoRA warm-up OUTSIDE the ``V(pi)``
    training-seed expectation (``ada_warmup_seed_outside_V``).
    """

    seed_rand: int | None = None
    seed_shuf: int | None = None
    seed_ada: int | None = None
    A_glob_hash: str = ""
    W_ada: int | None = None
    ada_importance_hash: str = ""
    ada_warmup_seed_outside_V: bool = True


@dataclass(frozen=True)
class RoutingPolicyValue:
    """Gate R1 routing policy-value outcome (v5 §3.5, §4.2; additive optional)."""

    R1_pass: bool = False
    gaps_per_control: Mapping[str, float] = field(default_factory=dict)
    margin: float | None = None                            # delta_R1
    per_contrast_lower_bounds: Mapping[str, float] = field(default_factory=dict)
    iut_simultaneous_bound: float | None = None            # L_{R1} = min_c L_c(alpha)


@dataclass(frozen=True)
class ComputeLedgerRecord:
    """Per-arm compute-match ledger record (v5 §3.7; additive optional)."""

    param_count: int | None = None
    optimizer_slots: int | None = None
    realized_flops: float | None = None
    wall_clock_s: float | None = None
    batch_policy: str = ""
    skip_flag: bool = False
    matched: bool = False


@dataclass(frozen=True)
class TrajectorySignatureV2:
    """V2 trajectory signature that PERSISTS the estimands (§2.10).

    Required estimands
    ------------------
    * ``endpoint_signature`` -- ``phi_end`` (Eq. 1), the FULL control.
    * ``magnitude`` / ``curvature`` -- ``{l: D_l}`` / ``{l: kappa_l}`` over
      ``layer_ids`` (Eq. 4 / Eq. 5).
    * ``projection_seeds`` -- ``{l: (seed, ...)}`` SW2 seeds (recomputability).

    Additive / optional estimands
    -----------------------------
    * ``slice_masks`` (persisted subsample masks), ``alignment_scores``
      (``rho_T``, residual ``beta_T``, ``lambda_ridge``), ``selection_scores``
      (``u(x)``, marginal gain), ``drift_estimates`` (``r_hat``, ``f_hat``),
      ``calibrator``, ``y_reliability``, ``router``, ``cluster_assignment_hash``,
      ``ridge_lambda``.

    The ``trajectory_hash`` recomputes from ALL persisted estimands via
    :func:`compute_trajectory_hash_v2`.
    """

    example_id: str
    layer_ids: tuple[int, ...]
    step_count: int
    token_count: int
    endpoint_signature: tuple[float, ...]
    magnitude: Mapping[int, float]               # D_l
    curvature: Mapping[int, float]               # kappa_l
    projection_seeds: Mapping[int, tuple[int, ...]]
    trajectory_hash: str = ""
    schema_version: str = SCHEMA_VERSION_V2
    # --- additive optional estimands (V2 deltas; absent => still valid) ---
    slice_masks: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    alignment_scores: Mapping[str, float] = field(default_factory=dict)
    selection_scores: Mapping[str, float] = field(default_factory=dict)
    drift_estimates: Mapping[str, float] = field(default_factory=dict)
    calibrator: CalibratorProvenance | None = None
    y_reliability: YReliability | None = None
    router: RouterOutputs | None = None
    cluster_assignment_hash: str = ""
    ridge_lambda: float | None = None
    # --- v5 additive optional estimands (absent => byte-identical hash to v4) ---
    psi_per_anchor: Mapping[int, float] = field(default_factory=dict)
    policy_value: Mapping[str, float] = field(default_factory=dict)  # per-arm V_hat
    pool_hashes: PoolHashes | None = None
    nuisance_map_hash: str = ""                                      # (B_lambda, b_lambda_Y)
    routing_policy_value: RoutingPolicyValue | None = None
    compute_ledger: Mapping[str, ComputeLedgerRecord] = field(default_factory=dict)
    control_provenance: ControlProvenance | None = None

    def recompute_hash(self) -> str:
        return compute_trajectory_hash_v2(self)

    def with_hash(self) -> "TrajectorySignatureV2":
        """Return a copy whose ``trajectory_hash`` is the recomputed integrity hash."""

        data = asdict(self)
        data["trajectory_hash"] = ""
        rehashed = TrajectorySignatureV2(**_rebuild_nested(data))
        object.__setattr__(rehashed, "trajectory_hash", rehashed.recompute_hash())
        return rehashed


def _rebuild_nested(data: dict) -> dict:
    """Reconstruct dataclass fields from an ``asdict`` mapping (frozen-safe copy)."""

    if data.get("calibrator") is not None:
        data["calibrator"] = CalibratorProvenance(**data["calibrator"])
    if data.get("y_reliability") is not None:
        data["y_reliability"] = YReliability(**data["y_reliability"])
    if data.get("router") is not None:
        router = dict(data["router"])
        router["layer_mask"] = tuple(router.get("layer_mask", ()))
        router["rank_per_layer"] = tuple(router.get("rank_per_layer", ()))
        router["anchor_mask"] = tuple(router.get("anchor_mask", ()))
        router["rank_per_anchor"] = tuple(router.get("rank_per_anchor", ()))
        data["router"] = RouterOutputs(**router)
    if data.get("pool_hashes") is not None:
        data["pool_hashes"] = PoolHashes(**data["pool_hashes"])
    if data.get("routing_policy_value") is not None:
        data["routing_policy_value"] = RoutingPolicyValue(**data["routing_policy_value"])
    if data.get("control_provenance") is not None:
        data["control_provenance"] = ControlProvenance(**data["control_provenance"])
    if data.get("compute_ledger"):
        data["compute_ledger"] = {
            k: ComputeLedgerRecord(**v) for k, v in data["compute_ledger"].items()
        }
    # Normalize tuple-typed fields that asdict turned into lists.
    data["layer_ids"] = tuple(data["layer_ids"])
    data["endpoint_signature"] = tuple(data["endpoint_signature"])
    data["projection_seeds"] = {k: tuple(v) for k, v in data["projection_seeds"].items()}
    data["slice_masks"] = {k: tuple(v) for k, v in data.get("slice_masks", {}).items()}
    return data


def _canonical(value):
    """JSON-canonicalize estimands with sorted keys and float rounding."""

    if isinstance(value, float):
        # Round to a stable precision so the recomputed hash is reproducible.
        return round(value, 12)
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def compute_trajectory_hash_v2(signature: TrajectorySignatureV2) -> str:
    """Integrity hash that recomputes from the persisted estimands (§2.10).

    Hashes a canonical JSON of all estimands EXCEPT the stored ``trajectory_hash``
    itself, so ``signature.recompute_hash()`` can be compared to the stored value
    to detect tampering or estimand drift.
    """

    payload = {
        "example_id": signature.example_id,
        "layer_ids": list(signature.layer_ids),
        "step_count": signature.step_count,
        "token_count": signature.token_count,
        "endpoint_signature": _canonical(list(signature.endpoint_signature)),
        "magnitude": _canonical(dict(signature.magnitude)),
        "curvature": _canonical(dict(signature.curvature)),
        "projection_seeds": {str(k): list(v) for k, v in signature.projection_seeds.items()},
        "slice_masks": {str(k): list(v) for k, v in signature.slice_masks.items()},
        "alignment_scores": _canonical(dict(signature.alignment_scores)),
        "selection_scores": _canonical(dict(signature.selection_scores)),
        "drift_estimates": _canonical(dict(signature.drift_estimates)),
        "calibrator": _canonical(asdict(signature.calibrator)) if signature.calibrator else None,
        "y_reliability": _canonical(asdict(signature.y_reliability)) if signature.y_reliability else None,
        "router": _canonical(asdict(signature.router)) if signature.router else None,
        "cluster_assignment_hash": signature.cluster_assignment_hash,
        "ridge_lambda": _canonical(signature.ridge_lambda) if signature.ridge_lambda is not None else None,
        "schema_version": signature.schema_version,
    }

    # v5 additive fields: fold into the hash ONLY when at least one is non-default,
    # so a v4-shaped record produces a BYTE-IDENTICAL hash to before (back-compat).
    v5_ext = _v5_extension_payload(signature)
    if v5_ext is not None:
        payload["v5_extension"] = v5_ext

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _v5_extension_payload(signature: "TrajectorySignatureV2") -> dict | None:
    """Canonical v5-extension sub-payload, or ``None`` when all v5 fields are default.

    Returning ``None`` for an all-default record keeps the hash byte-identical to
    the v4 hashing, so existing records validate unchanged.
    """

    has_any = (
        bool(signature.psi_per_anchor)
        or bool(signature.policy_value)
        or signature.pool_hashes is not None
        or bool(signature.nuisance_map_hash)
        or signature.routing_policy_value is not None
        or bool(signature.compute_ledger)
        or signature.control_provenance is not None
    )
    if not has_any:
        return None
    return {
        "psi_per_anchor": _canonical(dict(signature.psi_per_anchor)),
        "policy_value": _canonical(dict(signature.policy_value)),
        "pool_hashes": _canonical(asdict(signature.pool_hashes))
        if signature.pool_hashes
        else None,
        "nuisance_map_hash": signature.nuisance_map_hash,
        "routing_policy_value": _canonical(asdict(signature.routing_policy_value))
        if signature.routing_policy_value
        else None,
        "compute_ledger": _canonical(
            {k: asdict(v) for k, v in signature.compute_ledger.items()}
        ),
        "control_provenance": _canonical(asdict(signature.control_provenance))
        if signature.control_provenance
        else None,
    }


@dataclass(frozen=True)
class SelectionManifestV2:
    """Selection manifest carrying V1 and/or V2 signatures (additive).

    Mirrors :class:`neurotrace_it.schemas.SelectionManifest` but allows the
    ``signatures`` tuple to mix V1 :class:`TrajectorySignature` and V2
    :class:`TrajectorySignatureV2` records. ``server_authorized`` stays ``false``.
    """

    project: str
    candidate_pool_hash: str
    selected_example_ids: tuple[str, ...]
    baseline_ids: tuple[str, ...]
    signatures: tuple[object, ...]  # TrajectorySignature | TrajectorySignatureV2
    server_authorized: bool = False
    schema_version: str = SCHEMA_VERSION_V2


def validate_selection_manifest_v2(manifest: SelectionManifestV2) -> list[str]:
    """Validate a V2 manifest (back-compatible with V1 signatures).

    Reuses the V1 manifest-level invariants (project, server-not-authorized,
    endpoint baseline required, non-empty selection, unique signature ids, every
    selected example covered) by delegating to
    :func:`neurotrace_it.schemas.validate_selection_manifest` on a V1 view, then
    adds the V2 estimand-persistence checks for any V2 records: non-empty ``D`` /
    ``kappa`` over ``layer_ids``, present ``endpoint_signature``, present
    ``projection_seeds``, and a **recomputing** ``trajectory_hash``.
    """

    errors: list[str] = []

    # Manifest-level invariants (delegate to V1 validator using a V1 projection).
    v1_signatures: list[TrajectorySignature] = []
    for signature in manifest.signatures:
        if isinstance(signature, TrajectorySignature):
            v1_signatures.append(signature)
        elif isinstance(signature, TrajectorySignatureV2):
            v1_signatures.append(
                TrajectorySignature(
                    example_id=signature.example_id,
                    layer_ids=signature.layer_ids,
                    step_count=signature.step_count,
                    token_count=signature.token_count,
                    trajectory_hash=signature.trajectory_hash,
                )
            )
        else:  # pragma: no cover - defensive
            errors.append(f"unknown signature type: {type(signature).__name__}")

    from .schemas import SelectionManifest  # local import: additive, no cycle at module load

    v1_view = SelectionManifest(
        project=manifest.project,
        candidate_pool_hash=manifest.candidate_pool_hash,
        selected_example_ids=manifest.selected_example_ids,
        baseline_ids=manifest.baseline_ids,
        signatures=tuple(v1_signatures),
        server_authorized=manifest.server_authorized,
    )
    errors.extend(validate_selection_manifest(v1_view))

    # V2-specific estimand-persistence checks.
    for signature in manifest.signatures:
        if not isinstance(signature, TrajectorySignatureV2):
            continue
        example_id = signature.example_id
        if not signature.endpoint_signature:
            errors.append(f"signature {example_id} missing endpoint_signature (phi_end)")
        if not signature.projection_seeds:
            errors.append(f"signature {example_id} missing projection_seeds")
        for layer in signature.layer_ids:
            if layer not in signature.magnitude:
                errors.append(f"signature {example_id} missing D_l for layer {layer}")
            if layer not in signature.curvature:
                errors.append(f"signature {example_id} missing kappa_l for layer {layer}")
        if not signature.magnitude:
            errors.append(f"signature {example_id} has empty D (magnitude) map")
        if not signature.curvature:
            errors.append(f"signature {example_id} has empty kappa (curvature) map")
        recomputed = signature.recompute_hash()
        if signature.trajectory_hash != recomputed:
            errors.append(
                f"signature {example_id} trajectory_hash does not recompute "
                "from persisted estimands (integrity check failed)"
            )
    return errors

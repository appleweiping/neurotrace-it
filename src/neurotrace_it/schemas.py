from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectorySignature:
    example_id: str
    layer_ids: tuple[int, ...]
    step_count: int
    token_count: int
    trajectory_hash: str


@dataclass(frozen=True)
class SelectionManifest:
    project: str
    candidate_pool_hash: str
    selected_example_ids: tuple[str, ...]
    baseline_ids: tuple[str, ...]
    signatures: tuple[TrajectorySignature, ...]
    server_authorized: bool = False


def validate_selection_manifest(manifest: SelectionManifest) -> list[str]:
    errors: list[str] = []
    if manifest.project != "neurotrace-it":
        errors.append("project must be neurotrace-it")
    if manifest.server_authorized:
        errors.append("server_authorized must remain false in local manifests")
    if "endpoint_neuron_selection" not in manifest.baseline_ids:
        errors.append("endpoint_neuron_selection baseline is required")
    if not manifest.selected_example_ids:
        errors.append("selected_example_ids must not be empty")
    signature_ids = {signature.example_id for signature in manifest.signatures}
    if len(signature_ids) != len(manifest.signatures):
        errors.append("signature example ids must be unique")
    missing = [example_id for example_id in manifest.selected_example_ids if example_id not in signature_ids]
    if missing:
        errors.append(f"selected examples missing signatures: {missing}")
    for signature in manifest.signatures:
        if not signature.layer_ids:
            errors.append(f"signature {signature.example_id} has no layers")
        if signature.step_count <= 0 or signature.token_count <= 0:
            errors.append(f"signature {signature.example_id} must have positive step/token counts")
        if not signature.trajectory_hash:
            errors.append(f"signature {signature.example_id} missing trajectory hash")
    return errors


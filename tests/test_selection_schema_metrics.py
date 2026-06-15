from neurotrace_it.metrics import passes_cost_gate, passes_drift_gate, retention_adjusted_gain
from neurotrace_it.schemas import (
    SelectionManifest,
    TrajectorySignature,
    validate_selection_manifest,
)


def test_selection_manifest_requires_endpoint_baseline():
    manifest = SelectionManifest(
        project="neurotrace-it",
        candidate_pool_hash="abc",
        selected_example_ids=("ex1",),
        baseline_ids=("random_subset",),
        signatures=(
            TrajectorySignature(
                example_id="ex1",
                layer_ids=(1, 2),
                step_count=3,
                token_count=12,
                trajectory_hash="hash",
            ),
        ),
    )
    assert "endpoint_neuron_selection baseline is required" in validate_selection_manifest(manifest)


def test_selection_manifest_blocks_local_server_authorization():
    manifest = SelectionManifest(
        project="neurotrace-it",
        candidate_pool_hash="abc",
        selected_example_ids=("ex1",),
        baseline_ids=("endpoint_neuron_selection",),
        signatures=(
            TrajectorySignature(
                example_id="ex1",
                layer_ids=(1, 2),
                step_count=3,
                token_count=12,
                trajectory_hash="hash",
            ),
        ),
        server_authorized=True,
    )
    assert "server_authorized must remain false in local manifests" in validate_selection_manifest(manifest)


def test_retention_and_cost_gates():
    assert retention_adjusted_gain(0.05, 0.01, drift_weight=2.0) == 0.030000000000000002
    assert passes_drift_gate(0.005)
    assert not passes_drift_gate(0.02)
    assert passes_cost_gate(1.8)
    assert not passes_cost_gate(2.5)


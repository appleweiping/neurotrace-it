from neurotrace_it.contracts import EvidenceTier, validate_manifest


def test_evidence_tiers_are_explicit():
    assert EvidenceTier.PAPER_RESULT.value == "paper_result"
    assert EvidenceTier.PILOT.value == "pilot"


def test_manifest_requires_endpoint_baseline():
    manifest = {
        "project": "neurotrace-it",
        "server": {"authorized": False},
        "seeds": {"paper_minimum": 20},
        "baselines": ["random_subset", "full_data_it", "quality_score_selection"],
    }
    errors = validate_manifest(manifest)
    assert "endpoint neuron selection baseline is required" in errors


def test_manifest_rejects_server_authorization_in_scaffold():
    manifest = {
        "project": "neurotrace-it",
        "server": {"authorized": True},
        "seeds": {"paper_minimum": 20},
        "baselines": [
            "random_subset",
            "full_data_it",
            "endpoint_neuron_selection",
        ],
    }
    errors = validate_manifest(manifest)
    assert "initial scaffold must not authorize server execution" in errors


from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurotrace_it.contracts import NEUROTRACE_CONTRACT, required_paths


def require_markers(path: Path, markers: list[str], label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        for marker in missing:
            print(f"missing {label} marker: {marker}")
        return False
    return True


def main() -> int:
    missing = [path for path in required_paths(ROOT) if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing required doc: {path}")
        return 1

    config = ROOT / "configs" / "experiments" / "formal_neurotrace_it.yaml"
    text = config.read_text(encoding="utf-8")
    missing_markers = [
        marker for marker in NEUROTRACE_CONTRACT.required_config_markers if marker not in text
    ]
    if missing_markers:
        for marker in missing_markers:
            print(f"missing config marker: {marker}")
        return 1

    seeds = [
        line.strip()
        for line in (ROOT / "configs" / "seeds" / "paper_20.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if seeds != [str(index) for index in range(20)]:
        print("paper seed manifest must be exactly 0..19")
        return 1

    compute_text = (ROOT / "configs" / "compute" / "first_gate_budget.yaml").read_text(encoding="utf-8")
    for marker in ["buffer_percent: 30", "wall_clock_hours:", "storage_gb:"]:
        if marker not in compute_text:
            print(f"missing compute marker: {marker}")
            return 1

    if not require_markers(
        ROOT / "configs" / "baselines" / "baseline_registry.yaml",
        [
            "endpoint_neuron_selection:",
            "NAIT-style endpoint-neuron baseline",
            "https://arxiv.org/abs/2603.13201",
            "implementation_source:",
            "license:",
            "tuning_policy:",
            "input_access:",
            "fairness:",
        ],
        "baseline provenance",
    ):
        return 1

    if not require_markers(
        ROOT / "configs" / "experiments" / "first_gate.yaml",
        [
            "trajectory_endpoint_margin:",
            "endpoint_neuron_selection",
            "contamination_leakage_audit:",
            "max_retention_drift_abs:",
            "max_hallucination_drift_abs:",
            "authorized: false",
        ],
        "first-gate",
    ):
        return 1

    if not require_markers(
        ROOT / "schemas" / "selection_manifest.schema.json",
        [
            "\"server_authorized\"",
            "\"const\": false",
            "\"baseline_ids\"",
            "endpoint_neuron_selection",
        ],
        "selection schema",
    ):
        return 1

    print("NeuroTrace-IT local contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

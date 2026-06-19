"""Legacy v1-era selection-gate kernels (NOT the v5 confirmatory thresholds).

These small helpers predate the LATTICE-R v5 redesign. Only
``retention_adjusted_gain`` is still load-bearing -- it is the shared
gain-minus-drift kernel reused by ``selection.py`` and ``outcome_y.py``.

``passes_drift_gate`` / ``passes_cost_gate`` and their default cut-points
(``max_drift_abs=0.01`` was a v1 absolute-drift ceiling; ``max_multiplier=2.0``
a v1 cost-ratio ceiling) are **STALE and NON-CONFIRMATORY**: the v5 confirmatory
pipeline does NOT use them. v5 decides retention/hallucination/cost via the
LOCKED non-inferiority margins in ``configs/experiments/lattice_v5.yaml``
(``R2_margins.delta_ret/delta_hall/delta_cost``) tested by the bootstrap-``t``
gates (G2r/G2h/G2c, see ``analysis/matched_budget.py`` and
``analysis/routing_intervention.py``) -- NOT by these fixed-default booleans.
The defaults are retained ONLY so the legacy unit check
(``test_selection_schema_metrics.test_retention_and_cost_gates``) stays green;
they are not a pre-registered v5 decision threshold and must not be read as one.
"""

from __future__ import annotations


def retention_adjusted_gain(target_gain: float, retention_drift: float, *, drift_weight: float = 1.0) -> float:
    return target_gain - drift_weight * retention_drift


def passes_drift_gate(retention_drift_abs: float, *, max_drift_abs: float = 0.01) -> bool:
    """STALE v1 helper -- see module docstring. v5 uses ``R2_margins.delta_ret``."""

    return retention_drift_abs <= max_drift_abs


def passes_cost_gate(cost_multiplier: float, *, max_multiplier: float = 2.0) -> bool:
    """STALE v1 helper -- see module docstring. v5 uses ``R2_margins.delta_cost``."""

    return cost_multiplier <= max_multiplier


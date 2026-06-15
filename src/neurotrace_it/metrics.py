from __future__ import annotations


def retention_adjusted_gain(target_gain: float, retention_drift: float, *, drift_weight: float = 1.0) -> float:
    return target_gain - drift_weight * retention_drift


def passes_drift_gate(retention_drift_abs: float, *, max_drift_abs: float = 0.01) -> bool:
    return retention_drift_abs <= max_drift_abs


def passes_cost_gate(cost_multiplier: float, *, max_multiplier: float = 2.0) -> bool:
    return cost_multiplier <= max_multiplier


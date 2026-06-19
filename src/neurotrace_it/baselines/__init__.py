"""Baseline selectors for NeuroTrace-IT.

This subpackage holds faithful, matched-budget reimplementations of the
comparison baselines named in ``configs/baselines/baseline_registry.yaml``.

Deliverable #1 (per ``docs/redesign/REDESIGN_v4.md`` §2.1, §5) is the
endpoint-neuron (NAIT-style) baseline in :mod:`neurotrace_it.baselines.nait`.
It plays two roles in the registered design: the decisive matched-budget
comparator AND the FULL endpoint control block ``phi_end`` that the co-primary
ridge-partialling-out regression orthogonalizes against. No proposed-method
operator is measured for "gain over endpoints" until this baseline exists and
passes its unit tests.

DO-NOT-RUN: this module is pure, dependency-free numerical code. It performs no
model load, no server call, and no I/O. ``server.authorized`` stays ``false``.
"""

from __future__ import annotations

from .nait import (
    EndpointNeuronSignature,
    NaitSelectionResult,
    endpoint_score,
    endpoint_signature,
    nait_select,
    select_score,
)
from .nait_layerwise import (
    LayerwiseNaitModel,
    NaitLayerwiseScores,
    fit_layer_directions,
    layer_difference,
    principal_direction,
    score_layerwise,
    select_layerwise,
    sign_align,
    token_mean_summary,
)

__all__ = [
    "EndpointNeuronSignature",
    "NaitSelectionResult",
    "endpoint_score",
    "endpoint_signature",
    "nait_select",
    "select_score",
    # --- v5 faithful layerwise NAIT over L (additive) ---
    "LayerwiseNaitModel",
    "NaitLayerwiseScores",
    "fit_layer_directions",
    "layer_difference",
    "principal_direction",
    "score_layerwise",
    "select_layerwise",
    "sign_align",
    "token_mean_summary",
]

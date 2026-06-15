# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 8 | The sharpened claim is trajectory-vs-endpoint selection with retention/hallucination drift gates, not generic activation selection. |
| Feasibility | 8 | The first gate is bounded to matched-budget endpoint-vs-trajectory selection before full training scale. |
| Clarity | 8 | Endpoint baseline, target gain, retention drift, cost, and layer ablations define success. |
| Impact | 8 | Data selection that preserves capabilities while reducing drift is central to practical instruction tuning. |
| Testability | 9 | The project is killed if trajectory features fail against endpoint features under the same budget. |

Average: 8.2/10.

## Kill Argument

The strongest rejection argument is that NeuroTrace-IT is just NAIT with a more
expensive feature. If trajectory signatures do not beat endpoint activation
selection under identical data budgets, LoRA budgets, and retention metrics, the
project should pivot or downgrade to an analysis paper.

## Differentiation

| Closest Approach | Overlap | Differentiation | Strength |
| --- | --- | --- | --- |
| Neuron-aware data selection | Uses neuron activations for IT selection. | Uses full trajectory distributions and retention drift gates. | Medium |
| Influence/gradient selection | Selects data by expected downstream effect. | Uses neural process signatures and layer-function compatibility. | Medium |
| Layer-selective tuning | Adapts subsets of layers. | Layer policy is tied to data trajectory evidence. | Medium |

## Verdict

VERDICT: PROCEED
CONFIDENCE: Medium
BLOCKING ISSUE: None for local non-server initialization; server work remains gated by endpoint-baseline design review.
NEXT ACTION: Draft trajectory signature schema and submit the endpoint-vs-trajectory plan for ARIS review.

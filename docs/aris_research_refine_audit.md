# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 10 | The endpoint-vs-trajectory gate directly prevents NAIT relabeling and makes trajectory value the core claim. |
| Feasibility | 9 | The first gate is bounded with budget, cost cap, and raw-activation policy; server preflight remains external. |
| Clarity | 10 | Target gain, retention drift, hallucination drift, endpoint baseline, cost, and layer ablations define success. |
| Impact | 10 | Reliable instruction-tuning data selection is broadly useful if it preserves capability while improving target tasks. |
| Testability | 10 | Endpoint, drift, cost, and layer-policy gates can kill or narrow the claim. |

Average: 9.8/10.

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

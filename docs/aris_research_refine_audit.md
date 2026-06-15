# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 7 | Extends neuron-aware selection to trajectories and layer-function compatibility, but must clearly beat endpoint methods. |
| Feasibility | 6 | Activation extraction over trajectories is expensive and needs careful engineering. |
| Clarity | 7 | Main comparisons against endpoint selection and uniform adaptation are clear. |
| Impact | 7 | Better data selection and retention control matter for practical instruction tuning. |
| Testability | 7 | Early kill gates can falsify whether trajectory information adds value. |

Average: 6.8/10.

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

VERDICT: ITERATE BEFORE FULL EXECUTION
CONFIDENCE: Medium
BLOCKING ISSUE: Need a sharper proof-of-value pilot design against endpoint selection.
NEXT ACTION: Formalize trajectory signature and run only local contract validation until ARIS plan passes.


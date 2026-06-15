# Literature Boundary

## Must-Cite / Must-Beat Work

- NAIT / Neuron-Aware Data Selection in Instruction Tuning
  (`https://arxiv.org/abs/2603.13201`,
  `https://openreview.net/forum?id=uq6UWRgzMr`).
- Influence and gradient-based data selection for instruction tuning.
- Quality-score and diversity coreset selection.
- Layer-selective or parameter-efficient tuning baselines.
- Retention and catastrophic-forgetting evaluation in instruction tuning.

## Boundary Statement

NeuroTrace-IT does not claim that neuron-aware selection is new. The
contribution must be that trajectory-level signatures add value over endpoint
activation features under the same data/training budget and that this value is
visible after retention and hallucination drift are counted.

## Reviewer-Risk Notes

- If endpoint-neuron selection matches the method, the core novelty fails.
- If layer policy only adds complexity, it should be an ablation, not a main
  contribution.
- If target gains come with retention collapse, the paper cannot claim reliable
  instruction tuning.


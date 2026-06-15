# Risks And Blockers

## Active Blockers

| Blocker | Severity | Evidence | Mitigation |
| --- | --- | --- | --- |
| Server unavailable for activation extraction/training. | high | User instruction says server is temporarily unavailable. | Finish local design package and keep commands as TODO. |
| NAIT/endpoint-neuron novelty risk. | high | Current ICLR 2026 NAIT already uses neuron activation similarity for IT selection. | Make endpoint-neuron baseline mandatory and require trajectory gate. |
| Activation extraction cost may be too high. | medium | Full token/step/layer traces can be storage-heavy. | Cost gate and trajectory resolution sweep. |

## Scientific Risks

- Trajectory features may not beat endpoint features.
- Layer-wise policy may add complexity without gain.
- Retention drift may be noisy or dataset-dependent.

## Stop Rules

- If trajectory selection fails against endpoint selection under matched budget,
  pivot or downgrade to diagnostic analysis.
- If retention drift worsens beyond threshold, no reliability claim is allowed.
- If cost exceeds gate without strong gain, narrow to analysis rather than
  deployment-style method.


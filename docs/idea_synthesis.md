# Idea Synthesis

## Source Ideas Read

The seed idea came from the instruction-tuning note in
`D:\devtools\scratch\codex-hallucination-directions-20260615`. It proposed:

- neuron-aware data selection should consider reasoning trajectories, not only
  start/end token activations;
- different layers serve different functions, so layer-wise data selection or
  freezing may matter.

The raw idea file is not copied into this repository.

## Abstraction

The top-conference version is not "add more activation features to NAIT." It is:

> instruction examples should be selected by trajectory-level neural evidence
> only if that evidence improves target capability under the same budget and
> reduces semantic/hallucination drift.

## Rejected Directions

- activation steering at inference time;
- layer freezing without endpoint-neuron baselines;
- capability-only data selection without retention and drift checks;
- expensive trajectory signatures that do not beat endpoint activation.

## Selected Direction

NeuroTrace-IT keeps the user's trajectory/layer insight but makes it falsifiable:

1. endpoint neuron selection is a mandatory baseline;
2. layer policy is an ablation, not assumed contribution;
3. retention and hallucination drift are main metrics;
4. cost relative to endpoint selection is a gate.


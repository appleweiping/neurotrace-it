# Research Brief

## Problem

Instruction-tuning data selection methods can use quality scores, gradients,
influence estimates, or neuron activation similarity. Recent neuron-aware
approaches show that activation patterns are useful, but often compress an
example to start/end-token or aggregate activations. Reasoning examples are
processes: a math solution, coding trace, or multi-hop answer may activate
different layer functions over time. Selecting data from only endpoints can miss
whether the trajectory teaches the target capability or destabilizes retained
abilities.

NeuroTrace-IT asks:

> Do trajectory-level neural activation distributions and layer-function
> compatibility select better instruction-tuning data than endpoint activation
> similarity while reducing semantic and hallucination drift?

## Core Hypothesis

Instruction examples that share target reasoning trajectories but differ in
surface form are better training signals than examples that merely share
endpoint activations. Layer-wise adaptation should be guided by where those
trajectory signals live; freezing or adapting all layers equally is often
suboptimal.

## What Is New

- Activation trajectory signatures over steps, tokens, and layers.
- Joint adaptation/retention score balancing target gain and Fisher-style drift.
- Layer-function compatibility audit for layer-wise LoRA or freezing.
- Cross-task hallucination and semantic-retention gates as main evidence, not
  afterthoughts.

## Closest Work To Beat

- Neuron-aware instruction tuning data selection.
- Influence and gradient-based coreset selection.
- Quality-score and diversity-score instruction data filters.
- Neuron/layer selective safety or RAG tuning.
- Path-aware adaptation and stability methods.

## Kill Arguments

1. The trajectory score may be a more expensive version of endpoint activation
   similarity.
2. Layer-wise freezing may not help once LoRA rank and training budget are
   controlled.
3. Retention gains may come from selecting easier or shorter examples.
4. Cross-task hallucination drift may be too noisy to support a main claim.

## Target Venues

ICLR, NeurIPS, ICML, ACL, EMNLP, or NAACL.


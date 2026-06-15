# Motivation, Ablation, And Hyperparameter Plan

## Motivation Study

Goal: show endpoint activation similarity can miss trajectory-level differences
that matter for reasoning transfer or retention.

Planned figure:

- endpoint similarity vs trajectory similarity scatter;
- color by downstream retention-adjusted gain.

## Ablations

- endpoint activation only;
- trajectory without layer policy;
- layer policy without trajectory selection;
- no retention penalty;
- no hallucination drift gate;
- random subset;
- full data at matched compute where feasible.

## Hyperparameters

- trajectory aggregation window;
- layer subset;
- token/step sampling rate;
- retention penalty weight;
- LoRA rank;
- selected-data budget;
- selection diversity temperature.

## Required Curves

- target gain vs selected-data budget;
- retention drift vs target gain;
- cost vs trajectory resolution;
- layer subset sweep;
- retention penalty sweep.


# Research Plan

## Research Question

Do trajectory-level activation distributions and layer-function compatibility
select better instruction-tuning data than endpoint neuron activation similarity
while controlling retention and hallucination drift?

## Core Hypotheses

H1. Trajectory signatures identify useful examples missed by endpoint activation
similarity.

H2. Layer-function compatibility improves adaptation only for certain capability
families and should be ablated rather than assumed.

H3. Retention and hallucination drift expose data-selection failures hidden by
target-task accuracy.

## Work Packages

1. Signature schema: define token/step/layer activation trajectory records.
2. Selection score: balance target trajectory match against stability drift.
3. Baselines: random, full-data, quality, diversity, influence/gradient,
   endpoint-neuron, layer-selective without trajectory.
4. Evaluation: target capability, retention, hallucination drift, cost, and
   layer ablations.
5. Paper evidence: motivation, main table, ablations, hyperparameter curves,
   failure cases, and claim audit.

## Server Boundary

Local work ends at configs, docs, validators, and small contract tests.
Activation extraction and training require server approval.


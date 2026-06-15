# Data And Evaluation Plan

## Data Families

- Math instruction pool and held-out math reasoning eval.
- Code instruction pool and executable code eval.
- Multi-hop QA instruction pool and distractor eval.
- General instruction-following retention eval.
- Factuality/hallucination drift eval.

## Required Manifests

- candidate pool source, license, and hash;
- selected-example IDs and selection score hash;
- activation extraction config;
- train/valid/test split hash;
- contamination/leakage audit.

## Evaluation Tables

1. Target capability under fixed selected-data budget.
2. Retention and semantic drift.
3. Hallucination/factuality drift.
4. Layer policy ablations.
5. Selection and training cost.

## Fairness Policy

All selection methods use the same candidate pool, data budget, base model,
LoRA rank, training steps, validation policy, and evaluator.


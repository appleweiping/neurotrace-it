# Baseline Contract

All baselines must share:

- same base checkpoint and tokenizer;
- same candidate instruction pool;
- same selected data budget;
- same training steps, optimizer, LoRA rank, and evaluation code;
- same validation-only selection policy.

## Provenance Fields

- candidate pool manifest and hash;
- activation extraction configuration;
- layer list and token/step aggregation policy;
- selected-example IDs and hash;
- training config and seed;
- checkpoint hash or server path;
- evaluation command and metric file hash.

## Forbidden Comparisons

- comparing a trajectory-selected subset with endpoint selection at different
  data budgets;
- changing LoRA rank for only the proposed method;
- using test-set drift metrics to choose layer policy;
- reporting only target-task gains without retention and drift tables.


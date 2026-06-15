# Reproducibility Ledger

Every server artifact must record:

- git commit and dirty status;
- base model/tokenizer hash;
- candidate pool hash;
- activation extraction config;
- selected-example IDs and hashes;
- baseline source and commit;
- LoRA/training config and seed list;
- evaluator command and output hashes;
- ARIS evidence tier.

No paper claim may bypass this ledger.


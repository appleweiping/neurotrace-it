# Server Runbook

No server training is authorized at scaffold time.

## Required Before Server Use

1. ARIS experiment-plan review with numeric gates.
2. Compute estimate for activation extraction, selection, training, and
   evaluation.
3. Candidate instruction pool manifest and contamination audit.
4. Frozen baseline and proposed method configs.
5. User approval for exact server command.

## Lightweight Artifacts To Sync Back

- resolved configs;
- selected example IDs and hashes;
- aggregate metrics and paired-test inputs;
- retention and drift audit tables;
- layer ablation summaries;
- provenance JSON.

Do not commit raw corpora, checkpoints, full activation dumps, credentials, or
large logs.

